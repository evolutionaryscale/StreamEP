"""Shared base for the streaming-MoE GEMM kernel classes.

All four streaming kernels (`StreamingMoeA`, `StreamingMoeYBwd`,
`StreamingMoeY`, `StreamingMoeABwd`) are `quack.gemm_sm90.GemmSm90`
subclasses that differ only in their epilogue, but share the exact same
streaming *scheduler* behavior: a linear-claim + per-tile count-vs-target
spin via ``StreamingTileScheduler``. That shared behavior lives here so it
is written once.

`StreamingGemmBase` subclasses quack's concrete ``GemmDefaultSm90``
(= ``GemmDefaultEpiMixin`` + ``GemmSm90``): single inheritance, no mixin.
It carries the three members that are byte-identical across all four
kernels — ``get_scheduler_class``, ``get_scheduler_arguments``, and the
``__call__`` type-shim — and nothing epilogue-specific. Leaves supply their
own ``_epi_ops`` + ``epi_visit_subtile`` (and, for the Y-family, an
``epi_subtile_store`` scatter override via ``StreamingScatterBase``).

Note the A-family (`StreamingMoeA`, `StreamingMoeYBwd`) keeps the default
linear/D-store epilogue ops from ``GemmDefaultEpiMixin`` and adds a gated /
plain ``TileStore`` aux output; the Y-family reassigns ``_epi_ops`` down to
its scatter set (see ``StreamingScatterBase``). Because a subclass fully
controls ``_epi_ops`` by reassignment (``ComposableEpiMixin`` regenerates
``EpilogueParams`` from the subclass's tuple, and ``GemmDefaultEpiMixin``
never force-injects its ops — cf. ``GemmSymmetricMixin`` concatenating them
manually), inheriting ``GemmDefaultSm90`` costs the Y-family nothing.
"""

from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int32
from quack.cute_dsl_utils import mlir_namedtuple
from quack.gemm_default_epi import GemmDefaultSm90
from quack.gemm_sm90 import GemmSm90
from quack.tile_scheduler import PersistenceMode
from quack.varlen_utils import VarlenArguments

from stream_ep.stream_moe.tile_scheduler import (
    StreamingTileScheduler,
    StreamingTileSchedulerArguments,
)


# ---------------------------------------------------------------------------
# Host-facing scheduler-options NamedTuple. Mirrors TileSchedulerOptions but
# carries the streaming-specific tensors/pointers that the scheduler needs.
# Lives here (not in kernel_a) because all four streaming kernels take it.
# ---------------------------------------------------------------------------
@mlir_namedtuple
class StreamingTileSchedulerOptions(NamedTuple):
    max_active_clusters: Int32
    consumer_head: cute.Tensor  # [1] int32 — global linear claim counter
    # Per-tile ready spin source for kernel A's dispatch handoff. The
    # scheduler does `count[tile] == target[tile]` (count-vs-target). Dispatch's
    # metadata kernel fills `pool_arrival_target` with the per-tile firing
    # target; dispatch's Pass 2 release-adds into `pool_arrival_count`.
    pool_arrival_count: cute.Tensor   # [total_tiles] int32 — release-add destination
    pool_arrival_target: cute.Tensor  # [total_tiles] int32 — per-tile firing target
    expert_pool_block_offset: (
        cute.Tensor
    )  # [E_local + 1] int32 — pool-block prefix-sum. Source for the
    # warp-cooperative ballot lookup that retired per-claim `tile_id_to_expert`.
    total_tiles: Int32  # passed as scalar so get_grid_shape doesn't deref device tensor
    # Optional cross-stream launch-gate "started" flag (single int32 in
    # device memory). When supplied, the CTA that wins ``linear_idx == 0``
    # in ``_fetch_next_work_idx`` atomicAdd's this flag once. The host on
    # the consumer stream (typically communicate) issues
    # ``cuStreamBatchMemOp wait_value_geq`` against this flag before
    # launching combine_main / combine_grads_main, so combine's 80-CTA
    # sender grid can't grab SMs ahead of kernel_y / kernel_a_bwd's
    # 132-CTA grid. Pass ``None`` to disable (kernel_a / kernel_y_bwd
    # don't bump anything — only kernel_y / kernel_a_bwd own a flag).
    started_flag: Optional[cute.Tensor] = None  # [1] int32 device flag


# ---------------------------------------------------------------------------
# Shared streaming-GEMM base.
# ---------------------------------------------------------------------------
class StreamingGemmBase(GemmDefaultSm90):
    """Streaming scheduler behavior shared by all four streaming kernels.

    Holds only the scheduler hooks + the ``__call__`` type-shim; the
    epilogue (``_epi_ops`` / ``epi_visit_subtile`` / store path) is the
    concern of each leaf. Inherits the default linear epilogue from
    ``GemmDefaultSm90``; leaves that don't want it reassign ``_epi_ops`` and
    override ``epi_visit_subtile`` (which shadows the inherited linear math).
    """

    # -- scheduler hooks -----------------------------------------------------

    def get_scheduler_class(self, varlen_m: bool = False):
        return StreamingTileScheduler

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,  # (n, k, l=E_local) — n-dim tile count = ceil(n / tile_N)
        mD: Optional[cute.Tensor],
        scheduler_args: StreamingTileSchedulerOptions,
        varlen_args: VarlenArguments,
        epilogue_args,
    ):
        # mB's N-dim (mode 0) is the GEMM N: 2I for kernel A, I for y_bwd,
        # H for kernel Y / a_bwd. Read generically so the body is identical
        # for all four kernels.
        num_pid_n = cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1])
        E_local = cute.size(mB, mode=[2])
        return StreamingTileSchedulerArguments(
            problem_shape_ntile_mnl=(None, num_pid_n, E_local),
            consumer_head=scheduler_args.consumer_head,
            arrival_count=scheduler_args.pool_arrival_count,
            arrival_target=scheduler_args.pool_arrival_target,
            expert_pool_block_offset=scheduler_args.expert_pool_block_offset,
            total_tiles=scheduler_args.total_tiles,
            tile_shape_mn=self.cta_tile_shape_mnk[:2],
            cluster_shape_mnk=self.cluster_shape_mnk,
            scheduler_warp_id=self.ab_load_warp_id,
            persistence_mode=PersistenceMode.DYNAMIC,
            started_flag=scheduler_args.started_flag,
        )

    # -- launch type-shim ----------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args: tuple,
        scheduler_args: StreamingTileSchedulerOptions,
        varlen_args: Optional[VarlenArguments],
        stream: cuda.CUstream,
        mSFA: Optional[cute.Tensor] = None,
        mSFB: Optional[cute.Tensor] = None,
    ):
        """Type-shim override so CuTeDSL accepts StreamingTileSchedulerOptions
        as the scheduler_args type (base annotation is TileSchedulerOptions).

        The signature otherwise mirrors main's ``GemmSm90.__call__`` exactly:
        the trailing ``mSFA``/``mSFB`` scale-factor slots are part of the
        unified SM90/100/120 TMA arity (always None for StreamEP's plain bf16
        GEMMs; the compiled TVM-FFI arg spec bakes them in, so they must be
        present). Body delegates to ``GemmSm90.__call__`` unchanged.
        """
        GemmSm90.__call__(
            self,
            mA,
            mB,
            mD,
            mC,
            epilogue_args,
            scheduler_args,
            varlen_args,
            stream,
            mSFA,
            mSFB,
        )
