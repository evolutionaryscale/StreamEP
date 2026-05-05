"""Streaming-MoE kernel A (CuTeDSL, SM90, pool layout).

Forward kernel A of the problem-tile streaming pipeline:
  * Persistent CTAs pull tiles from a producer-fed queue (`tile_ready`).
  * For each claimed tile_id, the scheduler reads `expert_id =
    tile_id_to_expert[tile_id]` and computes `pid_m = tile_id -
    expert_pool_block_offset[expert_id]`.
  * Standard varlen_m strided TMA load of `pool[tile_id * tile_M : ..., :]`
    (the row offset = `cu_seqlens_m[expert_id] + pid_m * tile_m` lands at
    the correct expert-major pool row by construction).
  * GEMM against W1[expert_id], SwiGLU register-resident epilogue, TMA-store
    the I-half post-activation to `postact_a[tile_id * tile_M : ..., :]`.

Inherits the GEMM mainloop, SwiGLU epilogue, scheduler-warp + pipeline-state
machinery from `quack.gemm_act.GemmGatedSm90`. Streaming-specific behavior is
isolated to two overrides:
  (1) get_scheduler_class — return StreamingTileScheduler.
  (2) get_scheduler_arguments — build StreamingTileSchedulerArguments from
      pool-shape metadata.

The streaming scheduler uses the upstream 4-int sched payload
(pid_m, pid_n, batch_idx, is_valid) — no streaming-specific SMEM extension.
Kernel A's mainloop and postact path land at the right pool rows via
`cu_seqlens_m[batch_idx] + pid_m * tile_m` alone; tile_id is computed
locally inside the scheduler warp's queue-pull and used only for the
ready-spin and to derive expert_id/pid_m.
"""

from dataclasses import MISSING
from typing import Callable, NamedTuple, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import quack.utils as utils
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.activation import gate_fn_map
from quack.cache_utils import COMPILE_ONLY, jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    ParamsBase,
    get_device_capacity,
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.epi_ops import EpiOp
from quack.gemm_act import GemmGatedMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_tvm_ffi_utils import compile_gemm_kernel
from quack.rounding import RoundingMode
from quack.tile_scheduler import PersistenceMode
from quack.varlen_utils import VarlenArguments

from evolutionaryscale.models.moe.streaming_moe.ptx_helpers import (
    st_release_sys_global,
    threadfence_system,
)
from evolutionaryscale.models.moe.streaming_moe.streaming_tile_scheduler import (
    StreamingTileScheduler,
    StreamingTileSchedulerArguments,
)


# ---------------------------------------------------------------------------
# Per-tile a_ready release. Kernel A's downstream consumer is kernel Y, which
# acquire-spins on a_ready[tile_id] >= compute_seq before reading postact_a's
# per-tile slab. The release-store has to happen AFTER kernel A's TMA stores
# for THIS tile have actually committed to HBM — a thread-side
# `cp.async.bulk.wait_group(0)` drains the TMA store pipeline before the
# release.
#
# Multi-pid_n gating: a single tile_id is split across `num_pid_n` CTAs (one
# per N-stripe). The release-store fires ONCE per tile, when the last
# N-stripe completes. Atomic-add on `tile_n_stripes_done[tile_id]` provides
# the gating; the CTA whose atomic-add returns `num_pid_n - 1` is the last,
# fires `threadfence_system` + `st_release_sys_global(a_ready[tile_id], compute_seq)`.
# ---------------------------------------------------------------------------
@mlir_namedtuple
class TileReadyParams(NamedTuple):
    a_ready: cute.Tensor  # [total_tiles] int64 — A → Y release stamp
    tile_n_stripes_done: cute.Tensor  # [total_tiles] int32 — per-tile N-stripe arrival
    compute_seq: Int64  # value to release-store on hit-(num_pid_n - 1)
    num_pid_n: Int32  # ceil(2I / tile_N)
    tile_m: cutlass.Constexpr[
        int
    ]  # for tile_id = cu_seqlens_m[batch_idx] // tile_m + pid_m


class TileReadyRelease(EpiOp):
    """Per-tile a_ready[tile_id] = compute_seq release-store, with multi-pid_n
    gating and TMA-store drain.

    No SMEM. Just a cross-CTA arrival counter (gmem) + a release-store on the
    last-stripe.
    """

    def __init__(self, name: str = "tile_ready"):
        super().__init__(name)

    def param_fields(self):
        return [(self.name, object, MISSING)]

    def to_params(self, gemm, args):
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def end(
        self,
        gemm,
        param,
        state,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        if const_expr(param is None):
            return
        # Drain THIS CTA's TMA stores (cp.async.bulk.wait_group<0>) so that
        # postact_a is observable to kernel Y before a_ready[tile_id] flips.
        # Without this, the release-store can race ahead of the TMA hardware
        # and kernel Y reads stale postact_a.
        cute.arch.cp_async_bulk_wait_group(0, read=False)

        # Reconstruct tile_id from (batch_idx, pid_m).
        # cu_seqlens_m carries `expert_pool_block_offset * tile_m`, so
        # tile_id = expert_pool_block_offset[batch_idx] + pid_m
        #        = cu_seqlens_m[batch_idx] // tile_m + pid_m.
        batch_idx = tile_coord_mnkl[3]
        pid_m = tile_coord_mnkl[0]
        tile_id = (
            varlen_manager.params.cu_seqlens_m[batch_idx] // Int32(param.tile_m) + pid_m
        )

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        is_thread0 = (warp_idx == Int32(0)) & (lane_idx == Int32(0))

        if is_thread0:
            # Count this CTA's N-stripe as done. Atomic-add provides acq_rel
            # ordering across CTAs — the CTA that gets `prev == num_pid_n - 1`
            # observes all earlier CTAs' TMA-store drains too.
            stripes_ptr = utils.elem_pointer(param.tile_n_stripes_done, (tile_id,))
            prev = utils.atomic_add_i32(Int32(1), stripes_ptr)
            is_last_stripe = prev == (param.num_pid_n - Int32(1))
            if is_last_stripe:
                # All N-stripes for this tile have drained their TMA stores.
                # System fence then release-store so kernel Y on a different
                # stream can acquire-load with cross-stream visibility.
                threadfence_system()
                a_ready_ptr = utils.elem_pointer(param.a_ready, (tile_id,))
                st_release_sys_global(a_ready_ptr, param.compute_seq)


# ---------------------------------------------------------------------------
# Host-facing scheduler-options NamedTuple. Mirrors TileSchedulerOptions but
# carries the streaming-specific tensors/pointers that the scheduler needs.
# ---------------------------------------------------------------------------
@mlir_namedtuple
class StreamingTileSchedulerOptions(NamedTuple):
    max_active_clusters: Int32
    consumer_head: cute.Tensor  # [1] int32 — global linear claim counter
    tile_ready: (
        cute.Tensor
    )  # [total_tiles] int64 — release stamps from dispatch's Pass 2
    tile_id_to_expert: cute.Tensor  # [total_tiles] int32 — per-tile expert lookup
    expert_pool_block_offset: cute.Tensor  # [E_local + 1] int32 — pool-block prefix-sum
    dispatch_seq: Int64
    total_tiles: Int32  # passed as scalar so get_grid_shape doesn't deref device tensor


# ---------------------------------------------------------------------------
# Streaming kernel A class.
# ---------------------------------------------------------------------------
class StreamingMoeASm90(GemmGatedMixin, GemmSm90):
    """Streaming-MoE kernel A: standard strided varlen_m GEMM + SwiGLU with
    queue-pull scheduler. Pool layout means kernel A uses the base GEMM
    mainloop's varlen_m path verbatim — no per-tile gather indirection.

    Adds a per-tile `a_ready[tile_id] = compute_seq` release-store at the end
    of each tile's epilogue (after TMA-store drain + multi-pid_n gating) so
    kernel Y's per-tile acquire-spin observes a_ready cross-stream.
    """

    # Append TileReadyRelease to GemmGatedMixin's _epi_ops chain. Order
    # matters: the framework iterates ops for begin/end; the postact TileStore
    # must run before our release-drain so postact_a's TMA stores have been
    # COMMITTED before our wait_group(0) drains them.
    _epi_ops = (*GemmGatedMixin._epi_ops, TileReadyRelease("tile_ready"))
    _epi_param_bases = (ParamsBase,)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        tile_ready: TileReadyParams
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops + _extra_param_fields by
    # ComposableEpiMixin.__init_subclass__.

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
        trace_ptr: Optional[Int64] = None,
    ):
        """Type-shim override so CuTeDSL accepts StreamingTileSchedulerOptions
        as the scheduler_args type (base annotation is TileSchedulerOptions).
        Body delegates to GemmSm90.__call__ unchanged.
        """
        from quack.gemm_sm90 import GemmSm90 as _GemmSm90Base

        _GemmSm90Base.__call__(
            self,
            mA,
            mB,
            mD,
            mC,
            epilogue_args,
            scheduler_args,
            varlen_args,
            stream,
            trace_ptr,
        )

    # -- scheduler hooks -----------------------------------------------------

    def get_scheduler_class(self, varlen_m: bool = False):
        return StreamingTileScheduler

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,  # pool: (TK_padded, H)
        mB: cute.Tensor,  # W1: (2I, H, E_local)
        mD: Optional[cute.Tensor],  # None (no D for streaming kernel A)
        scheduler_args: StreamingTileSchedulerOptions,
        varlen_args: VarlenArguments,
        epilogue_args,
    ):
        # mB shape is (n=2I, k=H, l=E_local); n-dim tile count = ceil(2I / tile_N).
        num_pid_n = cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1])
        E_local = cute.size(mB, mode=[2])
        return StreamingTileSchedulerArguments(
            problem_shape_ntile_mnl=(None, num_pid_n, E_local),
            consumer_head=scheduler_args.consumer_head,
            tile_ready=scheduler_args.tile_ready,
            tile_id_to_expert=scheduler_args.tile_id_to_expert,
            expert_pool_block_offset=scheduler_args.expert_pool_block_offset,
            dispatch_seq=scheduler_args.dispatch_seq,
            total_tiles=scheduler_args.total_tiles,
            tile_shape_mn=self.cta_tile_shape_mnk[:2],
            cluster_shape_mnk=self.cluster_shape_mnk,
            persistence_mode=PersistenceMode.STREAMING,
        )

    # postact destination: inherited GemmGatedSm90.epi_setup_postact uses
    # `varlen_manager.offset_batch_epi(mPostAct, batch_idx)` (shift by
    # cu_seqlens_m[batch_idx]) + `local_tile((pid_m, pid_n))`. The combined
    # row offset is `cu_seqlens_m[batch_idx] + pid_m * tile_m =
    # expert_pool_block_offset[e] * tile_m + (tile_id - expert_pool_block_offset[e])
    # * tile_m = tile_id * tile_m`, which lands postact_a's per-tile slab — no
    # streaming-specific override needed.


# ---------------------------------------------------------------------------
# JIT compile factory.
# ---------------------------------------------------------------------------
@jit_cache
def _compile_streaming_moe_a(
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    postact_dtype: Type[cutlass.Numeric],
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    cluster_n: int,
    activation: str,
    device_capacity,
    *,
    store_preact: bool = False,
):
    assert device_capacity[0] == 9, "Streaming MoE kernel A is SM90-only for now"
    assert activation in gate_fn_map, f"Need a gated activation; got {activation}"

    H_sym = cute.sym_int()
    I2_sym = cute.sym_int()
    I_sym = cute.sym_int()
    E_sym = cute.sym_int()
    TK_padded_sym = cute.sym_int()
    Mflat_sym = cute.sym_int()  # total_tiles * tile_m, in postact's M dim
    total_tiles_sym = cute.sym_int()
    cu_seqlens_len_sym = cute.sym_int()  # E_local + 1 at runtime

    # A: pool (TK_padded, H), k-major (H is contiguous).
    mA = fake_tensor(a_dtype, (TK_padded_sym, H_sym), leading_dim=1, divisibility=8)
    # B: W1 (2I, H, E_local), k-major per expert (H contiguous), batch dim = E_local.
    mB = fake_tensor(b_dtype, (I2_sym, H_sym, E_sym), leading_dim=1, divisibility=8)
    # mD: optional pre-SwiGLU output. When `store_preact=True`, kernel A's
    # standard mD TMA-store path (inherited from GemmDefaultEpiMixin via
    # GemmGatedMixin) writes the [2I] accumulator (post-alpha/beta/RowVec/
    # ColVec, pre-act-fn) to gmem alongside the postact_a [I] write.
    # Bwd consumes preact via `kernel_a_bwd`'s SwiGLU-bwd in registers
    # (skipping the otherwise-required `pool @ W1.T` recompute GEMM); fwd
    # paths that don't need bwd activations leave it None.
    if store_preact:
        # Same pool layout as postact_a (Mflat = total_tiles * tile_m), but
        # full N=2I instead of half-N=I. n-major (2I contiguous), bf16.
        mD = fake_tensor(
            postact_dtype, (Mflat_sym, I2_sym), leading_dim=1, divisibility=8
        )
    else:
        mD = None
    mC = None
    # mPostAct: flat (total_tiles * tile_m, I), n-major (I contiguous).
    mPostAct = fake_tensor(
        postact_dtype, (Mflat_sym, I_sym), leading_dim=1, divisibility=8
    )

    # cu_seqlens_m drives the standard varlen_m m-offset for kernel A: each
    # entry is `expert_pool_block_offset[e] * tile_m`. Length E_local + 1.
    mCuSeqlensM = fake_tensor(
        cutlass.Int32, (cu_seqlens_len_sym,), leading_dim=0, divisibility=1
    )

    consumer_head = fake_tensor(cutlass.Int32, (cute.sym_int(),), divisibility=1)
    tile_ready = fake_tensor(cutlass.Int64, (total_tiles_sym,), divisibility=1)
    tile_id_to_expert = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)
    expert_pool_block_offset = fake_tensor(
        cutlass.Int32, (cu_seqlens_len_sym,), divisibility=1
    )
    a_ready = fake_tensor(cutlass.Int64, (total_tiles_sym,), divisibility=1)
    tile_n_stripes_done = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)

    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(0),  # set at runtime; 0 here keeps fake compile happy
        consumer_head=consumer_head,
        tile_ready=tile_ready,
        tile_id_to_expert=tile_id_to_expert,
        expert_pool_block_offset=expert_pool_block_offset,
        dispatch_seq=Int64(0),
        total_tiles=Int32(0),
    )

    tile_ready_params = TileReadyParams(
        a_ready=a_ready,
        tile_n_stripes_done=tile_n_stripes_done,
        compute_seq=Int64(0),
        num_pid_n=Int32(0),
        tile_m=tile_m,
    )

    epi_args = StreamingMoeASm90.EpilogueArguments(
        mPostAct=mPostAct,
        tile_ready=tile_ready_params,
        act_fn=gate_fn_map[activation],
        rounding_mode=RoundingMode.RN,
    )

    varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlensM, mCuSeqlensK=None, mAIdx=None)

    return compile_gemm_kernel(
        StreamingMoeASm90,
        a_dtype,
        (tile_m, tile_n),
        (cluster_m, cluster_n, 1),
        pingpong=False,
        persistent=True,
        gather_A=False,
        is_dynamic_persistent=False,
        device_capacity=device_capacity,
        mA=mA,
        mB=mB,
        mD=mD,
        mC=mC,
        epi_args=epi_args,
        scheduler_args=scheduler_args,
        varlen_args=varlen_args,
    )


# ---------------------------------------------------------------------------
# Test-only producer: walks tile_ready slot-by-slot and release-stores
# dispatch_seq on each, with a delay between fires. Used by tests to validate
# kernel A's per-tile spin without DeepEP.
# ---------------------------------------------------------------------------
class _StreamingTileProducer:
    @cute.jit
    def __call__(
        self,
        tile_ready: cute.Tensor,  # [total_tiles] int64
        total_tiles: cutlass.Int32,
        dispatch_seq: cutlass.Int64,
        delay_clocks: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(tile_ready, total_tiles, dispatch_seq, delay_clocks).launch(
            grid=[1, 1, 1], block=[1, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        tile_ready: cute.Tensor,
        total_tiles: cutlass.Int32,
        dispatch_seq: cutlass.Int64,
        delay_clocks: cutlass.Int32,
    ):
        from cutlass._mlir.dialects import nvvm
        from cutlass.cutlass_dsl import T
        from quack import utils

        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            for i in cutlass.range(total_tiles):
                start = cutlass.Int64(nvvm.read_ptx_sreg_clock64(T.i64()))
                end = start + cutlass.Int64(delay_clocks)
                while cutlass.Int64(nvvm.read_ptx_sreg_clock64(T.i64())) < end:
                    pass
                ready_ptr = utils.elem_pointer(tile_ready, (i,))
                threadfence_system()
                st_release_sys_global(ready_ptr, dispatch_seq)


@jit_cache
def _compile_streaming_tile_producer():
    total_tiles_sym = cute.sym_int()
    ready = fake_tensor(cutlass.Int64, (total_tiles_sym,), divisibility=1)
    op = _StreamingTileProducer()
    return cute.compile(
        op,
        ready,
        cutlass.Int32(0),
        cutlass.Int64(0),
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def fire_tiles_with_delay(
    tile_ready: torch.Tensor, dispatch_seq: int, delay_us: int = 50
) -> None:
    """Test helper: launches a single-thread producer kernel on the current
    CUDA stream that release-stores dispatch_seq into each slot of `tile_ready`
    with `delay_us` between fires.
    """
    assert tile_ready.dtype == torch.int64
    assert tile_ready.is_cuda and tile_ready.is_contiguous()
    total_tiles = tile_ready.shape[0]
    # H100 clock ~1.5 GHz → 1500 cycles/μs.
    delay_clocks = max(1, int(delay_us * 1500))
    compiled = _compile_streaming_tile_producer()
    compiled(
        tile_ready,
        cutlass.Int32(total_tiles),
        cutlass.Int64(dispatch_seq),
        cutlass.Int32(delay_clocks),
    )


def streaming_moe_a(
    pool: torch.Tensor,  # (TK_padded, H) bf16 — k-major (pool data, expert-major)
    W1: torch.Tensor,  # (E_local, 2I, H) bf16 — k-major per expert
    postact_a: torch.Tensor,  # (total_tiles, tile_M, I) bf16
    tile_id_to_expert: torch.Tensor,  # (total_tiles,) int32
    expert_pool_block_offset: torch.Tensor,  # (E_local + 1,) int32 — pool-block prefix sum
    tile_ready: torch.Tensor,  # (total_tiles,) int64 release stamps (input from dispatch)
    a_ready: torch.Tensor,  # (total_tiles,) int64 release stamps (output to kernel Y)
    dispatch_seq: int,
    compute_seq: int,
    *,
    preact_a: torch.Tensor | None = None,
    tile_m: int = 128,
    tile_n: int = 256,
    cluster_m: int = 1,
    cluster_n: int = 1,
    activation: str = "swiglu",
    num_sms: int | None = None,
) -> None:
    """Launch streaming-MoE kernel A on the caller's current CUDA stream (pool layout).

    ``num_sms`` caps the persistent-grid CTA count to the given value. When
    ``None`` (default) the kernel fills the GPU via ``get_max_active_clusters``.
    Smaller caps leave SMs available for kernel Y to run concurrently — see
    design.md §"SM budget".

    Caller is responsible for:
      - allocating ``postact_a`` ``(total_tiles, tile_M, I)`` ON THE SAME STREAM
        this function is called from (so the kernel's TMA stores are naturally
        ordered with the allocation; otherwise stale memory may leak through).
      - ensuring ``tile_ready`` is populated by the producer (DeepEP's
        ``Buffer.dispatch`` Pass 2 or a test stub) on a stream that release-stores
        ``tile_ready[tile_id] = dispatch_seq`` once the tile is ready. Kernel A's
        per-tile acquire-spin handles cross-stream visibility for ``tile_ready``
        and the dispatch metadata it transitively depends on.

    The internal ``consumer_head`` and ``tile_n_stripes_done`` counters are
    allocated on the calling stream so their zero-init is naturally ordered
    with the kernel.

    Per-tile a_ready release: at the end of each tile's epilogue (after all
    pid_n N-stripes have drained their TMA stores), kernel A release-stores
    ``a_ready[tile_id] = compute_seq`` with system scope. Kernel Y on
    ``compute_y_stream`` acquire-spins on this signal before reading the
    tile's postact_a slab. Multi-pid_n gating via an atomic-add to
    ``tile_n_stripes_done[tile_id]`` ensures the release fires once per tile,
    not once per N-stripe.

    Optional ``preact_a`` ``(total_tiles, tile_M, 2*I) bf16`` is the pre-SwiGLU
    accumulator destination for bwd. When passed, kernel A's standard mD
    TMA-store path (inherited from ``GemmGatedMixin → GemmDefaultEpiMixin``)
    writes the [2I] gate-up values to gmem alongside the postact_a [I] write.
    Saving preact lets ``kernel_a_bwd`` apply SwiGLU bwd in registers without
    a recompute GEMM (~370 µs/layer perf win at production); fwd-only paths
    leave ``preact_a=None`` to skip the extra TMA-store traffic. The two
    cases compile to separate kernels (different mD signature), keyed on
    ``store_preact`` in ``_compile_streaming_moe_a``'s jit_cache.
    """
    assert pool.is_cuda and W1.is_cuda and postact_a.is_cuda
    assert pool.dim() == 2 and pool.is_contiguous()
    assert W1.dim() == 3
    assert postact_a.dim() == 3
    total_tiles, postact_tile_m, I = postact_a.shape
    assert postact_tile_m == tile_m
    assert tile_id_to_expert.shape == (total_tiles,)
    assert tile_ready.shape == (total_tiles,) and tile_ready.dtype == torch.int64
    assert a_ready.shape == (total_tiles,) and a_ready.dtype == torch.int64
    H = pool.shape[1]
    E_local = W1.shape[0]
    assert expert_pool_block_offset.shape == (E_local + 1,)
    assert (
        W1.shape[1] == 2 * I
    ), f"W1 dim 1 must be 2*I = {2 * I}; got W1.shape={tuple(W1.shape)}"
    assert (
        W1.shape[2] == H
    ), f"W1 dim 2 (H) must match pool dim 1; got W1.shape={tuple(W1.shape)}, H={H}"
    two_I = W1.shape[1]
    if preact_a is not None:
        assert preact_a.is_cuda and preact_a.dim() == 3
        assert preact_a.shape == (total_tiles, tile_m, two_I), (
            f"preact_a must be (total_tiles, tile_m, 2*I) = "
            f"{(total_tiles, tile_m, two_I)}; got {tuple(preact_a.shape)}"
        )
        assert preact_a.dtype == postact_a.dtype, (
            f"preact_a dtype must match postact_a's; got {preact_a.dtype} vs "
            f"{postact_a.dtype}"
        )
    # Caller passes W1 as (E_local, 2I, H) k-major contiguous (each expert's
    # slab has H contiguous). We need the kernel to see shape (2I, H, E_local)
    # with leading_dim=1 (H is contiguous along K). torch.permute(1, 2, 0)
    # gives this layout WITHOUT a copy.
    W1_p = W1.permute(1, 2, 0)
    assert (
        W1_p.stride(1) == 1
    ), "W1[:,e,:] must be H-contiguous (caller passes k-major weights)"
    assert W1_p.shape == (two_I, H, E_local)

    # Flatten postact_a's leading two dims to (total_tiles * tile_m, I).
    postact_flat = postact_a.view(total_tiles * tile_m, I)
    # Flatten preact_a similarly when present — same Mflat dim, but full N=2I.
    preact_flat = (
        preact_a.view(total_tiles * tile_m, two_I) if preact_a is not None else None
    )

    # Build cu_seqlens_m = expert_pool_block_offset * tile_m. The standard
    # varlen_m path inside the GEMM uses this as the per-batch m-row offset:
    #   m_offset(tile) = cu_seqlens_m[batch_idx] + pid_m * tile_m
    #                  = expert_pool_block_offset[expert_id] * tile_m + tile_in_e * tile_m
    #                  = tile_id * tile_m
    # which lands at the correct pool row (pool is contiguous in tile_id order
    # by construction).
    cu_seqlens_m = (expert_pool_block_offset.to(torch.int32) * tile_m).contiguous()

    device_capacity = get_device_capacity(pool.device)
    assert device_capacity[0] == 9, "Streaming MoE kernel A is SM90-only for now"

    a_dtype = torch2cute_dtype_map[pool.dtype]
    b_dtype = torch2cute_dtype_map[W1.dtype]
    postact_dtype = torch2cute_dtype_map[postact_a.dtype]

    compiled_fn = _compile_streaming_moe_a(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        postact_dtype=postact_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        activation=activation,
        device_capacity=device_capacity,
        store_preact=preact_a is not None,
    )

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_m * cluster_n)
    if num_sms is not None:
        max_active_clusters = min(
            max_active_clusters, num_sms // (cluster_m * cluster_n)
        )

    # Internal scheduler counter — allocate on the calling stream (which is the
    # one the kernel will run on) so the zero-init is naturally ordered with
    # kernel A's `atomicAdd(consumer_head, 1)`. Allocating on a different stream
    # would race with the kernel's first atomic-claim and could leak stale
    # values from a recycled allocator slot, causing CTAs to early-exit.
    consumer_head = torch.zeros(1, dtype=torch.int32, device=pool.device)

    # Multi-pid_n N-stripe arrival counter. The CTA whose atomic-add returns
    # `num_pid_n - 1` is the last N-stripe to complete for its tile_id and
    # fires the per-tile a_ready release-store.
    num_pid_n = (2 * I + tile_n - 1) // tile_n
    tile_n_stripes_done = torch.zeros(
        total_tiles, dtype=torch.int32, device=pool.device
    )

    tile_ready_params = TileReadyParams(
        a_ready=a_ready,
        tile_n_stripes_done=tile_n_stripes_done,
        compute_seq=Int64(compute_seq),
        num_pid_n=Int32(num_pid_n),
        tile_m=None,  # Constexpr; burned in at compile, pass None at call time
    )

    epi_args = StreamingMoeASm90.EpilogueArguments(
        mPostAct=postact_flat,
        tile_ready=tile_ready_params,
        act_fn=None,  # Constexpr; pass None at call time
        rounding_mode=None,  # Constexpr; pass None at call time
    )
    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        consumer_head=consumer_head,
        tile_ready=tile_ready,
        tile_id_to_expert=tile_id_to_expert,
        expert_pool_block_offset=expert_pool_block_offset,
        dispatch_seq=Int64(dispatch_seq),
        total_tiles=Int32(total_tiles),
    )
    varlen_args = VarlenArguments(
        mCuSeqlensM=cu_seqlens_m, mCuSeqlensK=None, mAIdx=None
    )

    compiled_fn(
        pool, W1_p, preact_flat, None, epi_args, scheduler_args, varlen_args, None
    )
