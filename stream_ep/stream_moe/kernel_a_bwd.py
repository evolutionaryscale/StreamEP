"""Streaming-MoE kernel A bwd (CuTeDSL, SM90, pool layout — fused scatter).

Backward of fwd kernel A. Per the chain rule on `preact = pool @ W1[e].T`:
  dL/dpool[slot, :H] = dL/dswiglu_in[slot, :2I] @ W1[expert_for_slot]

`dL/dswiglu_in` is produced directly by `kernel_y_bwd`'s mD TMA-store
(SwiGLU bwd folded into kernel_y_bwd's epilogue — see `kernel_y_bwd.py`).

Kernel A_bwd runs on the SAME compute stream as kernel_y_bwd; same-stream
FIFO guarantees all of Y_bwd's mD/mPostAct TMA stores have drained and
dL_dweight atomic-adds are visible before A_bwd issues its first
instruction. No per-tile Y_bwd→A_bwd device-side signal is needed.

Per tile:
  * Streaming scheduler reuses dispatch_grads's per-tile pair
    (``bwd_dispatch_arrival_count`` / ``pool_arrival_target``) — at-target
    by the time A_bwd runs because Y_bwd already waited on it via the same
    scheduler protocol.
  * Standard varlen_m strided TMA load of
    `dL_dswiglu_in[tile_id * tile_M : ..., :]` (the row offset
    `cu_seqlens_m[expert_id] + pid_m * tile_m` lands on the right pool-major
    row by construction — same path fwd kernel A uses on pool).
  * NN GEMM against `W1[expert_id]`. mB is W1 permuted to (H, 2I, E_local)
    with H contiguous (n-major, leading_dim=0). The kernel-internal
    contraction `Σ_k A[m, k] * B[n, k]` evaluates to
      dL/dpool[m, h] = Σ_2I_idx dL/dswiglu_in[m, 2I_idx] * W1[e, 2I_idx, h]
    i.e. the (M, H) data-grad in registers.
  * R2S into a kernel-owned bf16 SMEM staging buffer, then per-warp coalesced
    atomic-scatter from SMEM into `dL_dx_per_r[r, :H]` via packed
    `red.global.add.bf16x2`. Per-slot `pool_recv_token` gating skips r=-1
    padding rows via PTX-level predication.
  * Per-tile end (last N-stripe gates): `bwd_k_local_remaining[r]` atomicSub
    by 1; on hit-zero `bwd_a_done_per_token[r]` release-stores
    `dispatch_seq` so the combine_grads sender (on its own stream) can pick
    up `dL_dx_per_r[r]` for RDMA push.

Strict subset of fwd kernel_y's epilogue — no weight multiply (the per-slot
pool_topk_weight was already absorbed into dL/dswiglu_in upstream by
kernel_y_bwd's epilogue: `dswiglu(gate, up, w*g) → (w*dgate, w*dup, postact)`
by chain-rule linearity in dpostact). All atomic-scatter mechanics
(predicated v4 bf16x2 issue, multi-pid_n bookkeeping gate, per-row last-stripe
release) are inherited verbatim from `kernel_y.py`.

Shares streaming machinery with fwd kernels:
  * `StreamingTileScheduler` for linear-claim + per-tile count-vs-target
    spin. Kernel A_bwd plumbs (``bwd_dispatch_arrival_count``,
    ``pool_arrival_target``) — same dispatch_grads handoff Y_bwd waited on.
  * Pool-layout `StreamingHandle` carries `pool_recv_token`,
    `bwd_k_local_remaining` (initialized from saved `k_local_total`),
    `bwd_a_done_per_token`, `dL_dx_per_r`,
    `expert_pool_block_offset`.
"""

from typing import NamedTuple, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from quack.cache_utils import COMPILE_ONLY, jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    get_device_capacity,
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.gemm_sm90 import GemmSm90
from quack.gemm_tvm_ffi_utils import compile_gemm_kernel
from quack.tile_scheduler import PersistenceMode
from quack.varlen_utils import VarlenArguments

from stream_ep.stream_moe.kernel_a import StreamingTileSchedulerOptions
from stream_ep.stream_moe.kernel_y import (
    AtomicScatterStore,
    ScatterParams,
    StreamingMoeY,
)
from stream_ep.stream_moe.tile_scheduler import (
    StreamingTileScheduler,
    StreamingTileSchedulerArguments,
)


# ---------------------------------------------------------------------------
# Streaming kernel A bwd class.
# ---------------------------------------------------------------------------
class StreamingMoeABwd(StreamingMoeY):
    """Streaming-MoE kernel A bwd: NN GEMM `dL/dswiglu_in @ W1` with fused
    atomic-scatter epilogue into `dL_dx_per_r`.

    Strict subset of fwd kernel_y — no per-slot weight multiply, otherwise
    structurally identical. Inherits the AtomicScatterStore EpiOp,
    `epi_subtile_store` (per-warp coalesced predicated v4 bf16x2 atomic-add),
    `epi_setup_postact`, and `epi_convert_postact` unchanged.

    Overrides:
      - `_epi_ops`: drop ColVecLoad — bwd has no per-slot weighting (the
        forward pool_topk_weight was absorbed into dL/dswiglu_in upstream
        by kernel_y_bwd's `dswiglu` + post-multiply on (dgate, dup);
        chain-rule linearity in dpostact bakes the weight into the
        dgate/dup pair we read here).
      - `EpilogueArguments`: drop `mColVecBroadcast`.
      - `epi_visit_subtile`: no-op — kernel_y's weight multiply has no
        analogue here.
      - `__call__` + `get_scheduler_arguments`: reuse the shared
        ``StreamingTileSchedulerOptions``; caller plumbs
        ``bwd_dispatch_arrival_count`` / ``pool_arrival_target`` (the same
        pair Y_bwd waited on; at-target by the time A_bwd runs because
        Y_bwd and A_bwd share a compute stream).
    """

    _epi_ops = (AtomicScatterStore("scatter"),)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        scatter: ScatterParams

    # `epi_to_underlying_arguments` inherited (the parent's version uses
    # `self._epi_ops_to_params_dict(args)`, which respects our narrower
    # `_epi_ops` tuple).

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        """No weight multiply — strict subset of fwd kernel_y's epilogue.

        The per-slot `pool_topk_weight` was baked into `dL/dswiglu_in` upstream
        by kernel_y_bwd's epilogue (which post-multiplied (dgate, dup) by the
        weight after `dswiglu`, exploiting chain-rule linearity in dpostact).
        By the time we run this kernel, the data-grad GEMM result
        `dL/dpool = dL/dswiglu_in @ W1` lands ALREADY-weighted in
        `tRS_rD`, so the atomic-scatter into `dL_dx_per_r` is a plain sum.
        """
        return None

    # -- scheduler hooks -----------------------------------------------------

    def get_scheduler_class(self, varlen_m: bool = False):
        return StreamingTileScheduler

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,  # dL_dswiglu_in: (TK_padded, 2I), k-major
        mB: cute.Tensor,  # W1 permuted: (H, 2I, E_local), n-major
        mD: Optional[cute.Tensor],  # None — output via atomic-scatter
        scheduler_args: StreamingTileSchedulerOptions,
        varlen_args: VarlenArguments,
        epilogue_args,
    ):
        # mB shape is (n=H, k=2I, l=E_local); n-dim tile count = ceil(H / tile_N).
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

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args,
        scheduler_args: StreamingTileSchedulerOptions,
        varlen_args: Optional[VarlenArguments],
        stream: cuda.CUstream,
        trace_ptr: Optional[Int64] = None,
    ):
        """Type-shim override so CuTeDSL accepts StreamingTileSchedulerOptions
        as the scheduler_args type. Body delegates to GemmSm90.__call__.
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
            trace_ptr,
        )


# ---------------------------------------------------------------------------
# JIT compile factory.
# ---------------------------------------------------------------------------
@jit_cache
def _compile_streaming_moe_a_bwd(
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    o_dtype: Type[cutlass.Numeric],
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    cluster_n: int,
    device_capacity,
):
    assert device_capacity[0] == 9, "Streaming MoE kernel A bwd is SM90-only for now"

    H_sym = cute.sym_int()
    I2_sym = cute.sym_int()
    E_sym = cute.sym_int()
    TK_padded_sym = cute.sym_int()
    Trecv_sym = cute.sym_int()
    total_tiles_sym = cute.sym_int()
    cu_seqlens_len_sym = cute.sym_int()  # = E_local + 1

    # A: dL_dswiglu_in flat (TK_padded, 2I), k-major (2I contracted along K,
    # contiguous in storage). Same pool-major flattening fwd kernel A applies
    # to pool.
    mA = fake_tensor(a_dtype, (TK_padded_sym, I2_sym), leading_dim=1, divisibility=8)
    # B: W1 permuted to (H, 2I, E_local), n-major (H contiguous along the
    # leading axis after `W1.permute(2, 1, 0)` on the host). With this layout
    # the kernel's contraction `Σ_k B[n, k]` evaluates `W1[k, n] = W1[2I_idx,
    # h]`, i.e. the NN GEMM `dL_dswiglu_in @ W1` we want.
    mB = fake_tensor(b_dtype, (H_sym, I2_sym, E_sym), leading_dim=0, divisibility=8)
    # No D / C — streaming kernel A bwd outputs via predicated atomic-scatter
    # into `dL_dx_per_r[T_recv, H]` (no trash row).
    mD = None
    mC = None

    # Scatter destination: (T_recv, H), n-major.
    mO = fake_tensor(o_dtype, (Trecv_sym, H_sym), leading_dim=1, divisibility=8)
    pool_recv_token = fake_tensor(
        cutlass.Int32, (TK_padded_sym,), leading_dim=0, divisibility=1
    )
    bwd_k_local_remaining = fake_tensor(
        cutlass.Int32, (cute.sym_int(),), leading_dim=0, divisibility=1
    )
    bwd_a_done_per_token = fake_tensor(
        cutlass.Int64, (cute.sym_int(),), leading_dim=0, divisibility=1
    )

    tile_n_stripes_done = fake_tensor(
        cutlass.Int32, (total_tiles_sym,), leading_dim=0, divisibility=1
    )
    expert_pool_block_offset_scatter = fake_tensor(
        cutlass.Int32, (cu_seqlens_len_sym,), leading_dim=0, divisibility=1
    )
    scatter = ScatterParams(
        mO=mO,
        pool_recv_token=pool_recv_token,
        k_local_remaining=bwd_k_local_remaining,
        y_done_per_token=bwd_a_done_per_token,
        tile_n_stripes_done=tile_n_stripes_done,
        expert_pool_block_offset=expert_pool_block_offset_scatter,
        T_recv=Int32(0),
        combine_seq=Int64(0),
        num_pid_n=Int32(0),
    )

    # cu_seqlens_m drives the standard varlen_m m-offset for mA's pool read.
    # Length E_local + 1, each entry = expert_pool_block_offset[e] * tile_m.
    mCuSeqlensM = fake_tensor(
        cutlass.Int32, (cu_seqlens_len_sym,), leading_dim=0, divisibility=1
    )

    consumer_head = fake_tensor(cutlass.Int32, (cute.sym_int(),), divisibility=1)
    bwd_dispatch_arrival_count = fake_tensor(
        cutlass.Int32, (total_tiles_sym,), divisibility=1
    )
    pool_arrival_target = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)
    expert_pool_block_offset = fake_tensor(
        cutlass.Int32, (cu_seqlens_len_sym,), divisibility=1
    )

    started_flag_fake = fake_tensor(cutlass.Int32, (cute.sym_int(),), divisibility=1)

    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(0),
        consumer_head=consumer_head,
        pool_arrival_count=bwd_dispatch_arrival_count,
        pool_arrival_target=pool_arrival_target,
        expert_pool_block_offset=expert_pool_block_offset,
        total_tiles=Int32(0),
        started_flag=started_flag_fake,
    )

    epi_args = StreamingMoeABwd.EpilogueArguments(scatter=scatter)

    varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlensM, mCuSeqlensK=None, mAIdx=None)

    return compile_gemm_kernel(
        StreamingMoeABwd,
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
# Host wrapper.
# ---------------------------------------------------------------------------
def streaming_moe_a_bwd(
    dL_dswiglu_in: torch.Tensor,  # (total_tiles, tile_m, 2*I) bf16
    W1: torch.Tensor,  # (E_local, 2*I, H) bf16 — k-major per expert
    dL_dx_per_r: torch.Tensor,  # (T_recv, H) bf16 — atomic-scatter destination
    pool_recv_token: torch.Tensor,  # (TK_padded,) int32
    bwd_k_local_remaining: torch.Tensor,  # (T_recv,) int32 — initialised from K_local_count
    bwd_a_done_per_token: torch.Tensor,  # (T_recv,) int64 — zero-init
    expert_pool_block_offset: torch.Tensor,  # (E_local + 1,) int32
    bwd_dispatch_arrival_count: torch.Tensor,  # (total_tiles,) int32 — dispatch_grads's per-tile counter (at-target by the time A_bwd runs)
    pool_arrival_target: torch.Tensor,  # (total_tiles,) int32 — per-tile firing target (shared with fwd)
    dispatch_seq: int,
    *,
    started_flag: torch.Tensor | None = None,  # (1,) int32 buffer-owned cross-stream launch-gate flag; first CTA bumps it. None → allocate throwaway (tests / standalone harnesses).
    tile_m: int = 128,
    tile_n: int = 128,
    cluster_m: int = 1,
    cluster_n: int = 1,
    num_sms: int | None = None,
) -> None:
    """Launch streaming-MoE kernel A bwd on the caller's current CUDA stream.

    Computes the data-grad ``dL/dpool = dL_dswiglu_in @ W1`` per tile and
    atomic-scatters into ``dL_dx_per_r[r, :H]`` (one logical recv-token row
    per pool slot via ``pool_recv_token[slot]``). Per-token bookkeeping
    (atomicSub on ``bwd_k_local_remaining`` + hit-zero release-store on
    ``bwd_a_done_per_token``) mirrors fwd kernel_y exactly so the
    combine_grads sender's per-token gate fires once kernel_a_bwd's last
    contributor for each recv-token has scattered.

    Kernel A_bwd runs on the same compute stream as kernel_y_bwd; same-stream
    FIFO covers cross-stage visibility, so the scheduler reuses
    dispatch_grads's per-tile pair (``bwd_dispatch_arrival_count`` /
    ``pool_arrival_target``). Both are at-target when A_bwd starts because
    Y_bwd already spun on them.

    ``num_sms`` caps the persistent-grid CTA count. ``None`` (default) fills
    the GPU.

    Caller is responsible for:
      - allocating ``dL_dx_per_r`` zero-init ON THE SAME STREAM this
        function runs on (so the kernel's first atomic-add lands on a
        known-zero buffer; otherwise stale memory leaks through).
      - allocating ``bwd_k_local_remaining`` initialised to
        ``K_local_count[r]`` for each recv-token (the bwd orchestrator
        ``cudaMemcpyAsync``'s this from the saved handle).
      - allocating ``bwd_a_done_per_token`` zero-init.

    The internal ``consumer_head`` and ``tile_n_stripes_done`` counters are
    allocated on the calling stream so their zero-init is naturally ordered
    with the kernel.
    """
    assert dL_dswiglu_in.is_cuda and W1.is_cuda and dL_dx_per_r.is_cuda
    assert dL_dswiglu_in.dim() == 3
    assert W1.dim() == 3
    assert dL_dx_per_r.dim() == 2
    total_tiles, in_tile_m, two_I = dL_dswiglu_in.shape
    assert (
        in_tile_m == tile_m
    ), f"dL_dswiglu_in middle dim must equal tile_m={tile_m}; got {in_tile_m}"
    T_recv, H = dL_dx_per_r.shape
    E_local = W1.shape[0]
    assert W1.shape == (
        E_local,
        two_I,
        H,
    ), f"W1 must be (E_local, 2*I, H) = {(E_local, two_I, H)}; got {tuple(W1.shape)}"
    # tile_n MUST divide the output N dim (= H). Non-divisible tile_n produces
    # silently-wrong stores in the data-grad GEMM's partial-tile path (the
    # bug that bit production at H=2048 with tile_n=192).
    assert H % tile_n == 0, (
        f"tile_n ({tile_n}) must divide H ({H}); H % tile_n = {H % tile_n}"
    )
    assert pool_recv_token.shape == (total_tiles * tile_m,)
    assert pool_recv_token.dtype == torch.int32
    assert bwd_k_local_remaining.shape == (T_recv,)
    assert bwd_k_local_remaining.dtype == torch.int32
    assert bwd_a_done_per_token.shape == (T_recv,)
    assert bwd_a_done_per_token.dtype == torch.int64
    assert expert_pool_block_offset.shape == (E_local + 1,)
    assert (
        bwd_dispatch_arrival_count.shape == (total_tiles,)
        and bwd_dispatch_arrival_count.dtype == torch.int32
    )
    assert (
        pool_arrival_target.shape == (total_tiles,)
        and pool_arrival_target.dtype == torch.int32
    )

    # Caller passes W1 as (E_local, 2I, H) k-major contiguous (each expert's
    # slab has H contiguous along the last axis — same layout fwd kernel A
    # consumes). For the bwd's NN GEMM we need the kernel-side tensor to be
    # (n=H, k=2I, l=E_local) with H contiguous (n-major), so the mainloop's
    # `Σ_k B[n, k]` evaluates `W1[2I_idx, h]`. `permute(2, 1, 0)` gives this
    # layout WITHOUT a copy: strides go (2I*H, H, 1) → (1, H, 2I*H).
    W1_p = W1.permute(2, 1, 0)
    assert W1_p.stride(0) == 1, (
        "W1.permute(2,1,0) must have H (axis 0) contiguous (n-major B); caller "
        "must pass W1 as (E_local, 2*I, H) k-major"
    )
    assert W1_p.shape == (H, two_I, E_local)

    # Flatten dL_dswiglu_in's leading two dims to (total_tiles * tile_m, 2I)
    # so the kernel sees a single varlen_m M dimension.
    dL_dswiglu_in_flat = dL_dswiglu_in.view(total_tiles * tile_m, two_I)

    # cu_seqlens_m = expert_pool_block_offset * tile_m drives the standard
    # varlen_m m-offset for mA's pool-major read:
    #   m_offset(tile) = cu_seqlens_m[batch_idx] + pid_m * tile_m = tile_id * tile_m
    cu_seqlens_m = (expert_pool_block_offset.to(torch.int32) * tile_m).contiguous()

    device_capacity = get_device_capacity(dL_dswiglu_in.device)
    assert device_capacity[0] == 9, "Streaming MoE kernel A bwd is SM90-only for now"

    a_dtype = torch2cute_dtype_map[dL_dswiglu_in.dtype]
    b_dtype = torch2cute_dtype_map[W1.dtype]
    o_dtype = torch2cute_dtype_map[dL_dx_per_r.dtype]

    compiled_fn = _compile_streaming_moe_a_bwd(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        o_dtype=o_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        device_capacity=device_capacity,
    )

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_m * cluster_n)
    if num_sms is not None:
        max_active_clusters = min(
            max_active_clusters, num_sms // (cluster_m * cluster_n)
        )

    # Internal scheduler counter — allocate on the calling stream so the
    # zero-init is naturally ordered with the kernel's first atomic-claim.
    consumer_head = torch.zeros(1, dtype=torch.int32, device=dL_dswiglu_in.device)

    # Multi-pid_n N-stripe arrival counter. The CTA whose atomic-add returns
    # `num_pid_n - 1` is the last N-stripe to complete for its tile_id and
    # owns the per-row bookkeeping (atomicSub bwd_k_local_remaining +
    # hit-zero release-store on bwd_a_done_per_token).
    num_pid_n = (H + tile_n - 1) // tile_n
    tile_n_stripes_done = torch.zeros(
        total_tiles, dtype=torch.int32, device=dL_dswiglu_in.device
    )

    scatter = ScatterParams(
        mO=dL_dx_per_r,
        pool_recv_token=pool_recv_token,
        k_local_remaining=bwd_k_local_remaining,
        y_done_per_token=bwd_a_done_per_token,
        tile_n_stripes_done=tile_n_stripes_done,
        expert_pool_block_offset=expert_pool_block_offset,
        T_recv=Int32(T_recv),
        combine_seq=Int64(dispatch_seq),
        num_pid_n=Int32(num_pid_n),
    )
    epi_args = StreamingMoeABwd.EpilogueArguments(scatter=scatter)
    if started_flag is None:
        started_flag = torch.zeros(1, dtype=torch.int32, device=dL_dswiglu_in.device)
    assert started_flag.shape == (1,) and started_flag.dtype == torch.int32
    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        consumer_head=consumer_head,
        pool_arrival_count=bwd_dispatch_arrival_count,
        pool_arrival_target=pool_arrival_target,
        expert_pool_block_offset=expert_pool_block_offset,
        total_tiles=Int32(total_tiles),
        started_flag=started_flag,
    )
    varlen_args = VarlenArguments(
        mCuSeqlensM=cu_seqlens_m, mCuSeqlensK=None, mAIdx=None
    )

    compiled_fn(
        dL_dswiglu_in_flat,
        W1_p,
        None,
        None,
        epi_args,
        scheduler_args,
        varlen_args,
        None,
    )
