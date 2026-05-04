"""Streaming-MoE kernel Y bwd (CuTeDSL, SM90, pool layout).

Backward of fwd kernel Y. Per the chain rule on `o = postact_a @ W2.T`:
  dL/dpostact_a = (dL/do @ W2) * pool_topk_weight   (per-row weighted)

Per tile:
  * Streaming scheduler acquire-spins on `bwd_y_ready[tile_id] >= dispatch_seq`.
  * Standard varlen_m strided TMA load of `dL_do_pool[tile_id * tile_M : ..., :]`
    (the row offset `cu_seqlens_m[expert_id] + pid_m * tile_m` lands on the
    correct pool-major row by construction — same path fwd kernel A uses on
    pool).
  * NN GEMM against `W2[expert_id]`. mB is W2 permuted to (I, H, E_local) with
    I contiguous (leading_dim=0, n-major). The kernel-internal contraction
    Σ_k A[m, k] * B[n, k] then evaluates to
      g[m, i] = Σ_h dL_do_pool[m, h] * W2[h, i]   = (dL_do_pool @ W2)[m, i]
    so g lands in registers as the unweighted gradient w.r.t. postact_a.
  * In-register multiply by `pool_topk_weight[slot]` via ColVecLoad
    (varlen_m-aware), yielding the weighted dL/dpostact_a in registers
    (mirrors fwd kernel Y's epilogue weight multiply, just on the bwd path).
  * TMA-store dL/dpostact_a to `dL_dpostact_a[tile_id * tile_m : ..., :I]`
    (default mD epilogue path inherited from GemmSm90 — pool-major output,
    same expert-major contiguity that lets dW2 land as a single grouped GEMM
    after this kernel finishes).
  * Per-tile end: drain TMA stores, multi-pid_n gate, release-store
    `bwd_a_ready[tile_id] = dispatch_seq` so kernel_a_bwd on a different
    stream can acquire-load with cross-stream visibility.

Shares streaming machinery with fwd kernels:
  * `StreamingTileScheduler` for linear-claim + per-tile ready spin
    (substitutions: `tile_ready` → `bwd_y_ready`, `dispatch_seq` from saved
    handle).
  * `TileReadyRelease` EpiOp from `streaming_kernel_a.py` (per-tile drain +
    multi-pid_n gating + system-scope release-store) — bwd reuses verbatim;
    only the destination tensor changes (bwd_a_ready instead of a_ready).

dL/dweight[slot] (the per-recv-slot dot product `postact_a[slot] · g[slot]`)
is intentionally NOT computed here yet. The plan: pick up `preact_a` (saved
from fwd kernel A's mD TMA-store path) as a per-tile gmem load via a
custom EpiOp, recompute postact_a in registers via the same paired-N
silu·mul fwd kernel A uses (`postact[i] = silu(preact[2i]) * preact[2i+1]`
— ~10 KFLOP/slot, element-wise, fully overlaps with the GEMM mainloop),
then accumulate `Σ_n postact[m, n] * g[m, n]` into a ColVecReduce buffer
for per-row dL/dweight. preact-based rather than postact-based because
save-preact / recompute-postact is the strictly better trade (preact
would otherwise require a full `pool @ W1.T` GEMM in kernel_a_bwd to
recover; postact recomputes element-wise — see bwd.md §"Pre-SwiGLU save
vs recompute"). Wired in once the surrounding combine_grads orchestration
is ready to consume `weight_grads[slot]`.
"""

from typing import NamedTuple, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.cache_utils import COMPILE_ONLY, jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    ParamsBase,
    get_device_capacity,
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_tvm_ffi_utils import compile_gemm_kernel
from quack.rounding import RoundingMode
from quack.tile_scheduler import PersistenceMode
from quack.varlen_utils import VarlenArguments

from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_a import (
    StreamingTileSchedulerOptions,
    TileReadyParams,
    TileReadyRelease,
)
from evolutionaryscale.models.moe.streaming_moe.streaming_tile_scheduler import (
    StreamingTileScheduler,
    StreamingTileSchedulerArguments,
)


# ---------------------------------------------------------------------------
# Streaming kernel Y bwd class.
# ---------------------------------------------------------------------------
class StreamingMoeYBwdSm90(GemmDefaultEpiMixin, GemmSm90):
    """Streaming-MoE kernel Y bwd: NN GEMM + per-row weight multiply + per-tile
    bwd_a_ready release.

    Inherits the standard mD TMA-store path from GemmDefaultEpiMixin. The only
    epilogue divergence from the default is a multiplicative ColVec broadcast
    (vs the default's additive one) — same override fwd kernel Y uses.

    Adds `TileReadyRelease("tile_ready")` to the inherited `_epi_ops` chain so
    that each tile's epilogue ends with a system-scope release-store on
    `bwd_a_ready[tile_id]` after the mD TMA stores have drained.
    """

    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, TileReadyRelease("tile_ready"))
    _epi_param_bases = (ParamsBase,)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        tile_ready: TileReadyParams
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops + _extra_param_fields by
    # ComposableEpiMixin.__init_subclass__.

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        """In-register weight multiply on the MMA accumulator subtile.

        Replaces GemmDefaultEpiMixin's additive ColVec branch with a
        multiplicative one — `pool_topk_weight[slot]` scales the per-row
        gradient instead of adding into it. Same pattern as fwd kernel Y's
        override.
        """
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        if const_expr(tDrColVec is not None):
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                tRS_rD[i] *= tDrColVec[i]
        return None

    # -- scheduler hooks -----------------------------------------------------

    def get_scheduler_class(self, varlen_m: bool = False):
        return StreamingTileScheduler

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,  # dL_do_pool: (TK_padded, H)
        mB: cute.Tensor,  # W2 permuted: (I, H, E_local), n-major
        mD: Optional[cute.Tensor],  # dL_dpostact_a flat: (Mflat, I)
        scheduler_args: StreamingTileSchedulerOptions,
        varlen_args: VarlenArguments,
        epilogue_args,
    ):
        # mB shape is (n=I, k=H, l=E_local); n-dim tile count = ceil(I / tile_N).
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
def _compile_streaming_moe_y_bwd(
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    cluster_n: int,
    device_capacity,
):
    assert device_capacity[0] == 9, "Streaming MoE kernel Y bwd is SM90-only for now"

    H_sym = cute.sym_int()
    I_sym = cute.sym_int()
    E_sym = cute.sym_int()
    TK_padded_sym = cute.sym_int()
    Mflat_sym = cute.sym_int()  # total_tiles * tile_m, in dL_dpostact_a's M dim
    total_tiles_sym = cute.sym_int()
    cu_seqlens_len_sym = cute.sym_int()  # E_local + 1 at runtime

    # A: dL_do_pool (TK_padded, H), k-major (H is contiguous; same layout fwd
    # kernel A uses on pool).
    mA = fake_tensor(a_dtype, (TK_padded_sym, H_sym), leading_dim=1, divisibility=8)
    # B: W2 permuted to (I, H, E_local), n-major (I is contiguous along the
    # leading axis after `W2.permute(2, 1, 0)` on the host). With this layout
    # the kernel's contraction Σ_k B[n, k] yields W2[k, n] = W2[h, i], i.e. the
    # NN GEMM dL_do_pool @ W2 we want.
    mB = fake_tensor(b_dtype, (I_sym, H_sym, E_sym), leading_dim=0, divisibility=8)
    # D: dL_dpostact_a flat (Mflat, I), n-major (I contiguous) — same shape +
    # layout as fwd kernel A's postact_a output (the bwd's mD plays the role
    # postact_a played in fwd: a pool-major per-tile slab).
    mD = fake_tensor(d_dtype, (Mflat_sym, I_sym), leading_dim=1, divisibility=8)
    # No C input in the MVP (dL/dweight per-slot fusion will add postact_a
    # here as mC + ColVecReduce on `postact_a · g`).
    mC = None

    # cu_seqlens_m drives the standard varlen_m m-offset for both mA's pool
    # read and mD's TMA-store row offset: cu_seqlens_m[batch_idx] = expert_pool_block_offset[e]
    # * tile_m so cu_seqlens_m[batch_idx] + pid_m * tile_m = tile_id * tile_m.
    mCuSeqlensM = fake_tensor(
        cutlass.Int32, (cu_seqlens_len_sym,), leading_dim=0, divisibility=1
    )

    # ColVecLoad's per-row weight broadcast (varlen_m). Shape (TK_padded,) fp32.
    pool_topk_weight = fake_tensor(
        cutlass.Float32, (TK_padded_sym,), leading_dim=0, divisibility=1
    )

    # Scheduler tensors
    consumer_head = fake_tensor(cutlass.Int32, (cute.sym_int(),), divisibility=1)
    bwd_y_ready = fake_tensor(cutlass.Int64, (total_tiles_sym,), divisibility=1)
    tile_id_to_expert = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)
    expert_pool_block_offset = fake_tensor(
        cutlass.Int32, (cu_seqlens_len_sym,), divisibility=1
    )
    bwd_a_ready = fake_tensor(cutlass.Int64, (total_tiles_sym,), divisibility=1)
    tile_n_stripes_done = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)

    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(0),
        consumer_head=consumer_head,
        tile_ready=bwd_y_ready,
        tile_id_to_expert=tile_id_to_expert,
        expert_pool_block_offset=expert_pool_block_offset,
        dispatch_seq=Int64(0),
        total_tiles=Int32(0),
    )

    tile_ready_params = TileReadyParams(
        a_ready=bwd_a_ready,
        tile_n_stripes_done=tile_n_stripes_done,
        compute_seq=Int64(0),
        num_pid_n=Int32(0),
        tile_m=tile_m,
    )

    epi_args = StreamingMoeYBwdSm90.EpilogueArguments(
        tile_ready=tile_ready_params,
        mColVecBroadcast=pool_topk_weight,
        rounding_mode=RoundingMode.RN,
    )

    varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlensM, mCuSeqlensK=None, mAIdx=None)

    return compile_gemm_kernel(
        StreamingMoeYBwdSm90,
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
def streaming_moe_y_bwd(
    dL_do_pool: torch.Tensor,  # (TK_padded, H) bf16 — pool-layout incoming gradient
    W2: torch.Tensor,  # (E_local, H, I) bf16 — k-major per expert (same as fwd)
    dL_dpostact_a: torch.Tensor,  # (total_tiles, tile_m, I) bf16 — pool-layout output
    pool_topk_weight: torch.Tensor,  # (TK_padded,) fp32 — per-slot weight (from saved handle)
    tile_id_to_expert: torch.Tensor,  # (total_tiles,) int32
    expert_pool_block_offset: torch.Tensor,  # (E_local + 1,) int32 — pool-block prefix sum
    bwd_y_ready: torch.Tensor,  # (total_tiles,) int64 — input ready stamps (from dispatch_grads)
    bwd_a_ready: torch.Tensor,  # (total_tiles,) int64 — output ready stamps (to kernel_a_bwd)
    dispatch_seq: int,
    *,
    tile_m: int = 128,
    tile_n: int = 256,
    cluster_m: int = 1,
    cluster_n: int = 1,
    num_sms: int | None = None,
) -> None:
    """Launch streaming-MoE kernel Y bwd on the caller's current CUDA stream.

    Computes the weighted gradient w.r.t. postact_a per tile:
      dL_dpostact_a[slot, :I] = pool_topk_weight[slot] * (dL_do_pool[slot, :H] @ W2[e])

    Streamed via per-tile acquire-spin on `bwd_y_ready` and per-tile
    release-store on `bwd_a_ready` (mirror of fwd kernel A's `tile_ready` /
    `a_ready` handshake — different tensors, identical signaling).

    ``num_sms`` caps the persistent-grid CTA count. ``None`` (default) fills the
    GPU; smaller caps leave SMs available for the other backward kernels to
    overlap.

    Caller is responsible for:
      - allocating ``dL_dpostact_a`` ``(total_tiles, tile_m, I)`` ON THE SAME
        STREAM this function is called from (so the kernel's TMA stores are
        naturally ordered with the allocation).
      - ensuring ``bwd_y_ready`` is populated by the producer
        (``dispatch_grads_main_kernel``'s Pass 2 or a test stub) on a stream
        that release-stores ``bwd_y_ready[tile_id] = dispatch_seq`` once the
        tile's ``dL_do_pool`` rows are ready. The per-tile acquire-spin
        handles cross-stream visibility.

    The internal ``consumer_head`` and ``tile_n_stripes_done`` counters are
    allocated on the calling stream so their zero-init is naturally ordered
    with the kernel.

    Per-tile bwd_a_ready release: at the end of each tile's epilogue (after
    all pid_n N-stripes have drained their TMA stores), the kernel
    release-stores ``bwd_a_ready[tile_id] = dispatch_seq`` with system scope.
    Kernel_a_bwd on its compute_a stream acquire-spins on this signal before
    reading the tile's ``dL_dpostact_a`` slab. Multi-pid_n gating via an
    atomic-add to ``tile_n_stripes_done[tile_id]`` ensures the release fires
    once per tile, not once per N-stripe.

    NOTE: dL/dweight per-slot scalars are NOT computed yet — to be fused into
    this kernel's epilogue (load postact_a as mC + ColVecReduce on the
    in-register ``postact_a · g`` dot product) before combine_grads is wired
    up. Fusing into the same tile keeps both Y-side gradients off the second
    pass over pool that a separate dW/weight kernel would entail.
    """
    assert dL_do_pool.is_cuda and W2.is_cuda and dL_dpostact_a.is_cuda
    assert dL_do_pool.dim() == 2 and dL_do_pool.is_contiguous()
    assert W2.dim() == 3
    assert dL_dpostact_a.dim() == 3
    total_tiles, postact_tile_m, I = dL_dpostact_a.shape
    assert postact_tile_m == tile_m
    H = dL_do_pool.shape[1]
    E_local = W2.shape[0]
    assert W2.shape == (E_local, H, I), (
        f"W2 must be (E_local, H, I); got {tuple(W2.shape)}, expected "
        f"{(E_local, H, I)}"
    )
    assert expert_pool_block_offset.shape == (E_local + 1,)
    assert pool_topk_weight.shape == (dL_do_pool.shape[0],)
    assert pool_topk_weight.dtype == torch.float32
    assert tile_id_to_expert.shape == (total_tiles,)
    assert bwd_y_ready.shape == (total_tiles,) and bwd_y_ready.dtype == torch.int64
    assert bwd_a_ready.shape == (total_tiles,) and bwd_a_ready.dtype == torch.int64

    # Caller passes W2 as (E_local, H, I) k-major (I contiguous; same layout
    # used by fwd kernel Y). For the bwd's NN GEMM we need the kernel-side
    # tensor to be (n=I, k=H, l=E_local) with I contiguous (n-major), so the
    # mainloop's `Σ_k B[n, k]` evaluates `W2[h, i]`. `permute(2, 1, 0)` gives
    # this layout WITHOUT a copy: strides go (H*I, I, 1) → (1, I, H*I).
    W2_p = W2.permute(2, 1, 0)
    assert W2_p.stride(0) == 1, (
        "W2.permute(2,1,0) must have I (axis 0) contiguous (n-major B); caller "
        "must pass W2 as (E_local, H, I) k-major"
    )
    assert W2_p.shape == (I, H, E_local)

    # Flatten dL_dpostact_a's leading two dims to (total_tiles * tile_m, I) so
    # the kernel sees a single varlen_m M dimension.
    dL_dpostact_flat = dL_dpostact_a.view(total_tiles * tile_m, I)

    # Build cu_seqlens_m = expert_pool_block_offset * tile_m. The varlen_m
    # path's per-batch m-row offset becomes
    #   m_offset(tile) = cu_seqlens_m[batch_idx] + pid_m * tile_m = tile_id * tile_m
    # which lands at the correct pool row (pool/dL_do_pool/dL_dpostact_a are
    # all expert-major contiguous in tile_id order by construction).
    cu_seqlens_m = (expert_pool_block_offset.to(torch.int32) * tile_m).contiguous()

    device_capacity = get_device_capacity(dL_do_pool.device)
    assert device_capacity[0] == 9, "Streaming MoE kernel Y bwd is SM90-only for now"

    a_dtype = torch2cute_dtype_map[dL_do_pool.dtype]
    b_dtype = torch2cute_dtype_map[W2.dtype]
    d_dtype = torch2cute_dtype_map[dL_dpostact_a.dtype]

    compiled_fn = _compile_streaming_moe_y_bwd(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        d_dtype=d_dtype,
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
    consumer_head = torch.zeros(1, dtype=torch.int32, device=dL_do_pool.device)

    # Multi-pid_n N-stripe arrival counter. The CTA whose atomic-add returns
    # `num_pid_n - 1` is the last N-stripe to complete for its tile_id and
    # fires the per-tile bwd_a_ready release-store.
    num_pid_n = (I + tile_n - 1) // tile_n
    tile_n_stripes_done = torch.zeros(
        total_tiles, dtype=torch.int32, device=dL_do_pool.device
    )

    tile_ready_params = TileReadyParams(
        a_ready=bwd_a_ready,
        tile_n_stripes_done=tile_n_stripes_done,
        compute_seq=Int64(dispatch_seq),
        num_pid_n=Int32(num_pid_n),
        tile_m=None,  # Constexpr; burned in at compile, pass None at call time
    )

    epi_args = StreamingMoeYBwdSm90.EpilogueArguments(
        tile_ready=tile_ready_params,
        mColVecBroadcast=pool_topk_weight,
        rounding_mode=None,  # Constexpr; pass None at call time
    )
    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        consumer_head=consumer_head,
        tile_ready=bwd_y_ready,
        tile_id_to_expert=tile_id_to_expert,
        expert_pool_block_offset=expert_pool_block_offset,
        dispatch_seq=Int64(dispatch_seq),
        total_tiles=Int32(total_tiles),
    )
    varlen_args = VarlenArguments(
        mCuSeqlensM=cu_seqlens_m, mCuSeqlensK=None, mAIdx=None
    )

    compiled_fn(
        dL_do_pool,
        W2_p,
        dL_dpostact_flat,
        None,
        epi_args,
        scheduler_args,
        varlen_args,
        None,
    )
