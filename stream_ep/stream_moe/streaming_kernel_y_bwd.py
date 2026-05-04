"""Streaming-MoE kernel Y bwd (CuTeDSL, SM90, pool layout).

Backward of fwd kernel Y, with SwiGLU bwd folded into the epilogue. Per
chain rule on `o = postact_a @ W2.T` and `postact_a = silu(gate) * up`:
  g[slot, :]               = dL_do_pool[slot, :] @ W2[expert_for_slot]
  dL/dpostact_a[slot, n]   = pool_topk_weight[slot] * g[slot, n]
  (dgate, dup, _)          = dswiglu(gate, up, dL/dpostact_a)
                           = (silu_grad(gate) * up * dpostact, silu(gate) * dpostact, postact)
  dL/dswiglu_in[slot, 2n]  = dgate;  dL/dswiglu_in[slot, 2n+1] = dup
  dL/dweight[slot]         = Σ_n postact[slot, n] * g[slot, n]   (UNWEIGHTED g)

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
  * **Epilogue: SwiGLU bwd + dL/dweight + dL/dswiglu_in store** (all
    register-resident). mC is `preact_a[tile, :2I]` — the pre-SwiGLU gate-up
    accumulator saved by fwd kernel A's mD TMA-store path. Storage is
    `(tile_m, 2I) bf16`; presented to the kernel as `(tile_m, I) fp32` via a
    host-side `.view(torch.float32)` (each fp32 element packs `(gate_i, up_i)`
    as bf16x2 — same f32-recast trick quack's `gemm_dgated` uses on its
    `PreAct` input). In epilogue:
    1. `tRS_rC` (fp32) is recast to bf16x2 via `cute.recast_tensor`, promoted
       to fp32 → (gate, up) f32 pairs.
    2. `dswiglu(gate, up, g_unweighted) → (dgate, dup, postact)` per element.
       Returns recomputed postact as a free byproduct — used directly for
       the ColVecReduce dot product without a separate silu·mul step.
    3. `ColVecReduce` accumulates `Σ_n postact[m, n] * g[m, n]` (UNWEIGHTED g)
       per row → per-(slot, N-stripe) fp32 partial sums for dL/dweight; final
       `dL/dweight = sum across N-stripes` (orchestrator does the trailing
       sum, ~1-line torch op).
    4. Per-row weight multiply scales (dgate, dup) by `pool_topk_weight[slot]`
       — SwiGLU bwd is linear in dpostact, so multiplying after dswiglu is
       equivalent to multiplying before, and lets the ColVecReduce see the
       unweighted g.
    5. Pack (dgate, dup) bf16x2 → fp32 view; standard mD TMA-store lands the
       result in `dL_dswiglu_in[tile, :2I]` (bf16 (M, 2I) on the host viewed
       as fp32 (M, I) — same f32-recast trick on the output side as on input).
  * Per-tile end: drain TMA stores, multi-pid_n gate, release-store
    `bwd_a_ready[tile_id] = dispatch_seq` so kernel_a_bwd on a different
    stream can acquire-load with cross-stream visibility. Threadfence_system
    inside TileReadyRelease.end() also flushes the per-(slot, N-stripe)
    ColVecReduce stores from each participating CTA, so combine_grads's
    later read of dL/dweight observes them post-fence.

Folding SwiGLU bwd here (vs running it as a separate step before kernel_a_bwd)
saves one read of preact_a from HBM in kernel_a_bwd at the cost of writing
2× the mD bytes (`dL/dswiglu_in [M, 2I]` vs `dL/dpostact_a [M, I]`). Net
~256 MB / layer saved at production. kernel_a_bwd's contract becomes a
vanilla streaming GEMM with a pre-materialised A operand — no in-kernel
SwiGLU bwd, no preact load.

Shares streaming machinery with fwd kernels:
  * `StreamingTileScheduler` for linear-claim + per-tile ready spin
    (substitutions: `tile_ready` → `bwd_y_ready`, `dispatch_seq` from saved
    handle).
  * `TileReadyRelease` EpiOp from `streaming_kernel_a.py` (per-tile drain +
    multi-pid_n gating + system-scope release-store) — bwd reuses verbatim;
    only the destination tensor changes (bwd_a_ready instead of a_ready).
"""

from typing import NamedTuple, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.activation import dswiglu
from quack.cache_utils import COMPILE_ONLY, jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    ParamsBase,
    get_device_capacity,
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.epi_ops import ColVecReduce, colvec_reduce_accumulate
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
    """Streaming-MoE kernel Y bwd: NN GEMM + SwiGLU bwd in epilogue +
    dL/dweight ColVecReduce + per-tile bwd_a_ready release.

    Inherits the standard mD TMA-store path from GemmDefaultEpiMixin. Both
    mC (preact) and mD (dL/dswiglu_in) use the f32-recast trick — host-side
    storage is bf16 (M, 2I), kernel sees fp32 (M, I) and recasts back to
    bf16x2 in-epilogue (`implicit_dtype = bf16`, mirroring quack's
    `gemm_dgated`). Compose the additional bwd-side EpiOps onto the
    inherited chain:
      - ColVecReduce("mColVecReduce") for the per-row dL/dweight dot product
        accumulator (writes per-(slot, N-stripe) partials at end of tile).
      - TileReadyRelease("tile_ready") for the system-scope `bwd_a_ready[tile_id]`
        release after mD TMA-store drain + multi-pid_n gating.

    `epi_visit_subtile` is fully overridden — it runs `dswiglu` against
    UNWEIGHTED `g` (= the GEMM result), gets `(dgate, dup, postact)` in one
    call (postact returned as a free byproduct, fed straight into the
    ColVecReduce dot product), then per-row-weight-multiplies (dgate, dup)
    AFTER the ColVecReduce since SwiGLU bwd is linear in dpostact. The
    per-row weight multiply on (dgate, dup) is equivalent to multiplying
    `g` first then running dswiglu, but lets ColVecReduce see UNWEIGHTED g
    for `dL/dweight = postact · g` (the chain rule).

    `implicit_dtype` (the bf16 dtype that mC AND mD's fp32-view-storage
    actually hold) is set via a `post_init` hook passed to
    `compile_gemm_kernel` — same plumbing quack's `gemm_dgated` uses.
    """

    _epi_ops = (
        *GemmDefaultEpiMixin._epi_ops,
        ColVecReduce("mColVecReduce"),
        TileReadyRelease("tile_ready"),
    )
    _epi_param_bases = (ParamsBase,)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        tile_ready: TileReadyParams
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops + _extra_param_fields by
    # ComposableEpiMixin.__init_subclass__.

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        """SwiGLU bwd in registers: outputs `dL/dswiglu_in` (M, 2I) packed
        as bf16x2 in fp32 via mD's f32-recast trick. ColVecReduce-accumulates
        `dL/dweight = postact · g` using the postact byproduct of `dswiglu`.

        tRS_rC arrives as an fp32 (M, N) register tensor — host-side
        `preact_a.view(torch.float32)` of the bf16 (M, 2N) preact slab (each
        fp32 element packs (gate_n, up_n) as bf16x2). Same f32-recast trick
        quack's `gemm_dgated` uses on its `PreAct` input.

        Order:
          (1) Recast tRS_rC fp32 → bf16x2 → fp32 (gate, up) f32 pairs.
          (2) `dswiglu(gate, up, g_unweighted) → (dgate, dup, postact)` per
              element. `tRS_rD` enters as the UNWEIGHTED GEMM result `g`;
              `dswiglu` returns the recomputed postact as its third element,
              avoiding a separate paired-N silu·mul recompute.
          (3) ColVecReduce-accumulate `Σ_n postact[m, n] * g[m, n]` per row
              (UNWEIGHTED g — chain rule for dL/dweight = postact · g).
          (4) Per-row weight multiply on (dgate, dup): SwiGLU bwd is linear
              in dpostact, so `w * dgate(g) = dgate(w * g)`. Multiplying
              after dswiglu is equivalent and lets ColVecReduce see the
              unweighted g.
          (5) Pack (dgate, dup) bf16x2 → fp32 view; restore in tRS_rD for
              the standard mD store path. mD's host storage is bf16
              (Mflat, 2I), viewed as fp32 (Mflat, I) before launch — the
              kernel sees fp32 mD with implicit_dtype=bf16 packing.
        """
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        tDrColVecReduce = epi_loop_tensors["mColVecReduce"]
        assert tRS_rC is not None, "kernel_y_bwd requires preact via mC"

        implicit_dtype = self.implicit_dtype
        # (1) Recast mC fp32 → bf16x2 → fp32 (gate, up) pair view.
        tRS_rXY_b16 = cute.recast_tensor(tRS_rC, implicit_dtype)
        tRS_rXY_f32 = cute.make_rmem_tensor(tRS_rXY_b16.layout, Float32)
        tRS_rXY_f32.store(tRS_rXY_b16.load().to(Float32))

        # (2) dswiglu on UNWEIGHTED g (= tRS_rD). Returns
        # (dgate, dup, postact) per element. Allocate paired-N output
        # (same layout as tRS_rXY_b16 since both are (M, 2I)).
        tRS_rdXY_f32 = cute.make_rmem_tensor(tRS_rXY_b16.layout, Float32)
        tRS_rPostAct = cute.make_rmem_tensor_like(tRS_rD, Float32)
        for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
            tRS_rdXY_f32[2 * i], tRS_rdXY_f32[2 * i + 1], tRS_rPostAct[i] = dswiglu(
                tRS_rXY_f32[2 * i], tRS_rXY_f32[2 * i + 1], tRS_rD[i]
            )

        # (3) ColVecReduce on UNWEIGHTED g — chain rule for dL/dweight.
        if const_expr(tDrColVecReduce is not None):
            colvec_reduce_accumulate(self, tDrColVecReduce, tRS_rD, rScale=tRS_rPostAct)

        # (4) Per-row weight multiply on (dgate, dup). Equivalent to
        # dswiglu(gate, up, w * g) by linearity in dout, but runs AFTER the
        # ColVecReduce so the dot product sees the unweighted g.
        if const_expr(tDrColVec is not None):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rdXY_f32[2 * i] *= tDrColVec[i]
                tRS_rdXY_f32[2 * i + 1] *= tDrColVec[i]

        # (5) Pack (dgate, dup) bf16x2 → fp32 view in tRS_rD for the
        # standard mD TMA-store. Lands in dL_dswiglu_in[tile, :2I] as bf16
        # via the host-side fp32 view.
        tRS_rdXY_b16 = cute.make_rmem_tensor(tRS_rdXY_f32.layout, implicit_dtype)
        tRS_rdXY_b16.store(tRS_rdXY_f32.load().to(implicit_dtype))
        tRS_rD.store(cute.recast_tensor(tRS_rdXY_b16, Float32).load())
        return None

    # -- scheduler hooks -----------------------------------------------------

    def get_scheduler_class(self, varlen_m: bool = False):
        return StreamingTileScheduler

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,  # dL_do_pool: (TK_padded, H)
        mB: cute.Tensor,  # W2 permuted: (I, H, E_local), n-major
        mD: Optional[cute.Tensor],  # dL_dswiglu_in flat: (Mflat, I) fp32-view
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
    implicit_dtype: Type[cutlass.Numeric],
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
    Mflat_sym = cute.sym_int()  # total_tiles * tile_m, in dL_dswiglu_in's M dim
    total_tiles_sym = cute.sym_int()
    cu_seqlens_len_sym = cute.sym_int()  # E_local + 1 at runtime
    n_tiles_sym = cute.sym_int()  # ceil(I / tile_n) — N-stripe count per tile

    # A: dL_do_pool (TK_padded, H), k-major (H is contiguous; same layout fwd
    # kernel A uses on pool).
    mA = fake_tensor(a_dtype, (TK_padded_sym, H_sym), leading_dim=1, divisibility=8)
    # B: W2 permuted to (I, H, E_local), n-major (I is contiguous along the
    # leading axis after `W2.permute(2, 1, 0)` on the host). With this layout
    # the kernel's contraction Σ_k B[n, k] yields W2[k, n] = W2[h, i], i.e. the
    # NN GEMM dL_do_pool @ W2 we want.
    mB = fake_tensor(b_dtype, (I_sym, H_sym, E_sym), leading_dim=0, divisibility=8)
    # D: dL_dswiglu_in flat. Storage on host is bf16 (Mflat, 2*I); we view as
    # fp32 (Mflat, I) before launch — each fp32 element packs (dgate_n, dup_n)
    # as bf16x2. d_dtype is fp32 (32-bit storage), implicit_dtype is bf16 (the
    # underlying type the kernel recasts to via `cute.recast_tensor` in
    # epi_visit_subtile, on both mC's input side and tRS_rD's output side).
    # divisibility=4 reflects fp32's 16-byte alignment requirement (4 fp32 = 16 B).
    mD = fake_tensor(d_dtype, (Mflat_sym, I_sym), leading_dim=1, divisibility=4)
    # C: preact_a flat. Same f32-recast trick as mD — host storage bf16
    # (Mflat, 2*I), kernel sees fp32 (Mflat, I).
    mC = fake_tensor(cutlass.Float32, (Mflat_sym, I_sym), leading_dim=1, divisibility=4)

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

    # ColVecReduce destination — per-(slot, N-stripe) fp32 partial sums.
    # Shape (Mflat, n_tiles) where n_tiles = ceil(I / tile_n) is the number
    # of N-stripes per tile. Final dL/dweight = sum across n_tiles dim
    # (orchestrator reduces post-hoc; cheap torch op).
    mColVecReduce = fake_tensor(
        cutlass.Float32, (Mflat_sym, n_tiles_sym), leading_dim=1, divisibility=1
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
        mColVecReduce=mColVecReduce,
        rounding_mode=RoundingMode.RN,
    )

    varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlensM, mCuSeqlensK=None, mAIdx=None)

    def _set_implicit_dtype(gemm_obj):
        # Tells epi_visit_subtile that mC's fp32 storage actually packs
        # implicit_dtype (bf16) elements as bf16x2. Same plumbing
        # quack's gemm_dgated uses on its PreAct input.
        gemm_obj.implicit_dtype = implicit_dtype

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
        post_init=_set_implicit_dtype,
    )


# ---------------------------------------------------------------------------
# Host wrapper.
# ---------------------------------------------------------------------------
def streaming_moe_y_bwd(
    dL_do_pool: torch.Tensor,  # (TK_padded, H) bf16 — pool-layout incoming gradient
    W2: torch.Tensor,  # (E_local, H, I) bf16 — k-major per expert (same as fwd)
    dL_dswiglu_in: torch.Tensor,  # (total_tiles, tile_m, 2*I) bf16 — pool-layout output
    pool_topk_weight: torch.Tensor,  # (TK_padded,) fp32 — per-slot weight (from saved handle)
    preact_a: torch.Tensor,  # (total_tiles, tile_m, 2*I) bf16 — saved from fwd kernel A's mD
    dL_dweight_per_stripe: torch.Tensor,  # (TK_padded, num_pid_n) fp32 — ColVecReduce partials
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

    Computes both Y-side gradients in one tile-streamed pass:
      g[slot, :]                = dL_do_pool[slot] @ W2[e]               (unweighted)
      dL_dpostact_a[slot, :I]   = pool_topk_weight[slot] * g[slot, :I]
      (dgate, dup, postact)     = dswiglu(gate, up, dL_dpostact_a)
      dL_dswiglu_in[slot, 2n]   = dgate;  dL_dswiglu_in[slot, 2n+1] = dup
      dL_dweight_per_stripe[slot, n_stripe]  partial-sums Σ_n postact[slot, n] * g[slot, n]

    SwiGLU bwd is folded into the epilogue — the recomputed postact is
    returned by `dswiglu` as a free byproduct (used directly for the
    ColVecReduce dot product), and the per-row weight multiply runs on
    (dgate, dup) AFTER `dswiglu` (chain rule is linear in dpostact).
    Output `dL_dswiglu_in` is the direct input to kernel_a_bwd's data-grad
    GEMM and to dW1's grouped GEMM — no separate SwiGLU-bwd materialisation
    step needed.

    Streamed via per-tile acquire-spin on `bwd_y_ready` and per-tile
    release-store on `bwd_a_ready` (mirror of fwd kernel A's `tile_ready` /
    `a_ready` handshake — different tensors, identical signaling).

    ``num_sms`` caps the persistent-grid CTA count. ``None`` (default) fills the
    GPU; smaller caps leave SMs available for the other backward kernels to
    overlap.

    Caller is responsible for:
      - allocating ``dL_dswiglu_in`` and ``dL_dweight_per_stripe`` ON THE SAME
        STREAM this function is called from (so the kernel's TMA stores +
        ColVecReduce gmem stores are naturally ordered with the allocations).
      - ensuring ``bwd_y_ready`` is populated by the producer
        (``dispatch_grads_main_kernel``'s Pass 2 or a test stub) on a stream
        that release-stores ``bwd_y_ready[tile_id] = dispatch_seq`` once the
        tile's ``dL_do_pool`` rows are ready. The per-tile acquire-spin
        handles cross-stream visibility.
      - reducing ``dL_dweight_per_stripe`` across its N-stripe dim to get the
        final per-slot ``dL_dweight``: ``dL_dweight = dL_dweight_per_stripe.sum(dim=-1)``.
        This is a cheap torch op orchestrated between kernel_y_bwd and
        combine_grads (mirror of sonic-moe's `db2_and_ds_kernel` post-hoc
        sum across H-blocks; we don't need a custom kernel for it).

    The internal ``consumer_head`` and ``tile_n_stripes_done`` counters are
    allocated on the calling stream so their zero-init is naturally ordered
    with the kernel.

    Per-tile bwd_a_ready release: at the end of each tile's epilogue (after
    all pid_n N-stripes have drained their TMA stores), the kernel
    release-stores ``bwd_a_ready[tile_id] = dispatch_seq`` with system scope.
    Kernel_a_bwd on its compute_a stream acquire-spins on this signal before
    reading the tile's ``dL_dswiglu_in`` slab. Multi-pid_n gating via an
    atomic-add to ``tile_n_stripes_done[tile_id]`` ensures the release fires
    once per tile, not once per N-stripe.

    Storage layouts (both use the f32-recast trick — same as quack's
    gemm_dgated):
      - ``preact_a``: bf16 (total_tiles, tile_m, 2*I), viewed as fp32
        (total_tiles*tile_m, I) before launch — each fp32 element packs
        (gate_n, up_n) as bf16x2. Kernel reads as mC.
      - ``dL_dswiglu_in``: bf16 (total_tiles, tile_m, 2*I), viewed as fp32
        (total_tiles*tile_m, I) before launch — each fp32 element packs
        (dgate_n, dup_n) as bf16x2. Kernel writes as mD.
    Both share the same `implicit_dtype` (bf16) which the kernel uses to
    recast tRS_rC / tRS_rD between fp32 storage and bf16x2 math views.
    """
    assert dL_do_pool.is_cuda and W2.is_cuda and dL_dswiglu_in.is_cuda
    assert dL_do_pool.dim() == 2 and dL_do_pool.is_contiguous()
    assert W2.dim() == 3
    assert dL_dswiglu_in.dim() == 3
    total_tiles, dswiglu_tile_m, two_I = dL_dswiglu_in.shape
    assert dswiglu_tile_m == tile_m
    assert two_I % 2 == 0, f"dL_dswiglu_in last dim must be 2*I (gate+up); got {two_I}"
    I = two_I // 2
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
    # preact contract — bf16 (total_tiles, tile_m, 2*I), same dtype as
    # dL_dswiglu_in (both share the f32-recast packing).
    assert preact_a.is_cuda and preact_a.dim() == 3
    assert preact_a.shape == (total_tiles, tile_m, 2 * I), (
        f"preact_a must be (total_tiles, tile_m, 2*I) = "
        f"{(total_tiles, tile_m, 2 * I)}; got {tuple(preact_a.shape)}"
    )
    assert preact_a.dtype == dL_dswiglu_in.dtype, (
        f"preact_a dtype must match dL_dswiglu_in's; got {preact_a.dtype} vs "
        f"{dL_dswiglu_in.dtype}"
    )
    assert (
        preact_a.element_size() == 2
    ), "preact_a must be 16-bit (bf16/fp16) for the f32-recast trick"
    assert (
        dL_dswiglu_in.element_size() == 2
    ), "dL_dswiglu_in must be 16-bit (bf16/fp16) for the f32-recast trick"
    # ColVecReduce partials: per-(slot, N-stripe) fp32. Caller sums across
    # the N-stripe dim post-hoc.
    num_pid_n = (I + tile_n - 1) // tile_n
    assert dL_dweight_per_stripe.is_cuda and dL_dweight_per_stripe.dim() == 2
    assert dL_dweight_per_stripe.shape == (dL_do_pool.shape[0], num_pid_n), (
        f"dL_dweight_per_stripe must be (TK_padded, num_pid_n) = "
        f"({dL_do_pool.shape[0]}, {num_pid_n}); got "
        f"{tuple(dL_dweight_per_stripe.shape)}"
    )
    assert dL_dweight_per_stripe.dtype == torch.float32
    assert dL_dweight_per_stripe.stride(1) == 1, (
        "dL_dweight_per_stripe must be n-major (n_stripes contiguous); "
        "ColVecReduce.end() expects this layout"
    )

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

    # Capture preact's bf16 dtype BEFORE the f32 view (compile-key + post_init).
    # Same `implicit_dtype` is used for both mC (preact in) and mD (dL_dswiglu_in
    # out) — both apply the f32-recast trick on the same underlying bf16 storage.
    preact_implicit_dtype = torch2cute_dtype_map[preact_a.dtype]
    # Flatten preact_a, view as fp32: bf16 (M, 2*I) → fp32 (M, I). Each fp32
    # element packs (gate_n, up_n) as bf16x2; the kernel recasts in
    # epi_visit_subtile via `cute.recast_tensor(tRS_rC, implicit_dtype)`.
    preact_flat = preact_a.view(total_tiles * tile_m, 2 * I).view(torch.float32)
    assert preact_flat.shape == (total_tiles * tile_m, I), (
        f"preact_flat fp32-view shape mismatch: got {tuple(preact_flat.shape)}, "
        f"expected {(total_tiles * tile_m, I)}"
    )
    # Flatten dL_dswiglu_in similarly, view as fp32: bf16 (M, 2*I) → fp32 (M, I).
    # mD's host storage is bf16 (M, 2I); the kernel sees fp32 (M, I) and the
    # epilogue recasts back to bf16x2 for the standard mD store.
    dL_dswiglu_in_flat = dL_dswiglu_in.view(total_tiles * tile_m, 2 * I).view(
        torch.float32
    )
    assert dL_dswiglu_in_flat.shape == (total_tiles * tile_m, I), (
        f"dL_dswiglu_in_flat fp32-view shape mismatch: got "
        f"{tuple(dL_dswiglu_in_flat.shape)}, expected {(total_tiles * tile_m, I)}"
    )

    # Build cu_seqlens_m = expert_pool_block_offset * tile_m. The varlen_m
    # path's per-batch m-row offset becomes
    #   m_offset(tile) = cu_seqlens_m[batch_idx] + pid_m * tile_m = tile_id * tile_m
    # which lands at the correct pool row (pool/dL_do_pool/dL_dswiglu_in are
    # all expert-major contiguous in tile_id order by construction).
    cu_seqlens_m = (expert_pool_block_offset.to(torch.int32) * tile_m).contiguous()

    device_capacity = get_device_capacity(dL_do_pool.device)
    assert device_capacity[0] == 9, "Streaming MoE kernel Y bwd is SM90-only for now"

    a_dtype = torch2cute_dtype_map[dL_do_pool.dtype]
    b_dtype = torch2cute_dtype_map[W2.dtype]
    # d_dtype is fp32 (mD's storage type the framework sees) — the
    # underlying bf16 (M, 2I) packs into fp32 (M, I) via the f32-recast
    # trick. `implicit_dtype = bf16` (the same as mC) tells the kernel
    # how to recast.
    d_dtype = torch2cute_dtype_map[torch.float32]

    compiled_fn = _compile_streaming_moe_y_bwd(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        d_dtype=d_dtype,
        implicit_dtype=preact_implicit_dtype,
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
        mColVecReduce=dL_dweight_per_stripe,
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
        dL_dswiglu_in_flat,
        preact_flat,
        epi_args,
        scheduler_args,
        varlen_args,
        None,
    )
