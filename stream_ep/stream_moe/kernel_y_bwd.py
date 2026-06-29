"""Streaming-MoE kernel Y bwd (CuTeDSL, SM90, pool layout).

Backward of fwd kernel Y, with SwiGLU bwd folded into the epilogue. Per
chain rule on `o = postact_a @ W2.T` and `postact_a = silu(gate) * up`:
  g[slot, :]               = dL_do_pool[slot, :] @ W2[expert_for_slot]
  dL/dpostact_a[slot, n]   = pool_topk_weight[slot] * g[slot, n]
  (dgate, dup, postact)    = dswiglu(gate, up, dL/dpostact_a)
                           = (silu_grad(gate) * up * dpostact, silu(gate) * dpostact, postact)
  dL/dswiglu_in[slot, 2n]  = dgate;  dL/dswiglu_in[slot, 2n+1] = dup
  dL/dweight[slot]         = Σ_n postact[slot, n] * g[slot, n]   (UNWEIGHTED g)
  postact_a_for_dW2[slot]  = pool_topk_weight[slot] * postact[slot]   (WEIGHTED, fed to dW2)

Per tile:
  * Streaming scheduler count-vs-target spins on
    `bwd_dispatch_arrival_count[tile] == pool_arrival_target[tile]` (the same
    target fwd uses; dispatch_grads's Pass 2 re-fires release-adds against it).
  * Standard varlen_m strided TMA load of `dL_do_pool[tile_id * tile_M : ..., :]`
    (the row offset `cu_seqlens_m[expert_id] + pid_m * tile_m` lands on the
    correct pool-major row by construction — same path fwd kernel A uses on
    pool).
  * NN GEMM against `W2[expert_id]`. mB is W2 permuted to (I, H, E_local) with
    I contiguous (leading_dim=0, n-major). The kernel-internal contraction
    Σ_k A[m, k] * B[n, k] then evaluates to
      g[m, i] = Σ_h dL_do_pool[m, h] * W2[h, i]   = (dL_do_pool @ W2)[m, i]
    so g lands in registers as the unweighted gradient w.r.t. postact_a.
  * **Padding predicate.** A second `ColVecLoad("mPaddingMask")` carries
    `pool_recv_token` (int32 → fp32) per row; before the colvec_reduce_accumulate
    and weight multiply, the epilogue conditionally assigns zero to (dgate, dup,
    postact, g) wherever recv_token < 0. Conditional assignment (vs multiply by
    zero) avoids `0 * inf = NaN` propagation from the matmul through padding-row
    garbage in `dL_do_pool` and `preact_a`. Padding rows of `dL_dswiglu_in`,
    `postact_a_for_dW2`, and the per-slot `dL_dweight` accumulator all land as
    clean zero — the host-side zero-init of those four tensors is no longer
    required.
  * **Epilogue: SwiGLU bwd + dL/dweight atomic + dL/dswiglu_in + postact_a_for_dW2 store**
    (all register-resident). mC is `preact_a[tile, :2I]` — the pre-SwiGLU gate-up
    accumulator saved by fwd kernel A's mD TMA-store path. Storage is
    `(tile_m, 2I) bf16`; presented to the kernel as `(tile_m, I) fp32` via a
    host-side `.view(torch.float32)` (each fp32 element packs `(gate_i, up_i)`
    as bf16x2 — same f32-recast trick quack's `gemm_dgated` uses on its
    `PreAct` input). In epilogue:
    1. `tRS_rC` (fp32) is recast to bf16x2 via `cute.recast_tensor`, promoted
       to fp32 → (gate, up) f32 pairs.
    2. `dswiglu(gate, up, g_unweighted) → (dgate, dup, postact)` per element.
       Returns recomputed postact as a free byproduct — used directly for
       both the dL/dweight dot product and the postact_a_for_dW2 store.
    3. `ColVecReduceAtomic` accumulates `Σ_n postact[m, n] * g[m, n]`
       (UNWEIGHTED g) per row, intra-warp shuffle + cross-warp reduce → one
       fp32 row sum per slot per pid_n CTA, then ``red.global.add.f32`` into
       a flat ``dL_dweight[slot]`` fp32 buffer. kernel_a_bwd runs on the
       same compute stream and is FIFO-ordered after Y_bwd retires, so the
       ``dL_dweight`` atomic-adds are visible to A_bwd (and through it to
       combine_grads's per-token gate) without any per-tile cross-stream
       release.
    4. Per-row weight multiply on (dgate, dup, postact): SwiGLU bwd is
       linear in dpostact, so `w * dgate(g) = dgate(w * g)`. Multiplying
       after dswiglu is equivalent and lets the dL/dweight dot product see
       the unweighted g. The same per-row weight is also applied to
       ``postact`` to produce ``postact_a_for_dW2 = w * postact`` for the
       second TMA-stored output (input to dW2's grouped GEMM).
    5. Pack (dgate, dup) bf16x2 → fp32 view; standard mD TMA-store lands
       the result in `dL_dswiglu_in[tile, :2I]` (bf16 (M, 2I) on the host
       viewed as fp32 (M, I) — same f32-recast trick on the output side as
       on input). ``postact_a_for_dW2`` rides ``mAuxOut`` (TileStore) — bf16
       (M, I) plain TMA-store, same path GemmActMixin uses for fwd postact.

Folding SwiGLU bwd here (vs running it as a separate step before kernel_a_bwd)
saves one read of preact_a from HBM in kernel_a_bwd at the cost of writing
2× the mD bytes (`dL/dswiglu_in [M, 2I]` vs `dL/dpostact_a [M, I]`). Net
~256 MB / layer saved at production. kernel_a_bwd's contract becomes a
vanilla streaming GEMM with a pre-materialised A operand — no in-kernel
SwiGLU bwd, no preact load.

Shares streaming machinery with fwd kernels:
  * `StreamingTileScheduler` for linear-claim + per-tile count-vs-target
    spin. Kernel Y_bwd plumbs (``bwd_dispatch_arrival_count``,
    ``pool_arrival_target``) — fired by dispatch_grads's Pass 2 release-adds.
  * kernel_a_bwd runs on the same compute stream and is FIFO-ordered after
    Y_bwd retires; no per-tile Y_bwd → A_bwd device-side signal is needed.
"""

from typing import Callable, NamedTuple, Optional, Type

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
from quack.epi_ops import ColVecLoad, colvec_reduce_accumulate
from quack.gemm_act import GemmActMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_tvm_ffi_utils import compile_gemm_kernel
from quack.rounding import RoundingMode
from quack.tile_scheduler import PersistenceMode
from quack.varlen_utils import VarlenArguments

from stream_ep.stream_moe.epi_ops import (
    ColVecReduceAtomic,
)
from stream_ep.stream_moe.kernel_a import StreamingTileSchedulerOptions
from stream_ep.stream_moe.tile_scheduler import (
    StreamingTileScheduler,
    StreamingTileSchedulerArguments,
)


# ---------------------------------------------------------------------------
# Streaming kernel Y bwd class.
# ---------------------------------------------------------------------------
class StreamingMoeYBwd(GemmActMixin, GemmSm90):
    """Streaming-MoE kernel Y bwd: NN GEMM + SwiGLU bwd in epilogue +
    in-kernel dL/dweight atomic-add + postact_a_for_dW2 TMA-store.

    Inherits the standard mD TMA-store path from GemmDefaultEpiMixin (via
    GemmActMixin), plus a SECOND TMA-store path via ``TileStore("mAuxOut")``
    that we repurpose for ``postact_a_for_dW2`` (host shape bf16 (M, I), no
    f32-recast — plain bf16 store, same layout fwd kernel A's mAuxOut uses).
    Both mC (preact) and mD (dL/dswiglu_in) still use the f32-recast trick —
    host-side storage is bf16 (M, 2I), kernel sees fp32 (M, I) and recasts
    back to bf16x2 in-epilogue (`implicit_dtype = bf16`, mirroring quack's
    `gemm_dgated`). Compose the additional bwd-side EpiOp onto the inherited
    chain:
      - ColVecReduceAtomic("mColVecReduce") for in-kernel atomic-add of the
        per-slot dL/dweight dot product (per-pid_n CTAs reduce intra-warp /
        cross-warp then ``red.global.add.f32`` into a flat ``(M,)`` fp32
        buffer; the in-kernel atomic collapses across pid_n stripes so no
        post-hoc ``.sum(dim=-1)`` or cross-stream event is needed).

    kernel_a_bwd runs on the same compute stream and FIFO-orders after Y_bwd
    retires, so dL_dweight's atomic-adds and the mAuxOut TMA stores are
    automatically visible — no per-stripe-CTA release-add is needed.

    `epi_visit_subtile` is fully overridden — runs `dswiglu` against
    UNWEIGHTED `g` (= the GEMM result), gets `(dgate, dup, postact)` in one
    call (postact returned as a free byproduct, fed into both the dL/dweight
    dot product and the postact_a_for_dW2 store), then per-row-weight-
    multiplies (dgate, dup, postact) AFTER the ColVecReduceAtomic since
    SwiGLU bwd is linear in dpostact and the dL/dweight dot product needs
    UNWEIGHTED g (the per-row weight is what dW2 needs in postact_a_for_dW2;
    chain rule yields `dW2[e] = Σ_slot postact[slot] * w[slot] · dL_do_pool[slot]`).

    `implicit_dtype` (the bf16 dtype that mC AND mD's fp32-view-storage
    actually hold) is set via a `post_init` hook passed to
    `compile_gemm_kernel` — same plumbing quack's `gemm_dgated` uses.

    GemmActMixin's ``act_fn`` field is unused — we override
    ``epi_visit_subtile`` to compute the weighted-postact register tensor
    directly and return it (the framework's standard ``epi_convert_postact``
    path then casts fp32 → bf16 for the ``mAuxOut`` TMA store). Pass
    ``act_fn=None`` in EpilogueArguments.
    """

    _epi_ops = (
        *GemmActMixin._epi_ops,
        # Second ColVecLoad alongside the inherited "mColVecBroadcast" (used
        # for `pool_topk_weight`). Carries `pool_recv_token` (int32, cast to
        # fp32 by ColVecLoad's begin_loop) so `epi_visit_subtile` can detect
        # padding rows (recv_token < 0) and conditionally zero (dgate, dup,
        # postact, g) BEFORE colvec_reduce_accumulate / weight multiply /
        # mD / mAuxOut stores. Conditional ASSIGNMENT (not multiply) is
        # required: matmul(garbage, W2) at padding rows can produce inf/NaN,
        # and `0 * inf = NaN`. Removes the need to host-zero-init `pool` /
        # `dL_do_pool` / `dL_dswiglu_in` / `postact_a_for_dW2`.
        ColVecLoad("mPaddingMask"),
        ColVecReduceAtomic("mColVecReduce"),
    )
    _epi_param_bases = (ParamsBase,)

    # y_bwd is memory-LATENCY-bound — the dominant ncu stall is the L1TEX
    # global-load scoreboard (warps waiting on global loads), ~66% of CPI at
    # baseline. A gating sweep (see markdowns/logbook.md) showed the stall is
    # the long-K (K=H=3072) W2/A *mainloop* load, NOT the per-subtile preact
    # reload: deepening the mainloop prefetch (ab_stage 3 -> 4, epi_c_stage=2)
    # drops the scoreboard stall 67% -> 52% and cuts duration ~13% (626 ->
    # 545 us at the 82ba5b shape), whereas deepening the preact prefetch
    # (epi_c 2 -> 3) does nothing. So y_bwd forces ab_stage=4.
    AB_STAGE_TARGET = 4
    EPI_C_STAGE_TARGET = 2

    @classmethod
    def _compute_stages(
        cls,
        cta_tile_shape_mnk,
        epi_tile,
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        epilogue_args,
        smem_capacity,
        occupancy,
        warp_shape_mnk=None,
    ):
        """Force ab_stage=4 (deep mainloop W2/A prefetch) for y_bwd.

        Replaces the default heuristic (which picks ab_stage=3) — y_bwd's
        bottleneck is the L1TEX global-load scoreboard on the long-K W2/A
        mainloop, so a deeper mainloop pipeline is the win (see class
        comment / markdowns/logbook.md). ``ab_stage`` and ``epi_c_stage``
        are pinned to ``AB_STAGE_TARGET`` / ``EPI_C_STAGE_TARGET``; the freed
        ``epi_stage`` is set to the largest value (>= 1) that still fits the
        per-CTA SMEM budget. The byte accounting mirrors the parent
        ``GemmSm90._compute_stages`` exactly so this stays a stage-policy
        override, not a divergent SMEM model.

        Raises ``ValueError`` (fast, descriptive) when even ``epi_stage=1``
        does not fit — i.e. ab_stage=4 + epi_c_stage=2 alone overflow the
        budget for the given tile. This replaces the opaque
        ``cudaErrorInvalidValue`` launch failure with an actionable message
        ("shrink tile_n_y_bwd"). The tile picker (default_tile_config in
        stream_moe.py) deliberately leaves room for ab_stage=4 at the
        supported shapes; a too-large tile override is the error path.
        """
        ab_stage = cls.AB_STAGE_TARGET
        epi_c_stage = cls.EPI_C_STAGE_TARGET

        # --- byte accounting (mirrors GemmSm90._compute_stages) ---
        epi_smem_bytes = cls.epi_smem_bytes(
            epilogue_args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk
        )
        has_tile_load = epi_smem_bytes.c_stage > 0
        epi_tile_elems = cute.size(cute.shape(epi_tile))
        d_bytes_per_stage = (
            epi_tile_elems * d_dtype.width // 8 if d_dtype is not None else 0
        )
        epi_bytes_per_stage = d_bytes_per_stage + epi_smem_bytes.d_stage

        # SMEM consumed by everything EXCEPT the epi (D-store) staging:
        #   unstaged epi + mbar/helper reservation + ab_stage A/B operands
        #   + epi_c_stage C-load staging.
        fixed_bytes = epi_smem_bytes.unstaged
        if c_dtype is not None:
            fixed_bytes += epi_tile_elems * c_dtype.width // 8 * epi_c_stage
        if has_tile_load:
            fixed_bytes += epi_smem_bytes.c_stage * epi_c_stage

        a_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        budget = smem_capacity // occupancy
        # Bytes left for the epi D-store staging after the pinned ab/epi_c
        # allocations and the mbar/helper reservation.
        epi_budget = (
            budget
            - mbar_helpers_bytes
            - fixed_bytes
            - ab_bytes_per_stage * ab_stage
        )
        epi_stage = epi_budget // epi_bytes_per_stage if epi_bytes_per_stage > 0 else 1

        if epi_stage < 1:
            needed = (
                mbar_helpers_bytes
                + fixed_bytes
                + ab_bytes_per_stage * ab_stage
                + epi_bytes_per_stage  # one epi stage
            )
            tile_m, tile_n = cta_tile_shape_mnk[0], cta_tile_shape_mnk[1]
            raise ValueError(
                f"StreamingMoeYBwd requires ab_stage={ab_stage} "
                f"(epi_c_stage={epi_c_stage}, epi_stage>=1) = {needed // 1024} KB "
                f"SMEM but the per-CTA budget is {budget // 1024} KB at "
                f"tile=({tile_m},{tile_n}); shrink tile_n_y_bwd."
            )
        return ab_stage, epi_stage, epi_c_stage

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mAuxOut: cute.Tensor  # postact_a_for_dW2 — bf16 (M, I)
        # Unused: overridden `epi_visit_subtile` bypasses `act_fn`. See class docstring.
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mPaddingMask: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops + _extra_param_fields by
    # ComposableEpiMixin.__init_subclass__.

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        """SwiGLU bwd in registers: outputs `dL/dswiglu_in` (M, 2I) packed
        as bf16x2 in fp32 via mD's f32-recast trick AND ``postact_a_for_dW2``
        (M, I) bf16 via mAuxOut (returned as ``tRS_rPostAct``).
        ColVecReduceAtomic-accumulates `dL/dweight = postact · g` using the
        postact byproduct of `dswiglu`, atomic-adding into a flat per-slot
        fp32 buffer.

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
          (3) ColVecReduceAtomic-accumulate `Σ_n postact[m, n] * g[m, n]`
              per row (UNWEIGHTED g — chain rule for dL/dweight =
              postact · g). The atomic-add to ``dL_dweight[slot]`` happens
              in ``ColVecReduceAtomic.end()`` after the in-CTA
              warp/cross-warp reduce.
          (4) Per-row weight multiply on (dgate, dup, postact): SwiGLU bwd
              is linear in dpostact, so `w * dgate(g) = dgate(w * g)`.
              Multiplying after dswiglu is equivalent and lets the dL/dweight
              dot product see the unweighted g. Same multiply applied to
              ``postact`` produces ``postact_a_for_dW2 = w * postact`` —
              what dW2's grouped GEMM needs (chain rule on
              ``o = Σ_k w * y`` puts the topk_weight on postact in dW2's
              outer product input).
          (5) Pack (dgate, dup) bf16x2 → fp32 view; restore in tRS_rD for
              the standard mD store path. mD's host storage is bf16
              (Mflat, 2I), viewed as fp32 (Mflat, I) before launch — the
              kernel sees fp32 mD with implicit_dtype=bf16 packing.
              ``tRS_rPostAct`` (weighted, fp32) is returned for the framework
              to convert and TMA-store via mAuxOut (bf16 (M, I)).
        """
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        tDrPadMask = epi_loop_tensors["mPaddingMask"]
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

        # (2.5) Padding-row predicate. `tDrPadMask` broadcasts pool_recv_token
        # (int32 → fp32) along N, so all elements at the same row m share the
        # same recv_token value. recv_token < 0 marks a padding slot; matmul
        # through dL_do_pool[padding] @ W2 may produce inf/NaN there.
        # Conditional ASSIGNMENT (not multiply) zeros the four register
        # tensors so downstream consumers — colvec_reduce_accumulate (step 3),
        # weight multiply (step 4), mD TMA-store (dL_dswiglu_in), and mAuxOut
        # TMA-store (postact_a_for_dW2) — see clean zeros at padding rows.
        # `0 * inf = NaN` would slip through a multiply-based mask; the
        # ternary compiles to PTX `select.f32`, no NaN propagation.
        if const_expr(tDrPadMask is not None):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                is_active = tDrPadMask[i] >= Float32(0.0)
                tRS_rD[i] = tRS_rD[i] if is_active else Float32(0.0)
                tRS_rPostAct[i] = tRS_rPostAct[i] if is_active else Float32(0.0)
                tRS_rdXY_f32[2 * i] = tRS_rdXY_f32[2 * i] if is_active else Float32(0.0)
                tRS_rdXY_f32[2 * i + 1] = (
                    tRS_rdXY_f32[2 * i + 1] if is_active else Float32(0.0)
                )

        # (3) ColVecReduceAtomic on UNWEIGHTED g — chain rule for dL/dweight.
        # The intra-CTA reduction stays in-register here; the atomic-add
        # to dL_dweight[slot] runs in ColVecReduceAtomic.end() after this
        # subtile loop completes.
        if const_expr(tDrColVecReduce is not None):
            colvec_reduce_accumulate(self, tDrColVecReduce, tRS_rD, rScale=tRS_rPostAct)

        # (4) Per-row weight multiply on (dgate, dup, postact). Equivalent
        # to dswiglu(gate, up, w * g) by linearity in dout, but runs AFTER
        # the ColVecReduceAtomic so the dot product sees the unweighted g.
        # ``tRS_rPostAct`` is multiplied to produce postact_a_for_dW2 in-place.
        if const_expr(tDrColVec is not None):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rdXY_f32[2 * i] *= tDrColVec[i]
                tRS_rdXY_f32[2 * i + 1] *= tDrColVec[i]
                tRS_rPostAct[i] *= tDrColVec[i]

        # (5) Pack (dgate, dup) bf16x2 → fp32 view in tRS_rD for the
        # standard mD TMA-store. Lands in dL_dswiglu_in[tile, :2I] as bf16
        # via the host-side fp32 view.
        tRS_rdXY_b16 = cute.make_rmem_tensor(tRS_rdXY_f32.layout, implicit_dtype)
        tRS_rdXY_b16.store(tRS_rdXY_f32.load().to(implicit_dtype))
        tRS_rD.store(cute.recast_tensor(tRS_rdXY_b16, Float32).load())
        # Return weighted postact for the framework's mAuxOut TMA-store path.
        # epi_convert_postact (inherited) handles fp32 → bf16 (postact_dtype).
        return tRS_rPostAct

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

    # Second ColVecLoad — pool_recv_token (int32, varlen_m via cu_seqlens_m).
    # Cast to fp32 by ColVecLoad's begin_loop; epi_visit_subtile compares
    # against 0 to detect padding rows (recv_token < 0). Shape (TK_padded,).
    pool_recv_token = fake_tensor(
        cutlass.Int32, (TK_padded_sym,), leading_dim=0, divisibility=1
    )

    # ColVecReduceAtomic destination — flat per-slot fp32 buffer that all
    # pid_n CTAs atomic-add into via red.global.add.f32. Shape (Mflat,).
    # No num_pid_n dim — the atomic-add collapses across stripes in-kernel,
    # so no post-hoc .sum() torch op or cross-stream event is needed.
    mColVecReduce = fake_tensor(
        cutlass.Float32, (Mflat_sym,), leading_dim=0, divisibility=1
    )

    # mAuxOut: postact_a_for_dW2 (M, I) bf16 — dW2 grouped GEMM's input,
    # written via TileStore TMA path (same machinery GemmActMixin uses for
    # fwd postact). Plain bf16 (no f32-recast); each pid_n CTA writes a
    # (tile_M, tile_N) slab with no cross-CTA collisions, so no atomics.
    mAuxOut = fake_tensor(b_dtype, (Mflat_sym, I_sym), leading_dim=1, divisibility=8)

    # Scheduler tensors
    consumer_head = fake_tensor(cutlass.Int32, (cute.sym_int(),), divisibility=1)
    bwd_dispatch_arrival_count = fake_tensor(
        cutlass.Int32, (total_tiles_sym,), divisibility=1
    )
    pool_arrival_target = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)
    expert_pool_block_offset = fake_tensor(
        cutlass.Int32, (cu_seqlens_len_sym,), divisibility=1
    )

    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(0),
        consumer_head=consumer_head,
        pool_arrival_count=bwd_dispatch_arrival_count,
        pool_arrival_target=pool_arrival_target,
        expert_pool_block_offset=expert_pool_block_offset,
        total_tiles=Int32(0),
    )

    epi_args = StreamingMoeYBwd.EpilogueArguments(
        mAuxOut=mAuxOut,
        act_fn=None,
        mColVecBroadcast=pool_topk_weight,
        mPaddingMask=pool_recv_token,
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
        StreamingMoeYBwd,
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
    postact_a_for_dW2: torch.Tensor,  # (total_tiles, tile_m, I) bf16 — weighted postact for dW2
    pool_topk_weight: torch.Tensor,  # (TK_padded,) fp32 — per-slot weight (from saved handle)
    pool_recv_token: torch.Tensor,  # (TK_padded,) int32 — -1 marks padding (from saved handle)
    preact_a: torch.Tensor,  # (total_tiles, tile_m, 2*I) bf16 — saved from fwd kernel A's mD
    dL_dweight: torch.Tensor,  # (TK_padded,) fp32 — ZERO-INIT; per-pid_n atomic-add target
    expert_pool_block_offset: torch.Tensor,  # (E_local + 1,) int32 — pool-block prefix sum
    bwd_dispatch_arrival_count: torch.Tensor,  # (total_tiles,) int32 — input count from dispatch_grads (live)
    pool_arrival_target: torch.Tensor,  # (total_tiles,) int32 — firing target (shared with fwd)
    *,
    tile_m: int = 128,
    tile_n: int = 256,
    cluster_m: int = 1,
    cluster_n: int = 1,
    num_sms: int | None = None,
) -> None:
    """Launch streaming-MoE kernel Y bwd on the caller's current CUDA stream.

    Computes Y-side gradients in one tile-streamed pass:
      g[slot, :]                = dL_do_pool[slot] @ W2[e]               (unweighted)
      dL_dpostact_a[slot, :I]   = pool_topk_weight[slot] * g[slot, :I]
      (dgate, dup, postact)     = dswiglu(gate, up, dL_dpostact_a)
      dL_dswiglu_in[slot, 2n]   = dgate;  dL_dswiglu_in[slot, 2n+1] = dup
      dL_dweight[slot]         += Σ_n postact[slot, n] * g[slot, n]   (per-pid_n atomic-add)
      postact_a_for_dW2[slot, n] = pool_topk_weight[slot] * postact[slot, n]

    SwiGLU bwd is folded into the epilogue — the recomputed postact is
    returned by `dswiglu` as a free byproduct (used directly for both the
    dL/dweight dot product and the postact_a_for_dW2 store). Per-row weight
    multiply runs on (dgate, dup, postact) AFTER `dswiglu`: SwiGLU bwd is
    linear in dpostact, so the multiply on (dgate, dup) is equivalent to
    multiplying g first (lets the dL/dweight dot product see UNWEIGHTED g);
    the multiply on postact is what dW2's grouped GEMM input requires
    (chain rule on `o = Σ_k w * y` puts the topk_weight on postact).

    ``dL_dweight`` is atomic-added in-kernel via ``red.global.add.f32`` —
    each pid_n CTA contributes its in-CTA-reduced row sum to the flat
    ``(TK_padded,)`` fp32 buffer. **Caller MUST zero-init ``dL_dweight``
    before launch on the same stream.** Kernel_a_bwd runs FIFO-ordered
    after kernel_y_bwd on the same compute stream, so by the time
    combine_grads (waiting on `a_bwd_started`) reads ``dL_dweight`` the
    atomics are drained — no per-tile cross-stream release needed.

    ``postact_a_for_dW2`` is TMA-stored via the standard mAuxOut path
    (GemmActMixin's TileStore). Each pid_n CTA writes its (tile_M, tile_N)
    slab; no cross-CTA collisions, no atomics.

    Output ``dL_dswiglu_in`` is the direct input to kernel_a_bwd's data-grad
    GEMM and to dW1's grouped GEMM — no separate SwiGLU-bwd materialisation
    step needed.

    Streamed via per-tile count-vs-target spin on
    (`bwd_dispatch_arrival_count`, `pool_arrival_target`) for the input
    handoff. The Y_bwd → A_bwd output handoff is implicit: same compute
    stream FIFO covers visibility.

    ``num_sms`` caps the persistent-grid CTA count. ``None`` (default) fills the
    GPU; smaller caps leave SMs available for the other backward kernels to
    overlap.

    Caller is responsible for:
      - allocating ``dL_dswiglu_in`` and ``postact_a_for_dW2`` ON THE SAME
        STREAM this function is called from (so the kernel's TMA stores
        are naturally ordered with the allocations).
      - **zero-initialising ``dL_dweight``** on the same stream (the kernel
        atomic-adds into it; non-zero starting values would corrupt).
      - ensuring ``bwd_dispatch_arrival_count`` / ``pool_arrival_target`` are
        populated by the producer (``dispatch_grads_main_kernel``'s Pass 2 or
        a test stub) on a stream that ``red.release.gpu.global.add.s32``s
        into ``bwd_dispatch_arrival_count[tile_id]`` until it equals
        ``pool_arrival_target[tile_id]``. The per-tile count-vs-target spin
        handles cross-stream visibility.

    The internal ``consumer_head`` counter is allocated on the calling
    stream so its zero-init is naturally ordered with the kernel.

    kernel_a_bwd reads dL_dswiglu_in, dL_dweight, and postact_a_for_dW2
    after Y_bwd retires (same-stream FIFO).

    Storage layouts:
      - ``preact_a``: bf16 (total_tiles, tile_m, 2*I), viewed as fp32
        (total_tiles*tile_m, I) before launch — each fp32 element packs
        (gate_n, up_n) as bf16x2. Kernel reads as mC. (f32-recast trick.)
      - ``dL_dswiglu_in``: bf16 (total_tiles, tile_m, 2*I), viewed as fp32
        (total_tiles*tile_m, I) before launch — each fp32 element packs
        (dgate_n, dup_n) as bf16x2. Kernel writes as mD. (f32-recast trick.)
      - ``postact_a_for_dW2``: bf16 (total_tiles, tile_m, I), viewed flat
        as bf16 (total_tiles*tile_m, I). Kernel writes as mAuxOut. Plain
        bf16 store (no f32-recast).
    The two f32-recast tensors share the same `implicit_dtype` (bf16) which
    the kernel uses to recast between fp32 storage and bf16x2 math views.
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
    # tile_n MUST divide the output N dim (= I, the moe_intermediate_size).
    # Non-divisible tile_n produces silently-wrong stores in the data-grad
    # GEMM's partial-tile path.
    assert I % tile_n == 0, (
        f"tile_n ({tile_n}) must divide I ({I}); I % tile_n = {I % tile_n}"
    )
    assert expert_pool_block_offset.shape == (E_local + 1,)
    assert pool_topk_weight.shape == (dL_do_pool.shape[0],)
    assert pool_topk_weight.dtype == torch.float32
    assert pool_recv_token.shape == (dL_do_pool.shape[0],), (
        f"pool_recv_token must be (TK_padded,) = ({dL_do_pool.shape[0]},); "
        f"got {tuple(pool_recv_token.shape)}"
    )
    assert pool_recv_token.dtype == torch.int32
    assert (
        bwd_dispatch_arrival_count.shape == (total_tiles,)
        and bwd_dispatch_arrival_count.dtype == torch.int32
    )
    assert (
        pool_arrival_target.shape == (total_tiles,)
        and pool_arrival_target.dtype == torch.int32
    )
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
    # postact_a_for_dW2: bf16 (total_tiles, tile_m, I) — TMA-stored via
    # mAuxOut. Same dtype as W2 (bf16); plain bf16 write, no f32-recast.
    assert postact_a_for_dW2.is_cuda and postact_a_for_dW2.dim() == 3
    assert postact_a_for_dW2.shape == (total_tiles, tile_m, I), (
        f"postact_a_for_dW2 must be (total_tiles, tile_m, I) = "
        f"{(total_tiles, tile_m, I)}; got {tuple(postact_a_for_dW2.shape)}"
    )
    assert postact_a_for_dW2.dtype == W2.dtype, (
        f"postact_a_for_dW2 dtype must match W2's; got {postact_a_for_dW2.dtype} "
        f"vs {W2.dtype}"
    )
    # dL_dweight: flat (TK_padded,) fp32 — atomic-add target. Caller MUST
    # zero-init before launch (the kernel atomic-adds, doesn't overwrite).
    num_pid_n = (I + tile_n - 1) // tile_n
    assert dL_dweight.is_cuda and dL_dweight.dim() == 1
    assert dL_dweight.shape == (dL_do_pool.shape[0],), (
        f"dL_dweight must be (TK_padded,) = ({dL_do_pool.shape[0]},); "
        f"got {tuple(dL_dweight.shape)}"
    )
    assert dL_dweight.dtype == torch.float32
    assert dL_dweight.is_contiguous()

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

    # Flatten postact_a_for_dW2 (total_tiles, tile_m, I) → (Mflat, I) bf16
    # for the mAuxOut TMA path. No f32-recast — plain bf16 store.
    postact_a_for_dW2_flat = postact_a_for_dW2.view(total_tiles * tile_m, I)
    assert postact_a_for_dW2_flat.shape == (total_tiles * tile_m, I), (
        f"postact_a_for_dW2_flat shape mismatch: got "
        f"{tuple(postact_a_for_dW2_flat.shape)}, expected {(total_tiles * tile_m, I)}"
    )

    epi_args = StreamingMoeYBwd.EpilogueArguments(
        mAuxOut=postact_a_for_dW2_flat,
        act_fn=None,  # weighted-postact computed inline in epi_visit_subtile
        mColVecBroadcast=pool_topk_weight,
        mPaddingMask=pool_recv_token,
        mColVecReduce=dL_dweight,
        rounding_mode=None,  # Constexpr; pass None at call time
    )
    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        consumer_head=consumer_head,
        pool_arrival_count=bwd_dispatch_arrival_count,
        pool_arrival_target=pool_arrival_target,
        expert_pool_block_offset=expert_pool_block_offset,
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
