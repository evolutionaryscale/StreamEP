"""Custom epilogue ops for the streaming-MoE pipeline.

Currently houses ``ColVecReduceAtomic`` — a variant of quack's
``ColVecReduce`` whose ``end_loop_finish`` atomic-adds the cross-warp-reduced
row sums into a flat per-row fp32 buffer instead of writing to a per-pid_n
column of a (M, num_pid_n) staging tensor.

Why a custom op: kernel_y_bwd's epilogue produces the dL/dtopk_weight
contribution per slot via ``Σ_n postact[m,n] * g[m,n]``. Each pid_n CTA
covers an N-stripe of the I dim; an upstream ``ColVecReduce`` would land
per-stripe partials in ``dL_dweight_per_stripe[slot, n_stripe]`` requiring
a post-hoc ``.sum(dim=-1)`` torch op — a second kernel on
streams.compute that would force combine_grads to wait globally for the
sum before its first packet ships, killing per-recv-token streaming.

With per-pid_n fp32 atomic-add directly into a flat ``dL_dweight[slot]``,
kernel_y_bwd's atomics drain before kernel_a_bwd retires (same-stream
FIFO) and combine_grads (gated on ``a_bwd_started``) sees the final
values. Combine_grads's per-token gate
(``bwd_a_done_per_token[r] >= dispatch_seq``, fired by kernel_a_bwd)
drives per-recv-token streaming overlap with combine's sender.

Atomic cost: TK_padded × num_pid_n_y_bwd ≈ 32K × 8 = 256K fp32 atomics
per layer, all hot in L2. Throughput-trivial on H100 — the L2 atomic-add
unit absorbs scatter patterns at near-DRAM-bandwidth rates and these are
sparse (one fp32 per slot per stripe), not bandwidth-bound.
"""

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from quack.epilogue.ops import ColVecReduce
from quack.utils import elem_pointer

from stream_ep.stream_moe.ptx_helpers import red_add_f32


class ColVecReduceAtomic(ColVecReduce):
    """``ColVecReduce`` variant that atomic-adds each row's reduced sum into
    a flat ``(M,)`` fp32 buffer instead of writing to a per-pid_n column of
    an ``(M, num_pid_n)`` staging tensor.

    The epilogue ``param`` is a 1D ``(M,)`` fp32 tensor (no per-pid_n column
    dim, no per-batch dim — varlen_m is handled via the same ``cu_seqlens_m``
    domain offset ``ColVecReduce`` uses, just on a 1D target). All pid_n CTAs
    for a given tile race-free atomic-add to the same ``(slot,)`` location;
    the intra-warp shuffle + cross-warp merge inside the CTA run identically
    to the parent class.

    **Only ``end_loop_finish`` (the post-barrier gmem write) is overridden.**
    ``end_loop_stage`` — the last-N-subtile gate, intra-warp reduce, and
    per-warp smem staging — is inherited from ``ColVecReduce`` unchanged: it
    never touches ``param``, so it is agnostic to the flat-vs-column layout,
    and the driver's single shared barrier (fired via the ``needs_barrier``
    it returns) orders the staging writes before our merge. ``begin``,
    ``begin_loop``, ``param_fields``, ``to_params``, ``smem_bytes``,
    ``smem_struct_field``, and ``get_smem_tensor`` are likewise inherited
    unchanged.

    Parent ``end_loop_finish`` writes ``gColVec[row_idx] = finalize(...)`` into
    ``param[.., pid_n]`` (requiring a rank-2/3 partial buffer, cf.
    ``sink_alloc_shape``). We keep its inter-warp merge verbatim, then replace
    the write with a flat ``red.global.add.f32`` into ``param[slot]`` —
    dropping the per-pid_n column index and its ``limit_n_tiles`` bound.
    kernel_y_bwd asserts ``I % tile_n == 0`` so every pid_n stripe is a real
    partial (no padding stripe to exclude from the sum).

    History: pre-``60d8808`` quack exposed a single per-subtile ``end_loop``
    hook and this class overrode that. Upstream split it into
    ``end_loop_stage`` → shared barrier → ``end_loop_finish``; overriding only
    the finish half (and inheriting the stage) is both smaller and keeps the
    intra-/inter-warp reduction in lockstep with upstream.
    """

    @cute.jit
    def end_loop_finish(self, gemm, param, staged, tile_coord_mnkl, varlen_manager):
        """Inter-warp merge from smem (verbatim from ``ColVecReduce``) then a
        flat ``red.global.add.f32`` into ``param[slot]`` instead of the
        parent's column-indexed store. varlen_m only (the only mode
        kernel_y_bwd uses)."""
        vals_m, tDcD_m, sExch = staged[0], staged[1], staged[2]
        warps_in_N, warp_n_idx, is_lane_n_leader = staged[3], staged[4], staged[5]
        use_swap_shuffle, num_slices, slice_elems, lane_g = (
            staged[6],
            staged[7],
            staged[8],
            staged[9],
        )
        num_vals = const_expr(len(vals_m))

        # ── Inter-warp merge from smem (verbatim from ColVecReduce.end_loop_finish) ──
        if const_expr(warps_in_N > 1):
            if const_expr(use_swap_shuffle):
                if warp_n_idx == 0 and lane_g < num_slices:
                    for j in cutlass.range_constexpr(slice_elems):
                        row_idx = tDcD_m[lane_g * slice_elems + j][0]
                        for warp_n in cutlass.range_constexpr(1, warps_in_N):
                            others = tuple(sExch[row_idx, warp_n - 1, k] for k in range(num_vals))
                            merged = self._merge(tuple(v[j] for v in vals_m), others)
                            for k in cutlass.range_constexpr(num_vals):
                                vals_m[k][j] = merged[k]
            else:
                if warp_n_idx == 0 and is_lane_n_leader:
                    for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                        row_idx = tDcD_m[m][0]
                        for warp_n in cutlass.range_constexpr(1, warps_in_N):
                            others = tuple(sExch[row_idx, warp_n - 1, k] for k in range(num_vals))
                            merged = self._merge(tuple(v[m] for v in vals_m), others)
                            for k in cutlass.range_constexpr(num_vals):
                                vals_m[k][m] = merged[k]

        # ── Flat atomic-add write (replaces parent's column-indexed store) ──
        # varlen_m only: no per-batch dim, no per-pid_n column, no limit_n_tiles
        # gate — every launched pid_n stripe is a real partial (I % tile_n == 0),
        # and all pid_n CTAs race-free atomic-add into the same (slot,) location.
        assert varlen_manager.varlen_m, (
            "ColVecReduceAtomic only supports varlen_m mode (the only mode "
            "kernel_y_bwd uses)"
        )
        tile_M = gemm.cta_tile_shape_mnk[0]
        batch_idx = tile_coord_mnkl[3]
        limit_m = min(
            varlen_manager.len_m(batch_idx) - tile_coord_mnkl[0] * tile_M, tile_M
        )
        mColVec = cute.domain_offset(
            (varlen_manager.params.cu_seqlens_m[batch_idx],), param
        )
        gColVec = cute.local_tile(mColVec, (tile_M,), (tile_coord_mnkl[0],))
        if const_expr(use_swap_shuffle):
            in_warp0 = True if const_expr(warps_in_N == 1) else warp_n_idx == 0
            if in_warp0 and lane_g < num_slices:
                for j in cutlass.range_constexpr(slice_elems):
                    row_idx = tDcD_m[lane_g * slice_elems + j][0]
                    if row_idx < limit_m:
                        red_add_f32(
                            elem_pointer(gColVec, (row_idx,)),
                            self._finalize(tuple(v[j] for v in vals_m)),
                        )
        else:
            should_write_gmem = (
                is_lane_n_leader
                if const_expr(warps_in_N == 1)
                else warp_n_idx == 0 and is_lane_n_leader
            )
            if should_write_gmem:
                for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                    row_idx = tDcD_m[m][0]
                    if row_idx < limit_m:
                        red_add_f32(
                            elem_pointer(gColVec, (row_idx,)),
                            self._finalize(tuple(v[m] for v in vals_m)),
                        )
