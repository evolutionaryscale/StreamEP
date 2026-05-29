"""Custom epilogue ops for the streaming-MoE pipeline.

Currently houses ``ColVecReduceAtomic`` — a variant of quack's
``ColVecReduce`` whose ``end()`` atomic-adds the cross-warp-reduced row sums
into a flat per-row fp32 buffer instead of writing to a per-pid_n column of
a (M, num_pid_n) staging tensor.

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

import operator
from functools import partial

import cutlass
import cutlass.cute as cute
import quack.layout_utils as layout_utils
from cutlass import Float32, const_expr
from quack.epi_ops import ColVecReduce, _get_lane_warp_layouts
from quack.sm90_utils import partition_for_epilogue
from quack.utils import elem_pointer

from stream_ep.stream_moe.ptx_helpers import red_add_f32


class ColVecReduceAtomic(ColVecReduce):
    """``ColVecReduce`` variant that atomic-adds the per-row reduced sum
    into a flat (M,) fp32 buffer instead of writing to ``param[m, pid_n]``.

    The epilogue ``param`` is a 1D ``(M,)`` fp32 tensor (no per-pid_n
    column dim, no per-batch dim — varlen_m is handled via the same
    ``cu_seqlens_m`` domain offset ``ColVecReduce`` uses, just on a 1D
    target). All pid_n CTAs for a given tile race-free atomic-add to the
    same (M,) location; the shuffle + cross-warp reduction inside the CTA
    still runs identically to the parent class.

    Inherits ``begin``, ``begin_loop``, ``param_fields``, ``to_params``,
    ``smem_bytes``, ``smem_struct_field``, ``get_smem_tensor`` from
    ``ColVecReduce`` unchanged — the only behavioural change is in the
    end-of-loop flush.

    v0.4.1 framework calls the per-subtile method ``end_loop`` (with the
    last-N-subtile gate moved inside the method). v0.3.11 called a single
    post-loop ``end()`` after the per-subtile fold. We override
    ``end_loop`` and gate on ``epi_coord[1] == epi_tile_shape[1] - 1`` to
    fire exactly once per CTA tile, matching the v0.3.11 post-loop
    invocation point.
    """

    @cute.jit
    def end_loop(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Intra-warp shuffle + optional inter-warp reduction → row sum →
        ``red.global.add.f32`` into ``param[slot]``.

        ``param`` shape contract: 1D ``(M,)`` fp32 in varlen_m mode (the
        only mode kernel_y_bwd uses). The varlen_m batch offset is
        applied via ``cute.domain_offset`` over ``cu_seqlens_m`` exactly
        as in the parent class, just against a 1D target.
        """
        if const_expr(param is None):
            return
        # Last-N-subtile gate (v0.4.1 calls end_loop per subtile).
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(gemm.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        if const_expr(epi_coord[1] != epi_tile_shape[1] - 1):
            return
        tDrReduce, sDrReduce = state[0], state[1]
        tiled_copy = tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s
        reference_src = tiled_copy_t2r is None

        # ── Derive lane/warp layouts (same as parent) ──
        lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(
            tiled_copy, reference_src
        )
        lanes_in_N = cute.size(lane_layout_MN, mode=[1])
        is_lane_n_leader = cute.arch.lane_idx() % lanes_in_N == 0

        # ── Intra-warp shuffle reduction across N lanes (same as parent) ──
        if const_expr(lanes_in_N > 1):
            assert lane_layout_MN.stride[1] == 1
            tDrReduce_flt = cute.filter_zeros(tDrReduce)
            for i in cutlass.range(cute.size(tDrReduce_flt), unroll_full=True):
                tDrReduce_flt[i] = cute.arch.warp_reduction(
                    tDrReduce_flt[i], operator.add, threads_in_group=lanes_in_N
                )

        warp_N = warp_layout_MN[1]
        warps_in_N = const_expr(cute.size(warp_N))
        # The v0.3.11 `max_warps_in_n` safety check is gone: v0.4.1's
        # ColVecReduce.smem_struct_field sizes the staging area from
        # `warp_shape_mnk` (passed to smem_bytes), which is by construction
        # >= warps_in_N at runtime. The parent class handles the bound.

        partition_for_epilogue_fn = partial(
            partition_for_epilogue,
            epi_tile=epi_tile,
            tiled_copy=tiled_copy,
            tidx=tidx,
            reference_src=tiled_copy_t2r is None,
        )
        tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]
        epilogue_barrier = gemm.epilogue_barrier
        batch_idx = tile_coord_mnkl[3]

        # ── Param indexing for 1D (M,) target with varlen_m batch offset ──
        # All pid_n CTAs for this tile_id atomic-add to the same (slot,)
        # locations; race-free via red.global.add.f32.
        assert varlen_manager.varlen_m, (
            "ColVecReduceAtomic only supports varlen_m mode (the only mode "
            "kernel_y_bwd uses)"
        )
        mColVec = cute.domain_offset(
            (varlen_manager.params.cu_seqlens_m[batch_idx],), param
        )
        gColVec = cute.local_tile(mColVec, (tile_M,), (tile_coord_mnkl[0],))
        limit_m = min(
            varlen_manager.len_m(batch_idx) - tile_coord_mnkl[0] * tile_M, tile_M
        )

        tDcD = partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
        tDrReduce_m = layout_utils.convert_layout_zero_stride(
            tDrReduce, tDrReduce.layout
        )[None, 0]
        tDcD_m = layout_utils.convert_layout_zero_stride(tDcD, tDrReduce.layout)[
            None, 0
        ]

        if const_expr(warps_in_N == 1):
            # Single warp covers the full tile_N — every is_lane_n_leader
            # owns one row's final sum, atomic-add directly.
            if is_lane_n_leader:
                for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                    row_idx = tDcD_m[m][0]
                    if row_idx < limit_m:
                        red_add_f32(elem_pointer(gColVec, (row_idx,)), tDrReduce_m[m])
        else:
            # Multi-warp tile — stage per-warp partials in SMEM, single
            # warp_n_idx==0 reduces across warps and atomic-adds.
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            warp_n_idx = warp_layout_MN.get_hier_coord(warp_idx)[1]
            if is_lane_n_leader:
                for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                    row_idx = tDcD_m[m][0]
                    if row_idx < limit_m:
                        sDrReduce[row_idx, warp_n_idx] = tDrReduce_m[m]
            epilogue_barrier.arrive_and_wait()
            if warp_n_idx == 0 and is_lane_n_leader:
                for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                    row_idx = tDcD_m[m][0]
                    if row_idx < limit_m:
                        row_sum = Float32(0.0)
                        for warp_n in cutlass.range_constexpr(warps_in_N):
                            row_sum += sDrReduce[row_idx, warp_n]
                        red_add_f32(elem_pointer(gColVec, (row_idx,)), row_sum)
