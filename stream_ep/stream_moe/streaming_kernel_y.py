"""Streaming-MoE kernel Y (CuTeDSL, SM90, pool layout — fused scatter).

Forward kernel Y of the streaming pipeline:
  * Persistent CTAs pull tiles from a producer-fed queue (`a_ready`).
  * For each claimed tile_id, the streaming scheduler reads
    `expert_id = tile_id_to_expert[tile_id]` and computes
    `pid_m = tile_id - expert_pool_block_offset[expert_id]`.
  * Standard varlen_m strided TMA load of
    `postact_a[tile_id * tile_M : ..., :]` (the row offset
    `cu_seqlens_m[expert_id] + pid_m * tile_m` lands at the right pool-major
    row by construction).
  * GEMM against `W2[expert_id]`, in-register weight multiply (broadcast along
    N via the standard ColVecLoad path with `cu_seqlens_m`-aware varlen_m),
    R2S into a kernel-Y-owned bf16 SMEM staging buffer, then per-warp
    coalesced atomic-scatter from SMEM into `o[recv_token, :]` via packed
    `red.global.add.bf16x2`.
  * On per-tile end, lane 0 of each warp atomicSubs `per_token_remaining[r]`
    for its rows; on hit-zero (the recv-token's last contribution landed),
    release-stores `compute_done_per_token[r] = combine_seq` so the combine
    sender (Phase D, on its own stream) can pick up `o[r]` for RDMA push.

The atomic-scatter staging SMEM is owned by `AtomicScatterStore` (an EpiOp,
declared here) — the framework's `sD` stays `None` (because `mD = None`).
This sidesteps both prior failure modes:
  - Option 4 (register-only): non-deterministic atomic drops from divergent
    branches around `red.global.add.bf16x2`. The SMEM staging gives every
    warp warp-uniform row selection — no within-warp divergence around `red`.
  - Option 5 (fork-edited `sD`): JIT compile timeout from threading
    `sD: Optional[cute.Tensor]` through the framework. Owning our own SMEM
    via the standard EpiOp `smem_struct_field` mechanism (mirrors `TileStore`,
    `RowVecLoad`, `ColVecLoad`, `ColVecReduce`) avoids fighting the framework.

Streaming machinery is shared with kernel A:
  * `StreamingTileScheduler` for linear-claim + per-tile ready spin.
    Substitution: `tile_ready` → `a_ready`, `dispatch_seq` → `compute_seq`.
  * Pool-layout `StreamingHandle` carries: `pool_recv_token`, `pool_topk_weight`,
    `per_token_remaining`, `compute_done_per_token`, `o`, `a_ready`,
    `tile_id_to_expert`, `expert_pool_block_offset` from `Buffer.dispatch`.

Padding rows (pool_recv_token[s] == -1) and other masked lanes: PTX-level
`@%p red.global.add.noftz.v4.bf16x2` predication skips the atomic at issue
time, so caller allocates `o` with the natural shape `(T_recv, H)` — no
trash row needed.
"""

from dataclasses import MISSING
from typing import NamedTuple, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import quack.copy_utils as copy_utils
import quack.sm90_utils as quack_sm90_utils
import quack.utils as utils
import torch
from cutlass import Int32, Int64, const_expr
from cutlass.utils import LayoutEnum
from quack.cache_utils import COMPILE_ONLY, jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    ParamsBase,
    get_device_capacity,
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.epi_composable import ComposableEpiMixin
from quack.epi_ops import ColVecLoad, EpiOp
from quack.gemm_sm90 import GemmSm90
from quack.gemm_tvm_ffi_utils import compile_gemm_kernel
from quack.tile_scheduler import PersistenceMode
from quack.varlen_utils import VarlenArguments

from evolutionaryscale.models.moe.streaming_moe.ptx_helpers import (
    pack_bf16x2,
    red_add_bf16x2_v4_pred,
    st_release_sys_global,
    threadfence_system,
)
from evolutionaryscale.models.moe.streaming_moe.streaming_tile_scheduler import (
    StreamingTileScheduler,
    StreamingTileSchedulerArguments,
)


# ---------------------------------------------------------------------------
# Bundled scatter param tensors. One field on EpilogueArguments carries all of
# them so the auto-generated EpilogueParams stays clean.
# ---------------------------------------------------------------------------
@mlir_namedtuple
class ScatterParams(NamedTuple):
    mO: cute.Tensor  # [T_recv, H]  bf16  — atomic-scatter destination
    pool_recv_token: cute.Tensor  # [TK_padded]        int32 — slot → r (-1 = padding)
    per_token_remaining: cute.Tensor  # [T_recv]           int32 — kernel Y atomicSubs
    compute_done_per_token: (
        cute.Tensor
    )  # [T_recv]           int64 — Y → combine release stamp
    tile_n_stripes_done: (
        cute.Tensor
    )  # [total_tiles]      int32 — per-tile_id N-stripe arrival counter
    expert_pool_block_offset: (
        cute.Tensor
    )  # [E_local + 1]      int32 — pool-block prefix-sum (for tile_id reconstruct)
    T_recv: Int32  # row count == mO.shape[0]
    combine_seq: Int64  # value to release-store on hit-zero
    num_pid_n: Int32  # N-stripe count per tile (= ceil(H / tile_N))


# ---------------------------------------------------------------------------
# AtomicScatterStore: an EpiOp that owns the bf16 staging SMEM + the per-tile
# pool_recv_token SMEM area. Mirrors TileStore's smem_struct_field pattern but
# nests two struct fields (staging + recv_token) into one sub-struct.
# ---------------------------------------------------------------------------
class AtomicScatterStore(EpiOp):
    """Per-tile bf16 staging buffer + per-warp coalesced atomic-scatter into o[r, :].

    Allocates two SMEM regions in one struct field `s_<name>`:
      - `staging`: bf16 tensor of shape `(epi_tile_M, epi_tile_N, epi_stage)`,
        swizzled per `make_smem_layout_epi` (the same layout `TileStore` uses for
        TMA-stored postact).
      - `recv_token`: int32 tensor of shape `(tile_M,)` — pool_recv_token slice
        for this tile. Loaded once per tile in `begin()` from gmem.

    The actual scatter logic lives in `StreamingMoeYSm90.epi_subtile_store`
    (the override of GemmSm90's hook). This op handles SMEM allocation,
    per-tile recv_token load, and the per-tile end bookkeeping.
    """

    def __init__(self, name: str = "scatter"):
        super().__init__(name)

    # --- Param plumbing -----------------------------------------------------
    def _layout_key(self):
        return f"{self.name}_smem_layout_staged"

    def param_fields(self):
        return [(self.name, object, MISSING), (self._layout_key(), object, MISSING)]

    def to_params(self, gemm, args):
        scatter = getattr(args, self.name)
        smem_layout_staged = quack_sm90_utils.make_smem_layout_epi(
            scatter.mO.element_type,
            LayoutEnum.from_tensor(scatter.mO),
            gemm.epi_tile,
            gemm.epi_stage,
        )
        return {self.name: scatter, self._layout_key(): smem_layout_staged}

    # --- SMEM allocation ----------------------------------------------------
    def smem_bytes(self, arg, cta_tile_shape_mnk, epi_tile):
        if arg is None:
            return 0
        # Conservative upper bound (epi_stage decided at runtime). Mirrors the
        # bytes TileStore would charge plus the small int32 recv_token area.
        bf16_bytes_per_stage = cute.size(cute.shape(epi_tile)) * 2
        return bf16_bytes_per_stage * 4 + cta_tile_shape_mnk[0] * 4

    def smem_struct_field(self, gemm, params):
        layout_key = self._layout_key()
        if not hasattr(params, layout_key):
            return None
        smem_layout_staged = getattr(params, layout_key)
        bf16_size = cute.cosize(smem_layout_staged)
        tile_M = gemm.cta_tile_shape_mnk[0]
        scatter_dtype = getattr(params, self.name).mO.element_type

        @cute.struct
        class ScatterStorage:
            staging: cute.struct.Align[
                cute.struct.MemRange[scatter_dtype, bf16_size], gemm.buffer_align_bytes
            ]
            recv_token: cute.struct.Align[cute.struct.MemRange[Int32, tile_M], 16]

        return (f"s_{self.name}", ScatterStorage)

    def get_smem_tensor(self, gemm, params, storage_epi):
        smem_layout_staged = getattr(params, self._layout_key())
        s_struct = getattr(storage_epi, f"s_{self.name}")
        staging_t = s_struct.staging.get_tensor(
            smem_layout_staged.outer, swizzle=smem_layout_staged.inner
        )
        recv_token_t = s_struct.recv_token.get_tensor(
            cute.make_layout(gemm.cta_tile_shape_mnk[0])
        )
        return (staging_t, recv_token_t)

    # --- Device lifecycle ---------------------------------------------------
    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """Once per tile: load pool_recv_token slice into SMEM via a tile_M-thread
        synchronous gmem→smem copy. Also computes pool_start = cu_seqlens_m[batch_idx]
        + pid_m * tile_M (the first pool slot for this tile).
        """
        if const_expr(param is None):
            return None
        staging_t, recv_token_t = smem_tensor
        # pool_start = cu_seqlens_m[batch_idx] + pid_m * tile_M.
        # cu_seqlens_m carries `expert_pool_block_offset * tile_m` (host-side build
        # in the streaming_moe_y wrapper); for the streaming scheduler
        # tile_coord_mnkl[0] = pid_m = tile_in_e and tile_coord_mnkl[3] = expert_id.
        pool_start = (
            ctx.varlen_manager.params.cu_seqlens_m[ctx.batch_idx]
            + ctx.tile_coord_mnkl[0] * ctx.tile_M
        )
        # Load tile_M ints from pool_recv_token[pool_start : pool_start + tile_M]
        # into recv_token_t. Each thread (tidx in [0, tile_M)) loads 1 int.
        # Off-tile threads (tidx >= tile_M) skip — the per-row consumers below
        # only read indices < tile_M.
        if ctx.tidx < ctx.tile_M:
            recv_token_t[ctx.tidx] = param.pool_recv_token[pool_start + ctx.tidx]
        # The framework's epi_begin wraps async ops in a cp_async_wait + barrier
        # if `needs_async_fence()` is True. We do a synchronous load above, so we
        # rely on the surrounding epilogue_barrier (called by the consumer side)
        # to make the load visible.
        return (staging_t, recv_token_t, pool_start)

    def begin_loop(self, gemm, state, epi_coord):
        # epi_subtile_store reads the staging + recv_token directly from `state`.
        return state

    def needs_async_fence(self):
        return False

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
        """Per-tile end: each row's atomic-scatter for THIS pid_n stripe has
        landed; decrement the per-recv-token remaining counter and on hit-zero
        release the per-token compute-done signal.

        Multi-pid_n gating: a single tile_id is split across `num_pid_n` CTAs
        (one per N-stripe). The per-row bookkeeping must fire ONCE PER TILE,
        not once per N-stripe — so the LAST N-stripe to complete (atomic-add
        on `tile_n_stripes_done[tile_id]` returning `num_pid_n - 1`) does the
        decrement. Atomic-add provides acq_rel ordering, so the last CTA sees
        all prior N-stripes' atomic-scatters.
        """
        if const_expr(param is None):
            return
        staging_t, recv_token_t, pool_start = state

        # Reconstruct tile_id from batch_idx + pid_m.
        # tile_id = expert_pool_block_offset[batch_idx] + pid_m.
        batch_idx = tile_coord_mnkl[3]
        pid_m = tile_coord_mnkl[0]
        tile_id = param.expert_pool_block_offset[batch_idx] + pid_m

        # Threadfence-gpu (the atomic-add below provides system-scope ordering).
        # Only thread 0 of warp 0 does the gate increment. Other threads sync
        # via epilogue_barrier. Then thread 0 broadcasts whether it's the last.
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        is_thread0 = (warp_idx == Int32(0)) & (lane_idx == Int32(0))

        # Use a single int register to share the "is last" bit across the CTA.
        # Each warp's threads will read the same is_last flag from the
        # epilogue_barrier rendez-vous.
        # Simpler approach: thread 0 atomic-adds, branches; other warps' lane 0
        # await on a shared atomic-loaded value or just have all warps decide
        # again via reading tile_n_stripes_done back. To minimize sync, all
        # threads of the CTA participate in the atomic-add (the value returned
        # is per-thread), but we only TRUST thread 0's result and broadcast.
        # Even simpler: have ONLY thread 0 do everything (atomic-add + per-row
        # bookkeeping). The bookkeeping is small (~tile_M scalar atomics), so
        # serializing on thread 0 is acceptable.

        if is_thread0:
            stripes_ptr = utils.elem_pointer(param.tile_n_stripes_done, (tile_id,))
            prev_stripes = utils.atomic_add_i32(Int32(1), stripes_ptr)
            is_last_stripe = prev_stripes == (param.num_pid_n - Int32(1))

            if is_last_stripe:
                # Make all atomic-scatters from all N-stripes globally visible
                # before the per-token release-store. The cross-CTA ordering
                # is already provided by the atomic-add above (acq_rel), but
                # add an explicit threadfence_system to be safe.
                threadfence_system()

                tile_M = const_expr(gemm.cta_tile_shape_mnk[0])
                T_recv = param.T_recv
                combine_seq = param.combine_seq

                for m in cutlass.range_constexpr(tile_M):
                    r = recv_token_t[m]
                    if r >= Int32(0) and r < T_recv:
                        rem_ptr = utils.elem_pointer(param.per_token_remaining, (r,))
                        prev = utils.atomic_add_i32(Int32(-1), rem_ptr)
                        if prev == Int32(1):
                            done_ptr = utils.elem_pointer(
                                param.compute_done_per_token, (r,)
                            )
                            st_release_sys_global(done_ptr, combine_seq)


# ---------------------------------------------------------------------------
# Host-facing scheduler-options NamedTuple. Mirrors kernel A's
# StreamingTileSchedulerOptions but renames `tile_ready` → `a_ready` to
# reflect the producer (kernel A's per-tile completion stamp).
# ---------------------------------------------------------------------------
@mlir_namedtuple
class StreamingMoeYSchedulerOptions(NamedTuple):
    max_active_clusters: Int32
    consumer_head: cute.Tensor
    a_ready: cute.Tensor
    tile_id_to_expert: cute.Tensor
    expert_pool_block_offset: cute.Tensor
    compute_seq: Int64
    total_tiles: Int32


# ---------------------------------------------------------------------------
# Streaming kernel Y class. Subclass GemmSm90 directly (not GemmDefaultEpiMixin
# or GemmActMixin) so we control which EpiOps participate.
# ---------------------------------------------------------------------------
class StreamingMoeYSm90(ComposableEpiMixin, GemmSm90):
    """Streaming-MoE kernel Y: streaming GEMM with fused atomic-scatter epilogue.

    Composition:
      - ColVecLoad("mColVecBroadcast"): per-row weight broadcast along N.
        Caller passes pool_topk_weight as args.mColVecBroadcast; varlen_m mode
        with cu_seqlens_m = expert_pool_block_offset * tile_m offsets correctly
        to pool_start = expert_pool_block_offset[batch_idx] * tile_m + pid_m * tile_m.
      - AtomicScatterStore("scatter"): owns the bf16 staging SMEM and the
        per-tile pool_recv_token SMEM area. End-of-tile bookkeeping fires
        compute_done_per_token[r] on hit-zero.

    Overrides:
      - epi_visit_subtile: in-register weight multiply (replaces the additive
        bias path of GemmDefaultEpiMixin).
      - epi_subtile_store: R2S into AtomicScatterStore's staging; per-warp
        coalesced atomic-scatter from staging into mO[r, n_origin:].
      - epi_setup_postact: returns None (no postact).
    """

    _epi_ops = (ColVecLoad("mColVecBroadcast"), AtomicScatterStore("scatter"))
    _epi_param_bases = (ParamsBase,)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        scatter: ScatterParams
        mColVecBroadcast: Optional[cute.Tensor] = None

    # EpilogueParams auto-generated by ComposableEpiMixin.

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        return self.EpilogueParams(**self._epi_ops_to_params_dict(args))

    def epi_setup_postact(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        return None

    @cute.jit
    def epi_convert_postact(
        self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
    ):
        return tRS_rPostAct

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        """In-register weight multiply on the MMA accumulator subtile.

        ColVecLoad's begin_loop has already populated `mColVecBroadcast` as a
        register tensor with the same per-thread layout as `tRS_rD`, with the
        per-row weight broadcast along N. So `tRS_rD[i] *= weight[i]` works
        element-wise.
        """
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        if const_expr(tDrColVec is not None):
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                tRS_rD[i] *= tDrColVec[i]
        return None

    @cute.jit
    def epi_subtile_store(
        self,
        params,
        epi_loop_tensors,
        tRS_rD,
        tRS_sD,
        tiled_copy_r2s,
        copy_D,
        has_D,
        postact_ctx,
        tRS_rPostAct_out,
        epi_store_pipeline,
        epilogue_barrier,
        tile_coord_mnkl,
        epi_coord,
        num_prev_subtiles,
        epi_idx,
        tidx,
        is_tma_warp,
    ):
        """Per-subtile R2S → bf16 SMEM → per-warp coalesced atomic-scatter into o.

        Pipeline state hygiene: producer_acquire/commit on epi_store_pipeline
        are kept balanced even though we never call copy_D (gotcha #6 from
        logbook 2026-04-30).
        """
        scatter = params.scatter
        staging_t, recv_token_t, pool_start = epi_loop_tensors["scatter"]

        # 1. Producer-acquire (balance pipeline state across the persistent loop).
        if is_tma_warp:
            epi_store_pipeline.producer_acquire()
        epilogue_barrier.arrive_and_wait()

        # 2. R2S: cvt_copy register acc → staging SMEM (per-stage slot).
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sScatter = thr_copy_r2s.partition_D(staging_t)
        epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
        copy_utils.cvt_copy(
            tiled_copy_r2s, tRS_rD, tRS_sScatter[None, None, None, epi_buffer]
        )

        # 3. Sync to make staging visible to all warps.
        cute.arch.fence_view_async_shared()
        epilogue_barrier.arrive_and_wait()

        # 4. Per-warp coalesced atomic-scatter from staging.
        # staging_t shape: (epi_tile_M, epi_tile_N, epi_stage), n_major.
        # epi_coord = (epi_m, epi_n) within the cta-tile's (tile_M / epi_tile_M,
        # tile_N / epi_tile_N) grid of subtiles.
        epi_tile_M = const_expr(self.epi_tile[0])
        epi_tile_N = const_expr(self.epi_tile[1])
        cta_tile_N = const_expr(self.cta_tile_shape_mnk[1])

        n_origin = tile_coord_mnkl[1] * Int32(cta_tile_N) + epi_coord[1] * Int32(
            epi_tile_N
        )
        m_subtile_origin = epi_coord[0] * Int32(epi_tile_M)

        # Vector-packed bf16x2 atomic-add. Each `red.global.add.v4.bf16x2`
        # writes 8 bf16 (16 bytes).
        #
        # The naive partition (32 lanes co-cover ONE row's `epi_tile_N`
        # bf16) wastes lanes whenever `epi_tile_N // 8 < 32`. At our
        # default `epi_tile_N=32` only 4 lanes/row carry useful work; the
        # other 28 issuing zero-mask v4 atomics turns into 4× wasted HBM
        # bandwidth (each masked lane still does a 16-byte read-modify-write
        # of zeros). Verified empirically: naive-mask v4 made kernel Y 45%
        # slower than the scalar bf16x2 path.
        #
        # Fix: flatten `(row, v4_chunk)` into a single per-warp work index.
        # Each lane handles one `(m_in_subtile, v4_chunk)` pair per inner
        # iter — distributing work across rows so all 32 lanes are busy.
        # At our defaults (`epi_tile_M=128, num_epi_warps=4, epi_tile_N=32`):
        #   rows_per_warp = 32, v4_chunks_per_row = 4
        #   work_per_warp = 128 = 4 * 32  (no remainder; all lanes effective)
        # Total atomics issued per warp drops from 32*32=1024 (16 effective
        # + 16 masked per row × 32 rows) to 32*4=128 v4 atomics, *all
        # effective*. 8× reduction in atomic ops vs scalar; same total bytes.
        v4_chunks_per_row = const_expr((epi_tile_N + 7) // 8)
        # Per-warp work units = rows_per_warp * v4_chunks_per_row. Inner
        # iters loop over `ceil(work_per_warp / 32)`. Lanes whose work_idx
        # exceeds `work_per_warp` are masked (rare; only when work doesn't
        # divide evenly into 32-lane groups).
        # Computed below after rows_per_warp is known.

        # Per-stage staging view. We read individual bf16 from SMEM (which
        # honors swizzle correctly) and pack into bf16x2 manually — see
        # `pack_bf16x2` in ptx_helpers.py for why `cute.recast_tensor` on
        # swizzled SMEM does NOT preserve bf16x2 pair-adjacency.
        stage_view = staging_t[None, None, epi_buffer]  # (epi_tile_M, epi_tile_N)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        num_epi_warps = const_expr(self.num_epi_warps)
        rows_per_warp = const_expr((epi_tile_M + num_epi_warps - 1) // num_epi_warps)
        work_per_warp = const_expr(rows_per_warp * v4_chunks_per_row)
        work_iters_per_warp = const_expr((work_per_warp + 32 - 1) // 32)

        T_recv = scatter.T_recv

        for w in cutlass.range_constexpr(work_iters_per_warp):
            work_idx = Int32(w) * Int32(32) + Int32(lane_idx)
            work_in_range = work_idx < Int32(work_per_warp)
            work_safe = work_idx if work_in_range else Int32(0)
            # Decompose: row offset within warp's slab and v4-chunk within row.
            # constexpr divisor folds to shift+and at compile time when
            # v4_chunks_per_row is a power of 2.
            m_off = work_safe // Int32(v4_chunks_per_row)
            v4_chunk = work_safe % Int32(v4_chunks_per_row)
            m_in_subtile = Int32(warp_idx) * Int32(rows_per_warp) + m_off
            row_in_range = (m_in_subtile < Int32(epi_tile_M)) & work_in_range
            m_safe = m_in_subtile if row_in_range else Int32(0)
            m_local_in_tile = m_subtile_origin + m_safe
            r_raw = recv_token_t[m_local_in_tile]
            # Padding rows (r_raw == -1), out-of-range warp rows, and
            # work_idx-out-of-range lanes all collapse into one predicate.
            # PTX-level @%p on the v4 atomic skips the instruction entirely
            # at issue time — no HBM op, no atomic side-effect — so the
            # address can be any clamped in-bounds pointer.
            atomic_pred = (r_raw >= Int32(0)) & (r_raw < T_recv) & row_in_range
            r_safe = r_raw if atomic_pred else Int32(0)

            # 8 adjacent bf16 starting at bf16-idx (v4_chunk * 8).
            bf16_base = v4_chunk * Int32(8)
            b0 = stage_view[m_safe, bf16_base + Int32(0)]
            b1 = stage_view[m_safe, bf16_base + Int32(1)]
            b2 = stage_view[m_safe, bf16_base + Int32(2)]
            b3 = stage_view[m_safe, bf16_base + Int32(3)]
            b4 = stage_view[m_safe, bf16_base + Int32(4)]
            b5 = stage_view[m_safe, bf16_base + Int32(5)]
            b6 = stage_view[m_safe, bf16_base + Int32(6)]
            b7 = stage_view[m_safe, bf16_base + Int32(7)]
            p0 = pack_bf16x2(b0, b1)
            p1 = pack_bf16x2(b2, b3)
            p2 = pack_bf16x2(b4, b5)
            p3 = pack_bf16x2(b6, b7)
            # v4 needs 16-byte alignment (8 bf16). bf16_base is multiple
            # of 8 by construction; n_origin is multiple of epi_tile_N
            # (≥ 8 by config); together address is 16-byte aligned.
            n_global = n_origin + bf16_base
            o_row = scatter.mO[r_safe, None]
            o_row_as_i32 = cute.recast_tensor(o_row, Int32)
            target_ptr = utils.elem_pointer(o_row_as_i32, (n_global // Int32(2),))
            red_add_bf16x2_v4_pred(target_ptr, p0, p1, p2, p3, Int32(atomic_pred))

        # 5. Producer-commit (balance pipeline state).
        cute.arch.fence_view_async_shared()
        epilogue_barrier.arrive_and_wait()
        if is_tma_warp:
            epi_store_pipeline.producer_commit()

    # -- scheduler hooks -----------------------------------------------------

    def get_scheduler_class(self, varlen_m: bool = False):
        return StreamingTileScheduler

    def get_scheduler_arguments(
        self,
        mA,
        mB,
        mD,
        scheduler_args: StreamingMoeYSchedulerOptions,
        varlen_args: VarlenArguments,
        epilogue_args,
    ):
        # mB shape: (n=H, k=I, l=E_local).
        num_pid_n = cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1])
        E_local = cute.size(mB, mode=[2])
        return StreamingTileSchedulerArguments(
            problem_shape_ntile_mnl=(None, num_pid_n, E_local),
            consumer_head=scheduler_args.consumer_head,
            tile_ready=scheduler_args.a_ready,
            tile_id_to_expert=scheduler_args.tile_id_to_expert,
            expert_pool_block_offset=scheduler_args.expert_pool_block_offset,
            dispatch_seq=scheduler_args.compute_seq,
            total_tiles=scheduler_args.total_tiles,
            tile_shape_mn=self.cta_tile_shape_mnk[:2],
            cluster_shape_mnk=self.cluster_shape_mnk,
            persistence_mode=PersistenceMode.STREAMING,
        )

    @cute.jit
    def __call__(
        self,
        mA,
        mB,
        mD,
        mC,
        epilogue_args,
        scheduler_args: StreamingMoeYSchedulerOptions,
        varlen_args,
        stream,
        trace_ptr=None,
    ):
        """Type-shim override so CuTeDSL accepts StreamingMoeYSchedulerOptions
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
def _compile_streaming_moe_y(
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    o_dtype: Type[cutlass.Numeric],
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    cluster_n: int,
    device_capacity,
):
    assert device_capacity[0] == 9, "Streaming MoE kernel Y is SM90-only"

    I_sym = cute.sym_int()
    H_sym = cute.sym_int()
    E_sym = cute.sym_int()
    TK_padded_sym = cute.sym_int()
    Trecv_sym = cute.sym_int()
    total_tiles_sym = cute.sym_int()
    cu_seqlens_len_sym = cute.sym_int()  # = E_local + 1

    # A: postact_a flat (TK_padded, I), k-major (I contiguous).
    mA = fake_tensor(a_dtype, (TK_padded_sym, I_sym), leading_dim=1, divisibility=8)
    # B: W2 (H, I, E_local), k-major per expert (I contiguous), batch dim = E_local.
    mB = fake_tensor(b_dtype, (H_sym, I_sym, E_sym), leading_dim=1, divisibility=8)
    # No D / C — streaming kernel Y outputs via predicated atomic-scatter
    # into o[T_recv, H] (no trash row).
    mD = None
    mC = None

    # Scatter destination: (T_recv, H), n-major.
    mO = fake_tensor(o_dtype, (Trecv_sym, H_sym), leading_dim=1, divisibility=8)
    pool_recv_token = fake_tensor(
        cutlass.Int32, (TK_padded_sym,), leading_dim=0, divisibility=1
    )
    per_token_remaining = fake_tensor(
        cutlass.Int32, (cute.sym_int(),), leading_dim=0, divisibility=1
    )
    compute_done_per_token = fake_tensor(
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
        per_token_remaining=per_token_remaining,
        compute_done_per_token=compute_done_per_token,
        tile_n_stripes_done=tile_n_stripes_done,
        expert_pool_block_offset=expert_pool_block_offset_scatter,
        T_recv=Int32(0),
        combine_seq=Int64(0),
        num_pid_n=Int32(0),
    )

    # ColVecLoad's per-row weight broadcast (varlen_m). Shape (TK_padded,) fp32.
    pool_topk_weight = fake_tensor(
        cutlass.Float32, (TK_padded_sym,), leading_dim=0, divisibility=1
    )

    # cu_seqlens_m drives the standard varlen_m m-offset for both A's pool read
    # and ColVecLoad's per-row weight slice. Length E_local + 1.
    mCuSeqlensM = fake_tensor(
        cutlass.Int32, (cu_seqlens_len_sym,), leading_dim=0, divisibility=1
    )

    consumer_head = fake_tensor(cutlass.Int32, (cute.sym_int(),), divisibility=1)
    a_ready = fake_tensor(cutlass.Int64, (total_tiles_sym,), divisibility=1)
    tile_id_to_expert = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)
    expert_pool_block_offset = fake_tensor(
        cutlass.Int32, (cu_seqlens_len_sym,), divisibility=1
    )

    scheduler_args = StreamingMoeYSchedulerOptions(
        max_active_clusters=Int32(0),
        consumer_head=consumer_head,
        a_ready=a_ready,
        tile_id_to_expert=tile_id_to_expert,
        expert_pool_block_offset=expert_pool_block_offset,
        compute_seq=Int64(0),
        total_tiles=Int32(0),
    )

    epi_args = StreamingMoeYSm90.EpilogueArguments(
        scatter=scatter, mColVecBroadcast=pool_topk_weight
    )

    varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlensM, mCuSeqlensK=None, mAIdx=None)

    return compile_gemm_kernel(
        StreamingMoeYSm90,
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
def streaming_moe_y(
    postact_a: torch.Tensor,  # (total_tiles, tile_m, I) bf16
    W2: torch.Tensor,  # (E_local, H, I) bf16 — k-major per expert
    o: torch.Tensor,  # (T_recv, H) bf16 — atomic-scatter destination
    pool_recv_token: torch.Tensor,  # (TK_padded,) int32
    pool_topk_weight: torch.Tensor,  # (TK_padded,) float32
    per_token_remaining: torch.Tensor,  # (T_recv,) int32
    compute_done_per_token: torch.Tensor,  # (T_recv,) int64
    tile_id_to_expert: torch.Tensor,  # (total_tiles,) int32
    expert_pool_block_offset: torch.Tensor,  # (E_local + 1,) int32
    a_ready: torch.Tensor,  # (total_tiles,) int64
    compute_seq: int,
    combine_seq: int,
    *,
    tile_m: int = 128,
    tile_n: int = 128,
    cluster_m: int = 1,
    cluster_n: int = 1,
    num_sms: int | None = None,
) -> None:
    """Launch streaming-MoE kernel Y (fused atomic-scatter) on the caller's
    current CUDA stream.

    ``num_sms`` caps the persistent-grid CTA count to the given value. When
    ``None`` (default) the kernel fills the GPU. Smaller caps leave SMs
    available for kernel A to run concurrently — see design.md §"SM budget".

    Caller is responsible for:
      - allocating ``o`` with shape ``(T_recv, H)``, zero-initialized, on
        the same stream this function is called from. Padding rows and
        other masked lanes are handled via PTX-level predicated atomic-add
        (no trash row needed).
      - allocating ``per_token_remaining`` with the K_local count for each
        recv-token (DeepEP's dispatch sets this in Pass B's per-pool-slot block).
      - allocating ``compute_done_per_token`` zero-initialized.
      - ensuring ``a_ready`` is populated by kernel A (or a test stub) on a
        stream that release-stores ``a_ready[tile_id] = compute_seq`` once
        the tile's postact_a is ready.
    """
    assert postact_a.is_cuda and W2.is_cuda and o.is_cuda
    assert postact_a.dim() == 3
    assert W2.dim() == 3
    assert o.dim() == 2
    total_tiles, postact_tile_m, I = postact_a.shape
    assert postact_tile_m == tile_m
    T_recv, H = o.shape
    E_local = W2.shape[0]
    assert W2.shape == (E_local, H, I), (
        f"W2 must be (E_local, H, I); got {tuple(W2.shape)}, expected "
        f"{(E_local, H, I)}"
    )
    assert pool_recv_token.shape == (total_tiles * tile_m,)
    assert pool_recv_token.dtype == torch.int32
    assert pool_topk_weight.shape == (total_tiles * tile_m,)
    assert pool_topk_weight.dtype == torch.float32
    assert per_token_remaining.shape == (T_recv,)
    assert per_token_remaining.dtype == torch.int32
    assert compute_done_per_token.shape == (T_recv,)
    assert compute_done_per_token.dtype == torch.int64
    assert tile_id_to_expert.shape == (total_tiles,)
    assert expert_pool_block_offset.shape == (E_local + 1,)
    assert a_ready.shape == (total_tiles,)
    assert a_ready.dtype == torch.int64

    # Caller passes W2 as (E_local, H, I) k-major contiguous. We need the
    # kernel to see shape (H, I, E_local) with leading_dim=1 (I contiguous
    # along K). torch.permute(1, 2, 0) gives this layout WITHOUT a copy.
    W2_p = W2.permute(1, 2, 0)
    assert W2_p.stride(1) == 1, "W2[:, e, :] must be I-contiguous (k-major weights)"
    assert W2_p.shape == (H, I, E_local)

    # Flatten postact_a's leading two dims.
    postact_flat = postact_a.view(total_tiles * tile_m, I)

    # cu_seqlens_m = expert_pool_block_offset * tile_m drives both:
    #   (a) varlen_m m-offset for postact_a tile reads (kernel mainloop)
    #   (b) ColVecLoad's per-row weight slice via cute.domain_offset
    cu_seqlens_m = (expert_pool_block_offset.to(torch.int32) * tile_m).contiguous()

    device_capacity = get_device_capacity(postact_a.device)
    assert device_capacity[0] == 9, "Streaming MoE kernel Y is SM90-only"

    a_dtype = torch2cute_dtype_map[postact_a.dtype]
    b_dtype = torch2cute_dtype_map[W2.dtype]
    o_dtype = torch2cute_dtype_map[o.dtype]

    compiled_fn = _compile_streaming_moe_y(
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
    # zero-init is naturally ordered with kernel Y's atomic claims.
    consumer_head = torch.zeros(1, dtype=torch.int32, device=postact_a.device)

    # tile_n_stripes_done gates per-tile bookkeeping when num_pid_n > 1: the
    # CTA whose atomic-add returns `num_pid_n - 1` is the last N-stripe and
    # does the per-row decrement + compute_done release.
    num_pid_n = (H + tile_n - 1) // tile_n
    tile_n_stripes_done = torch.zeros(
        total_tiles, dtype=torch.int32, device=postact_a.device
    )

    scatter = ScatterParams(
        mO=o,
        pool_recv_token=pool_recv_token,
        per_token_remaining=per_token_remaining,
        compute_done_per_token=compute_done_per_token,
        tile_n_stripes_done=tile_n_stripes_done,
        expert_pool_block_offset=expert_pool_block_offset,
        T_recv=Int32(T_recv),
        combine_seq=Int64(combine_seq),
        num_pid_n=Int32(num_pid_n),
    )
    epi_args = StreamingMoeYSm90.EpilogueArguments(
        scatter=scatter, mColVecBroadcast=pool_topk_weight
    )
    scheduler_args = StreamingMoeYSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        consumer_head=consumer_head,
        a_ready=a_ready,
        tile_id_to_expert=tile_id_to_expert,
        expert_pool_block_offset=expert_pool_block_offset,
        compute_seq=Int64(compute_seq),
        total_tiles=Int32(total_tiles),
    )
    varlen_args = VarlenArguments(
        mCuSeqlensM=cu_seqlens_m, mCuSeqlensK=None, mAIdx=None
    )

    compiled_fn(
        postact_flat, W2_p, None, None, epi_args, scheduler_args, varlen_args, None
    )


# ---------------------------------------------------------------------------
# Test-only producer: walks a_ready slot-by-slot and release-stores
# compute_seq on each, with delay between fires. Used by tests to validate
# kernel Y's per-tile spin without kernel A.
# ---------------------------------------------------------------------------
class _StreamingTileProducerY:
    @cute.jit
    def __call__(
        self,
        a_ready: cute.Tensor,
        total_tiles: cutlass.Int32,
        compute_seq: cutlass.Int64,
        delay_clocks: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(a_ready, total_tiles, compute_seq, delay_clocks).launch(
            grid=[1, 1, 1], block=[1, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        a_ready: cute.Tensor,
        total_tiles: cutlass.Int32,
        compute_seq: cutlass.Int64,
        delay_clocks: cutlass.Int32,
    ):
        from cutlass._mlir.dialects import nvvm
        from cutlass.cutlass_dsl import T

        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            for i in cutlass.range(total_tiles):
                start = cutlass.Int64(nvvm.read_ptx_sreg_clock64(T.i64()))
                end = start + cutlass.Int64(delay_clocks)
                while cutlass.Int64(nvvm.read_ptx_sreg_clock64(T.i64())) < end:
                    pass
                ready_ptr = utils.elem_pointer(a_ready, (i,))
                threadfence_system()
                st_release_sys_global(ready_ptr, compute_seq)


@jit_cache
def _compile_streaming_tile_producer_y():
    total_tiles_sym = cute.sym_int()
    ready = fake_tensor(cutlass.Int64, (total_tiles_sym,), divisibility=1)
    op = _StreamingTileProducerY()
    return cute.compile(
        op,
        ready,
        cutlass.Int32(0),
        cutlass.Int64(0),
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def fire_a_ready_with_delay(
    a_ready: torch.Tensor, compute_seq: int, delay_us: int = 50
) -> None:
    """Test helper: launches a single-thread producer kernel on the current
    CUDA stream that release-stores compute_seq into each slot of a_ready
    with delay_us between fires.
    """
    assert a_ready.dtype == torch.int64
    assert a_ready.is_cuda and a_ready.is_contiguous()
    total_tiles = a_ready.shape[0]
    delay_clocks = max(1, int(delay_us * 1500))
    compiled = _compile_streaming_tile_producer_y()
    compiled(
        a_ready,
        cutlass.Int32(total_tiles),
        cutlass.Int64(compute_seq),
        cutlass.Int32(delay_clocks),
    )
