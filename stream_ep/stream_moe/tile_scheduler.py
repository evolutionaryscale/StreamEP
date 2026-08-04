"""Linear-claim tile scheduler for the streaming-MoE pipeline (pool layout).

Persistent CTAs claim work via `atomic_add(consumer_head, 1)`; each linear work
index decomposes into `(tile_id, pid_n)`. Every per-tile ready spin uses one
uniform protocol:

    while ld_acquire_gpu_global_i32(arrival_count[tile]) != arrival_target[tile]:
        pass

Producers (dispatch's Pass 2, dispatch_grads's Pass 2) each issue one
`red.release.gpu.global.add.s32` per contribution. `.release` semantics on the
add chain with the consumer's `.acquire` load to publish prior pool / postact
writes; `.gpu` scope is enough intra-GPU (different streams, same device).

  * Fwd handoff: caller passes `pool_arrival_count` / `pool_arrival_target`
    from the dispatch metadata. Kernel A spins on these; kernel Y, FIFO-
    ordered after A on the same compute stream, also passes them (its spin
    no-ops).
  * Bwd handoff: caller passes `bwd_dispatch_arrival_count` /
    `pool_arrival_target` from the dispatch_grads metadata. Kernel_y_bwd
    spins on these; kernel_a_bwd, FIFO-ordered after Y_bwd on compute,
    reuses them (its spin no-ops).

`pool_arrival_target[tile]` is the per-tile firing target (variable across
tiles).

Expert/pid_m are derived from `tile_id` by a warp-cooperative ballot lookup
over `expert_pool_block_offset` — each scheduler-warp lane loads one entry (or
kNumExpertsPerLane entries for E_local > 31), `vote_ballot_sync(cum <= tile_id)`
+ `popc` returns `expert_id + 1`, and a `shuffle_sync` from the matching lane
gives the cum for `pid_m`. The pool row offset `cu_seqlens_m[expert_id] +
pid_m * tile_m` lands at the right rows via the standard varlen_m TMA path —
no per-tile gather.

Wave behavior is structural (not enforced): dispatch's Pass 2 fires
`pool_arrival_count` in expert-major order at substream end. Linear claim
order == tile_id order == expert-major order, so 80 CTAs naturally converge
on the same expert at the same time and L2 holds 1-2 W1[e] slabs throughout.

Scheduler payload (sched_smem): the upstream 4-int layout
``(pid_m, pid_n, batch_idx, is_valid)``. tile_id is computed locally in the
scheduler warp's `_fetch_next_work_idx` (used for the spin and to derive
expert_id/pid_m) but not propagated to consumer warps — kernel A's mainloop
and postact path both hit the right pool rows via
``cu_seqlens_m[batch_idx] + pid_m * tile_m`` alone.

This file lives outside the quack tree so the streaming-MoE additions can be
maintained alongside the rest of the streaming-MoE pipeline in stream_ep.stream_moe.
StreamingTileScheduler subclasses quack's TileScheduler and overrides ONLY the
two streaming policy hooks — `_fetch_next_work_idx` (the global atomic claim +
per-tile ready-spin) and `_delinearize_work_idx` (the MoE expert-ballot
index->coord decode) — plus the streaming Params/Arguments/get_grid_shape and
the all-atomic `initial_work_tile_info`. The scheduler-warp SMEM coord
broadcast + pipeline (`write_work_tile_to_smem`, `get_current_work`,
`advance_to_next_work`, the loop-carry) are INHERITED from the base as-is.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import quack.utils as utils
from cutlass import Boolean, Int32, const_expr
from quack.fast_math import FastDivmod
from quack.tile_scheduler import PersistenceMode, TileScheduler, WorkTileInfo

from stream_ep.stream_moe.ptx_helpers import ld_acquire_gpu_global_i32


@dataclass
class StreamingTileSchedulerArguments:
    """Arguments for the streaming-MoE tile scheduler (pool layout).

    Every per-tile ready spin is the same protocol:

        while ld_acquire_gpu_global_i32(arrival_count[tile]) != arrival_target[tile]:
            pass

    Each producer issues `red.release.gpu.global.add.s32(arrival_count[tile], delta)`
    per contribution; the consumer terminates once the per-tile count hits the
    per-tile target. Both dispatch and epilogue handoffs share this protocol.

    Fwd handoff: caller passes `pool_arrival_count` / `pool_arrival_target`
    from the dispatch metadata. Used by kernel A; kernel Y (FIFO-ordered
    after A on the same compute stream) also passes them and its spin
    no-ops.

    Bwd handoff: caller passes `bwd_dispatch_arrival_count` /
    `pool_arrival_target` from the dispatch_grads metadata. Used by
    kernel_y_bwd; kernel_a_bwd (FIFO-ordered after Y_bwd on compute) reuses
    them and its spin no-ops.

    Pool layout: kernel A reads `pool` (expert-major, BLOCK_M-padded) via
    standard strided TMA — no per-tile gather indirection. Each tile's m-row
    range = `[tile_id * tile_M, (tile_id + 1) * tile_M)` in pool. The base
    GEMM kernel's varlen_m path lands the right rows when given
    ``cu_seqlens_m = expert_pool_block_offset * tile_m`` and the per-tile
    pid_m = tile_id - expert_pool_block_offset[expert_id].

    Expert lookup is warp-cooperative (no per-claim `tile_id_to_expert` GMEM
    read): each scheduler-warp lane loads one entry of
    ``expert_pool_block_offset`` (or kNumExpertsPerLane entries for
    E_local > 31), and a `vote_ballot_sync` + `popc` over `cum <= tile_id`
    yields ``expert_id + 1`` directly.
    """

    problem_shape_ntile_mnl: cute.Shape  # (None, num_pid_n, num_local_experts)
    consumer_head: cute.Tensor  # [1] int32 — global linear claim counter
    arrival_count: cute.Tensor  # [total_tiles] int32 — release-add destination
    arrival_target: cute.Tensor  # [total_tiles] int32 — per-tile firing target
    expert_pool_block_offset: (
        cute.Tensor
    )  # [E_local + 1] int32 — pool-block prefix-sum; consulted by the
    # warp-cooperative ballot lookup below (one entry per scheduler-warp lane,
    # kNumExpertsPerLane entries for E_local > 31).
    total_tiles: Int32  # passed as scalar so launch-time get_grid_shape doesn't deref device tensor
    tile_shape_mn: cutlass.Constexpr[cute.Shape]  # (tile_M, tile_N)
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    # Warp id of the producer (atomic-claim) warp. Each kernel passes
    # `self.ab_load_warp_id` here (or the math-side equivalent if those ever
    # need to act as scheduler). The `initial_work_tile_info` override below
    # compares `cute.arch.warp_idx()` against this value to identify the
    # single producer per CTA, replacing the v0.3.11 `is_scheduler_warp`
    # parameter that the caller used to pass into `setup_initial_work_tile`.
    scheduler_warp_id: cutlass.Constexpr[Int32]
    persistence_mode: cutlass.Constexpr[PersistenceMode] = PersistenceMode.DYNAMIC
    # Split-K is unused by the streaming pipeline; a constexpr 1 makes the base
    # TileScheduler's inherited split-K methods (get_split_k_tile_range,
    # get_combined_batch_idx, and the framework's per-tile call at
    # gemm_sm90.get_split_k_tile_range) compile to their no-split path.
    num_split_k: cutlass.Constexpr[int] = 1
    # Optional [1] int32 device flag. When supplied, the CTA that wins
    # `linear_idx == 0` atomicAdd's this flag once at the top of
    # `_fetch_next_work_idx`. Used by kernel_y / kernel_a_bwd to signal
    # to the consumer (combine_main / combine_grads_main on the
    # communicate stream) that at least one block is co-resident on an
    # SM. None disables the bump (kernel_a / kernel_y_bwd path).
    started_flag: Optional[cute.Tensor] = None


class StreamingTileScheduler(TileScheduler):
    """Linear-claim tile scheduler for the streaming-MoE pipeline (pool
    layout).

    Each persistent CTA's scheduler warp atomic-add-claims a linear work index
    `linear_idx = atomic_add(consumer_head, 1)`. The linear index decomposes
    into `(tile_id, pid_n) = divmod(linear_idx, num_pid_n)`. The per-tile
    ready spin is count-vs-target on
    `arrival_count[tile] == arrival_target[tile]` — one protocol for every
    handoff (dispatch's per-tile-variable target tensor, epilogue handoffs
    use a uniform-fill target tensor). Expert/pid_m are derived from
    `tile_id` by a
    warp-cooperative ballot lookup over `expert_pool_block_offset` (replaces
    the per-claim `tile_id_to_expert` + `expert_pool_block_offset` GMEM reads
    with one warp-collective ballot+popc and one shuffle). The standard
    varlen_m path's `cu_seqlens_m[expert_id] + pid_m * tile_m` formula then
    lands the correct pool row.

    Wave behavior for free: dispatch's Pass 2 fires `pool_arrival_count`
    release-adds in expert-major order at substream end. Linear claim order
    == tile_id order == expert-major order, so 80 CTAs naturally converge
    on the same expert at the same time and L2 holds 1-2 W1[e] slabs
    throughout.

    The work tile produced for the consumer warps carries the upstream-shape
    tuple `(pid_m, pid_n, None, batch_idx)`:
      - `pid_m = tile_in_e` (drives the cu_seqlens_m row offset)
      - `pid_n` (the N-stripe)
      - K-slot is unused (None), matching VarlenMTileScheduler's convention
      - `batch_idx = expert_id` (used by the kernel body to select W1[e])
    """

    @dataclass
    class Params:
        consumer_head: cute.Tensor
        arrival_count: cute.Tensor
        arrival_target: cute.Tensor
        expert_pool_block_offset: cute.Tensor
        total_tiles: Int32
        num_pid_n: Int32
        num_pid_n_fdd: FastDivmod
        tile_shape_mn: cutlass.Constexpr[cute.Shape]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        scheduler_warp_id: cutlass.Constexpr[Int32]
        persistence_mode: cutlass.Constexpr[PersistenceMode]
        num_split_k: cutlass.Constexpr[int] = 1
        started_flag: Optional[cute.Tensor] = None

        @staticmethod
        @cute.jit
        def create(
            args: StreamingTileSchedulerArguments, *, loc=None, ip=None
        ) -> "StreamingTileScheduler.Params":
            num_pid_n = cute.ceil_div(
                args.problem_shape_ntile_mnl[1], args.cluster_shape_mnk[1]
            )
            return StreamingTileScheduler.Params(
                consumer_head=args.consumer_head,
                arrival_count=args.arrival_count,
                arrival_target=args.arrival_target,
                expert_pool_block_offset=args.expert_pool_block_offset,
                total_tiles=args.total_tiles,
                num_pid_n=num_pid_n,
                num_pid_n_fdd=FastDivmod(num_pid_n),
                tile_shape_mn=args.tile_shape_mn,
                cluster_shape_mnk=args.cluster_shape_mnk,
                scheduler_warp_id=args.scheduler_warp_id,
                persistence_mode=args.persistence_mode,
                num_split_k=args.num_split_k,
                started_flag=args.started_flag,
            )

    # __init__, create, write_work_tile_to_smem, get_current_work,
    # advance_to_next_work, and the MLIR loop-carry (__extract/__new) are all
    # INHERITED from quack.tile_scheduler.TileScheduler. StreamEP customizes
    # only the two streaming *policy* hooks — _fetch_next_work_idx (the global
    # atomic claim + per-tile ready-spin) and _delinearize_work_idx (the MoE
    # expert-ballot index->coord decode) — plus the streaming
    # Params/Arguments/get_grid_shape and the all-atomic initial_work_tile_info.
    # The base owns the scheduler-warp SMEM coord broadcast + pipeline (which
    # StreamEP used to shadow, and which drifted from main).

    @staticmethod
    def to_underlying_arguments(
        args: StreamingTileSchedulerArguments, *, loc=None, ip=None
    ) -> Params:
        return StreamingTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params, max_active_clusters: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32, Int32]:
        # Grid is sized to fill compute SMs. total_tiles is passed as a scalar
        # (not derived from cumulative_tiles_before_e[num_local_experts]) so
        # we don't need to dereference a device tensor at host launch time.
        total_work = params.total_tiles * params.num_pid_n
        num_persistent_clusters = cutlass.min(
            max_active_clusters,
            cute.ceil_div(total_work, cute.size(params.cluster_shape_mnk)),
        )
        return (
            params.cluster_shape_mnk[0],
            params.cluster_shape_mnk[1],
            params.cluster_shape_mnk[2] * num_persistent_clusters,
        )

    @cute.jit
    def _fetch_next_work_idx(self, *, loc=None, ip=None) -> Int32:
        """Scheduler-warp-only. Global atomic claim + per-tile ready spin.

        Returns the claimed linear work index (the base
        ``TileScheduler._fetch_next_work_idx`` contract). The index->coord
        decode (the MoE expert ballot) lives in ``_delinearize_work_idx``; the
        SMEM broadcast of the decoded coord and the pipeline are inherited from
        the base scheduler.

        Lane 0 does ``linear_idx = atomic_add(consumer_head, 1)`` — a single
        global counter, so linear claim order == expert-major order (dispatch's
        Pass 2 fires ``pool_arrival_count`` release-adds in expert-major order,
        so CTAs walk experts in waves and L2 holds 1-2 W1[e] slabs). StreamEP
        claims ALL tiles atomically, unlike the base DYNAMIC scheduler's
        static-prefix + atomic-tail (hence the all-atomic
        ``initial_work_tile_info`` override). ``linear_idx >= total_tiles *
        num_pid_n`` means the queue is drained; ``_delinearize`` then returns
        is_valid=False and consumer warps exit.

        The per-tile count-vs-target ready spin (the dispatch handoff) runs
        HERE, before the ballot in ``_delinearize``: the acquire-load fences the
        ballot's ``expert_pool_block_offset`` reads (validated empirically —
        ballot-before-spin exposed a stale-L1 window). ``arrival_count`` must be
        fresh-zero per dispatch (the dispatch / dispatch_grads C++ slab
        zero-inits it), else a stale count lets a later CTA skip the wait.
        """
        params = self.params
        total_work = params.total_tiles * params.num_pid_n
        linear_idx = Int32(-1)
        if cute.arch.lane_idx() == 0:
            head_ptr = utils.elem_pointer(params.consumer_head, (Int32(0),))
            linear_idx = cute.arch.atomic_add(head_ptr, Int32(1))
        linear_idx = cute.arch.shuffle_sync(linear_idx, 0)

        # Cross-stream launch-gate bump: the CTA that wins linear_idx == 0 bumps
        # started_flag once so combine (on the communicate stream) can't grab SMs
        # ahead of this kernel. Only kernel_y / kernel_a_bwd carry a non-None
        # started_flag; skipped at compile time otherwise.
        if const_expr(params.started_flag is not None):
            if linear_idx == Int32(0) and cute.arch.lane_idx() == 0:
                flag_ptr = utils.elem_pointer(params.started_flag, (Int32(0),))
                cute.arch.atomic_add(flag_ptr, Int32(1))

        # Per-tile ready spin (lane 0), gated to in-range claims so a drained
        # claim never spins on an out-of-bounds arrival_count[tile_id]. tile_id =
        # linear_idx // num_pid_n (the same divmod _delinearize repeats for the
        # ballot).
        if linear_idx < total_work:
            tile_id, _ = divmod(linear_idx, params.num_pid_n_fdd)
            if cute.arch.lane_idx() == 0:
                count_ptr = utils.elem_pointer(params.arrival_count, (tile_id,))
                target = params.arrival_target[tile_id]
                while ld_acquire_gpu_global_i32(count_ptr) != target:
                    pass
        return linear_idx

    @cute.jit
    def _delinearize_work_idx(
        self, work_idx: Int32, *, block_zero_only: bool = False, loc=None, ip=None
    ) -> WorkTileInfo:
        """Decode a claimed linear work index into its tile coord via the
        warp-cooperative MoE expert ballot (the base ``_delinearize_work_idx``
        contract). Returns the CTA-0 (leader) coord; the base's
        ``write_work_tile_to_smem`` adds each peer's cluster offset (hence
        ``block_zero_only``, which StreamEP's cluster-N sharing gets for free).

        ``(tile_id, pid_n) = divmod(work_idx, num_pid_n)``; ``expert_id`` from a
        ballot over ``expert_pool_block_offset``; ``pid_m = tile_id -
        block_offset[expert_id]``; ``tile_coord_mnkl = (pid_m, pid_n, None,
        expert_id)``. A drained index returns a zero coord + is_valid=False.
        """
        params = self.params
        total_work = params.total_tiles * params.num_pid_n
        is_valid_i32 = Int32(work_idx < total_work)
        pid_n = Int32(0)
        pid_m = Int32(0)
        expert_id = Int32(0)
        if is_valid_i32 != 0:
            # linear_idx is one CLUSTER's claim along cluster-N; num_pid_n
            # already divides total_pid_n by cluster_n, so multiply back to the
            # leader-CTA pid_n (peers add their bidy offset in the base's
            # write_work_tile_to_smem).
            tile_id, cluster_pid_n = divmod(work_idx, params.num_pid_n_fdd)
            pid_n = cluster_pid_n * Int32(params.cluster_shape_mnk[1])

            # Warp-cooperative expert lookup. expert_pool_block_offset has length
            # E_local + 1 (last entry = total_tiles); kNumExpertsPerLane=2 covers
            # E_local up to 63. Out-of-range slots get INT_MAX so they never
            # match; cum[E_local]=total_tiles never matches since tile_id < it.
            kNumExpertsPerLane = const_expr(2)
            num_local_experts = (
                cute.size(params.expert_pool_block_offset, mode=[0]) - Int32(1)
            )
            lane_idx = cute.arch.lane_idx()
            INF = Int32(0x7FFFFFFF)
            cum_slots = []
            for i in cutlass.range_constexpr(kNumExpertsPerLane):
                e_idx = lane_idx + Int32(i * 32)
                cum_v = INF
                if e_idx <= num_local_experts:
                    cum_v = params.expert_pool_block_offset[e_idx]
                cum_slots.append(cum_v)

            # Count cums <= tile_id; monotone non-decreasing so count == expert_id + 1.
            n_matched_total = Int32(0)
            for i in cutlass.range_constexpr(kNumExpertsPerLane):
                n_matched_total += cute.arch.popc(
                    cute.arch.vote_ballot_sync(cum_slots[i] <= tile_id)
                )
            expert_id = n_matched_total - Int32(1)

            # pid_m = tile_id - block_offset[expert_id]; shuffle the matching cum.
            expert_lane = expert_id % Int32(32)
            expert_cum = Int32(0)
            if const_expr(kNumExpertsPerLane == 1):
                expert_cum = cute.arch.shuffle_sync(cum_slots[0], expert_lane)
            else:
                expert_slot = expert_id // Int32(32)
                for i in cutlass.range_constexpr(kNumExpertsPerLane):
                    cand = cute.arch.shuffle_sync(cum_slots[i], expert_lane)
                    if expert_slot == Int32(i):
                        expert_cum = cand
            pid_m = tile_id - expert_cum

        # tile_coord_mnkl: (pid_m, pid_n, K-slot=None, batch_idx=expert_id).
        tile_coord_mnkl = (pid_m, pid_n, None, expert_id)
        return WorkTileInfo(tile_coord_mnkl, Boolean(is_valid_i32))

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        """Streaming variant of the base `initial_work_tile_info`.

        The first work tile must come from the producer's atomic-claim +
        queue spin + sched_smem write — there is no static initial
        `_current_work_idx` to decompose. Both v0.4.1 call sites
        (`quack/gemm_sm90.py:682` AB-load context, `:810` math context)
        invoke this with no args; we recompute `is_scheduler_warp` here from
        `cute.arch.warp_idx() == self.params.scheduler_warp_id` (kernel side
        sets `scheduler_warp_id = self.ab_load_warp_id`). The producer warp
        runs the atomic claim + smem write inside `advance_to_next_work`;
        every other warp is a no-op claim that returns immediately and then
        consumer-waits in `get_current_work`.

        Math warps at the `:810` call site always see
        `warp_idx != scheduler_warp_id` (they live in a disjoint warp range),
        so they fall through to the consumer-wait path automatically. The
        pipeline's release-acquire pairing across the producer write and
        consumer read covers cross-warp visibility.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        is_scheduler_warp = warp_idx == self.params.scheduler_warp_id
        if const_expr(cute.size(self.params.cluster_shape_mnk) > 1):
            is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
        self.advance_to_next_work(is_scheduler_warp=is_scheduler_warp, loc=loc, ip=ip)
        return self.get_current_work(loc=loc, ip=ip)
