"""Linear-claim tile scheduler for the streaming-MoE pipeline (pool layout).

Persistent CTAs claim work via `atomic_add(consumer_head, 1)`; each linear work
index decomposes into `(tile_id, pid_n)`. The per-tile ready spin is selected
at JIT time via `SpinKind`:

  * `COUNT_VS_TARGET` (kernel A, kernel_y_bwd): spin on
    `pool_arrival_count[tile] == pool_arrival_target[tile]`. The producer
    (dispatch's Pass 2 / dispatch_grads's Pass 2) fires
    `red.release.gpu.global.add.s32` per contribution; multi-producer.
  * `ACQUIRE_VS_SEQ` (kernel Y, kernel_a_bwd): spin on
    `ld_acquire_gpu_global(stamp[tile]) >= seq`. The producer (kernel A /
    kernel_y_bwd) release-stores a monotonic int64 stamp once per tile
    after its multi-stripe TMA drain; single-producer.

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
The base TileScheduler + supporting infrastructure (PipelineStateWAdvance,
FastDivmod, the upstream PersistenceMode enum) are imported from quack as-is.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import quack.utils as utils
from cutlass import Boolean, Int32, Int64, const_expr
from quack.fast_math import FastDivmod
from quack.pipeline import PipelineStateWAdvance
from quack.tile_scheduler import PersistenceMode, TileScheduler
from quack.utils import store_shared_remote_x4

from stream_ep.stream_moe.ptx_helpers import (
    ld_acquire_gpu_global,
    ld_acquire_gpu_global_i32,
)


class SpinKind(IntEnum):
    """How `_fetch_next_work_idx` synchronizes with the producer of a tile.

    Both kinds use the same warp-cooperative ballot expert-lookup; only the
    per-tile "is this tile ready" spin differs.

    COUNT_VS_TARGET: spin on
        `pool_arrival_count[tile_id] == pool_arrival_target[tile_id]`.
        Producer fires a `red.release.gpu.global.add.s32` per contribution
        (multi-producer); consumer terminates once count hits the firing
        target. Used by kernel A (consumes dispatch's Pass 2) and
        `kernel_y_bwd` (consumes dispatch_grads' Pass 2).

    ACQUIRE_VS_SEQ: spin on `ld_acquire_gpu_global(stamp[tile_id]) >= seq`.
        Producer release-stores a monotonic int64 seq stamp (single-producer
        per tile, with multi-stripe gating done inside the producer epilogue);
        consumer terminates once the stamp catches up. Used by kernel Y
        (consumes kernel A's `a_ready`) and `kernel_a_bwd` (consumes
        `kernel_y_bwd`'s `bwd_a_ready`).
    """

    COUNT_VS_TARGET = 0
    ACQUIRE_VS_SEQ = 1


@dataclass
class StreamingTileSchedulerArguments:
    """Arguments for the streaming-MoE tile scheduler (pool layout). Produced by
    DeepEP's Buffer.dispatch and consumed by the QuACK streaming kernel.

    The scheduler supports two per-tile spin protocols, selected at compile
    time via the `spin_kind` constexpr:
      * `SpinKind.COUNT_VS_TARGET` (kernel A, kernel_y_bwd): spin on
        `pool_arrival_count[tile_id] == pool_arrival_target[tile_id]`.
        Multi-producer release-add protocol; producer fires
        `red.release.gpu.global.add.s32` per contribution.
      * `SpinKind.ACQUIRE_VS_SEQ` (kernel Y, kernel_a_bwd): spin on
        `ld_acquire_gpu_global(stamp[tile_id]) >= seq`. Single-producer
        release-stamp protocol; producer issues one `st_release_gpu_global`
        per tile after the in-kernel multi-stripe gating.

    Both kinds share the warp-cooperative ballot expert-lookup. The unused
    field set (count/target for ACQUIRE_VS_SEQ kernels; stamp/seq for
    COUNT_VS_TARGET kernels) gets placeholder tensors; the constexpr branch
    in `_fetch_next_work_idx` dead-code-eliminates the unused path at JIT
    compile time.

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
    # Per-tile spin source. Exactly one of (count, target) vs (stamp, seq) is
    # live per kernel; the other pair carries placeholder tensors. Selected
    # by `spin_kind` constexpr.
    pool_arrival_count: cute.Tensor  # [total_tiles] int32  — COUNT_VS_TARGET
    pool_arrival_target: cute.Tensor  # [total_tiles] int32  — COUNT_VS_TARGET
    stamp: cute.Tensor  # [total_tiles] int64                  — ACQUIRE_VS_SEQ
    dispatch_seq: Int64  # the int the consumer compares stamp against
    # Common.
    expert_pool_block_offset: (
        cute.Tensor
    )  # [E_local + 1] int32 — pool-block prefix-sum; consulted by the
    # warp-cooperative ballot lookup below (one entry per scheduler-warp lane,
    # kNumExpertsPerLane entries for E_local > 31).
    total_tiles: Int32  # passed as scalar so launch-time get_grid_shape doesn't deref device tensor
    tile_shape_mn: cutlass.Constexpr[cute.Shape]  # (tile_M, tile_N)
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    spin_kind: cutlass.Constexpr[SpinKind] = SpinKind.COUNT_VS_TARGET
    persistence_mode: cutlass.Constexpr[PersistenceMode] = PersistenceMode.STREAMING


class StreamingTileScheduler(TileScheduler):
    """Linear-claim tile scheduler for streaming-MoE kernel A (pool layout).

    Each persistent CTA's scheduler warp atomic-add-claims a linear work index
    `linear_idx = atomic_add(consumer_head, 1)`. The linear index decomposes
    into `(tile_id, pid_n) = divmod(linear_idx, num_pid_n)`. The scheduler
    selects the per-tile ready spin via `SpinKind`: count-vs-target on
    (`pool_arrival_count`, `pool_arrival_target`) for the multi-producer
    dispatch handoffs, or acquire-vs-seq on (`stamp`, `dispatch_seq`) for
    the single-producer epilogue handoffs. Expert/pid_m are derived from
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
        pool_arrival_count: cute.Tensor
        pool_arrival_target: cute.Tensor
        stamp: cute.Tensor
        dispatch_seq: Int64
        expert_pool_block_offset: cute.Tensor
        total_tiles: Int32
        num_pid_n: Int32
        num_pid_n_fdd: FastDivmod
        tile_shape_mn: cutlass.Constexpr[cute.Shape]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        spin_kind: cutlass.Constexpr[SpinKind]
        persistence_mode: cutlass.Constexpr[PersistenceMode]

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
                pool_arrival_count=args.pool_arrival_count,
                pool_arrival_target=args.pool_arrival_target,
                stamp=args.stamp,
                dispatch_seq=args.dispatch_seq,
                expert_pool_block_offset=args.expert_pool_block_offset,
                total_tiles=args.total_tiles,
                num_pid_n=num_pid_n,
                num_pid_n_fdd=FastDivmod(num_pid_n),
                tile_shape_mn=args.tile_shape_mn,
                cluster_shape_mnk=args.cluster_shape_mnk,
                spin_kind=args.spin_kind,
                persistence_mode=args.persistence_mode,
            )

    def __init__(
        self,
        current_work_idx: Int32,
        num_tiles_executed: Int32,
        current_expert: Int32,
        current_pid_m: Int32,
        current_pid_n: Int32,
        sched_smem: Optional[cute.Tensor],
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        pipeline_state: PipelineStateWAdvance,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        # Streaming scheduler state, persisted across the persistent loop's
        # iterations via the MLIR pytree round-trip:
        #   _current_expert: the expert containing the current tile (derived
        #     in _fetch_next_work_idx from the warp-cooperative ballot lookup
        #     over expert_pool_block_offset). Surfaced via tile_coord_mnkl[3]
        #     for W1[e] selection.
        #   _current_pid_m: tile_in_e = tile_id - expert_pool_block_offset[expert_id].
        #     Drives the standard varlen_m path's m-offset calculation
        #     (`cu_seqlens_m[expert_id] + pid_m * tile_m`) which lands at the
        #     correct pool row.
        #   _current_pid_n: the N-stripe of the most recently claimed work,
        #     surfaced via tile_coord_mnkl[1].
        # tile_id is computed locally in `_fetch_next_work_idx` (used for the
        # ready spin and to derive expert_id/pid_m); not stashed on self
        # because no consumer code reads it.
        self._current_work_idx = current_work_idx
        self.num_tiles_executed = num_tiles_executed
        self._current_expert = current_expert
        self._current_pid_m = current_pid_m
        self._current_pid_n = current_pid_n
        self._sched_smem = sched_smem
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: StreamingTileSchedulerArguments, *, loc=None, ip=None
    ) -> Params:
        return StreamingTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sched_smem: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "StreamingTileScheduler":
        # Initial work index: each persistent CTA starts unclaimed (work_idx = -1
        # means "not yet fetched"). The scheduler warp does its first
        # atomic_add(consumer_head) inside _fetch_next_work_idx during the first
        # advance_to_next_work call.
        stages = (
            const_expr(cute.size(sched_smem, mode=[1])) if sched_smem is not None else 0
        )
        return StreamingTileScheduler(
            current_work_idx=Int32(-1),
            num_tiles_executed=Int32(0),
            current_expert=Int32(0),
            current_pid_m=Int32(0),
            current_pid_n=Int32(0),
            sched_smem=sched_smem,
            scheduler_pipeline=scheduler_pipeline,
            pipeline_state=PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params=params,
            loc=loc,
            ip=ip,
        )

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
        """Scheduler-warp-only. Linear claim with per-tile ready spin.

        Lane 0 does ``linear_idx = atomic_add(consumer_head, 1)``. If
        ``linear_idx >= total_tiles * num_pid_n`` the kernel is exhausted and
        we return is_valid=0; consumer warps see is_valid_tile=False and exit
        the persistent loop. Otherwise:

          1. Decompose ``(tile_id, pid_n) = divmod(linear_idx, num_pid_n)``.
          2. **Warp-cooperative expert lookup** (overlaps with the spin below).
             Each scheduler-warp lane loads one entry of
             ``expert_pool_block_offset`` (or kNumExpertsPerLane entries for
             E_local > 31). ``vote_ballot_sync(cum <= tile_id)`` + ``popc``
             returns ``expert_id + 1`` directly (one PTX op of warp-collective
             work, no per-claim ``tile_id_to_expert`` GMEM read).
             ``pid_m = tile_id - expert_pool_block_offset[expert_id]`` falls out
             of a single ``shuffle_sync`` from the lane holding the matching
             cum.
          3. Lane 0 runs the per-tile ready spin selected by ``spin_kind``:
             either count-vs-target on
             ``pool_arrival_count[tile] == pool_arrival_target[tile]`` (kernel
             A / kernel_y_bwd), or acquire-vs-seq on
             ``ld_acquire(stamp[tile]) >= dispatch_seq`` (kernel Y /
             kernel_a_bwd).

        Combined with the consumer's varlen_m path that reads
        ``cu_seqlens_m = expert_pool_block_offset * tile_m``, the m-offset
        ``cu_seqlens_m[expert_id] + pid_m * tile_m`` lands at the correct
        pool row.

        Because dispatch's Pass 2 fires `pool_arrival_count` release-adds in
        expert-major order at substream end, linear-claim CTAs naturally walk
        experts in waves: 80 CTAs all start on expert 0's tile range, drain
        it, advance to expert 1, etc.
        """
        params = self.params
        total_work = params.total_tiles * params.num_pid_n
        linear_idx = Int32(-1)
        if cute.arch.lane_idx() == 0:
            head_ptr = utils.elem_pointer(params.consumer_head, (Int32(0),))
            linear_idx = utils.atomic_add_i32(1, head_ptr)
        linear_idx = cute.arch.shuffle_sync(linear_idx, 0)

        is_valid_i32 = Int32(linear_idx < total_work)
        pid_n = Int32(0)
        pid_m = Int32(0)
        expert_id = Int32(0)
        if is_valid_i32 != 0:
            # Each linear_idx represents one CLUSTER's claim along the cluster-N
            # axis. `params.num_pid_n` already divides total_pid_n by cluster_n,
            # so the divmod gives a "cluster_pid_n" in [0, total_pid_n / cluster_n).
            # We multiply by cluster_n to recover the leader-CTA pid_n; peer CTAs
            # in the same cluster will add their own bidy_in_cluster offset on
            # the receive side (see write_work_tile_to_smem).
            tile_id, cluster_pid_n = divmod(linear_idx, params.num_pid_n_fdd)
            pid_n = cluster_pid_n * Int32(params.cluster_shape_mnk[1])

            # Lane 0 spins on the per-tile ready signal. Spin-then-ballot
            # ordering is load-bearing (the acquire-load fences the subsequent
            # ballot reads of `expert_pool_block_offset`; validated
            # empirically — ballot-before-spin exposed a stale-L1 coherence
            # window for the first ~5 iters).
            if cute.arch.lane_idx() == 0:
                if const_expr(params.spin_kind == SpinKind.COUNT_VS_TARGET):
                    # Multi-producer release-add: each producer fires
                    # `red.release.gpu.global.add.s32`; consumer terminates
                    # once count hits target.
                    count_ptr = utils.elem_pointer(
                        params.pool_arrival_count, (tile_id,)
                    )
                    target = params.pool_arrival_target[tile_id]
                    while ld_acquire_gpu_global_i32(count_ptr) != target:
                        pass
                else:
                    # ACQUIRE_VS_SEQ: single-producer release-stamp; consumer
                    # terminates once the int64 stamp >= the seq this kernel
                    # was launched with.
                    stamp_ptr = utils.elem_pointer(params.stamp, (tile_id,))
                    while ld_acquire_gpu_global(stamp_ptr) < cutlass.Int64(
                        params.dispatch_seq
                    ):
                        pass

            # Warp-cooperative expert lookup. `expert_pool_block_offset` has
            # length E_local + 1 (last entry = total_tiles); `kNumExpertsPerLane`
            # is fixed at 2 here, sized for E_local up to 63 (covers all
            # configured world sizes: intranode E=384/8=48, internode E=384/16=24).
            # Out-of-range slots get INT_MAX sentinel so they never match the
            # ballot; the upper-sentinel cum[E_local]=total_tiles is loaded but
            # also never matches because tile_id < total_tiles.
            kNumExpertsPerLane = const_expr(2)
            # num_local_experts derived from the tensor shape (runtime Int32);
            # comparison gates per-slot loads to in-range indices.
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

            # Count cums <= tile_id across all slots. Since expert_pool_block_offset
            # is monotone non-decreasing and INF for out-of-range, the count
            # equals expert_id + 1 (number of indices 0..expert_id where
            # cum <= tile_id).
            n_matched_total = Int32(0)
            for i in cutlass.range_constexpr(kNumExpertsPerLane):
                n_matched_total += cute.arch.popc(
                    cute.arch.vote_ballot_sync(cum_slots[i] <= tile_id)
                )
            expert_id = n_matched_total - Int32(1)

            # pid_m = tile_id - expert_pool_block_offset[expert_id].
            # Shuffle the matching cum from (expert_slot, expert_lane).
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

        self._current_expert = expert_id
        self._current_pid_m = pid_m
        self._current_pid_n = pid_n
        return is_valid_i32

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> cutlass.utils.WorkTileInfo:
        # _fetch_next_work_idx stashed the per-tile claim result onto self.
        if const_expr(is_valid is None):
            is_valid = work_idx != Int32(0)
        # tile_coord_mnkl[0] (pid_m): tile_in_e — drives varlen_m m-offset.
        # tile_coord_mnkl[2] (the K slot): None — same convention as
        #   VarlenMTileScheduler. Kernel A's mainloop and postact path don't
        #   read it; kernel Y will reconstruct tile_id from
        #   `expert_pool_block_offset[batch_idx] + pid_m` if it needs the
        #   per-pool-slot reverse-map index.
        # tile_coord_mnkl[3] (batch_idx): expert_id — kernel body W1[e] select.
        tile_coord_mnkl = (
            self._current_pid_m,
            self._current_pid_n,
            None,
            self._current_expert,
        )
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    @cute.jit
    def write_work_tile_to_smem(
        self, work_tile_info: cutlass.utils.WorkTileInfo, *, loc=None, ip=None
    ):
        """Write 4 ints to _sched_smem: (pid_m, pid_n, batch_idx=expert_id, is_valid).

        Single-cluster path: thread-local SMEM store + pipeline producer_commit
        (matches the upstream cluster=1 fast path).

        Multi-cluster path: scheduler-warp lanes 0..cluster_size-1 each
        ``store_shared_remote_x4`` the 4-int payload to one peer CTA's SMEM,
        adding `bidx_in_cluster` to pid_m and `bidy_in_cluster` to pid_n so
        each peer sees its correct (pid_m, pid_n) tile coord.

        For our streaming MoE pool layout cluster_M is currently constrained
        to 1: pool tiles are bound to specific experts (via tile_id_to_expert),
        so a cluster spanning two consecutive tile_ids could straddle an
        expert boundary. cluster_N is the useful axis — multiple CTAs share
        the same A operand (pool rows) and TMA-multicast it on the way in.
        """
        params = self.params
        if const_expr(self._sched_smem is not None):
            pipeline_state_producer = PipelineStateWAdvance(
                self._pipeline_state.stages,
                self._pipeline_state.count,
                self._pipeline_state.index,
                self._pipeline_state.phase ^ 1,
            )
            self._scheduler_pipeline.producer_acquire(pipeline_state_producer)
            sched_data = [
                work_tile_info.tile_idx[0],  # pid_m (leader-CTA value)
                work_tile_info.tile_idx[1],  # pid_n (leader-CTA value, see _fetch)
                work_tile_info.tile_idx[3],  # batch_idx = expert_id
                Int32(work_tile_info.is_valid_tile),
            ]
            lane_idx = cute.arch.lane_idx()
            # cluster_M==1 invariant for streaming pool layout (see docstring).
            assert (
                params.cluster_shape_mnk[0] == 1
            ), "StreamingTileScheduler requires cluster_shape_mnk[0] == 1 (cluster_M=1)"
            pipeline_idx = self._pipeline_state.index
            if const_expr(cute.size(params.cluster_shape_mnk) == 1):
                # Single-CTA cluster — thread-local SMEM write.
                if lane_idx == 0:
                    for i in cutlass.range_constexpr(4):
                        self._sched_smem[i, pipeline_idx] = sched_data[i]
                    self._scheduler_pipeline.producer_commit(self._pipeline_state)
            else:
                # Multi-cluster: lanes 0..cluster_size-1 each push the 4-int
                # payload to one peer CTA's SMEM via st.async.shared::cluster.
                # The async store's mbarrier::complete_tx also serves as the
                # producer-side commit (the consumer_wait on the matching
                # pipeline stage will release once all peers' tx bytes land).
                if lane_idx < cute.size(params.cluster_shape_mnk):
                    peer_cta_rank_in_cluster = lane_idx
                    # Cluster CTA ordering: x fastest, then y, then z. With
                    # cluster_M==1 fixed, bidx_in_cluster is always 0.
                    bidy_in_cluster = peer_cta_rank_in_cluster % params.cluster_shape_mnk[1]
                    mbar_ptr = self._scheduler_pipeline.producer_get_barrier(
                        self._pipeline_state
                    )
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar_ptr, 16, peer_cta_rank_in_cluster
                    )
                    store_shared_remote_x4(
                        sched_data[0],
                        sched_data[1] + bidy_in_cluster,
                        sched_data[2],
                        sched_data[3],
                        smem_ptr=self._sched_smem[None, pipeline_idx].iterator,
                        mbar_ptr=mbar_ptr,
                        peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                    )

    @cute.jit
    def setup_initial_work_tile(
        self, is_scheduler_warp: bool | Boolean = False, *, loc=None, ip=None
    ) -> cutlass.utils.WorkTileInfo:
        """For streaming, the first work tile must come from the producer's
        atomic-claim + queue spin + sched_smem write — there is no static
        initial `_current_work_idx`. Both producer and consumer warps call
        this; producer's `advance_to_next_work` does the fetch+write, consumer's
        is a no-op; both then read the populated sched_smem via
        `get_current_work`.
        """
        self.advance_to_next_work(is_scheduler_warp=is_scheduler_warp, loc=loc, ip=ip)
        return self.get_current_work(loc=loc, ip=ip)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        """Read the upstream 4-int payload from sched_smem
        (pid_m, pid_n, batch_idx=expert_id, is_valid) and produce a
        WorkTileInfo with tile_idx ``(pid_m, pid_n, None, batch_idx)``.
        """
        params = self.params
        self._scheduler_pipeline.consumer_wait(self._pipeline_state)
        pid_m, pid_n, batch_idx, is_valid_i32 = [
            self._sched_smem[i, self._pipeline_state.index] for i in range(4)
        ]
        if const_expr(cute.size(params.cluster_shape_mnk) > 1):
            cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            self._scheduler_pipeline.consumer_release(self._pipeline_state)
        self._pipeline_state.advance()
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, Boolean(is_valid_i32))

    @cute.jit
    def advance_to_next_work(
        self,
        is_scheduler_warp: bool | Boolean = False,
        *,
        advance_count: int = 1,
        loc=None,
        ip=None,
    ):
        """Streaming variant. Same flow as TileScheduler.advance_to_next_work but
        always uses the STREAMING fetch path.
        """
        self.num_tiles_executed += Int32(advance_count)
        if const_expr(self._pipeline_state is not None and advance_count > 1):
            self._pipeline_state.advance_iters(advance_count - 1)
        if is_scheduler_warp:
            self._current_work_idx = self._fetch_next_work_idx(loc=loc, ip=ip)
            work_tile_info = self._delinearize_work_idx(
                self._current_work_idx, block_zero_only=True, loc=loc, ip=ip
            )
            self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._current_work_idx,
            self.num_tiles_executed,
            self._current_expert,
            self._current_pid_m,
            self._current_pid_n,
            self._sched_smem,
            self._scheduler_pipeline,
            self._pipeline_state,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_idx,
                self.num_tiles_executed,
                self._current_expert,
                self._current_pid_m,
                self._current_pid_n,
                self._sched_smem,
                self._scheduler_pipeline,
                self._pipeline_state,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)
