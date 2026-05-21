#include "api.cuh"
#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"
#include <cooperative_groups.h>

namespace stream_ep {

namespace intranode {

// Per-sub-batch chunk size in the dispatch receiver. Bounds the batch_slot SMEM
// scratch independent of the channel queue depth — Pass A allocates pool slots
// for at most kReceiverChunkSize (chunk, k) pairs at a time before Pass B drains
// them. Used both in the kernel and at the launch site for SMEM sizing.
constexpr int kReceiverChunkSize = 32;

// Compile-time bounds for the streaming dispatch path. All three derive from
// "must fit in a 32-lane warp": E_local indexes per-(c, src, e) SMEM slots that
// lane 0 walks unrolled; topk and ranks both index lane masks (uint32 dst_mask
// in the metadata kernel, lane-as-rank patterns in the dispatch / combine
// kernels). Increase only if the kernel logic is widened correspondingly.
constexpr int kMaxLocalExpertsPerRank = 64;
constexpr int kMaxTopK = 32;
constexpr int kMaxRanks = 32;

// Receiver-state SMEM layout, used by the dispatch kernel's receiver block. Sits
// after the per-warp TMA buffer slabs in dynamic SMEM:
//   smem_per_substream_seen [num_ranks][E_local]                       — per-(c, src, e) cumulative count
//   smem_batch_slot         [num_ranks][kReceiverChunkSize][num_topk]  — per-sub-batch slot map
__host__ __device__ inline int receiver_state_smem_bytes(int num_ranks, int E_local, int num_topk) {
    return (num_ranks * E_local +
            num_ranks * kReceiverChunkSize * num_topk) * static_cast<int>(sizeof(int));
}

// ── Shared device helpers (used by both fwd and bwd sender code paths) ──

// Block the calling warp until the destination IPC queue has at least
// `num_required_free` empty slots. Lane-elected polling on `channel_head_idx`;
// caller must `__syncwarp` after only if it needs all lanes to observe the
// release point (this helper already does it).
//
// Used by fwd dispatch sender, fwd combine sender, and (after backward lands)
// the bwd dispatch_grads / combine_grads senders. They all share the same
// IPC ringbuffer protocol; only the per-call free-slot requirement differs.
__device__ __forceinline__ void sender_wait_for_queue_space(int cached_tail_idx,
                                                            uint64_t* channel_head_idx,
                                                            int64_t nvl_seq,
                                                            int num_recv_buffer_tokens,
                                                            int num_required_free,
                                                            int rank,
                                                            int responsible_channel,
                                                            const char* role) {
    auto start_time = clock64();
    if (elect_one_sync()) {
        while (true) {
            // Genstamp read: head slot encodes `(seq32, value32)`. Seq
            // mismatch ⇒ stale residue from a prior iter, treat head as 0
            // (this iter's start; sender then sees num_used = cached_tail,
            // and either spins until receiver writes a real head with this
            // iter's seq, or proceeds if cached_tail leaves enough room).
            uint64_t raw = static_cast<uint64_t>(ld_volatile_global(channel_head_idx));
            int head = nvl_seq_match(raw, nvl_seq) ? nvl_unpack_value(raw) : 0;
            int num_used_slots = cached_tail_idx - head;
            if (num_recv_buffer_tokens - num_used_slots >= num_required_free)
                break;
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("DeepEP timeout for %s, rank %d, responsible_channel = %d\n",
                       role, rank, responsible_channel);
                trap();
            }
        }
    }
    __syncwarp();
}


// Streaming-MoE consolidated dispatch metadata kernel. Cooperative-grid:
// launched with `num_blocks_v2` blocks × kNumThreadsV2 threads. Block 0 owns
// the cross-rank IPC exchange (Phases 1-5, single-warp / single-block work);
// all blocks participate in the wide-parallel post-exchange phases (6-8).
//
// IPC slab at `buffer_ptrs[R] + streaming_section_offset` carries two adjacent
// inboxes (zeroed on Buffer construction; cross-rank stores via per-(c, src)
// unique slots — no atomicAdds across senders):
//   - e_inbox[c, src, e]: per-(channel, src, local_expert) (token, k) count
//   - u_inbox[c, src]:    per-(channel, src) UNIQUE token count
//
// Phases (vs v1: 1-5 are the same single-block path; 6-8 are now grid-wide):
//   1-5 (block 0): build local SMEM histogram from topk_idx, bulk-store to
//        peers' IPC inboxes, bracketed by two `barrier_block` calls so peer
//        writes are observable on the post-barrier read.
//   GRID SYNC.
//   6 (all blocks, grid-stride): copy inbox -> seen_per_substream, then a
//      grid-stride atomicAdd reduce into expert_frequency. Replaces v1's
//      thread-per-expert serial sum (E_local threads, the rest idle).
//   GRID SYNC.
//   7 (block 0 thread 0): smem_pool_blk prefix, total_tiles, mapped counters,
//      rank_prefix_matrix column. Small serial work — O(E + kNumRanks).
//   GRID SYNC.
//   8 (one block per expert, round-robin): base_pool parallel prefix-scan
//      over (c, src) lex order. Per expert: SMEM-staged scan with cooperative
//      threads. Replaces v1's per-thread `acc += seen[cs, e]` 320-iter serial
//      dependency loop. E ≤ num_blocks at production, so all experts run in
//      parallel in one grid wave.
//   GRID SYNC.
//   9 (all blocks, grid-stride): tile_id_to_expert + pool_arrival_target —
//      one thread per tile. Replaces v1's per-expert serial tile walk.
//
// Pool layout: each expert's region in `pool` starts at
// `expert_pool_block_offset[e] * BLOCK_M` (BLOCK_M = tile_m) and is padded up to
// a BLOCK_M multiple. base_pool[c, src, e] is the substream's first slot for
// expert e (deterministic given cached routing).
// Phase A (cross-rank IPC exchange): single block, 60 KB SMEM, NOT cooperative.
// Builds local SMEM histogram by scanning topk_idx, bracketed by two
// barrier_block calls so peer writes are observable on the post-barrier read.
// This kernel does ONLY Phase 1-5 from the v1 single-kernel design; Phase 6-9
// lives in `_phase_b_kernel` below, launched FIFO-after on the same stream.
// Splitting lets phase B run with much smaller per-block SMEM (~1.5 KB vs
// 60 KB), which:
//   (a) avoids the cooperative-launch occupancy ceiling (1 block/SM at 60 KB
//       SMEM → 32 cooperative blocks fit cleanly across 132 SMs);
//   (b) lets the cooperative scheduler pack more concurrent blocks per SM in
//       Phase B (4-6 at ~5 KB), so SM-availability variance from the prior
//       iter's combine tail no longer pins this kernel to 1-block-per-SM.
template <int kNumRanks>
__global__ void streaming_dispatch_metadata_phase_a_kernel(
        const topk_idx_t* topk_idx,
        int* expert_frequency,
        int* expert_pool_block_offset,
        int* base_pool,
        int* seen_per_substream,
        int* rank_prefix_matrix,
        int* tile_id_to_expert,
        int* pool_arrival_target,
        int* total_tiles_out,
        int* num_recv_mapped,
        int* num_recv_per_expert_mapped,
        int* total_tiles_mapped,
        int num_tokens,
        int num_topk,
        int num_experts_per_rank,
        int num_channels,
        int64_t streaming_section_offset,
        void** buffer_ptrs,
        int** barrier_signal_ptrs,
        int rank,
        int tile_m,
        int expert_alignment) {
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    constexpr int kNumWarpsV2 = 256 / 32;  // matches kNumThreadsV2 in launcher.

    const int block_id = static_cast<int>(blockIdx.x);
    const int num_blocks = static_cast<int>(gridDim.x);
    const int thread_id = static_cast<int>(threadIdx.x);
    const int num_threads = static_cast<int>(blockDim.x);
    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;
    const int num_warps = num_threads / 32;
    const int grid_tid = block_id * num_threads + thread_id;
    const int grid_threads = num_blocks * num_threads;

    const int E = num_experts_per_rank;
    const int e_inbox_size = num_channels * kNumRanks * E;
    const int slab_e_size = num_channels * E;       // per-dst slice of smem_local_e
    const int slab_u_size = num_channels;            // per-dst slice of smem_local_u

    extern __shared__ int smem[];
    int* smem_local_e = smem;                                          // [kNumRanks * slab_e_size]
    int* smem_local_u = smem_local_e + kNumRanks * slab_e_size;        // [kNumRanks * slab_u_size]

    // ──────────────────────────────────────────────────────────────────────
    // PHASES 1-5: block 0 only. Cross-rank histogram exchange via IPC.
    // Other blocks idle here — they can't help (cross-rank IPC + barrier_block
    // are coordinated through peer-rank atomic counters that don't shard
    // across blocks). They'll join the grid-wide reduction in Phase 6+.
    // ──────────────────────────────────────────────────────────────────────
    if (block_id == 0) {
        // Phase 1: zero local histograms.
        for (int i = thread_id; i < kNumRanks * slab_e_size; i += num_threads)
            smem_local_e[i] = 0;
        for (int i = thread_id; i < kNumRanks * slab_u_size; i += num_threads)
            smem_local_u[i] = 0;
        __syncthreads();

        // Phase 2: cross-rank handshake.
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        // Phase 3: build local histograms.
        int num_tokens_per_channel = (num_tokens + num_channels - 1) / num_channels;
        for (int t = thread_id; t < num_tokens; t += num_threads) {
            int channel_id = t / num_tokens_per_channel;
            uint64_t dst_mask = 0;
            for (int k = 0; k < num_topk; ++k) {
                int e_global = static_cast<int>(topk_idx[t * num_topk + k]);
                if (e_global < 0)
                    continue;
                int dst_rank = e_global / E;
                int e_local = e_global - dst_rank * E;
                atomicAdd(&smem_local_e[dst_rank * slab_e_size + channel_id * E + e_local], 1);
                uint64_t bit = 1ULL << dst_rank;
                if (!(dst_mask & bit)) {
                    dst_mask |= bit;
                    atomicAdd(&smem_local_u[dst_rank * slab_u_size + channel_id], 1);
                }
            }
        }
        __syncthreads();

        // Phase 4: bulk store to peers' inboxes (each (sender, dst) writes unique slots).
        for (int dst = warp_id; dst < kNumRanks; dst += num_warps) {
            auto* peer_e = reinterpret_cast<int*>(
                static_cast<uint8_t*>(buffer_ptrs[dst]) + streaming_section_offset);
            auto* peer_u = peer_e + e_inbox_size;
            for (int i = lane_id; i < slab_e_size; i += 32) {
                int c = i / E;
                int e = i - c * E;
                peer_e[c * kNumRanks * E + rank * E + e] = smem_local_e[dst * slab_e_size + i];
            }
            for (int c = lane_id; c < num_channels; c += 32)
                peer_u[c * kNumRanks + rank] = smem_local_u[dst * slab_u_size + c];
        }
        __syncthreads();

        // Phase 5: cross-rank barrier — peers' phase-4 stores observable on
        // post-barrier read.
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
    }
}  // end streaming_dispatch_metadata_phase_a_kernel

// Phase B (post-exchange compute): cooperative grid, ~1.5 KB SMEM/block.
// Reads from this rank's IPC inbox (peer-populated by Phase A) and computes
// the metadata outputs the dispatch hot path consumes. Launched FIFO-after
// the Phase A kernel on the same stream.
template <int kNumRanks>
__global__ void streaming_dispatch_metadata_phase_b_kernel(
        int* expert_frequency,
        int* expert_pool_block_offset,
        int* base_pool,
        int* seen_per_substream,
        int* rank_prefix_matrix,
        int* tile_id_to_expert,
        int* pool_arrival_target,
        int* total_tiles_out,
        int* num_recv_mapped,
        int* num_recv_per_expert_mapped,
        int* total_tiles_mapped,
        int num_experts_per_rank,
        int num_channels,
        int64_t streaming_section_offset,
        void** buffer_ptrs,
        int rank,
        int tile_m,
        int expert_alignment) {
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    constexpr int kNumWarpsV2 = 256 / 32;

    const int block_id = static_cast<int>(blockIdx.x);
    const int num_blocks = static_cast<int>(gridDim.x);
    const int thread_id = static_cast<int>(threadIdx.x);
    const int num_threads = static_cast<int>(blockDim.x);
    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;
    const int num_warps = num_threads / 32;
    const int grid_tid = block_id * num_threads + thread_id;
    const int grid_threads = num_blocks * num_threads;

    const int E = num_experts_per_rank;
    const int e_inbox_size = num_channels * kNumRanks * E;

    extern __shared__ int smem[];

    // ──────────────────────────────────────────────────────────────────────
    // PHASE 6: seen_per_substream copy + expert_frequency warp-reduce.
    //
    // Two grid-parallel work items folded into one block of work:
    //   - seen_per_substream[idx] = my_e_inbox[idx] (grid-stride, all blocks).
    //   - expert_frequency[e] = warp-reduce sum over (c, src) of inbox[c,src,e].
    //     One warp owns one expert; the warp's lanes stride over cs_total,
    //     accumulating partials in a register, then a 5-step __shfl_down_sync
    //     reduces to the warp leader which writes expert_frequency[e].
    //
    // No need for atomicAdd or pre-zero of expert_frequency — each (block,
    // warp) writes a disjoint (e) slot. Saves one grid.sync vs the
    // atomic-reduce pattern.
    // ──────────────────────────────────────────────────────────────────────
    auto* my_e_inbox = reinterpret_cast<int*>(
        static_cast<uint8_t*>(buffer_ptrs[rank]) + streaming_section_offset);
    auto* my_u_inbox = my_e_inbox + e_inbox_size;

    const int seen_total = num_channels * kNumRanks * E;

    // 6a: copy inbox -> seen_per_substream (contiguous, all blocks grid-stride).
    for (int idx = grid_tid; idx < seen_total; idx += grid_threads)
        seen_per_substream[idx] = my_e_inbox[idx];

    // 6b: warp-per-expert reduce over (c, src) -> expert_frequency[e].
    // Block handles num_warps experts; iterate if E > num_blocks * num_warps.
    const int cs_total_phase6 = num_channels * kNumRanks;
    const int experts_per_iter6 = num_blocks * kNumWarpsV2;
    for (int e_iter = 0; ; ++e_iter) {
        int e = block_id * kNumWarpsV2 + warp_id + e_iter * experts_per_iter6;
        if (e >= E) break;

        int sum = 0;
        for (int cs = lane_id; cs < cs_total_phase6; cs += 32)
            sum += my_e_inbox[cs * E + e];
        // Warp-level reduction (sum across lanes).
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
        if (lane_id == 0)
            expert_frequency[e] = sum;
    }

    // Grid-wide barrier #2: seen_per_substream + expert_frequency fully
    // written before Phase 7 reads expert_frequency for the prefix.
    grid.sync();

    // ──────────────────────────────────────────────────────────────────────
    // PHASE 7: smem_pool_blk prefix, total_tiles, mapped counters,
    // rank_prefix_matrix column. Block 0 thread 0 — small serial work.
    // ──────────────────────────────────────────────────────────────────────
    if (block_id == 0 && thread_id == 0) {
        int cum_blocks = 0;
        expert_pool_block_offset[0] = 0;
        for (int e = 0; e < E; ++e) {
            int n_blocks_e = (expert_frequency[e] + tile_m - 1) / tile_m;
            cum_blocks += n_blocks_e;
            expert_pool_block_offset[e + 1] = cum_blocks;
        }
        *total_tiles_out = cum_blocks;
        *total_tiles_mapped = cum_blocks;

        // num_recv_per_expert_mapped (host-mapped, alignment).
        for (int e = 0; e < E; ++e) {
            int aligned = (expert_frequency[e] + expert_alignment - 1) / expert_alignment * expert_alignment;
            num_recv_per_expert_mapped[e] = aligned;
        }

        // rank_prefix_matrix column for this rank + total recv counter.
        // smem_per_src[i] = sum over c of u_inbox[c, i] — total unique tokens
        // from sender i to this rank.
        int cum_src = 0;
        for (int i = 0; i < kNumRanks; ++i) {
            int per_src = 0;
            for (int c = 0; c < num_channels; ++c)
                per_src += my_u_inbox[c * kNumRanks + i];
            cum_src += per_src;
            rank_prefix_matrix[i * kNumRanks + rank] = cum_src;
        }
        *num_recv_mapped = cum_src;
    }

    // Grid-wide barrier #4: expert_pool_block_offset / total_tiles / mapped
    // counters visible to all blocks for phases 8-9.
    grid.sync();

    // ──────────────────────────────────────────────────────────────────────
    // PHASE 8: base_pool — per-expert exclusive prefix-scan over (c, src)
    // lex order, starting from `expert_pool_block_offset[e] * tile_m`.
    //
    // One block per expert (round-robin if E > num_blocks). Within a block:
    // chunked warp-parallel scan (Kogge-Stone within warp + warp 0 scans
    // warp sums). Replaces the per-thread acc-serial-dep loop of v1.
    // ──────────────────────────────────────────────────────────────────────
    const int cs_total = num_channels * kNumRanks;
    int* s_scan = smem;                          // [cs_total] scan workspace.
    int* s_warp_sums = s_scan + cs_total;        // [kNumWarpsV2] warp-sum scratch.

    for (int e_iter = 0; ; ++e_iter) {
        int e = block_id + e_iter * num_blocks;
        if (e >= E) break;

        const int e_start_offset = expert_pool_block_offset[e] * tile_m;

        // 8a: load seen[cs, e] into SMEM in (c, src) lex order.
        for (int cs = thread_id; cs < cs_total; cs += num_threads)
            s_scan[cs] = seen_per_substream[cs * E + e];
        __syncthreads();

        // 8b: chunked warp-parallel exclusive scan with initial carry =
        // e_start_offset. For cs_total > num_threads, scan in chunks of
        // num_threads, propagating a running carry between chunks.
        int carry = e_start_offset;
        for (int chunk_start = 0; chunk_start < cs_total; chunk_start += num_threads) {
            int idx = chunk_start + thread_id;
            int orig_x = (idx < cs_total) ? s_scan[idx] : 0;

            // Warp-level inclusive scan via Kogge-Stone shuffle.
            int x = orig_x;
            int y;
            y = __shfl_up_sync(0xffffffff, x, 1);  if (lane_id >= 1)  x += y;
            y = __shfl_up_sync(0xffffffff, x, 2);  if (lane_id >= 2)  x += y;
            y = __shfl_up_sync(0xffffffff, x, 4);  if (lane_id >= 4)  x += y;
            y = __shfl_up_sync(0xffffffff, x, 8);  if (lane_id >= 8)  x += y;
            y = __shfl_up_sync(0xffffffff, x, 16); if (lane_id >= 16) x += y;
            // x is now the inclusive scan within the warp.

            // Last lane of each warp writes warp_sum.
            if (lane_id == 31)
                s_warp_sums[warp_id] = x;
            __syncthreads();

            // Warp 0 does inclusive scan over warp_sums (kNumWarpsV2 = 8 entries).
            if (warp_id == 0) {
                int ws = (lane_id < kNumWarpsV2) ? s_warp_sums[lane_id] : 0;
                int wy;
                wy = __shfl_up_sync(0xffffffff, ws, 1); if (lane_id >= 1) ws += wy;
                wy = __shfl_up_sync(0xffffffff, ws, 2); if (lane_id >= 2) ws += wy;
                wy = __shfl_up_sync(0xffffffff, ws, 4); if (lane_id >= 4) ws += wy;
                if (lane_id < kNumWarpsV2) s_warp_sums[lane_id] = ws;
            }
            __syncthreads();

            // Compose exclusive scan output:
            //   exclusive_at_idx = (warp_inclusive - orig_x) + warp_prefix + carry
            int warp_prefix = (warp_id > 0) ? s_warp_sums[warp_id - 1] : 0;
            int exclusive_x = (x - orig_x) + warp_prefix + carry;

            if (idx < cs_total)
                s_scan[idx] = exclusive_x;

            // Update carry for the next chunk: full block sum is the last
            // warp's inclusive total, which sits at s_warp_sums[kNumWarpsV2 - 1].
            __syncthreads();
            carry += s_warp_sums[kNumWarpsV2 - 1];
            __syncthreads();
        }

        // 8c: write base_pool from SMEM.
        for (int cs = thread_id; cs < cs_total; cs += num_threads)
            base_pool[cs * E + e] = s_scan[cs];
        __syncthreads();
    }

    // Phase 9 reads expert_pool_block_offset (already synced in barrier #3)
    // and expert_frequency (synced in barrier #2). Phase 8 writes base_pool
    // and Phase 9 writes tile_id_to_expert + pool_arrival_target — disjoint
    // outputs, so no sync needed between Phase 8 and Phase 9.

    // ──────────────────────────────────────────────────────────────────────
    // PHASE 9: tile_id_to_expert + pool_arrival_target — grid-stride.
    // One thread per tile. Linear search over E to find owning expert (E ≤ 64,
    // cheap; could be binary search if E grows).
    // ──────────────────────────────────────────────────────────────────────
    int total_tiles = *total_tiles_out;
    for (int tile_id = grid_tid; tile_id < total_tiles; tile_id += grid_threads) {
        int e_found = 0;
        for (int e = 0; e < E; ++e) {
            if (tile_id < expert_pool_block_offset[e + 1]) {
                e_found = e;
                break;
            }
        }
        int e_start = expert_pool_block_offset[e_found];
        int n_tiles_e = expert_pool_block_offset[e_found + 1] - e_start;
        int n_e = expert_frequency[e_found];
        int t = tile_id - e_start;
        tile_id_to_expert[tile_id] = e_found;
        pool_arrival_target[tile_id] = (t == n_tiles_e - 1) ? (n_e - t * tile_m) : tile_m;
    }
}

void streaming_dispatch_metadata(const topk_idx_t* topk_idx,
                                 int* expert_frequency,
                                 int* expert_pool_block_offset,
                                 int* base_pool,
                                 int* seen_per_substream,
                                 int* rank_prefix_matrix,
                                 int* tile_id_to_expert,
                                 int* pool_arrival_target,
                                 int* total_tiles_out,
                                 int* num_recv_mapped,
                                 int* num_recv_per_expert_mapped,
                                 int* total_tiles_mapped,
                                 int num_tokens,
                                 int num_topk,
                                 int num_experts_per_rank,
                                 int num_channels,
                                 int64_t streaming_section_offset,
                                 void** buffer_ptrs,
                                 int** barrier_signal_ptrs,
                                 int rank,
                                 int num_ranks,
                                 int tile_m,
                                 int expert_alignment,
                                 cudaStream_t stream) {
    // Two-kernel split — Phase A (cross-rank IPC, single-block, 60 KB SMEM,
    // non-cooperative) followed by Phase B (cooperative grid, ~1.5 KB SMEM
    // per block). The Phase A→Phase B boundary uses stream FIFO (no event
    // needed); the cheap kernel-launch dep replaces what was a grid.sync()
    // in the single-kernel v2 design.

    // ── Phase A kernel: 1 block, 256 threads, 60 KB SMEM, NOT cooperative ──
    constexpr int kNumThreadsV2 = 256;
    int smem_bytes_a = num_ranks * num_channels * (num_experts_per_rank + 1) * sizeof(int);

#define STREAMING_DISPATCH_METADATA_PHASE_A_LAUNCH(ranks)                                    \
    EP_HOST_ASSERT(cudaFuncSetAttribute(                                                     \
                       streaming_dispatch_metadata_phase_a_kernel<ranks>,                    \
                       cudaFuncAttributeMaxDynamicSharedMemorySize,                          \
                       smem_bytes_a) == cudaSuccess);                                        \
    {                                                                                        \
        cudaLaunchConfig_t cfg_a = {1, kNumThreadsV2, 0, stream, nullptr, 0};                \
        cfg_a.dynamicSmemBytes = smem_bytes_a;                                               \
        LAUNCH_KERNEL(&cfg_a,                                                                \
                      streaming_dispatch_metadata_phase_a_kernel<ranks>,                     \
                      topk_idx,                                                              \
                      expert_frequency, expert_pool_block_offset, base_pool,                 \
                      seen_per_substream, rank_prefix_matrix,                                \
                      tile_id_to_expert, pool_arrival_target,                                \
                      total_tiles_out, num_recv_mapped,                                      \
                      num_recv_per_expert_mapped, total_tiles_mapped,                        \
                      num_tokens, num_topk, num_experts_per_rank, num_channels,              \
                      streaming_section_offset, buffer_ptrs, barrier_signal_ptrs,            \
                      rank, tile_m, expert_alignment);                                       \
    }                                                                                        \
    break

    SWITCH_RANKS(STREAMING_DISPATCH_METADATA_PHASE_A_LAUNCH);

#undef STREAMING_DISPATCH_METADATA_PHASE_A_LAUNCH

    // ── Phase B kernel: 32 cooperative blocks × 256 threads, small SMEM ──
    // SMEM: cs_total (= num_channels × kNumRanks) ints for parallel-scan
    // workspace + kNumWarpsV2 ints for warp-sum scratch. At intranode
    // production (num_channels=40, kNumRanks=8), cs_total=320 ints =
    // 1.28 KB + 32 bytes ≈ 1.3 KB per block — well within H100's 228 KB
    // per-SM SMEM, so 32 cooperative blocks pack ~4-6 blocks per SM if the
    // scheduler chooses (vs 1 block/SM at 60 KB).
    constexpr int kNumBlocksV2 = 32;
    int smem_bytes_b = (num_channels * num_ranks + (kNumThreadsV2 / 32)) * sizeof(int);

#define STREAMING_DISPATCH_METADATA_PHASE_B_LAUNCH(ranks)                                    \
    EP_HOST_ASSERT(cudaFuncSetAttribute(                                                     \
                       streaming_dispatch_metadata_phase_b_kernel<ranks>,                    \
                       cudaFuncAttributeMaxDynamicSharedMemorySize,                          \
                       smem_bytes_b) == cudaSuccess);                                        \
    {                                                                                        \
        cudaLaunchConfig_t cfg_b = {kNumBlocksV2, kNumThreadsV2, 0, stream, nullptr, 0};     \
        cudaLaunchAttribute attr_b[2];                                                       \
        attr_b[0].id = cudaLaunchAttributeCooperative;                                       \
        attr_b[0].val.cooperative = 1;                                                       \
        attr_b[1].id = cudaLaunchAttributeClusterDimension;                                  \
        attr_b[1].val.clusterDim.x = 1;                                                      \
        attr_b[1].val.clusterDim.y = 1;                                                      \
        attr_b[1].val.clusterDim.z = 1;                                                      \
        cfg_b.attrs = attr_b;                                                                \
        cfg_b.numAttrs = 2;                                                                  \
        cfg_b.dynamicSmemBytes = smem_bytes_b;                                               \
        LAUNCH_KERNEL(&cfg_b,                                                                \
                      streaming_dispatch_metadata_phase_b_kernel<ranks>,                     \
                      expert_frequency, expert_pool_block_offset, base_pool,                 \
                      seen_per_substream, rank_prefix_matrix,                                \
                      tile_id_to_expert, pool_arrival_target,                                \
                      total_tiles_out, num_recv_mapped,                                      \
                      num_recv_per_expert_mapped, total_tiles_mapped,                        \
                      num_experts_per_rank, num_channels,                                    \
                      streaming_section_offset, buffer_ptrs,                                 \
                      rank, tile_m, expert_alignment);                                       \
    }                                                                                        \
    break

    SWITCH_RANKS(STREAMING_DISPATCH_METADATA_PHASE_B_LAUNCH);

#undef STREAMING_DISPATCH_METADATA_PHASE_B_LAUNCH
}

template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch_main_kernel(
        DispatchPoolOut pool_out,
        DispatchPerTokenOut per_token_out,
        DispatchInputs inputs,
        DispatchTileSignal tile_signal,
        DispatchShape shape,
        DispatchEnv env) {
    // Bump the "kernel started" flag for the host-queued cuStreamWaitValue
    // gate. Block 0 thread 0 only — the flag is single-writer-per-kernel-call
    // monotonic. Host queues `wait flag >= issued_count` on the compute stream
    // before launching kernel_a so kernel_a doesn't grab SMs first under CDMC>1
    // + torch.compile, where SM contention would otherwise leave dispatch_main
    // queued behind kernel_a.
    if (blockIdx.x == 0 && threadIdx.x == 0 && tile_signal.started_flag != nullptr)
        atomicAdd(tile_signal.started_flag, 1);
    const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const bool is_sender = sm_id % 2 == 0;
    EP_DEVICE_ASSERT(num_sms % 2 == 0);

    // Several warps are response for a single rank
    const auto num_threads_per_rank = kNumThreads / kNumRanks;
    const auto num_channels = num_sms / 2;
    const auto responsible_rank = (static_cast<int>(thread_id)) / num_threads_per_rank;
    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    const auto responsible_channel = sm_id / 2;

    int num_experts_per_rank = shape.num_experts / kNumRanks;
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 and shape.num_topk > 0);
    // Pool-layout SMEM scratch is sized by E_local at runtime; a compile-time
    // upper bound keeps the receiver-block per-(chunk, k) slot computation
    // unrolled. The 32-lane bound covers any practical MoE: even Mixtral-style
    // routing has num_experts_per_rank well under 32 at world_size ≥ 8.
    EP_DEVICE_ASSERT(num_experts_per_rank <= kMaxLocalExpertsPerRank);
    EP_DEVICE_ASSERT(shape.num_topk <= kMaxTopK);
    EP_DEVICE_ASSERT((inputs.topk_idx == nullptr) == (inputs.topk_weights == nullptr));

    // Calculate pointers by the specific layout
    // `rank_prefix_matrix`: kNumRanks * kNumRanks * sizeof(int)
    auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(env.buffer_ptrs[is_sender ? responsible_rank : env.rank]) +
                                       kNumRanks * kNumRanks * sizeof(int));
    int target_rank = is_sender ? env.rank : responsible_rank;
    auto num_channels_total = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;

    // Channel buffer metadata
    // Senders are responsible for tails, and receivers are responsible for heads
    // Stored on the receiver side
    // The retired signals are actually boolean flags, but to align with 16 bytes, we make it `int64_t`
    // `start_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `end_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
    // Meta slots: 64-bit `(seq32, value32)` genstamped. Reader filters by
    // seq match — stale residue from a prior iter is treated as "not
    // written" (or as zero head for backpressure). See utils.cuh
    // `nvl_pack` / `nvl_seq_match` / `nvl_unpack_value`.
    auto channel_start_offset = Buffer<uint64_t>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset = Buffer<uint64_t>(ptr, num_channels_total, channel_rank_offset);
    auto channel_head_idx = Buffer<uint64_t>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx = Buffer<uint64_t>(ptr, num_channels_total, channel_rank_offset);
    const int64_t nvl_seq = tile_signal.dispatch_seq << 1;

    // Channel data buffers, stored on the receiver side
    auto channel_x_buffers = Buffer<int4>(
        ptr, num_channels_total * env.num_recv_buffer_tokens * shape.hidden_int4,
        channel_rank_offset * env.num_recv_buffer_tokens * shape.hidden_int4);
    auto channel_src_idx_buffers = Buffer<int>(
        ptr, num_channels_total * env.num_recv_buffer_tokens,
        channel_rank_offset * env.num_recv_buffer_tokens);
    auto channel_topk_idx_buffers = Buffer<topk_idx_t>(
        ptr, num_channels_total * env.num_recv_buffer_tokens * shape.num_topk,
        channel_rank_offset * env.num_recv_buffer_tokens * shape.num_topk);
    auto channel_topk_weights_buffers = Buffer<float>(
        ptr, num_channels_total * env.num_recv_buffer_tokens * shape.num_topk,
        channel_rank_offset * env.num_recv_buffer_tokens * shape.num_topk);

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto half_hidden_int4 = shape.hidden_int4 / 2;
    auto half_hidden_bytes = half_hidden_int4 * static_cast<int>(sizeof(int4));
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + half_hidden_bytes);
    uint32_t tma_phase = 0;
    if (elect_one_sync()) {
        mbarrier_init(tma_mbarrier, 1);
        fence_barrier_init();
        EP_DEVICE_ASSERT(shape.hidden_int4 % 2 == 0 and half_hidden_bytes + sizeof(uint64_t) <= kNumTMABytesPerWarp);
    }
    __syncwarp();
#endif

    if (is_sender) {
        // Workers for sending
        constexpr int num_send_warps = kNumThreads / 32;
        constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
        const auto send_thread_id = thread_id;
        const auto send_warp_id_in_rank = send_thread_id % num_threads_per_rank / 32;
        EP_DEVICE_ASSERT(kNumRanks <= kMaxRanks);
        EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

        // Compute the channel-prefix locally over `is_token_in_rank`. One warp
        // per (channel, dst_rank) pair scans tokens [0, channel_end_of_c) and
        // splits the count into start (i < channel_start) and end (full range)
        // using lane-strided accumulation + warp_reduce.
        if (send_warp_id_in_rank == 0) {
            int channel_start, channel_end;
            get_channel_task_range(shape.num_tokens, num_channels, responsible_channel, channel_start, channel_end);
            int start_count = 0, end_count = 0;
            for (int i = lane_id; i < channel_end; i += 32) {
                int v = inputs.is_token_in_rank[i * kNumRanks + responsible_rank];
                if (i < channel_start) start_count += v;
                else end_count += v;
            }
            start_count = warp_reduce_sum(start_count);
            end_count = warp_reduce_sum(end_count) + start_count;

            // Send offset by `-value - 1`, e.g. 0 -> -1, 1 -> -2
            // NOTES: this is for distinguishing zero tokens. With
            // genstamps the seq tag is the primary "written-this-iter"
            // signal; the negative bias is preserved for symmetry with
            // internode and so legacy callers still see consistent
            // values on unpack.
            if (elect_one_sync()) {
                st_relaxed_sys_global(channel_start_offset.buffer(), nvl_pack(nvl_seq, -start_count - 1));
                st_relaxed_sys_global(channel_end_offset.buffer(),   nvl_pack(nvl_seq, -end_count   - 1));
                // Persist inclusive cumulative through this channel for combine.
                tile_signal.channel_prefix_matrix[responsible_rank * num_channels + responsible_channel] = end_count;
            }
        }
        __syncwarp();

        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(shape.num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

        // Iterate over all tokens and send by chunks
        int cached_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            sender_wait_for_queue_space(cached_channel_tail_idx,
                                        channel_head_idx.buffer(),
                                        nvl_seq,
                                        env.num_recv_buffer_tokens,
                                        env.num_max_send_tokens,
                                        env.rank, responsible_channel,
                                        "dispatch senders");

            int chunk_token_idx = 0;
            while (chunk_token_idx < env.num_max_send_tokens and token_idx < token_end_idx) {
                // NOTES: for the same token, the warp assigned to save `send_head` may be different from the warp assigned to send the
                // following data
                if (token_idx % num_send_warps_per_rank == send_warp_id_in_rank and elect_one_sync())
                    per_token_out.send_head[token_idx * kNumRanks + responsible_rank] =
                        inputs.is_token_in_rank[token_idx * kNumRanks + responsible_rank] ? cached_channel_tail_idx : -1;

                // Skip if not selected
                if (not inputs.is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                    token_idx++;
                    continue;
                }

                // Get an empty slot
                int dst_slot_idx = (cached_channel_tail_idx++) % env.num_recv_buffer_tokens;
                if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
                    // Copy data
                    auto shifted_channel_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * shape.hidden_int4;
                    auto shifted_x = inputs.x + token_idx * shape.hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, shape.hidden_int4, shifted_channel_x_buffers, shifted_x, __ldg, st_na_global);

                    // Copy source index
                    if (elect_one_sync())
                        channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);

                    // Copy `topk_idx` and `topk_weights` with transformed index
                    if (lane_id < shape.num_topk) {
                        // Top-k index
                        int recv_expert_begin = responsible_rank * num_experts_per_rank,
                            recv_expert_end = (responsible_rank + 1) * num_experts_per_rank;
                        auto idx_value = __ldg(inputs.topk_idx + token_idx * shape.num_topk + lane_id);
                        idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end) ? idx_value - recv_expert_begin : -1;
                        channel_topk_idx_buffers[dst_slot_idx * shape.num_topk + lane_id] = idx_value;

                        // Top-k weights
                        auto weight_value = __ldg(inputs.topk_weights + token_idx * shape.num_topk + lane_id);
                        weight_value = (idx_value >= 0) ? weight_value : 0.0f;
                        channel_topk_weights_buffers[dst_slot_idx * shape.num_topk + lane_id] = weight_value;
                    }

                }

                // Move token index
                chunk_token_idx++, token_idx++;
            }

            // Move tail index
            // NOTES: here all warps should share the same new tail
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), nvl_pack(nvl_seq, cached_channel_tail_idx));
        }

        // Reverse-scan `send_head[token, responsible_rank]` over the range we
        // just wrote, encoding sentinels in-place. Fused from
        // `encode_combine_heads_kernel`'s former blocks 1..N (the encode kernel
        // now retains only block 0 / slab memset+barrier). One warp per
        // rank-group does the scan (other 2 exit). The per-chunk
        // `bar.sync %0, %1` at the chunk-loop tail above syncs all warps in
        // this rank-group on each iteration, so by loop exit all
        // `send_head[*, responsible_rank]` writes are visible to warp 0.
        // Visibility downstream is by communicate-stream FIFO (this rank's
        // combine_main_kernel reads the same sentinels) — no new
        // cross-rank release-acquire pair is needed, and the sentinels carry
        // through to bwd combine_grads because `handle.send_head` is the
        // same tensor.
        if (send_warp_id_in_rank == 0 and per_token_out.send_head != nullptr) {
            int last_head = kReverseScanSentinel;
            #pragma unroll
            for (int token_idx_tail = token_end_idx - 1; token_idx_tail >= token_start_idx; token_idx_tail -= 32) {
                int token_idx = token_idx_tail - lane_id, expected_head = 0;
                auto current_head = (token_idx >= token_start_idx)
                    ? __ldg(per_token_out.send_head + token_idx * kNumRanks + responsible_rank)
                    : -1;
                for (int i = 0; i < min(32, token_idx_tail - token_start_idx + 1); ++i) {
                    const int head = __shfl_sync(0xffffffff, current_head, i);
                    if (head < 0) {
                        if (lane_id == i)
                            expected_head = -last_head - 1;
                    } else {
                        last_head = head;
                    }
                }
                if (current_head < 0 and token_idx >= token_start_idx)
                    per_token_out.send_head[token_idx * kNumRanks + responsible_rank] = expected_head;
            }
        }
    } else {
        // Workers for receiving — pool layout. Each landed (token, k) pair routing
        // to a local expert e gets its own pool slot, allocated deterministically
        // as `slot = base_pool[c, src, e] + smem_seen[e]++`. Slot allocation is
        // chunk-major in lane 0 (sequential SMEM increments — fast and cheap), so
        // slot order = (chunk_idx_in_substream, k) lex regardless of how the
        // sender chunked or how the receiver was paced. The data copy still uses
        // 3-warp parallelism reading slot positions from the SMEM batch_slot scratch.
        // After all batches drain, Pass 2 fires pool_arrival_count (release-add)
        // in expert-major order so kernel A's count-vs-target spin unblocks in
        // tile_id-monotonic order (preserves W1[e] L2 caching).
        //
        // Per-iter inner-loop processes at most kReceiverChunkSize chunks (defined
        // at namespace scope); if the sender pushed a larger backlog into the
        // queue, we run multiple inner sub-iters before advancing the queue head.
        constexpr int num_recv_warps = kNumThreads / 32;
        constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
        const auto recv_thread_id = thread_id;
        const auto recv_thread_id_in_rank = recv_thread_id % num_threads_per_rank;
        const auto recv_warp_id_in_rank = recv_thread_id_in_rank / 32;
        const int E = num_experts_per_rank;
        EP_DEVICE_ASSERT(kNumRanks <= kMaxRanks);
        EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);

        // ── Receiver-state SMEM (sized by `receiver_state_smem_bytes` at the
        // launch site; placed after the per-warp TMA buffer slabs in dynamic SMEM).
        //   smem_per_substream_seen [kNumRanks][E] — cumulative (chunk, k) pair
        //     count routed to each local expert across all batches so far. Plus
        //     base_pool[c, src, e], gives the next slot for that (c, src, e).
        //   smem_batch_slot [kNumRanks][kReceiverChunkSize][num_topk] — per-(chunk,
        //     k) slot for the current sub-batch; -1 = nonlocal. Computed by lane 0
        //     in Pass A; consumed by all warps in Pass B (data copy + per-slot writes).
#ifndef DISABLE_SM90_FEATURES
        int* smem_per_substream_seen = reinterpret_cast<int*>(
            smem_buffer + kNumTMABytesPerWarp * num_recv_warps);
#else
        extern __shared__ int smem_recv_state[];
        int* smem_per_substream_seen = smem_recv_state;
#endif
        const int seen_stride = E;
        const int slot_stride = kReceiverChunkSize * shape.num_topk;
        int* smem_batch_slot = smem_per_substream_seen + kNumRanks * seen_stride;

        // Initialize cumulative seen for this substream's local experts.
        if (recv_thread_id_in_rank < E)
            smem_per_substream_seen[responsible_rank * seen_stride + recv_thread_id_in_rank] = 0;
        __syncthreads();

        // Calculate offset
        auto rank_prefix_matrix = static_cast<int*>(env.buffer_ptrs[env.rank]);
        int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + env.rank] : 0;

        // Receive channel offset. Genstamp reads: spin until the slot's
        // seq matches this iter's `nvl_seq` (= "sender wrote this iter").
        // Stale residue from prior iter mismatches; receiver keeps spinning.
        int total_offset, num_tokens_to_recv;
        if (elect_one_sync()) {
            while (true) {
                uint64_t raw = static_cast<uint64_t>(ld_volatile_global(channel_start_offset.buffer()));
                if (nvl_seq_match(raw, nvl_seq)) { total_offset = nvl_unpack_value(raw); break; }
            }
            while (true) {
                uint64_t raw = static_cast<uint64_t>(ld_volatile_global(channel_end_offset.buffer()));
                if (nvl_seq_match(raw, nvl_seq)) { num_tokens_to_recv = nvl_unpack_value(raw); break; }
            }
            total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
            if (recv_warp_id_in_rank == 0)
                per_token_out.recv_channel_prefix_matrix[responsible_rank * num_channels + responsible_channel] = total_offset;
            num_tokens_to_recv -= total_offset;
        }
        total_offset = __shfl_sync(0xffffffff, total_offset, 0);
        total_offset += rank_offset;  // becomes the recv-token row index for chunk_idx 0
        num_tokens_to_recv = __shfl_sync(0xffffffff, num_tokens_to_recv, 0);

        // Shared tail indices for different warps within a rank-group.
        __shared__ volatile int shared_channel_tail_idx[kNumRanks];

        const int substream_csrc = responsible_channel * kNumRanks + responsible_rank;
        const int* base_pool_substream = tile_signal.base_pool + substream_csrc * E;
        int* seen_substream       = smem_per_substream_seen + responsible_rank * seen_stride;
        int* batch_slot_substream = smem_batch_slot         + responsible_rank * slot_stride;

        auto start_time = clock64();
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            // Wait for queue tail (one thread per rank-group polls; others wait at bar).
            // Genstamp: ignore stale-seq reads (tail = 0 effectively).
            while (recv_thread_id_in_rank == 0) {
                uint64_t raw = ld_acquire_sys_global(channel_tail_idx.buffer());
                cached_channel_tail_idx = nvl_seq_match(raw, nvl_seq) ? nvl_unpack_value(raw) : 0;
                if (cached_channel_head_idx != cached_channel_tail_idx) {
                    shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
                    break;
                }
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP timeout for dispatch receivers, rank %d, responsible_channel = %d, tokens remained: %d\n",
                           env.rank, responsible_channel, num_tokens_to_recv);
                    trap();
                }
            }

            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];
            int batch_total = cached_channel_tail_idx - cached_channel_head_idx;

            // Inner sub-batch loop: bound the batch_slot SMEM scratch.
            int sub_start = 0;
            while (sub_start < batch_total) {
                int sub_size = min(kReceiverChunkSize, batch_total - sub_start);

                // ── Pass A: lane 0 of warp 0 sequentially walks (chunk, k) pairs in
                // chunk-major lex order, allocates pool slots from base_pool +
                // smem_per_substream_seen, writes batch_slot_substream[chunk, k]. The
                // sequential allocation makes slots independent of batch boundaries:
                // slot for (chunk_idx_in_substream, k) is the same in every dispatch
                // run with the same cached routing.
                if (recv_warp_id_in_rank == 0 and lane_id == 0) {
                    for (int sub_chunk = 0; sub_chunk < sub_size; ++sub_chunk) {
                        int chunk_idx = sub_start + sub_chunk;
                        int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % env.num_recv_buffer_tokens;
                        int* slot_row = batch_slot_substream + sub_chunk * shape.num_topk;
                        for (int k = 0; k < shape.num_topk; ++k) {
                            int e_local = static_cast<int>(
                                channel_topk_idx_buffers[token_idx_in_buffer * shape.num_topk + k]);
                            if (e_local >= 0 and e_local < E) {
                                int slot = base_pool_substream[e_local] + seen_substream[e_local];
                                seen_substream[e_local]++;
                                slot_row[k] = slot;
                            } else {
                                slot_row[k] = -1;
                            }
                        }
                    }
                }
                asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));

                // ── Pass B: per-(chunk, k) data copy + auxiliary writes. Each warp
                // owns chunks where sub_chunk % num_recv_warps_per_rank == warp_id;
                // lane 0 of the warp owns the per-pool-slot scalar writes (weight,
                // recv_token, k_slot); the warp's TMA buffer is reused across the K
                // pieces (load piece into SMEM, broadcast to multiple slot stores).
                for (int sub_chunk = recv_warp_id_in_rank; sub_chunk < sub_size;
                     sub_chunk += num_recv_warps_per_rank) {
                    int chunk_idx = sub_start + sub_chunk;
                    int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % env.num_recv_buffer_tokens;
                    int recv_token_id = total_offset + chunk_idx;
                    int* slot_row = batch_slot_substream + sub_chunk * shape.num_topk;

                    auto shifted_buffer_x_int4 = channel_x_buffers.buffer() + token_idx_in_buffer * shape.hidden_int4;
#ifndef DISABLE_SM90_FEATURES
                    // 2-piece pattern: tma_load piece p into tma_buffer, then for
                    // each local k tma_store piece p to pool[slot] + p*half_hidden.
                    #pragma unroll
                    for (int piece = 0; piece < 2; ++piece) {
                        tma_store_wait<0>();
                        if (elect_one_sync()) {
                            tma_load_1d(tma_buffer,
                                        shifted_buffer_x_int4 + piece * half_hidden_int4,
                                        tma_mbarrier, half_hidden_bytes);
                            mbarrier_arrive_and_expect_tx(tma_mbarrier, half_hidden_bytes);
                            mbarrier_wait(tma_mbarrier, tma_phase);
                            for (int k = 0; k < shape.num_topk; ++k) {
                                int slot = slot_row[k];
                                if (slot < 0) continue;
                                tma_store_1d(tma_buffer,
                                             pool_out.pool + static_cast<int64_t>(slot) * shape.hidden_int4
                                                  + piece * half_hidden_int4,
                                             half_hidden_bytes, false);
                            }
                        }
                        __syncwarp();
                    }
#else
                    for (int k = 0; k < shape.num_topk; ++k) {
                        int slot = slot_row[k];
                        if (slot < 0) continue;
                        auto shifted_pool_int4 = pool_out.pool + static_cast<int64_t>(slot) * shape.hidden_int4;
                        UNROLLED_WARP_COPY(5, lane_id, shape.hidden_int4, shifted_pool_int4, shifted_buffer_x_int4,
                                           ld_nc_global, st_na_global);
                    }
#endif

                    // Per-pool-slot scalar writes + per-recv-token K_local count.
                    // Intranode: each recv-token is delivered by exactly one
                    // substream, so the K_local count emitted here is final
                    // (no cross-substream accumulation needed). Pass 2's
                    // __threadfence (.gpu) + pool_arrival_count release-add
                    // after this make k_local_remaining[r] visible to kernel Y
                    // before kernel Y's first atomicSub on the same address.
                    //
                    // Phase F additions: recv_token_to_slots[r, :K] and
                    // k_local_total[r] are persisted here as well — both are
                    // consumed by the backward path (dispatch_grads receiver
                    // gathers slots; bwd setup memcpy's k_local_total into
                    // bwd_k_local_remaining). They share the lane-0 K-loop
                    // since both `slot_row[k]` and `recv_token_id` are already
                    // in scope. recv_token_to_slots gets a write for EVERY k
                    // (the value is -1 for non-local k); k_local_total
                    // duplicates k_local_remaining's value into a write-once
                    // buffer that fwd never decrements.
                    if (lane_id == 0) {
                        int k_local_total_val = 0;
                        for (int k = 0; k < shape.num_topk; ++k) {
                            int slot = slot_row[k];
                            per_token_out.recv_token_to_slots[recv_token_id * shape.num_topk + k] = slot;
                            if (slot < 0) continue;
                            pool_out.pool_topk_weight[slot] = ld_nc_global(
                                channel_topk_weights_buffers.buffer() + token_idx_in_buffer * shape.num_topk + k);
                            pool_out.pool_recv_token[slot] = recv_token_id;
                            pool_out.pool_k_slot[slot] = k;
                            ++k_local_total_val;
                        }
                        if (k_local_total_val > 0) {
                            per_token_out.k_local_remaining[recv_token_id] = k_local_total_val;
                            per_token_out.k_local_total[recv_token_id] = k_local_total_val;
                        }
                    }

                }

                asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
                sub_start += sub_size;
            }

            // Move queue head. Genstamp-pack with this iter's `nvl_seq` so
            // the sender's backpressure spin distinguishes a real head
            // publish from prior-iter residue at the same slot.
            cached_channel_head_idx += batch_total;
            total_offset += batch_total;
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 and elect_one_sync())
                st_relaxed_sys_global(channel_head_idx.buffer(), nvl_pack(nvl_seq, cached_channel_head_idx));

            num_tokens_to_recv -= batch_total;
        }

        // ── Pass 2: substream-end expert-major firing of pool_arrival_count.
        // After all batches drain, walk experts in order; for each pool block
        // this substream contributed to, `red.release.gpu.global.add.s32` the
        // substream's per-block count into pool_arrival_count[block]. Kernel A's
        // scheduler spins until count == pool_arrival_target[block]; expert-
        // major firing makes that unblock happen in tile_id-monotonic order
        // (preserves W1[e] L2 caching).
#ifndef DISABLE_SM90_FEATURES
        tma_store_wait<0>();
#endif
        asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
        // Every thread in the rank-group fences its own prior pool writes at
        // device scope. ``pool_recv_token`` / ``pool_topk_weight`` /
        // ``pool_k_slot`` / ``k_local_remaining`` / ``recv_token_to_slots`` /
        // ``k_local_total`` are written by lane 0 of EVERY receiver warp
        // during the batch loop; a thread-0-only fence would only cover that
        // one thread's writes, and block-scope ``bar.sync`` doesn't propagate
        // beyond the block. ``__threadfence()`` (device-scope) is sufficient
        // because all consumers (kernel A acquiring ``pool_arrival_count``;
        // kernel Y reading the pool scalars after Y_started) run on the SAME
        // GPU. ``.sys`` scope was previously used as a carry-over from
        // before the 2-stream graph cleanup — cheaper to fence within L2
        // than across the PCIe / NVLink boundary.
        __threadfence();
        asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
        if (recv_thread_id_in_rank == 0) {
            for (int e = 0; e < E; ++e) {
                int n_writes_for_e = seen_substream[e];
                if (n_writes_for_e == 0) continue;
                fire_pool_blocks(base_pool_substream[e], n_writes_for_e, shape.tile_m,
                                 tile_signal.pool_arrival_count);
            }
        }
    }
}

void launch_dispatch_main(const DispatchPoolOut& pool_out,
                          const DispatchPerTokenOut& per_token_out,
                          const DispatchInputs& inputs,
                          const DispatchTileSignal& tile_signal,
                          const DispatchShape& shape,
                          const DispatchEnv& env,
                          int num_ranks,
                          cudaStream_t stream,
                          int num_sms) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 8192;
    constexpr int kNumWarps = kNumThreads / 32;

    int E_local = shape.num_experts / num_ranks;
    int receiver_state_bytes = receiver_state_smem_bytes(num_ranks, E_local, shape.num_topk);
#ifndef DISABLE_SM90_FEATURES
    int smem_size = kNumTMABytesPerWarp * kNumWarps + receiver_state_bytes;
#else
    int smem_size = receiver_state_bytes;
#endif

#define DISPATCH_LAUNCH_CASE(ranks)                                                  \
    {                                                                                \
        auto kernel = dispatch_main_kernel<ranks, kNumThreads, kNumTMABytesPerWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                                           \
        LAUNCH_KERNEL(&cfg, kernel,                                                  \
                      pool_out, per_token_out, inputs,                               \
                      tile_signal, shape, env);                                      \
    }                                                                                \
    break

    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    cfg.dynamicSmemBytes = smem_size;
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

// ── Backward: dispatch_grads_main_kernel ─────────────────────────────────
// Ships dL/dy[t] rows from origin → expert ranks using the same routing as
// fwd dispatch. Differences from fwd dispatch_main_kernel:
//   - sender ships data only (no topk_idx / topk_weights / src_idx).
//   - receiver SKIPS Pass A — slots come from recv_token_to_slots[r, :K]
//     (persisted by fwd Pass B). No SMEM batch_slot, no per-(c, src, e) seen
//     counter — Pass 2 reads seen_per_substream[c, src, e] from the metadata
//     kernel's persisted output.
//   - receiver writes ONLY into dL_do_pool[slot] (K-fanout per packet).
//     No scalar pool-slot writes (pool_topk_weight / pool_recv_token /
//     pool_k_slot / k_local_remaining are already populated by fwd).
//   - Pass 2 fires `red.release.gpu.global.add.s32` into
//     bwd_dispatch_arrival_count; the bwd-Y scheduler spins until count ==
//     pool_arrival_target[block] (the same target fwd uses, re-fired on the
//     bwd ready signal).
template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch_grads_main_kernel(
        DispatchGradsIO io,
        DispatchGradsRouting routing,
        DispatchGradsTileSignal tile_signal,
        DispatchGradsShape shape,
        DispatchEnv env) {
    // Mirror of the fwd dispatch_main_kernel entry gate — see comment there.
    if (blockIdx.x == 0 && threadIdx.x == 0 && tile_signal.started_flag != nullptr)
        atomicAdd(tile_signal.started_flag, 1);
    const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const bool is_sender = sm_id % 2 == 0;
    EP_DEVICE_ASSERT(num_sms % 2 == 0);

    const auto num_threads_per_rank = kNumThreads / kNumRanks;
    const auto num_channels = num_sms / 2;
    const auto responsible_rank = thread_id / num_threads_per_rank;
    const auto responsible_channel = sm_id / 2;

    int num_experts_per_rank = shape.num_experts / kNumRanks;
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 and shape.num_topk > 0);
    EP_DEVICE_ASSERT(num_experts_per_rank <= kMaxLocalExpertsPerRank);
    EP_DEVICE_ASSERT(shape.num_topk <= kMaxTopK);

    // Channel buffer pointers — same IPC slab layout as fwd dispatch (both
    // share the dispatch ring). We only touch start_offset / end_offset / head
    // / tail / x_buffers; topk_idx / topk_weights / src_idx buffers exist in
    // the slab but go untouched by bwd.
    auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(env.buffer_ptrs[is_sender ? responsible_rank : env.rank]) +
                                       kNumRanks * kNumRanks * sizeof(int));
    int target_rank = is_sender ? env.rank : responsible_rank;
    auto num_channels_total = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;

    auto channel_start_offset = Buffer<uint64_t>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset = Buffer<uint64_t>(ptr, num_channels_total, channel_rank_offset);
    auto channel_head_idx = Buffer<uint64_t>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx = Buffer<uint64_t>(ptr, num_channels_total, channel_rank_offset);
    // Bwd uses the same dispatch_seq but stamps the LSB phase bit so its
    // writes don't collide with fwd dispatch's at the same dispatch_seq
    // (same as internode dispatch_grads at internode.cu:2380).
    const int64_t nvl_seq = (tile_signal.dispatch_seq << 1) | 1;

    auto channel_x_buffers = Buffer<int4>(
        ptr, num_channels_total * env.num_recv_buffer_tokens * shape.hidden_int4,
        channel_rank_offset * env.num_recv_buffer_tokens * shape.hidden_int4);
    // Skip past topk_idx / topk_weights regions (untouched by bwd but they
    // reserve space in the slab — pointer arithmetic must match fwd).
    auto channel_src_idx_buffers = Buffer<int>(
        ptr, num_channels_total * env.num_recv_buffer_tokens,
        channel_rank_offset * env.num_recv_buffer_tokens);
    (void)channel_src_idx_buffers;
    auto channel_topk_idx_buffers = Buffer<topk_idx_t>(
        ptr, num_channels_total * env.num_recv_buffer_tokens * shape.num_topk,
        channel_rank_offset * env.num_recv_buffer_tokens * shape.num_topk);
    (void)channel_topk_idx_buffers;
    auto channel_topk_weights_buffers = Buffer<float>(
        ptr, num_channels_total * env.num_recv_buffer_tokens * shape.num_topk,
        channel_rank_offset * env.num_recv_buffer_tokens * shape.num_topk);
    (void)channel_topk_weights_buffers;

    // TMA stuffs (mirror fwd dispatch's setup).
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto half_hidden_int4 = shape.hidden_int4 / 2;
    auto half_hidden_bytes = half_hidden_int4 * static_cast<int>(sizeof(int4));
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + half_hidden_bytes);
    uint32_t tma_phase = 0;
    if (elect_one_sync()) {
        mbarrier_init(tma_mbarrier, 1);
        fence_barrier_init();
        EP_DEVICE_ASSERT(shape.hidden_int4 % 2 == 0 and half_hidden_bytes + sizeof(uint64_t) <= kNumTMABytesPerWarp);
    }
    __syncwarp();
#endif

    if (is_sender) {
        constexpr int num_send_warps = kNumThreads / 32;
        constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
        const auto send_thread_id = thread_id;
        const auto send_warp_id_in_rank = send_thread_id % num_threads_per_rank / 32;
        EP_DEVICE_ASSERT(kNumRanks <= kMaxRanks);
        EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

        // Recompute channel_prefix locally (same as fwd dispatch sender).
        // No need to re-write channel_prefix_matrix to global — fwd already
        // populated it; bwd's only consumer here is start/end offset.
        if (send_warp_id_in_rank == 0) {
            int channel_start, channel_end;
            get_channel_task_range(shape.num_tokens, num_channels, responsible_channel, channel_start, channel_end);
            int start_count = 0, end_count = 0;
            for (int i = lane_id; i < channel_end; i += 32) {
                int v = io.is_token_in_rank[i * kNumRanks + responsible_rank];
                if (i < channel_start) start_count += v;
                else end_count += v;
            }
            start_count = warp_reduce_sum(start_count);
            end_count = warp_reduce_sum(end_count) + start_count;

            if (elect_one_sync()) {
                st_relaxed_sys_global(channel_start_offset.buffer(), nvl_pack(nvl_seq, -start_count - 1));
                st_relaxed_sys_global(channel_end_offset.buffer(),   nvl_pack(nvl_seq, -end_count   - 1));
            }
        }
        __syncwarp();

        int token_start_idx, token_end_idx;
        get_channel_task_range(shape.num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

        int cached_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            sender_wait_for_queue_space(cached_channel_tail_idx,
                                        channel_head_idx.buffer(),
                                        nvl_seq,
                                        env.num_recv_buffer_tokens,
                                        env.num_max_send_tokens,
                                        env.rank, responsible_channel,
                                        "dispatch_grads senders");

            int chunk_token_idx = 0;
            while (chunk_token_idx < env.num_max_send_tokens and token_idx < token_end_idx) {
                if (not io.is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                    token_idx++;
                    continue;
                }

                int dst_slot_idx = (cached_channel_tail_idx++) % env.num_recv_buffer_tokens;
                if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
                    auto shifted_channel_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * shape.hidden_int4;
                    auto shifted_dL_dy = io.dL_dy + token_idx * shape.hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, shape.hidden_int4, shifted_channel_x_buffers, shifted_dL_dy, __ldg, st_na_global);
                }

                chunk_token_idx++, token_idx++;
            }

            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), nvl_pack(nvl_seq, cached_channel_tail_idx));
        }
    } else {
        // Receiver: gather K slots per packet from recv_token_to_slots, write
        // dL_do_pool[slot] K times. No Pass A, no SMEM batch_slot.
        constexpr int num_recv_warps = kNumThreads / 32;
        constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
        const auto recv_thread_id = thread_id;
        const auto recv_thread_id_in_rank = recv_thread_id % num_threads_per_rank;
        const auto recv_warp_id_in_rank = recv_thread_id_in_rank / 32;
        const int E = num_experts_per_rank;
        EP_DEVICE_ASSERT(kNumRanks <= kMaxRanks);
        EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);

        // Rank prefix matrix is passed explicitly (NOT read from IPC slab):
        // fwd combine's `encode_combine_heads` zeros the leading bytes of the
        // slab before bwd runs, so the slab copy is gone by the time we need
        // it. Persistent tensor lives on the StreamingHandle.
        const int* rank_prefix_matrix = routing.rank_prefix_matrix;
        int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + env.rank] : 0;

        int total_offset, num_tokens_to_recv;
        if (elect_one_sync()) {
            while (true) {
                uint64_t raw = static_cast<uint64_t>(ld_volatile_global(channel_start_offset.buffer()));
                if (nvl_seq_match(raw, nvl_seq)) { total_offset = nvl_unpack_value(raw); break; }
            }
            while (true) {
                uint64_t raw = static_cast<uint64_t>(ld_volatile_global(channel_end_offset.buffer()));
                if (nvl_seq_match(raw, nvl_seq)) { num_tokens_to_recv = nvl_unpack_value(raw); break; }
            }
            total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
            num_tokens_to_recv -= total_offset;
        }
        total_offset = __shfl_sync(0xffffffff, total_offset, 0);
        total_offset += rank_offset;
        num_tokens_to_recv = __shfl_sync(0xffffffff, num_tokens_to_recv, 0);

        __shared__ volatile int shared_channel_tail_idx[kNumRanks];

        const int substream_csrc = responsible_channel * kNumRanks + responsible_rank;
        const int* base_pool_substream = routing.base_pool + substream_csrc * E;
        const int* seen_substream = routing.seen_per_substream + substream_csrc * E;

        auto start_time = clock64();
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            while (recv_thread_id_in_rank == 0) {
                uint64_t raw = ld_acquire_sys_global(channel_tail_idx.buffer());
                cached_channel_tail_idx = nvl_seq_match(raw, nvl_seq) ? nvl_unpack_value(raw) : 0;
                if (cached_channel_head_idx != cached_channel_tail_idx) {
                    shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
                    break;
                }
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP timeout for dispatch_grads receivers, rank %d, responsible_channel = %d, tokens remained: %d\n",
                           env.rank, responsible_channel, num_tokens_to_recv);
                    trap();
                }
            }

            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];
            int batch_total = cached_channel_tail_idx - cached_channel_head_idx;

            // Direct per-(chunk) processing — no inner sub-batch loop bounded
            // by SMEM batch_slot (Pass A is gone). Each warp owns chunks where
            // chunk_idx % num_recv_warps_per_rank == warp_id.
            for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < batch_total;
                 chunk_idx += num_recv_warps_per_rank) {
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % env.num_recv_buffer_tokens;
                int recv_token_id = total_offset + chunk_idx;

                auto shifted_buffer_x_int4 = channel_x_buffers.buffer() + token_idx_in_buffer * shape.hidden_int4;
#ifndef DISABLE_SM90_FEATURES
                #pragma unroll
                for (int piece = 0; piece < 2; ++piece) {
                    tma_store_wait<0>();
                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer,
                                    shifted_buffer_x_int4 + piece * half_hidden_int4,
                                    tma_mbarrier, half_hidden_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, half_hidden_bytes);
                        mbarrier_wait(tma_mbarrier, tma_phase);
                        for (int k = 0; k < shape.num_topk; ++k) {
                            int slot = routing.recv_token_to_slots[recv_token_id * shape.num_topk + k];
                            if (slot < 0) continue;
                            tma_store_1d(tma_buffer,
                                         io.dL_do_pool + static_cast<int64_t>(slot) * shape.hidden_int4
                                              + piece * half_hidden_int4,
                                         half_hidden_bytes, false);
                        }
                    }
                    __syncwarp();
                }
#else
                for (int k = 0; k < shape.num_topk; ++k) {
                    int slot = routing.recv_token_to_slots[recv_token_id * shape.num_topk + k];
                    if (slot < 0) continue;
                    auto shifted_pool_int4 = io.dL_do_pool + static_cast<int64_t>(slot) * shape.hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, shape.hidden_int4, shifted_pool_int4, shifted_buffer_x_int4,
                                       ld_nc_global, st_na_global);
                }
#endif
            }

            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            cached_channel_head_idx += batch_total;
            total_offset += batch_total;
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 and elect_one_sync())
                st_relaxed_sys_global(channel_head_idx.buffer(), nvl_pack(nvl_seq, cached_channel_head_idx));

            num_tokens_to_recv -= batch_total;
        }

        // Pass 2 — same shape as fwd dispatch's Pass 2, but reads
        // `seen_per_substream` from global (instead of fwd's SMEM
        // `seen_substream`) and fires `bwd_dispatch_arrival_count` instead of
        // fwd's `pool_arrival_count`.
#ifndef DISABLE_SM90_FEATURES
        tma_store_wait<0>();
#endif
        asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
        // Device-scope fence: dL_do_pool is local and consumed by
        // kernel_y_bwd on the same GPU. See the matching comment in
        // ``dispatch_main_kernel`` above for the full rationale.
        __threadfence();
        asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
        if (recv_thread_id_in_rank == 0) {
            for (int e = 0; e < E; ++e) {
                int n_writes_for_e = seen_substream[e];
                if (n_writes_for_e == 0) continue;
                fire_pool_blocks(base_pool_substream[e], n_writes_for_e, shape.tile_m,
                                 tile_signal.bwd_dispatch_arrival_count);
            }
        }
    }
}

void launch_dispatch_grads_main(const DispatchGradsIO& io,
                                const DispatchGradsRouting& routing,
                                const DispatchGradsTileSignal& tile_signal,
                                const DispatchGradsShape& shape,
                                const DispatchEnv& env,
                                int num_ranks,
                                cudaStream_t stream,
                                int num_sms) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 8192;
    constexpr int kNumWarps = kNumThreads / 32;
#ifndef DISABLE_SM90_FEATURES
    int smem_size = kNumTMABytesPerWarp * kNumWarps;
#else
    int smem_size = 0;
#endif

#define DISPATCH_GRADS_LAUNCH_CASE(ranks)                                                  \
    {                                                                                      \
        auto kernel = dispatch_grads_main_kernel<ranks, kNumThreads, kNumTMABytesPerWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                                                 \
        LAUNCH_KERNEL(&cfg, kernel,                                                        \
                      io, routing, tile_signal, shape, env);                               \
    }                                                                                      \
    break

    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    cfg.dynamicSmemBytes = smem_size;
    SWITCH_RANKS(DISPATCH_GRADS_LAUNCH_CASE);
#undef DISPATCH_GRADS_LAUNCH_CASE
}

// `encode_combine_heads_kernel` was deleted as part of the
// disjoint-regions + genstamps refactor. Its two former jobs are now
// handled elsewhere:
//   - per-channel reverse-scan of `send_head`: fused into
//     `dispatch_main_kernel`'s sender block tail.
//   - bracketed slab head/tail memset: no longer needed — combine's
//     (head_idx, tail_idx) live in a physically disjoint region from
//     dispatch's (start_offset, end_offset) (see api.cuh
//     `intranode_get_dispatch_section_bytes`), and the meta slots are
//     64-bit genstamped so iter-to-iter staleness is filtered by seq
//     mismatch without needing a memset.

// Combine kernel — used by BOTH forward combine and backward combine_grads.
// The per-direction differences are all in args:
//
//   ┌─────────────────────────┬────────────────── fwd ──────────────────┬────────────────── bwd ─────────────────┐
//   │ recv_x                  │ out[num_tokens, H] bf16                  │ dL/dx[num_tokens, H] bf16              │
//   │ recv_topk_weights_out   │ recv_topk_weights[num_tokens, K] fp32    │ dL/dtopk_weights[num_tokens, K] fp32   │
//   │ x                       │ handle.o[T_recv, H]                      │ dL/dx_per_r[T_recv, H]                 │
//   │ per_slot_weights        │ pool_topk_weight[TK_padded] fp32         │ weight_grads[TK_padded] fp32           │
//   │ recv_token_to_slots     │ same — populated by fwd dispatch Pass B  │ same                                   │
//   │ y_done_per_token  │ kernel_y forward release stamp           │ kernel_a_bwd release stamp             │
//   │ combine_seq             │ dispatch_seq (fwd's value)               │ dispatch_seq (bwd uses same int)       │
//   └─────────────────────────┴──────────────────────────────────────────┴────────────────────────────────────────┘
//
// The wire format and reduction logic are direction-independent: each sender
// ships one packet per recv-token (bf16 H values + num_topk fp32 weights with
// 0 in non-local slots); the receiver gathers K_dst_ranks packets per source
// token and sums both halves. For bwd weight-grads, only one sender's packet
// has the actual non-zero weight for each (t, k) — the sum reduces correctly
// since the rest are zero.
template <typename dtype_t, int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp,
          bool kSendTopkWeights>
__global__ void __launch_bounds__(kNumThreads, 1) combine_main_kernel(dtype_t* recv_x,
                                                          float* recv_topk_weights_out,
                                                          const dtype_t* x,
                                                          const float* per_slot_weights,
                                                          const int* recv_token_to_slots,
                                                          const int* rank_prefix_matrix,
                                                          const int* channel_prefix_matrix,
                                                          int* send_head,
                                                          const int64_t* y_done_per_token,
                                                          int64_t combine_seq,
                                                          int combine_phase,
                                                          int num_tokens,
                                                          int num_recv_tokens,
                                                          int hidden,
                                                          int num_topk,
                                                          void** buffer_ptrs,
                                                          // Byte offset on each peer's `buffer_ptrs[i]` past dispatch's
                                                          // sub-buffer chain (see api.cuh `intranode_get_dispatch_section_bytes`).
                                                          // Combine's meta + data buffers live at
                                                          // `buffer_ptrs[i] + dispatch_section_bytes`; physically disjoint from
                                                          // dispatch's chain at `buffer_ptrs[i] + 0`. Removes the prior
                                                          // head/tail alias with dispatch's start/end_offset that required
                                                          // the `encode_combine_heads` memset.
                                                          int64_t dispatch_section_bytes,
                                                          int rank,
                                                          int num_max_send_tokens,
                                                          int num_recv_buffer_tokens) {
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto sm_id = static_cast<int>(blockIdx.x), lane_id = get_lane_id();
    const auto num_channels = num_sms / 2;
    const bool is_sender = sm_id % 2 == 0;
    const int responsible_channel = sm_id / 2;
    // Genstamp seq for combine's meta slots. Fwd combine and bwd
    // combine_grads share the same `combine_seq` (= handle.dispatch_seq)
    // but stamp the LSB phase bit so they don't collide at the same
    // physical slot. Mirrors internode combine_main_kernel:3220.
    const int64_t nvl_seq = (combine_seq << 1) | (combine_phase & 1);
    EP_DEVICE_ASSERT(num_topk <= kMaxTopK);

    constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
    int hidden_int4 = hidden * sizeof(dtype_t) / sizeof(int4);
    int hidden_int4_aligned = align_down(hidden_int4, 32);
    auto x_int4 = reinterpret_cast<const int4*>(x);
    auto recv_int4 = reinterpret_cast<int4*>(recv_x);

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
#endif

    if (is_sender) {
        // Workers for sending
        // Several warps are responsible for a single rank
        constexpr int num_send_warps_per_rank = (kNumThreads / 32) / kNumRanks;
        constexpr int num_send_warps = num_send_warps_per_rank * kNumRanks;
        const auto num_threads_per_rank = num_send_warps_per_rank * 32;
        const auto send_thread_id = thread_id;
        const auto send_warp_id = send_thread_id / 32;
        const auto send_rank_id = (responsible_channel + send_warp_id) % kNumRanks;
        const auto send_warp_id_in_rank = send_warp_id / kNumRanks;
        EP_STATIC_ASSERT(num_send_warps * 32 == kNumThreads, "Invalid warp count");

        // Calculate pointers by the specific layout. Base offset by
        // `dispatch_section_bytes` lands us in combine's disjoint
        // sub-buffer chain (see api.cuh `intranode_get_dispatch_section_bytes`).
        auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[send_rank_id]) +
                                           dispatch_section_bytes);
        auto num_channels_total = num_channels * kNumRanks;
        auto channel_rank_offset = responsible_channel * kNumRanks + rank;

        // Channel meta data — combine's own (head_idx, tail_idx) as
        // genstamped uint64.
        // `head_idx`: kNumChannels * kNumRanks * sizeof(uint64_t)
        // `tail_idx`: kNumChannels * kNumRanks * sizeof(uint64_t)
        // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
        // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)   ← skipped (kept for layout parity with dispatch section)
        // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
        auto channel_head_idx = Buffer<uint64_t>(ptr, num_channels_total, channel_rank_offset);
        auto channel_tail_idx = Buffer<uint64_t>(ptr, num_channels_total, channel_rank_offset);
        auto channel_x_buffers = Buffer<int4>(
            ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
        ptr = reinterpret_cast<void*>(static_cast<int8_t*>(ptr) +
                                      num_channels_total * num_recv_buffer_tokens * sizeof(int));
        auto channel_topk_weights_buffers = Buffer<float>(
            ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);

        // Get tasks
        // NOTES: `channel_offset` is already shifted
        int rank_offset = send_rank_id > 0 ? rank_prefix_matrix[(send_rank_id - 1) * kNumRanks + rank] : 0;
        int num_rank_tokens = rank_prefix_matrix[send_rank_id * kNumRanks + rank] - rank_offset;
        int channel_offset = channel_prefix_matrix[send_rank_id * num_channels + responsible_channel];
        int num_channel_tokens =
            (responsible_channel == num_channels - 1 ? num_rank_tokens
                                                     : channel_prefix_matrix[send_rank_id * num_channels + responsible_channel + 1]) -
            channel_offset;
        int token_start_idx = rank_offset + channel_offset, token_end_idx = rank_offset + channel_offset + num_channel_tokens;

        // Iterate over all tokens and send by chunks
        int current_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            int num_round_tokens = min(num_max_send_tokens, token_end_idx - static_cast<int>(token_idx));
            sender_wait_for_queue_space(current_channel_tail_idx,
                                        channel_head_idx.buffer(),
                                        nvl_seq,
                                        num_recv_buffer_tokens,
                                        num_round_tokens,
                                        rank, responsible_channel,
                                        "combine senders");

            // Send by chunk
            #pragma unroll
            for (int i = send_warp_id_in_rank; i < num_round_tokens; i += num_send_warps_per_rank) {
                // Phase-D per-token gate: kernel Y release-stores `combine_seq` into
                // y_done_per_token[my_token] once k_local_remaining[my_token]
                // hits zero (all K_local(my_token) contributions to o[my_token] have
                // landed). Spin until the gate clears, then issue the warp-cooperative
                // copy below — the acquire pairs with kernel Y's `.gpu`-scope release
                // (intra-GPU: kernel Y and combine sender on the same device).
                // Kernel Y's `threadfence_system` before the release carries
                // o[my_token]'s writes to system scope for the downstream NVL send.
                auto my_token = token_idx + i;
                if (elect_one_sync()) {
                    auto gate_start = clock64();
                    while (ld_acquire_gpu_global(&y_done_per_token[my_token]) < combine_seq) {
                        if (clock64() - gate_start > NUM_TIMEOUT_CYCLES) {
                            printf("DeepEP timeout for combine sender gate, rank %d, channel %d, token %d\n",
                                   rank, responsible_channel, static_cast<int>(my_token));
                            trap();
                        }
                    }
                }
                __syncwarp();

                // Get an empty slot
                int dst_slot_idx = (current_channel_tail_idx + i) % num_recv_buffer_tokens;

                // Copy data
                auto shifted_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                auto shifted_x = x_int4 + my_token * hidden_int4;
                UNROLLED_WARP_COPY(4, lane_id, hidden_int4, shifted_x_buffers, shifted_x, ld_nc_global, st_na_global);

                // Send the per-(my_token, k) weight via slot lookup. Bwd
                // only: weight_grads[slot] = dL/dweight at that (r, k_local).
                // For non-local k (slot == -1) we ship 0 — receiver's K-way
                // sum then yields the correct (t, k) value, since exactly
                // one sender has the non-zero contribution per (t, k). Fwd
                // combine drops this payload entirely (kernel Y already
                // pre-multiplies pool_topk_weight per row).
                if constexpr (kSendTopkWeights) {
                    if (num_topk > 0 and lane_id < num_topk) {
                        int slot = recv_token_to_slots[my_token * num_topk + lane_id];
                        float w = (slot >= 0) ? __ldg(per_slot_weights + slot) : 0.0f;
                        channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = w;
                    }
                }
            }
            token_idx += num_round_tokens;
            current_channel_tail_idx += num_round_tokens;

            // Move tail index
            asm volatile("bar.sync %0, %1;" ::"r"(send_rank_id), "r"(num_threads_per_rank));
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), nvl_pack(nvl_seq, current_channel_tail_idx));
        }
    } else {
        // Workers for receiving
        // One warp for moving the queue head, others for reduction
        constexpr int num_recv_warps = kNumThreads / 32;
        const auto recv_warp_id = thread_id / 32;
        EP_DEVICE_ASSERT(kNumRanks <= kMaxRanks and kNumThreads > 32);
        EP_DEVICE_ASSERT(thread_id >= 0 and kNumThreads % 32 == 0);

        // Shared head, tail and retired flags for receiver warps
        __shared__ volatile int warp_channel_head_idx[num_recv_warps][kNumRanks];
        __shared__ volatile int channel_tail_idx[kNumRanks];
        __shared__ volatile bool warp_retired[num_recv_warps];
        if (thread_id < num_recv_warps)
            warp_retired[thread_id] = false;
        if (lane_id < kNumRanks)
            warp_channel_head_idx[recv_warp_id][lane_id] = 0;
        if (thread_id < kNumRanks)
            channel_tail_idx[thread_id] = 0;
        asm volatile("bar.sync 0, %0;" ::"r"(kNumThreads));

        if (thread_id < 32) {
            // Combine's (head, tail) slots live in combine's own disjoint
            // sub-buffer chain at `buffer_ptrs[rank] + dispatch_section_bytes`.
            // They're 64-bit genstamped; reader filters by `nvl_seq` match.
            uint64_t* channel_head_idx_ptr = reinterpret_cast<uint64_t*>(
                static_cast<int8_t*>(buffer_ptrs[rank]) + dispatch_section_bytes)
                + responsible_channel * kNumRanks + lane_id;
            uint64_t* channel_tail_idx_ptr = channel_head_idx_ptr + num_channels * kNumRanks;

            // Queue head updater
            int last_head = 0;
            while (lane_id < kNumRanks) {
                // Check retired
                bool retired = true;
                #pragma unroll
                for (int i = 1; i < num_recv_warps; ++i)
                    retired = retired and warp_retired[i];
                if (retired)
                    break;

                // Update queue tail. Stale-seq → treat tail as 0 (i.e.,
                // sender hasn't published this iter yet).
                uint64_t raw_tail = ld_acquire_sys_global(channel_tail_idx_ptr);
                channel_tail_idx[lane_id] = nvl_seq_match(raw_tail, nvl_seq) ? nvl_unpack_value(raw_tail) : 0;

                // Update minimum head
                int min_head = std::numeric_limits<int>::max();
                #pragma unroll
                for (int i = 1; i < num_recv_warps; ++i)
                    if (not warp_retired[i])
                        min_head = min(min_head, warp_channel_head_idx[i][lane_id]);
                if (min_head != std::numeric_limits<int>::max() and min_head > last_head)
                    st_relaxed_sys_global(channel_head_idx_ptr, nvl_pack(nvl_seq, last_head = min_head));
            }
        } else {
            // Receivers
            // Channel metadata
            // All lanes will use data buffer, but only rank lane will use `head/tail/src_idx`
            Buffer<int4> channel_x_buffers[kNumRanks];
            Buffer<float> channel_topk_weights_buffers[kNumRanks];

            // Calculate pointers by the specific layout
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++i) {
                auto channel_rank_offset = responsible_channel * kNumRanks + i;
                auto num_channels_total = num_channels * kNumRanks;
                // Combine sub-buffer base: skip past dispatch's section and
                // combine's own (head_idx, tail_idx) genstamped uint64
                // meta to land on `channel_x_buffers`.
                auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[rank]) +
                                                   dispatch_section_bytes +
                                                   2 * num_channels * kNumRanks * sizeof(uint64_t));

                // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
                channel_x_buffers[i] = Buffer<int4>(ptr,
                                                    num_channels_total * num_recv_buffer_tokens * hidden_int4,
                                                    channel_rank_offset * num_recv_buffer_tokens * hidden_int4);

                // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
                ptr = reinterpret_cast<void*>(static_cast<int8_t*>(ptr) + num_channels_total * num_recv_buffer_tokens * sizeof(int));

                // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
                channel_topk_weights_buffers[i] = Buffer<float>(
                    ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
            }

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_recv_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

            // Iterate over all tokens and combine
            for (int64_t token_idx = token_start_idx + recv_warp_id - 1; token_idx < token_end_idx; token_idx += num_recv_warps - 1) {
                // Read expected head
                int expected_head = -1;
                if (lane_id < kNumRanks)
                    expected_head = ld_nc_global(send_head + token_idx * kNumRanks + lane_id);

                auto start_time = clock64();
                while (__any_sync(0xffffffff, channel_tail_idx[lane_id] <= expected_head and expected_head >= 0)) {
                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for combine receivers, rank %d, responsible_channel = %d, expect = %d\n",
                               rank,
                               responsible_channel,
                               expected_head);
                        trap();
                    }
                }
                __syncwarp();

                // Broadcast current heads
                int num_topk_ranks = 0, topk_ranks[kNumRanks], slot_indices[kNumRanks];
                #pragma unroll
                for (int i = 0; i < kNumRanks; ++i) {
                    auto expected_head_i = __shfl_sync(0xffffffff, expected_head, i);
                    if (expected_head_i >= 0) {
                        slot_indices[num_topk_ranks] = expected_head_i % num_recv_buffer_tokens;
                        topk_ranks[num_topk_ranks++] = i;
                    }
                }

                // Wait shared memory release
#ifndef DISABLE_SM90_FEATURES
                tma_store_wait<0>();
                __syncwarp();
#endif

                // Reduce data with pipeline
                constexpr int kNumStages = 8;
                EP_STATIC_ASSERT(kNumStages * 32 * sizeof(int4) <= kNumTMABytesPerWarp, "Invalid count");
                #pragma unroll
                for (int i = lane_id; i < hidden_int4; i += 32) {
                    // Read buffers
                    int4 recv_value_int4[kNumRanks];
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++j)
                        recv_value_int4[j] = ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() + slot_indices[j] * hidden_int4 + i);

                    // Reduce all-to-all results into fp32 accumulator.
                    float values[kDtypePerInt4] = {};
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++j) {
                        auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
                        #pragma unroll
                        for (int k = 0; k < kDtypePerInt4; ++k)
                            values[k] += static_cast<float>(recv_value_dtypes[k]);
                    }

                    // Cast back to `dtype_t`
                    int4 out_int4;
                    auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
                    #pragma unroll
                    for (int j = 0; j < kDtypePerInt4; ++j)
                        out_dtypes[j] = static_cast<dtype_t>(values[j]);

#ifndef DISABLE_SM90_FEATURES
                    if (i < hidden_int4_aligned) {
                        // Wait TMA arrival
                        tma_store_wait<kNumStages - 1>();
                        __syncwarp();

                        // Write into TMA buffer
                        auto tma_stage_idx = (i / 32) % kNumStages;
                        reinterpret_cast<int4*>(tma_buffer)[tma_stage_idx * 32 + lane_id] = out_int4;

                        // Issue TMA
                        tma_store_fence();
                        __syncwarp();
                        if (elect_one_sync()) {
                            auto tma_bytes = min(32, hidden_int4 - i) * static_cast<int>(sizeof(int4));
                            tma_store_1d(reinterpret_cast<int4*>(tma_buffer) + tma_stage_idx * 32,
                                         recv_int4 + token_idx * hidden_int4 + i,
                                         tma_bytes,
                                         false);
                        }
                        __syncwarp();
                    } else {
#endif
                        recv_int4[token_idx * hidden_int4 + i] = out_int4;
#ifndef DISABLE_SM90_FEATURES
                    }
#endif
                }

                // Reduce `topk_weights` — bwd combine_grads only. Sink is
                // dL/dtopk_weights[t, k]. Fwd combine omits the K-weight
                // wire payload + this reduce (see header `kSendTopkWeights`).
                if constexpr (kSendTopkWeights) {
                    if (lane_id < num_topk) {
                        float value = 0;
                        #pragma unroll
                        for (int i = 0; i < num_topk_ranks; ++i)
                            value += ld_nc_global(channel_topk_weights_buffers[topk_ranks[i]].buffer() + slot_indices[i] * num_topk + lane_id);
                        recv_topk_weights_out[token_idx * num_topk + lane_id] = value;
                    }
                }

                // Update head
                if (lane_id < kNumRanks)
                    warp_channel_head_idx[recv_warp_id][lane_id] = (expected_head < 0) ? -expected_head - 1 : expected_head + 1;
            }

            // Retired
            __syncwarp();
            if (elect_one_sync())
                warp_retired[recv_warp_id] = true;
        }
    }
}

void launch_combine_main(cudaDataType_t type,
             void* recv_x,
             float* recv_topk_weights_out,
             const void* x,
             const float* per_slot_weights,
             const int* recv_token_to_slots,
             const int* rank_prefix_matrix,
             const int* channel_prefix_matrix,
             int* send_head,
             const int64_t* y_done_per_token,
             int64_t combine_seq,
             bool is_fwd,
             int num_tokens,
             int num_recv_tokens,
             int hidden,
             int num_topk,
             void** buffer_ptrs,
             int64_t dispatch_section_bytes,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_sms,
             int num_max_send_tokens,
             int num_recv_buffer_tokens) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 4096;
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
#endif

    // Fwd combine and bwd combine_grads share `combine_seq` (= handle.dispatch_seq);
    // the LSB phase bit distinguishes which direction stamped a given slot.
    const int combine_phase = is_fwd ? 0 : 1;

#define COMBINE_LAUNCH_CASE_IMPL(dtype, ranks, kSendTopkWeights)                                                  \
    {                                                                                                             \
        auto kernel = combine_main_kernel<dtype, ranks, kNumThreads, kNumTMABytesPerWarp, kSendTopkWeights>;      \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                                     \
        LAUNCH_KERNEL(&cfg,                                                    \
                      kernel,                                                  \
                      reinterpret_cast<dtype*>(recv_x),                        \
                      recv_topk_weights_out,                                   \
                      reinterpret_cast<const dtype*>(x),                       \
                      per_slot_weights,                                        \
                      recv_token_to_slots,                                     \
                      rank_prefix_matrix,                                      \
                      channel_prefix_matrix,                                   \
                      send_head,                                               \
                      y_done_per_token,                                        \
                      combine_seq,                                             \
                      combine_phase,                                           \
                      num_tokens,                                              \
                      num_recv_tokens,                                         \
                      hidden,                                                  \
                      num_topk,                                                \
                      buffer_ptrs,                                             \
                      dispatch_section_bytes,                                  \
                      rank,                                                    \
                      num_max_send_tokens,                                     \
                      num_recv_buffer_tokens);                                 \
    }
#define COMBINE_LAUNCH_CASE(dtype, ranks)                                          \
    {                                                                              \
        if (is_fwd) {                                                              \
            COMBINE_LAUNCH_CASE_IMPL(dtype, ranks, false)                          \
        } else {                                                                   \
            COMBINE_LAUNCH_CASE_IMPL(dtype, ranks, true)                           \
        }                                                                          \
    }                                                                              \
    break
#define COMBINE_DTYPE_LAUNCH_CASE(dtype)                 \
    SWITCH_RANKS_WITH_DTYPE(dtype, COMBINE_LAUNCH_CASE); \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving
    EP_HOST_ASSERT(num_sms % 2 == 0);
    EP_HOST_ASSERT(kNumThreads >= num_ranks * 32);
    // No PDL predecessor — encode_combine_heads was deleted as part of the
    // disjoint-regions + genstamps refactor.
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_TYPES(COMBINE_DTYPE_LAUNCH_CASE);
#undef COMBINE_DTYPE_LAUNCH_CASE
#undef COMBINE_LAUNCH_CASE
#undef COMBINE_LAUNCH_CASE_IMPL
}

}  // namespace intranode

}  // namespace stream_ep
