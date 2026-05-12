#include "api.cuh"
#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

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
                                                            int* channel_head_idx,
                                                            int num_recv_buffer_tokens,
                                                            int num_required_free,
                                                            int rank,
                                                            int responsible_channel,
                                                            const char* role) {
    auto start_time = clock64();
    if (elect_one_sync()) {
        while (true) {
            // We only consider the worst case (cached_tail - head); counting
            // the actual freed-but-not-yet-published slots is not worth it.
            int num_used_slots = cached_tail_idx - ld_volatile_global(channel_head_idx);
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


// Streaming-MoE consolidated dispatch metadata. Single-block kernel that does
// the cross-rank (token, k) count exchange and emits the pool-shape outputs the
// dispatch hot path consumes (everything sized by E_local + R + (c, src, e),
// known before the host poll).
//
// IPC slab at `buffer_ptrs[R] + streaming_section_offset` carries two adjacent
// inboxes (zeroed on Buffer construction; cross-rank stores via per-(c, src)
// unique slots — no atomicAdds across senders):
//   - e_inbox[c, src, e]: per-(channel, src, local_expert) (token, k) count
//   - u_inbox[c, src]:    per-(channel, src) UNIQUE token count
//
// Phases:
//   1. Zero local SMEM histograms (per-(dst, c, e) and per-(dst, c)).
//   2. barrier_block — peers' kernels are running, IPC slabs ready to write.
//   3. Local SMEM histograms by scanning topk_idx (with dst_mask to dedupe k
//      slots that share the same dst).
//   4. Bulk store local histograms to peers' IPC inboxes — each (sender, dst)
//      pair writes a unique region (no contention, no atomics).
//   5. barrier_block — peers' phase-4 stores observable.
//   6. Read own inbox, derive pool-shape outputs:
//        expert_frequency[E_local], expert_pool_block_offset[E_local + 1],
//        base_pool[c, src, e_local], rank_prefix_matrix[R, R] (this rank's column),
//        total_tiles (host + device), num_recv (host),
//        num_recv_per_expert[E_local] (host).
//   7. base_pool fill: per-expert serial accumulation over (c, src) lex order.
//   8. Per-tile arrays: `tile_id_to_expert[total_tiles]` and
//      `pool_arrival_target[total_tiles]`. Pre-allocated at `total_tiles_max`
//      by the host (sized by num_tokens × num_topk × num_ranks / tile_m + E,
//      ~8 KB at production); only the prefix [0, total_tiles) is written.
//
// Pool layout: each expert's region in `pool` starts at
// `expert_pool_block_offset[e] * BLOCK_M` (BLOCK_M = tile_m) and is padded up to
// a BLOCK_M multiple. base_pool[c, src, e] is the substream's first slot for
// expert e (deterministic given cached routing).
template <int kNumRanks>
__global__ void streaming_dispatch_metadata_kernel(
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
    auto thread_id = static_cast<int>(threadIdx.x);
    auto num_threads = static_cast<int>(blockDim.x);
    auto warp_id = thread_id / 32;
    auto lane_id = thread_id % 32;
    auto num_warps = num_threads / 32;

    const int E = num_experts_per_rank;
    const int e_inbox_size = num_channels * kNumRanks * E;
    const int slab_e_size = num_channels * E;       // per-dst slice of smem_local_e
    const int slab_u_size = num_channels;            // per-dst slice of smem_local_u

    extern __shared__ int smem[];
    int* smem_local_e = smem;                                          // [kNumRanks * slab_e_size]
    int* smem_local_u = smem_local_e + kNumRanks * slab_e_size;             // [kNumRanks * slab_u_size]

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

    // Phase 5: cross-rank barrier — peers' phase-4 stores now observable.
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

    // Phase 6: read own inbox, derive metadata. Reuse the (now-stale) local SMEM.
    auto* my_e_inbox = reinterpret_cast<int*>(
        static_cast<uint8_t*>(buffer_ptrs[rank]) + streaming_section_offset);
    auto* my_u_inbox = my_e_inbox + e_inbox_size;

    int* smem_freq     = smem;                       // [E]
    int* smem_pool_blk = smem_freq + E;                 // [E + 1]
    int* smem_per_src  = smem_pool_blk + (E + 1);       // [kNumRanks]

    // expert_frequency[e] = sum over (c, src) of e_inbox[c, src, e].
    for (int e = thread_id; e < E; e += num_threads) {
        int sum = 0;
        for (int cs = 0; cs < num_channels * kNumRanks; ++cs)
            sum += my_e_inbox[cs * E + e];
        smem_freq[e] = sum;
        expert_frequency[e] = sum;
    }
    // smem_per_src[src] = sum over c of u_inbox[c, src].
    for (int src = thread_id; src < kNumRanks; src += num_threads) {
        int total = 0;
        for (int c = 0; c < num_channels; ++c)
            total += my_u_inbox[c * kNumRanks + src];
        smem_per_src[src] = total;
    }
    __syncthreads();

    if (thread_id == 0) {
        int cum_blocks = 0;
        smem_pool_blk[0] = 0;
        for (int e = 0; e < E; ++e) {
            int n_blocks_e = (smem_freq[e] + tile_m - 1) / tile_m;
            cum_blocks += n_blocks_e;
            smem_pool_blk[e + 1] = cum_blocks;
        }
        *total_tiles_out = cum_blocks;
        *total_tiles_mapped = cum_blocks;

        int total_unique = 0;
        for (int i = 0; i < kNumRanks; ++i)
            total_unique += smem_per_src[i];
        *num_recv_mapped = total_unique;

        for (int e = 0; e < E; ++e) {
            int aligned = (smem_freq[e] + expert_alignment - 1) / expert_alignment * expert_alignment;
            num_recv_per_expert_mapped[e] = aligned;
        }

        // rank_prefix_matrix: this rank fills its own column. Cumulative unique
        // tokens from senders 0..i to this rank — read by combine on rank j.
        int cum_src = 0;
        for (int i = 0; i < kNumRanks; ++i) {
            cum_src += smem_per_src[i];
            rank_prefix_matrix[i * kNumRanks + rank] = cum_src;
        }
    }
    __syncthreads();
    for (int e = thread_id; e < E + 1; e += num_threads)
        expert_pool_block_offset[e] = smem_pool_blk[e];

    // Phase 7: base_pool[c, src, e] = pool slot start (in pool-row units) for
    // substream (c, src) writes for expert e:
    //   base_pool[c, src, e] = smem_pool_blk[e] * tile_m
    //                          + Σ over (c', src') < (c, src) lex of e_inbox[c', src', e].
    // E ≤ NUM_MAX_LOCAL_EXPERTS so we can run one thread per expert in parallel.
    //
    // Also persist seen_per_substream[c, src, e] = my_e_inbox[c, src, e]
    // (the per-substream-per-expert recv count) for the backward path. The
    // bwd dispatch_grads receiver doesn't run Pass A — it gathers slots from
    // recv_token_to_slots — so it can't reconstruct fwd's SMEM seen_substream
    // counter. Persisting it here lets bwd Pass 2 reuse fwd's per-block atomic
    // accounting verbatim: walk experts, atomic-add seen_per_substream[cs, e]
    // worth of writes, fire bwd_y_ready when count == tile_m.
    // Cost: ~17 KB int32 at production (66 channels × 8 ranks × 8 experts).
    for (int e = thread_id; e < E; e += num_threads) {
        int acc = smem_pool_blk[e] * tile_m;
        for (int cs = 0; cs < num_channels * kNumRanks; ++cs) {
            base_pool[cs * E + e] = acc;
            int n = my_e_inbox[cs * E + e];
            seen_per_substream[cs * E + e] = n;
            acc += n;
        }
    }

    // Phase 8: per-tile arrays for the dispatch hot path. Each thread owns one
    // expert and walks its tile range [smem_pool_blk[e], smem_pool_blk[e+1]) writing
    //   tile_id_to_expert[tile_id]   = e
    //   pool_arrival_target[tile_id] = tile_m for full tiles,
    //                                  smem_freq[e] - tile_in_e * tile_m for the
    //                                  last (possibly partial) tile per expert.
    // No sync required vs. phase 7: writes target disjoint global regions and
    // smem_pool_blk / smem_freq remain valid in SMEM since phase 6 (the only writers).
    for (int e = thread_id; e < E; e += num_threads) {
        int e_start = smem_pool_blk[e];
        int e_end = smem_pool_blk[e + 1];
        int n_tiles = e_end - e_start;
        int n_e = smem_freq[e];
        for (int t = 0; t < n_tiles; ++t) {
            int tile_id = e_start + t;
            tile_id_to_expert[tile_id] = e;
            pool_arrival_target[tile_id] = (t == n_tiles - 1) ? (n_e - t * tile_m) : tile_m;
        }
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
#define STREAMING_DISPATCH_METADATA_LAUNCH_CASE(ranks)                                       \
    EP_HOST_ASSERT(cudaFuncSetAttribute(                                                     \
                       streaming_dispatch_metadata_kernel<ranks>,                            \
                       cudaFuncAttributeMaxDynamicSharedMemorySize,                          \
                       smem_bytes) == cudaSuccess);                                          \
    LAUNCH_KERNEL(&cfg,                                                                      \
                  streaming_dispatch_metadata_kernel<ranks>,                                 \
                  topk_idx,                                                                  \
                  expert_frequency,                                                          \
                  expert_pool_block_offset,                                                  \
                  base_pool,                                                                 \
                  seen_per_substream,                                                        \
                  rank_prefix_matrix,                                                        \
                  tile_id_to_expert,                                                         \
                  pool_arrival_target,                                                       \
                  total_tiles_out,                                                           \
                  num_recv_mapped,                                                           \
                  num_recv_per_expert_mapped,                                                \
                  total_tiles_mapped,                                                        \
                  num_tokens,                                                                \
                  num_topk,                                                                  \
                  num_experts_per_rank,                                                      \
                  num_channels,                                                              \
                  streaming_section_offset,                                                  \
                  buffer_ptrs,                                                               \
                  barrier_signal_ptrs,                                                       \
                  rank,                                                                      \
                  tile_m,                                                                    \
                  expert_alignment);                                                         \
    break

    SETUP_LAUNCH_CONFIG(1, 256, stream);
    // SMEM: phase-3 histograms dominate (kNumRanks × num_channels × (E + 1) ints).
    // At production E=384, R=8, num_sms=80 this is ~62KB — above H100's default
    // 48KB dynamic-SMEM cap, so we must opt up via cudaFuncSetAttribute before
    // the launch (otherwise the launch fails with the misleading
    // cudaErrorCooperativeLaunchTooLarge: "too many blocks in cooperative launch").
    int smem_bytes = num_ranks * num_channels * (num_experts_per_rank + 1) * sizeof(int);
    cfg.dynamicSmemBytes = smem_bytes;
    SWITCH_RANKS(STREAMING_DISPATCH_METADATA_LAUNCH_CASE);
#undef STREAMING_DISPATCH_METADATA_LAUNCH_CASE
}

template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch_main_kernel(
        DispatchPoolOut pool_out,
        DispatchPerTokenOut per_token_out,
        DispatchInputs inputs,
        DispatchTileSignal tile_signal,
        DispatchShape shape,
        DispatchEnv env) {
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
    auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);

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
            // NOTES: this is for distinguishing zero tokens
            if (elect_one_sync()) {
                st_relaxed_sys_global(channel_start_offset.buffer(), -start_count - 1);
                st_relaxed_sys_global(channel_end_offset.buffer(), -end_count - 1);
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
                st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
        }
    } else {
        // Workers for receiving — pool layout. Each landed (token, k) pair routing
        // to a local expert e gets its own pool slot, allocated deterministically
        // as `slot = base_pool[c, src, e] + smem_seen[e]++`. Slot allocation is
        // chunk-major in lane 0 (sequential SMEM increments — fast and cheap), so
        // slot order = (chunk_idx_in_substream, k) lex regardless of how the
        // sender chunked or how the receiver was paced. The data copy still uses
        // 3-warp parallelism reading slot positions from the SMEM batch_slot scratch.
        // After all batches drain, Pass 2 fires tile_ready in expert-major order
        // so kernel A sees firings in tile_id-monotonic order (preserves W1[e] L2
        // caching).
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

        // Receive channel offset.
        int total_offset, num_tokens_to_recv;
        if (elect_one_sync()) {
            while ((total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0)
                ;
            while ((num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0)
                ;
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
            while (recv_thread_id_in_rank == 0) {
                cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());
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
                    // __threadfence_system + tile_ready release-store sequence
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

            // Move queue head.
            cached_channel_head_idx += batch_total;
            total_offset += batch_total;
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 and elect_one_sync())
                st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx);

            num_tokens_to_recv -= batch_total;
        }

        // ── Pass 2: substream-end expert-major firing of tile_ready.
        // After all batches drain, walk experts in order; for each pool block this
        // substream contributed to, atomic-add the substream's per-block count and
        // (if the block is now full) release-store tile_ready[block] = dispatch_seq.
        // Iterating experts in order across substream blocks gives the kernel-A
        // scheduler a tile_id-monotonic firing stream (preserves W1[e] L2 caching).
#ifndef DISABLE_SM90_FEATURES
        tma_store_wait<0>();
#endif
        asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
        // Every thread in the rank-group fences its own prior writes at system
        // scope. ``pool_recv_token`` / ``pool_topk_weight`` / ``pool_k_slot`` are
        // written by lane 0 of EVERY receiver warp during the batch loop;
        // ``__threadfence_system()`` on a single thread (a thread-0-only fence)
        // only covers that one thread's writes. Block-scope ``bar.sync`` makes
        // other warps' writes visible WITHIN the block but doesn't propagate them
        // to system scope. Cross-stream consumers (kernel A acquiring tile_ready,
        // kernel Y acquiring a_ready) need system-scope visibility for those
        // per-warp writes; without this fence they could read stale -1 from the
        // pool's N memset init.
        __threadfence_system();
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
//   - Pass 2 atomic-adds into bwd_dispatch_arrival_count and release-stores
//     bwd_y_ready[block] = dispatch_seq when count == pool_arrival_target
//     (the same target fwd uses; we re-fire it on the bwd ready signal).
template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch_grads_main_kernel(
        DispatchGradsIO io,
        DispatchGradsRouting routing,
        DispatchGradsTileSignal tile_signal,
        DispatchGradsShape shape,
        DispatchEnv env) {
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

    auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);

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
                st_relaxed_sys_global(channel_start_offset.buffer(), -start_count - 1);
                st_relaxed_sys_global(channel_end_offset.buffer(), -end_count - 1);
            }
        }
        __syncwarp();

        int token_start_idx, token_end_idx;
        get_channel_task_range(shape.num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

        int cached_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            sender_wait_for_queue_space(cached_channel_tail_idx,
                                        channel_head_idx.buffer(),
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
                st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
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
            while ((total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0)
                ;
            while ((num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0)
                ;
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
                cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());
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
                st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx);

            num_tokens_to_recv -= batch_total;
        }

        // Pass 2 — same shape as fwd dispatch's Pass 2, but reads
        // `seen_per_substream` from global (instead of fwd's SMEM
        // `seen_substream`) and fires `bwd_y_ready` instead of `tile_ready`.
#ifndef DISABLE_SM90_FEATURES
        tma_store_wait<0>();
#endif
        asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
        __threadfence_system();
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

template <int kNumRanks>
__global__ void encode_combine_heads_kernel(
    void** buffer_ptrs, int* send_head, int num_channels, int num_recv_tokens, int num_memset_int, int** barrier_signal_ptrs, int rank) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    if (sm_id == 0) {
        // Barrier before cleaning
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        // Clean
        auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
        auto ptr = static_cast<int*>(buffer_ptrs[rank]);
        #pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            ptr[i] = 0;

        // Barrier after cleaning
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
    } else {
        const auto channel_id = sm_id - 1;
        const auto thread_id = static_cast<int>(threadIdx.x);
        const auto rank_id = thread_id / 32;
        const auto lane_id = thread_id % 32;
        if (rank_id >= kNumRanks)
            return;

        int token_start_idx, token_end_idx;
        get_channel_task_range(num_recv_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        int last_head = kReverseScanSentinel;
        #pragma unroll
        for (int token_idx_tail = token_end_idx - 1; token_idx_tail >= token_start_idx; token_idx_tail -= 32) {
            int token_idx = token_idx_tail - lane_id, expected_head = 0;
            auto current_head = (token_idx >= token_start_idx) ? __ldg(send_head + token_idx * kNumRanks + rank_id) : -1;
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
                send_head[token_idx * kNumRanks + rank_id] = expected_head;
        }
    }
}

void encode_combine_heads(void** buffer_ptrs,
                          int* send_head,
                          int num_channels,
                          int num_recv_tokens,
                          int num_memset_int,
                          int** barrier_signal_ptrs,
                          int rank,
                          int num_ranks,
                          cudaStream_t stream) {
#define ENCODE_COMBINE_HEADS(ranks)                    \
    LAUNCH_KERNEL(&cfg,                                \
                  encode_combine_heads_kernel<ranks>,  \
                  buffer_ptrs,                         \
                  send_head,                           \
                  num_channels,                        \
                  num_recv_tokens,                     \
                  num_memset_int,                      \
                  barrier_signal_ptrs,                 \
                  rank);                               \
    break

    const int num_threads = std::max(128, 32 * num_ranks);
    EP_HOST_ASSERT(num_ranks <= num_threads);
    EP_HOST_ASSERT(num_threads <= 1024);
    EP_HOST_ASSERT(1 + num_channels <= num_channels * 2);
    SETUP_LAUNCH_CONFIG(1 + num_channels, num_threads, stream);
    SWITCH_RANKS(ENCODE_COMBINE_HEADS);
#undef ENCODE_COMBINE_HEADS
}

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
template <typename dtype_t, int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
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
                                                          int num_tokens,
                                                          int num_recv_tokens,
                                                          int hidden,
                                                          int num_topk,
                                                          void** buffer_ptrs,
                                                          int rank,
                                                          int num_max_send_tokens,
                                                          int num_recv_buffer_tokens) {
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto sm_id = static_cast<int>(blockIdx.x), lane_id = get_lane_id();
    const auto num_channels = num_sms / 2;
    const bool is_sender = sm_id % 2 == 0;
    const int responsible_channel = sm_id / 2;
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

        // Calculate pointers by the specific layout
        auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[send_rank_id]));
        auto num_channels_total = num_channels * kNumRanks;
        auto channel_rank_offset = responsible_channel * kNumRanks + rank;

        // Channel meta data
        // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
        // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
        // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
        // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)   ← skipped (unused by combine; populated by dispatch only)
        // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
        auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
        auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
        auto channel_x_buffers = Buffer<int4>(
            ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
        // Skip past src_idx region (slab layout fixed by dispatch; combine
        // doesn't write or read it).
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

                // Send the per-(my_token, k) weight via slot lookup. Fwd:
                // pool_topk_weight[slot] = topk_weights[t, k]; bwd:
                // weight_grads[slot] = dL/dweight at that (r, k_local). For
                // non-local k (slot == -1) we ship 0 — receiver's K-way sum
                // then yields the correct (t, k) value, since exactly one
                // sender has the non-zero contribution per (t, k).
                if (num_topk > 0 and lane_id < num_topk) {
                    int slot = recv_token_to_slots[my_token * num_topk + lane_id];
                    float w = (slot >= 0) ? __ldg(per_slot_weights + slot) : 0.0f;
                    channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = w;
                }
            }
            token_idx += num_round_tokens;
            current_channel_tail_idx += num_round_tokens;

            // Move tail index
            asm volatile("bar.sync %0, %1;" ::"r"(send_rank_id), "r"(num_threads_per_rank));
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), current_channel_tail_idx);
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
            int* channel_head_idx_ptr = static_cast<int*>(buffer_ptrs[rank]) + responsible_channel * kNumRanks + lane_id;
            int* channel_tail_idx_ptr = channel_head_idx_ptr + num_channels * kNumRanks;

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

                // Update queue tail
                channel_tail_idx[lane_id] = ld_acquire_sys_global(channel_tail_idx_ptr);

                // Update minimum head
                int min_head = std::numeric_limits<int>::max();
                #pragma unroll
                for (int i = 1; i < num_recv_warps; ++i)
                    if (not warp_retired[i])
                        min_head = min(min_head, warp_channel_head_idx[i][lane_id]);
                if (min_head != std::numeric_limits<int>::max() and min_head > last_head)
                    st_relaxed_sys_global(channel_head_idx_ptr, last_head = min_head);
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
                // `head_idx` & `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
                auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[rank]) + 2 * num_channels * kNumRanks * sizeof(int));

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

                // Reduce `topk_weights` (fwd: combined topk_weights output;
                // bwd: dL/dtopk_weights output — same code, different sink).
                if (lane_id < num_topk) {
                    float value = 0;
                    #pragma unroll
                    for (int i = 0; i < num_topk_ranks; ++i)
                        value += ld_nc_global(channel_topk_weights_buffers[topk_ranks[i]].buffer() + slot_indices[i] * num_topk + lane_id);
                    recv_topk_weights_out[token_idx * num_topk + lane_id] = value;
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
             int num_tokens,
             int num_recv_tokens,
             int hidden,
             int num_topk,
             void** buffer_ptrs,
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

#define COMBINE_LAUNCH_CASE(dtype, ranks)                                                  \
    {                                                                                      \
        auto kernel = combine_main_kernel<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \
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
                      y_done_per_token,                                  \
                      combine_seq,                                             \
                      num_tokens,                                              \
                      num_recv_tokens,                                         \
                      hidden,                                                  \
                      num_topk,                                                \
                      buffer_ptrs,                                             \
                      rank,                                                    \
                      num_max_send_tokens,                                     \
                      num_recv_buffer_tokens);                                 \
    }                                                                          \
    break
#define COMBINE_DTYPE_LAUNCH_CASE(dtype)                 \
    SWITCH_RANKS_WITH_DTYPE(dtype, COMBINE_LAUNCH_CASE); \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving
    EP_HOST_ASSERT(num_sms % 2 == 0);
    EP_HOST_ASSERT(kNumThreads >= num_ranks * 32);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_TYPES(COMBINE_DTYPE_LAUNCH_CASE);
#undef COMBINE_DTYPE_LAUNCH_CASE
#undef COMBINE_LAUNCH_CASE
}

}  // namespace intranode

}  // namespace stream_ep
