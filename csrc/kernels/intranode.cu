#include "api.cuh"
#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace deep_ep {

namespace intranode {

// Per-sub-batch chunk size in the dispatch receiver. Bounds the batch_slot SMEM
// scratch independent of the channel queue depth — Pass A allocates pool slots
// for at most kReceiverChunkSize (chunk, k) pairs at a time before Pass B drains
// them. Used both in the kernel and at the launch site for SMEM sizing.
constexpr int kReceiverChunkSize = 32;

// Receiver-state SMEM layout, used by the dispatch kernel's receiver block. Sits
// after the per-warp TMA buffer slabs in dynamic SMEM:
//   smem_per_substream_seen [num_ranks][E_local]                       — per-(c, src, e) cumulative count
//   smem_batch_slot         [num_ranks][kReceiverChunkSize][num_topk]  — per-sub-batch slot map
__host__ __device__ inline int receiver_state_smem_bytes(int num_ranks, int E_local, int num_topk) {
    return (num_ranks * E_local +
            num_ranks * kReceiverChunkSize * num_topk) * static_cast<int>(sizeof(int));
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
        int* rank_prefix_matrix,
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
    const int slab_e_size = num_channels * E;       // per-dst slice of local_e
    const int slab_u_size = num_channels;            // per-dst slice of local_u

    extern __shared__ int smem[];
    int* local_e = smem;                                          // [kNumRanks * slab_e_size]
    int* local_u = local_e + kNumRanks * slab_e_size;             // [kNumRanks * slab_u_size]

    // Phase 1: zero local histograms.
    for (int i = thread_id; i < kNumRanks * slab_e_size; i += num_threads)
        local_e[i] = 0;
    for (int i = thread_id; i < kNumRanks * slab_u_size; i += num_threads)
        local_u[i] = 0;
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
            atomicAdd(&local_e[dst_rank * slab_e_size + channel_id * E + e_local], 1);
            uint64_t bit = 1ULL << dst_rank;
            if (!(dst_mask & bit)) {
                dst_mask |= bit;
                atomicAdd(&local_u[dst_rank * slab_u_size + channel_id], 1);
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
            peer_e[c * kNumRanks * E + rank * E + e] = local_e[dst * slab_e_size + i];
        }
        for (int c = lane_id; c < num_channels; c += 32)
            peer_u[c * kNumRanks + rank] = local_u[dst * slab_u_size + c];
    }
    __syncthreads();

    // Phase 5: cross-rank barrier — peers' phase-4 stores now observable.
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

    // Phase 6: read own inbox, derive metadata. Reuse the (now-stale) local SMEM.
    auto* my_e_inbox = reinterpret_cast<int*>(
        static_cast<uint8_t*>(buffer_ptrs[rank]) + streaming_section_offset);
    auto* my_u_inbox = my_e_inbox + e_inbox_size;

    int* s_freq     = smem;                       // [E]
    int* s_pool_blk = s_freq + E;                 // [E + 1]
    int* s_per_src  = s_pool_blk + (E + 1);       // [kNumRanks]

    // expert_frequency[e] = sum over (c, src) of e_inbox[c, src, e].
    for (int e = thread_id; e < E; e += num_threads) {
        int sum = 0;
        for (int cs = 0; cs < num_channels * kNumRanks; ++cs)
            sum += my_e_inbox[cs * E + e];
        s_freq[e] = sum;
        expert_frequency[e] = sum;
    }
    // s_per_src[src] = sum over c of u_inbox[c, src].
    for (int src = thread_id; src < kNumRanks; src += num_threads) {
        int total = 0;
        for (int c = 0; c < num_channels; ++c)
            total += my_u_inbox[c * kNumRanks + src];
        s_per_src[src] = total;
    }
    __syncthreads();

    if (thread_id == 0) {
        int cum_blocks = 0;
        s_pool_blk[0] = 0;
        for (int e = 0; e < E; ++e) {
            int n_blocks_e = (s_freq[e] + tile_m - 1) / tile_m;
            cum_blocks += n_blocks_e;
            s_pool_blk[e + 1] = cum_blocks;
        }
        *total_tiles_out = cum_blocks;
        *total_tiles_mapped = cum_blocks;

        int total_unique = 0;
        for (int i = 0; i < kNumRanks; ++i)
            total_unique += s_per_src[i];
        *num_recv_mapped = total_unique;

        for (int e = 0; e < E; ++e) {
            int aligned = (s_freq[e] + expert_alignment - 1) / expert_alignment * expert_alignment;
            num_recv_per_expert_mapped[e] = aligned;
        }

        // rank_prefix_matrix: this rank fills its own column. Cumulative unique
        // tokens from senders 0..i to this rank — read by combine on rank j.
        int cum_src = 0;
        for (int i = 0; i < kNumRanks; ++i) {
            cum_src += s_per_src[i];
            rank_prefix_matrix[i * kNumRanks + rank] = cum_src;
        }
    }
    __syncthreads();
    for (int e = thread_id; e < E + 1; e += num_threads)
        expert_pool_block_offset[e] = s_pool_blk[e];

    // Phase 7: base_pool[c, src, e] = pool slot start (in pool-row units) for
    // substream (c, src) writes for expert e:
    //   base_pool[c, src, e] = s_pool_blk[e] * tile_m
    //                          + Σ over (c', src') < (c, src) lex of e_inbox[c', src', e].
    // E ≤ NUM_MAX_LOCAL_EXPERTS so we can run one thread per expert in parallel.
    for (int e = thread_id; e < E; e += num_threads) {
        int acc = s_pool_blk[e] * tile_m;
        for (int cs = 0; cs < num_channels * kNumRanks; ++cs) {
            base_pool[cs * E + e] = acc;
            acc += my_e_inbox[cs * E + e];
        }
    }
}

void streaming_dispatch_metadata(const topk_idx_t* topk_idx,
                                 int* expert_frequency,
                                 int* expert_pool_block_offset,
                                 int* base_pool,
                                 int* rank_prefix_matrix,
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
    LAUNCH_KERNEL(&cfg,                                                                      \
                  streaming_dispatch_metadata_kernel<ranks>,                                 \
                  topk_idx,                                                                  \
                  expert_frequency,                                                          \
                  expert_pool_block_offset,                                                  \
                  base_pool,                                                                 \
                  rank_prefix_matrix,                                                        \
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
    int smem_bytes = num_ranks * num_channels * (num_experts_per_rank + 1) * sizeof(int);
    cfg.dynamicSmemBytes = smem_bytes;
    SWITCH_RANKS(STREAMING_DISPATCH_METADATA_LAUNCH_CASE);
#undef STREAMING_DISPATCH_METADATA_LAUNCH_CASE
}

// Pool-layout per-tile arrays. Sized by `total_tiles` (known only after the host
// poll on streaming_total_tiles), so this is a separate launch from the metadata
// kernel. Outputs:
//   - tile_id_to_expert[total_tiles]      int32 — which expert this tile belongs to.
//   - pool_arrival_target[total_tiles]    int32 — write count needed for the tile
//       to be considered "ready" (BLOCK_M for full tiles, leftover for the last
//       partial tile of each expert).
__global__ void tile_arrays_init_kernel(const int* __restrict__ expert_frequency,
                                        const int* __restrict__ expert_pool_block_offset,
                                        int* tile_id_to_expert,
                                        int* pool_arrival_target,
                                        int E,
                                        int total_tiles,
                                        int tile_m) {
    int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_id >= total_tiles)
        return;

    int e = 0;
    while (e < E && expert_pool_block_offset[e + 1] <= tile_id)
        ++e;
    int e_start = expert_pool_block_offset[e];
    int e_end = expert_pool_block_offset[e + 1];
    int tile_in_e = tile_id - e_start;

    tile_id_to_expert[tile_id] = e;
    if (tile_id == e_end - 1) {
        int rem = expert_frequency[e] - tile_in_e * tile_m;
        pool_arrival_target[tile_id] = rem;
    } else {
        pool_arrival_target[tile_id] = tile_m;
    }
}

void tile_arrays_init(const int* expert_frequency,
                      const int* expert_pool_block_offset,
                      int* tile_id_to_expert,
                      int* pool_arrival_target,
                      int num_experts_per_rank,
                      int total_tiles,
                      int tile_m,
                      cudaStream_t stream) {
    if (total_tiles == 0)
        return;
    int threads = 128;
    int blocks = (total_tiles + threads - 1) / threads;
    tile_arrays_init_kernel<<<blocks, threads, 0, stream>>>(
        expert_frequency, expert_pool_block_offset, tile_id_to_expert, pool_arrival_target,
        num_experts_per_rank, total_tiles, tile_m);
}

template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch(int4* pool,
                                                           float* pool_x_scales,
                                                           float* pool_topk_weight,
                                                           int* pool_recv_token,
                                                           int* pool_k_slot,
                                                           int* recv_src_idx,
                                                           float* recv_topk_weights,
                                                           int* recv_channel_offset,
                                                           int* send_head,
                                                           int* per_token_remaining,
                                                           const int4* x,
                                                           const float* x_scales,
                                                           const topk_idx_t* topk_idx,
                                                           const float* topk_weights,
                                                           const bool* is_token_in_rank,
                                                           int* channel_prefix_matrix,
                                                           const int* base_pool,
                                                           int* pool_arrival_count,
                                                           const int* pool_arrival_target,
                                                           int64_t* tile_ready,
                                                           int64_t dispatch_seq,
                                                           int num_tokens,
                                                           int hidden_int4,
                                                           int num_topk,
                                                           int num_experts,
                                                           int num_scales,
                                                           int scale_token_stride,
                                                           int scale_hidden_stride,
                                                           void** buffer_ptrs,
                                                           int rank,
                                                           int num_max_send_tokens,
                                                           int num_recv_buffer_tokens,
                                                           int tile_m) {
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

    int num_experts_per_rank = num_experts / kNumRanks;
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 and num_topk > 0);
    // Pool-layout SMEM scratch is sized by E_local at runtime; a compile-time
    // upper bound keeps the receiver-block per-(chunk, k) slot computation
    // unrolled. 32 covers any practical MoE: even Mixtral-style routing has
    // num_experts_per_rank well under 32 at world_size ≥ 8.
    EP_DEVICE_ASSERT(num_experts_per_rank <= 32);
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
    EP_DEVICE_ASSERT(recv_topk_weights != nullptr);

    // Calculate pointers by the specific layout
    // `rank_prefix_matrix`: kNumRanks * kNumRanks * sizeof(int)
    auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[is_sender ? responsible_rank : rank]) +
                                       kNumRanks * kNumRanks * sizeof(int));
    int target_rank = is_sender ? rank : responsible_rank;
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
    // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
    // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
    // `topk_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(topk_idx_t)
    // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
    // `x_scales_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_scales * sizeof(float)
    auto channel_x_buffers = Buffer<int4>(
        ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
    auto channel_src_idx_buffers =
        Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
    auto channel_topk_idx_buffers = Buffer<topk_idx_t>(
        ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_topk_weights_buffers =
        Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_x_scales_buffers = Buffer<float>(
        ptr, num_channels_total * num_recv_buffer_tokens * num_scales, channel_rank_offset * num_recv_buffer_tokens * num_scales);

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto half_hidden_int4 = hidden_int4 / 2;
    auto half_hidden_bytes = half_hidden_int4 * static_cast<int>(sizeof(int4));
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + half_hidden_bytes);
    uint32_t tma_phase = 0;
    if (elect_one_sync()) {
        mbarrier_init(tma_mbarrier, 1);
        fence_barrier_init();
        EP_DEVICE_ASSERT(hidden_int4 % 2 == 0 and half_hidden_bytes + sizeof(uint64_t) <= kNumTMABytesPerWarp);
    }
    __syncwarp();
#endif

    if (is_sender) {
        // Workers for sending
        constexpr int num_send_warps = kNumThreads / 32;
        constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
        const auto send_thread_id = thread_id;
        const auto send_warp_id_in_rank = send_thread_id % num_threads_per_rank / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32);
        EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

        // Compute the channel-prefix locally over `is_token_in_rank`. One warp
        // per (channel, dst_rank) pair scans tokens [0, channel_end_of_c) and
        // splits the count into start (i < channel_start) and end (full range)
        // using lane-strided accumulation + warp_reduce.
        if (send_warp_id_in_rank == 0) {
            int channel_start, channel_end;
            get_channel_task_range(num_tokens, num_channels, responsible_channel, channel_start, channel_end);
            int start_count = 0, end_count = 0;
            for (int i = lane_id; i < channel_end; i += 32) {
                int v = is_token_in_rank[i * kNumRanks + responsible_rank];
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
                channel_prefix_matrix[responsible_rank * num_channels + responsible_channel] = end_count;
            }
        }
        __syncwarp();

        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

        // Iterate over all tokens and send by chunks
        int cached_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            // NOTES: the head index received by different warps may not be the same
            auto start_time = clock64();
            if (elect_one_sync()) {
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    int num_used_slots = cached_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
                    if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens)
                        break;

                    // Rare cases to loop again
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for dispatch senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();
                    }
                }
            }
            __syncwarp();

            int chunk_token_idx = 0;
            while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
                // NOTES: for the same token, the warp assigned to save `send_head` may be different from the warp assigned to send the
                // following data
                if (token_idx % num_send_warps_per_rank == send_warp_id_in_rank and elect_one_sync())
                    send_head[token_idx * kNumRanks + responsible_rank] =
                        is_token_in_rank[token_idx * kNumRanks + responsible_rank] ? cached_channel_tail_idx : -1;

                // Skip if not selected
                if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                    token_idx++;
                    continue;
                }

                // Get an empty slot
                int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;
                if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
                    // Copy data
                    auto shifted_channel_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                    auto shifted_x = x + token_idx * hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_channel_x_buffers, shifted_x, __ldg, st_na_global);

                    // Copy source index
                    if (elect_one_sync())
                        channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);

                    // Copy `topk_idx` and `topk_weights` with transformed index
                    if (lane_id < num_topk) {
                        // Top-k index
                        int recv_expert_begin = responsible_rank * num_experts_per_rank,
                            recv_expert_end = (responsible_rank + 1) * num_experts_per_rank;
                        auto idx_value = __ldg(topk_idx + token_idx * num_topk + lane_id);
                        idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end) ? idx_value - recv_expert_begin : -1;
                        channel_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] = idx_value;

                        // Top-k weights
                        auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);
                        weight_value = (idx_value >= 0) ? weight_value : 0.0f;
                        channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = weight_value;
                    }

                    // Copy `x_scales`
                    #pragma unroll
                    for (int i = lane_id; i < num_scales; i += 32) {
                        auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                        channel_x_scales_buffers[dst_slot_idx * num_scales + i] = __ldg(x_scales + offset);
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
        EP_DEVICE_ASSERT(kNumRanks <= 32);
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
        int* recv_state = reinterpret_cast<int*>(
            smem_buffer + kNumTMABytesPerWarp * num_recv_warps);
#else
        extern __shared__ int recv_state_no_tma[];
        int* recv_state = recv_state_no_tma;
#endif
        const int seen_stride = E;
        const int slot_stride = kReceiverChunkSize * num_topk;
        int* smem_per_substream_seen = recv_state;
        int* smem_batch_slot         = smem_per_substream_seen + kNumRanks * seen_stride;
        (void)num_max_send_tokens;

        // Initialize cumulative seen for this substream's local experts.
        if (recv_thread_id_in_rank < E)
            smem_per_substream_seen[responsible_rank * seen_stride + recv_thread_id_in_rank] = 0;
        __syncthreads();

        // Calculate offset
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank] : 0;

        // Receive channel offset.
        int total_offset, num_tokens_to_recv;
        if (elect_one_sync()) {
            while ((total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0)
                ;
            while ((num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0)
                ;
            total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
            if (recv_warp_id_in_rank == 0)
                recv_channel_offset[responsible_rank * num_channels + responsible_channel] = total_offset;
            num_tokens_to_recv -= total_offset;
        }
        total_offset = __shfl_sync(0xffffffff, total_offset, 0);
        total_offset += rank_offset;  // becomes the recv-token row index for chunk_idx 0
        num_tokens_to_recv = __shfl_sync(0xffffffff, num_tokens_to_recv, 0);

        // Shared tail indices for different warps within a rank-group.
        __shared__ volatile int shared_channel_tail_idx[kNumRanks];

        const int substream_csrc = responsible_channel * kNumRanks + responsible_rank;
        const int* base_pool_substream = base_pool + substream_csrc * E;
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
                           rank, responsible_channel, num_tokens_to_recv);
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
                        int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                        int* slot_row = batch_slot_substream + sub_chunk * num_topk;
                        for (int k = 0; k < num_topk; ++k) {
                            int e_local = static_cast<int>(
                                channel_topk_idx_buffers[token_idx_in_buffer * num_topk + k]);
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
                    int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                    int recv_token_id = total_offset + chunk_idx;
                    int* slot_row = batch_slot_substream + sub_chunk * num_topk;

                    auto shifted_buffer_x_int4 = channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;
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
                            for (int k = 0; k < num_topk; ++k) {
                                int slot = slot_row[k];
                                if (slot < 0) continue;
                                tma_store_1d(tma_buffer,
                                             pool + static_cast<int64_t>(slot) * hidden_int4
                                                  + piece * half_hidden_int4,
                                             half_hidden_bytes, false);
                            }
                        }
                        __syncwarp();
                    }
#else
                    for (int k = 0; k < num_topk; ++k) {
                        int slot = slot_row[k];
                        if (slot < 0) continue;
                        auto shifted_pool_int4 = pool + static_cast<int64_t>(slot) * hidden_int4;
                        UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_pool_int4, shifted_buffer_x_int4,
                                           ld_nc_global, st_na_global);
                    }
#endif

                    // Per-pool-slot scalar writes + per-recv-token K_local count.
                    // Intranode: each recv-token is delivered by exactly one
                    // substream, so the K_local count emitted here is final
                    // (no cross-substream accumulation needed). Pass 2's
                    // __threadfence_system + tile_ready release-store sequence
                    // after this make per_token_remaining[r] visible to kernel Y
                    // before kernel Y's first atomicSub on the same address.
                    if (lane_id == 0) {
                        int k_local_count = 0;
                        for (int k = 0; k < num_topk; ++k) {
                            int slot = slot_row[k];
                            if (slot < 0) continue;
                            pool_topk_weight[slot] = ld_nc_global(
                                channel_topk_weights_buffers.buffer() + token_idx_in_buffer * num_topk + k);
                            pool_recv_token[slot] = recv_token_id;
                            pool_k_slot[slot] = k;
                            ++k_local_count;
                        }
                        if (k_local_count > 0)
                            per_token_remaining[recv_token_id] = k_local_count;
                    }

                    // x_scales: per-(slot, scales_idx). Lanes split scales_idx within K.
                    if (num_scales > 0) {
                        for (int k = 0; k < num_topk; ++k) {
                            int slot = slot_row[k];
                            if (slot < 0) continue;
                            for (int i = lane_id; i < num_scales; i += 32) {
                                pool_x_scales[static_cast<int64_t>(slot) * num_scales + i] =
                                    ld_nc_global(channel_x_scales_buffers.buffer() + token_idx_in_buffer * num_scales + i);
                            }
                        }
                    }

                    // Per-recv-token writes (combine consumes recv_src_idx + recv_topk_weights).
                    if (lane_id == 0)
                        recv_src_idx[recv_token_id] = ld_nc_global(channel_src_idx_buffers.buffer() + token_idx_in_buffer);
                    if (lane_id < num_topk) {
                        recv_topk_weights[static_cast<int64_t>(recv_token_id) * num_topk + lane_id] =
                            ld_nc_global(channel_topk_weights_buffers.buffer() + token_idx_in_buffer * num_topk + lane_id);
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
        if (recv_thread_id_in_rank == 0) {
            __threadfence_system();
            for (int e = 0; e < E; ++e) {
                int n_writes_for_e = seen_substream[e];
                if (n_writes_for_e == 0) continue;
                int slot_start_e = base_pool_substream[e];
                int slot_end_e = slot_start_e + n_writes_for_e;
                int first_block = slot_start_e / tile_m;
                int last_block = (slot_end_e - 1) / tile_m;
                for (int block_id = first_block; block_id <= last_block; ++block_id) {
                    int block_slot_start = block_id * tile_m;
                    int block_slot_end = block_slot_start + tile_m;
                    int writes_in_block =
                        min(slot_end_e, block_slot_end) - max(slot_start_e, block_slot_start);
                    int cnt_before = atomicAdd(&pool_arrival_count[block_id], writes_in_block);
                    if (cnt_before + writes_in_block == pool_arrival_target[block_id]) {
                        memory_fence();
                        st_release_sys_global(tile_ready + block_id, dispatch_seq);
                    }
                }
            }
        }
    }
}

void dispatch(void* pool,
              float* pool_x_scales,
              float* pool_topk_weight,
              int* pool_recv_token,
              int* pool_k_slot,
              int* recv_src_idx,
              float* recv_topk_weights,
              int* recv_channel_offset,
              int* send_head,
              int* per_token_remaining,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              const bool* is_token_in_rank,
              int* channel_prefix_matrix,
              const int* base_pool,
              int* pool_arrival_count,
              const int* pool_arrival_target,
              int64_t* tile_ready,
              int64_t dispatch_seq,
              int num_tokens,
              int hidden_int4,
              int num_topk,
              int num_experts,
              int num_scales,
              int scale_token_stride,
              int scale_hidden_stride,
              void** buffer_ptrs,
              int rank,
              int num_ranks,
              cudaStream_t stream,
              int num_sms,
              int num_max_send_tokens,
              int num_recv_buffer_tokens,
              int tile_m) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 8192;
    constexpr int kNumWarps = kNumThreads / 32;

    int E_local = num_experts / num_ranks;
    int receiver_state_bytes = receiver_state_smem_bytes(num_ranks, E_local, num_topk);
#ifndef DISABLE_SM90_FEATURES
    int smem_size = kNumTMABytesPerWarp * kNumWarps + receiver_state_bytes;
#else
    int smem_size = receiver_state_bytes;
#endif

    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(ranks)                                      \
    {                                                                    \
        auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                               \
        LAUNCH_KERNEL(&cfg,                                              \
                      kernel,                                            \
                      reinterpret_cast<int4*>(pool),                     \
                      pool_x_scales,                                     \
                      pool_topk_weight,                                  \
                      pool_recv_token,                                   \
                      pool_k_slot,                                       \
                      recv_src_idx,                                      \
                      recv_topk_weights,                                 \
                      recv_channel_offset,                               \
                      send_head,                                         \
                      per_token_remaining,                               \
                      reinterpret_cast<const int4*>(x),                  \
                      x_scales,                                          \
                      topk_idx,                                          \
                      topk_weights,                                      \
                      is_token_in_rank,                                  \
                      channel_prefix_matrix,                             \
                      base_pool,                                         \
                      pool_arrival_count,                                \
                      pool_arrival_target,                               \
                      tile_ready,                                        \
                      dispatch_seq,                                      \
                      num_tokens,                                        \
                      hidden_int4,                                       \
                      num_topk,                                          \
                      num_experts,                                       \
                      num_scales,                                        \
                      scale_token_stride,                                \
                      scale_hidden_stride,                               \
                      buffer_ptrs,                                       \
                      rank,                                              \
                      num_max_send_tokens,                               \
                      num_recv_buffer_tokens,                            \
                      tile_m);                                           \
    }                                                                    \
    break

    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    cfg.dynamicSmemBytes = smem_size;
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_combine(
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

        // NOTES: `1 << 25` is a heuristic large number
        int last_head = 1 << 25;
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

void cached_notify_combine(void** buffer_ptrs,
                           int* send_head,
                           int num_channels,
                           int num_recv_tokens,
                           int num_memset_int,
                           int** barrier_signal_ptrs,
                           int rank,
                           int num_ranks,
                           cudaStream_t stream) {
#define CACHED_NOTIFY_COMBINE(ranks)            \
    LAUNCH_KERNEL(&cfg,                         \
                  cached_notify_combine<ranks>, \
                  buffer_ptrs,                  \
                  send_head,                    \
                  num_channels,                 \
                  num_recv_tokens,              \
                  num_memset_int,               \
                  barrier_signal_ptrs,          \
                  rank);                        \
    break

    const int num_threads = std::max(128, 32 * num_ranks);
    EP_HOST_ASSERT(num_ranks <= num_threads);
    EP_HOST_ASSERT(num_threads <= 1024);
    EP_HOST_ASSERT(1 + num_channels <= num_channels * 2);
    SETUP_LAUNCH_CONFIG(1 + num_channels, num_threads, stream);
    SWITCH_RANKS(CACHED_NOTIFY_COMBINE);
#undef CACHED_NOTIFY_COMBINE
}

template <typename dtype_t, int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) combine(dtype_t* recv_x,
                                                          float* recv_topk_weights,
                                                          const dtype_t* x,
                                                          const float* topk_weights,
                                                          const dtype_t* bias_0,
                                                          const dtype_t* bias_1,
                                                          const int* src_idx,
                                                          const int* rank_prefix_matrix,
                                                          const int* channel_prefix_matrix,
                                                          int* send_head,
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
    EP_DEVICE_ASSERT(num_topk <= 32);

    constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
    int hidden_int4 = hidden * sizeof(dtype_t) / sizeof(int4);
    int hidden_int4_aligned = align_down(hidden_int4, 32);
    auto x_int4 = reinterpret_cast<const int4*>(x);
    auto bias_0_int4 = reinterpret_cast<const int4*>(bias_0);
    auto bias_1_int4 = reinterpret_cast<const int4*>(bias_1);
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
        // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
        // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
        auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
        auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
        auto channel_x_buffers = Buffer<int4>(
            ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
        auto channel_src_idx_buffers =
            Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
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
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            auto start_time = clock64();
            int num_round_tokens = min(num_max_send_tokens, token_end_idx - static_cast<int>(token_idx));
            if (elect_one_sync()) {
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    int num_used_slots = current_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
                    if (num_recv_buffer_tokens - num_used_slots >= num_round_tokens)
                        break;

                    // Rare cases to loop again
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for combine senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();
                    }
                }
            }
            __syncwarp();

            // Send by chunk
            #pragma unroll
            for (int i = send_warp_id_in_rank; i < num_round_tokens; i += num_send_warps_per_rank) {
                // Get an empty slot
                int dst_slot_idx = (current_channel_tail_idx + i) % num_recv_buffer_tokens;

                // Copy data
                auto shifted_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                auto shifted_x = x_int4 + (token_idx + i) * hidden_int4;
                UNROLLED_WARP_COPY(4, lane_id, hidden_int4, shifted_x_buffers, shifted_x, ld_nc_global, st_na_global);

                // Send source index
                if (elect_one_sync())
                    channel_src_idx_buffers[dst_slot_idx] = __ldg(src_idx + token_idx + i);

                // Send `topk_weights`
                if (num_topk > 0 and lane_id < num_topk)
                    channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] =
                        __ldg(topk_weights + (token_idx + i) * num_topk + lane_id);
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
        EP_DEVICE_ASSERT(kNumRanks <= 32 and kNumThreads > 32);
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
                    // Read bias
                    // TODO: make it as a template
                    int4 bias_0_value_int4 =
                        bias_0_int4 != nullptr ? __ldg(bias_0_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);
                    int4 bias_1_value_int4 =
                        bias_1_int4 != nullptr ? __ldg(bias_1_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);

                    // Read buffers
                    int4 recv_value_int4[kNumRanks];
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++j)
                        recv_value_int4[j] = ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() + slot_indices[j] * hidden_int4 + i);

                    // Reduce bias
                    float values[kDtypePerInt4];
                    auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
                    auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);
                    #pragma unroll
                    for (int j = 0; j < kDtypePerInt4; ++j)
                        values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);

                    // Reduce all-to-all results
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

                // Reduce `topk_weights`
                if (lane_id < num_topk) {
                    float value = 0;
                    #pragma unroll
                    for (int i = 0; i < num_topk_ranks; ++i)
                        value += ld_nc_global(channel_topk_weights_buffers[topk_ranks[i]].buffer() + slot_indices[i] * num_topk + lane_id);
                    recv_topk_weights[token_idx * num_topk + lane_id] = value;
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

void combine(cudaDataType_t type,
             void* recv_x,
             float* recv_topk_weights,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* src_idx,
             const int* rank_prefix_matrix,
             const int* channel_prefix_matrix,
             int* send_head,
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

#define COMBINE_LAUNCH_CASE(dtype, ranks)                                      \
    {                                                                          \
        auto kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                                     \
        LAUNCH_KERNEL(&cfg,                                                    \
                      kernel,                                                  \
                      reinterpret_cast<dtype*>(recv_x),                        \
                      recv_topk_weights,                                       \
                      reinterpret_cast<const dtype*>(x),                       \
                      topk_weights,                                            \
                      reinterpret_cast<const dtype*>(bias_0),                  \
                      reinterpret_cast<const dtype*>(bias_1),                  \
                      src_idx,                                                 \
                      rank_prefix_matrix,                                      \
                      channel_prefix_matrix,                                   \
                      send_head,                                               \
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

}  // namespace deep_ep
