#include "api.cuh"
#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace deep_ep {

namespace intranode {


// Sender-side per-(dst_rank, channel) token counts. One block per destination rank
// scans this rank's `is_token_in_rank` channel-locally and produces the cumulative
// `channel_prefix_matrix[num_ranks, num_channels]` consumed by the dispatch sender
// and combine. No cross-rank exchange; receiver-side metadata (num_recv, per-expert
// counts, rank_prefix_matrix) is produced by `streaming_metadata_init` from
// `recv_count`. See new_design.md §"The count exchange".
template <int kNumRanks>
__global__ void notify_dispatch(int num_tokens,
                                int num_channels,
                                const bool* is_token_in_rank,
                                int* channel_prefix_matrix) {
    auto dst_rank = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto lane_id = thread_id % 32, warp_id = thread_id / 32, num_warps = num_threads / 32;

    for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        int count = 0;
        for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32)
            count += is_token_in_rank[i * kNumRanks + dst_rank];
        count = warp_reduce_sum(count);
        if (elect_one_sync())
            channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
    }
    __syncthreads();

    if (thread_id == 0) {
        #pragma unroll
        for (int i = 1; i < num_channels; ++i)
            channel_prefix_matrix[dst_rank * num_channels + i] += channel_prefix_matrix[dst_rank * num_channels + i - 1];
    }
}

void notify_dispatch(int num_ranks,
                     int num_tokens,
                     const bool* is_token_in_rank,
                     int* channel_prefix_matrix,
                     cudaStream_t stream,
                     int num_channels) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)   \
    LAUNCH_KERNEL(&cfg,                      \
                  notify_dispatch<ranks>,    \
                  num_tokens,                \
                  num_channels,              \
                  is_token_in_rank,          \
                  channel_prefix_matrix);    \
    break

    constexpr int kNumThreads = 128;
    SETUP_LAUNCH_CONFIG(num_ranks, kNumThreads, stream);
    SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

// Streaming-MoE count exchange.
//
// Two-phase design, no cross-rank atomicAdds:
//   1. Local histogram in SMEM. Each thread iterates per-token (with `dst_mask` to
//      dedupe k slots that share the same dst). LOCAL atomicAdds into SMEM count
//      every routed (token, k) pair into `local_e[dst, c, e]` and every unique
//      (token, dst) into `local_u[dst, c]`.
//   2. Bulk store to peers. Each (sender, dst) pair writes to a UNIQUE region of
//      dst's IPC inbox (slot [c, src=sender, e]). No write contention across
//      senders, so plain stores — no atomicAdds.
//
// This avoids T × K cross-rank atomicAdds per dispatch (~50K at production); the
// cross-rank traffic is now bounded by the histogram size: (num_dst × num_channels
// × E_local) ints per sender, total ~12 KB at production intranode.
//
// Two adjacent inboxes per rank at `buffer_ptrs[R] + streaming_section_offset`:
//   - e_inbox[c, src, e]: per-(channel, src, local_expert) (token, k) count
//   - u_inbox[c, src]:    per-(channel, src) UNIQUE token count
template <int kNumRanks>
__global__ void streaming_count_exchange(const topk_idx_t* topk_idx,
                                         int* recv_count_out,
                                         int* recv_unique_per_source_out,
                                         int num_tokens,
                                         int num_topk,
                                         int num_experts_per_rank,
                                         int num_channels,
                                         int64_t streaming_section_offset,
                                         void** buffer_ptrs,
                                         int** barrier_signal_ptrs,
                                         int rank) {
    auto thread_id = static_cast<int>(threadIdx.x);
    auto num_threads = static_cast<int>(blockDim.x);
    auto warp_id = thread_id / 32;
    auto lane_id = thread_id % 32;
    auto num_warps = num_threads / 32;

    const int E = num_experts_per_rank;
    const int e_inbox_size = num_channels * kNumRanks * E;
    const int u_inbox_size = num_channels * kNumRanks;
    const int slab_e_size = num_channels * E;       // per-dst slice of local_e
    const int slab_u_size = num_channels;            // per-dst slice of local_u

    // SMEM histograms: per-(dst, c, e) and per-(dst, c).
    extern __shared__ int smem[];
    int* local_e = smem;                                         // [kNumRanks * slab_e_size]
    int* local_u = local_e + kNumRanks * slab_e_size;            // [kNumRanks * slab_u_size]

    // Zero local histograms.
    for (int i = thread_id; i < kNumRanks * slab_e_size; i += num_threads)
        local_e[i] = 0;
    for (int i = thread_id; i < kNumRanks * slab_u_size; i += num_threads)
        local_u[i] = 0;
    __syncthreads();

    // Cross-rank handshake: peers' kernels are running, IPC slabs ready to write.
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

    // Phase 1: build local histograms with SMEM atomicAdds.
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

    // Phase 2: bulk store our local histograms to each peer's inbox at slot
    // [src=rank]. Each (sender, dst) pair writes to a unique region — no atomics.
    for (int dst = warp_id; dst < kNumRanks; dst += num_warps) {
        auto* peer_e = reinterpret_cast<int*>(
            static_cast<uint8_t*>(buffer_ptrs[dst]) + streaming_section_offset);
        auto* peer_u = peer_e + e_inbox_size;
        // local_e[dst, c, e] -> peer_e[c, src=rank, e]
        for (int i = lane_id; i < slab_e_size; i += 32) {
            int c = i / E;
            int e = i - c * E;
            peer_e[c * kNumRanks * E + rank * E + e] = local_e[dst * slab_e_size + i];
        }
        // local_u[dst, c] -> peer_u[c, src=rank]
        for (int c = lane_id; c < num_channels; c += 32) {
            peer_u[c * kNumRanks + rank] = local_u[dst * slab_u_size + c];
        }
    }
    __syncthreads();

    // Cross-rank barrier: all peers' phase-2 stores observable to me.
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

    // Phase 3: copy our own inbox to output tensors.
    auto* my_e_inbox =
        reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[rank]) + streaming_section_offset);
    auto* my_u_inbox = my_e_inbox + e_inbox_size;
    for (int i = thread_id; i < e_inbox_size; i += num_threads)
        recv_count_out[i] = my_e_inbox[i];
    for (int i = thread_id; i < u_inbox_size; i += num_threads)
        recv_unique_per_source_out[i] = my_u_inbox[i];
}

void streaming_count_exchange(const topk_idx_t* topk_idx,
                              int* recv_count_out,
                              int* recv_unique_per_source_out,
                              int num_tokens,
                              int num_topk,
                              int num_experts_per_rank,
                              int num_channels,
                              int64_t streaming_section_offset,
                              void** buffer_ptrs,
                              int** barrier_signal_ptrs,
                              int rank,
                              int num_ranks,
                              cudaStream_t stream) {
#define STREAMING_COUNT_EXCHANGE_LAUNCH_CASE(ranks)                                          \
    LAUNCH_KERNEL(&cfg,                                                                      \
                  streaming_count_exchange<ranks>,                                           \
                  topk_idx,                                                                  \
                  recv_count_out,                                                             \
                  recv_unique_per_source_out,                                                 \
                  num_tokens,                                                                \
                  num_topk,                                                                  \
                  num_experts_per_rank,                                                      \
                  num_channels,                                                              \
                  streaming_section_offset,                                                  \
                  buffer_ptrs,                                                               \
                  barrier_signal_ptrs,                                                       \
                  rank);                                                                     \
    break

    SETUP_LAUNCH_CONFIG(1, 256, stream);
    // Local histograms in SMEM: per-(dst, c, e) + per-(dst, c).
    int smem_bytes = num_ranks * num_channels * (num_experts_per_rank + 1) * sizeof(int);
    cfg.dynamicSmemBytes = smem_bytes;
    SWITCH_RANKS(STREAMING_COUNT_EXCHANGE_LAUNCH_CASE);
#undef STREAMING_COUNT_EXCHANGE_LAUNCH_CASE
}

// Streaming-MoE pre-compute. Single block; reads recv_count[c, src, e_local] and
// emits the prefix-sum-style metadata derived from the count exchange (see
// new_design.md §"Pre-compute on receiver"). All output shapes are known
// statically from (num_channels, num_ranks, num_experts_per_rank); no padding.
//
// Subsumes the receiver-side metadata that stock notify_dispatch produced: writes
// num_recv (= sum recv_count) and num_recv_per_expert (aligned per expert_alignment)
// to host-mapped slots, fills this rank's column of rank_prefix_matrix, and lands
// total_tiles in both a device int and a host-mapped slot. The dispatch flow polls
// {num_recv, num_recv_per_expert, total_tiles} as a single combined sync.
__global__ void streaming_metadata_init_kernel(const int* __restrict__ recv_count,
                                               const int* __restrict__ recv_unique_per_source,
                                               int* expert_frequency,
                                               int* expert_frequency_offset,
                                               int* base,
                                               int* cumulative_tiles_before_e,
                                               int* per_source_rank_remaining,
                                               int* rank_prefix_matrix,
                                               int* total_tiles_out,
                                               int* num_recv_mapped,
                                               int* num_recv_per_expert_mapped,
                                               int* total_tiles_mapped,
                                               int num_channels,
                                               int num_ranks,
                                               int num_experts_per_rank,
                                               int tile_m,
                                               int expert_alignment,
                                               int my_rank) {
    auto thread_id = static_cast<int>(threadIdx.x);
    auto num_threads = static_cast<int>(blockDim.x);
    int num_csrc = num_channels * num_ranks;
    int E = num_experts_per_rank;

    extern __shared__ int smem[];
    int* s_freq = smem;                       // [E]
    int* s_freq_off = s_freq + E;             // [E + 1]
    int* s_per_src = s_freq_off + E + 1;      // [num_ranks]

    // expert_frequency[e] = sum over (c, src) of recv_count[c, src, e]  (pair counts).
    for (int e = thread_id; e < E; e += num_threads) {
        int sum = 0;
        for (int cs = 0; cs < num_csrc; ++cs)
            sum += recv_count[cs * E + e];
        s_freq[e] = sum;
        expert_frequency[e] = sum;
    }
    // per_source_rank_remaining[c, src] = unique token count for substream (c, src).
    // s_per_src[src] = sum over c (cumulative for rank_prefix_matrix and num_recv).
    for (int src = thread_id; src < num_ranks; src += num_threads) {
        int total = 0;
        for (int c = 0; c < num_channels; ++c) {
            int cs = c * num_ranks + src;
            int u = recv_unique_per_source[cs];
            per_source_rank_remaining[cs] = u;
            total += u;
        }
        s_per_src[src] = total;
    }
    __syncthreads();

    // Sequential scans (E ≤ NUM_MAX_LOCAL_EXPERTS, fits in one thread).
    if (thread_id == 0) {
        s_freq_off[0] = 0;
        int cum_off = 0, cum_tiles = 0;
        cumulative_tiles_before_e[0] = 0;
        for (int e = 0; e < E; ++e) {
            cum_off += s_freq[e];
            s_freq_off[e + 1] = cum_off;
            int tpe = (s_freq[e] + tile_m - 1) / tile_m;
            cum_tiles += tpe;
            cumulative_tiles_before_e[e + 1] = cum_tiles;
        }
        *total_tiles_out = cum_tiles;

        // num_recv = Σ over (c, src) of unique tokens per source (NOT the pair count).
        int total_unique = 0;
        for (int i = 0; i < num_ranks; ++i)
            total_unique += s_per_src[i];

        // num_recv_per_expert[e] aligned up to expert_alignment.
        for (int e = 0; e < E; ++e) {
            int aligned = (s_freq[e] + expert_alignment - 1) / expert_alignment * expert_alignment;
            num_recv_per_expert_mapped[e] = aligned;
        }
        *num_recv_mapped = total_unique;
        *total_tiles_mapped = cum_tiles;

        // rank_prefix_matrix: this rank fills its own column (combine on rank j only
        // reads column j). Cumulative unique tokens from senders 0..i to this rank.
        int cum_src = 0;
        for (int i = 0; i < num_ranks; ++i) {
            cum_src += s_per_src[i];
            rank_prefix_matrix[i * num_ranks + my_rank] = cum_src;
        }
    }
    __syncthreads();
    for (int e = thread_id; e < E + 1; e += num_threads)
        expert_frequency_offset[e] = s_freq_off[e];

    // base[c, src, e] = expert_frequency_offset[e] + Σ over (c', src') < (c, src) lex of recv_count[c', src', e]
    for (int e = thread_id; e < E; e += num_threads) {
        int acc = s_freq_off[e];
        for (int cs = 0; cs < num_csrc; ++cs) {
            base[cs * E + e] = acc;
            acc += recv_count[cs * E + e];
        }
    }
}

// Streaming-MoE post-dispatch slot assignment. One block per (channel, src_rank)
// substream; one warp per block. Each lane handles a deterministic slice of the
// substream's (chunk, k) pairs (idx = lane + 32, +64, ...). Pass 1 counts per-e
// occurrences in lane-private SMEM slots; cross-lane exclusive prefix scan per e
// gives `thread_offset[lane, e]`. Pass 2 walks the substream in **expert-major
// order**: for each expert e in [0, E), each lane scans its stride-32 slice of
// (chunk, k) pairs, processes only those matching e, and assigns each pair its
// slot via `slot = base[c, src, e] + thread_offset[lane, e] + local_seen[lane,
// e]++`. Counters are SMEM-private to each lane (single-writer, no atomics), so
// the slot for a given (chunk, k) is the same on every run with the same input.
// Bit-deterministic by construction.
//
// Expert-major ordering ensures that across all (channel, src) blocks, every
// `tile_remaining[tile_id]` decrement for expert e fires before any decrement
// for expert e+1. Tiles thus fire onto `tile_ready[tile_id]` in expert-monotonic
// order, giving consumers a wave-scheduled view of the firing stream without
// any global cross-block synchronization.
template <int kNumRanks>
__global__ void streaming_slot_assign_kernel(const topk_idx_t* __restrict__ recv_topk_idx,
                                             const int* __restrict__ recv_channel_offset,
                                             const int* __restrict__ rank_prefix_matrix,
                                             const int* __restrict__ per_source_rank_remaining,
                                             const int* __restrict__ base_table,
                                             const int* __restrict__ expert_frequency_offset,
                                             const int* __restrict__ cumulative_tiles_before_e,
                                             const int* __restrict__ substream_ready,
                                             int* tile_records_recv_x_rows,
                                             int* tile_records_k_slots,
                                             int* tile_records_expert_id,
                                             int* tile_remaining,
                                             int64_t* tile_ready,
                                             int my_rank,
                                             int num_channels,
                                             int num_experts_per_rank,
                                             int num_topk,
                                             int tile_m,
                                             int64_t dispatch_seq) {
    const int c = blockIdx.x / kNumRanks;
    const int src = blockIdx.x - c * kNumRanks;
    const int lane = threadIdx.x;
    const int E = num_experts_per_rank;

    // per_source_rank_remaining[c, src] is the unique-token count for substream
    // (c, src) — exactly the substream's row count in recv_x. Doubles as the
    // zero-substream early-out and the iteration bound (no need to read a c+1
    // recv_channel_offset, which would create a cross-substream dependency for
    // streaming).
    int substream_token_count = per_source_rank_remaining[c * kNumRanks + src];
    if (substream_token_count == 0)
        return;

    // Streaming gate: spin until the dispatch main kernel has finished writing
    // this substream's recv_topk_idx. Per-block, per-substream — does not block
    // on any other substream's progress. If substream_ready is nullptr (standalone
    // test path with no dispatch upstream), skip the wait.
    if (substream_ready != nullptr && lane == 0) {
        while (ld_acquire_sys_global(&substream_ready[c * kNumRanks + src]) == 0) {
        }
    }
    __syncwarp();

    int rank_offset = (src > 0) ? rank_prefix_matrix[(src - 1) * kNumRanks + my_rank] : 0;
    int channel_offset = recv_channel_offset[src * num_channels + c];
    const int substream_start = rank_offset + channel_offset;
    const int total_pairs = substream_token_count * num_topk;

    extern __shared__ int s_streaming[];
    int* s_count = s_streaming;                      // [32 * E] -- pass 1 count (preserved)
    int* s_offset = s_streaming + 32 * E;            // [32 * E] -- exclusive prefix (thread_offset)
    int* s_seen = s_streaming + 2 * 32 * E;          // [32 * E] -- per-lane consumed counter

    for (int e = 0; e < E; ++e) {
        s_count[lane * E + e] = 0;
        s_seen[lane * E + e] = 0;
    }
    __syncwarp();

    for (int idx = lane; idx < total_pairs; idx += 32) {
        int chunk = idx / num_topk;
        int k = idx - chunk * num_topk;
        int row = substream_start + chunk;
        int e = static_cast<int>(recv_topk_idx[row * num_topk + k]);
        if (e >= 0 && e < E)
            s_count[lane * E + e]++;
    }
    __syncwarp();

    for (int e = 0; e < E; ++e) {
        int val = s_count[lane * E + e];
        int x = val;
        #pragma unroll
        for (int i = 1; i < 32; i *= 2) {
            int y = __shfl_up_sync(0xffffffff, x, i);
            if (lane >= i)
                x += y;
        }
        s_offset[lane * E + e] = x - val;
    }
    __syncwarp();

    const int csrc_idx = c * kNumRanks + src;
    for (int e = 0; e < E; ++e) {
        // Skip the entire substream scan when this lane has no pairs for expert e.
        // Most lanes touch only a handful of experts → most outer iterations skip.
        if (s_count[lane * E + e] != 0) {
            const int base_e = base_table[csrc_idx * E + e];
            const int thread_offset_e = s_offset[lane * E + e];
            const int feo_e = expert_frequency_offset[e];
            const int ctbe_e = cumulative_tiles_before_e[e];

            for (int idx = lane; idx < total_pairs; idx += 32) {
                int chunk = idx / num_topk;
                int k = idx - chunk * num_topk;
                int row = substream_start + chunk;
                int e_pair = static_cast<int>(recv_topk_idx[row * num_topk + k]);
                if (e_pair == e) {
                    int slot = base_e + thread_offset_e + s_seen[lane * E + e];
                    s_seen[lane * E + e]++;
                    int rel = slot - feo_e;
                    int local_tile_idx = rel / tile_m;
                    int row_in_tile = rel - local_tile_idx * tile_m;
                    int tile_id = ctbe_e + local_tile_idx;
                    int row_idx = tile_id * tile_m + row_in_tile;

                    tile_records_recv_x_rows[row_idx] = row;
                    tile_records_k_slots[row_idx] = k;
                    tile_records_expert_id[tile_id] = e;

                    int rem = atomicSub(tile_remaining + tile_id, 1);
                    if (rem == 1) {
                        memory_fence();
                        st_release_sys_global(tile_ready + tile_id, dispatch_seq);
                    }
                }
            }
        }
        __syncwarp();
    }
}

void streaming_slot_assign(const topk_idx_t* recv_topk_idx,
                           const int* recv_channel_offset,
                           const int* rank_prefix_matrix,
                           const int* base_table,
                           const int* expert_frequency_offset,
                           const int* cumulative_tiles_before_e,
                           const int* substream_ready,
                           int* tile_records_recv_x_rows,
                           int* tile_records_k_slots,
                           int* tile_records_expert_id,
                           int* tile_remaining,
                           int64_t* tile_ready,
                           const int* per_source_rank_remaining,
                           int rank,
                           int num_ranks,
                           int num_channels,
                           int num_experts_per_rank,
                           int num_topk,
                           int tile_m,
                           int64_t dispatch_seq,
                           cudaStream_t stream) {
    const int total_blocks = num_channels * num_ranks;
    const int smem_bytes = 3 * 32 * num_experts_per_rank * static_cast<int>(sizeof(int));

#define STREAMING_SLOT_ASSIGN_LAUNCH_CASE(ranks)                                              \
    streaming_slot_assign_kernel<ranks><<<total_blocks, 32, smem_bytes, stream>>>(            \
        recv_topk_idx, recv_channel_offset, rank_prefix_matrix, per_source_rank_remaining,    \
        base_table, expert_frequency_offset, cumulative_tiles_before_e, substream_ready,      \
        tile_records_recv_x_rows, tile_records_k_slots, tile_records_expert_id,               \
        tile_remaining, tile_ready,                                                           \
        rank, num_channels, num_experts_per_rank, num_topk, tile_m, dispatch_seq);            \
    break

    SWITCH_RANKS(STREAMING_SLOT_ASSIGN_LAUNCH_CASE);
#undef STREAMING_SLOT_ASSIGN_LAUNCH_CASE

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        EPException ex("CUDA", __FILE__, __LINE__, cudaGetErrorString(e));
        fprintf(stderr, "%s\n", ex.what());
        throw ex;
    }
}

void streaming_metadata_init(const int* recv_count,
                             const int* recv_unique_per_source,
                             int* expert_frequency,
                             int* expert_frequency_offset,
                             int* base,
                             int* cumulative_tiles_before_e,
                             int* per_source_rank_remaining,
                             int* rank_prefix_matrix,
                             int* total_tiles_out,
                             int* num_recv_mapped,
                             int* num_recv_per_expert_mapped,
                             int* total_tiles_mapped,
                             int num_channels,
                             int num_ranks,
                             int num_experts_per_rank,
                             int tile_m,
                             int expert_alignment,
                             int my_rank,
                             cudaStream_t stream) {
    int num_threads = 256;
    int E = num_experts_per_rank;
    // SMEM: s_freq[E] + s_freq_off[E+1] + s_per_src[num_ranks]
    int smem_bytes = (E + (E + 1) + num_ranks) * sizeof(int);
    streaming_metadata_init_kernel<<<1, num_threads, smem_bytes, stream>>>(
        recv_count,
        recv_unique_per_source,
        expert_frequency,
        expert_frequency_offset,
        base,
        cumulative_tiles_before_e,
        per_source_rank_remaining,
        rank_prefix_matrix,
        total_tiles_out,
        num_recv_mapped,
        num_recv_per_expert_mapped,
        total_tiles_mapped,
        num_channels,
        num_ranks,
        num_experts_per_rank,
        tile_m,
        expert_alignment,
        my_rank);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        EPException ex("CUDA", __FILE__, __LINE__, cudaGetErrorString(e));
        fprintf(stderr, "%s\n", ex.what());
        throw ex;
    }
}

// Streaming-MoE tile_remaining initialization. tile_remaining[i] is the number of
// (chunk, k) entries that must atomicSub-reach this tile before it fires onto
// tile_ready_queue. Full tiles get tile_m; the last (partial) tile of each expert
// gets the leftover. Linear scan over experts inside each thread (E_local small).
__global__ void tile_remaining_init_kernel(const int* __restrict__ expert_frequency,
                                           const int* __restrict__ cumulative_tiles_before_e,
                                           int* tile_remaining,
                                           int E,
                                           int total_tiles,
                                           int tile_m) {
    int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_id >= total_tiles)
        return;

    int e = 0;
    while (e < E && cumulative_tiles_before_e[e + 1] <= tile_id)
        ++e;
    int e_start = cumulative_tiles_before_e[e];
    int e_end = cumulative_tiles_before_e[e + 1];
    if (tile_id == e_end - 1) {
        int rem = expert_frequency[e] - (e_end - e_start - 1) * tile_m;
        tile_remaining[tile_id] = rem;
    } else {
        tile_remaining[tile_id] = tile_m;
    }
}

void tile_remaining_init(const int* expert_frequency,
                         const int* cumulative_tiles_before_e,
                         int* tile_remaining,
                         int num_experts_per_rank,
                         int total_tiles,
                         int tile_m,
                         cudaStream_t stream) {
    if (total_tiles == 0)
        return;
    int threads = 128;
    int blocks = (total_tiles + threads - 1) / threads;
    tile_remaining_init_kernel<<<blocks, threads, 0, stream>>>(
        expert_frequency, cumulative_tiles_before_e, tile_remaining,
        num_experts_per_rank, total_tiles, tile_m);
}

template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch(int4* recv_x,
                                                           float* recv_x_scales,
                                                           int* recv_src_idx,
                                                           topk_idx_t* recv_topk_idx,
                                                           float* recv_topk_weights,
                                                           int* recv_channel_offset,
                                                           int* send_head,
                                                           const int4* x,
                                                           const float* x_scales,
                                                           const topk_idx_t* topk_idx,
                                                           const float* topk_weights,
                                                           const bool* is_token_in_rank,
                                                           const int* channel_prefix_matrix,
                                                           int num_tokens,
                                                           int num_worst_tokens,
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
                                                           int* substream_ready) {
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
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
    EP_DEVICE_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

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

        // Send offset by `-value - 1`, e.g. 0 -> -1, 1 -> -2
        // NOTES: this is for distinguishing zero tokens
        if (send_warp_id_in_rank == 0 and elect_one_sync()) {
            int value = responsible_channel > 0 ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1] : 0;
            st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1);
            value = channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
            st_relaxed_sys_global(channel_end_offset.buffer(), -value - 1);
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
        // Workers for receiving and copying into buffer
        constexpr int num_recv_warps = kNumThreads / 32;
        constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
        const auto recv_thread_id = thread_id;
        const auto recv_thread_id_in_rank = recv_thread_id % num_threads_per_rank;
        const auto recv_warp_id_in_rank = recv_thread_id_in_rank / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32);
        EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);

        // Calculate offset first
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank] : 0;

        // Receive channel offset
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
        total_offset += rank_offset;
        num_tokens_to_recv = __shfl_sync(0xffffffff, num_tokens_to_recv, 0);

        // Shared tail indices for different warps
        __shared__ volatile int shared_channel_tail_idx[kNumRanks];

        auto start_time = clock64();
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            // NOTES: unlike the sender, the receiver must ensure that the tail indices hold by different warps are the same
            while (recv_thread_id_in_rank == 0) {
                cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());

                // Ready to copy
                if (cached_channel_head_idx != cached_channel_tail_idx) {
                    shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
                    break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP timeout for dispatch receivers, rank %d, responsible_channel = %d, tokens remained: %d\n",
                           rank,
                           responsible_channel,
                           num_tokens_to_recv);
                    trap();
                }
            }

            // Synchronize queue tail
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens; chunk_idx += num_recv_warps_per_rank) {
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                auto shifted_buffer_x_int4 = channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;
                auto shifted_recv_x_int4 = recv_x + static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;
#ifndef DISABLE_SM90_FEATURES
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    tma_store_wait<0>();
                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer, shifted_buffer_x_int4 + i * half_hidden_int4, tma_mbarrier, half_hidden_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, half_hidden_bytes);
                        mbarrier_wait(tma_mbarrier, tma_phase);
                        tma_store_1d(tma_buffer, shifted_recv_x_int4 + i * half_hidden_int4, half_hidden_bytes, false);
                    }
                }
                __syncwarp();
#else
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_recv_x_int4, shifted_buffer_x_int4, ld_nc_global, st_na_global);
#endif
            }

            // Copy `src_idx`
            #pragma unroll 4
            for (int chunk_idx = cached_channel_head_idx + recv_thread_id_in_rank; chunk_idx < cached_channel_tail_idx;
                 chunk_idx += 32 * num_recv_warps_per_rank)
                recv_src_idx[total_offset + chunk_idx - cached_channel_head_idx] =
                    ld_nc_global(channel_src_idx_buffers.buffer() + chunk_idx % num_recv_buffer_tokens);

            // Copy `topk_idx` and `topk_weights`
            #pragma unroll 4
            for (int idx = recv_thread_id_in_rank; idx < num_recv_tokens * num_topk; idx += 32 * num_recv_warps_per_rank) {
                int chunk_idx = idx / num_topk, token_topk_idx = idx % num_topk;
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                auto recv_idx = static_cast<int64_t>(total_offset + chunk_idx) * num_topk + token_topk_idx;
                auto buffer_idx = token_idx_in_buffer * num_topk + token_topk_idx;
                auto e_val = ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);
                recv_topk_idx[recv_idx] = e_val;
                recv_topk_weights[recv_idx] = ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);
            }

            // Copy `x_scales`
            #pragma unroll 4
            for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_scales; i += 32 * num_recv_warps_per_rank) {
                int chunk_idx = i / num_scales, scales_idx = i % num_scales;
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                recv_x_scales[static_cast<int64_t>(total_offset + chunk_idx) * num_scales + scales_idx] =
                    ld_nc_global(channel_x_scales_buffers.buffer() + token_idx_in_buffer * num_scales + scales_idx);
            }

            // Move queue
            cached_channel_head_idx += num_recv_tokens;
            total_offset += num_recv_tokens;
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 and elect_one_sync())
                st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx);

            // Exit
            num_tokens_to_recv -= num_recv_tokens;
        }

        // Streaming-MoE: this thread group has finished writing recv_topk_idx (and
        // recv_x, recv_src_idx, ...) for substream (responsible_channel, responsible_rank).
        // Release-store substream_ready[c, s] so streaming_slot_assign block (c, s) can
        // start its pass-2 work without waiting for the rest of the dispatch kernel.
#ifndef DISABLE_SM90_FEATURES
        tma_store_wait<0>();
#endif
        asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
        if (recv_thread_id_in_rank == 0 and substream_ready != nullptr) {
            __threadfence_system();
            st_release_sys_global(&substream_ready[responsible_channel * kNumRanks + responsible_rank], 1);
        }
    }

    // Clean unused `recv_topk_idx` as -1
    if (num_worst_tokens > 0) {
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        const auto num_recv_tokens = rank_prefix_matrix[(kNumRanks - 1) * kNumRanks + rank];
        const auto clean_start = num_recv_tokens * num_topk + sm_id * kNumThreads;
        const auto clean_end = num_worst_tokens * num_topk;
        const auto clean_stride = num_sms * kNumThreads;
        #pragma unroll
        for (int i = clean_start + thread_id; i < clean_end; i += clean_stride)
            recv_topk_idx[i] = -1;
    }
}

void dispatch(void* recv_x,
              float* recv_x_scales,
              int* recv_src_idx,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              int* recv_channel_offset,
              int* send_head,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              const bool* is_token_in_rank,
              const int* channel_prefix_matrix,
              int num_tokens,
              int num_worst_tokens,
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
              int* substream_ready) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 8192;
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
#endif

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(ranks)                                      \
    {                                                                    \
        auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                               \
        LAUNCH_KERNEL(&cfg,                                              \
                      kernel,                                            \
                      reinterpret_cast<int4*>(recv_x),                   \
                      recv_x_scales,                                     \
                      recv_src_idx,                                      \
                      recv_topk_idx,                                     \
                      recv_topk_weights,                                 \
                      recv_channel_offset,                               \
                      send_head,                                         \
                      reinterpret_cast<const int4*>(x),                  \
                      x_scales,                                          \
                      topk_idx,                                          \
                      topk_weights,                                      \
                      is_token_in_rank,                                  \
                      channel_prefix_matrix,                             \
                      num_tokens,                                        \
                      num_worst_tokens,                                  \
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
                      substream_ready);                                  \
    }                                                                    \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
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
