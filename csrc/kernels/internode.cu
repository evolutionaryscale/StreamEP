#include <functional>
#include <optional>

#include "api.cuh"
#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "ibgda_device.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace stream_ep {

namespace internode {

extern nvshmem_team_t cpu_rdma_team;

struct SourceMeta {
    int src_rdma_rank, is_token_in_nvl_rank_bits;

    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

    __forceinline__ SourceMeta() = default;

    // TODO: faster encoding
    __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
        src_rdma_rank = rdma_rank;
        is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
        #pragma unroll
        for (int i = 1; i < NUM_MAX_NVL_PEERS; ++i)
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
    }

    __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const { return (is_token_in_nvl_rank_bits >> nvl_rank) & 1; }
};

EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

int get_source_meta_bytes() {
    return sizeof(SourceMeta);
}

// `get_num_bytes_per_token`, `get_rdma_clean_meta`, `get_nvl_clean_meta` are
// declared as inline __host__ __device__ helpers in api.cuh so deep_ep.cpp
// can reuse them when sizing the metadata kernel's cleanup region.

template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

template <bool kLowLatencyMode>
__forceinline__ __device__ void nvshmem_sync_with_same_gpu_idx(const nvshmem_team_t& rdma_team) {
    kLowLatencyMode ? void(nvshmem_sync(rdma_team)) : nvshmem_sync_all();
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming-MoE consolidated dispatch metadata for internode (RDMA + NVL).
//
// Folded single-kernel architecture mirroring `intranode::streaming_dispatch_metadata`'s
// shape phase-for-phase. Performs the cross-rank (token, k) count exchange
// + channel-prefix-matrix derivation + streaming-superset emission (pool-
// shape outputs the dispatch hot path consumes, everything sized by
// E_local / num_world_ranks / num_channels / total_tiles, all known before
// the host poll on `pool[T_recv, hidden]`).
//
// Block layout: (1 + kNumRDMARanks, kNumThreads):
//   - Block 0: NVSHMEM cleanup + cross-rank handshake, local histograms from
//     topk_idx, RDMA put + NVL peer writes, cross-rank barrier, derive
//     metadata + write host-mapped counters + channel matrices, then the
//     streaming-superset phases (build seen_per_substream / base_pool /
//     expert_frequency / expert_pool_block_offset / rank_prefix_matrix /
//     tile_id_to_expert / pool_arrival_target / streaming_total_tiles).
//   - Blocks 1..kNumRDMARanks: per-(dst_rdma_rank) channel prefix matrix
//     calculation, derives is_token_in_rank locally from topk_idx.
//
// Streaming SymBuffer layout (per (src_world → dst_rdma) slab):
//     int32[num_channels][NUM_MAX_NVL_PEERS][E_local]
//   RDMA pairs same-NVL-slot ranks (dst rank == (dst_rdma, src_nvl)), so
//   each receiver's recv buffer holds one slab per (src_rdma, src_nvl=this_rank's_nvl).
//   The src_nvl axis is filled by the NVL-aggregation phase that follows
//   (each NVL rank within dst_rdma extracts its `dst_nvl` slice from the
//   RDMA-received slabs and propagates to its 7 NVL peers via direct
//   buffer_ptrs writes — same RDMA-then-NVL two-hop count-aggregation
//   pattern used for the per-rank/per-expert recv counters).
template <bool kLowLatencyMode, int kNumRDMARanks>
__global__ void streaming_dispatch_metadata_kernel(
        const topk_idx_t* topk_idx,
        // Counters (host-mapped, written)
        int* moe_recv_counter_mapped,
        int* moe_recv_rdma_counter_mapped,
        int* moe_recv_expert_counter_mapped,
        int* streaming_total_tiles_mapped,
        // Channel prefix matrices (sender-side cumulative)
        int* rdma_channel_prefix_matrix,
        int* recv_rdma_rank_prefix_sum,
        int* gbl_channel_prefix_matrix,
        int* recv_gbl_rank_prefix_sum,
        // Streaming-superset outputs
        int* expert_frequency,
        int* expert_pool_block_offset,
        int* base_pool,
        int* seen_per_substream,
        int* rank_prefix_matrix,
        int* tile_id_to_expert,
        int* pool_arrival_target,
        int* total_tiles_device,
        // Shape
        int num_tokens,
        int num_topk,
        int num_experts,
        int num_channels,
        int expert_alignment,
        int tile_m,
        // Cleanup
        int rdma_clean_offset,
        int rdma_num_int_clean,
        int nvl_clean_offset,
        int nvl_num_int_clean,
        // Streaming SymBuffer offset within rdma_buffer_ptr (post notify_dispatch's payload)
        int64_t streaming_rdma_offset,
        // Env
        void* rdma_buffer_ptr,
        void** buffer_ptrs,
        int** barrier_signal_ptrs,
        int rank,
        const nvshmem_team_t rdma_team) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);
    auto warp_id = thread_id / 32;
    auto lane_id = get_lane_id();
    auto num_threads = static_cast<int>(blockDim.x);
    auto num_warps = num_threads / 32;

    auto rdma_rank = rank / NUM_MAX_NVL_PEERS;
    auto nvl_rank = rank % NUM_MAX_NVL_PEERS;
    const int num_ranks = kNumRDMARanks * NUM_MAX_NVL_PEERS;
    const int E_local = num_experts / num_ranks;
    const int num_rdma_experts = num_experts / kNumRDMARanks;
    EP_DEVICE_ASSERT(E_local <= NUM_MAX_LOCAL_EXPERTS);

    // Layout the streaming SymBuffer: per (src_world → dst_rdma) slab of
    //   [num_channels][NUM_MAX_NVL_PEERS][E_local] int32.
    const int kStreamSlabInts = num_channels * NUM_MAX_NVL_PEERS * E_local;
    void* streaming_rdma_base = static_cast<uint8_t*>(rdma_buffer_ptr) + streaming_rdma_offset;
    auto streaming_rdma_recv = SymBuffer<int>(streaming_rdma_base, kStreamSlabInts, kNumRDMARanks);

    // ─────────────────────────────────────────────────────────────────────
    // Blocks 1..kNumRDMARanks: per-(dst_rdma_rank) channel prefix matrix
    // (sender-side). Computes `gbl_channel_prefix_matrix[dst_world, c]` and
    // `rdma_channel_prefix_matrix[dst_rdma, c]` from THIS rank's topk_idx
    // (the routing-bitmap derivation that get_dispatch_layout used to
    // perform host-side; folded into this kernel now).
    // ─────────────────────────────────────────────────────────────────────
    if (sm_id != 0) {
        int dst_rdma_rank = sm_id - 1;
        int num_tokens_per_channel = (num_tokens + num_channels - 1) / num_channels;
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int t_start = channel_id * num_tokens_per_channel;
            int t_end = min(t_start + num_tokens_per_channel, num_tokens);

            int total_count = 0;
            int per_nvl_count[NUM_MAX_NVL_PEERS] = {0};
            for (int t = t_start + lane_id; t < t_end; t += 32) {
                // Derive per-(dst_nvl) routing bitmap for this token by
                // walking topk_idx and bucketing (any k → dst_world ↦ bit).
                int per_nvl[NUM_MAX_NVL_PEERS] = {0};
                #pragma unroll
                for (int k = 0; k < num_topk; ++k) {
                    int e_global = static_cast<int>(__ldg(topk_idx + static_cast<int64_t>(t) * num_topk + k));
                    if (e_global < 0) continue;
                    int dst_world = e_global / E_local;
                    if (dst_world / NUM_MAX_NVL_PEERS != dst_rdma_rank) continue;
                    int dst_nvl = dst_world - dst_rdma_rank * NUM_MAX_NVL_PEERS;
                    per_nvl[dst_nvl] = 1;  // any-k OR
                }
                int any_in_rdma = 0;
                #pragma unroll
                for (int n = 0; n < NUM_MAX_NVL_PEERS; ++n) {
                    per_nvl_count[n] += per_nvl[n];
                    any_in_rdma |= per_nvl[n];
                }
                total_count += any_in_rdma;
            }

            total_count = warp_reduce_sum(total_count);
            #pragma unroll
            for (int n = 0; n < NUM_MAX_NVL_PEERS; ++n)
                per_nvl_count[n] = warp_reduce_sum(per_nvl_count[n]);

            if (elect_one_sync()) {
                #pragma unroll
                for (int n = 0; n < NUM_MAX_NVL_PEERS; ++n)
                    gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + n) * num_channels + channel_id] = per_nvl_count[n];
                rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] = total_count;
            }
        }

        __syncthreads();
        if (thread_id == 0) {
            auto prefix_row = rdma_channel_prefix_matrix + dst_rdma_rank * num_channels;
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                prefix_row[i] += prefix_row[i - 1];
        }
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        if (thread_id < NUM_MAX_NVL_PEERS) {
            auto prefix_row = gbl_channel_prefix_matrix + (dst_rdma_rank * NUM_MAX_NVL_PEERS + thread_id) * num_channels;
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                prefix_row[i] += prefix_row[i - 1];
        }
        return;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Block 0: cross-rank exchange + counters + streaming superset.
    // ─────────────────────────────────────────────────────────────────────

    EP_DEVICE_ASSERT(num_warps > 1);
    EP_DEVICE_ASSERT(kNumRDMARanks <= num_threads);
    EP_DEVICE_ASSERT(num_rdma_experts <= num_threads);
    EP_DEVICE_ASSERT(NUM_MAX_NVL_PEERS <= num_threads);
    EP_DEVICE_ASSERT(E_local <= num_threads);

    // SMEM layout:
    //   smem_streaming_hist [kNumRDMARanks * kStreamSlabInts] int32
    //     = streaming-superset histogram per (dst_rdma, c, dst_nvl, e_local).
    //   smem_local_per_rank        [num_ranks]      int32 (deduped per token)
    //   smem_local_per_rdma_rank   [kNumRDMARanks]  int32 (deduped per token)
    //   smem_local_per_expert      [num_experts]    int32 ((token, k) pairs)
    extern __shared__ int smem_buf[];
    int* smem_streaming_hist     = smem_buf;
    int* smem_local_per_rank     = smem_streaming_hist + kNumRDMARanks * kStreamSlabInts;
    int* smem_local_per_rdma     = smem_local_per_rank + num_ranks;
    int* smem_local_per_expert   = smem_local_per_rdma + kNumRDMARanks;

    // ── Phase A1: zero local histograms.
    for (int i = thread_id; i < kNumRDMARanks * kStreamSlabInts; i += num_threads)
        smem_streaming_hist[i] = 0;
    for (int i = thread_id; i < num_ranks; i += num_threads)
        smem_local_per_rank[i] = 0;
    for (int i = thread_id; i < kNumRDMARanks; i += num_threads)
        smem_local_per_rdma[i] = 0;
    for (int i = thread_id; i < num_experts; i += num_threads)
        smem_local_per_expert[i] = 0;
    __syncthreads();

    // ── Phase A2: build local histograms from topk_idx.
    // Per (token, k): increment streaming_hist (count (token, k) pairs);
    // dedupe per-token via uint64 register bitmask for per-rank /
    // per-rdma-rank unique counts. Mirrors intranode metadata Phase 3.
    int num_tokens_per_channel_var = (num_tokens + num_channels - 1) / num_channels;
    EP_DEVICE_ASSERT(num_ranks <= 64);
    EP_DEVICE_ASSERT(kNumRDMARanks <= 64);
    for (int t = thread_id; t < num_tokens; t += num_threads) {
        int channel_id = t / num_tokens_per_channel_var;
        if (channel_id >= num_channels) channel_id = num_channels - 1;
        uint64_t world_mask = 0, rdma_mask = 0;
        #pragma unroll
        for (int k = 0; k < num_topk; ++k) {
            int e_global = static_cast<int>(__ldg(topk_idx + static_cast<int64_t>(t) * num_topk + k));
            if (e_global < 0) continue;
            int dst_world = e_global / E_local;
            int dst_rdma = dst_world / NUM_MAX_NVL_PEERS;
            int dst_nvl  = dst_world - dst_rdma * NUM_MAX_NVL_PEERS;
            int e_local  = e_global - dst_world * E_local;

            // Streaming-superset histogram: per (dst_rdma, c, dst_nvl, e_local).
            atomicAdd(&smem_streaming_hist[((dst_rdma * num_channels + channel_id) * NUM_MAX_NVL_PEERS + dst_nvl) * E_local + e_local], 1);

            // (token, k)-pair count per expert.
            atomicAdd(&smem_local_per_expert[e_global], 1);

            // Deduped per-rank / per-rdma-rank unique-token counts.
            uint64_t world_bit = 1ULL << dst_world;
            if (!(world_mask & world_bit)) {
                world_mask |= world_bit;
                atomicAdd(&smem_local_per_rank[dst_world], 1);
            }
            uint64_t rdma_bit = 1ULL << dst_rdma;
            if (!(rdma_mask & rdma_bit)) {
                rdma_mask |= rdma_bit;
                atomicAdd(&smem_local_per_rdma[dst_rdma], 1);
            }
        }
    }
    __syncthreads();

    // ── Phase A3: NVSHMEM cleanup (drain prior-iter WRs) + intra-NVL barrier.
    {
        auto qps_per_rdma_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;
        for (int i = thread_id; i < qps_per_rdma_rank * (kNumRDMARanks - 1); i += num_threads) {
            auto dst_rdma_rank = (i / qps_per_rdma_rank + rdma_rank + 1) % kNumRDMARanks;
            auto qp_id = i % qps_per_rdma_rank;
            nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), qp_id);
        }
        __syncthreads();

        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        barrier_block<NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);
    }

    // ── Phase A4: build + send notify_dispatch's count payload via RDMA.
    auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
    auto rdma_recv_num_tokens_mixed = SymBuffer<int>(rdma_buffer_ptr,
        NUM_MAX_NVL_PEERS + num_rdma_experts + 1, kNumRDMARanks);

    // Cleanup the inter-iter RDMA scratch region.
    EP_DEVICE_ASSERT(rdma_recv_num_tokens_mixed.total_bytes <= rdma_clean_offset * sizeof(int));
    #pragma unroll
    for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
        rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;

    // Build per-dst_rdma payload from SMEM histograms.
    for (int i = thread_id; i < num_ranks; i += num_threads)
        rdma_recv_num_tokens_mixed.send_buffer(i / NUM_MAX_NVL_PEERS)[i % NUM_MAX_NVL_PEERS] = smem_local_per_rank[i];
    for (int i = thread_id; i < num_experts; i += num_threads)
        rdma_recv_num_tokens_mixed.send_buffer(i / num_rdma_experts)[NUM_MAX_NVL_PEERS + i % num_rdma_experts] =
            smem_local_per_expert[i];
    if (thread_id < kNumRDMARanks)
        rdma_recv_num_tokens_mixed.send_buffer(thread_id)[NUM_MAX_NVL_PEERS + num_rdma_experts] =
            smem_local_per_rdma[thread_id];

    // Build per-dst_rdma streaming slab in send_buffer (block-stride copy).
    for (int d = 0; d < kNumRDMARanks; ++d) {
        int* dst = streaming_rdma_recv.send_buffer(d);
        int* src = smem_streaming_hist + d * kStreamSlabInts;
        for (int i = thread_id; i < kStreamSlabInts; i += num_threads)
            dst[i] = src[i];
    }
    __syncthreads();

    // Issue RDMA puts (count payload + streaming slab) per dst_rdma_rank.
    for (int i = warp_id; i < kNumRDMARanks; i += num_warps) {
        if (i != rdma_rank) {
            nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank)),
                                              reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.send_buffer(i)),
                                              (NUM_MAX_NVL_PEERS + num_rdma_experts + 1) * sizeof(int),
                                              translate_dst_rdma_rank<kLowLatencyMode>(i, nvl_rank),
                                              0, lane_id, 0);
            nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(streaming_rdma_recv.recv_buffer(rdma_rank)),
                                              reinterpret_cast<uint64_t>(streaming_rdma_recv.send_buffer(i)),
                                              kStreamSlabInts * sizeof(int),
                                              translate_dst_rdma_rank<kLowLatencyMode>(i, nvl_rank),
                                              0, lane_id, 0);
        } else {
            UNROLLED_WARP_COPY(1, lane_id, NUM_MAX_NVL_PEERS + num_rdma_experts + 1,
                               rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank),
                               rdma_recv_num_tokens_mixed.send_buffer(i),
                               ld_volatile_global, st_na_global);
            UNROLLED_WARP_COPY(1, lane_id, kStreamSlabInts,
                               streaming_rdma_recv.recv_buffer(rdma_rank),
                               streaming_rdma_recv.send_buffer(i),
                               ld_volatile_global, st_na_global);
        }
    }
    __syncthreads();

    // Wait for in-flight WRs + cross-RDMA-team sync.
    if (thread_id < kNumRDMARanks and thread_id != rdma_rank)
        nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(thread_id, nvl_rank), 0);
    __syncthreads();
    if (thread_id == 0)
        nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
    __syncthreads();

    // ── Phase A5: NVL aggregation of count payload.
    auto nvl_send_buffer = thread_id < NUM_MAX_NVL_PEERS ? buffer_ptrs[thread_id] : nullptr;
    auto nvl_recv_buffer = buffer_ptrs[nvl_rank];
    auto nvl_reduced_num_tokens_per_expert = Buffer<int>(nvl_recv_buffer, num_rdma_experts).advance_also(nvl_send_buffer);
    auto nvl_send_num_tokens_per_rank = AsymBuffer<int>(nvl_send_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
    auto nvl_send_num_tokens_per_expert = AsymBuffer<int>(nvl_send_buffer, E_local, NUM_MAX_NVL_PEERS);
    auto nvl_recv_num_tokens_per_rank = AsymBuffer<int>(nvl_recv_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
    auto nvl_recv_num_tokens_per_expert = AsymBuffer<int>(nvl_recv_buffer, E_local, NUM_MAX_NVL_PEERS);

    auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
    EP_DEVICE_ASSERT(nvl_reduced_num_tokens_per_expert.total_bytes + nvl_send_num_tokens_per_rank.total_bytes +
                         nvl_send_num_tokens_per_expert.total_bytes <=
                     nvl_clean_offset * sizeof(int));
    #pragma unroll
    for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
        nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;

    // Reduce per-expert tokens received (sum over kNumRDMARanks senders).
    if (thread_id < num_rdma_experts) {
        int sum = 0;
        #pragma unroll
        for (int i = 0; i < kNumRDMARanks; ++i)
            sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + thread_id];
        nvl_reduced_num_tokens_per_expert[thread_id] = sum;
    }
    __syncthreads();

    // Reduce per-RDMA-rank received tokens → moe_recv_rdma_counter + recv_rdma_rank_prefix_sum.
    if (thread_id == 0) {
        int sum = 0;
        #pragma unroll
        for (int i = 0; i < kNumRDMARanks; ++i) {
            sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + num_rdma_experts];
            recv_rdma_rank_prefix_sum[i] = sum;
        }
        while (ld_volatile_global(moe_recv_rdma_counter_mapped) != -1)
            ;
        *moe_recv_rdma_counter_mapped = sum;
    }

    // Each NVL peer (lane=peer) writes its "for me" counts into peer's NVL buffer.
    if (thread_id < NUM_MAX_NVL_PEERS) {
        #pragma unroll
        for (int i = 0; i < kNumRDMARanks; ++i)
            nvl_send_num_tokens_per_rank.buffer(nvl_rank)[i] = rdma_recv_num_tokens_mixed.recv_buffer(i)[thread_id];
        #pragma unroll
        for (int i = 0; i < E_local; ++i)
            nvl_send_num_tokens_per_expert.buffer(nvl_rank)[i] = nvl_reduced_num_tokens_per_expert[thread_id * E_local + i];
    }
    barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);

    // Reduce → moe_recv_counter + recv_gbl_rank_prefix_sum.
    if (thread_id == 0) {
        int sum = 0;
        #pragma unroll
        for (int i = 0; i < num_ranks; ++i) {
            int src_rdma = i / NUM_MAX_NVL_PEERS, src_nvl = i % NUM_MAX_NVL_PEERS;
            sum += nvl_recv_num_tokens_per_rank.buffer(src_nvl)[src_rdma];
            recv_gbl_rank_prefix_sum[i] = sum;
        }
        while (ld_volatile_global(moe_recv_counter_mapped) != -1)
            ;
        *moe_recv_counter_mapped = sum;
    }

    // Per-local-expert reductions → moe_recv_expert_counter (with alignment).
    if (thread_id < E_local) {
        int sum = 0;
        #pragma unroll
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
            sum += nvl_recv_num_tokens_per_expert.buffer(i)[thread_id];
        sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
        while (ld_volatile_global(moe_recv_expert_counter_mapped + thread_id) != -1)
            ;
        moe_recv_expert_counter_mapped[thread_id] = sum;
    }

    // Final NVL barrier — counters all observable.
    if (thread_id == 32)
        nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
    barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);

    // ─────────────────────────────────────────────────────────────────────
    // Phase B: streaming-superset.
    //
    // After Phase A, this rank's streaming_rdma_recv.recv_buffer(s_rdma)
    // holds the slab from sender (s_rdma, src_nvl=this_nvl_rank), shape
    // [num_channels][NUM_MAX_NVL_PEERS][E_local]. The dst_nvl axis spans
    // contributions to all 8 NVL peers at this RDMA rank — only the slice
    // dst_nvl == this rank's nvl_rank is for THIS rank.
    //
    // We need full per-(c, src_world, e_local) where src_world spans all
    // num_world_ranks. The src_nvl axis is filled by an NVL exchange:
    // each NVL rank reads its RDMA-received slabs, extracts the
    // dst_nvl=peer's_nvl slice, and writes to peer's NVL buffer. After the
    // exchange, each NVL rank has [s_rdma][src_nvl_in_rdma][c][e_local]
    // contributions for itself.
    // ─────────────────────────────────────────────────────────────────────

    // ── NVL exchange: propagate the src_nvl axis among the 8 NVL peers.
    //
    // Layout the per-rank NVL slab: per writer (src_nvl) slot of
    //   [kNumRDMARanks][num_channels][E_local] int32.
    // (Mirrors notify_dispatch's nvl_send/recv pattern: writer w writes to
    //  peer m's region at slot indexed by w; receiver reads its own region
    //  at slot=src_nvl.)
    const int kNvlSlotInts = kNumRDMARanks * num_channels * E_local;
    auto nvl_streaming_send = AsymBuffer<int>(nvl_send_buffer, kNvlSlotInts, NUM_MAX_NVL_PEERS);
    auto nvl_streaming_recv = AsymBuffer<int>(nvl_recv_buffer, kNvlSlotInts, NUM_MAX_NVL_PEERS);

    EP_DEVICE_ASSERT(nvl_reduced_num_tokens_per_expert.total_bytes + nvl_send_num_tokens_per_rank.total_bytes +
                         nvl_send_num_tokens_per_expert.total_bytes + nvl_streaming_send.total_bytes <=
                     nvl_clean_offset * sizeof(int));

    // Compute peer m's nvl_streaming region byte offset (same offset in
    // every peer's NVL buffer, IPC peers are mirror-allocated). nvl_streaming_recv
    // is constructed against THIS rank's NVL buffer, so the offset is
    // recv.buffer(0) - buffer_ptrs[nvl_rank]. Same value across all threads.
    const int64_t nvl_streaming_offset_bytes =
        reinterpret_cast<uint8_t*>(nvl_streaming_recv.buffer(0)) -
        reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]);

    // Each warp dispatches to ONE peer m; lanes parallelize the
    // (src_rdma, c, e) inner loop. Writer slot at peer m is indexed by
    // this rank's nvl_rank.
    for (int m = warp_id; m < NUM_MAX_NVL_PEERS; m += num_warps) {
        int* peer_streaming_base = reinterpret_cast<int*>(
            static_cast<uint8_t*>(buffer_ptrs[m]) + nvl_streaming_offset_bytes);
        int* peer_writer_slot = peer_streaming_base + nvl_rank * kNvlSlotInts;
        for (int idx = lane_id; idx < kNumRDMARanks * num_channels * E_local; idx += 32) {
            int src_rdma = idx / (num_channels * E_local);
            int rem = idx - src_rdma * (num_channels * E_local);
            int c = rem / E_local;
            int e = rem - c * E_local;
            int v = ld_nc_global(streaming_rdma_recv.recv_buffer(src_rdma)
                                 + (c * NUM_MAX_NVL_PEERS + m) * E_local + e);
            st_na_global(peer_writer_slot + idx, v);
        }
    }
    __syncthreads();
    barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);

    // ── Build seen_per_substream from this rank's NVL recv buffer.
    // After exchange, this rank's nvl_streaming_recv has NUM_MAX_NVL_PEERS
    // slots (one per writer src_nvl). Slot[w] = [src_rdma][c][e_local]
    // contributions from sender (src_rdma, src_nvl=w) going to this rank.
    //
    // seen_per_substream[c, src_world, e_local] where src_world = src_rdma*8+w.
    for (int idx = thread_id; idx < num_channels * num_ranks * E_local; idx += num_threads) {
        int c = idx / (num_ranks * E_local);
        int rem = idx - c * (num_ranks * E_local);
        int src_world = rem / E_local;
        int e = rem - src_world * E_local;
        int src_rdma = src_world / NUM_MAX_NVL_PEERS;
        int src_nvl = src_world - src_rdma * NUM_MAX_NVL_PEERS;
        int* slot_w = nvl_streaming_recv.buffer(src_nvl);
        int v = ld_volatile_global(slot_w + (src_rdma * num_channels + c) * E_local + e);
        seen_per_substream[idx] = v;
    }
    __syncthreads();

    // ── Phase B3: derive expert_frequency, expert_pool_block_offset,
    // base_pool, rank_prefix_matrix, total_tiles.
    int* smem_freq = smem_buf;                 // reuse scratch
    int* smem_pool_blk = smem_freq + E_local;
    // (smem_buf is already past use of streaming_hist; safe to reuse.)
    __syncthreads();

    // expert_frequency[e] = sum over (c, src_world) of seen_per_substream[c, src_world, e].
    for (int e = thread_id; e < E_local; e += num_threads) {
        int sum = 0;
        for (int cs = 0; cs < num_channels * num_ranks; ++cs)
            sum += seen_per_substream[cs * E_local + e];
        smem_freq[e] = sum;
        expert_frequency[e] = sum;
    }
    __syncthreads();

    if (thread_id == 0) {
        int cum_blocks = 0;
        smem_pool_blk[0] = 0;
        for (int e = 0; e < E_local; ++e) {
            int n_blocks = (smem_freq[e] + tile_m - 1) / tile_m;
            cum_blocks += n_blocks;
            smem_pool_blk[e + 1] = cum_blocks;
        }
        *total_tiles_device = cum_blocks;
        *streaming_total_tiles_mapped = cum_blocks;

        // rank_prefix_matrix[i, rank]: cumulative unique tokens from senders 0..i.
        // (NUM_NVL × kNumRDMARanks ranks, all in the same world view.)
        int cum = 0;
        for (int i = 0; i < num_ranks; ++i) {
            cum = recv_gbl_rank_prefix_sum[i];
            rank_prefix_matrix[i * num_ranks + rank] = cum;
        }
    }
    __syncthreads();
    for (int e = thread_id; e < E_local + 1; e += num_threads)
        expert_pool_block_offset[e] = smem_pool_blk[e];

    // base_pool[c, src_world, e] = expert_pool_block_offset[e]*tile_m
    //   + Σ over (c', s') < (c, s) lex of seen_per_substream[c', s', e].
    // Per-expert serial accumulator (E_local ≤ NUM_MAX_LOCAL_EXPERTS so
    // we can run one thread per expert in parallel).
    for (int e = thread_id; e < E_local; e += num_threads) {
        int acc = smem_pool_blk[e] * tile_m;
        for (int cs = 0; cs < num_channels * num_ranks; ++cs) {
            base_pool[cs * E_local + e] = acc;
            acc += seen_per_substream[cs * E_local + e];
        }
    }

    // tile_id_to_expert + pool_arrival_target.
    for (int e = thread_id; e < E_local; e += num_threads) {
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
                                 int* moe_recv_counter_mapped,
                                 int* moe_recv_rdma_counter_mapped,
                                 int* moe_recv_expert_counter_mapped,
                                 int* streaming_total_tiles_mapped,
                                 int* rdma_channel_prefix_matrix,
                                 int* recv_rdma_rank_prefix_sum,
                                 int* gbl_channel_prefix_matrix,
                                 int* recv_gbl_rank_prefix_sum,
                                 int* expert_frequency,
                                 int* expert_pool_block_offset,
                                 int* base_pool,
                                 int* seen_per_substream,
                                 int* rank_prefix_matrix,
                                 int* tile_id_to_expert,
                                 int* pool_arrival_target,
                                 int* total_tiles_device,
                                 int num_tokens,
                                 int num_topk,
                                 int num_experts,
                                 int num_channels,
                                 int hidden_int4,
                                 int expert_alignment,
                                 int tile_m,
                                 int rdma_clean_offset,
                                 int rdma_num_int_clean,
                                 int nvl_clean_offset,
                                 int nvl_num_int_clean,
                                 int64_t streaming_rdma_offset,
                                 void* rdma_buffer_ptr,
                                 void** buffer_ptrs,
                                 int** barrier_signal_ptrs,
                                 int rank,
                                 int num_ranks,
                                 cudaStream_t stream,
                                 int64_t num_rdma_bytes,
                                 int64_t num_nvl_bytes) {
    constexpr int kNumThreads = 512;
    const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    const int E_local = num_experts / num_ranks;
    const int kStreamSlabInts = num_channels * NUM_MAX_NVL_PEERS * E_local;

    // SMEM = streaming_hist + per_rank + per_rdma + per_expert.
    int smem_bytes =
        (num_rdma_ranks * kStreamSlabInts + num_ranks + num_rdma_ranks + num_experts) * sizeof(int);

#define STREAMING_DISPATCH_METADATA_LAUNCH_CASE(num_rdma_ranks)                                       \
    {                                                                                                  \
        auto k = streaming_dispatch_metadata_kernel<false, num_rdma_ranks>;                            \
        EP_HOST_ASSERT(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,            \
                                            smem_bytes) == cudaSuccess);                              \
        cfg.dynamicSmemBytes = smem_bytes;                                                             \
        LAUNCH_KERNEL(&cfg, k,                                                                         \
                      topk_idx,                                                                        \
                      moe_recv_counter_mapped,                                                         \
                      moe_recv_rdma_counter_mapped,                                                    \
                      moe_recv_expert_counter_mapped,                                                  \
                      streaming_total_tiles_mapped,                                                    \
                      rdma_channel_prefix_matrix,                                                      \
                      recv_rdma_rank_prefix_sum,                                                       \
                      gbl_channel_prefix_matrix,                                                       \
                      recv_gbl_rank_prefix_sum,                                                        \
                      expert_frequency,                                                                \
                      expert_pool_block_offset,                                                        \
                      base_pool,                                                                       \
                      seen_per_substream,                                                              \
                      rank_prefix_matrix,                                                              \
                      tile_id_to_expert,                                                               \
                      pool_arrival_target,                                                             \
                      total_tiles_device,                                                              \
                      num_tokens, num_topk, num_experts, num_channels,                                 \
                      expert_alignment, tile_m,                                                        \
                      rdma_clean_offset, rdma_num_int_clean,                                           \
                      nvl_clean_offset, nvl_num_int_clean,                                             \
                      streaming_rdma_offset,                                                           \
                      rdma_buffer_ptr, buffer_ptrs, barrier_signal_ptrs, rank,                         \
                      cpu_rdma_team);                                                                  \
    }                                                                                                  \
    break

    SETUP_LAUNCH_CONFIG(1 + num_rdma_ranks, kNumThreads, stream);
    SWITCH_RDMA_RANKS(STREAMING_DISPATCH_METADATA_LAUNCH_CASE);
#undef STREAMING_DISPATCH_METADATA_LAUNCH_CASE
}

// At most 8 RDMA ranks to be sent
constexpr int get_num_topk_rdma_ranks(int num_rdma_ranks) {
    return num_rdma_ranks < 8 ? num_rdma_ranks : 8;
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming dispatch_main: pool-layout, dropless. Mirrors `intranode::
// dispatch_main_kernel` (intranode.cu:366–819) — same six-struct contract,
// same Pass A (slot allocation) + Pass B (per-pool-slot scalars + per-recv-
// token reverse-map) + Pass 2 fire (per-block atomic-add → release-store
// `tile_ready[block_id] = dispatch_seq`) on the NVL receiver. The internode
// delta is the upstream RDMA→NVL forwarding: kRDMASender / kRDMASender-
// Coordinator / kRDMAAndNVLForwarder / kForwarderCoordinator stage data
// across the RDMA + NVL hops with no slot logic; only the NVL receiver
// (which owns the destination's `pool` memory) does Pass A/B/2.
//
// Per `internode.md` §"Where slot allocation lives — NVL receiver", placing
// Pass A on the receiver (rather than baking slot indices into the NVL
// message header from the forwarder) avoids cross-NVL-peer `base_pool` IPC
// plumbing, keeps the receiver symmetric with intranode (same single-writer
// proof: each (channel, src_nvl_rank) warp uniquely owns the
// `(c, src_world, e_local)` tuple), and keeps the forwarder lean (it stays
// the current bottleneck per the TMA-double-buffer perf opt called out in
// `internode.md` §"Optional perf follow-ups").
// ─────────────────────────────────────────────────────────────────────────────

// Per-warp lane-0-resident scratch for Pass A's K-loop slot mapping.
// num_topk ≤ 32 is asserted at kernel entry.
constexpr int kMaxTopK = 32;

// SMEM bytes for the NVL receiver's per-warp `seen[src_rdma][e_local]` slab,
// sized at runtime by E_local. Total receiver-side SMEM:
//   NUM_MAX_NVL_PEERS warps × (kNumRDMARanks × E_local × 4) bytes.
// At worst case (16 RDMA × 64 E_local) = 32 KB across 8 warps. Forwarder
// blocks share the same SMEM allocation but leave the slab unused.
__host__ __device__ inline int receiver_seen_smem_bytes(int num_rdma_ranks, int E_local) {
    return NUM_MAX_NVL_PEERS * num_rdma_ranks * E_local * static_cast<int>(sizeof(int));
}

template <int kNumRDMARanks,
          int kNumTMABytesPerWarp,
          int kNumDispatchRDMASenderWarps,
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
__global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32), 1)
dispatch_main_kernel(DispatchPoolOut pool_out,
                     DispatchPerTokenOut per_token_out,
                     DispatchInputs inputs,
                     DispatchTileSignal tile_signal,
                     DispatchShape shape,
                     DispatchEnv env) {
    enum class WarpRole { kRDMASender, kRDMASenderCoordinator, kRDMAAndNVLForwarder, kForwarderCoordinator, kNVLReceivers };

    const auto num_sms = static_cast<int>(gridDim.x);
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_channels = num_sms / 2, channel_id = sm_id / 2;
    const bool is_forwarder = sm_id % 2 == 0;
    const auto rdma_rank = env.rank / NUM_MAX_NVL_PEERS, nvl_rank = env.rank % NUM_MAX_NVL_PEERS;
    const int num_world_ranks = kNumRDMARanks * NUM_MAX_NVL_PEERS;
    const int E_local = shape.num_experts / num_world_ranks;
    const int local_expert_begin = env.rank * E_local;
    const int local_expert_end = local_expert_begin + E_local;

    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe == num_channels or ibgda_get_state()->num_rc_per_pe >= num_sms);
    EP_DEVICE_ASSERT(shape.num_topk <= 32);
    EP_DEVICE_ASSERT(E_local > 0 and E_local <= NUM_MAX_LOCAL_EXPERTS);

    const auto role_meta = [=]() -> std::pair<WarpRole, int> {
        if (is_forwarder) {
            if (warp_id < NUM_MAX_NVL_PEERS) {
                return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id) % NUM_MAX_NVL_PEERS};
            } else {
                return {WarpRole::kForwarderCoordinator, warp_id - NUM_MAX_NVL_PEERS};
            }
        } else if (warp_id < kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASender, -1};
        } else if (warp_id == kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASenderCoordinator, -1};
        } else {
            return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS};
        }
    }();
    auto warp_role = role_meta.first;
    auto target_rank = role_meta.second;
    EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS);

    // RDMA symmetric layout (same wire format as the legacy internode dispatch).
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    auto hidden_bytes = shape.hidden_int4 * sizeof(int4);
    auto num_bytes_per_token = get_num_bytes_per_token(shape.hidden_int4, shape.num_topk, shape.num_topk);
    auto rdma_channel_data = SymBuffer<uint8_t>(env.rdma_buffer_ptr,
        env.num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_meta = SymBuffer<int>(env.rdma_buffer_ptr,
        NUM_MAX_NVL_PEERS * 2 + 2, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(env.rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(env.rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

    // NVL buffer layouts (rs_wr = "read for senders, write for receivers";
    // ws_rr = "write for senders, read for receivers").
    void *rs_wr_buffer_ptr = nullptr, *ws_rr_buffer_ptr = nullptr;
    int rs_wr_rank = 0, ws_rr_rank = 0;
    if (warp_role == WarpRole::kRDMAAndNVLForwarder)
        rs_wr_buffer_ptr = env.buffer_ptrs[nvl_rank], ws_rr_buffer_ptr = env.buffer_ptrs[target_rank],
        rs_wr_rank = nvl_rank, ws_rr_rank = target_rank;
    if (warp_role == WarpRole::kNVLReceivers)
        rs_wr_buffer_ptr = env.buffer_ptrs[target_rank], ws_rr_buffer_ptr = env.buffer_ptrs[nvl_rank],
        rs_wr_rank = target_rank, ws_rr_rank = nvl_rank;

    auto nvl_channel_x = AsymBuffer<uint8_t>(ws_rr_buffer_ptr,
                                             env.num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                                             NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                             .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_start =
        AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_end = AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                                      .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_head =
        AsymBuffer<int>(rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, ws_rr_rank).advance_also(ws_rr_buffer_ptr);
    auto nvl_channel_tail =
        AsymBuffer<int>(ws_rr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);

    // RDMA sender warp synchronization
    __shared__ int rdma_send_channel_lock[kNumRDMARanks];
    __shared__ int rdma_send_channel_tail[kNumRDMARanks];
    __shared__ uint32_t rdma_send_channel_window[kNumRDMARanks];
    auto sync_rdma_sender_smem = []() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumDispatchRDMASenderWarps + 1) * 32)); };

    // TMA buffer slabs (per (channel-block, NVL-peer-warp)). Forwarder warps and
    // NVL receiver warps both index by `target_rank` ∈ [0, NUM_MAX_NVL_PEERS),
    // sharing the layout (mirrors legacy).
    extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
    auto tma_buffer = smem_tma_buffer + target_rank * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);
    uint32_t tma_phase = 0;
    if ((warp_role == WarpRole::kRDMAAndNVLForwarder or warp_role == WarpRole::kNVLReceivers) and elect_one_sync()) {
        mbarrier_init(tma_mbarrier, 1);
        fence_barrier_init();
        EP_DEVICE_ASSERT(num_bytes_per_token + sizeof(uint64_t) <= kNumTMABytesPerWarp);
    }
    __syncwarp();

    // Forwarder warp synchronization
    __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
    __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
    auto sync_forwarder_smem = []() { asm volatile("barrier.sync 1, %0;" ::"r"((NUM_MAX_NVL_PEERS + 1) * 32)); };

    // NVL receiver synchronization (custom barrier 2 — only the
    // NUM_MAX_NVL_PEERS receiver warps participate, not the RDMA senders sharing
    // this block).
    auto sync_nvl_receivers = []() { asm volatile("barrier.sync 2, %0;" ::"r"(NUM_MAX_NVL_PEERS * 32)); };

    if (warp_role == WarpRole::kRDMASender) {
        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(shape.num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // Send number of tokens in this channel by `-value - 1`
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            auto dst_ptr =
                dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
            if (lane_id < NUM_MAX_NVL_PEERS) {
                dst_ptr[lane_id] =
                    -(channel_id == 0
                          ? 0
                          : inputs.gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_channels + channel_id - 1]) -
                    1;
            } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] =
                    -inputs.gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id - NUM_MAX_NVL_PEERS) * num_channels +
                                                      channel_id] -
                    1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : inputs.rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1]) - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
                dst_ptr[lane_id] = -inputs.rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] - 1;
            }
            __syncwarp();

            if (dst_rdma_rank != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                                  translate_dst_rdma_rank<false>(dst_rdma_rank, nvl_rank),
                                                  channel_id, lane_id, 0);
            }
        }
        sync_rdma_sender_smem();

        // Iterate over tokens and copy into buffer
        int64_t token_idx;
        int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;
        auto send_buffer = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);
        for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
            uint64_t is_token_in_rank_uint64 = 0;
            if (lane_id < kNumRDMARanks) {
                is_token_in_rank_uint64 =
                    __ldg(reinterpret_cast<const uint64_t*>(inputs.is_token_in_rank + token_idx * num_world_ranks + lane_id * NUM_MAX_NVL_PEERS));
                global_rdma_tail_idx += (is_token_in_rank_uint64 != 0);
            }
            __syncwarp();

            if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
                continue;
            auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;

            auto start_time = clock64();
            while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= env.num_max_rdma_chunked_recv_tokens) {
                cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));
                if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch RDMA sender timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA lane: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, cached_rdma_channel_head, rdma_tail_idx);
                    trap();
                }
            }
            __syncwarp();

            // Store RDMA head for combine
            if (lane_id < kNumRDMARanks)
                per_token_out.send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;

            // Broadcast tails
            SourceMeta src_meta;
            int num_topk_ranks = 0, topk_ranks[kNumTopkRDMARanks];
            void* dst_send_buffers[kNumTopkRDMARanks];
            #pragma unroll
            for (int i = 0, slot_idx; i < kNumRDMARanks; ++i)
                if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {
                    slot_idx = slot_idx % env.num_max_rdma_chunked_recv_tokens;
                    topk_ranks[num_topk_ranks] = i;
                    auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i);
                    auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);
                    if (lane_id == num_topk_ranks)
                        src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);
                    dst_send_buffers[num_topk_ranks++] =
                        reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) + slot_idx * num_bytes_per_token;
                }
            EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRDMARanks);

            // Copy `x` into symmetric send buffer
            auto st_broadcast = [=](const int key, const int4& value) {
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);
            };
            UNROLLED_WARP_COPY(5, lane_id, shape.hidden_int4, 0, inputs.x + token_idx * shape.hidden_int4, ld_nc_global, st_broadcast);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + shape.hidden_int4;

            // Copy source metadata
            if (lane_id < num_topk_ranks)
                st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;

            // Copy `topk_idx` and `topk_weights`
            #pragma unroll
            for (int i = lane_id; i < shape.num_topk * num_topk_ranks; i += 32) {
                auto rank_idx = i / shape.num_topk, copy_idx = i % shape.num_topk;
                auto idx_value = static_cast<int>(ld_nc_global(inputs.topk_idx + token_idx * shape.num_topk + copy_idx));
                auto weight_value = ld_nc_global(inputs.topk_weights + token_idx * shape.num_topk + copy_idx);
                st_na_global(reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx, idx_value);
                st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) + shape.num_topk + copy_idx, weight_value);
            }
            __syncwarp();

            // Release the transaction in the window
            if (is_token_in_rank_uint64 != 0) {
                acquire_lock(rdma_send_channel_lock + lane_id);
                auto latest_tail = rdma_send_channel_tail[lane_id];
                auto offset = rdma_tail_idx - latest_tail;
                while (offset >= 32) {
                    release_lock(rdma_send_channel_lock + lane_id);
                    acquire_lock(rdma_send_channel_lock + lane_id);
                    latest_tail = rdma_send_channel_tail[lane_id];
                    offset = rdma_tail_idx - latest_tail;
                }
                auto window = rdma_send_channel_window[lane_id] | (1u << offset);
                if (offset == 0) {
                    auto num_empty_slots = (~window) == 0 ? 32 : __ffs(~window) - 1;
                    st_release_cta(rdma_send_channel_tail + lane_id, latest_tail + num_empty_slots);
                    window >>= num_empty_slots;
                }
                rdma_send_channel_window[lane_id] = window;
                release_lock(rdma_send_channel_lock + lane_id);
            }
            __syncwarp();
        }
    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
        EP_DEVICE_ASSERT(env.num_max_rdma_chunked_recv_tokens % env.num_max_rdma_chunked_send_tokens == 0);

        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_lock[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_tail[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_window[lane_id] = 0) : 0;

        sync_rdma_sender_smem();

        int num_tokens_to_send = 0;
        if (lane_id < kNumRDMARanks) {
            num_tokens_to_send = inputs.rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
            if (channel_id > 0)
                num_tokens_to_send -= inputs.rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];
        }

        int last_issued_tail = 0;
        auto start_time = clock64();
        while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                printf("DeepEP RDMA sender coordinator timeout, channel: %d, IB: %d, nvl %d, dst IB: %d, tail: %d, remaining: %d\n",
                       channel_id, rdma_rank, nvl_rank, lane_id, last_issued_tail, num_tokens_to_send);
                trap();
            }

            for (int i = 0, synced_num_tokens_to_send; i < kNumRDMARanks; ++i) {
                int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;
                synced_num_tokens_to_send = __shfl_sync(0xffffffff, num_tokens_to_send, dst_rdma_rank);
                if (synced_num_tokens_to_send == 0)
                    continue;

                auto processed_tail =
                    __shfl_sync(0xffffffff, ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank)), 0);
                auto synced_last_issued_tail = __shfl_sync(0xffffffff, last_issued_tail, dst_rdma_rank);
                auto num_tokens_processed = processed_tail - synced_last_issued_tail;
                if (num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < env.num_max_rdma_chunked_send_tokens)
                    continue;

                auto num_tokens_to_issue = min(num_tokens_processed, env.num_max_rdma_chunked_send_tokens);
                EP_DEVICE_ASSERT(num_tokens_to_issue >= 0 and num_tokens_to_issue <= synced_num_tokens_to_send);

                if (dst_rdma_rank != rdma_rank) {
                    auto dst_slot_idx = synced_last_issued_tail % env.num_max_rdma_chunked_recv_tokens;
                    EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= env.num_max_rdma_chunked_recv_tokens);
                    const size_t num_bytes_per_msg = num_bytes_per_token * num_tokens_to_issue;
                    const auto dst_ptr =
                        reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + dst_slot_idx * num_bytes_per_token);
                    const auto src_ptr =
                        reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + dst_slot_idx * num_bytes_per_token);
                    nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                      translate_dst_rdma_rank<false>(dst_rdma_rank, nvl_rank),
                                                      channel_id, lane_id, 0);
                } else {
                    memory_fence();
                }
                __syncwarp();

                if (lane_id == dst_rdma_rank) {
                    last_issued_tail += num_tokens_to_issue;
                    num_tokens_to_send -= num_tokens_to_issue;
                    nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank),
                                                    num_tokens_to_issue,
                                                    translate_dst_rdma_rank<false>(dst_rdma_rank, nvl_rank),
                                                    channel_id,
                                                    dst_rdma_rank == rdma_rank);
                }
                __syncwarp();
            }
        }
    } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
        // RDMA consumers and NVL producers. Bulk-copy stage — no slot logic,
        // no per-(c, src, e) bookkeeping. Wire format on NVL ring is the same
        // as legacy internode dispatch (data + SourceMeta + topk_idx + topk_weights).
        const auto dst_nvl_rank = target_rank;

        int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;
        EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
        auto start_time = clock64();
        if (lane_id < kNumRDMARanks) {
            while (true) {
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
                auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
                auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
                if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
                    int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);
                    st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);

                    src_rdma_channel_prefix = -meta_2 - 1;
                    auto src_rdma_channel_prefix_1 = -meta_3 - 1;
                    num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
                    per_token_out.recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = src_rdma_channel_prefix_1;
                    src_rdma_channel_prefix += lane_id == 0 ? 0 : inputs.recv_rdma_rank_prefix_sum[lane_id - 1];
                    EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);
                    break;
                }

                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch forwarder timeout (RDMA meta), channel: %d, RDMA: %d, nvl: %d, src RDMA lane: %d, dst NVL: %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, dst_nvl_rank);
                    trap();
                }
            }
        }
        __syncwarp();

        int* send_nvl_head_for_lane = per_token_out.send_nvl_head + (src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank);
        sync_forwarder_smem();

        int src_rdma_rank = sm_id % kNumRDMARanks;
        int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
        int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0, rdma_nvl_token_idx = 0;
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
            start_time = clock64();
            while (true) {
                const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
                if (env.num_max_nvl_chunked_recv_tokens - num_used_slots >= env.num_max_nvl_chunked_send_tokens)
                    break;
                cached_nvl_channel_head = __shfl_sync(0xffffffffu, ld_volatile_global(nvl_channel_head.buffer()), 0);

                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch forwarder timeout (NVL check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank);
                    trap();
                }
            }

            start_time = clock64();
            while (true) {
                src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
                if (__shfl_sync(0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
                    if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
                        cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank)));
                    if (__shfl_sync(0xffffffff, cached_rdma_channel_tail > cached_rdma_channel_head, src_rdma_rank))
                        break;
                }
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf("DeepEP dispatch forwarder timeout (RDMA check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank);
                    trap();
                }
            }
            auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);
            auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);

            for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++i) {
                auto rdma_slot_idx = i % env.num_max_rdma_chunked_recv_tokens;
                auto shifted = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_token;
                auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes));
                lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;
                bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);
                if (lane_id == src_rdma_rank) {
                    auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
                    rdma_nvl_token_idx += is_in_dst_nvl_rank;
                    send_nvl_head_for_lane[i * NUM_MAX_NVL_PEERS] = cached_head;
                }
                if (not is_in_dst_nvl_rank)
                    continue;

                int dst_slot_idx = (cached_nvl_channel_tail++) % env.num_max_nvl_chunked_recv_tokens;
                auto dst_shifted = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;

                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, num_bytes_per_token, false);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, num_bytes_per_token);
                }
                __syncwarp();
                mbarrier_wait(tma_mbarrier, tma_phase);
                if (elect_one_sync())
                    tma_store_1d(tma_buffer, dst_shifted, num_bytes_per_token);
                __syncwarp();

                if ((++num_tokens_sent) == env.num_max_nvl_chunked_send_tokens)
                    src_rdma_tail = i + 1;

                tma_store_wait<0>();
                __syncwarp();
            }

            if (lane_id == src_rdma_rank)
                forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);

            __syncwarp();
            if (elect_one_sync())
                st_release_sys_global(nvl_channel_tail.buffer(), cached_nvl_channel_tail);
        }

        __syncwarp();
        if (elect_one_sync())
            forward_channel_retired[dst_nvl_rank] = true;
    } else if (warp_role == WarpRole::kForwarderCoordinator) {
        if (target_rank > 0)
            return;

        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        #pragma unroll
        for (int i = lane_id; i < kNumRDMARanks * NUM_MAX_NVL_PEERS; i += 32)
            forward_channel_head[i % NUM_MAX_NVL_PEERS][i / NUM_MAX_NVL_PEERS] = 0;
        if (lane_id < NUM_MAX_NVL_PEERS)
            forward_channel_retired[lane_id] = false;
        sync_forwarder_smem();

        int last_head = 0, target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;
        while (true) {
            int min_head = std::numeric_limits<int>::max();
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                if (not forward_channel_retired[i])
                    min_head = min(min_head, forward_channel_head[i][target_rdma]);
            if (__all_sync(0xffffffff, min_head == std::numeric_limits<int>::max()))
                break;

            if (min_head != std::numeric_limits<int>::max() and min_head >= last_head + env.num_max_rdma_chunked_send_tokens and
                lane_id < kNumRDMARanks) {
                nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank),
                                                min_head - last_head,
                                                translate_dst_rdma_rank<false>(lane_id, nvl_rank),
                                                channel_id + num_channels,
                                                lane_id == rdma_rank);
                last_head = min_head;
            }

            __nanosleep(NUM_WAIT_NANOSECONDS);
        }
    } else {
        // ── kNVLReceivers: pool-layout streaming receiver ───────────────────
        // One warp per src_nvl_rank (= target_rank). Each warp drains its
        // (channel, src_nvl_rank) NVL ring stream and:
        //   Pass A: lane-0 K-loop computes slot = base_pool[c, src_world,
        //           e_local] + warp_local_seen[src_rdma_rank][e_local]++.
        //           Single-writer per (c, src_world, e_local) tuple — no atomic.
        //   Pass B: TMA-load message data; K-fanout TMA-store to pool[slot] for
        //           each local k; lane-0 writes per-pool-slot scalars
        //           (pool_topk_weight, pool_recv_token, pool_k_slot) and
        //           per-recv-token reverse-map (recv_token_to_slots[r, :K],
        //           per_token_remaining[r], k_local_count[r]); lane-0 writes
        //           recv_src_meta (combine plumbing).
        //   Pass 2 fire (substream-end): per-warp `__threadfence_system()` +
        //           cross-NVL-receiver-warp `bar.sync 2` (for system-scope
        //           visibility of other warps' writes contributing to the same
        //           tile via base_pool stacking) → lane-0 expert-major walk
        //           over (e, src_rdma_rank); atomic-add into pool_arrival_count;
        //           on completion, release-store tile_ready[block_id] = dispatch_seq.
        const int src_nvl_rank = target_rank;
        int total_offset = 0;

        if (lane_id < kNumRDMARanks and lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
            total_offset = inputs.recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank - 1];

        int start_offset = 0, end_offset = 0, num_tokens_to_recv;
        auto start_time = clock64();
        while (lane_id < kNumRDMARanks) {
            start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
            end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
            if (start_offset < 0 and end_offset < 0) {
                start_offset = -start_offset - 1, end_offset = -end_offset - 1;
                total_offset += start_offset;
                break;
            }
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("DeepEP dispatch NVL receiver timeout (prefix), channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, src nvl: %d\n",
                       channel_id, rdma_rank, nvl_rank, lane_id, src_nvl_rank);
                trap();
            }
        }
        num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

        // Save for combine usage (per-(src_world, channel) recv-token cumulative).
        if (lane_id < kNumRDMARanks)
            per_token_out.recv_gbl_channel_prefix_matrix[(lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] = total_offset;
        __syncwarp();

        // ── Per-warp Pass A counter table: warp_local_seen[src_rdma][e_local].
        // Stored in SMEM (sized at runtime by E_local; per-warp slab indexed
        // by `target_rank` ∈ [0, NUM_MAX_NVL_PEERS)). Lane 0 reads/writes it;
        // lane 0 walks Pass 2 fire over the same slab.
        // Slab placement: right after the NUM_MAX_NVL_PEERS TMA buffer slabs.
        int* warp_local_seen = reinterpret_cast<int*>(
            smem_tma_buffer + NUM_MAX_NVL_PEERS * kNumTMABytesPerWarp
            + target_rank * kNumRDMARanks * E_local * static_cast<int>(sizeof(int)));
        for (int i = lane_id; i < kNumRDMARanks * E_local; i += 32)
            warp_local_seen[i] = 0;
        __syncwarp();

        // base_pool slice for this warp's substreams: indexed by [src_rdma][e]
        // inside the kernel via (channel_id * num_world_ranks + src_world) * E_local.
        const int* base_pool_for_channel = tile_signal.base_pool + channel_id * num_world_ranks * E_local;

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            start_time = clock64();
            while (true) {
                if (cached_channel_head_idx != cached_channel_tail_idx)
                    break;
                cached_channel_tail_idx = __shfl_sync(0xffffffff, ld_acquire_sys_global(nvl_channel_tail.buffer()), 0);
                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch NVL receiver timeout (tail), channel: %d, RDMA: %d, nvl: %d, src NVL: %d\n",
                           channel_id, rdma_rank, nvl_rank, src_nvl_rank);
                    trap();
                }
            }

            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++chunk_idx, --num_tokens_to_recv) {
                int token_idx_in_buffer = (cached_channel_head_idx++) % env.num_max_nvl_chunked_recv_tokens;
                auto shifted = nvl_channel_x.buffer() + token_idx_in_buffer * num_bytes_per_token;
                auto meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes));
                int src_rdma_rank = meta.src_rdma_rank;
                int recv_token_idx = __shfl_sync(0xffffffff, total_offset, src_rdma_rank);
                (lane_id == src_rdma_rank) ? (total_offset += 1) : 0;

                int src_world = src_rdma_rank * NUM_MAX_NVL_PEERS + src_nvl_rank;
                auto topk_idx_in_msg     = reinterpret_cast<const int*>(shifted + hidden_bytes + sizeof(SourceMeta));
                auto topk_weights_in_msg = reinterpret_cast<const float*>(shifted + hidden_bytes + sizeof(SourceMeta) + shape.num_topk * sizeof(int));

                // ── Pass A: lane-0 K-loop, slot allocation. ──
                int slot_row[kMaxTopK];
                if (lane_id == 0) {
                    int* base_pool_substream = const_cast<int*>(base_pool_for_channel + src_world * E_local);
                    int* seen_for_src = warp_local_seen + src_rdma_rank * E_local;
                    for (int k = 0; k < shape.num_topk; ++k) {
                        int e_global = static_cast<int>(__ldg(topk_idx_in_msg + k));
                        int e_local = (e_global >= local_expert_begin and e_global < local_expert_end) ? e_global - local_expert_begin : -1;
                        if (e_local >= 0) {
                            int slot = base_pool_substream[e_local] + seen_for_src[e_local];
                            seen_for_src[e_local] += 1;
                            slot_row[k] = slot;
                        } else {
                            slot_row[k] = -1;
                        }
                    }
                }
                // No need to broadcast: only lane 0 issues the K-fanout TMA stores
                // and per-pool-slot scalar writes. Other lanes participate in the
                // TMA load below via the warp's shared mbarrier.

                // ── Pass B: TMA-load hidden, K-fanout TMA-store. ──
                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, hidden_bytes);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, hidden_bytes);
                }
                __syncwarp();
                mbarrier_wait(tma_mbarrier, tma_phase);

                if (lane_id == 0) {
                    for (int k = 0; k < shape.num_topk; ++k) {
                        int slot = slot_row[k];
                        if (slot < 0) continue;
                        tma_store_1d(tma_buffer,
                                     pool_out.pool + static_cast<int64_t>(slot) * shape.hidden_int4,
                                     hidden_bytes, false);
                    }
                }
                __syncwarp();

                // ── Pass B (continued): per-pool-slot scalars + per-recv-token scalars. ──
                if (lane_id == 0) {
                    int k_local_count_val = 0;
                    for (int k = 0; k < shape.num_topk; ++k) {
                        int slot = slot_row[k];
                        per_token_out.recv_token_to_slots[recv_token_idx * shape.num_topk + k] = slot;
                        if (slot < 0) continue;
                        pool_out.pool_topk_weight[slot] = ld_nc_global(topk_weights_in_msg + k);
                        pool_out.pool_recv_token[slot] = recv_token_idx;
                        pool_out.pool_k_slot[slot] = k;
                        ++k_local_count_val;
                    }
                    if (k_local_count_val > 0) {
                        per_token_out.per_token_remaining[recv_token_idx] = k_local_count_val;
                        per_token_out.k_local_count[recv_token_idx] = k_local_count_val;
                    }
                }

                // Combine plumbing: recv_src_meta per recv_token.
                if (elect_one_sync())
                    st_na_global(reinterpret_cast<SourceMeta*>(per_token_out.recv_src_meta) + recv_token_idx, meta);

                // Wait for K-fanout TMA stores to publish before the next token
                // reuses tma_buffer.
                tma_store_wait<0>();
                __syncwarp();
            }

            if (elect_one_sync())
                st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx);
        }

        // ── Pass 2 fire: substream-end.
        // Every NVL receiver thread does its own __threadfence_system() to
        // publish its prior writes (per-pool-slot scalars + recv_token_to_slots
        // are written by lane 0 of every receiver warp; the fence on every
        // thread covers that thread's writes). The cross-receiver-warp
        // bar.sync 2 ensures all 8 NVL receiver warps in this block reach
        // the fence point AND finish their fences before any warp's Pass 2
        // walk fires `tile_ready` for a tile that other warps contributed
        // to. Mirrors `intranode.cu:781–797`.
        tma_store_wait<0>();
        __syncwarp();
        __threadfence_system();
        sync_nvl_receivers();

        if (lane_id == 0) {
            for (int e = 0; e < E_local; ++e) {
                #pragma unroll 1
                for (int src_rdma_rank = 0; src_rdma_rank < kNumRDMARanks; ++src_rdma_rank) {
                    int n_writes_for_e = warp_local_seen[src_rdma_rank * E_local + e];
                    if (n_writes_for_e == 0) continue;
                    int src_world = src_rdma_rank * NUM_MAX_NVL_PEERS + src_nvl_rank;
                    int slot_start_e = base_pool_for_channel[src_world * E_local + e];
                    int slot_end_e = slot_start_e + n_writes_for_e;
                    int first_block = slot_start_e / shape.tile_m;
                    int last_block = (slot_end_e - 1) / shape.tile_m;
                    for (int block_id = first_block; block_id <= last_block; ++block_id) {
                        int block_slot_start = block_id * shape.tile_m;
                        int block_slot_end = block_slot_start + shape.tile_m;
                        int writes_in_block =
                            min(slot_end_e, block_slot_end) - max(slot_start_e, block_slot_start);
                        int cnt_before = atomicAdd(&tile_signal.pool_arrival_count[block_id], writes_in_block);
                        if (cnt_before + writes_in_block == tile_signal.pool_arrival_target[block_id]) {
                            memory_fence();
                            st_release_sys_global(tile_signal.tile_ready + block_id, tile_signal.dispatch_seq);
                        }
                    }
                }
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
                          int num_rdma_ranks,
                          int num_channels,
                          cudaStream_t stream) {
    constexpr int kNumDispatchRDMASenderWarps = 7;
    constexpr int kNumTMABytesPerWarp = 16384;

    int num_world_ranks = num_rdma_ranks * NUM_MAX_NVL_PEERS;
    int E_local = shape.num_experts / num_world_ranks;
    int smem_size = kNumTMABytesPerWarp * NUM_MAX_NVL_PEERS
                  + receiver_seen_smem_bytes(num_rdma_ranks, E_local);

#define DISPATCH_LAUNCH_CASE(num_rdma_ranks_)                                          \
    {                                                                                  \
        auto kernel = dispatch_main_kernel<num_rdma_ranks_,                            \
                                           kNumTMABytesPerWarp,                        \
                                           kNumDispatchRDMASenderWarps>;               \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                                             \
        LAUNCH_KERNEL(&cfg, kernel,                                                    \
                      pool_out, per_token_out, inputs, tile_signal, shape, env);       \
    }                                                                                  \
    break

    SETUP_LAUNCH_CONFIG(num_channels * 2, (kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32, stream);
    int num_ranks = num_world_ranks;  // for SWITCH_RDMA_RANKS
    SWITCH_RDMA_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

}  // namespace internode

}  // namespace stream_ep
