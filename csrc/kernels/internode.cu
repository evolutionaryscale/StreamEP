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

// `get_num_bytes_per_token` is declared as an inline __host__ __device__
// helper in api.cuh so stream_ep.cpp can reuse it for buffer sizing.

template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

// Gen-stamp encoding for the NVL dispatch / combine ring slots. Each slot
// is a 64-bit pair `(seq, value)` — high 32 bits carry the generation tag
// (low 32 bits of `tile_signal.dispatch_seq` / `combine_seq`, monotonic
// across the training run), low 32 bits carry the payload. The reader
// checks the high half matches the current iter's seq before consuming the
// low half — that distinguishes this-iter writes from prior-iter residue
// and removes the need for inter-iter cleanup memsets on
// `nvl_channel_prefix_start/end`, `nvl_channel_head`, and
// `nvl_channel_tail`. NVLink writes are intra-node coherent, so no amo /
// L2-line-pack trick is required (unlike the RDMA meta region — see the
// 128-byte alignment block below for the sentinel-amo coherence rationale).
//
// Wrap horizon: 32-bit seq window, one increment per kernel call, so
// ~2^31 distinct iter tags / phase before aliasing. With the phase bit
// the caller stamps in the LSB (fwd vs bwd of one layer), that's still
// ~2^30 phase-distinct logical iters — ~16M training steps × 64 layers
// before any aliasing risk. Effectively infinite for production training.
__forceinline__ __device__ uint64_t nvl_pack(int64_t seq, int value) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(seq)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(value));
}
__forceinline__ __device__ bool nvl_seq_match(uint64_t packed, int64_t seq) {
    return static_cast<uint32_t>(packed >> 32) == static_cast<uint32_t>(seq);
}
__forceinline__ __device__ int nvl_unpack_value(uint64_t packed) {
    return static_cast<int>(static_cast<uint32_t>(packed));
}

// Signed-difference atomicMax for the persistent `reader_prev_*` arrays.
// The on-the-wire NIC AMO is 4-byte (mlx5 ATOMIC_MASKED_FA), so the slot's
// low 32 bits accumulate modulo 2^32. The persistent reader_prev_*[c, peer]
// is uint32, storing only the low 32 bits — across-iter wrap is handled by
// the read-side via modular uint32 subtraction `(cur_low - prev_low) mod 2^32`
// (always < 2^31 at production since per-iter advance ≪ 2^32), so no high
// bits / cumulative reconstruction is needed.
//
// Within an iter, multiple warps in different roles observe the slot at
// slightly different moments and call this helper. The slot is monotone
// within an iter (per-iter advance ≪ 2^32, no intra-iter wrap), so the
// latest observation has the largest cur_low. Across iters, writebacks are
// stream-ordered (no race). The cross-iter wrap boundary breaks plain
// `atomicMax<uint32>`: at iter N+1, candidates can straddle the 2^32 edge,
// and unsigned max would prefer the pre-wrap (near 2^32-1) value over the
// post-wrap (near 0) value, even though the post-wrap value is the latest.
// The signed-difference CAS below picks the candidate that is "ahead" in
// modular arithmetic regardless of where the wrap edge falls.
__forceinline__ __device__ void atomicmax_reader_prev_cumulative(uint32_t* prev_addr,
                                                                  uint32_t cur_low) {
    uint32_t old = *prev_addr;
    while (static_cast<int32_t>(cur_low - old) > 0) {
        uint32_t prev = atomicCAS(prev_addr, old, cur_low);
        if (prev == old) break;
        old = prev;
    }
}

// Round `gbl_ptr` up to a 128-byte boundary in place. Used before allocating
// the RDMA dispatch meta SymBuffer so each 32-int (=128B) slab aligns to a
// single H100 L2 cache line — required for sentinel-amo coherence: sender
// bulk_put → amo_nonfetch_add(slot 30) on the same QP; the
// amo invalidates the entire 128B line on the receiver, so subsequent reads
// of slots 0..17 return HBM values rather than stale L2). The 128B base
// alignment of `rdma_buffer_ptr` (NUM_BUFFER_ALIGNMENT_BYTES) is preserved
// only at the start of the heap; intermediate SymBuffer advances depend on
// `num_bytes_per_token * num_max_rdma_chunked_recv_tokens * num_rdma_ranks`
// which is only int4-aligned. A `get_rdma_buffer_size_hint` slack of
// NUM_BUFFER_ALIGNMENT_BYTES covers the up-to-127-byte gap this introduces.
__forceinline__ __device__ void align_meta_base_to_l2_line(void*& gbl_ptr) {
    constexpr uintptr_t kMask = NUM_BUFFER_ALIGNMENT_BYTES - 1;
    auto p = reinterpret_cast<uintptr_t>(gbl_ptr);
    gbl_ptr = reinterpret_cast<void*>((p + kMask) & ~kMask);
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
        // Streaming SymBuffer offset within rdma_buffer_ptr (post the leading count-exchange payload)
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
    EP_DEVICE_ASSERT(num_ranks <= 128);
    EP_DEVICE_ASSERT(kNumRDMARanks <= 64);
    for (int t = thread_id; t < num_tokens; t += num_threads) {
        int channel_id = t / num_tokens_per_channel_var;
        if (channel_id >= num_channels) channel_id = num_channels - 1;
        __uint128_t world_mask = 0;
        uint64_t rdma_mask = 0;
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
            __uint128_t world_bit = ((__uint128_t)1) << dst_world;
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

    // ── Phase A4: build + send the count payload via RDMA.
    // No upfront inter-iter IBGDA quiet + nvshmem_sync + NVL barrier is
    // needed — every polled slot is iter-disambiguated by its cumulative
    // protocol (RDMA head/tail and NVL gen-stamp), so dispatch ring slots
    // no longer need a pre-iter cross-rank drain. The metadata kernel's
    // own RDMA puts in this phase remain bracketed by their own quiet +
    // sync below; only the upfront cross-iter drain is gone.
    auto rdma_recv_num_tokens_mixed = SymBuffer<int>(rdma_buffer_ptr,
        NUM_MAX_NVL_PEERS + num_rdma_experts + 1, kNumRDMARanks);

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
    // Per-NVL-slot send/recv pattern: writer w writes to peer m's region at
    // slot indexed by w; receiver reads its own region at slot=src_nvl.
    const int kNvlSlotInts = kNumRDMARanks * num_channels * E_local;
    auto nvl_streaming_send = AsymBuffer<int>(nvl_send_buffer, kNvlSlotInts, NUM_MAX_NVL_PEERS);
    auto nvl_streaming_recv = AsymBuffer<int>(nvl_recv_buffer, kNvlSlotInts, NUM_MAX_NVL_PEERS);

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

    // base_pool[c, src_world, e]: per-expert, partition the slot range
    // into NVL-local substreams (`src_rdma == rdma_rank`) first, then
    // RDMA-remote substreams (`src_rdma != rdma_rank`). Within each
    // partition the order is `(c, src_world)` lex.
    //
    // Rationale: `pool_arrival_count[block]` waits on the slowest contributor
    // to that block. NVL-local substreams complete ~µs after dispatch start;
    // RDMA-routed substreams complete after the round-trip + forwarder cost.
    // Without this partition, slow RDMA tiles can land at low `tile_id`s and
    // HOL-block the linear-claim CTAs in kernel A. With it, low `tile_id`s
    // within each expert are NVL-only paths (low latency) and high `tile_id`s
    // are RDMA-routed paths (high latency); producer fire order and consumer
    // claim order stay aligned with transport heterogeneity. Determinism is
    // preserved (the NVL-local set is a deterministic function of
    // `src_rdma == rdma_rank`).
    //
    // Tile-boundary cost: if `tile_m` doesn't divide the NVL-local count for
    // an expert, exactly one tile per expert straddles the partition and
    // contains both NVL and RDMA slots; that tile is RDMA-bound (waits on
    // the slowest contributor as before). Accepted as-is — bounding to one
    // tile per expert is a small upper bound on pool overhead.
    for (int e = thread_id; e < E_local; e += num_threads) {
        int acc = smem_pool_blk[e] * tile_m;
        // Pass 1: NVL-local substreams.
        for (int cs = 0; cs < num_channels * num_ranks; ++cs) {
            int src_world = cs % num_ranks;
            int src_rdma = src_world / NUM_MAX_NVL_PEERS;
            if (src_rdma != rdma_rank) continue;
            base_pool[cs * E_local + e] = acc;
            acc += seen_per_substream[cs * E_local + e];
        }
        // Pass 2: RDMA-remote substreams.
        for (int cs = 0; cs < num_channels * num_ranks; ++cs) {
            int src_world = cs % num_ranks;
            int src_rdma = src_world / NUM_MAX_NVL_PEERS;
            if (src_rdma == rdma_rank) continue;
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
// token reverse-map) + Pass 2 fire (per-block `red.release.gpu.global.add.s32`
// into `pool_arrival_count[block_id]`) on the NVL receiver. The internode
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
    align_meta_base_to_l2_line(env.rdma_buffer_ptr);
    auto rdma_channel_meta = SymBuffer<int>(env.rdma_buffer_ptr,
        kRdmaMetaSlabInts, kNumRDMARanks, channel_id, num_channels);
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
        AsymBuffer<uint64_t>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_end = AsymBuffer<uint64_t>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                                      .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_head =
        AsymBuffer<uint64_t>(rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, ws_rr_rank).advance_also(ws_rr_buffer_ptr);
    auto nvl_channel_tail =
        AsymBuffer<uint64_t>(ws_rr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);

    // NVL gen-stamp tag carried in the high 32 bits of each NVL slot.
    // Phase-distinct from `dispatch_grads_main_kernel`'s tag (low bit 1 for
    // bwd) so the fwd-dispatch leftover (seq tag = nvl_seq) doesn't alias
    // the same-iter bwd-dispatch reader (looking for nvl_seq | 1) on the
    // shared dispatch NVL ring. `tile_signal.dispatch_seq` is the user-
    // supplied per-call counter, monotonic across the training run; the
    // 32-bit window after the phase bit (~2 B distinct iters) is the wrap
    // horizon. Per-tile release-adds (`pool_arrival_count`) and the kernel-A
    // count-vs-target spin operate on int32 counts, not the int64 dispatch_seq;
    // dispatch_seq is consumed only by the NVL ring's nvl_seq generation here.
    const int64_t nvl_seq = tile_signal.dispatch_seq << 1;

    // RDMA sender warp synchronization
    __shared__ int rdma_send_channel_lock[kNumRDMARanks];
    __shared__ int rdma_send_channel_tail[kNumRDMARanks];
    __shared__ uint32_t rdma_send_channel_window[kNumRDMARanks];
    auto sync_rdma_sender_smem = []() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumDispatchRDMASenderWarps + 1) * 32)); };

    // TMA buffer slabs (per (channel-block, NVL-peer-warp)). Forwarder warps and
    // NVL receiver warps both index by `target_rank` ∈ [0, NUM_MAX_NVL_PEERS),
    // sharing the layout (mirrors legacy). Two-stage pipeline: each warp owns
    // two `num_bytes_per_token`-sized stage buffers (16-byte aligned) + two
    // 8-byte mbarriers at the tail. Stage `s` (∈ {0, 1}) is one outstanding
    // TMA load; the next iter's load issues into stage `1-s` before this iter
    // waits on its store, so load latency is hidden behind the store-wait.
    constexpr int kNumStages = 2;
    const int kStageStride = (num_bytes_per_token + 15) & ~15;
    extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
    auto tma_buffer = [=](int s) {
        return smem_tma_buffer + target_rank * kNumTMABytesPerWarp + s * kStageStride;
    };
    auto tma_mbarrier = [=](int s) {
        return reinterpret_cast<uint64_t*>(
            smem_tma_buffer + target_rank * kNumTMABytesPerWarp
            + kNumStages * kStageStride + s * static_cast<int>(sizeof(uint64_t)));
    };
    uint32_t tma_phase[kNumStages] = {0, 0};
    if ((warp_role == WarpRole::kRDMAAndNVLForwarder or warp_role == WarpRole::kNVLReceivers) and elect_one_sync()) {
        #pragma unroll
        for (int s = 0; s < kNumStages; ++s)
            mbarrier_init(tma_mbarrier(s), 1);
        fence_barrier_init();
        EP_DEVICE_ASSERT(kNumStages * kStageStride + kNumStages * static_cast<int>(sizeof(uint64_t)) <= kNumTMABytesPerWarp);
    }
    __syncwarp();

    // Forwarder warp synchronization
    __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
    __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
    auto sync_forwarder_smem = []() { asm volatile("barrier.sync 1, %0;" ::"r"((NUM_MAX_NVL_PEERS + 1) * 32)); };

    if (warp_role == WarpRole::kRDMASender) {
        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(shape.num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // Publish per-channel meta to each dst_rdma_rank: 18 raw data ints in
        // slots 0..17 (NUM_MAX_NVL_PEERS*2 start_sum/end_sum + 2 rdma_channel
        // prefix start/end) + a cumulative-across-iters sentinel at slot 30.
        // Remote dst: bulk_put → amo_nonfetch_add(slot 30, 1) on the same QP;
        // RC ordering places the put before the amo, and the amo invalidates
        // the receiver's 128B L2 line so the forwarder's reads of slots 0..17
        // fetch fresh from HBM (see kRdmaMetaSlabInts in api.cuh).
        // Local dst: direct stores + __threadfence + atomicAdd(slot 30) — the
        // forwarder is on the same GPU and observes the data via the L2.
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            auto dst_ptr =
                dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
            if (lane_id < NUM_MAX_NVL_PEERS) {
                dst_ptr[lane_id] =
                    channel_id == 0
                        ? 0
                        : inputs.gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_channels + channel_id - 1];
            } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] =
                    inputs.gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id - NUM_MAX_NVL_PEERS) * num_channels +
                                                     channel_id];
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = channel_id == 0 ? 0 : inputs.rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
                dst_ptr[lane_id] = inputs.rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            }
            __syncwarp();

            if (dst_rdma_rank != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                                  translate_dst_rdma_rank<false>(dst_rdma_rank, nvl_rank),
                                                  channel_id, lane_id, 0);
                if (lane_id == 0) {
                    nvshmemi_ibgda_amo_nonfetch_add(
                        rdma_channel_meta.recv_buffer(rdma_rank) + kRdmaMetaSentinelSlot,
                        1,
                        translate_dst_rdma_rank<false>(dst_rdma_rank, nvl_rank),
                        channel_id);
                }
            } else if (lane_id == 0) {
                __threadfence();
                atomicAdd(dst_ptr + kRdmaMetaSentinelSlot, 1);
            }
        }
        sync_rdma_sender_smem();

        // Iterate over tokens and copy into buffer.
        //
        // The RDMA head slot accumulates across all iters (peer's forwarder
        // coordinator publishes deltas via amo_nonfetch_add). Seed
        // prev_at_entry from the persistent reader-state array (written
        // at the prior kernel's exit, stream-ordered, race-free). Iter-
        // local cached_rdma_channel_head tracks this iter's advance only.
        int64_t token_idx;
        uint32_t prev_rdma_channel_head_at_entry = lane_id < kNumRDMARanks
            ? env.reader_prev_head[channel_id * kNumRDMARanks + lane_id]
            : 0u;
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
                cached_rdma_channel_head =
                    static_cast<int>(static_cast<uint32_t>(ld_acquire_sys_global(rdma_channel_head.buffer(lane_id))) - prev_rdma_channel_head_at_entry);
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

        // Writeback observed cumulative head to persistent reader_prev. All
        // sender warps share the role; each lane atomicMax's its observed
        // slot value to keep the array monotonic without a CTA-level sync.
        // Stream-ordered with the next kernel: the writeback is visible
        // before the next kernel reads it. False-low values cause the next
        // iter's sender to over-spin on backpressure (no correctness
        // impact); false-high values can't happen because the slot
        // accumulates monotonically.
        if (lane_id < kNumRDMARanks) {
            atomicmax_reader_prev_cumulative(
                env.reader_prev_head + channel_id * kNumRDMARanks + lane_id,
                static_cast<uint32_t>(ld_acquire_sys_global(rdma_channel_head.buffer(lane_id))));
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
        // Seed prev_at_entry from the persistent sentinel array (written at
        // the prior kernel's exit, stream-ordered, race-free). Spin on the
        // cumulative slot 30 advancing past prev; once tripped, the bulk_put
        // delivering slots 0..17 is RC-ordered before the amo and the L2
        // line is freshly invalidated, so plain ld_volatile reads return the
        // current iter's raw data.
        int prev_meta_sentinel_at_entry = lane_id < kNumRDMARanks
            ? env.dispatch_meta_sentinel_prev[channel_id * kNumRDMARanks + lane_id]
            : 0;
        if (lane_id < kNumRDMARanks) {
            while (true) {
                auto cur_sentinel = ld_acquire_sys_global(
                    rdma_channel_meta.recv_buffer(lane_id) + kRdmaMetaSentinelSlot);
                if (cur_sentinel > prev_meta_sentinel_at_entry) {
                    auto meta_0 = ld_acquire_sys_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                    auto meta_1 = ld_acquire_sys_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
                    auto meta_2 = ld_acquire_sys_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
                    auto meta_3 = ld_acquire_sys_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
                    int start_sum = meta_0, end_sum = meta_1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id,
                                         nvl_pack(nvl_seq, start_sum));
                    st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id,
                                         nvl_pack(nvl_seq, end_sum));

                    src_rdma_channel_prefix = meta_2;
                    auto src_rdma_channel_prefix_1 = meta_3;
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

        // Seed prev-at-entry from the persistent reader_prev_tail array.
        // Slot accumulates across iters; cached_rdma_channel_tail tracks
        // this iter's advance only. See dispatch_main_kernel kRDMASender
        // for the rationale.
        int src_rdma_rank = sm_id % kNumRDMARanks;
        uint32_t prev_rdma_channel_tail_at_entry = lane_id < kNumRDMARanks
            ? env.reader_prev_tail[channel_id * kNumRDMARanks + lane_id]
            : 0u;
        int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
        int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0, rdma_nvl_token_idx = 0;
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
            start_time = clock64();
            while (true) {
                const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
                if (env.num_max_nvl_chunked_recv_tokens - num_used_slots >= env.num_max_nvl_chunked_send_tokens)
                    break;
                uint64_t raw_head = __shfl_sync(0xffffffffu, ld_volatile_global(nvl_channel_head.buffer()), 0);
                if (nvl_seq_match(raw_head, nvl_seq))
                    cached_nvl_channel_head = nvl_unpack_value(raw_head);

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
                        cached_rdma_channel_tail =
                            static_cast<int>(static_cast<uint32_t>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank))) - prev_rdma_channel_tail_at_entry);
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

            // Two-stage prefetch pipeline: while iter `i` waits on its store,
            // iter `i+1`'s load is already in flight on the TMA unit. Forward-
            // iters that skip the TMA (not-in-dst-nvl-rank) do NOT advance the
            // stage cursor — the next forwarded iter inherits the pending head
            // slot. Pending queue is at most `kNumStages` deep.
            struct PendingForward { uint8_t* dst_shifted; };
            PendingForward pending[kNumStages] = {};
            int pending_count = 0;
            int issue_stage = 0;
            int drain_stage = 0;

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

                // Drain the oldest pending stage if both stages are in flight.
                if (pending_count == kNumStages) {
                    mbarrier_wait(tma_mbarrier(drain_stage), tma_phase[drain_stage]);
                    if (elect_one_sync())
                        tma_store_1d(tma_buffer(drain_stage), pending[drain_stage].dst_shifted, num_bytes_per_token);
                    __syncwarp();
                    tma_store_wait<0>();
                    __syncwarp();
                    drain_stage = (drain_stage + 1) % kNumStages;
                    pending_count -= 1;
                }

                // Issue load for the current token into the free stage. Runs
                // async on the TMA unit, overlapping with any in-flight store
                // from a prior iter.
                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer(issue_stage), shifted, tma_mbarrier(issue_stage), num_bytes_per_token, false);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier(issue_stage), num_bytes_per_token);
                }
                __syncwarp();
                pending[issue_stage].dst_shifted = dst_shifted;
                issue_stage = (issue_stage + 1) % kNumStages;
                pending_count += 1;

                if ((++num_tokens_sent) == env.num_max_nvl_chunked_send_tokens)
                    src_rdma_tail = i + 1;
            }

            // Drain remaining in-flight stages in FIFO order.
            while (pending_count > 0) {
                mbarrier_wait(tma_mbarrier(drain_stage), tma_phase[drain_stage]);
                if (elect_one_sync())
                    tma_store_1d(tma_buffer(drain_stage), pending[drain_stage].dst_shifted, num_bytes_per_token);
                __syncwarp();
                tma_store_wait<0>();
                __syncwarp();
                drain_stage = (drain_stage + 1) % kNumStages;
                pending_count -= 1;
            }

            if (lane_id == src_rdma_rank)
                forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);

            __syncwarp();
            if (elect_one_sync())
                st_release_sys_global(nvl_channel_tail.buffer(),
                                     nvl_pack(nvl_seq, cached_nvl_channel_tail));
        }

        __syncwarp();
        if (elect_one_sync())
            forward_channel_retired[dst_nvl_rank] = true;

        // Writeback observed cumulative tail + meta-sentinel to the persistent
        // arrays. Forwarder exits only after draining the expected count from
        // each src_rdma — so ld(rdma_channel_tail) at exit >= prev_at_entry +
        // this iter's expected count, and ld(slab[lane].slot[30]) >=
        // prev_meta_sentinel_at_entry + 1 (this iter's amo bumps slot 30 by
        // 1). atomicMax across multiple forwarder warps preserves monotonicity.
        // Stream-ordered with the next kernel's read.
        if (lane_id < kNumRDMARanks) {
            atomicmax_reader_prev_cumulative(
                env.reader_prev_tail + channel_id * kNumRDMARanks + lane_id,
                static_cast<uint32_t>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id))));
            int latest_sentinel = ld_acquire_sys_global(
                rdma_channel_meta.recv_buffer(lane_id) + kRdmaMetaSentinelSlot);
            atomicMax(env.dispatch_meta_sentinel_prev + channel_id * kNumRDMARanks + lane_id, latest_sentinel);
        }
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
        //           k_local_remaining[r], k_local_total[r]); lane-0 writes
        //           recv_src_meta (combine plumbing).
        //   Pass 2 fire (substream-end): per-warp `__threadfence_system()` +
        //           cross-NVL-receiver-warp `bar.sync 2` (for system-scope
        //           visibility of other warps' writes contributing to the same
        //           tile via base_pool stacking) → lane-0 expert-major walk
        //           over (e, src_rdma_rank); `red.release.gpu.global.add.s32`
        //           into pool_arrival_count[block_id]. Kernel A's scheduler
        //           spins until count == pool_arrival_target[block_id].
        const int src_nvl_rank = target_rank;

        int total_offset = 0;

        if (lane_id < kNumRDMARanks and lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
            total_offset = inputs.recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank - 1];

        int start_offset = 0, end_offset = 0, num_tokens_to_recv;
        auto start_time = clock64();
        while (lane_id < kNumRDMARanks) {
            uint64_t raw_start = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
            uint64_t raw_end   = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
            if (nvl_seq_match(raw_start, nvl_seq) and
                nvl_seq_match(raw_end, nvl_seq)) {
                start_offset = nvl_unpack_value(raw_start);
                end_offset   = nvl_unpack_value(raw_end);
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
        if (lane_id < kNumRDMARanks) {
            per_token_out.recv_gbl_channel_prefix_matrix[(lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] = total_offset;
        }
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
        const int* seen_for_channel = tile_signal.seen_per_substream + channel_id * num_world_ranks * E_local;

        // Eager Pass 2 fire bookkeeping: lane-0 register bitmask of `e_local`s
        // this warp has already fired pool_arrival_count for (across all
        // src_rdma_rank src_worlds). Each warp's substream binding is to a specific
        // src_nvl_rank — `e_local` alone identifies the (src_world, e) tuple
        // we may have already fired, because seen-per-substream completion
        // for a given (src_rdma, e) only fires once across the warp's life,
        // and a `(src_rdma_a, e) → fired` does not block `(src_rdma_b, e) →
        // fire` since each (src_rdma, e) gets its own atomicAdd's worth of
        // writes_in_block. So the mask is per-(src_rdma, e) — we need
        // 2D bookkeeping. Use kNumRDMARanks bitmasks of E_local bits each;
        // since NUM_MAX_LOCAL_EXPERTS = 64 and kNumRDMARanks ≤ 16, this is
        // a uint64_t[kNumRDMARanks] register array.
        uint64_t completed_mask[kNumRDMARanks] = {0};

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            start_time = clock64();
            while (true) {
                if (cached_channel_head_idx != cached_channel_tail_idx)
                    break;
                {
                    uint64_t raw_tail = __shfl_sync(0xffffffff, ld_volatile_global(nvl_channel_tail.buffer()), 0);
                    if (nvl_seq_match(raw_tail, nvl_seq))
                        cached_channel_tail_idx = nvl_unpack_value(raw_tail);
                }
                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch NVL receiver timeout (tail), channel: %d, RDMA: %d, nvl: %d, src NVL: %d\n",
                           channel_id, rdma_rank, nvl_rank, src_nvl_rank);
                    trap();
                }
            }

            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;

            // Pre-loop: prefetch the first token's hidden-bytes load into stage 0.
            // The inner-for body waits on this and then prefetches stage 1, etc.
            if (num_recv_tokens > 0) {
                int prefetch_buf = cached_channel_head_idx % env.num_max_nvl_chunked_recv_tokens;
                auto prefetch_shifted = nvl_channel_x.buffer() + prefetch_buf * num_bytes_per_token;
                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer(0), prefetch_shifted, tma_mbarrier(0), hidden_bytes);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier(0), hidden_bytes);
                }
                __syncwarp();
            }

            for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++chunk_idx, --num_tokens_to_recv) {
                const int s = chunk_idx % kNumStages;
                const int ns = (chunk_idx + 1) % kNumStages;

                int token_idx_in_buffer = (cached_channel_head_idx++) % env.num_max_nvl_chunked_recv_tokens;
                auto shifted = nvl_channel_x.buffer() + token_idx_in_buffer * num_bytes_per_token;
                auto meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes));
                int src_rdma_rank = meta.src_rdma_rank;
                int recv_token_idx = __shfl_sync(0xffffffff, total_offset, src_rdma_rank);
                (lane_id == src_rdma_rank) ? (total_offset += 1) : 0;

                int src_world = src_rdma_rank * NUM_MAX_NVL_PEERS + src_nvl_rank;
                auto topk_idx_in_msg     = reinterpret_cast<const int*>(shifted + hidden_bytes + sizeof(SourceMeta));
                auto topk_weights_in_msg = reinterpret_cast<const float*>(shifted + hidden_bytes + sizeof(SourceMeta) + shape.num_topk * sizeof(int));

                // ── Pass A: lane-0 K-loop, slot allocation. Reads topk_idx
                //    from the global NVL ring (independent of TMA). Also
                //    records `e_local_row[k]` for the per-iter eager-fire
                //    check below. ──
                int slot_row[kMaxTopK];
                int e_local_row[kMaxTopK];
                if (lane_id == 0) {
                    int* base_pool_substream = const_cast<int*>(base_pool_for_channel + src_world * E_local);
                    int* seen_for_src = warp_local_seen + src_rdma_rank * E_local;
                    for (int k = 0; k < shape.num_topk; ++k) {
                        int e_global = static_cast<int>(__ldg(topk_idx_in_msg + k));
                        int e_local = (e_global >= local_expert_begin and e_global < local_expert_end) ? e_global - local_expert_begin : -1;
                        e_local_row[k] = e_local;
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

                // ── Pass B: wait this iter's load, prefetch next, K-fanout store. ──
                mbarrier_wait(tma_mbarrier(s), tma_phase[s]);

                // Prefetch next iter's load (overlaps with the K-fanout below).
                // The prior iter's `tma_store_wait<0>` drained the stores reading
                // from `tma_buffer(ns)` two iters ago, so the buffer is free.
                if (chunk_idx + 1 < num_recv_tokens) {
                    int next_buf = cached_channel_head_idx % env.num_max_nvl_chunked_recv_tokens;
                    auto next_shifted = nvl_channel_x.buffer() + next_buf * num_bytes_per_token;
                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer(ns), next_shifted, tma_mbarrier(ns), hidden_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier(ns), hidden_bytes);
                    }
                    __syncwarp();
                }

                if (lane_id == 0) {
                    for (int k = 0; k < shape.num_topk; ++k) {
                        int slot = slot_row[k];
                        if (slot < 0) continue;
                        tma_store_1d(tma_buffer(s),
                                     pool_out.pool + static_cast<int64_t>(slot) * shape.hidden_int4,
                                     hidden_bytes, false);
                    }
                }
                __syncwarp();

                // ── Pass B (continued): per-pool-slot scalars + per-recv-token scalars. ──
                if (lane_id == 0) {
                    int k_local_total_val = 0;
                    for (int k = 0; k < shape.num_topk; ++k) {
                        int slot = slot_row[k];
                        per_token_out.recv_token_to_slots[recv_token_idx * shape.num_topk + k] = slot;
                        if (slot < 0) continue;
                        pool_out.pool_topk_weight[slot] = ld_nc_global(topk_weights_in_msg + k);
                        pool_out.pool_recv_token[slot] = recv_token_idx;
                        pool_out.pool_k_slot[slot] = k;
                        ++k_local_total_val;
                    }
                    if (k_local_total_val > 0) {
                        per_token_out.k_local_remaining[recv_token_idx] = k_local_total_val;
                        per_token_out.k_local_total[recv_token_idx] = k_local_total_val;
                    }
                }

                // Combine plumbing: recv_src_meta per recv_token.
                if (elect_one_sync())
                    st_na_global(reinterpret_cast<SourceMeta*>(per_token_out.recv_src_meta) + recv_token_idx, meta);

                // Drain this iter's K-fanout stores before stage `s` is reused
                // (two iters from now) by the next prefetch's load — and before
                // the eager-fire check below issues atomicAdds whose target
                // ordering depends on the pool writes being sys-visible.
                tma_store_wait<0>();
                __syncwarp();

                // ── Eager Pass 2 fire.
                // For each unique `e_local` this iter touched on this src_world
                // (= src_rdma × NUM_MAX_NVL_PEERS + src_nvl), check whether
                // `warp_local_seen[src_rdma][e_local]` just hit the metadata-
                // kernel-computed `seen_per_substream[c, src_world, e_local]`.
                // If yes, this warp has finished its contribution to expert
                // `e_local` in this substream: fence + `red.release.gpu.global.add.s32`
                // the writes-in-block count into pool_arrival_count[block].
                //
                // Visibility: each warp's `__threadfence_system()` before its
                // own release-add orders this warp's pool writes (K-fanout
                // TMA stores + per-pool-slot scalars) sys-visible before the
                // add lands. The release semantics of `red.release.gpu.global.add`
                // then chain with kernel A's acquire-load on the same address;
                // by the time count reaches pool_arrival_target[block] on the
                // consumer side, every contributor's fenced pool writes are
                // observable. Mirrors intranode's single-fusion pattern without
                // the cross-warp `bar.sync 2` — each warp's own fence-before-
                // release-add carries the invariant.
                if (lane_id == 0) {
                    uint64_t newly_complete = 0;
                    for (int k = 0; k < shape.num_topk; ++k) {
                        int e_local = e_local_row[k];
                        if (e_local < 0) continue;
                        uint64_t bit = 1ULL << e_local;
                        if ((completed_mask[src_rdma_rank] | newly_complete) & bit) continue;
                        int my_seen = warp_local_seen[src_rdma_rank * E_local + e_local];
                        int expected = ld_nc_global(seen_for_channel + src_world * E_local + e_local);
                        if (my_seen == expected) newly_complete |= bit;
                    }

                    if (newly_complete != 0) {
                        completed_mask[src_rdma_rank] |= newly_complete;
                        __threadfence_system();
                        for (int e_local = 0; e_local < E_local; ++e_local) {
                            if (!((newly_complete >> e_local) & 1)) continue;
                            int my_seen = warp_local_seen[src_rdma_rank * E_local + e_local];
                            int slot_start_e = base_pool_for_channel[src_world * E_local + e_local];
                            fire_pool_blocks(slot_start_e, my_seen, shape.tile_m,
                                             tile_signal.pool_arrival_count);
                        }
                    }
                }
                __syncwarp();
            }

            if (elect_one_sync())
                st_relaxed_sys_global(nvl_channel_head.buffer(),
                                     nvl_pack(nvl_seq, cached_channel_head_idx));
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

// Per-recv-token reduction shared between the kNVLAndRDMAForwarder
// (NVL→RDMA, NUM_MAX_NVL_PEERS contributions per token, TMA-staged) and the
// kRDMAReceiver (RDMA→origin, kNumRDMARanks contributions per token,
// non-TMA). Reduces hidden bf16 + per-(t, k) topk weights across the K
// contributing source ranks for one recv-token.
//
// `is_token_in_rank` = "this lane's rank actually contributed to this
// token" — derived from `head_idx >= 0` (sentinel-encoded, set by
// encode_combine_heads). Lane `i` holds the head + flag for rank `i`;
// the helper warp-shuffles to assemble the per-token (rank, slot) topk
// list.
//
// Streaming variant: biases (bias_0/bias_1) are dropped vs the legacy
// `combine_token`. The intranode streaming combine never had biases either;
// they were a legacy DeepEP non-streaming concept.
template <int kNumRanks,
          typename dtype_t,
          int kMaxNumRanks,
          bool kUseTMA,
          int kNumStages,
          int kNumTMALoadBytes = 0,
          bool kSendTopkWeights = true,
          typename GetAddrFn,
          typename ReceiveTWFn>
__device__ int combine_token(bool is_token_in_rank,
                             int head_idx,
                             int lane_id,
                             int hidden_int4,
                             int num_topk,
                             int4* combined_row,
                             float* combined_topk_weights,
                             int num_max_recv_tokens,
                             const GetAddrFn& get_addr_fn,
                             const ReceiveTWFn& recv_tw_fn,
                             uint8_t* smem_ptr,
                             uint32_t (&tma_phase)[kNumStages]) {
    constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

    // Lane i holds (is_token_in_rank, head_idx) for rank i. Warp-shuffle to
    // assemble the per-token list of contributing (rank, slot) pairs.
    EP_STATIC_ASSERT(kMaxNumRanks <= 32, "Too many ranks");
    int num_topk_ranks = 0, topk_ranks[kMaxNumRanks], slot_indices[kMaxNumRanks];
    #pragma unroll
    for (int i = 0; i < kNumRanks; ++i)
        if (__shfl_sync(0xffffffff, is_token_in_rank, i)) {
            slot_indices[num_topk_ranks] = __shfl_sync(0xffffffff, head_idx, i) % num_max_recv_tokens;
            topk_ranks[num_topk_ranks++] = i;
        }
    EP_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);
    EP_STATIC_ASSERT(kNumStages == 2, "Only support 2 stages");

    if constexpr (kUseTMA) {
        constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16;
        EP_DEVICE_ASSERT(hidden_int4 % 32 == 0);

        auto tma_load_buffer = [=](const int& i, const int& j) -> int4* {
            return reinterpret_cast<int4*>(smem_ptr + i * kNumTMABufferBytesPerStage + j * kNumTMALoadBytes);
        };
        auto tma_store_buffer = [=](const int& i) -> int4* {
            return reinterpret_cast<int4*>(smem_ptr + i * kNumTMABufferBytesPerStage + NUM_MAX_NVL_PEERS * kNumTMALoadBytes);
        };
        auto tma_mbarrier = [=](const int& i) -> uint64_t* {
            return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + (NUM_MAX_NVL_PEERS + 1) * kNumTMALoadBytes);
        };

        // Prefetch stage 0
        if (lane_id < num_topk_ranks)
            tma_load_1d(tma_load_buffer(0, lane_id),
                        get_addr_fn(topk_ranks[lane_id], slot_indices[lane_id], 0),
                        tma_mbarrier(0), kNumTMALoadBytes);
        mbarrier_arrive_and_expect_tx(tma_mbarrier(0), lane_id < num_topk_ranks ? kNumTMALoadBytes : 0);
        __syncwarp();

        for (int shifted = 0, iter = 0; shifted < hidden_int4; shifted += 32, iter += 1) {
            const int stage_idx = iter % kNumStages;
            const int next_stage_idx = (iter + 1) % kNumStages;

            if (shifted + 32 < hidden_int4) {
                if (lane_id < num_topk_ranks)
                    tma_load_1d(tma_load_buffer(next_stage_idx, lane_id),
                                get_addr_fn(topk_ranks[lane_id], slot_indices[lane_id], shifted + 32),
                                tma_mbarrier(next_stage_idx),
                                kNumTMALoadBytes);
                mbarrier_arrive_and_expect_tx(tma_mbarrier(next_stage_idx), lane_id < num_topk_ranks ? kNumTMALoadBytes : 0);
                __syncwarp();
            }

            mbarrier_wait(tma_mbarrier(stage_idx), tma_phase[stage_idx]);
            float values[kDtypePerInt4] = {0};
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j) {
                auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(tma_load_buffer(stage_idx, j) + lane_id);
                #pragma unroll
                for (int k = 0; k < kDtypePerInt4; ++k)
                    values[k] += static_cast<float>(recv_value_dtypes[k]);
            }

            tma_store_wait<kNumStages - 1>();

            auto out_dtypes = reinterpret_cast<dtype_t*>(tma_store_buffer(stage_idx) + lane_id);
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++j)
                out_dtypes[j] = static_cast<dtype_t>(values[j]);
            tma_store_fence();
            __syncwarp();

            if (elect_one_sync())
                tma_store_1d(tma_store_buffer(stage_idx), combined_row + shifted, kNumTMALoadBytes);
            __syncwarp();
        }

        tma_store_wait<0>();
    } else {
        #pragma unroll
        for (int i = lane_id; i < hidden_int4; i += 32) {
            int4 recv_value_int4[kMaxNumRanks];
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j)
                recv_value_int4[j] = ld_nc_global(get_addr_fn(topk_ranks[j], slot_indices[j], i));

            float values[kDtypePerInt4] = {0};
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j) {
                auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
                #pragma unroll
                for (int k = 0; k < kDtypePerInt4; ++k)
                    values[k] += static_cast<float>(recv_value_dtypes[k]);
            }

            int4 out_int4;
            auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++j)
                out_dtypes[j] = static_cast<dtype_t>(values[j]);
            st_na_global(combined_row + i, out_int4);
        }
    }

    // Reduce per-(t, k) topk weights — bwd only. In fwd, kernel Y already
    // pre-multiplies pool_topk_weight[slot] per row before the atomic
    // scatter into o[r], so out[t,:H] = Σ_k w_k·y_k is reconstructed by
    // the data reduce above; the per-K wire payload is not shipped (see
    // combine_main_kernel's num_bytes_per_token / sender pack / both
    // forwarder + receiver reduce sites). Bwd combine_grads still ships
    // and reduces dL/dweight[t, k] per K from each source.
    if constexpr (kSendTopkWeights) {
        if (lane_id < num_topk) {
            float value = 0;
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);
            st_na_global(combined_topk_weights + lane_id, value);
        }
    }

    return topk_ranks[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// encode_combine_heads — pre-combine fixup. See api.cuh for the architectural
// contract; this is the kernel body. Block layout (single warp per block):
//
//   blocks [0, num_channels):
//             reverse-order sentinel encoding of `combined_rdma_head`
//             ([num_combined_tokens, num_rdma_ranks]). One block per
//             channel; lane_id = dst_rdma_rank. Walks the channel's
//             recv-token range in reverse and replaces `< 0` entries
//             with `-last_head - 1`.
//   blocks [num_channels, num_channels * (1 + num_rdma_ranks)):
//             reverse-order sentinel encoding of `combined_nvl_head`
//             ([num_rdma_recv_tokens, NUM_MAX_NVL_PEERS]). One block per
//             (channel, dst_rdma_rank); TMA-batched along the token axis,
//             lane_id = NVL peer. Same `last_head` reverse-cumulative
//             pattern, just along the NVL lane.
//
// Block-per-(channel | (channel, dst_rdma_rank)) avoids the warp-per-
// channel packing that capped num_channels ≤ 32 (one block ≤ 32 warps).
// Independent work items per block — no cross-block sync.
//
// TODO(fuse-into-combine): the reverse-scan logic in this kernel could
// move into `combine_main_kernel` at the same per-(channel)/per-(channel,
// dst_rdma_rank) granularity — combine's kRDMAReceiver and kNVLAndRDMA
// Forwarder warps already own those scopes. Fusing eliminates the
// separate kernel launch (~10-15 µs), the HBM round-trip on send_*_head
// (combine could read once, encode in registers/SMEM, consume), and the
// FIFO-implies-gate reasoning currently linking the two kernels. The four
// sender-side / prefix-matrix sentinels we acquire here would move into
// combine_main verbatim. Cost: ~50-80 LOC of reverse-scan logic + a SMEM
// scratch in combine_main (which already runs near its SMEM budget — need
// to measure). Tracked at Stage 1 acceptance time as a follow-up; see
// markdowns/cdmc-partial-fix.md §"Stage 1 — Prefix-matrix sentinel".
// ─────────────────────────────────────────────────────────────────────────────
template <int kNumTMABytesPerWarp>
__global__ void encode_combine_heads_kernel(
    int* combined_rdma_head,
    int num_combined_tokens,
    int num_channels,
    const int* rdma_channel_prefix_matrix,
    const int* rdma_rank_prefix_sum,
    int* combined_nvl_head,
    int num_ranks) {
    auto block_id = static_cast<int>(blockIdx.x);
    auto lane_id = get_lane_id();

    auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

    if (block_id < num_channels) {
        // rdma_head reverse pass for `channel = block_id`. lane_id = dst_rdma_rank.
        EP_DEVICE_ASSERT(num_rdma_ranks <= 32);

        if (lane_id < num_rdma_ranks) {
            int channel = block_id;
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, channel, token_start_idx, token_end_idx);

            // Sentinel for "no real head ahead"; any combine receiver that hits
            // this is past the channel's tail.
            int last_head = kReverseScanSentinel;
            for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
                auto current_head = __ldg(combined_rdma_head + token_idx * num_rdma_ranks + lane_id);
                if (current_head < 0) {
                    combined_rdma_head[token_idx * num_rdma_ranks + lane_id] = -last_head - 1;
                } else {
                    last_head = current_head;
                }
            }
        }

    } else {
        // nvl_head reverse pass for one (channel, dst_rdma_rank). lane_id = NVL peer.
        EP_DEVICE_ASSERT(rdma_channel_prefix_matrix != nullptr and rdma_rank_prefix_sum != nullptr);
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Too many NVL peers");

        int nvl_block = block_id - num_channels;
        int channel = nvl_block % num_channels;
        int dst_rdma_rank = nvl_block / num_channels;

        constexpr int tma_batch_size = kNumTMABytesPerWarp - sizeof(uint64_t);
        constexpr int num_bytes_per_token = sizeof(int) * NUM_MAX_NVL_PEERS;
        constexpr int num_tokens_per_batch = tma_batch_size / num_bytes_per_token;
        EP_STATIC_ASSERT(num_bytes_per_token % 16 == 0, "num_bytes_per_token should be divisible by 16");

        extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
        auto tma_buffer = smem_tma_buffer;
        auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + tma_batch_size);
        uint32_t tma_phase = 0;
        if (elect_one_sync()) {
            mbarrier_init(tma_mbarrier, 1);
            fence_barrier_init();
        }
        __syncwarp();

        int token_start_idx = channel == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel - 1];
        int token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel];
        int shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
        token_start_idx += shift, token_end_idx += shift;

        int last_head = kReverseScanSentinel;
        for (int batch_end_idx = token_end_idx; batch_end_idx > token_start_idx; batch_end_idx -= num_tokens_per_batch) {
            auto batch_start_idx = max(token_start_idx, batch_end_idx - num_tokens_per_batch);

            if (elect_one_sync()) {
                tma_load_1d(tma_buffer,
                            combined_nvl_head + batch_start_idx * NUM_MAX_NVL_PEERS,
                            tma_mbarrier,
                            (batch_end_idx - batch_start_idx) * num_bytes_per_token);
                mbarrier_arrive_and_expect_tx(tma_mbarrier, (batch_end_idx - batch_start_idx) * num_bytes_per_token);
            }
            mbarrier_wait(tma_mbarrier, tma_phase);
            __syncwarp();

            for (int token_idx = batch_end_idx - 1; token_idx >= batch_start_idx; --token_idx) {
                if (lane_id < NUM_MAX_NVL_PEERS) {
                    auto current_head =
                        reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id];
                    if (current_head < 0) {
                        reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id] =
                            -last_head - 1;
                    } else {
                        last_head = current_head;
                    }
                }
            }
            tma_store_fence();
            __syncwarp();

            if (elect_one_sync())
                tma_store_1d(tma_buffer,
                             combined_nvl_head + batch_start_idx * NUM_MAX_NVL_PEERS,
                             (batch_end_idx - batch_start_idx) * num_bytes_per_token);
            tma_store_wait<0>();
            __syncwarp();
        }
    }
}

void encode_combine_heads(int hidden_int4,
                          int num_topk,
                          int num_ranks,
                          int num_channels,
                          int num_combined_tokens,
                          int* combined_rdma_head,
                          const int* rdma_channel_prefix_matrix,
                          const int* rdma_rank_prefix_sum,
                          int* combined_nvl_head,
                          cudaStream_t stream) {
    // Block-per-(channel | (channel, dst_rdma_rank)), 1 warp per block.
    // 4096 B SMEM per nvl_head block packs ~127 tokens/batch
    // (= (4096 − 8) / (4 × NUM_MAX_NVL_PEERS)). rdma_head blocks don't use
    // SMEM but share the same launch attribute (cheap). Grid scales linearly
    // with num_channels and num_rdma_ranks — no per-block warp cap.
    constexpr int kNumTMABytesPerWarp = 4096;
    constexpr int num_threads = 32;
    const int smem_size = kNumTMABytesPerWarp;
    const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    const int num_blocks = num_channels * (1 + num_rdma_ranks);

    EP_HOST_ASSERT(num_channels > 0 and num_rdma_ranks > 0);

    auto kernel = encode_combine_heads_kernel<kNumTMABytesPerWarp>;
    SETUP_LAUNCH_CONFIG(num_blocks, num_threads, stream);
    SET_SHARED_MEMORY_FOR_TMA(kernel);
    LAUNCH_KERNEL(&cfg, kernel,
                  combined_rdma_head, num_combined_tokens, num_channels,
                  rdma_channel_prefix_matrix, rdma_rank_prefix_sum,
                  combined_nvl_head,
                  num_ranks);
}

// ─────────────────────────────────────────────────────────────────────────────
// dispatch_grads_main_kernel — bwd dispatch, mirrors `dispatch_main_kernel`
// shape (kRDMASender / kRDMASenderCoordinator / kRDMAAndNVLForwarder /
// kForwarderCoordinator / kNVLReceivers) but for the bwd path: ships
// dL/dy[t] origin → expert ranks via the same RDMA + NVL hierarchy as fwd
// dispatch. The receiver SKIPS Pass A — slot lookups come from
// `recv_token_to_slots[r, :K]` (persisted by fwd Pass B). Pass 2 fires
// `red.release.gpu.global.add.s32` into `bwd_dispatch_arrival_count[block]`;
// the bwd-Y scheduler spins until count == `pool_arrival_target[block]`
// (re-uses the same target fwd uses).
//
// Wire format reuses fwd's per-token bytes layout (data + SourceMeta +
// topk_idx + topk_weights bytes). Bwd writes only the data + SourceMeta
// regions; topk_* bytes are untouched (zero from encode_combine_heads
// cleanup). Saves layout-divergence complexity at the cost of ~5–10%
// wasted RDMA + NVL bandwidth per token.
// ─────────────────────────────────────────────────────────────────────────────
template <int kNumRDMARanks,
          int kNumTMABytesPerWarp,
          int kNumDispatchRDMASenderWarps,
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
__global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32), 1)
dispatch_grads_main_kernel(DispatchGradsIO io,
                           DispatchGradsRouting routing,
                           DispatchGradsTileSignal tile_signal,
                           DispatchGradsShape shape,
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
    (void)local_expert_begin; (void)local_expert_end;

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

    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    auto hidden_bytes = shape.hidden_int4 * sizeof(int4);
    auto num_bytes_per_token = get_num_bytes_per_token(shape.hidden_int4, shape.num_topk, shape.num_topk);
    auto rdma_channel_data = SymBuffer<uint8_t>(env.rdma_buffer_ptr,
        env.num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
    align_meta_base_to_l2_line(env.rdma_buffer_ptr);
    auto rdma_channel_meta = SymBuffer<int>(env.rdma_buffer_ptr,
        kRdmaMetaSlabInts, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(env.rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(env.rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

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
        AsymBuffer<uint64_t>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_end = AsymBuffer<uint64_t>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                                      .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_head =
        AsymBuffer<uint64_t>(rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, ws_rr_rank).advance_also(ws_rr_buffer_ptr);
    auto nvl_channel_tail =
        AsymBuffer<uint64_t>(ws_rr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);

    // NVL gen-stamp tag with the bwd phase bit set in the LSB. Phase-distinct
    // from fwd `dispatch_main_kernel`'s tag (low bit 0) so fwd's leftover on
    // the shared dispatch NVL ring doesn't alias bwd's current-iter reader,
    // even though both kernels see the same `tile_signal.dispatch_seq` value
    // within a layer (see `stream_moe.py`: fwd dispatch and bwd
    // dispatch_grads share the per-call `dispatch_seq` int).
    const int64_t nvl_seq = (tile_signal.dispatch_seq << 1) | 1;

    __shared__ int rdma_send_channel_lock[kNumRDMARanks];
    __shared__ int rdma_send_channel_tail[kNumRDMARanks];
    __shared__ uint32_t rdma_send_channel_window[kNumRDMARanks];
    auto sync_rdma_sender_smem = []() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumDispatchRDMASenderWarps + 1) * 32)); };

    // Two-stage TMA pipeline (mirrors `dispatch_main_kernel`). See the
    // companion comment there for the layout / why-this-works.
    constexpr int kNumStages = 2;
    const int kStageStride = (num_bytes_per_token + 15) & ~15;
    extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
    auto tma_buffer = [=](int s) {
        return smem_tma_buffer + target_rank * kNumTMABytesPerWarp + s * kStageStride;
    };
    auto tma_mbarrier = [=](int s) {
        return reinterpret_cast<uint64_t*>(
            smem_tma_buffer + target_rank * kNumTMABytesPerWarp
            + kNumStages * kStageStride + s * static_cast<int>(sizeof(uint64_t)));
    };
    uint32_t tma_phase[kNumStages] = {0, 0};
    if ((warp_role == WarpRole::kRDMAAndNVLForwarder or warp_role == WarpRole::kNVLReceivers) and elect_one_sync()) {
        #pragma unroll
        for (int s = 0; s < kNumStages; ++s)
            mbarrier_init(tma_mbarrier(s), 1);
        fence_barrier_init();
        EP_DEVICE_ASSERT(kNumStages * kStageStride + kNumStages * static_cast<int>(sizeof(uint64_t)) <= kNumTMABytesPerWarp);
    }
    __syncwarp();

    __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
    __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
    auto sync_forwarder_smem = []() { asm volatile("barrier.sync 1, %0;" ::"r"((NUM_MAX_NVL_PEERS + 1) * 32)); };


    if (warp_role == WarpRole::kRDMASender) {
        // Sender: identical structure to fwd dispatch, but reads `io.dL_dy`
        // and skips topk_idx / topk_weights metadata packing (those bytes
        // are untouched in the wire — receiver doesn't read them for bwd).
        int token_start_idx, token_end_idx;
        get_channel_task_range(shape.num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // See dispatch_main_kernel kRDMASender for the sentinel-amo / 128B
        // L2-line packing rationale. Mirrored here verbatim — fwd dispatch
        // and bwd dispatch_grads share the meta SymBuffer + the
        // dispatch_meta_sentinel_prev array (both run on streams.dispatch).
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            auto dst_ptr =
                dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
            if (lane_id < NUM_MAX_NVL_PEERS) {
                dst_ptr[lane_id] =
                    channel_id == 0
                        ? 0
                        : routing.gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_channels + channel_id - 1];
            } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] =
                    routing.gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id - NUM_MAX_NVL_PEERS) * num_channels +
                                                      channel_id];
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = channel_id == 0 ? 0 : routing.rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
                dst_ptr[lane_id] = routing.rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            }
            __syncwarp();

            if (dst_rdma_rank != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                                  translate_dst_rdma_rank<false>(dst_rdma_rank, nvl_rank),
                                                  channel_id, lane_id, 0);
                if (lane_id == 0) {
                    nvshmemi_ibgda_amo_nonfetch_add(
                        rdma_channel_meta.recv_buffer(rdma_rank) + kRdmaMetaSentinelSlot,
                        1,
                        translate_dst_rdma_rank<false>(dst_rdma_rank, nvl_rank),
                        channel_id);
                }
            } else if (lane_id == 0) {
                __threadfence();
                atomicAdd(dst_ptr + kRdmaMetaSentinelSlot, 1);
            }
        }
        sync_rdma_sender_smem();

        // See dispatch_main_kernel kRDMASender for the prev-at-entry rationale.
        int64_t token_idx;
        uint32_t prev_rdma_channel_head_at_entry = lane_id < kNumRDMARanks
            ? env.reader_prev_head[channel_id * kNumRDMARanks + lane_id]
            : 0u;
        int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;
        auto send_buffer = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);
        for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
            uint64_t is_token_in_rank_uint64 = 0;
            if (lane_id < kNumRDMARanks) {
                is_token_in_rank_uint64 =
                    __ldg(reinterpret_cast<const uint64_t*>(io.is_token_in_rank + token_idx * num_world_ranks + lane_id * NUM_MAX_NVL_PEERS));
                global_rdma_tail_idx += (is_token_in_rank_uint64 != 0);
            }
            __syncwarp();

            if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
                continue;
            auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;

            auto start_time = clock64();
            while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= env.num_max_rdma_chunked_recv_tokens) {
                cached_rdma_channel_head =
                    static_cast<int>(static_cast<uint32_t>(ld_acquire_sys_global(rdma_channel_head.buffer(lane_id))) - prev_rdma_channel_head_at_entry);
                if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch_grads RDMA sender timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA lane: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, cached_rdma_channel_head, rdma_tail_idx);
                    trap();
                }
            }
            __syncwarp();

            // Build per-(dst_rdma_rank) destination buffer pointers.
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

            // Copy `dL/dy` into symmetric send buffers.
            auto st_broadcast = [=](const int key, const int4& value) {
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);
            };
            UNROLLED_WARP_COPY(5, lane_id, shape.hidden_int4, 0, io.dL_dy + token_idx * shape.hidden_int4, ld_nc_global, st_broadcast);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + shape.hidden_int4;

            // Copy SourceMeta — needed by receiver to derive src_rdma_rank.
            // Skip topk_idx / topk_weights writes (bwd doesn't use them).
            if (lane_id < num_topk_ranks)
                st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);
            __syncwarp();

            // Release the transaction in the window.
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

        // Writeback observed cumulative head; see dispatch_main_kernel
        // kRDMASender for rationale.
        if (lane_id < kNumRDMARanks) {
            atomicmax_reader_prev_cumulative(
                env.reader_prev_head + channel_id * kNumRDMARanks + lane_id,
                static_cast<uint32_t>(ld_acquire_sys_global(rdma_channel_head.buffer(lane_id))));
        }
    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
        // RDMA sender coordinator — same as fwd: issue chunked RDMA puts to
        // dst_rdma_ranks as windows fill up. Per-channel num_tokens_to_send
        // comes from `recv_rdma_channel_prefix_matrix` (recv-side: from this
        // rank's perspective, "tokens this rank's RDMA inbox holds for
        // dst_rdma_rank in this channel"). For bwd, this is symmetric — bwd
        // sends from the same per-channel partition fwd dispatch did.
        EP_DEVICE_ASSERT(env.num_max_rdma_chunked_recv_tokens % env.num_max_rdma_chunked_send_tokens == 0);

        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_lock[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_tail[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_window[lane_id] = 0) : 0;

        sync_rdma_sender_smem();

        int num_tokens_to_send = 0;
        if (lane_id < kNumRDMARanks) {
            num_tokens_to_send = routing.rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
            if (channel_id > 0)
                num_tokens_to_send -= routing.rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];
        }

        int last_issued_tail = 0;
        auto start_time = clock64();
        while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                printf("DeepEP dispatch_grads RDMA sender coordinator timeout, channel: %d, IB: %d, nvl %d, dst IB: %d, tail: %d, remaining: %d\n",
                       channel_id, rdma_rank, nvl_rank, lane_id, last_issued_tail, num_tokens_to_send);
                trap();
            }

            for (int i = 0, synced_num_tokens_to_send; i < kNumRDMARanks; ++i) {
                int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;
                synced_num_tokens_to_send = __shfl_sync(0xffffffff, num_tokens_to_send, dst_rdma_rank);
                if (synced_num_tokens_to_send == 0)
                    continue;

                // DeepEP commit 77f97f7 ("Fix the tail loading issue"): broadcast
                // lane-0's read so all lanes agree on processed_tail. Without
                // this, lane-divergent reads of rdma_send_channel_tail cause
                // the coordinator to make inconsistent progress decisions and
                // stall on iter 2+ of multi-iter workloads. The fix is already
                // present at line 1129 in dispatch_main_kernel; this is the bwd
                // copy that missed it.
                int processed_tail = __shfl_sync(0xffffffff, ld_acquire_cta(rdma_send_channel_tail + dst_rdma_rank), 0);
                int synced_last_issued_tail = __shfl_sync(0xffffffff, last_issued_tail, dst_rdma_rank);
                int num_tokens_processed = processed_tail - synced_last_issued_tail;
                if (num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < env.num_max_rdma_chunked_send_tokens)
                    continue;

                int num_tokens_in_chunk = min(num_tokens_processed, env.num_max_rdma_chunked_send_tokens);
                if (dst_rdma_rank != rdma_rank) {
                    const size_t num_bytes_per_msg = num_tokens_in_chunk * num_bytes_per_token;
                    auto src_slot_idx = synced_last_issued_tail % env.num_max_rdma_chunked_recv_tokens;
                    const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + src_slot_idx * num_bytes_per_token);
                    const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + src_slot_idx * num_bytes_per_token);
                    nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                      translate_dst_rdma_rank<false>(dst_rdma_rank, nvl_rank),
                                                      channel_id, lane_id, 0);
                } else {
                    memory_fence();
                }

                __syncwarp();
                if (lane_id == 0) {
                    nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank),
                                                    num_tokens_in_chunk,
                                                    translate_dst_rdma_rank<false>(dst_rdma_rank, nvl_rank),
                                                    channel_id,
                                                    dst_rdma_rank == rdma_rank);
                }

                if (lane_id == dst_rdma_rank) {
                    last_issued_tail += num_tokens_in_chunk;
                    num_tokens_to_send -= num_tokens_in_chunk;
                }
            }
        }
    } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
        // RDMA consumers and NVL producers. Bulk-copy stage — no slot logic,
        // no per-(c, src, e) bookkeeping. Wire format on NVL ring is the same
        // as fwd dispatch (data + SourceMeta + unused topk_idx + topk_weights
        // bytes). Bwd doesn't write back to `recv_rdma_channel_prefix_matrix`
        // or `send_nvl_head` — those were populated by fwd dispatch and live
        // on the StreamingHandle. Otherwise this is verbatim from fwd.
        const auto dst_nvl_rank = target_rank;

        int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;
        EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
        auto start_time = clock64();
        // See dispatch_main_kernel forwarder for the sentinel-amo rationale.
        int prev_meta_sentinel_at_entry = lane_id < kNumRDMARanks
            ? env.dispatch_meta_sentinel_prev[channel_id * kNumRDMARanks + lane_id]
            : 0;
        if (lane_id < kNumRDMARanks) {
            while (true) {
                // HCA-written slot — force a system-scope memory fence before
                // each spin read. .sys-acquire alone establishes happens-before
                // ordering but doesn't guarantee L2 invalidation against PCIe
                // writes from the HCA on every spin iteration; without the
                // explicit fence the GPU keeps the cache line hot in L2 and
                // misses the HCA's atomic write entirely (iter 2+ symptom on
                // 4-node, peer-3-specific stall on GPU 7).
                // TODO(jaimec00): revisit if this is necessary / we can do something cheaper
                __threadfence_system();
                auto cur_sentinel = ld_acquire_sys_global(
                    rdma_channel_meta.recv_buffer(lane_id) + kRdmaMetaSentinelSlot);
                if (cur_sentinel > prev_meta_sentinel_at_entry) {
                    auto meta_0 = ld_acquire_sys_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                    auto meta_1 = ld_acquire_sys_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
                    auto meta_2 = ld_acquire_sys_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
                    auto meta_3 = ld_acquire_sys_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
                    int start_sum = meta_0, end_sum = meta_1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id,
                                         nvl_pack(nvl_seq, start_sum));
                    st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id,
                                         nvl_pack(nvl_seq, end_sum));

                    src_rdma_channel_prefix = meta_2;
                    auto src_rdma_channel_prefix_1 = meta_3;
                    num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
                    src_rdma_channel_prefix += lane_id == 0 ? 0 : routing.recv_rdma_rank_prefix_sum[lane_id - 1];
                    EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);
                    break;
                }

                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch_grads forwarder timeout (RDMA meta), channel: %d, RDMA: %d, nvl: %d, src RDMA lane: %d, dst NVL: %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, dst_nvl_rank);
                    trap();
                }
            }
        }
        __syncwarp();
        sync_forwarder_smem();

        // See dispatch_main_kernel forwarder for the prev-at-entry rationale.
        int src_rdma_rank = sm_id % kNumRDMARanks;
        uint32_t prev_rdma_channel_tail_at_entry = lane_id < kNumRDMARanks
            ? env.reader_prev_tail[channel_id * kNumRDMARanks + lane_id]
            : 0u;
        int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
        int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0;
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
            start_time = clock64();
            while (true) {
                const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
                if (env.num_max_nvl_chunked_recv_tokens - num_used_slots >= env.num_max_nvl_chunked_send_tokens)
                    break;
                uint64_t raw_head = __shfl_sync(0xffffffffu, ld_volatile_global(nvl_channel_head.buffer()), 0);
                if (nvl_seq_match(raw_head, nvl_seq))
                    cached_nvl_channel_head = nvl_unpack_value(raw_head);

                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch_grads forwarder timeout (NVL check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank);
                    trap();
                }
            }

            start_time = clock64();
            while (true) {
                src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
                if (__shfl_sync(0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
                    if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
                        cached_rdma_channel_tail =
                            static_cast<int>(static_cast<uint32_t>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank))) - prev_rdma_channel_tail_at_entry);
                    if (__shfl_sync(0xffffffff, cached_rdma_channel_tail > cached_rdma_channel_head, src_rdma_rank))
                        break;
                }
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf("DeepEP dispatch_grads forwarder timeout (RDMA check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank);
                    trap();
                }
            }
            auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);
            auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);

            // Two-stage prefetch pipeline (mirrors fwd dispatch forwarder).
            struct PendingForward { uint8_t* dst_shifted; };
            PendingForward pending[kNumStages] = {};
            int pending_count = 0;
            int issue_stage = 0;
            int drain_stage = 0;

            for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++i) {
                auto rdma_slot_idx = i % env.num_max_rdma_chunked_recv_tokens;
                auto shifted = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_token;
                auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes));
                lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;
                bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);
                if (not is_in_dst_nvl_rank)
                    continue;

                int dst_slot_idx = (cached_nvl_channel_tail++) % env.num_max_nvl_chunked_recv_tokens;
                auto dst_shifted = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;

                if (pending_count == kNumStages) {
                    mbarrier_wait(tma_mbarrier(drain_stage), tma_phase[drain_stage]);
                    if (elect_one_sync())
                        tma_store_1d(tma_buffer(drain_stage), pending[drain_stage].dst_shifted, num_bytes_per_token);
                    __syncwarp();
                    tma_store_wait<0>();
                    __syncwarp();
                    drain_stage = (drain_stage + 1) % kNumStages;
                    pending_count -= 1;
                }

                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer(issue_stage), shifted, tma_mbarrier(issue_stage), num_bytes_per_token, false);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier(issue_stage), num_bytes_per_token);
                }
                __syncwarp();
                pending[issue_stage].dst_shifted = dst_shifted;
                issue_stage = (issue_stage + 1) % kNumStages;
                pending_count += 1;

                if ((++num_tokens_sent) == env.num_max_nvl_chunked_send_tokens)
                    src_rdma_tail = i + 1;
            }

            while (pending_count > 0) {
                mbarrier_wait(tma_mbarrier(drain_stage), tma_phase[drain_stage]);
                if (elect_one_sync())
                    tma_store_1d(tma_buffer(drain_stage), pending[drain_stage].dst_shifted, num_bytes_per_token);
                __syncwarp();
                tma_store_wait<0>();
                __syncwarp();
                drain_stage = (drain_stage + 1) % kNumStages;
                pending_count -= 1;
            }

            if (lane_id == src_rdma_rank)
                forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);

            __syncwarp();
            if (elect_one_sync())
                st_release_sys_global(nvl_channel_tail.buffer(),
                                     nvl_pack(nvl_seq, cached_nvl_channel_tail));
        }

        __syncwarp();
        if (elect_one_sync())
            forward_channel_retired[dst_nvl_rank] = true;

        // Writeback observed cumulative tail + meta-sentinel to the persistent
        // arrays. Forwarder exits only after draining the expected count from
        // each src_rdma — so ld(rdma_channel_tail) at exit >= prev_at_entry +
        // this iter's expected count, and ld(slab[lane].slot[30]) >=
        // prev_meta_sentinel_at_entry + 1 (this iter's amo bumps slot 30 by
        // 1). atomicMax across multiple forwarder warps preserves monotonicity.
        // Stream-ordered with the next kernel's read.
        if (lane_id < kNumRDMARanks) {
            atomicmax_reader_prev_cumulative(
                env.reader_prev_tail + channel_id * kNumRDMARanks + lane_id,
                static_cast<uint32_t>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id))));
            int latest_sentinel = ld_acquire_sys_global(
                rdma_channel_meta.recv_buffer(lane_id) + kRdmaMetaSentinelSlot);
            atomicMax(env.dispatch_meta_sentinel_prev + channel_id * kNumRDMARanks + lane_id, latest_sentinel);
        }
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
        // ── kNVLReceivers (bwd): drain NVL ring, derive recv_token_id from
        // iteration order (per-src_rdma_rank counter), look up
        // `recv_token_to_slots[r, :K]`, K-fanout-write `dL_do_pool[slot]`.
        // Pass 2 fires `bwd_dispatch_arrival_count` using `seen_per_substream`
        // from the routing struct.
        const int src_nvl_rank = target_rank;

        int total_offset = 0;

        // `recv_gbl_channel_prefix_matrix[src_world, channel]` was written by
        // fwd dispatch's NVL receiver as `recv_gbl_rank_prefix_sum[src_world-1]
        // + per-channel start_offset` — the starting recv_token_id for this
        // (src_world, channel) substream's tokens in recv-x. Read directly
        // (vs fwd's runtime derivation from prefix-announce).
        if (lane_id < kNumRDMARanks) {
            int src_world = lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank;
            total_offset = routing.recv_gbl_channel_prefix_matrix[src_world * num_channels + channel_id];
        }
        __syncwarp();

        int num_tokens_to_recv = 0;
        int start_offset = 0, end_offset = 0;
        auto start_time = clock64();
        while (lane_id < kNumRDMARanks) {
            uint64_t raw_start = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
            uint64_t raw_end   = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
            if (nvl_seq_match(raw_start, nvl_seq) and
                nvl_seq_match(raw_end, nvl_seq)) {
                start_offset = nvl_unpack_value(raw_start);
                end_offset   = nvl_unpack_value(raw_end);
                break;
            }
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("DeepEP dispatch_grads NVL receiver timeout (prefix), channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, src nvl: %d\n",
                       channel_id, rdma_rank, nvl_rank, lane_id, src_nvl_rank);
                trap();
            }
        }
        num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

        const int* seen_for_channel = routing.seen_per_substream + channel_id * num_world_ranks * E_local;
        const int* base_pool_for_channel = routing.base_pool + channel_id * num_world_ranks * E_local;

        // Per-warp Pass-A-equivalent counter table for eager-fire bookkeeping,
        // mirrors fwd `warp_local_seen`. Placed in SMEM right after the
        // NUM_MAX_NVL_PEERS TMA stage slabs. Lane 0 reads/writes it.
        int* warp_local_seen = reinterpret_cast<int*>(
            smem_tma_buffer + NUM_MAX_NVL_PEERS * kNumTMABytesPerWarp
            + target_rank * kNumRDMARanks * E_local * static_cast<int>(sizeof(int)));
        for (int i = lane_id; i < kNumRDMARanks * E_local; i += 32)
            warp_local_seen[i] = 0;
        __syncwarp();

        // Eager Pass 2 fire dedup mask, per-src_rdma_rank (see fwd dispatch
        // for the rationale comment).
        uint64_t completed_mask[kNumRDMARanks] = {0};

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            start_time = clock64();
            while (true) {
                if (cached_channel_head_idx != cached_channel_tail_idx)
                    break;
                {
                    uint64_t raw_tail = __shfl_sync(0xffffffff, ld_volatile_global(nvl_channel_tail.buffer()), 0);
                    if (nvl_seq_match(raw_tail, nvl_seq))
                        cached_channel_tail_idx = nvl_unpack_value(raw_tail);
                }
                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch_grads NVL receiver timeout (tail), channel: %d, RDMA: %d, nvl: %d, src NVL: %d\n",
                           channel_id, rdma_rank, nvl_rank, src_nvl_rank);
                    trap();
                }
            }

            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;

            // Pre-loop: prefetch first token's load into stage 0.
            if (num_recv_tokens > 0) {
                int prefetch_buf = cached_channel_head_idx % env.num_max_nvl_chunked_recv_tokens;
                auto prefetch_shifted = nvl_channel_x.buffer() + prefetch_buf * num_bytes_per_token;
                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer(0), prefetch_shifted, tma_mbarrier(0), hidden_bytes);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier(0), hidden_bytes);
                }
                __syncwarp();
            }

            for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++chunk_idx, --num_tokens_to_recv) {
                const int s = chunk_idx % kNumStages;
                const int ns = (chunk_idx + 1) % kNumStages;

                int token_idx_in_buffer = (cached_channel_head_idx++) % env.num_max_nvl_chunked_recv_tokens;
                auto shifted = nvl_channel_x.buffer() + token_idx_in_buffer * num_bytes_per_token;
                auto meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes));
                int src_rdma_rank = meta.src_rdma_rank;
                int recv_token_idx = __shfl_sync(0xffffffff, total_offset, src_rdma_rank);
                (lane_id == src_rdma_rank) ? (total_offset += 1) : 0;

                // Wait this iter's load, then prefetch next, then K-fanout store.
                mbarrier_wait(tma_mbarrier(s), tma_phase[s]);

                if (chunk_idx + 1 < num_recv_tokens) {
                    int next_buf = cached_channel_head_idx % env.num_max_nvl_chunked_recv_tokens;
                    auto next_shifted = nvl_channel_x.buffer() + next_buf * num_bytes_per_token;
                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer(ns), next_shifted, tma_mbarrier(ns), hidden_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier(ns), hidden_bytes);
                    }
                    __syncwarp();
                }

                // Bwd Pass A equivalent: per K, derive (slot, e_local) and
                // increment the per-warp `warp_local_seen[src_rdma][e_local]`
                // counter. e_local comes from `tile_id_to_expert[slot/tile_m]`
                // (the metadata kernel built it; small and L2-resident).
                int e_local_row[kMaxTopK];
                if (lane_id == 0) {
                    int* seen_for_src = warp_local_seen + src_rdma_rank * E_local;
                    for (int k = 0; k < shape.num_topk; ++k) {
                        int slot = routing.recv_token_to_slots[recv_token_idx * shape.num_topk + k];
                        if (slot < 0) {
                            e_local_row[k] = -1;
                            continue;
                        }
                        int tile_id = slot / shape.tile_m;
                        int e_local = __ldg(routing.tile_id_to_expert + tile_id);
                        e_local_row[k] = e_local;
                        seen_for_src[e_local] += 1;
                        tma_store_1d(tma_buffer(s),
                                     io.dL_do_pool + static_cast<int64_t>(slot) * shape.hidden_int4,
                                     hidden_bytes, false);
                    }
                }
                __syncwarp();
                tma_store_wait<0>();
                __syncwarp();

                // ── Eager Pass 2 fire (bwd).
                // Same shape as fwd's eager fire (see dispatch_main_kernel for
                // the visibility argument). Release-adds into
                // `bwd_dispatch_arrival_count[block]` when this substream's
                // contribution to expert `e_local` is complete;
                // `seen_per_substream` is the metadata-kernel-computed expected
                // count, matching fwd's count.
                if (lane_id == 0) {
                    uint64_t newly_complete = 0;
                    for (int k = 0; k < shape.num_topk; ++k) {
                        int e_local = e_local_row[k];
                        if (e_local < 0) continue;
                        uint64_t bit = 1ULL << e_local;
                        if ((completed_mask[src_rdma_rank] | newly_complete) & bit) continue;
                        int src_world = src_rdma_rank * NUM_MAX_NVL_PEERS + src_nvl_rank;
                        int my_seen = warp_local_seen[src_rdma_rank * E_local + e_local];
                        int expected = ld_nc_global(seen_for_channel + src_world * E_local + e_local);
                        if (my_seen == expected) newly_complete |= bit;
                    }

                    if (newly_complete != 0) {
                        completed_mask[src_rdma_rank] |= newly_complete;
                        __threadfence_system();
                        int src_world = src_rdma_rank * NUM_MAX_NVL_PEERS + src_nvl_rank;
                        for (int e_local = 0; e_local < E_local; ++e_local) {
                            if (!((newly_complete >> e_local) & 1)) continue;
                            int my_seen = warp_local_seen[src_rdma_rank * E_local + e_local];
                            int slot_start_e = base_pool_for_channel[src_world * E_local + e_local];
                            fire_pool_blocks(slot_start_e, my_seen, shape.tile_m,
                                             tile_signal.bwd_dispatch_arrival_count);
                        }
                    }
                }
                __syncwarp();
            }

            if (elect_one_sync())
                st_relaxed_sys_global(nvl_channel_head.buffer(),
                                     nvl_pack(nvl_seq, cached_channel_head_idx));
        }
    }
}

void launch_dispatch_grads_main(const DispatchGradsIO& io,
                                const DispatchGradsRouting& routing,
                                const DispatchGradsTileSignal& tile_signal,
                                const DispatchGradsShape& shape,
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

#define DISPATCH_GRADS_LAUNCH_CASE(num_rdma_ranks_)                                       \
    {                                                                                     \
        auto kernel = dispatch_grads_main_kernel<num_rdma_ranks_,                         \
                                                 kNumTMABytesPerWarp,                     \
                                                 kNumDispatchRDMASenderWarps>;            \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                                                \
        LAUNCH_KERNEL(&cfg, kernel, io, routing, tile_signal, shape, env);                \
    }                                                                                     \
    break

    SETUP_LAUNCH_CONFIG(num_channels * 2, (kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32, stream);
    int num_ranks = num_world_ranks;
    SWITCH_RDMA_RANKS(DISPATCH_GRADS_LAUNCH_CASE);
#undef DISPATCH_GRADS_LAUNCH_CASE
}

// ─────────────────────────────────────────────────────────────────────────────
// combine_main_kernel — three warp roles + coordinator on the unified
// fwd/bwd arg surface. Mirrors `intranode::combine_main_kernel` shape (same
// arg semantics for `recv_x` / `recv_topk_weights_out` / `x` /
// `per_slot_weights` / `recv_token_to_slots` / `y_done_per_token` /
// `combine_seq`); internode adds RDMA staging via a forwarder + receiver
// pair on top of NVL senders. Streaming gate at `kNVLSender` only —
// downstream stages ride on existing ring-buffer flow control.
//
//   ┌─────────────────────────┬────────────────── fwd ──────────────────┬────────────────── bwd ─────────────────┐
//   │ recv_x                  │ out[num_combined_tokens, H] bf16         │ dL/dx[num_combined_tokens, H] bf16     │
//   │ recv_topk_weights_out   │ recv_topk_weights[num_combined_tokens, K]│ dL/dtopk_weights[num_combined_tokens, K]│
//   │ x                       │ handle.o[num_tokens, H]                  │ dL/dx_per_r[num_tokens, H]             │
//   │ per_slot_weights        │ pool_topk_weight[TK_padded] fp32         │ weight_grads[TK_padded] fp32           │
//   │ recv_token_to_slots     │ same — populated by fwd dispatch Pass B  │ same                                   │
//   │ y_done_per_token  │ kernel_y forward release stamp           │ kernel_a_bwd release stamp             │
//   │ combine_seq             │ dispatch_seq (fwd's value)               │ dispatch_seq (bwd uses same int)       │
//   └─────────────────────────┴──────────────────────────────────────────┴────────────────────────────────────────┘
//
// `num_tokens` here = recv-side count this rank holds in `x` (= dispatch's
// `T_recv` from its perspective). `num_combined_tokens` = source-side count
// (= input token count, the rows of `recv_x` output).
//
// Three-warp-role data flow:
//   kNVLSender (dst rank)  → NVL ring → kNVLAndRDMAForwarder (drains NVL,
//   reduces NUM_MAX_NVL_PEERS contributions per token, RDMA-puts to origin
//   RDMA rank) → RDMA ring → kRDMAReceiver (origin rank, drains RDMA inbox,
//   reduces kNumRDMARanks contributions per token into recv_x[t]).
// Coordinator updates the queue heads (consumer-side flow control).
// ─────────────────────────────────────────────────────────────────────────────
template <bool kLowLatencyMode,
          int kNumRDMARanks,
          typename dtype_t,
          int kNumCombineForwarderWarps,
          int kNumTMABytesPerSenderWarp,
          int kNumTMABytesPerForwarderWarp,
          bool kSendTopkWeights,
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks),
          int kNumWarpsPerForwarder = (kNumCombineForwarderWarps / kNumRDMARanks > 0) ? kNumCombineForwarderWarps / kNumRDMARanks : 1,
          int kNumForwarders = kNumRDMARanks * kNumWarpsPerForwarder,
          int kNumRDMAReceivers = kNumForwarders - NUM_MAX_NVL_PEERS>
__global__ void __launch_bounds__((kNumForwarders + 1) * 32, 1) combine_main_kernel(
    int4* recv_x,
    float* recv_topk_weights_out,
    const int4* x,
    const float* per_slot_weights,
    const int* recv_token_to_slots,
    const int* combined_rdma_head,
    const int* combined_nvl_head,
    const SourceMeta* src_meta,
    const int* recv_rdma_channel_prefix_matrix,
    const int* recv_rdma_rank_prefix_sum,
    const int* gbl_channel_prefix_matrix,
    const int64_t* y_done_per_token,
    int64_t combine_seq,
    int combine_phase,
    int num_tokens,
    int num_combined_tokens,
    int hidden,
    int num_topk,
    void* rdma_buffer_ptr,
    int num_max_rdma_chunked_send_tokens,
    int num_max_rdma_chunked_recv_tokens,
    void** buffer_ptrs,
    int num_max_nvl_chunked_send_tokens,
    int num_max_nvl_chunked_recv_tokens,
    int rank,
    int num_ranks,
    uint32_t* combine_reader_prev_head,
    uint32_t* combine_reader_prev_tail) {
    enum class WarpRole { kNVLSender, kNVLAndRDMAForwarder, kRDMAReceiver, kCoordinator };

    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const auto num_channels = static_cast<int>(gridDim.x) / 2, channel_id = sm_id / 2;
    const bool is_forwarder_sm = sm_id % 2 == 1;

    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);
    const auto hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));
    const auto hidden_bytes = hidden_int4 * sizeof(int4);
    // Per-token slab stride. Stays at the bwd payload (hidden + SourceMeta +
    // K floats) across fwd and bwd combines: the host-side IPC slab layout
    // (RDMA SymBuffer / NVL AsymBuffer) is sized assuming this stride, and
    // the channel sub-slabs are laid out at fixed `num_max_*_chunked_*_tokens
    // * num_bytes_per_token` offsets. Shrinking the stride in fwd would
    // mismatch the host layout → mid-launch slot reads/writes spill into
    // adjacent (channel, src_rank) regions → async CUDA aborts downstream.
    // The K-weight bytes in each slot are simply left uninitialized in fwd
    // (sender skips packing, receiver skips reading — see `kSendTopkWeights`
    // gates below).
    const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, 0, num_topk);

    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto role_meta = [=]() -> std::pair<WarpRole, int> {
        auto warp_id = thread_id / 32;
        if (not is_forwarder_sm) {
            if (warp_id < NUM_MAX_NVL_PEERS) {
                auto shuffled_warp_id = warp_id;
                shuffled_warp_id = (shuffled_warp_id + channel_id) % NUM_MAX_NVL_PEERS;
                return {WarpRole::kNVLSender, shuffled_warp_id};
            } else if (warp_id < kNumForwarders) {
                return {WarpRole::kRDMAReceiver, warp_id - NUM_MAX_NVL_PEERS};
            } else {
                return {WarpRole::kCoordinator, 0};
            }
        } else {
            if (warp_id < kNumForwarders) {
                auto shuffled_warp_id = (warp_id + channel_id) % kNumForwarders;
                return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};
            } else {
                return {WarpRole::kCoordinator, 0};
            }
        }
    }();
    auto warp_role = role_meta.first;
    auto warp_id = role_meta.second;

    EP_DEVICE_ASSERT(num_warps == kNumForwarders + 1);
    auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks;

    // NVL gen-stamp tag for the combine ring slots. Phase bit (0 for fwd
    // combine, 1 for bwd combine_grads — supplied by the host launcher)
    // ensures the combine ring's leftover from one phase doesn't alias the
    // other within the same layer, where `combine_seq` is shared. Release
    // stamps (`y_done_per_token`, written by kernel Y / kernel_a_bwd)
    // remain on the unshifted `combine_seq` value and gate the streaming
    // sender unchanged.
    const int64_t nvl_seq = (combine_seq << 1) | (combine_phase & 1);

    if (warp_role == WarpRole::kNVLSender) {
        // NVL producers: read x[token_idx] from local memory, TMA-load into
        // the dst NVL peer's combine ring buffer, pack SourceMeta + per-(t, k)
        // weight (looked up via recv_token_to_slots → per_slot_weights).
        const auto dst_nvl_rank = warp_id;

        auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];
        auto nvl_channel_x = AsymBuffer<uint8_t>(dst_buffer_ptr,
                                                 num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                                                 NUM_MAX_NVL_PEERS,
                                                 channel_id,
                                                 num_channels,
                                                 nvl_rank)
                                 .advance_also(local_buffer_ptr);
        auto nvl_channel_head = AsymBuffer<uint64_t>(local_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, dst_nvl_rank)
                                    .advance_also(dst_buffer_ptr);
        auto nvl_channel_tail = AsymBuffer<uint64_t>(dst_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
                                    .advance_also(local_buffer_ptr);

        // Two-stage TMA pipeline (mirrors `dispatch_main_kernel`'s NVL receiver).
        constexpr int kNumStages = 2;
        const int kStageStride = (num_bytes_per_token + 15) & ~15;
        extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
        auto tma_buffer = [=](int s) {
            return smem_tma_buffer + dst_nvl_rank * kNumTMABytesPerSenderWarp + s * kStageStride;
        };
        auto tma_mbarrier = [=](int s) {
            return reinterpret_cast<uint64_t*>(
                smem_tma_buffer + dst_nvl_rank * kNumTMABytesPerSenderWarp
                + kNumStages * kStageStride + s * static_cast<int>(sizeof(uint64_t)));
        };
        uint32_t tma_phase[kNumStages] = {0, 0};
        if (elect_one_sync()) {
            #pragma unroll
            for (int s = 0; s < kNumStages; ++s)
                mbarrier_init(tma_mbarrier(s), 1);
            fence_barrier_init();
            EP_DEVICE_ASSERT(kNumStages * kStageStride + kNumStages * static_cast<int>(sizeof(uint64_t)) <= kNumTMABytesPerSenderWarp);
        }
        __syncwarp();

        // Per-RDMA-source token range for this (channel, dst_nvl_rank).
        int token_start_idx = 0, token_end_idx = 0;
        if (lane_id < kNumRDMARanks) {
            int prefix_idx = (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels + channel_id;
            token_start_idx = gbl_channel_prefix_matrix[prefix_idx];
            token_end_idx = (prefix_idx == num_channels * num_ranks - 1) ? num_tokens : gbl_channel_prefix_matrix[prefix_idx + 1];
        }
        __syncwarp();

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

        int current_rdma_idx = channel_id % kNumRDMARanks;
        while (true) {
            if (__all_sync(0xffffffff, token_start_idx >= token_end_idx))
                break;

            // Wait for any RDMA-source slot to have queue space.
            bool is_lane_ready = false;
            auto start_time = clock64();
            while (true) {
                int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
                is_lane_ready = lane_id < kNumRDMARanks and token_start_idx < token_end_idx and
                    num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens;
                if (__any_sync(0xffffffff, is_lane_ready))
                    break;

                if (lane_id < kNumRDMARanks and token_start_idx < token_end_idx) {
                    // NVL gen-stamp read: head is iter-tagged by the coordinator's
                    // force-write + per-progress writes. Pre-force-write reads
                    // carry the prior iter's seq and are ignored; cached_head
                    // stays at its last-valid value (initial 0) until a matching
                    // read arrives.
                    uint64_t raw_head = ld_volatile_global(nvl_channel_head.buffer() + lane_id);
                    if (nvl_seq_match(raw_head, nvl_seq))
                        cached_channel_head_idx = nvl_unpack_value(raw_head);
                }

                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    uint64_t raw_head = ld_volatile_global(nvl_channel_head.buffer() + lane_id);
                    int head_seq_ok = nvl_seq_match(raw_head, nvl_seq) ? nvl_unpack_value(raw_head) : -1;
                    printf("DeepEP combine NVL sender timeout, channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, RDMA lane: %d, head (seq-ok or -1): %d, tail: "
                           "%d, start: %d, end: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id,
                           head_seq_ok,
                           cached_channel_tail_idx, token_start_idx, token_end_idx);
                    trap();
                }
            }

            for (int i = 0; i < kNumRDMARanks; ++i) {
                current_rdma_idx = (current_rdma_idx + 1) % kNumRDMARanks;
                if (__shfl_sync(0xffffffff, (token_start_idx >= token_end_idx) or (not is_lane_ready), current_rdma_idx))
                    continue;

                auto token_idx = static_cast<int64_t>(__shfl_sync(0xffffffff, token_start_idx, current_rdma_idx));
                int num_tokens_in_chunk =
                    __shfl_sync(0xffffffff, min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx), current_rdma_idx);

                // Pre-batch: gate-wait + prefetch the first token's hidden-bytes
                // load into stage 0. The inner-for body waits this iter, prefetches
                // the next, K=1 store, and drains via `tma_store_wait<0>` — so each
                // iter's load latency overlaps with the prior iter's store-wait.
                if (num_tokens_in_chunk > 0) {
                    int64_t first_token_idx = token_idx;
                    if (elect_one_sync()) {
                        auto gate_start = clock64();
                        while (ld_acquire_sys_global(&y_done_per_token[first_token_idx]) < combine_seq) {
                            if (clock64() - gate_start > NUM_TIMEOUT_CYCLES) {
                                // Diagnostic: print device-observed value at
                                // trap so we can compare with host's view of
                                // the same memory (the doc's original
                                // host-vs-device check). Three possible
                                // shapes:
                                //   observed == 0           → matches doc's
                                //     Z_post race (memset clobbered kernel_y).
                                //   observed in (0, seq)    → kernel_y started
                                //     writing this row but didn't finish.
                                //   observed == reread      → stable, the value
                                //     truly hasn't advanced (most likely).
                                //   observed != reread      → race; value
                                //     arrived between two reads (unlikely with
                                //     .sys-acquire but worth printing).
                                auto observed = ld_acquire_sys_global(&y_done_per_token[first_token_idx]);
                                auto reread = ld_acquire_sys_global(&y_done_per_token[first_token_idx]);
                                printf("DeepEP combine NVL sender gate timeout, channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, "
                                       "token: %ld, seq: %ld, observed: %ld, reread: %ld, addr: %p\n",
                                       channel_id, rdma_rank, nvl_rank, dst_nvl_rank,
                                       static_cast<int64_t>(first_token_idx), combine_seq,
                                       observed, reread, (void*)&y_done_per_token[first_token_idx]);
                                trap();
                            }
                        }
                    }
                    __syncwarp();

                    auto shifted_x_first = x + first_token_idx * hidden_int4;
                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer(0), shifted_x_first, tma_mbarrier(0), hidden_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier(0), hidden_bytes);
                    }
                    __syncwarp();
                }

                for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
                    const int s = chunk_idx % kNumStages;
                    const int ns = (chunk_idx + 1) % kNumStages;

                    int dst_slot_idx = 0;
                    if (lane_id == current_rdma_idx) {
                        dst_slot_idx = (cached_channel_tail_idx++) % num_max_nvl_chunked_recv_tokens_per_rdma;
                        dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;
                    }
                    dst_slot_idx = __shfl_sync(0xffffffff, dst_slot_idx, current_rdma_idx);

                    auto shifted_x_buffers = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;

                    // Wait for this iter's load (prefetched pre-batch or by prior iter).
                    mbarrier_wait(tma_mbarrier(s), tma_phase[s]);

                    if (lane_id == num_topk)
                        *reinterpret_cast<SourceMeta*>(tma_buffer(s) + hidden_bytes) = ld_nc_global(src_meta + token_idx);

                    // Per-(token_idx, k) weight: `recv_token_to_slots[token_idx, k]`
                    // gives the pool slot for this (recv-token, k) pair, which holds
                    // weight_grads for bwd. For non-local k (slot == -1) we ship 0 —
                    // receiver's K-way sum then yields the correct (t, k) value
                    // since exactly one sender has the non-zero contribution. Fwd
                    // combine omits the K-weight payload entirely (see header above
                    // `num_bytes_per_token`).
                    if constexpr (kSendTopkWeights) {
                        if (lane_id < num_topk) {
                            int slot = recv_token_to_slots[token_idx * num_topk + lane_id];
                            float w = (slot >= 0) ? __ldg(per_slot_weights + slot) : 0.0f;
                            *reinterpret_cast<float*>(tma_buffer(s) + hidden_bytes + sizeof(SourceMeta) + lane_id * sizeof(float)) = w;
                        }
                    }

                    tma_store_fence();
                    __syncwarp();
                    if (elect_one_sync())
                        tma_store_1d(tma_buffer(s), shifted_x_buffers, num_bytes_per_token, false);

                    // Prefetch next iter's gate-wait + load (overlaps with store-wait).
                    if (chunk_idx + 1 < num_tokens_in_chunk) {
                        int64_t next_token_idx = token_idx + 1;
                        if (elect_one_sync()) {
                            auto gate_start = clock64();
                            while (ld_acquire_sys_global(&y_done_per_token[next_token_idx]) < combine_seq) {
                                if (clock64() - gate_start > NUM_TIMEOUT_CYCLES) {
                                    printf("DeepEP combine NVL sender gate timeout, channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, "
                                           "token: %ld, seq: %ld\n",
                                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank,
                                           static_cast<int64_t>(next_token_idx), combine_seq);
                                    trap();
                                }
                            }
                        }
                        __syncwarp();

                        auto shifted_x_next = x + next_token_idx * hidden_int4;
                        if (elect_one_sync()) {
                            tma_load_1d(tma_buffer(ns), shifted_x_next, tma_mbarrier(ns), hidden_bytes);
                            mbarrier_arrive_and_expect_tx(tma_mbarrier(ns), hidden_bytes);
                        }
                        __syncwarp();
                    }

                    // Drain this iter's store before stage `s` is reused.
                    tma_store_wait<0>();
                    __syncwarp();
                }
                lane_id == current_rdma_idx ? (token_start_idx = static_cast<int>(token_idx)) : 0;
            }

            tma_store_wait<0>();
            __syncwarp();
            if (lane_id < kNumRDMARanks and is_lane_ready)
                st_release_sys_global(nvl_channel_tail.buffer() + lane_id,
                                      nvl_pack(nvl_seq, cached_channel_tail_idx));
        }
    } else {
        // Combiners and coordinators
        auto rdma_channel_data = SymBuffer<int8_t>(
            rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

        void* local_nvl_buffer = buffer_ptrs[nvl_rank];
        void* nvl_buffers[NUM_MAX_NVL_PEERS];
        #pragma unroll
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
            nvl_buffers[i] = buffer_ptrs[i];
        auto nvl_channel_x =
            AsymBuffer<uint8_t>(
                local_nvl_buffer, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, NUM_MAX_NVL_PEERS, channel_id, num_channels)
                .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
        auto nvl_channel_head =
            AsymBuffer<uint64_t, NUM_MAX_NVL_PEERS>(nvl_buffers, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
                .advance_also(local_nvl_buffer);
        auto nvl_channel_tail = AsymBuffer<uint64_t>(local_nvl_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels)
                                    .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);

        __shared__ volatile int forwarder_nvl_head[kNumForwarders][NUM_MAX_NVL_PEERS];
        __shared__ volatile bool forwarder_retired[kNumForwarders];
        __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers][kNumRDMARanks];
        __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers];
        auto sync_forwarder_smem = [=]() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumForwarders + 1) * 32)); };
        auto sync_rdma_receiver_smem = [=]() { asm volatile("barrier.sync 1, %0;" ::"r"((kNumRDMAReceivers + 1) * 32)); };

        if (warp_role == WarpRole::kNVLAndRDMAForwarder) {
            // Drain NVL ring on dst rank, reduce K_local NVL-source contributions,
            // RDMA-put to origin rank.
            const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder;
            const auto sub_warp_id = warp_id % kNumWarpsPerForwarder;
            auto send_buffer =
                dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank) : rdma_channel_data.send_buffer(dst_rdma_rank);
            auto sync_large_warp = [=]() {
                if (kNumWarpsPerForwarder == 1) {
                    __syncwarp();
                } else {
                    asm volatile("bar.sync %0, %1;" ::"r"(dst_rdma_rank + 2), "r"(kNumWarpsPerForwarder * 32));
                }
            };
            EP_STATIC_ASSERT(kNumWarpsPerForwarder == 1 or kNumRDMARanks + 2 <= 16, "Barriers are not enough");

            constexpr int kNumStages = 2;
            constexpr int kNumTMALoadBytes = sizeof(int4) * 32;
            constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16;
            EP_STATIC_ASSERT(kNumTMABufferBytesPerStage * kNumStages <= kNumTMABytesPerForwarderWarp, "TMA buffer not large enough");

            extern __shared__ __align__(1024) uint8_t smem_buffer[];
            auto smem_ptr = smem_buffer + warp_id * kNumStages * kNumTMABufferBytesPerStage;
            auto tma_mbarrier = [=](const int& i) {
                return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1));
            };
            uint32_t tma_phase[kNumStages] = {0};
            if (lane_id < kNumStages) {
                mbarrier_init(tma_mbarrier(lane_id), 32);
                fence_barrier_init();
            }
            __syncwarp();

            nvl_channel_x.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * num_bytes_per_token);
            nvl_channel_head.advance(dst_rdma_rank);
            nvl_channel_tail.advance(dst_rdma_rank);

            EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
            lane_id < NUM_MAX_NVL_PEERS ? (forwarder_nvl_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (forwarder_retired[warp_id] = false) : false;
            sync_forwarder_smem();

            int cached_nvl_channel_tail_idx = 0;
            // Combine head slot accumulates across iters; seed prev_at_entry.
            uint32_t prev_rdma_channel_head_at_entry =
                combine_reader_prev_head[channel_id * kNumRDMARanks + dst_rdma_rank];

            // `recv_rdma_channel_prefix_matrix` was gated by
            // `encode_combine_heads` (which runs first on the same combine
            // stream) — same-stream FIFO covers our read here. The gbl
            // matrix gate inside kNVLSender (above) is independent and
            // remains. See markdowns/cdmc-partial-fix.md §"Stage 1".
            int num_tokens_to_combine = recv_rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            int num_tokens_prefix = channel_id == 0 ? 0 : recv_rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            num_tokens_to_combine -= num_tokens_prefix;
            num_tokens_prefix += dst_rdma_rank == 0 ? 0 : recv_rdma_rank_prefix_sum[dst_rdma_rank - 1];
            auto combined_nvl_head_local = combined_nvl_head + num_tokens_prefix * NUM_MAX_NVL_PEERS;

            for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine; token_start_idx += num_max_rdma_chunked_send_tokens) {
                auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
                auto num_chunked_tokens = token_end_idx - token_start_idx;
                auto start_time = clock64();
                while (sub_warp_id == 0 and lane_id == 0) {
                    int cur_iter_head = static_cast<int>(static_cast<uint32_t>(ld_acquire_sys_global(rdma_channel_head.buffer(dst_rdma_rank))) - prev_rdma_channel_head_at_entry);
                    int num_used_slots = token_start_idx - cur_iter_head;
                    if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens)
                        break;
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP combine forwarder (RDMA check) timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA: %d, iter_head: %d, "
                               "tail: %d, chunked: %d\n",
                               channel_id, rdma_rank, nvl_rank, dst_rdma_rank, cur_iter_head,
                               token_start_idx, num_chunked_tokens);
                        trap();
                    }
                }
                sync_large_warp();

                for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder) {
                    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                    int expected_head = -1;
                    if (lane_id < NUM_MAX_NVL_PEERS) {
                        expected_head = ld_nc_global(combined_nvl_head_local + token_idx * NUM_MAX_NVL_PEERS + lane_id);
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head);
                    }

                    start_time = clock64();
                    while (cached_nvl_channel_tail_idx <= expected_head) {
                        // NVL gen-stamp read: only accept seq-matching tail
                        // values. The sender's iter-entry force-write +
                        // per-chunk packed writes ensure the slot eventually
                        // carries this iter's tag; pre-tag residue is ignored.
                        uint64_t raw_tail = ld_volatile_global(nvl_channel_tail.buffer(lane_id));
                        if (nvl_seq_match(raw_tail, nvl_seq))
                            cached_nvl_channel_tail_idx = nvl_unpack_value(raw_tail);
                        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < NUM_MAX_NVL_PEERS) {
                            printf("DeepEP combine forwarder (NVL check) timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, dst RDMA: %d, "
                                   "tail: %d, waiting: %d, total: %d, sub: %d, large: %d, expected: %d\n",
                                   channel_id, rdma_rank, nvl_rank, lane_id, dst_rdma_rank,
                                   cached_nvl_channel_tail_idx, token_idx, num_tokens_to_combine,
                                   sub_warp_id, kNumWarpsPerForwarder, expected_head);
                            trap();
                        }
                    }

                    auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
                    void* shifted = send_buffer + rdma_slot_idx * num_bytes_per_token;
                    auto get_addr_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                        return reinterpret_cast<int4*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token) +
                            hidden_int4_idx;
                    };
                    auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float {
                        return ld_nc_global(reinterpret_cast<float*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token +
                                                                     hidden_bytes + sizeof(SourceMeta)) +
                                            topk_idx);
                    };
                    combine_token<NUM_MAX_NVL_PEERS, dtype_t, NUM_MAX_NVL_PEERS, true, kNumStages, kNumTMALoadBytes, kSendTopkWeights>(
                        expected_head >= 0,
                        expected_head,
                        lane_id,
                        hidden_int4,
                        num_topk,
                        static_cast<int4*>(shifted),
                        reinterpret_cast<float*>(static_cast<int8_t*>(shifted) + hidden_bytes + sizeof(SourceMeta)),
                        num_max_nvl_chunked_recv_tokens_per_rdma,
                        get_addr_fn,
                        recv_tw_fn,
                        smem_ptr,
                        tma_phase);

                    if (lane_id < NUM_MAX_NVL_PEERS)
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head + 1);
                }
                sync_large_warp();

                if (sub_warp_id == kNumWarpsPerForwarder - 1) {
                    if (dst_rdma_rank != rdma_rank) {
                        auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
                        const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_token;
                        const auto dst_ptr =
                            reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        const auto src_ptr =
                            reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                          translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                          channel_id, lane_id, 0);
                    } else {
                        memory_fence();
                    }

                    __syncwarp();
                    if (elect_one_sync()) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank),
                                                        num_chunked_tokens,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                        channel_id,
                                                        dst_rdma_rank == rdma_rank);
                    }
                }
            }

            __syncwarp();
            if (elect_one_sync())
                forwarder_retired[warp_id] = true;

            // Writeback observed cumulative head to persistent reader_prev_head
            // for this combine forwarder's dst_rdma_rank. Single-warp writer
            // per (channel, dst_rdma) — no atomic needed in principle, but
            // atomicMax keeps shape symmetric with the dispatch path.
            if (lane_id == 0) {
                atomicmax_reader_prev_cumulative(
                    combine_reader_prev_head + channel_id * kNumRDMARanks + dst_rdma_rank,
                    static_cast<uint32_t>(ld_acquire_sys_global(rdma_channel_head.buffer(dst_rdma_rank))));
            }
        } else if (warp_role == WarpRole::kRDMAReceiver) {
            // On origin rank: drain RDMA inbox, reduce K_dst contributions per
            // source token into recv_x[token_idx] + recv_topk_weights_out[token_idx, k].
            EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
            lane_id < kNumRDMARanks ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;
            sync_rdma_receiver_smem();

            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Combine tail slot accumulates across iters; seed prev_at_entry.
            uint32_t prev_rdma_channel_tail_at_entry = lane_id < kNumRDMARanks
                ? combine_reader_prev_tail[channel_id * kNumRDMARanks + lane_id]
                : 0u;
            int cached_channel_tail_idx = 0;
            for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers) {
                EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                int expected_head = -1;
                if (lane_id < kNumRDMARanks) {
                    expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks + lane_id);
                    (expected_head < 0) ? (rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1)
                                        : (rdma_receiver_rdma_head[warp_id][lane_id] = expected_head);
                }

                auto start_time = clock64();
                while (cached_channel_tail_idx <= expected_head) {
                    cached_channel_tail_idx =
                        static_cast<int>(static_cast<uint32_t>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id))) - prev_rdma_channel_tail_at_entry);
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP combine RDMA receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, tail: %d, "
                               "waiting: %ld, expect: %d\n",
                               channel_id, rdma_rank, nvl_rank, lane_id,
                               cached_channel_tail_idx, token_idx, expected_head);
                        trap();
                    }
                }
                __syncwarp();

                auto get_addr_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                    return reinterpret_cast<int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_token) +
                        hidden_int4_idx;
                };
                auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float {
                    return ld_nc_global(reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rdma_rank) +
                                                                       slot_idx * num_bytes_per_token + hidden_bytes + sizeof(SourceMeta)) +
                                        topk_idx);
                };
                uint32_t dummy_tma_phases[2];
                combine_token<kNumRDMARanks, dtype_t, kNumTopkRDMARanks, false, 2, 0, kSendTopkWeights>(
                    expected_head >= 0, expected_head, lane_id, hidden_int4, num_topk,
                    recv_x + token_idx * hidden_int4,
                    recv_topk_weights_out + token_idx * num_topk,
                    num_max_rdma_chunked_recv_tokens,
                    get_addr_fn, recv_tw_fn,
                    nullptr, dummy_tma_phases);
            }

            __syncwarp();
            if (elect_one_sync())
                rdma_receiver_retired[warp_id] = true;

            // Writeback observed cumulative tail to persistent reader_prev_tail.
            // Multiple RDMA receiver warps share the role; atomicMax keeps
            // the array monotonic.
            if (lane_id < kNumRDMARanks) {
                atomicmax_reader_prev_cumulative(
                    combine_reader_prev_tail + channel_id * kNumRDMARanks + lane_id,
                    static_cast<uint32_t>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id))));
            }
        } else {
            // Coordinator
            is_forwarder_sm ? sync_forwarder_smem() : sync_rdma_receiver_smem();
            const auto num_warps_per_rdma_rank = kNumForwarders / kNumRDMARanks;

            int last_rdma_head = 0;
            int last_nvl_head[kNumRDMARanks] = {0};
            int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
            int dst_nvl_rank = lane_id < NUM_MAX_NVL_PEERS ? lane_id : 0;
            EP_STATIC_ASSERT(kNumCombineForwarderWarps <= 32, "Invalid number of forwarder warps");

            while (true) {
                if (not is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumRDMAReceivers or rdma_receiver_retired[lane_id]))
                    break;
                if (is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumForwarders or forwarder_retired[lane_id]))
                    break;

                if (not is_forwarder_sm) {
                    int min_head = std::numeric_limits<int>::max();
                    #pragma unroll
                    for (int i = 0; i < kNumRDMAReceivers; ++i)
                        if (not rdma_receiver_retired[i])
                            min_head = min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);
                    if (min_head != std::numeric_limits<int>::max() and min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and
                        lane_id < kNumRDMARanks) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank),
                                                        min_head - last_rdma_head,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                        channel_id + num_channels,
                                                        dst_rdma_rank == rdma_rank);
                        last_rdma_head = min_head;
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < kNumRDMARanks; ++i) {
                        int min_head = std::numeric_limits<int>::max();
                        #pragma unroll
                        for (int j = 0; j < num_warps_per_rdma_rank; ++j)
                            if (not forwarder_retired[i * num_warps_per_rdma_rank + j])
                                min_head = min(min_head, forwarder_nvl_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank]);
                        if (min_head != std::numeric_limits<int>::max() and min_head > last_nvl_head[i] and lane_id < NUM_MAX_NVL_PEERS) {
                            last_nvl_head[i] = min_head;
                            st_relaxed_sys_global(nvl_channel_head.buffer_by(dst_nvl_rank) + i,
                                                  nvl_pack(nvl_seq, min_head));
                        }
                    }
                }

                __nanosleep(NUM_WAIT_NANOSECONDS);
            }
        }
    }
}

void launch_combine_main(cudaDataType_t type,
                         void* recv_x,
                         float* recv_topk_weights_out,
                         const void* x,
                         const float* per_slot_weights,
                         const int* recv_token_to_slots,
                         const int* combined_rdma_head,
                         const int* combined_nvl_head,
                         const void* src_meta,
                         const int* recv_rdma_channel_prefix_matrix,
                         const int* recv_rdma_rank_prefix_sum,
                         const int* gbl_channel_prefix_matrix,
                         const int64_t* y_done_per_token,
                         int64_t combine_seq,
                         int combine_phase,
                         bool is_fwd,
                         int num_tokens,
                         int num_combined_tokens,
                         int hidden,
                         int num_topk,
                         void* rdma_buffer_ptr,
                         int num_max_rdma_chunked_send_tokens,
                         int num_max_rdma_chunked_recv_tokens,
                         void** buffer_ptrs,
                         int num_max_nvl_chunked_send_tokens,
                         int num_max_nvl_chunked_recv_tokens,
                         int rank,
                         int num_ranks,
                         cudaStream_t stream,
                         int num_channels,
                         uint32_t* combine_reader_prev_head,
                         uint32_t* combine_reader_prev_tail) {
    constexpr int kNumCombineForwarderWarps = 24;
    constexpr int kNumTMABytesPerSenderWarp = 16384;
    constexpr int kNumTMABytesPerForwarderWarp = 9248;
    constexpr int smem_size =
        std::max(kNumTMABytesPerSenderWarp * NUM_MAX_NVL_PEERS, kNumTMABytesPerForwarderWarp * kNumCombineForwarderWarps);

#define COMBINE_LAUNCH_CASE_IMPL(num_rdma_ranks, kSendTopkWeights)                   \
    {                                                                                \
        auto kernel = combine_main_kernel<false,                                     \
                                          num_rdma_ranks,                            \
                                          nv_bfloat16,                               \
                                          kNumCombineForwarderWarps,                 \
                                          kNumTMABytesPerSenderWarp,                 \
                                          kNumTMABytesPerForwarderWarp,              \
                                          kSendTopkWeights>;                         \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                                           \
        LAUNCH_KERNEL(&cfg, kernel,                                                  \
                      reinterpret_cast<int4*>(recv_x),                               \
                      recv_topk_weights_out,                                         \
                      reinterpret_cast<const int4*>(x),                              \
                      per_slot_weights,                                              \
                      recv_token_to_slots,                                           \
                      combined_rdma_head, combined_nvl_head,                         \
                      reinterpret_cast<const SourceMeta*>(src_meta),                 \
                      recv_rdma_channel_prefix_matrix,                               \
                      recv_rdma_rank_prefix_sum,                                     \
                      gbl_channel_prefix_matrix,                                     \
                      y_done_per_token,                                              \
                      combine_seq, combine_phase,                                    \
                      num_tokens, num_combined_tokens, hidden, num_topk,             \
                      rdma_buffer_ptr,                                               \
                      num_max_rdma_chunked_send_tokens,                              \
                      num_max_rdma_chunked_recv_tokens,                              \
                      buffer_ptrs,                                                   \
                      num_max_nvl_chunked_send_tokens,                               \
                      num_max_nvl_chunked_recv_tokens,                               \
                      rank, num_ranks,                                               \
                      combine_reader_prev_head, combine_reader_prev_tail);           \
    }
#define COMBINE_LAUNCH_CASE(num_rdma_ranks)                                          \
    {                                                                                \
        if (is_fwd) {                                                                \
            COMBINE_LAUNCH_CASE_IMPL(num_rdma_ranks, false)                          \
        } else {                                                                     \
            COMBINE_LAUNCH_CASE_IMPL(num_rdma_ranks, true)                           \
        }                                                                            \
    }                                                                                \
    break

    int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    auto num_warps_per_forwarder = std::max(kNumCombineForwarderWarps / num_rdma_ranks, 1);
    int num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;
    EP_HOST_ASSERT(num_rdma_ranks <= kNumCombineForwarderWarps);
    EP_HOST_ASSERT(num_forwarder_warps > NUM_MAX_NVL_PEERS and num_forwarder_warps % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks >
                   std::max(num_max_rdma_chunked_send_tokens, num_max_nvl_chunked_send_tokens));
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks - num_warps_per_forwarder >= num_max_nvl_chunked_send_tokens);
    EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder);
    EP_HOST_ASSERT(type == CUDA_R_16BF);

    SETUP_LAUNCH_CONFIG(num_channels * 2, (num_forwarder_warps + 1) * 32, stream);
    SWITCH_RDMA_RANKS(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
#undef COMBINE_LAUNCH_CASE_IMPL
}

}  // namespace internode

}  // namespace stream_ep
