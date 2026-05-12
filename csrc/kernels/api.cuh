#pragma once

#include <vector>

#include "configs.cuh"

namespace stream_ep {

// Intranode runtime
namespace intranode {

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream);

}  // namespace intranode

// Internode runtime
namespace internode {

std::vector<uint8_t> get_unique_id();

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks);

void* alloc(size_t size, size_t alignment);

void free(void* ptr);

void barrier();

void finalize();

}  // namespace internode

// Intranode kernels
namespace intranode {

// Streaming-MoE consolidated dispatch metadata. Single launch that does the
// cross-rank (token, k) count exchange and emits all pool-shape outputs known
// before the host poll on `total_tiles`:
//   expert_frequency[E_local], expert_pool_block_offset[E_local + 1],
//   base_pool[num_channels, num_ranks, E_local],
//   rank_prefix_matrix[R, R] (this rank's column),
//   tile_id_to_expert[total_tiles_max], pool_arrival_target[total_tiles_max]
//     (only the [0, total_tiles) prefix is written; caller pre-allocates at
//     `total_tiles_max = ceil(N * K * R / tile_m) + E_local`),
//   total_tiles (host-mapped + device int), num_recv (host-mapped),
//   num_recv_per_expert[E_local] (host-mapped, aligned).
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
                                 cudaStream_t stream);

// Argument groupings for the streaming dispatch main kernel. Each struct
// captures one logical concern; passing all six instead of 31 loose args makes
// it easy to add / rename / reorder a single field — touch the struct, not
// every signature copy.
struct DispatchPoolOut {
    int4* pool;                    // [TK_padded, hidden]   data (int4-vector)
    float* pool_topk_weight;       // [TK_padded]           per-pool-slot weight
    int* pool_recv_token;          // [TK_padded]           slot → recv-token id (-1 = padding)
    int* pool_k_slot;              // [TK_padded]           slot → k (-1 = padding)
};

struct DispatchPerTokenOut {
    int* recv_channel_prefix_matrix;  // [num_ranks, num_channels]  receiver-side cumulative
    int* send_head;                // [num_tokens, num_ranks]
    int* per_token_remaining;      // [T_recv]              K_local(r); kernel Y atomicSubs to 0
    // Backward-pass scaffolding written by Pass B's per-recv-token lane-0 K-loop:
    int* recv_token_to_slots;      // [T_recv, num_topk]    (r, k) → pool slot, -1 for non-local k
    int* k_local_count;            // [T_recv]              K_local(r); write-once mirror of per_token_remaining
};

struct DispatchInputs {
    const int4* x;                 // [num_tokens, hidden]  (int4-vector)
    const topk_idx_t* topk_idx;    // [num_tokens, num_topk]
    const float* topk_weights;     // [num_tokens, num_topk]
    const bool* is_token_in_rank;  // [num_tokens, num_ranks]
};

struct DispatchTileSignal {
    int* channel_prefix_matrix;    // [num_ranks, num_channels]  sender-side cumulative
    const int* base_pool;          // [num_channels, num_ranks, E_local]
    int* pool_arrival_count;       // [total_tiles]   atomic-add target during pass 2
    const int* pool_arrival_target;  // [total_tiles] firing target
    int64_t* tile_ready;           // [total_tiles]   per-tile release stamp
    int64_t dispatch_seq;
};

struct DispatchShape {
    int num_tokens;
    int hidden_int4;
    int num_topk;
    int num_experts;
    int tile_m;
};

struct DispatchEnv {
    void** buffer_ptrs;
    int rank;
    int num_max_send_tokens;
    int num_recv_buffer_tokens;
};

void launch_dispatch_main(const DispatchPoolOut& pool_out,
                          const DispatchPerTokenOut& per_token_out,
                          const DispatchInputs& inputs,
                          const DispatchTileSignal& tile_signal,
                          const DispatchShape& shape,
                          const DispatchEnv& env,
                          int num_ranks,
                          cudaStream_t stream,
                          int num_sms);

// Backward dispatch_grads: ships dL/dy from origin → expert ranks along the
// same routing as forward dispatch (sender uses is_token_in_rank, receiver
// looks up slots from recv_token_to_slots written by fwd Pass B). No Pass A
// (slots are pre-computed), no scalar metadata writes (already populated).
struct DispatchGradsIO {
    int4* dL_do_pool;                 // [TK_padded, hidden_int4] receiver writes K times per recv-token
    const int4* dL_dy;                // [num_tokens, hidden_int4]   sender reads
    const bool* is_token_in_rank;     // [num_tokens, num_ranks]     same routing as fwd dispatch
};

struct DispatchGradsRouting {
    const int* recv_token_to_slots;   // [T_recv, num_topk]                     bwd Pass B slot lookup
    const int* base_pool;             // [num_channels, num_ranks, E_local]     Pass 2: per-substream slot start
    const int* seen_per_substream;    // [num_channels, num_ranks, E_local]     Pass 2: per-substream-per-expert recv count
    // Passed explicitly (NOT read from IPC slab leading bytes) — fwd combine's
    // `cached_notify_combine` zeros that region before bwd runs, so the IPC
    // slab can't be the source. Persistent tensor lives on the StreamingHandle.
    const int* rank_prefix_matrix;    // [num_ranks, num_ranks]                receiver: per-source-rank token offset
};

struct DispatchGradsTileSignal {
    int* bwd_dispatch_arrival_count;  // [total_tiles] int32  atomic-add target during Pass 2
    const int* pool_arrival_target;   // [total_tiles] int32  firing target (same as fwd's)
    int64_t* bwd_y_ready;             // [total_tiles] int64  per-tile release stamp (consumed by kernel_y_bwd)
    int64_t dispatch_seq;
};

struct DispatchGradsShape {
    int num_tokens;
    int hidden_int4;
    int num_topk;
    int num_experts;
    int tile_m;
};

void launch_dispatch_grads_main(const DispatchGradsIO& io,
                                const DispatchGradsRouting& routing,
                                const DispatchGradsTileSignal& tile_signal,
                                const DispatchGradsShape& shape,
                                const DispatchEnv& env,
                                int num_ranks,
                                cudaStream_t stream,
                                int num_sms);

void cached_notify_combine(void** buffer_ptrs,
                           int* send_head,
                           int num_channels,
                           int num_recv_tokens,
                           int num_memset_int,
                           int** barrier_signal_ptrs,
                           int rank,
                           int num_ranks,
                           cudaStream_t stream);

// combine_main_kernel — used by both forward combine and backward combine_grads.
// Per-direction differences are entirely in args (per_slot_weights tensor,
// gate variable, output destinations). See the kernel header comment in
// intranode.cu for the full args table.
void launch_combine_main(cudaDataType_t type,
             void* recv_x,
             float* recv_topk_weights_out,
             const void* x,
             const float* per_slot_weights,
             const int* recv_token_to_slots,
             const int* rank_prefix_matrix,
             const int* channel_prefix_matrix,
             int* send_head,
             const int64_t* compute_done_per_token,
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
             int num_recv_buffer_tokens);

}  // namespace intranode

// Internode kernels
namespace internode {

int get_source_meta_bytes();

// Inline mirror of `get_source_meta_bytes()` for use in __host__ __device__
// helpers below (the regular function is .cpp-side only). Kept in sync with
// the SourceMeta struct definition in internode.cu.
__host__ __device__ inline int get_source_meta_bytes_inline() {
    return 8;  // sizeof(SourceMeta) — { int src_rdma_rank; int is_token_in_nvl_rank_bits; }
}

// Wire-format byte size for one NVL/RDMA-staged token (data + SourceMeta +
// topk_idx + topk_weights). Used by the metadata kernel (cleanup-region
// sizing) and the dispatch_main kernel (stride into the channel data ring).
__host__ __device__ inline int get_num_bytes_per_token(int hidden_int4, int num_topk_idx, int num_topk_weights) {
    auto raw = hidden_int4 * static_cast<int>(sizeof(int4))
               + get_source_meta_bytes_inline()
               + num_topk_idx * static_cast<int>(sizeof(int))
               + num_topk_weights * static_cast<int>(sizeof(float));
    auto a = static_cast<int>(sizeof(int4));
    return ((raw + a - 1) / a) * a;
}

// RDMA dispatch meta SymBuffer slab size (ints per (channel, dst_rdma) slab).
// 32 ints = 128 bytes = one H100 L2 cache line. Slots 0..17 carry the data
// (NUM_MAX_NVL_PEERS*2 start_sum + NUM_MAX_NVL_PEERS*2 end_sum + 2 prefix);
// slots 18..29 are padding; slot 30 is the cumulative-across-iters sentinel
// (8-byte-aligned, mlx5 ATOMIC_FAA target); slot 31 is reserved. The
// sentinel-amo coherence trick (sender bulk_put + amo on slot 30; reader
// observes slot 30 > prev_at_entry then reads slots 0..17 plain) requires
// each slab to occupy exactly one L2 line. The meta SymBuffer base is
// 128B-aligned by `align_meta_base_to_l2_line` in internode.cu.
#define kRdmaMetaSlabInts 32
#define kRdmaMetaSentinelSlot 30
static_assert((NUM_MAX_NVL_PEERS * 2 + 2) <= 18,
              "Meta data slots overflow the 0..17 region of the 32-int slab");

// Streaming-MoE consolidated dispatch metadata for internode (RDMA + NVL).
// Folded single-kernel architecture mirroring `intranode::streaming_dispatch_metadata`'s
// shape phase-for-phase, with NVL primitives replaced by RDMA equivalents
// where the topology demands it. Runs cross-rank count exchange + channel
// prefix matrices + host-mapped recv counters in the leading phases, then
// the streaming-superset phases (expert frequency, base_pool,
// seen_per_substream, per-tile arrays) at the end of the same launch.
//
// Outputs (per-rank, on the dispatch stream):
//   - rdma_channel_prefix_matrix[num_rdma_ranks, num_channels]
//   - gbl_channel_prefix_matrix[num_world_ranks, num_channels]
//   - recv_rdma_rank_prefix_sum[num_rdma_ranks]
//   - recv_gbl_rank_prefix_sum[num_world_ranks]
//   - moe_recv_counter / moe_recv_rdma_counter / moe_recv_expert_counter[E_local]
//     (host-mapped — drive the host poll for `pool[T_recv, hidden]` allocation)
//   - streaming_total_tiles (host-mapped)
//   - expert_frequency[E_local]
//   - expert_pool_block_offset[E_local + 1]   (tile-unit prefix sum)
//   - base_pool[num_channels, num_world_ranks, E_local]
//   - seen_per_substream[num_channels, num_world_ranks, E_local]
//   - rank_prefix_matrix[num_world_ranks, num_world_ranks] (this rank's column)
//   - tile_id_to_expert[total_tiles_max]   (caller narrows post-poll)
//   - pool_arrival_target[total_tiles_max] (caller narrows post-poll)
//   - total_tiles_device[1]
//
// RDMA-payload layout for the streaming-superset histogram exchange (new
// SymBuffer alongside the leading count-exchange payload):
//   per (src_world_rank → dst_rdma_rank) slab, contiguous as
//     [num_channels][NUM_MAX_NVL_PEERS][E_local] int32
//   Size per slab: num_channels * NUM_MAX_NVL_PEERS * E_local * 4 bytes.
//   RDMA pairs same-NVL-slot ranks (dst rank == (dst_rdma, src_nvl)), so
//   the dst rank's recv buffer holds one slab per (src_rdma, src_nvl=this_rank's_nvl)
//   pair — the src_nvl axis is filled by the NVL aggregation phase that
//   follows (each NVL rank within dst_rdma extracts its `dst_nvl` slice
//   from the RDMA-received slabs and writes it to its 7 NVL peers).
void streaming_dispatch_metadata(const topk_idx_t* topk_idx,
                                 // Counters (host-mapped, written)
                                 int* moe_recv_counter_mapped,
                                 int* moe_recv_rdma_counter_mapped,
                                 int* moe_recv_expert_counter_mapped,
                                 int* streaming_total_tiles_mapped,
                                 // Channel prefix matrices
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
                                 int hidden_int4,
                                 int expert_alignment,
                                 int tile_m,
                                 // Streaming SymBuffer offset within rdma_buffer_ptr
                                 // (placed AFTER the leading count-exchange payload).
                                 int64_t streaming_rdma_offset,
                                 // Env
                                 void* rdma_buffer_ptr,
                                 void** buffer_ptrs,
                                 int** barrier_signal_ptrs,
                                 int rank,
                                 int num_ranks,
                                 cudaStream_t stream,
                                 int64_t num_rdma_bytes,
                                 int64_t num_nvl_bytes);

// Argument groupings for the streaming internode dispatch main kernel. Same
// six-struct shape as `intranode::Dispatch*` (api.cuh:76–121); internode-
// specific deltas live inside `DispatchPerTokenOut` (combine plumbing —
// recv_src_meta + send_rdma_head + send_nvl_head + recv_*_channel_prefix_*),
// `DispatchInputs` (sender/forwarder reads from metadata kernel), and
// `DispatchEnv` (RDMA buffer + NVL/RDMA chunked send/recv sizes).
struct DispatchPoolOut {
    int4* pool;                    // [TK_padded, hidden_int4]   data (int4-vector)
    float* pool_topk_weight;       // [TK_padded]                per-pool-slot weight
    int* pool_recv_token;          // [TK_padded]                slot → recv-token id (-1 = padding)
    int* pool_k_slot;              // [TK_padded]                slot → k (-1 = padding)
};

struct DispatchPerTokenOut {
    // Streaming-essential per-recv-token outputs (mirror intranode):
    int* per_token_remaining;      // [T_recv]                   K_local(r); kernel Y atomicSubs to 0
    int* recv_token_to_slots;      // [T_recv, num_topk]         (r, k) → pool slot, -1 for non-local k
    int* k_local_count;            // [T_recv]                   write-once K_local mirror

    // Combine plumbing (internode-specific):
    void* recv_src_meta;                   // [T_recv, get_source_meta_bytes()] (uint8)
    int*  send_rdma_head;                  // [num_tokens, num_rdma_ranks]
    int*  send_nvl_head;                   // [num_rdma_recv_tokens, NUM_MAX_NVL_PEERS]
    int*  recv_rdma_channel_prefix_matrix; // [num_rdma_ranks, num_channels]
    int*  recv_gbl_channel_prefix_matrix;  // [num_world_ranks, num_channels]
};

struct DispatchInputs {
    const int4* x;                      // [num_tokens, hidden_int4]
    const topk_idx_t* topk_idx;         // [num_tokens, num_topk]
    const float* topk_weights;          // [num_tokens, num_topk]
    const bool* is_token_in_rank;       // [num_tokens, num_world_ranks]
    // Sender / forwarder reads from metadata kernel:
    const int* rdma_channel_prefix_matrix;  // [num_rdma_ranks, num_channels]
    const int* recv_rdma_rank_prefix_sum;   // [num_rdma_ranks]
    const int* gbl_channel_prefix_matrix;   // [num_world_ranks, num_channels]
    const int* recv_gbl_rank_prefix_sum;    // [num_world_ranks]
};

struct DispatchTileSignal {
    const int* base_pool;            // [num_channels, num_world_ranks, E_local]
    const int* seen_per_substream;   // [num_channels, num_world_ranks, E_local]
                                     //   eager-fire target: NVL receiver compares its
                                     //   per-warp `seen[src_rdma][e_local]` against this
                                     //   per-iter; on match, fires `tile_ready` for the
                                     //   completed expert's blocks (vs the substream-end
                                     //   walk intranode still uses).
    int* pool_arrival_count;         // [total_tiles]
    const int* pool_arrival_target;  // [total_tiles]
    int64_t* tile_ready;             // [total_tiles]
    int64_t dispatch_seq;
};

struct DispatchShape {
    int num_tokens;
    int hidden_int4;
    int num_topk;
    int num_experts;
    int tile_m;
};

struct DispatchEnv {
    void* rdma_buffer_ptr;
    void** buffer_ptrs;              // [NUM_MAX_NVL_PEERS]
    int rank;
    int num_max_rdma_chunked_send_tokens;
    int num_max_rdma_chunked_recv_tokens;
    int num_max_nvl_chunked_send_tokens;
    int num_max_nvl_chunked_recv_tokens;
    // Persistent reader_prev arrays for the RDMA head/tail SymBuffer slots.
    // [num_channels × num_rdma_ranks] uint32 each — matches the NIC's
    // 4-byte AMO width. Read at warp entry, written back at kernel exit by
    // `atomicmax_reader_prev_cumulative`. See `Buffer` in `deep_ep.hpp` for
    // the role / lifetime. Shared by fwd dispatch + bwd dispatch_grads
    // (both on streams.dispatch — stream-ordered, so the writeback at the
    // end of fwd is visible at the start of bwd).
    uint32_t* reader_prev_head;
    uint32_t* reader_prev_tail;
    // Persistent prev-sentinel array for the RDMA dispatch meta region.
    // [num_channels × num_rdma_ranks] int32. Same lifetime/protocol as
    // reader_prev_{head,tail}, but tracks the slab[c, src_rdma].slot[30]
    // sentinel (cumulative amo). The forwarder seeds prev at warp entry,
    // spins on `ld(slot 30) > prev`, reads raw data slots 0..17 once
    // tripped, and atomicMaxes the latest slot 30 value back at exit.
    int* dispatch_meta_sentinel_prev;
};

void launch_dispatch_main(const DispatchPoolOut& pool_out,
                          const DispatchPerTokenOut& per_token_out,
                          const DispatchInputs& inputs,
                          const DispatchTileSignal& tile_signal,
                          const DispatchShape& shape,
                          const DispatchEnv& env,
                          int num_rdma_ranks,
                          int num_channels,
                          cudaStream_t stream);

// Backward dispatch_grads (internode). Same architectural shape as
// `intranode::launch_dispatch_grads_main` (sender ships dL/dy → expert
// ranks; receiver uses fwd-persisted `recv_token_to_slots` to scatter
// `dL_do_pool[slot]` K times per packet; Pass 2 fires `bwd_y_ready`),
// scaled to the RDMA + NVL hierarchy. Reuses fwd dispatch's wire format
// — the per-token bytes carry data + SourceMeta + topk_* but bwd only
// writes/reads the data region; metadata bytes are zero from
// `cached_notify_combine`'s buffer cleanup.
struct DispatchGradsIO {
    int4* dL_do_pool;                 // [TK_padded, hidden_int4] receiver writes K times per recv-token
    const int4* dL_dy;                // [num_tokens, hidden_int4] sender reads
    const bool* is_token_in_rank;     // [num_tokens, num_world_ranks] same routing as fwd dispatch
};

struct DispatchGradsRouting {
    const int* recv_token_to_slots;   // [T_recv, num_topk]                       bwd Pass B slot lookup
    const int* base_pool;             // [num_channels, num_world_ranks, E_local] Pass 2: per-substream slot start
    const int* seen_per_substream;    // [num_channels, num_world_ranks, E_local] Pass 2: per-substream-per-expert recv count
    const int* tile_id_to_expert;     // [total_tiles] int32                      Eager Pass 2: slot → e_local via slot/tile_m
    // Sender-side prefix matrices (drive sender + sender_coordinator's
    // per-channel send counts and the negative-encoded meta the forwarder
    // reads to derive `num_tokens_to_recv_from_rdma`).
    const int* gbl_channel_prefix_matrix;          // [num_world_ranks, num_channels]   sender-side, per-(dst_world, channel)
    const int* rdma_channel_prefix_matrix;         // [num_rdma_ranks, num_channels]    sender-side, per-(dst_rdma, channel)
    // Receiver-side: forwarder uses these for per-(src_rdma) base offset
    // computation; receiver uses recv_gbl_channel_prefix_matrix to map
    // (channel, src_world) → starting recv_token_id.
    const int* recv_rdma_rank_prefix_sum;          // [num_rdma_ranks]
    const int* recv_gbl_channel_prefix_matrix;     // [num_world_ranks, num_channels]
};

struct DispatchGradsTileSignal {
    int* bwd_dispatch_arrival_count;  // [total_tiles] int32  atomic-add target during Pass 2
    const int* pool_arrival_target;   // [total_tiles] int32  firing target (same as fwd's)
    int64_t* bwd_y_ready;             // [total_tiles] int64  per-tile release stamp (consumed by kernel_y_bwd)
    int64_t dispatch_seq;
};

struct DispatchGradsShape {
    int num_tokens;
    int hidden_int4;
    int num_topk;
    int num_experts;
    int tile_m;
};

void launch_dispatch_grads_main(const DispatchGradsIO& io,
                                const DispatchGradsRouting& routing,
                                const DispatchGradsTileSignal& tile_signal,
                                const DispatchGradsShape& shape,
                                const DispatchEnv& env,
                                int num_rdma_ranks,
                                int num_channels,
                                cudaStream_t stream);

// Pre-combine fixup: in-place reverse-order sentinel encoding of
// `combined_rdma_head` (input: dispatch's `send_rdma_head`) and
// `combined_nvl_head` (input: dispatch's `send_nvl_head`). For tokens whose
// head entry is `< 0` (no contribution from that source), encode the *next*
// real head ahead of it as `-last_head - 1`; combine's receivers use this
// to skip cleanly past gaps without re-reading the counter region.
//
// (The legacy buffer-cleanup half of this kernel — IBGDA quiet + RDMA team
// sync + NVL barrier + memset of dispatch ring-control regions — is gone:
// every polled slot is iter-disambiguated by the cumulative head/tail /
// RDMA meta sentinel-amo / NVL gen-stamp protocols. Block 0 of the kernel
// is now an early return to preserve block-id offsets in blocks 1+.)
void cached_notify_combine(int hidden_int4,
                           int num_topk,
                           int num_ranks,
                           int num_channels,
                           int num_combined_tokens,
                           int* combined_rdma_head,
                           const int* rdma_channel_prefix_matrix,
                           const int* rdma_rank_prefix_sum,
                           int* combined_nvl_head,
                           cudaStream_t stream);

// combine_main_kernel — used by both forward combine and backward
// combine_grads. Same arg surface as `intranode::launch_combine_main` for the
// unified args (recv_x, recv_topk_weights_out, x, per_slot_weights,
// recv_token_to_slots, compute_done_per_token, combine_seq); internode adds
// the RDMA-staging plumbing (combined_rdma_head, combined_nvl_head,
// src_meta, recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
// gbl_channel_prefix_matrix). Streaming gate at kNVLSender only.
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
                         const int64_t* compute_done_per_token,
                         int64_t combine_seq,
                         // 0 = fwd combine, 1 = bwd combine_grads. Phase-
                         // distinguishes the NVL gen-stamp tag so fwd's slot
                         // residue can't alias the bwd reader on the same
                         // (channel, dst_nvl, rdma_src) slot within one layer
                         // where `combine_seq` is shared between the two
                         // phases.
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
                         cudaStream_t stream,
                         int num_channels,
                         // Persistent reader_prev arrays for the combine
                         // direction's RDMA head/tail. Same role and layout
                         // as `DispatchEnv::reader_prev_*`. Shared by fwd
                         // combine + bwd combine_grads (both on
                         // streams.combine — stream-ordered).
                         uint32_t* combine_reader_prev_head,
                         uint32_t* combine_reader_prev_tail);

}  // namespace internode

}  // namespace stream_ep
