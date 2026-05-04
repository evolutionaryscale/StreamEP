#pragma once

#include <vector>

#include "configs.cuh"

namespace deep_ep {

// Intranode runtime
namespace intranode {

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream);

}  // namespace intranode

// Internode runtime
namespace internode {

std::vector<uint8_t> get_unique_id();

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode);

void* alloc(size_t size, size_t alignment);

void free(void* ptr);

void barrier();

void finalize();

}  // namespace internode

// Layout kernels
namespace layout {

void get_dispatch_layout(const topk_idx_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         cudaStream_t stream);

}  // namespace layout

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
    float* pool_x_scales;          // [TK_padded, num_scales] (FP8 only)
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
    const float* x_scales;         // [num_tokens, num_scales] (FP8 only)
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
    int num_scales;
    int scale_token_stride;
    int scale_hidden_stride;
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

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_rdma_rank,
                     int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     const bool* is_token_in_rank,
                     int num_tokens,
                     int num_worst_tokens,
                     int num_channels,
                     int hidden_int4,
                     int num_scales,
                     int num_topk,
                     int expert_alignment,
                     int* rdma_channel_prefix_matrix,
                     int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix,
                     int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr,
                     int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int64_t num_rdma_bytes,
                     int64_t num_nvl_bytes,
                     bool low_latency_mode);

void dispatch(void* recv_x,
              float* recv_x_scales,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              void* recv_src_meta,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              int* send_rdma_head,
              int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix,
              int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix,
              const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix,
              const int* recv_gbl_rank_prefix_sum,
              const bool* is_token_in_rank,
              int num_tokens,
              int num_worst_tokens,
              int hidden_int4,
              int num_scales,
              int num_topk,
              int num_experts,
              int scale_token_stride,
              int scale_hidden_stride,
              void* rdma_buffer_ptr,
              int num_max_rdma_chunked_send_tokens,
              int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs,
              int num_max_nvl_chunked_send_tokens,
              int num_max_nvl_chunked_recv_tokens,
              int rank,
              int num_ranks,
              bool is_cached_dispatch,
              cudaStream_t stream,
              int num_channels,
              bool low_latency_mode);

void cached_notify(int hidden_int4,
                   int num_scales,
                   int num_topk_idx,
                   int num_topk_weights,
                   int num_ranks,
                   int num_channels,
                   int num_combined_tokens,
                   int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix,
                   const int* rdma_rank_prefix_sum,
                   int* combined_nvl_head,
                   void* rdma_buffer_ptr,
                   int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs,
                   int num_max_nvl_chunked_recv_tokens,
                   int** barrier_signal_ptrs,
                   int rank,
                   cudaStream_t stream,
                   int64_t num_rdma_bytes,
                   int64_t num_nvl_bytes,
                   bool is_cached_dispatch,
                   bool low_latency_mode);

void combine(cudaDataType_t type,
             void* combined_x,
             float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* combined_rdma_head,
             const int* combined_nvl_head,
             const void* src_meta,
             const int* rdma_channel_prefix_matrix,
             const int* rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
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
             bool low_latency_mode);

}  // namespace internode

// Internode low-latency kernels
namespace internode_ll {

void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              int rank,
                              int num_ranks,
                              int* mask_buffer,
                              int* sync_buffer,
                              cudaStream_t stream);

void dispatch(void* packed_recv_x,
              void* packed_recv_x_scales,
              int* packed_recv_src_info,
              int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* mask_buffer,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x,
              int* rdma_recv_count,
              void* rdma_x,
              const void* x,
              const topk_idx_t* topk_idx,
              int* next_clean,
              int num_next_clean_int,
              int num_tokens,
              int hidden,
              int num_max_dispatch_tokens_per_rank,
              int num_topk,
              int num_experts,
              int rank,
              int num_ranks,
              bool use_fp8,
              bool round_scale,
              bool use_ue8m0,
              void* workspace,
              int num_device_sms,
              cudaStream_t stream,
              int phases);

void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* mask_buffer,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             bool use_logfmt,
             void* workspace,
             int num_device_sms,
             cudaStream_t stream,
             int phases,
             bool zero_copy);

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* output_mask_tensor, cudaStream_t stream);

void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask, cudaStream_t stream);

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream);

}  // namespace internode_ll

}  // namespace deep_ep
