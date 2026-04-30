#pragma once

// Forcibly disable NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>

#include <tuple>
#include <vector>

#include "config.hpp"
#include "event.hpp"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME deep_ep_cpp
#endif

namespace shared_memory {

union MemHandleInner {
    cudaIpcMemHandle_t cuda_ipc_mem_handle;
    CUmemFabricHandle cu_mem_fabric_handle;
};

struct MemHandle {
    MemHandleInner inner;
    size_t size;
};

constexpr size_t HANDLE_SIZE = sizeof(MemHandle);

class SharedMemoryAllocator {
public:
    SharedMemoryAllocator(bool use_fabric);
    void malloc(void** ptr, size_t size);
    void free(void* ptr);
    void get_mem_handle(MemHandle* mem_handle, void* ptr);
    void open_mem_handle(void** ptr, MemHandle* mem_handle);
    void close_mem_handle(void* ptr);

private:
    bool use_fabric;
};
}  // namespace shared_memory

namespace deep_ep {

struct Buffer {
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

private:
    // Low-latency mode buffer
    int low_latency_buffer_idx = 0;
    bool low_latency_mode = false;

    // NVLink Buffer
    int64_t num_nvl_bytes;
    void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;

    // NVSHMEM Buffer
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;

    // Shrink mode buffer
    bool enable_shrink = false;
    int* mask_buffer_ptr = nullptr;
    int* sync_buffer_ptr = nullptr;

    // Device info and communication
    int device_id;
    int num_device_sms;
    int rank, rdma_rank, nvl_rank;
    int num_ranks, num_rdma_ranks, num_nvl_ranks;
    shared_memory::MemHandle ipc_handles[NUM_MAX_NVL_PEERS];

    // Stream for communication
    at::cuda::CUDAStream comm_stream;

    // After IPC/NVSHMEM synchronization, this flag will be true
    bool available = false;

    // Whether explicit `destroy()` is required.
    bool explicitly_destroy;
    // After `destroy()` be called, this flag will be true
    bool destroyed = false;

    // Barrier signals
    int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    int** barrier_signal_ptrs_gpu = nullptr;

    // Workspace
    void* workspace = nullptr;

    // Streaming-MoE count-exchange inbox: per-(channel, src_rank, local_expert) int32.
    // Lives in the same IPC slab as buffer_ptrs / barrier_signal_ptrs. Sized for the
    // worst case (max channels = num_device_sms / 2, max experts = NUM_MAX_LOCAL_EXPERTS).
    int64_t streaming_section_offset = 0;
    int64_t streaming_section_bytes = 0;

    // Host-side MoE info
    volatile int* moe_recv_counter = nullptr;
    int* moe_recv_counter_mapped = nullptr;

    // Host-side expert-level MoE info
    volatile int* moe_recv_expert_counter = nullptr;
    int* moe_recv_expert_counter_mapped = nullptr;

    // Host-side RDMA-level MoE info
    volatile int* moe_recv_rdma_counter = nullptr;
    int* moe_recv_rdma_counter_mapped = nullptr;

    // Streaming-MoE total_tiles host-mapped slot. Written by streaming_metadata_init,
    // polled by the dispatch flow alongside moe_recv_counter / moe_recv_expert_counter
    // as the single sync point per layer.
    volatile int* streaming_total_tiles = nullptr;
    int* streaming_total_tiles_mapped = nullptr;

    shared_memory::SharedMemoryAllocator shared_memory_allocator;

public:
    Buffer(int rank,
           int num_ranks,
           int64_t num_nvl_bytes,
           int64_t num_rdma_bytes,
           bool low_latency_mode,
           bool explicitly_destroy,
           bool enable_shrink,
           bool use_fabric);

    ~Buffer() noexcept(false);

    bool is_available() const;

    bool is_internode_available() const;

    int get_num_rdma_ranks() const;

    int get_rdma_rank() const;

    int get_root_rdma_rank(bool global) const;

    int get_local_device_id() const;

    pybind11::bytearray get_local_ipc_handle() const;

    pybind11::bytearray get_local_nvshmem_unique_id() const;

    torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const;

    torch::Stream get_comm_stream() const;

    void sync(const std::vector<int>& device_ids,
              const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
              const std::optional<pybind11::bytearray>& root_unique_id_opt);

    void destroy();

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>> get_dispatch_layout(
        const torch::Tensor& topk_idx,
        int num_experts,
        std::optional<EventHandle>& previous_event,
        bool async,
        bool allocate_on_comm_stream);

    // Streaming-MoE consolidated dispatch (intranode, pool layout). Single host
    // call producing pool-shape outputs. Two kernel launches + one host sync:
    //   1. streaming_dispatch_metadata: fused count_exchange + metadata derivation.
    //   2. host poll on {moe_recv_counter, moe_recv_expert_counter, streaming_total_tiles}.
    //   3. tile_arrays_init: per-tile (tile_id_to_expert, pool_arrival_target).
    //   4. dispatch: pool-layout receiver writes pool[TK_padded, H], pool_x_scales,
    //      pool_topk_weight, pool_recv_token, pool_k_slot. Pass 2 fires tile_ready
    //      in expert-major order (preserves wave caching of W1[e]).
    //
    // Return tuple:
    //   0  pool                  Tensor[TK_padded, hidden]   pool data (expert-major, BLOCK_M-padded)
    //   1  pool_x_scales         optional<Tensor>            FP8 scales (pool-layout)
    //   2  pool_topk_weight      Tensor[TK_padded]           per-pool-slot weight
    //   3  pool_recv_token       Tensor[TK_padded]           per-pool-slot recv-token id (-1 = padding)
    //   4  pool_k_slot           Tensor[TK_padded]           per-pool-slot k (-1 = padding)
    //   5  recv_topk_weights     Tensor[T_recv, num_topk]    per-recv-token weights (combine input)
    //   6  recv_src_idx          Tensor[T_recv]              recv-token → source token idx (combine input)
    //   7  send_head             Tensor[T, R]                sender ring slot (combine input)
    //   8  num_recv_tokens_per_expert_list  vector<int>      per-expert counts (host)
    //   9  rank_prefix_matrix    Tensor[R, R]                cumulative recv-tokens per source rank
    //   10 channel_prefix_matrix Tensor[R, num_channels]     sender-side per-(rank, channel) cum count
    //   11 recv_channel_prefix_matrix  Tensor[R, num_channels]
    //   12 expert_frequency      Tensor[E_local]             per-expert (token, k) pair count
    //   13 expert_pool_block_offset    Tensor[E_local + 1]   pool-block prefix-sum (in BLOCK_M-tile units)
    //   14 base_pool             Tensor[num_channels, R, E_local]  per-substream-per-expert pool slot start
    //   15 tile_id_to_expert     Tensor[total_tiles]         per-tile expert lookup
    //   16 pool_arrival_target   Tensor[total_tiles]         per-tile write count for tile_ready firing
    //   17 tile_ready            Tensor[total_tiles] int64   per-tile release stamp
    //   18 a_ready               Tensor[total_tiles] int64   per-tile kernel-A→Y release stamp (zero-init)
    //   19 per_token_remaining   Tensor[T_recv] int32        K_local(r); kernel Y atomicSubs
    //   20 compute_done_per_token  Tensor[T_recv] int64      per-token Y→combine release stamp (zero-init)
    //   21 o                     Tensor[T_recv, hidden]      kernel Y atomic-scatter destination (zero-init)
    //   22 total_tiles           int                         scalar count
    //   23 event                 optional<EventHandle>
    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               std::vector<int>,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               int,
               std::optional<EventHandle>>
    intranode_dispatch(const torch::Tensor& x,
                       const std::optional<torch::Tensor>& x_scales,
                       const torch::Tensor& topk_idx,
                       const torch::Tensor& topk_weights,
                       const torch::Tensor& is_token_in_rank,
                       int num_experts,
                       int expert_alignment,
                       int tile_m,
                       int64_t dispatch_seq,
                       const Config& config,
                       std::optional<EventHandle>& previous_event,
                       bool async,
                       bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> intranode_combine(
        const torch::Tensor& x,
        const std::optional<torch::Tensor>& topk_weights,
        const std::optional<torch::Tensor>& bias_0,
        const std::optional<torch::Tensor>& bias_1,
        const torch::Tensor& src_idx,
        const torch::Tensor& rank_prefix_matrix,
        const torch::Tensor& channel_prefix_matrix,
        const torch::Tensor& send_head,
        const Config& config,
        std::optional<EventHandle>& previous_event,
        bool async,
        bool allocate_on_comm_stream);

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::vector<int>,
               torch::Tensor,
               torch::Tensor,
               std::optional<torch::Tensor>,
               torch::Tensor,
               std::optional<torch::Tensor>,
               torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    internode_dispatch(const torch::Tensor& x,
                       const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx,
                       const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank,
                       const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                       const torch::Tensor& is_token_in_rank,
                       const std::optional<torch::Tensor>& num_tokens_per_expert,
                       int cached_num_recv_tokens,
                       int cached_num_rdma_recv_tokens,
                       const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,
                       const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                       const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,
                       const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                       int expert_alignment,
                       int num_worst_tokens,
                       const Config& config,
                       std::optional<EventHandle>& previous_event,
                       bool async,
                       bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> internode_combine(
        const torch::Tensor& x,
        const std::optional<torch::Tensor>& topk_weights,
        const std::optional<torch::Tensor>& bias_0,
        const std::optional<torch::Tensor>& bias_1,
        const torch::Tensor& src_meta,
        const torch::Tensor& is_combined_token_in_rank,
        const torch::Tensor& rdma_channel_prefix_matrix,
        const torch::Tensor& rdma_rank_prefix_sum,
        const torch::Tensor& gbl_channel_prefix_matrix,
        const torch::Tensor& combined_rdma_head,
        const torch::Tensor& combined_nvl_head,
        const Config& config,
        std::optional<EventHandle>& previous_event,
        bool async,
        bool allocate_on_comm_stream);

    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               std::optional<EventHandle>,
               std::optional<std::function<void()>>>
    low_latency_dispatch(const torch::Tensor& x,
                         const torch::Tensor& topk_idx,
                         const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                         const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                         int num_max_dispatch_tokens_per_rank,
                         int num_experts,
                         bool use_fp8,
                         bool round_scale,
                         bool use_ue8m0,
                         bool async,
                         bool return_recv_hook);

    std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> low_latency_combine(
        const torch::Tensor& x,
        const torch::Tensor& topk_idx,
        const torch::Tensor& topk_weights,
        const torch::Tensor& src_info,
        const torch::Tensor& layout_range,
        const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
        int num_max_dispatch_tokens_per_rank,
        int num_experts,
        bool use_logfmt,
        bool zero_copy,
        bool async,
        bool return_recv_hook,
        const std::optional<torch::Tensor>& out = std::nullopt);

    torch::Tensor get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const;

    void low_latency_update_mask_buffer(int rank_to_mask, bool mask);

    void low_latency_query_mask_buffer(const torch::Tensor& mask_status);

    void low_latency_clean_mask_buffer();
};

}  // namespace deep_ep
