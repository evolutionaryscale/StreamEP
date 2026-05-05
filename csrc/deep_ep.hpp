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
#define TORCH_EXTENSION_NAME stream_ep_cpp
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

// All outputs of `Buffer::intranode_dispatch`. Bound to Python via pybind11
// with attribute access (see PYBIND11_MODULE in deep_ep.cpp); the Python-side
// `Buffer.dispatch` repacks these into the public `StreamingHandle` dataclass.
//
// Field grouping:
//   pool_*                   pool data + per-pool-slot scalars (kernel A reads,
//                            kernel Y / combine read via pool_recv_token).
//   recv_*, send_head        per-recv-token combine inputs.
//   *_prefix_matrix          per-(sender, channel) cumulative counts (sender +
//                            receiver views).
//   expert_*, base_pool      pool-shape metadata (kernel A scheduler).
//   tile_id_to_expert,
//   pool_arrival_target,
//   tile_ready, a_ready      per-tile arrays (scheduler + cross-stream signals).
//   per_token_remaining,
//   compute_done_per_token,
//   o                        kernel Y atomic-scatter destination + Y→combine gate.
//   recv_token_to_slots,
//   k_local_count            backward-pass scaffolding written by fwd Pass B
//                            (dispatch_grads receiver gathers slots; bwd setup
//                            memcpy's k_local_count into bwd_per_token_remaining).
struct StreamingDispatchOutputs {
    torch::Tensor pool;
    torch::Tensor pool_topk_weight;
    torch::Tensor pool_recv_token;
    torch::Tensor pool_k_slot;

    torch::Tensor send_head;

    torch::Tensor rank_prefix_matrix;
    torch::Tensor channel_prefix_matrix;
    torch::Tensor recv_channel_prefix_matrix;

    torch::Tensor expert_frequency;
    torch::Tensor expert_pool_block_offset;
    torch::Tensor base_pool;
    torch::Tensor seen_per_substream;

    torch::Tensor tile_id_to_expert;
    torch::Tensor pool_arrival_target;
    torch::Tensor tile_ready;

    torch::Tensor a_ready;
    torch::Tensor per_token_remaining;
    torch::Tensor compute_done_per_token;
    torch::Tensor o;

    torch::Tensor recv_token_to_slots;
    torch::Tensor k_local_count;

    int total_tiles;

    EventHandle metadata_done_event;
};

struct Buffer {
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

private:
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

    // Streaming-MoE total_tiles host-mapped slot. Written by streaming_dispatch_metadata,
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

    void sync(const std::vector<int>& device_ids,
              const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
              const std::optional<pybind11::bytearray>& root_unique_id_opt);

    void destroy();

    // Streaming-MoE consolidated dispatch (intranode, pool layout). Two kernels
    // + one host sync per call: a fused metadata kernel (cross-rank count
    // exchange + per-tile arrays), a host poll on
    // {moe_recv_counter, moe_recv_expert_counter, streaming_total_tiles}, and
    // the dispatch main kernel (pool-layout receiver). Outputs returned as a
    // `StreamingDispatchOutputs` struct with named fields (see top of header
    // for field semantics). `metadata_done_event` is recorded between the two
    // kernels — consumer streams `wait_event` on it to safely read metadata
    // tensors without serializing against dispatch main.
    //
    // Streams: kernels run on `at::cuda::getCurrentCUDAStream()` — caller-managed.
    StreamingDispatchOutputs intranode_dispatch(
        const torch::Tensor& x,
        const torch::Tensor& topk_idx,
        const torch::Tensor& topk_weights,
        const torch::Tensor& is_token_in_rank,
        int num_experts,
        int expert_alignment,
        int tile_m,
        int64_t dispatch_seq,
        const Config& config);

    // Backward dispatch_grads: ship dL/dy[t] origin → expert ranks along the
    // same (t, dst_rank) routing as forward dispatch. Receiver writes K times
    // into dL_do_pool[slot] for each landed packet, using `recv_token_to_slots`
    // (populated by fwd Pass B) to look up slots without rerunning Pass A.
    // No metadata kernel, no host poll, no IPC barrier on the metadata path —
    // the only cross-rank sync is the small barrier between the channel-control
    // memset and the kernel launch. Returns (dL_do_pool, bwd_y_ready); caller
    // (orchestrator) holds bwd_y_ready for kernel_y_bwd to acquire-spin on.
    std::tuple<torch::Tensor, torch::Tensor> intranode_dispatch_grads(
        const torch::Tensor& dL_dy,
        const torch::Tensor& is_token_in_rank,
        const torch::Tensor& recv_token_to_slots,
        const torch::Tensor& base_pool,
        const torch::Tensor& seen_per_substream,
        const torch::Tensor& pool_arrival_target,
        const torch::Tensor& rank_prefix_matrix,
        int num_experts,
        int num_topk,
        int tile_m,
        int64_t TK_padded,
        int64_t dispatch_seq,
        const Config& config);

    // Combine: reduces per-recv-token x[r] back to source ranks. Used for both
    // forward (sums weighted x → out) and backward combine_grads (sums per-recv-token
    // dL/dx → per-source-token dL/dx). Per-direction differences are entirely in
    // args; same kernel underlies both. See `combine_main_kernel` in intranode.cu.
    //
    //   per_slot_weights      fwd: handle.pool_topk_weight    bwd: weight_grads
    //   recv_token_to_slots   handle.recv_token_to_slots (same for both directions)
    //   x                     fwd: handle.o                   bwd: dL/dx_per_r
    //   bias_0 / bias_1       optional (fwd only)             nullopt for bwd
    //   compute_done_per_token  fwd: kernel_y release stamp   bwd: kernel_a_bwd release stamp
    //   combine_seq           caller's monotonic int (`dispatch_seq`)
    std::tuple<torch::Tensor, torch::Tensor> intranode_combine(
        const torch::Tensor& x,
        const torch::Tensor& per_slot_weights,
        const torch::Tensor& recv_token_to_slots,
        const torch::Tensor& rank_prefix_matrix,
        const torch::Tensor& channel_prefix_matrix,
        const torch::Tensor& send_head,
        const torch::Tensor& compute_done_per_token,
        int64_t combine_seq,
        const Config& config);

    std::tuple<torch::Tensor,
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
               std::optional<torch::Tensor>>
    internode_dispatch(const torch::Tensor& x,
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
                       const Config& config);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>> internode_combine(
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
        const Config& config);

};

}  // namespace deep_ep
