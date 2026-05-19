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

namespace stream_ep {

// All outputs of `Buffer::intranode_dispatch`. Bound to Python via pybind11
// with attribute access (see PYBIND11_MODULE in stream_ep.cpp); the Python-side
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
//   pool_arrival_count       per-tile arrays. Scheduler spins on
//                            `pool_arrival_count[tile] == pool_arrival_target[tile]`
//                            (set by dispatch's Pass 2 `red.release.gpu.add`).
//   k_local_remaining,
//   y_done_per_token,
//   o                        kernel Y atomic-scatter destination + Y→combine gate.
//   recv_token_to_slots,
//   k_local_total            backward-pass scaffolding written by fwd Pass B
//                            (dispatch_grads receiver gathers slots; bwd setup
//                            memcpy's k_local_total into bwd_k_local_remaining).

struct StreamingDispatchOutputs {
    torch::Tensor pool;
    torch::Tensor pool_topk_weight;
    torch::Tensor pool_recv_token;
    torch::Tensor pool_k_slot;

    // Intranode combine plumbing. For internode, `send_head` /
    // `channel_prefix_matrix` / `recv_channel_prefix_matrix` are empty and
    // the `*_rdma_*` / `*_gbl_*` / `recv_src_meta` fields below carry the
    // equivalent reverse-routing data.
    torch::Tensor send_head;

    // Per-source-rank cumulative recv-token offset. Sized [num_ranks, num_ranks]
    // for intranode and [num_world_ranks, num_world_ranks] for internode; only
    // this rank's column is populated.
    torch::Tensor rank_prefix_matrix;
    torch::Tensor channel_prefix_matrix;
    torch::Tensor recv_channel_prefix_matrix;

    torch::Tensor expert_frequency;
    torch::Tensor expert_pool_block_offset;
    torch::Tensor base_pool;
    torch::Tensor seen_per_substream;

    torch::Tensor tile_id_to_expert;
    torch::Tensor pool_arrival_target;
    torch::Tensor pool_arrival_count;

    torch::Tensor k_local_remaining;
    torch::Tensor y_done_per_token;
    torch::Tensor o;

    torch::Tensor recv_token_to_slots;
    torch::Tensor k_local_total;

    int total_tiles;

    EventHandle metadata_done_event;

    // Internode-only combine plumbing. Empty tensors for the intranode path.
    // Produced by the metadata kernel (the four prefix tensors below) and by
    // the dispatch_main kernel's forwarder + NVL receiver (`send_rdma_head` /
    // `send_nvl_head` / `recv_rdma_channel_prefix_matrix` /
    // `recv_gbl_channel_prefix_matrix` / `recv_src_meta`). The combine path
    // (yet to be refactored) will consume these alongside `encode_combine_heads`'s
    // `combined_rdma_head` / `combined_nvl_head` outputs.
    torch::Tensor rdma_channel_prefix_matrix;       // [num_rdma_ranks, num_channels]
    torch::Tensor recv_rdma_rank_prefix_sum;        // [num_rdma_ranks]
    torch::Tensor gbl_channel_prefix_matrix;        // [num_world_ranks, num_channels]
    torch::Tensor recv_gbl_rank_prefix_sum;         // [num_world_ranks]
    torch::Tensor recv_rdma_channel_prefix_matrix;  // [num_rdma_ranks, num_channels]
    torch::Tensor recv_gbl_channel_prefix_matrix;   // [num_world_ranks, num_channels]
    torch::Tensor send_rdma_head;                   // [num_tokens, num_rdma_ranks]
    torch::Tensor send_nvl_head;                    // [num_rdma_recv_tokens, NUM_MAX_NVL_PEERS]
    torch::Tensor recv_src_meta;                    // [T_recv, get_source_meta_bytes()]  uint8
};

struct Buffer {
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

private:
    // NVLink Buffer
    int64_t num_nvl_bytes;
    void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;

    // NVSHMEM Buffer. Allocated as `2 * num_rdma_bytes` and split into two
    // halves: the first half (`rdma_buffer_ptr`) serves fwd dispatch + bwd
    // dispatch_grads (both on streams.dispatch); the second half
    // (`rdma_buffer_ptr_combine = rdma_buffer_ptr + num_rdma_bytes`) serves
    // fwd combine + bwd combine_grads (both on streams.combine). Disjoint
    // halves eliminate a latent SymBuffer aliasing where combine's smaller
    // per-token bytes (no topk_idx in the wire format) put combine's
    // head/tail offsets inside dispatch's data region. NVSHMEM symmetric
    // heap means `+ num_rdma_bytes` is the same offset on every rank.
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;
    void* rdma_buffer_ptr_combine = nullptr;

    // Persistent ring control state. The cross-host head/tail slots in the
    // RDMA SymBuffers accumulate across iters via amo_nonfetch_add (the
    // sender publishes deltas, the slot's value is the cumulative across
    // all iters). Each kernel's reader reads `prev` from these arrays at
    // warp entry, computes iter-local cached values as `cur - prev`, then
    // writes the latest cumulative back to the array at kernel exit. The
    // arrays are written in lockstep with kernel exit (no race with peer
    // amos), unlike an in-kernel `ld(slot)` seed which is racy because
    // peer amos land asynchronously through the NIC.
    //
    // Sized [max_num_channels × num_rdma_ranks] uint32, ~256 B each at
    // production; ~1 KB total. Allocated zero-init in `Buffer::Buffer`,
    // freed in destructor. Storage is uint32 (matching the on-the-wire
    // 4-byte AMO width); cross-iter wrap is absorbed by modular uint32
    // subtraction in the read-side `cur - prev` math and a signed-difference
    // CAS in the write-side helper (`atomicmax_reader_prev_cumulative`).
    // Wrap horizon is effectively infinite — the protocol never compares
    // across-iter slot values directly, so it doesn't care that the slot
    // (and the array) cycle through 2^32 every ~500k training steps.
    uint32_t* dispatch_reader_prev_head = nullptr;
    uint32_t* dispatch_reader_prev_tail = nullptr;
    uint32_t* combine_reader_prev_head  = nullptr;
    uint32_t* combine_reader_prev_tail  = nullptr;
    // Persistent prev-sentinel array for the RDMA dispatch meta region (C3).
    // The meta SymBuffer's slot 30 ("kRdmaMetaSentinelSlot") accumulates
    // across iters via amo_nonfetch_add — sender bulk_puts the 18 data ints
    // into slots 0..17, then issues an amo on slot 30 to invalidate the L2
    // line and signal availability. The forwarder reads its prev from this
    // array at warp entry, spins on `ld(slot 30) > prev`, reads raw data
    // slots, and atomicMaxes the latest cumulative back at exit.
    // Sized [max_num_channels × num_rdma_ranks] int32. Same lifetime as
    // reader_prev_*; freed in destructor; shared by fwd dispatch + bwd
    // dispatch_grads (same dispatch stream).
    int* dispatch_meta_sentinel_prev = nullptr;

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

    // "Kernel started" flags for the cross-stream launch gate. Each is a
    // single int32 in device memory. dispatch_main_kernel / dispatch_grads_main_kernel
    // atomicAdd these once at entry (block 0 thread 0). The host issues
    // `cuStreamBatchMemOp` wait_value_geq on the compute stream before
    // launching the consumer kernel (kernel_a / kernel_y_bwd), forcing the
    // consumer to wait until the dispatch kernel's block 0 is actually
    // co-resident — so dispatch grabs SMs first.
    int* dispatch_main_started_flag = nullptr;
    int* dispatch_grads_started_flag = nullptr;
    int dispatch_main_issued_count = 0;
    int dispatch_grads_issued_count = 0;

    // Caller's compute-stream cudaStream_t (set once from Python via
    // `set_compute_stream_handle`; zero until set). Used to
    // `record_stream` every per-call `torch::empty`/`torch::zeros` slab
    // and pool tensor in dispatch / combine / dispatch_grads so the
    // caching allocator waits for compute (consumer) — not just for
    // communicate (allocation stream) — before reusing the storage. The
    // kernel_y / kernel_a writes to slabs like `y_done_per_token`, `o`,
    // and pool run on `compute`; without this the allocator can recycle
    // a slab while compute is still touching it and the next iter's
    // memset clobbers in-flight values.
    int64_t compute_stream_handle_ = 0;

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

    // Wait on the compute stream until the most-recently-launched
    // dispatch_main_kernel / dispatch_grads_main_kernel has actually entered
    // execution (block 0 thread 0 atomicAdd'd the started_flag). Implemented
    // as a host-queued `cuStreamBatchMemOp` wait_value_geq. The wait fires at
    // the GPU front-end so the compute stream's pending kernel sits without
    // consuming an SM; once dispatch has its block 0 co-resident, the wait
    // passes and the compute kernel can launch onto the remaining SMs.
    // Streams: takes a torch CUDAStream — typically the caller's compute stream.
    void wait_dispatch_main_started(int64_t stream_handle);
    void wait_dispatch_grads_started(int64_t stream_handle);

    // Register the caller's compute-stream cudaStream_t. Subsequent
    // dispatch / combine / dispatch_grads calls record_stream every
    // per-call slab/pool tensor onto it so the caching allocator waits
    // for compute (consumer) before reusing the storage. Called once
    // per (Buffer, StreamHolder) pair by `stream_moe_func`; safe to call
    // multiple times.
    void set_compute_stream_handle(int64_t stream_handle);

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
    // memset and the kernel launch. Returns (dL_do_pool,
    // bwd_dispatch_arrival_count); caller (orchestrator) holds the count for
    // kernel_y_bwd to spin on (count == pool_arrival_target).
    std::tuple<torch::Tensor, torch::Tensor, EventHandle> intranode_dispatch_grads(
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
    //   y_done_per_token  fwd: kernel_y release stamp   bwd: kernel_a_bwd release stamp
    //   combine_seq           caller's monotonic int (`dispatch_seq`)
    // `is_fwd` = true (fwd combine): drops the per-K topk-weight wire
    // payload + receiver reduce; returns `c10::nullopt` in the second slot
    // (Python surfaces as `None`). Kernel Y already pre-multiplies
    // pool_topk_weight per row, so `recv_x[t]` is the full Σ_k w_k·y_k.
    // `is_fwd` = false (bwd combine_grads): unchanged — ships dL/dweight
    // per K, receiver sums into `recv_topk_weights_out`.
    std::tuple<torch::Tensor, c10::optional<torch::Tensor>> intranode_combine(
        const torch::Tensor& x,
        const torch::Tensor& per_slot_weights,
        const torch::Tensor& recv_token_to_slots,
        const torch::Tensor& rank_prefix_matrix,
        const torch::Tensor& channel_prefix_matrix,
        const torch::Tensor& send_head,
        const torch::Tensor& y_done_per_token,
        int64_t combine_seq,
        bool is_fwd,
        const Config& config);

    // Streaming-MoE consolidated dispatch (internode, pool layout). Mirrors
    // the intranode entry point: one folded metadata kernel + one host poll +
    // one dispatch_main kernel. Returns the same `StreamingDispatchOutputs`
    // shape as intranode, plus the internode-specific combine-plumbing
    // tensors (see struct definition above).
    //
    // Streams: kernels run on `at::cuda::getCurrentCUDAStream()` — caller-managed.
    StreamingDispatchOutputs internode_dispatch(
        const torch::Tensor& x,
        const torch::Tensor& topk_idx,
        const torch::Tensor& topk_weights,
        const torch::Tensor& is_token_in_rank,
        int num_experts,
        int expert_alignment,
        int tile_m,
        int64_t dispatch_seq,
        const Config& config);

    // Streaming-MoE bwd dispatch (internode, pool layout). Mirrors
    // `Buffer::intranode_dispatch_grads`: ships dL/dy[t] origin → expert ranks
    // along the same (t, dst_rank) routing as fwd dispatch. Receiver writes
    // dL_do_pool[slot] K times per packet using `recv_token_to_slots`
    // (populated by fwd Pass B), no Pass A. Returns
    // (dL_do_pool, bwd_dispatch_arrival_count).
    //
    // No metadata kernel, no host poll — the routing tensors live on
    // `StreamingDispatchOutputs` from the fwd dispatch and are reused here.
    //
    // Streams: kernel runs on `at::cuda::getCurrentCUDAStream()`.
    std::tuple<torch::Tensor, torch::Tensor, EventHandle> internode_dispatch_grads(
        const torch::Tensor& dL_dy,
        const torch::Tensor& is_token_in_rank,
        const StreamingDispatchOutputs& dispatch_out,
        int64_t TK_padded,
        int64_t dispatch_seq,
        const Config& config);

    // Streaming-MoE combine (internode, pool layout). Two kernels per call:
    // `internode::encode_combine_heads` (buffer cleanup + reverse-order
    // sentinel encoding of `send_rdma_head` / `send_nvl_head`) followed by
    // `internode::launch_combine_main` (three-warp-role NVL→RDMA→origin
    // reduction). Mirrors `Buffer::intranode_combine`'s two-kernel-one-method
    // pattern (`stream_ep.cpp:1218 + 1235`); same arg semantics for the
    // unified surface (x = handle.o for fwd / dL/dx_per_r for bwd;
    // per_slot_weights = pool_topk_weight for fwd / weight_grads for bwd;
    // y_done_per_token / combine_seq drive the streaming gate at
    // `kNVLSender`).
    //
    // Both passes mutate `dispatch_out.send_rdma_head` and
    // `dispatch_out.send_nvl_head` in place (the sentinel encoding from the
    // first pass IS the input to the second) — caller should treat them as
    // consumed after this call.
    //
    // Streams: kernels run on `at::cuda::getCurrentCUDAStream()`.
    // See `intranode_combine` for `is_fwd` semantics — same contract.
    std::tuple<torch::Tensor, c10::optional<torch::Tensor>> internode_combine(
        const torch::Tensor& x,
        const torch::Tensor& per_slot_weights,
        const StreamingDispatchOutputs& dispatch_out,
        const torch::Tensor& y_done_per_token,
        int64_t combine_seq,
        // 0 = fwd combine, 1 = bwd combine_grads. Forwarded into
        // `internode::launch_combine_main` to phase-distinguish the NVL
        // gen-stamp tag on the combine ring.
        int64_t combine_phase,
        bool is_fwd,
        const Config& config);

};

}  // namespace stream_ep
