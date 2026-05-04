#include "deep_ep.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>
#include <pybind11/functional.h>
#include <torch/python.h>

#include <chrono>
#include <memory>

#include "kernels/api.cuh"
#include "kernels/configs.cuh"

namespace shared_memory {
void cu_mem_set_access_all(void* ptr, size_t size) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    CUmemAccessDesc access_desc[device_count];
    for (int idx = 0; idx < device_count; ++idx) {
        access_desc[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc[idx].location.id = idx;
        access_desc[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    CU_CHECK(cuMemSetAccess((CUdeviceptr)ptr, size, access_desc, device_count));
}

void cu_mem_free(void* ptr) {
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemRelease(handle));
}

size_t get_size_align_to_granularity(size_t size_raw, size_t granularity) {
    size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
    if (size == 0)
        size = granularity;
    return size;
}

SharedMemoryAllocator::SharedMemoryAllocator(bool use_fabric) : use_fabric(use_fabric) {}

void SharedMemoryAllocator::malloc(void** ptr, size_t size_raw) {
    if (use_fabric) {
        CUdevice device;
        CU_CHECK(cuCtxGetDevice(&device));

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        prop.location.id = device;

        size_t granularity = 0;
        CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        size_t size = get_size_align_to_granularity(size_raw, granularity);

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemCreate(&handle, size, &prop, 0));

        CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    } else {
        CUDA_CHECK(cudaMalloc(ptr, size_raw));
    }
}

void SharedMemoryAllocator::free(void* ptr) {
    if (use_fabric) {
        cu_mem_free(ptr);
    } else {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void SharedMemoryAllocator::get_mem_handle(MemHandle* mem_handle, void* ptr) {
    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    mem_handle->size = size;

    if (use_fabric) {
        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

        CU_CHECK(cuMemExportToShareableHandle(&mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    } else {
        CUDA_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
    }
}

void SharedMemoryAllocator::open_mem_handle(void** ptr, MemHandle* mem_handle) {
    if (use_fabric) {
        size_t size = mem_handle->size;

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));

        CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, 0, 0, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    } else {
        CUDA_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));
    }
}

void SharedMemoryAllocator::close_mem_handle(void* ptr) {
    if (use_fabric) {
        cu_mem_free(ptr);
    } else {
        CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
    }
}
}  // namespace shared_memory

namespace deep_ep {

Buffer::Buffer(int rank,
               int num_ranks,
               int64_t num_nvl_bytes,
               int64_t num_rdma_bytes,
               bool low_latency_mode,
               bool explicitly_destroy,
               bool enable_shrink,
               bool use_fabric)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      enable_shrink(enable_shrink),
      low_latency_mode(low_latency_mode),
      explicitly_destroy(explicitly_destroy),
      shared_memory_allocator(use_fabric) {
    // Metadata memory
    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

    // Common checks
    EP_STATIC_ASSERT(NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                   (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                   (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(num_nvl_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);

    // Get ranks
    CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
#ifdef DISABLE_NVSHMEM
    EP_HOST_ASSERT(num_rdma_ranks == 1 and not low_latency_mode and "NVSHMEM is disabled during compilation");
#endif

    // Get device info
    cudaDeviceProp device_prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    num_device_sms = device_prop.multiProcessorCount;

    // Number of per-channel bytes cannot be large
    EP_HOST_ASSERT(ceil_div<int64_t>(num_nvl_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(ceil_div<int64_t>(num_rdma_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());

    if (num_nvl_bytes > 0) {
        // Streaming-MoE inbox sizing: bound by max channels (num_device_sms / 2) and
        // NUM_MAX_LOCAL_EXPERTS. Each rank holds two inboxes:
        //   - `e_inbox[num_channels, num_ranks, num_local_experts]` int32 — per-(c, src, e)
        //     (token, k) counts (counts every routed pair, used for the streaming pipeline).
        //   - `u_inbox[num_channels, num_ranks]` int32 — per-(c, src) UNIQUE token counts
        //     (one increment per (token, src→dst) pair, regardless of how many of dst's
        //     local experts the token routes to). Used for num_recv and combine gating.
        // The inboxes are laid out adjacently at `streaming_section_offset`.
        int max_num_channels = num_device_sms / 2;
        int64_t e_inbox_bytes =
            static_cast<int64_t>(max_num_channels) * NUM_MAX_NVL_PEERS * NUM_MAX_LOCAL_EXPERTS * sizeof(int);
        int64_t u_inbox_bytes = static_cast<int64_t>(max_num_channels) * NUM_MAX_NVL_PEERS * sizeof(int);
        streaming_section_bytes = e_inbox_bytes + u_inbox_bytes;
        streaming_section_bytes =
            ((streaming_section_bytes + NUM_BUFFER_ALIGNMENT_BYTES - 1) / NUM_BUFFER_ALIGNMENT_BYTES) * NUM_BUFFER_ALIGNMENT_BYTES;
        streaming_section_offset = num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes;

        // Local IPC: alloc local memory and set local IPC handles
        shared_memory_allocator.malloc(&buffer_ptrs[nvl_rank],
                                       num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes +
                                           streaming_section_bytes);
        shared_memory_allocator.get_mem_handle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);
        buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

        // Set barrier signals
        barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        barrier_signal_ptrs_gpu =
            reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        auto current_stream = at::cuda::getCurrentCUDAStream();
        CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, current_stream));
        CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + streaming_section_offset, 0,
                                   streaming_section_bytes, current_stream));
    }

    // Create 32 MiB workspace
    CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, at::cuda::getCurrentCUDAStream()));

    // MoE counter
    CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    // MoE expert-level counter
    CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
        moe_recv_expert_counter[i] = -1;

    // MoE RDMA-level counter
    if (num_rdma_ranks > 0) {
        CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
        *moe_recv_rdma_counter = -1;
    }

    // Streaming-MoE total_tiles sync slot
    CUDA_CHECK(cudaMallocHost(&streaming_total_tiles, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&streaming_total_tiles_mapped, const_cast<int*>(streaming_total_tiles), 0));
    *streaming_total_tiles = -1;
}

Buffer::~Buffer() noexcept(false) {
    if (not explicitly_destroy) {
        destroy();
    } else if (not destroyed) {
        printf("WARNING: destroy() was not called before DeepEP buffer destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

bool Buffer::is_available() const {
    return available;
}

bool Buffer::is_internode_available() const {
    return is_available() and num_ranks > NUM_MAX_NVL_PEERS;
}

int Buffer::get_num_rdma_ranks() const {
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
    return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const {
    return device_id;
}

pybind11::bytearray Buffer::get_local_ipc_handle() const {
    const shared_memory::MemHandle& handle = ipc_handles[nvl_rank];
    return {reinterpret_cast<const char*>(&handle), sizeof(handle)};
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(rdma_rank == 0 and "Only RDMA rank 0 can get NVSHMEM unique ID");
    auto unique_id = internode::get_unique_id();
    return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) + offset;
    auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
    return torch::from_blob(base_ptr, num_bytes / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));
}

void Buffer::destroy() {
    EP_HOST_ASSERT(not destroyed);

    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    if (num_nvl_bytes > 0) {
        // Barrier
        intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, at::cuda::getCurrentCUDAStream());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Close remote IPC
        if (is_available()) {
            for (int i = 0; i < num_nvl_ranks; ++i)
                if (i != nvl_rank)
                    shared_memory_allocator.close_mem_handle(buffer_ptrs[i]);
        }

        // Free local buffer and error flag
        shared_memory_allocator.free(buffer_ptrs[nvl_rank]);
    }

    // Free NVSHMEM
#ifndef DISABLE_NVSHMEM
    if (is_available() and num_rdma_bytes > 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
        internode::barrier();
        internode::free(rdma_buffer_ptr);
        if (enable_shrink) {
            internode::free(mask_buffer_ptr);
            internode::free(sync_buffer_ptr);
        }
        internode::finalize();
    }
#endif

    // Free workspace and MoE counter
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));

    // Free chunked mode staffs
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));

    // Free streaming sync slot
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(streaming_total_tiles)));

    destroyed = true;
    available = false;
}

void Buffer::sync(const std::vector<int>& device_ids,
                  const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
                  const std::optional<pybind11::bytearray>& root_unique_id_opt) {
    EP_HOST_ASSERT(not is_available());

    // Sync IPC handles
    if (num_nvl_bytes > 0) {
        EP_HOST_ASSERT(num_ranks == device_ids.size());
        EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
            EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            EP_HOST_ASSERT(handle_str.size() == shared_memory::HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE);
                shared_memory_allocator.open_mem_handle(&buffer_ptrs[i], &ipc_handles[i]);
                barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
            } else {
                EP_HOST_ASSERT(std::memcmp(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE) == 0);
            }
        }

        // Copy all buffer and barrier signal pointers to GPU
        CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Sync NVSHMEM handles and allocate memory
#ifndef DISABLE_NVSHMEM
    if (num_rdma_bytes > 0) {
        // Initialize NVSHMEM
        EP_HOST_ASSERT(root_unique_id_opt.has_value());
        std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
        auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
        std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(), root_unique_id_opt->size());
        auto nvshmem_rank = low_latency_mode ? rank : rdma_rank;
        auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
        EP_HOST_ASSERT(nvshmem_rank == internode::init(root_unique_id, nvshmem_rank, num_nvshmem_ranks, low_latency_mode));
        internode::barrier();

        // Allocate
        rdma_buffer_ptr = internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);

        // Clean buffer (mainly for low-latency mode)
        CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

        // Allocate and clean shrink buffer
        if (enable_shrink) {
            int num_mask_buffer_bytes = num_ranks * sizeof(int);
            int num_sync_buffer_bytes = num_ranks * sizeof(int);
            mask_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_mask_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            sync_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_sync_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            CUDA_CHECK(cudaMemset(mask_buffer_ptr, 0, num_mask_buffer_bytes));
            CUDA_CHECK(cudaMemset(sync_buffer_ptr, 0, num_sync_buffer_bytes));
        }

        // Barrier
        internode::barrier();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif

    // Ready to use
    available = true;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor>
Buffer::get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    // All kernels run on the caller's stream. Allocations naturally land here
    // because PyTorch's caching allocator uses `getCurrentCUDAStream`.
    auto stream = at::cuda::getCurrentCUDAStream();

    auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kCUDA));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA));
    if (is_internode_available())
        num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    layout::get_dispatch_layout(topk_idx.data_ptr<topk_idx_t>(),
                                num_tokens_per_rank.data_ptr<int>(),
                                num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,
                                num_tokens_per_expert.data_ptr<int>(),
                                is_token_in_rank.data_ptr<bool>(),
                                num_tokens,
                                num_topk,
                                num_ranks,
                                num_experts,
                                stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank};
}

StreamingDispatchOutputs Buffer::intranode_dispatch(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& x_scales,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& is_token_in_rank,
    int num_experts,
    int expert_alignment,
    int tile_m,
    int64_t dispatch_seq,
    const Config& config) {
    // One channel uses two blocks (sender + receiver).
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    int num_local_experts = num_experts / num_ranks;
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_local_experts <= NUM_MAX_LOCAL_EXPERTS);
    EP_HOST_ASSERT(tile_m > 0);
    EP_HOST_ASSERT(expert_alignment > 0);

    // Type & shape checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == x.size(0) and topk_weights.size(0) == x.size(0));
    EP_HOST_ASSERT(topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    int num_topk = static_cast<int>(topk_idx.size(1));

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // All kernels + allocations run on the caller's current stream. The caller
    // is expected to have set its `dispatch_stream` as current via
    // `with torch.cuda.stream(dispatch_stream)` before calling. PyTorch's caching
    // allocator uses `getCurrentCUDAStream`, so allocations land on this same
    // stream and are naturally ordered with the kernels we launch.
    auto stream = at::cuda::getCurrentCUDAStream();

    // Reset host-mapped sync slots before metadata writes.
    *moe_recv_counter = -1;
    for (int i = 0; i < num_local_experts; ++i)
        moe_recv_expert_counter[i] = -1;
    *streaming_total_tiles = -1;

    // ── Pre-host-poll consolidated allocation. ──────────────────────────────
    //
    // The metadata kernel's outputs + the dispatch sender's per-(rank,channel)
    // prefix matrix all live on `dispatch_stream`. We bundle them into a single
    // int8 slab and view typed sub-tensors via `narrow` + `view(dtype)`.
    //
    // Why one slab + one memset instead of 8 separate `torch::empty/zeros`:
    // each per-tensor `torch::empty` costs ~5–10 µs of host-side aten-op +
    // caching-allocator overhead at production shape, and each `torch::zeros`
    // adds another `aten::zero_` + `cudaLaunchKernel` for the per-tensor
    // memset. Trace evidence (profiles/phase_d_reorder/) showed ~150 µs of
    // host-side work between the metadata kernel ending and dispatch_main
    // launching, dominated by ~12 individual aten ops; consolidating into
    // one alloc + one memset reclaims the bulk of that latency.
    //
    // Per-tensor init requirements (audit):
    //   rank_prefix_matrix       atomicAdd target in metadata phase 4–5 → MUST be 0.
    //   {channel,base,...}_*     written by metadata kernel; init value irrelevant.
    //   tile_id_to_expert        written by phase 8 at indices < total_tiles; tail unused.
    //   pool_arrival_target      written by phase 8 at indices < total_tiles; tail unused.
    // Single bundle-wide memset(0) is correctness-preserving and simplest.
    //
    // `total_tiles_max` is the upper-bound tile count (every (token, k) at
    // every sender contributes at most one slot, plus up to E_local extra
    // tiles from per-expert ceil-padding). Production: 8192×4×8/128 + 8 =
    // 2056 ints (~8 KB per array). The dispatch return slices these down to
    // the polled `total_tiles` so consumers see only the valid prefix.
    auto i32_opts = dtype(torch::kInt32).device(torch::kCUDA);
    auto i8_opts = dtype(torch::kInt8).device(torch::kCUDA);

    int64_t total_tiles_max = static_cast<int64_t>(num_tokens) * num_topk * num_ranks / tile_m + num_local_experts + 1;

    auto align16 = [](int64_t x) { return (x + 15) & ~static_cast<int64_t>(15); };
    int64_t pre_off_channel_prefix_matrix    = align16(0);
    int64_t pre_off_expert_frequency         = align16(pre_off_channel_prefix_matrix    + static_cast<int64_t>(num_ranks) * num_channels * sizeof(int));
    int64_t pre_off_expert_pool_block_offset = align16(pre_off_expert_frequency         + static_cast<int64_t>(num_local_experts) * sizeof(int));
    int64_t pre_off_base_pool                = align16(pre_off_expert_pool_block_offset + static_cast<int64_t>(num_local_experts + 1) * sizeof(int));
    int64_t pre_off_rank_prefix_matrix       = align16(pre_off_base_pool                + static_cast<int64_t>(num_channels) * num_ranks * num_local_experts * sizeof(int));
    int64_t pre_off_total_tiles_device       = align16(pre_off_rank_prefix_matrix       + static_cast<int64_t>(num_ranks) * num_ranks * sizeof(int));
    int64_t pre_off_tile_id_to_expert        = align16(pre_off_total_tiles_device       + static_cast<int64_t>(1) * sizeof(int));
    int64_t pre_off_pool_arrival_target      = align16(pre_off_tile_id_to_expert        + total_tiles_max * sizeof(int));
    int64_t pre_total_bytes                  = align16(pre_off_pool_arrival_target      + total_tiles_max * sizeof(int));

    auto pre_bundle = torch::empty({pre_total_bytes}, i8_opts);
    auto* pre_base = static_cast<int8_t*>(pre_bundle.data_ptr());
    CUDA_CHECK(cudaMemsetAsync(pre_base, 0, pre_total_bytes, stream));

    // Lambda capturing `pre_bundle` by value to keep storage alive for the
    // lifetime of any returned view. Each `at::from_blob` clones this lambda
    // (incrementing pre_bundle's refcount); on tensor free, the storage
    // deleter destructs the lambda, releasing the refcount. Bundle freed back
    // to the caching allocator once all views are dropped.
    auto pre_keep = [pre_bundle](void*) {};

    auto channel_prefix_matrix    = at::from_blob(pre_base + pre_off_channel_prefix_matrix,    {num_ranks, num_channels},                            pre_keep, i32_opts);
    auto expert_frequency         = at::from_blob(pre_base + pre_off_expert_frequency,         {num_local_experts},                                  pre_keep, i32_opts);
    auto expert_pool_block_offset = at::from_blob(pre_base + pre_off_expert_pool_block_offset, {num_local_experts + 1},                              pre_keep, i32_opts);
    auto base_pool                = at::from_blob(pre_base + pre_off_base_pool,                {num_channels, num_ranks, num_local_experts},         pre_keep, i32_opts);
    auto rank_prefix_matrix       = at::from_blob(pre_base + pre_off_rank_prefix_matrix,       {num_ranks, num_ranks},                               pre_keep, i32_opts);
    auto total_tiles_device       = at::from_blob(pre_base + pre_off_total_tiles_device,       {1},                                                  pre_keep, i32_opts);
    auto tile_id_to_expert        = at::from_blob(pre_base + pre_off_tile_id_to_expert,        {total_tiles_max},                                    pre_keep, i32_opts);
    auto pool_arrival_target      = at::from_blob(pre_base + pre_off_pool_arrival_target,      {total_tiles_max},                                    pre_keep, i32_opts);

    intranode::streaming_dispatch_metadata(topk_idx.data_ptr<topk_idx_t>(),
                                           expert_frequency.data_ptr<int>(),
                                           expert_pool_block_offset.data_ptr<int>(),
                                           base_pool.data_ptr<int>(),
                                           rank_prefix_matrix.data_ptr<int>(),
                                           tile_id_to_expert.data_ptr<int>(),
                                           pool_arrival_target.data_ptr<int>(),
                                           total_tiles_device.data_ptr<int>(),
                                           moe_recv_counter_mapped,
                                           moe_recv_expert_counter_mapped,
                                           streaming_total_tiles_mapped,
                                           num_tokens,
                                           num_topk,
                                           num_local_experts,
                                           num_channels,
                                           streaming_section_offset,
                                           buffer_ptrs_gpu,
                                           barrier_signal_ptrs_gpu,
                                           rank,
                                           num_ranks,
                                           tile_m,
                                           expert_alignment,
                                           stream);

    // The dispatch main kernel reads rank_prefix_matrix from buffer_ptrs[rank]
    // (the IPC slab, offset 0) and the channel queue metadata immediately after.
    // The queue metadata (start_offset, end_offset, head_idx, tail_idx —
    // 4 × num_channels × num_ranks ints) must be zeroed each dispatch so the
    // receiver's spin on `channel_start_offset != 0` doesn't latch onto a stale
    // value from a prior dispatch.
    int num_memset_int = num_channels * num_ranks * 4;
    EP_HOST_ASSERT((num_ranks * num_ranks + num_memset_int) * sizeof(int) <= num_nvl_bytes);
    CUDA_CHECK(cudaMemcpyAsync(buffer_ptrs[nvl_rank],
                               rank_prefix_matrix.data_ptr<int>(),
                               num_ranks * num_ranks * sizeof(int),
                               cudaMemcpyDeviceToDevice,
                               stream));
    CUDA_CHECK(cudaMemsetAsync(static_cast<int*>(buffer_ptrs[nvl_rank]) + num_ranks * num_ranks,
                               0,
                               num_memset_int * sizeof(int),
                               stream));
    intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_ranks, stream);

    // Single host poll: {num_recv, num_recv_per_expert[E_local], total_tiles}.
    int num_recv_tokens = -1;
    int total_tiles = -1;
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            num_recv_tokens = static_cast<int>(*moe_recv_counter);
            total_tiles = static_cast<int>(*streaming_total_tiles);
            bool ready = (num_recv_tokens >= 0) and (total_tiles >= 0);
            for (int i = 0; i < num_local_experts and ready; ++i)
                ready &= moe_recv_expert_counter[i] >= 0;
            if (ready)
                break;
            if (std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - start_time).count() > NUM_CPU_TIMEOUT_SECS)
                throw std::runtime_error("DeepEP error: CPU recv timeout");
        }
    }
    auto num_recv_tokens_per_expert_list =
        std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
    int64_t TK_padded = static_cast<int64_t>(total_tiles) * tile_m;

    // ── Post-host-poll consolidated allocation. ─────────────────────────────
    //
    // All small/medium per-recv-token + per-tile outputs go into ONE int8
    // slab with three `cudaMemsetAsync` calls (different offsets, different
    // fill bytes, with `metadata_done` recorded between them). Bundling
    // saves ~80 µs of host aten-op overhead vs the previous 11 separate
    // torch::zeros/empty calls.
    //
    // The `metadata_done` event placement matters for kernel A streaming
    // overlap: consumer streams wait on metadata_done before reading any
    // metadata tensors, but only some of the post-poll buffers are needed
    // by kernel A's start (it spins on tile_ready). So we split the Z region
    // into two halves around the event:
    //
    //   Z_pre (zeroed BEFORE metadata_done — kernel A / kernel Y read these
    //          via the dispatch_main → tile_ready → kernel_A chain):
    //     pool_topk_weight, recv_src_idx, recv_topk_weights,
    //     recv_channel_prefix_matrix, send_head, pool_arrival_count,
    //     tile_ready, a_ready
    //
    //   N (0xFF = -1, BEFORE metadata_done — kernel Y reads via predicate):
    //     pool_recv_token, pool_k_slot
    //
    //   Z_post (zeroed AFTER metadata_done — kernel Y atomic-scatter
    //          destinations + combine sender state. Kernel Y waits on
    //          a_ready (kernel A's release), which serializes after this
    //          memset on dispatch_stream → no race):
    //     per_token_remaining, compute_done_per_token, o
    //
    // Kept separate (memory wall — too big to bundle without forcing a
    // different size class on PyTorch's caching allocator each iter):
    //   pool[TK_padded, hidden] (~290 MB at production; no init needed —
    //     kernel Y's predicate skips padding slots).
    //   pool_x_scales (FP8 path; conditional).
    auto pool = torch::empty({TK_padded, hidden}, x.options());

    // Pool-shape x_scales (FP8 path; otherwise unused). Kept separate from
    // the post-poll bundle since it's conditional and FP8-specific.
    auto pool_x_scales = std::optional<torch::Tensor>();
    float* pool_x_scales_ptr = nullptr;
    if (x_scales.has_value()) {
        pool_x_scales = torch::zeros({TK_padded, num_scales}, x_scales->options());
        pool_x_scales_ptr = static_cast<float*>(pool_x_scales->data_ptr());
    }

    int64_t hidden_bytes_per_recv_token = static_cast<int64_t>(hidden) * x.element_size();

    // Z_pre region offsets (zeroed before metadata_done event).
    int64_t off_pool_topk_weight    = align16(0);
    int64_t off_recv_src_idx        = align16(off_pool_topk_weight    + TK_padded * sizeof(float));
    int64_t off_recv_topk_weights   = align16(off_recv_src_idx        + static_cast<int64_t>(num_recv_tokens) * sizeof(int));
    int64_t off_recv_channel_prefix = align16(off_recv_topk_weights   + static_cast<int64_t>(num_recv_tokens) * num_topk * sizeof(float));
    int64_t off_send_head           = align16(off_recv_channel_prefix + static_cast<int64_t>(num_ranks) * num_channels * sizeof(int));
    int64_t off_pool_arrival_count  = align16(off_send_head           + static_cast<int64_t>(num_tokens) * num_ranks * sizeof(int));
    int64_t off_tile_ready          = align16(off_pool_arrival_count  + static_cast<int64_t>(total_tiles) * sizeof(int));
    int64_t off_a_ready             = align16(off_tile_ready          + static_cast<int64_t>(total_tiles) * sizeof(int64_t));
    int64_t z_pre_bytes             = align16(off_a_ready             + static_cast<int64_t>(total_tiles) * sizeof(int64_t));

    // N region offsets (-1 fill, before metadata_done).
    int64_t off_pool_recv_token = align16(z_pre_bytes);
    int64_t off_pool_k_slot     = align16(off_pool_recv_token + TK_padded * sizeof(int));
    int64_t n_end               = align16(off_pool_k_slot     + TK_padded * sizeof(int));

    // Z_post region offsets (zeroed after metadata_done).
    int64_t off_per_token_remaining    = align16(n_end);
    int64_t off_compute_done_per_token = align16(off_per_token_remaining    + static_cast<int64_t>(num_recv_tokens) * sizeof(int));
    int64_t off_o                      = align16(off_compute_done_per_token + static_cast<int64_t>(num_recv_tokens) * sizeof(int64_t));
    int64_t post_total_bytes           = align16(off_o                      + static_cast<int64_t>(num_recv_tokens) * hidden_bytes_per_recv_token);

    auto post_bundle = torch::empty({post_total_bytes}, i8_opts);
    auto* post_base = static_cast<int8_t*>(post_bundle.data_ptr());

    // Z_pre + N memsets (queued BEFORE metadata_done event recording).
    CUDA_CHECK(cudaMemsetAsync(post_base,                0x00, z_pre_bytes,         stream));
    CUDA_CHECK(cudaMemsetAsync(post_base + z_pre_bytes,  0xFF, n_end - z_pre_bytes, stream));

    auto post_keep = [post_bundle](void*) {};
    auto i64_opts = dtype(torch::kInt64).device(torch::kCUDA);
    auto f32_opts = dtype(torch::kFloat32).device(torch::kCUDA);

    // Z_pre + N views (consumed by kernel A / kernel Y / combine via
    // tile_ready / a_ready / per-token-gate chains; serialized after the
    // memsets above via metadata_done event).
    auto pool_topk_weight           = at::from_blob(post_base + off_pool_topk_weight,    {TK_padded},                          post_keep, f32_opts);
    auto recv_src_idx               = at::from_blob(post_base + off_recv_src_idx,        {num_recv_tokens},                    post_keep, i32_opts);
    auto recv_topk_weights          = at::from_blob(post_base + off_recv_topk_weights,   {num_recv_tokens, num_topk},          post_keep, topk_weights.options());
    auto recv_channel_prefix_matrix = at::from_blob(post_base + off_recv_channel_prefix, {num_ranks, num_channels},            post_keep, i32_opts);
    auto send_head                  = at::from_blob(post_base + off_send_head,           {num_tokens, num_ranks},              post_keep, i32_opts);
    auto pool_arrival_count         = at::from_blob(post_base + off_pool_arrival_count,  {total_tiles},                        post_keep, i32_opts);
    auto tile_ready                 = at::from_blob(post_base + off_tile_ready,          {total_tiles},                        post_keep, i64_opts);
    auto a_ready                    = at::from_blob(post_base + off_a_ready,             {total_tiles},                        post_keep, i64_opts);
    auto pool_recv_token            = at::from_blob(post_base + off_pool_recv_token,     {TK_padded},                          post_keep, i32_opts);
    auto pool_k_slot                = at::from_blob(post_base + off_pool_k_slot,         {TK_padded},                          post_keep, i32_opts);

    // Narrow the per-tile arrays from total_tiles_max → total_tiles for the
    // returned views (the metadata kernel wrote indices < total_tiles). One
    // narrow per tensor here (vs replacing with from_blob): the source
    // tensor (already from_blob'd into pre_bundle) keeps its storage alive
    // and `narrow` on a from_blob'd tensor returns another view sharing the
    // same storage — net cost is one extra `aten::narrow` event per tensor,
    // unavoidable since the visible size is only known post-poll.
    tile_id_to_expert   = tile_id_to_expert.narrow(0, 0, total_tiles);
    pool_arrival_target = pool_arrival_target.narrow(0, 0, total_tiles);

    // Record the metadata-done event between the metadata kernel + Z_pre/N
    // memsets and the dispatch main kernel. Consumer streams (kernel A,
    // kernel Y, combine sender) wait on this event to safely read metadata
    // tensors (expert_pool_block_offset, pool_recv_token, pool_topk_weight,
    // etc.) without serializing against the dispatch main kernel —
    // preserving the per-tile dispatch→A streaming overlap.
    auto metadata_done_event = EventHandle(stream);

    // Z_post memset (queued AFTER metadata_done event recording — kernel A
    // doesn't depend on these tensors at start; kernel Y reaches them only
    // through the a_ready release-acquire chain which serializes after
    // dispatch_main and kernel A on dispatch_stream / compute_a_stream).
    CUDA_CHECK(cudaMemsetAsync(post_base + n_end, 0x00, post_total_bytes - n_end, stream));

    // Z_post views.
    auto per_token_remaining    = at::from_blob(post_base + off_per_token_remaining,    {num_recv_tokens},          post_keep, i32_opts);
    auto compute_done_per_token = at::from_blob(post_base + off_compute_done_per_token, {num_recv_tokens},          post_keep, i64_opts);
    auto o                      = at::from_blob(post_base + off_o,                      {num_recv_tokens, hidden},  post_keep, x.options());

    EP_HOST_ASSERT(
        num_ranks * num_ranks * sizeof(int) +
            num_channels * num_ranks * sizeof(int) +
            num_channels * num_ranks * sizeof(int) +
            num_channels * num_ranks * sizeof(int) * 2 +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * pool.element_size() +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(topk_idx_t) +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales
        <= num_nvl_bytes);

    intranode::launch_dispatch_main(pool.data_ptr(),
                        pool_x_scales_ptr,
                        pool_topk_weight.data_ptr<float>(),
                        pool_recv_token.data_ptr<int>(),
                        pool_k_slot.data_ptr<int>(),
                        recv_src_idx.data_ptr<int>(),
                        recv_topk_weights.data_ptr<float>(),
                        recv_channel_prefix_matrix.data_ptr<int>(),
                        send_head.data_ptr<int>(),
                        per_token_remaining.data_ptr<int>(),
                        x.data_ptr(),
                        x_scales_ptr,
                        topk_idx.data_ptr<topk_idx_t>(),
                        topk_weights.data_ptr<float>(),
                        is_token_in_rank.data_ptr<bool>(),
                        channel_prefix_matrix.data_ptr<int>(),
                        base_pool.data_ptr<int>(),
                        pool_arrival_count.data_ptr<int>(),
                        pool_arrival_target.data_ptr<int>(),
                        tile_ready.data_ptr<int64_t>(),
                        dispatch_seq,
                        num_tokens,
                        static_cast<int>(hidden * pool.element_size() / sizeof(int4)),
                        num_topk,
                        num_experts,
                        num_scales,
                        scale_token_stride,
                        scale_hidden_stride,
                        buffer_ptrs_gpu,
                        rank,
                        num_ranks,
                        stream,
                        config.num_sms,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens,
                        tile_m);

    return StreamingDispatchOutputs{
        .pool                       = pool,
        .pool_x_scales              = pool_x_scales,
        .pool_topk_weight           = pool_topk_weight,
        .pool_recv_token            = pool_recv_token,
        .pool_k_slot                = pool_k_slot,
        .recv_topk_weights          = recv_topk_weights,
        .recv_src_idx               = recv_src_idx,
        .send_head                  = send_head,
        .num_recv_tokens_per_expert = num_recv_tokens_per_expert_list,
        .rank_prefix_matrix         = rank_prefix_matrix,
        .channel_prefix_matrix      = channel_prefix_matrix,
        .recv_channel_prefix_matrix = recv_channel_prefix_matrix,
        .expert_frequency           = expert_frequency,
        .expert_pool_block_offset   = expert_pool_block_offset,
        .base_pool                  = base_pool,
        .tile_id_to_expert          = tile_id_to_expert,
        .pool_arrival_target        = pool_arrival_target,
        .tile_ready                 = tile_ready,
        .a_ready                    = a_ready,
        .per_token_remaining        = per_token_remaining,
        .compute_done_per_token     = compute_done_per_token,
        .o                          = o,
        .total_tiles                = total_tiles,
        .metadata_done_event        = metadata_done_event,
    };
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> Buffer::intranode_combine(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& topk_weights,
    const std::optional<torch::Tensor>& bias_0,
    const std::optional<torch::Tensor>& bias_1,
    const torch::Tensor& src_idx,
    const torch::Tensor& rank_prefix_matrix,
    const torch::Tensor& channel_prefix_matrix,
    const torch::Tensor& send_head,
    const torch::Tensor& compute_done_per_token,
    int64_t combine_seq,
    const Config& config) {
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and
                   rank_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and
                   channel_prefix_matrix.scalar_type() == torch::kInt32);
    // Phase-D per-token gate: kernel Y release-stores `combine_seq` into
    // compute_done_per_token[r] once per_token_remaining[r] hits zero. The
    // combine sender's per-warp send loop spins on the same address before its
    // data copy of o[r], pairing with kernel Y's `.sys`-scope release.
    EP_HOST_ASSERT(compute_done_per_token.dim() == 1 and compute_done_per_token.is_contiguous() and
                   compute_done_per_token.scalar_type() == torch::kInt64);

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));
    EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
    // compute_done_per_token is indexed by combine sender's iteration variable
    // `token_idx` ∈ [0, num_tokens), which (in combine's naming convention)
    // is the number of input rows of `x` = handle.o.size(0) = T_recv from
    // dispatch's perspective. Note combine's `num_recv_tokens` is dispatch's
    // source-token count (output rows of combine), NOT the size of the gate.
    EP_HOST_ASSERT(compute_done_per_token.size(0) == num_tokens);
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

    // All kernels + allocations run on the caller's current stream.
    auto stream = at::cuda::getCurrentCUDAStream();

    int num_topk = 0;
    auto recv_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }

    // Launch barrier and reset queue head and tail
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
    intranode::cached_notify_combine(buffer_ptrs_gpu,
                                     send_head.data_ptr<int>(),
                                     num_channels,
                                     num_recv_tokens,
                                     num_channels * num_ranks * 2,
                                     barrier_signal_ptrs_gpu,
                                     rank,
                                     num_ranks,
                                     stream);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            EP_HOST_ASSERT(bias.size(0) == num_recv_tokens and bias.size(1) == hidden);
            bias_ptrs[i] = bias.data_ptr();
        }

    // Combine data
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * x.element_size() +  // Data buffer
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +             // Source index buffer
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float)  // Top-k weight buffer
                   <= num_nvl_bytes);
    intranode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                       recv_x.data_ptr(),
                       recv_topk_weights_ptr,
                       x.data_ptr(),
                       topk_weights_ptr,
                       bias_ptrs[0],
                       bias_ptrs[1],
                       src_idx.data_ptr<int>(),
                       rank_prefix_matrix.data_ptr<int>(),
                       channel_prefix_matrix.data_ptr<int>(),
                       send_head.data_ptr<int>(),
                       compute_done_per_token.data_ptr<int64_t>(),
                       combine_seq,
                       num_tokens,
                       num_recv_tokens,
                       hidden,
                       num_topk,
                       buffer_ptrs_gpu,
                       rank,
                       num_ranks,
                       stream,
                       config.num_sms,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens);

    return {recv_x, recv_topk_weights};
}

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
           std::optional<torch::Tensor>>
Buffer::internode_dispatch(const torch::Tensor& x,
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
                           const Config& config) {
#ifndef DISABLE_NVSHMEM
    // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
    // If users of DeepEP need to execute other Python code on other threads, such as KV transfer, their code will get stuck due to GIL
    // unless we release GIL here.
    pybind11::gil_scoped_release release;

    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and cached_rdma_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and
                       cached_rdma_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and
                       cached_gbl_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
         hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    topk_idx_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // All kernels + allocations run on the caller's current stream.
    auto stream = at::cuda::getCurrentCUDAStream();

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
    auto rdma_channel_prefix_matrix = torch::Tensor();
    auto recv_rdma_rank_prefix_sum = torch::Tensor();
    auto gbl_channel_prefix_matrix = torch::Tensor();
    auto recv_gbl_rank_prefix_sum = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
        rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
        recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
        gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
        recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

        // Just a barrier and clean flags
        internode::cached_notify(hidden_int4,
                                 num_scales,
                                 num_topk,
                                 num_topk,
                                 num_ranks,
                                 num_channels,
                                 0,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 rdma_buffer_ptr,
                                 config.num_max_rdma_chunked_recv_tokens,
                                 buffer_ptrs_gpu,
                                 config.num_max_nvl_chunked_recv_tokens,
                                 barrier_signal_ptrs_gpu,
                                 rank,
                                 stream,
                                 config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                 num_nvl_bytes,
                                 true,
                                 low_latency_mode);
    } else {
        rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

        // Send sizes
        *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        internode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                   moe_recv_counter_mapped,
                                   num_ranks,
                                   num_tokens_per_rdma_rank->data_ptr<int>(),
                                   moe_recv_rdma_counter_mapped,
                                   num_tokens_per_expert->data_ptr<int>(),
                                   moe_recv_expert_counter_mapped,
                                   num_experts,
                                   is_token_in_rank.data_ptr<bool>(),
                                   num_tokens,
                                   num_worst_tokens,
                                   num_channels,
                                   hidden_int4,
                                   num_scales,
                                   num_topk,
                                   expert_alignment,
                                   rdma_channel_prefix_matrix.data_ptr<int>(),
                                   recv_rdma_rank_prefix_sum.data_ptr<int>(),
                                   gbl_channel_prefix_matrix.data_ptr<int>(),
                                   recv_gbl_rank_prefix_sum.data_ptr<int>(),
                                   rdma_buffer_ptr,
                                   config.num_max_rdma_chunked_recv_tokens,
                                   buffer_ptrs_gpu,
                                   config.num_max_nvl_chunked_recv_tokens,
                                   barrier_signal_ptrs_gpu,
                                   rank,
                                   stream,
                                   config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                   num_nvl_bytes,
                                   low_latency_mode);

        // Synchronize total received tokens and tokens per expert
        if (num_worst_tokens > 0) {
            num_recv_tokens = num_worst_tokens;
            num_rdma_recv_tokens = num_worst_tokens;
        } else {
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);
                num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                    NUM_CPU_TIMEOUT_SECS) {
                    printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank, num_recv_tokens, num_rdma_recv_tokens);
                    for (int i = 0; i < num_local_experts; ++i)
                        printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                    throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
                }
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_src_meta = std::optional<torch::Tensor>();
    auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto send_rdma_head = std::optional<torch::Tensor>();
    auto send_nvl_head = std::optional<torch::Tensor>();
    if (not cached_mode) {
        recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kCUDA));
        recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(torch::kCUDA));
    }

    // Assign pointers
    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                             : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Launch data dispatch
    // NOTES: the buffer size checks are moved into the `.cu` file
    internode::dispatch(recv_x.data_ptr(),
                        recv_x_scales_ptr,
                        recv_topk_idx_ptr,
                        recv_topk_weights_ptr,
                        cached_mode ? nullptr : recv_src_meta->data_ptr(),
                        x.data_ptr(),
                        x_scales_ptr,
                        topk_idx_ptr,
                        topk_weights_ptr,
                        cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),
                        cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
                        cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
                        cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
                        rdma_channel_prefix_matrix.data_ptr<int>(),
                        recv_rdma_rank_prefix_sum.data_ptr<int>(),
                        gbl_channel_prefix_matrix.data_ptr<int>(),
                        recv_gbl_rank_prefix_sum.data_ptr<int>(),
                        is_token_in_rank.data_ptr<bool>(),
                        num_tokens,
                        num_worst_tokens,
                        hidden_int4,
                        num_scales,
                        num_topk,
                        num_experts,
                        scale_token_stride,
                        scale_hidden_stride,
                        rdma_buffer_ptr,
                        config.num_max_rdma_chunked_send_tokens,
                        config.num_max_rdma_chunked_recv_tokens,
                        buffer_ptrs_gpu,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens,
                        rank,
                        num_ranks,
                        cached_mode,
                        stream,
                        num_channels,
                        low_latency_mode);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> Buffer::internode_combine(
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
    const Config& config) {
#ifndef DISABLE_NVSHMEM
    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);
    EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and
                   is_combined_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and
                   rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and
                   rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and
                   gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and
                   combined_rdma_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and combined_nvl_head.scalar_type() == torch::kInt32);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
         hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());
    EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and rdma_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and
                   combined_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

    // All kernels + allocations run on the caller's current stream.
    auto stream = at::cuda::getCurrentCUDAStream();

    // Top-k checks
    int num_topk = 0;
    auto combined_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
        combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
    }

    // Extra check for avoid-dead-lock design
    EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

    // Launch barrier and reset queue head and tail
    internode::cached_notify(hidden_int4,
                             0,
                             0,
                             num_topk,
                             num_ranks,
                             num_channels,
                             num_combined_tokens,
                             combined_rdma_head.data_ptr<int>(),
                             rdma_channel_prefix_matrix.data_ptr<int>(),
                             rdma_rank_prefix_sum.data_ptr<int>(),
                             combined_nvl_head.data_ptr<int>(),
                             rdma_buffer_ptr,
                             config.num_max_rdma_chunked_recv_tokens,
                             buffer_ptrs_gpu,
                             config.num_max_nvl_chunked_recv_tokens,
                             barrier_signal_ptrs_gpu,
                             rank,
                             stream,
                             config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                             num_nvl_bytes,
                             false,
                             low_latency_mode);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
            bias_ptrs[i] = bias.data_ptr();
        }

    // Launch data combine
    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    internode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                       combined_x.data_ptr(),
                       combined_topk_weights_ptr,
                       is_combined_token_in_rank.data_ptr<bool>(),
                       x.data_ptr(),
                       topk_weights_ptr,
                       bias_ptrs[0],
                       bias_ptrs[1],
                       combined_rdma_head.data_ptr<int>(),
                       combined_nvl_head.data_ptr<int>(),
                       src_meta.data_ptr(),
                       rdma_channel_prefix_matrix.data_ptr<int>(),
                       rdma_rank_prefix_sum.data_ptr<int>(),
                       gbl_channel_prefix_matrix.data_ptr<int>(),
                       num_tokens,
                       num_combined_tokens,
                       hidden,
                       num_topk,
                       rdma_buffer_ptr,
                       config.num_max_rdma_chunked_send_tokens,
                       config.num_max_rdma_chunked_recv_tokens,
                       buffer_ptrs_gpu,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens,
                       rank,
                       num_ranks,
                       stream,
                       num_channels,
                       low_latency_mode);

    // Return values
    return {combined_x, combined_topk_weights};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    auto clean_meta_0 = layout.buffers[0].clean_meta();
    auto clean_meta_1 = layout.buffers[1].clean_meta();

    auto check_boundary = [=](void* ptr, size_t num_bytes) {
        auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
        EP_HOST_ASSERT(0 <= offset and offset + num_bytes <= num_rdma_bytes);
    };
    check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
    check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

    internode_ll::clean_low_latency_buffer(clean_meta_0.first,
                                           clean_meta_0.second,
                                           clean_meta_1.first,
                                           clean_meta_1.second,
                                           rank,
                                           num_ranks,
                                           mask_buffer_ptr,
                                           sync_buffer_ptr,
                                           at::cuda::getCurrentCUDAStream());
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const torch::Tensor& x,
                             const torch::Tensor& topk_idx,
                             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                             const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                             int num_max_dispatch_tokens_per_rank,
                             int num_experts,
                             bool use_fp8,
                             bool round_scale,
                             bool use_ue8m0,
                             bool return_recv_hook) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    // By default using `ptp128c` FP8 cast
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);

    // Diagnosis tensors
    if (cumulative_local_expert_recv_stats.has_value()) {
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts / num_ranks);
    }
    if (dispatch_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_topk = static_cast<int>(topk_idx.size(1));
    auto num_local_experts = num_experts / num_ranks;

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // All kernels + allocations run on the caller's current stream.
    auto launch_stream = at::cuda::getCurrentCUDAStream();

    // Allocate packed tensors
    auto packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                      x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
    auto packed_recv_src_info =
        torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto packed_recv_count = torch::empty({num_local_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // Allocate column-majored scales
    auto packed_recv_x_scales = std::optional<torch::Tensor>();
    void* packed_recv_x_scales_ptr = nullptr;
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

    if (use_fp8) {
        // TODO: support unaligned cases
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (not use_ue8m0) {
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 128, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
        } else {
            EP_HOST_ASSERT(round_scale);
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 512, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kInt).device(torch::kCUDA));
        }
        packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::dispatch(
            packed_recv_x.data_ptr(),
            packed_recv_x_scales_ptr,
            packed_recv_src_info.data_ptr<int>(),
            packed_recv_layout_range.data_ptr<int64_t>(),
            packed_recv_count.data_ptr<int>(),
            mask_buffer_ptr,
            cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
            dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
            buffer.dispatch_rdma_recv_data_buffer,
            buffer.dispatch_rdma_recv_count_buffer,
            buffer.dispatch_rdma_send_buffer,
            x.data_ptr(),
            topk_idx.data_ptr<topk_idx_t>(),
            next_clean_meta.first,
            next_clean_meta.second,
            num_tokens,
            hidden,
            num_max_dispatch_tokens_per_rank,
            num_topk,
            num_experts,
            rank,
            num_ranks,
            use_fp8,
            round_scale,
            use_ue8m0,
            workspace,
            num_device_sms,
            launch_stream,
            phases);
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, recv_hook};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, std::optional<std::function<void()>>> Buffer::low_latency_combine(
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
    bool return_recv_hook,
    const std::optional<torch::Tensor>& out) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
    EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and layout_range.size(1) == num_ranks);

    if (combine_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and combine_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto hidden = static_cast<int>(x.size(2));
    auto num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // All kernels + allocations run on the caller's current stream.
    auto launch_stream = at::cuda::getCurrentCUDAStream();

    // Allocate output tensor
    torch::Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        combined_x = out.value();
    } else {
        combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::combine(combined_x.data_ptr(),
                              buffer.combine_rdma_recv_data_buffer,
                              buffer.combine_rdma_recv_flag_buffer,
                              buffer.combine_rdma_send_buffer,
                              x.data_ptr(),
                              topk_idx.data_ptr<topk_idx_t>(),
                              topk_weights.data_ptr<float>(),
                              src_info.data_ptr<int>(),
                              layout_range.data_ptr<int64_t>(),
                              mask_buffer_ptr,
                              combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                              next_clean_meta.first,
                              next_clean_meta.second,
                              num_combined_tokens,
                              hidden,
                              num_max_dispatch_tokens_per_rank,
                              num_topk,
                              num_experts,
                              rank,
                              num_ranks,
                              use_logfmt,
                              workspace,
                              num_device_sms,
                              launch_stream,
                              phases,
                              zero_copy);
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {combined_x, recv_hook};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

torch::Tensor Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
#ifndef DISABLE_NVSHMEM
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);

    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                            {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                            {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                            torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

void Buffer::low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < num_ranks);
    internode_ll::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, at::cuda::getCurrentCUDAStream());
}

void Buffer::low_latency_query_mask_buffer(const torch::Tensor& mask_status) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == num_ranks && mask_status.scalar_type() == torch::kInt32);

    internode_ll::query_mask_buffer(
        mask_buffer_ptr, num_ranks, reinterpret_cast<int*>(mask_status.data_ptr()), at::cuda::getCurrentCUDAStream());
}

void Buffer::low_latency_clean_mask_buffer() {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    internode_ll::clean_mask_buffer(mask_buffer_ptr, num_ranks, at::cuda::getCurrentCUDAStream());
}

}  // namespace deep_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    pybind11::class_<deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6,
             py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6,
             py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait)
        .def("wait", &deep_ep::EventHandle::wait);

    pybind11::class_<deep_ep::StreamingDispatchOutputs>(m, "StreamingDispatchOutputs")
        .def_readonly("pool",                       &deep_ep::StreamingDispatchOutputs::pool)
        .def_readonly("pool_x_scales",              &deep_ep::StreamingDispatchOutputs::pool_x_scales)
        .def_readonly("pool_topk_weight",           &deep_ep::StreamingDispatchOutputs::pool_topk_weight)
        .def_readonly("pool_recv_token",            &deep_ep::StreamingDispatchOutputs::pool_recv_token)
        .def_readonly("pool_k_slot",                &deep_ep::StreamingDispatchOutputs::pool_k_slot)
        .def_readonly("recv_topk_weights",          &deep_ep::StreamingDispatchOutputs::recv_topk_weights)
        .def_readonly("recv_src_idx",               &deep_ep::StreamingDispatchOutputs::recv_src_idx)
        .def_readonly("send_head",                  &deep_ep::StreamingDispatchOutputs::send_head)
        .def_readonly("num_recv_tokens_per_expert", &deep_ep::StreamingDispatchOutputs::num_recv_tokens_per_expert)
        .def_readonly("rank_prefix_matrix",         &deep_ep::StreamingDispatchOutputs::rank_prefix_matrix)
        .def_readonly("channel_prefix_matrix",      &deep_ep::StreamingDispatchOutputs::channel_prefix_matrix)
        .def_readonly("recv_channel_prefix_matrix", &deep_ep::StreamingDispatchOutputs::recv_channel_prefix_matrix)
        .def_readonly("expert_frequency",           &deep_ep::StreamingDispatchOutputs::expert_frequency)
        .def_readonly("expert_pool_block_offset",   &deep_ep::StreamingDispatchOutputs::expert_pool_block_offset)
        .def_readonly("base_pool",                  &deep_ep::StreamingDispatchOutputs::base_pool)
        .def_readonly("tile_id_to_expert",          &deep_ep::StreamingDispatchOutputs::tile_id_to_expert)
        .def_readonly("pool_arrival_target",        &deep_ep::StreamingDispatchOutputs::pool_arrival_target)
        .def_readonly("tile_ready",                 &deep_ep::StreamingDispatchOutputs::tile_ready)
        .def_readonly("a_ready",                    &deep_ep::StreamingDispatchOutputs::a_ready)
        .def_readonly("per_token_remaining",        &deep_ep::StreamingDispatchOutputs::per_token_remaining)
        .def_readonly("compute_done_per_token",     &deep_ep::StreamingDispatchOutputs::compute_done_per_token)
        .def_readonly("o",                          &deep_ep::StreamingDispatchOutputs::o)
        .def_readonly("total_tiles",                &deep_ep::StreamingDispatchOutputs::total_tiles)
        .def_readonly("metadata_done_event",        &deep_ep::StreamingDispatchOutputs::metadata_done_event);

    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_nvshmem_unique_id", &deep_ep::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("sync", &deep_ep::Buffer::sync)
        .def("destroy", &deep_ep::Buffer::destroy)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine)
        .def("low_latency_update_mask_buffer", &deep_ep::Buffer::low_latency_update_mask_buffer)
        .def("low_latency_query_mask_buffer", &deep_ep::Buffer::low_latency_query_mask_buffer)
        .def("low_latency_clean_mask_buffer", &deep_ep::Buffer::low_latency_clean_mask_buffer)
        .def("get_next_low_latency_combine_buffer", &deep_ep::Buffer::get_next_low_latency_combine_buffer);

    m.def("is_sm90_compiled", deep_ep::is_sm90_compiled);
    m.attr("topk_idx_t") =
        py::reinterpret_borrow<py::object>((PyObject*)torch::getTHPDtype(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value));
}
