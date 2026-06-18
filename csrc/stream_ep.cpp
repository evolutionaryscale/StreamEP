#include "stream_ep.hpp"

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

namespace stream_ep {

Buffer::Buffer(int rank,
               int num_ranks,
               int64_t num_nvl_bytes,
               int64_t num_rdma_bytes,
               bool explicitly_destroy,
               bool enable_shrink,
               bool use_fabric)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      enable_shrink(enable_shrink),
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
                   num_rdma_bytes <= std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_nvl_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks and num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS);
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS);

    // Get ranks
    CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
#ifdef DISABLE_NVSHMEM
    EP_HOST_ASSERT(num_rdma_ranks == 1 and "NVSHMEM is disabled during compilation");
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
        // Once-only NVL slab zero-init. Iter-0 readers of the genstamped
        // meta slots see packed (seq=0, value=0); seq mismatches the
        // first kernel's `nvl_seq` (which starts at >= 1 via
        // Buffer._next_seq), so iter-0 spins until the first sender's
        // write lands. The genstamp protocol means subsequent iters need
        // no per-iter memset + barrier — stale residue is rejected by
        // seq-mismatch on read.
        CUDA_CHECK(cudaMemsetAsync(buffer_ptrs[nvl_rank], 0, num_nvl_bytes, current_stream));
        CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + streaming_section_offset, 0,
                                   streaming_section_bytes, current_stream));
    }

    // Create 32 MiB workspace
    CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, at::cuda::getCurrentCUDAStream()));

    // "Kernel started" flags — single int32 each, zero-init. See stream_ep.hpp
    // for the cross-stream launch-gate rationale.
    CUDA_CHECK(cudaMalloc(&dispatch_main_started_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dispatch_grads_started_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&kernel_y_started_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&kernel_a_bwd_started_flag, sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(dispatch_main_started_flag, 0, sizeof(int),
                               at::cuda::getCurrentCUDAStream()));
    CUDA_CHECK(cudaMemsetAsync(dispatch_grads_started_flag, 0, sizeof(int),
                               at::cuda::getCurrentCUDAStream()));
    CUDA_CHECK(cudaMemsetAsync(kernel_y_started_flag, 0, sizeof(int),
                               at::cuda::getCurrentCUDAStream()));
    CUDA_CHECK(cudaMemsetAsync(kernel_a_bwd_started_flag, 0, sizeof(int),
                               at::cuda::getCurrentCUDAStream()));

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
        printf("WARNING: destroy() was not called before StreamEP buffer destruction, which can leak resources.\n");
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
        if (dispatch_reader_prev_head)     CUDA_CHECK(cudaFree(dispatch_reader_prev_head));
        if (dispatch_reader_prev_tail)     CUDA_CHECK(cudaFree(dispatch_reader_prev_tail));
        if (combine_reader_prev_head)      CUDA_CHECK(cudaFree(combine_reader_prev_head));
        if (combine_reader_prev_tail)      CUDA_CHECK(cudaFree(combine_reader_prev_tail));
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

    // Free cross-stream launch-gate flags
    CUDA_CHECK(cudaFree(dispatch_main_started_flag));
    CUDA_CHECK(cudaFree(dispatch_grads_started_flag));
    CUDA_CHECK(cudaFree(kernel_y_started_flag));
    CUDA_CHECK(cudaFree(kernel_a_bwd_started_flag));

    destroyed = true;
    available = false;
}

// cuStreamBatchMemOp-based wait: queues a GPU-front-end wait on `stream` until
// `*flag >= target`. The wait sits at the front-end — does NOT consume an SM.
// Used to gate kernel_a / kernel_y_bwd launches behind their respective
// dispatch_main / dispatch_grads_main kernels actually entering execution, so
// dispatch grabs SMs first under CDMC>1 + torch.compile.
static void wait_value_geq_on_stream(CUstream stream, int* flag, int target) {
    CUstreamBatchMemOpParams op{};
    op.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    op.waitValue.address = reinterpret_cast<CUdeviceptr>(flag);
    op.waitValue.value = static_cast<cuuint32_t>(target);
    op.waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
    CUresult err = cuStreamBatchMemOp(stream, 1, &op, 0);
    if (err != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(err, &msg);
        EP_HOST_ASSERT(false and msg);
    }
}

void Buffer::wait_dispatch_main_started(int64_t stream_handle) {
    if (dispatch_main_issued_count == 0) return;  // nothing launched yet
    wait_value_geq_on_stream(reinterpret_cast<CUstream>(stream_handle),
                             dispatch_main_started_flag,
                             dispatch_main_issued_count);
}

void Buffer::wait_dispatch_grads_started(int64_t stream_handle) {
    if (dispatch_grads_issued_count == 0) return;
    wait_value_geq_on_stream(reinterpret_cast<CUstream>(stream_handle),
                             dispatch_grads_started_flag,
                             dispatch_grads_issued_count);
}

void Buffer::wait_kernel_y_started(int64_t stream_handle) {
    if (kernel_y_issued_count == 0) return;
    wait_value_geq_on_stream(reinterpret_cast<CUstream>(stream_handle),
                             kernel_y_started_flag,
                             kernel_y_issued_count);
}

void Buffer::wait_kernel_a_bwd_started(int64_t stream_handle) {
    if (kernel_a_bwd_issued_count == 0) return;
    wait_value_geq_on_stream(reinterpret_cast<CUstream>(stream_handle),
                             kernel_a_bwd_started_flag,
                             kernel_a_bwd_issued_count);
}

void Buffer::bump_kernel_y_issued() {
    kernel_y_issued_count++;
}

void Buffer::bump_kernel_a_bwd_issued() {
    kernel_a_bwd_issued_count++;
}

torch::Tensor Buffer::kernel_y_started_flag_tensor() const {
    // Buffer-owned device memory; the deleter is a no-op (cudaFree happens
    // in destroy()). Returned as a 1-element int32 tensor view.
    auto opts = at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_id);
    return at::from_blob(kernel_y_started_flag, {1}, [](void*) {}, opts);
}

torch::Tensor Buffer::kernel_a_bwd_started_flag_tensor() const {
    auto opts = at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_id);
    return at::from_blob(kernel_a_bwd_started_flag, {1}, [](void*) {}, opts);
}

void Buffer::set_compute_stream_handle(int64_t stream_handle) {
    compute_stream_handle_ = stream_handle;
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
        auto nvshmem_rank = rdma_rank;
        auto num_nvshmem_ranks = num_rdma_ranks;
        EP_HOST_ASSERT(nvshmem_rank == internode::init(root_unique_id, nvshmem_rank, num_nvshmem_ranks));
        internode::barrier();

        // Allocate. Total = 2 * num_rdma_bytes; first half for dispatch-side
        // SymBuffers, second half for combine-side. See `stream_ep.hpp`.
        rdma_buffer_ptr = internode::alloc(num_rdma_bytes * 2, NUM_BUFFER_ALIGNMENT_BYTES);
        rdma_buffer_ptr_combine = static_cast<uint8_t*>(rdma_buffer_ptr) + num_rdma_bytes;

        // Clean buffer
        CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes * 2));

        // Persistent reader_prev arrays for the RDMA head/tail slots. See
        // `stream_ep.hpp` for layout / role. Sized for the worst case
        // (max_num_channels × num_rdma_ranks). Local memory (not symmetric);
        // each rank tracks its own reader's last-observed slot value (uint32
        // matching the NIC's 4-byte AMO; cross-iter wrap absorbed by
        // modular subtraction + signed-difference CAS).
        const int max_num_channels = num_device_sms / 2;
        const int64_t prev_array_bytes_32 =
            static_cast<int64_t>(max_num_channels) * num_rdma_ranks * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&dispatch_reader_prev_head,    prev_array_bytes_32));
        CUDA_CHECK(cudaMalloc(&dispatch_reader_prev_tail,    prev_array_bytes_32));
        CUDA_CHECK(cudaMalloc(&combine_reader_prev_head,     prev_array_bytes_32));
        CUDA_CHECK(cudaMalloc(&combine_reader_prev_tail,     prev_array_bytes_32));
        CUDA_CHECK(cudaMemset(dispatch_reader_prev_head,    0, prev_array_bytes_32));
        CUDA_CHECK(cudaMemset(dispatch_reader_prev_tail,    0, prev_array_bytes_32));
        CUDA_CHECK(cudaMemset(combine_reader_prev_head,     0, prev_array_bytes_32));
        CUDA_CHECK(cudaMemset(combine_reader_prev_tail,     0, prev_array_bytes_32));

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

// Helpers for `Buffer::intranode_dispatch`. Hidden from the public surface
// (anonymous namespace); see the corresponding section of intranode_dispatch
// where each is called for the rationale.
namespace {

constexpr inline int64_t align16(int64_t x) {
    return (x + 15) & ~static_cast<int64_t>(15);
}

// 16-byte-aligned offset accumulator for packing typed sub-tensors into a
// single int8 slab (see allocate_pre_poll_bundle / allocate_post_poll_bundle).
// One `reserve<T>(count)` per field; single source of truth for the layout,
// so reordering fields doesn't require re-threading offset arithmetic.
class SlabBuilder {
public:
    template <typename T>
    int64_t reserve(int64_t count) {
        return reserve_bytes(count * static_cast<int64_t>(sizeof(T)));
    }
    int64_t reserve_bytes(int64_t bytes) {
        int64_t off = align16(cursor_);
        cursor_ = align16(off + bytes);
        return off;
    }
    int64_t total_bytes() const { return cursor_; }
private:
    int64_t cursor_ = 0;
};

// Tell the caching allocator that the consumer stream identified by
// `consumer_stream_handle` (raw cudaStream_t cast to int64_t) also touches
// `t`'s storage. Without this, the allocator only tracks the allocation
// stream (= caller's current stream = communicate) and can recycle the
// storage while consumer-stream kernels (kernel_a / kernel_y, bwd path)
// are still reading or writing. No-op when handle is 0.
//
// Only meaningful for tensors whose storage is the caching allocator's
// (i.e. `torch::empty`/`torch::zeros`); views constructed via `at::from_blob`
// have a custom deleter and aren't allocator-tracked — recording on a view
// is silently a no-op.
inline void record_consumer_stream(const torch::Tensor& t, int64_t consumer_stream_handle) {
    if (consumer_stream_handle == 0) return;
    t.record_stream(at::cuda::getStreamFromExternal(
        reinterpret_cast<cudaStream_t>(consumer_stream_handle),
        t.device().index()));
}

void validate_dispatch_inputs(const torch::Tensor& x,
                              const torch::Tensor& topk_idx,
                              const torch::Tensor& topk_weights,
                              const torch::Tensor& is_token_in_rank,
                              int num_experts, int num_ranks, int num_local_experts,
                              int expert_alignment, int tile_m) {
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_local_experts <= NUM_MAX_LOCAL_EXPERTS);
    EP_HOST_ASSERT(tile_m > 0);
    EP_HOST_ASSERT(expert_alignment > 0);

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
}

// Pre-host-poll bundle: the metadata kernel's outputs + the dispatch sender's
// per-(rank,channel) prefix matrix all live on dispatch_stream. We bundle them
// into a single int8 slab and view typed sub-tensors via at::from_blob.
//
// Why one slab + one memset instead of 8 separate `torch::empty/zeros`: each
// per-tensor `torch::empty` costs ~5–10 µs of host-side aten-op +
// caching-allocator overhead at production shape, and each `torch::zeros` adds
// another `aten::zero_` + `cudaLaunchKernel` for the per-tensor memset. Trace
// evidence (profiles/phase_d_reorder/) showed ~150 µs of host-side work
// between the metadata kernel ending and dispatch_main launching, dominated
// by ~12 individual aten ops; consolidating into one alloc + one memset
// reclaims the bulk of that latency.
//
// Per-tensor init requirements (audit):
//   rank_prefix_matrix       atomicAdd target in metadata phase 4–5 → MUST be 0.
//   {channel,base,...}_*     written by metadata kernel; init value irrelevant.
//   tile_id_to_expert        written by phase 8 at indices < total_tiles; tail unused.
//   pool_arrival_target      written by phase 8 at indices < total_tiles; tail unused.
// Single bundle-wide memset(0) is correctness-preserving and simplest.
//
// `total_tiles_max` is the upper-bound tile count (every (token, k) at every
// sender contributes at most one slot, plus up to E_local extra tiles from
// per-expert ceil-padding). Production: 8192×4×8/128 + 8 = 2056 ints (~8 KB
// per array). Caller narrows the per-tile views to the polled `total_tiles`
// after the host poll.
struct PrePollBundle {
    torch::Tensor channel_prefix_matrix;     // [R, num_channels]
    torch::Tensor expert_frequency;          // [E_local]
    torch::Tensor expert_pool_block_offset;  // [E_local + 1]
    torch::Tensor base_pool;                 // [num_channels, R, E_local]
    torch::Tensor seen_per_substream;        // [num_channels, R, E_local] — bwd Pass 2 input
    torch::Tensor rank_prefix_matrix;        // [R, R]
    torch::Tensor total_tiles_device;        // [1]
    torch::Tensor tile_id_to_expert;         // [total_tiles_max] (caller narrows)
    torch::Tensor pool_arrival_target;       // [total_tiles_max] (caller narrows)
};

PrePollBundle allocate_pre_poll_bundle(int64_t total_tiles_max,
                                       int num_ranks,
                                       int num_channels,
                                       int num_local_experts,
                                       at::cuda::CUDAStream stream,
                                       int64_t consumer_stream_handle) {
    auto i32_opts = dtype(torch::kInt32).device(torch::kCUDA);
    auto i8_opts  = dtype(torch::kInt8).device(torch::kCUDA);

    SlabBuilder b;
    int64_t off_channel_prefix_matrix    = b.reserve<int>(static_cast<int64_t>(num_ranks) * num_channels);
    int64_t off_expert_frequency         = b.reserve<int>(num_local_experts);
    int64_t off_expert_pool_block_offset = b.reserve<int>(num_local_experts + 1);
    int64_t off_base_pool                = b.reserve<int>(static_cast<int64_t>(num_channels) * num_ranks * num_local_experts);
    int64_t off_seen_per_substream       = b.reserve<int>(static_cast<int64_t>(num_channels) * num_ranks * num_local_experts);
    int64_t off_rank_prefix_matrix       = b.reserve<int>(static_cast<int64_t>(num_ranks) * num_ranks);
    int64_t off_total_tiles_device       = b.reserve<int>(1);
    int64_t off_tile_id_to_expert        = b.reserve<int>(total_tiles_max);
    int64_t off_pool_arrival_target      = b.reserve<int>(total_tiles_max);

    auto bundle = torch::empty({b.total_bytes()}, i8_opts);
    record_consumer_stream(bundle, consumer_stream_handle);
    auto* base = static_cast<int8_t*>(bundle.data_ptr());
    CUDA_CHECK(cudaMemsetAsync(base, 0, b.total_bytes(), stream));

    // Lambda capturing the bundle by value to keep storage alive for the
    // lifetime of any returned view. Each at::from_blob clones the lambda
    // (incrementing the bundle's refcount); on tensor free, the storage
    // deleter destructs the lambda, releasing the refcount.
    auto keep = [bundle](void*) {};

    return PrePollBundle{
        .channel_prefix_matrix    = at::from_blob(base + off_channel_prefix_matrix,    {num_ranks, num_channels},                            keep, i32_opts),
        .expert_frequency         = at::from_blob(base + off_expert_frequency,         {num_local_experts},                                  keep, i32_opts),
        .expert_pool_block_offset = at::from_blob(base + off_expert_pool_block_offset, {num_local_experts + 1},                              keep, i32_opts),
        .base_pool                = at::from_blob(base + off_base_pool,                {num_channels, num_ranks, num_local_experts},         keep, i32_opts),
        .seen_per_substream       = at::from_blob(base + off_seen_per_substream,       {num_channels, num_ranks, num_local_experts},         keep, i32_opts),
        .rank_prefix_matrix       = at::from_blob(base + off_rank_prefix_matrix,       {num_ranks, num_ranks},                               keep, i32_opts),
        .total_tiles_device       = at::from_blob(base + off_total_tiles_device,       {1},                                                  keep, i32_opts),
        .tile_id_to_expert        = at::from_blob(base + off_tile_id_to_expert,        {total_tiles_max},                                    keep, i32_opts),
        .pool_arrival_target      = at::from_blob(base + off_pool_arrival_target,      {total_tiles_max},                                    keep, i32_opts),
    };
}

struct HostPollResult {
    int num_recv_tokens;
    int total_tiles;
};

HostPollResult host_poll_recv_counts(volatile int* moe_recv_counter,
                                     volatile int* moe_recv_expert_counter,
                                     volatile int* streaming_total_tiles,
                                     int num_local_experts) {
    HostPollResult r{};
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        r.num_recv_tokens = static_cast<int>(*moe_recv_counter);
        r.total_tiles     = static_cast<int>(*streaming_total_tiles);
        bool ready = (r.num_recv_tokens >= 0) and (r.total_tiles >= 0);
        for (int i = 0; i < num_local_experts and ready; ++i)
            ready &= moe_recv_expert_counter[i] >= 0;
        if (ready) break;
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_time).count() > NUM_CPU_TIMEOUT_SECS)
            throw std::runtime_error("StreamEP error: CPU recv timeout");
    }
    return r;
}

// Post-host-poll bundle: small/medium per-recv-token + per-tile outputs go
// into ONE int8 slab with three cudaMemsetAsync calls (different offsets,
// different fill bytes), with `metadata_done_event` recorded between them.
// Bundling saves ~80 µs of host aten-op overhead vs separate torch::zeros/empty
// calls.
//
// The event placement matters for kernel A streaming overlap: consumer streams
// wait on metadata_done before reading any metadata tensors, but only some of
// the post-poll buffers are needed by kernel A's start (it spins on
// `pool_arrival_count == pool_arrival_target`).  So we split the Z region
// into two halves around the event:
//
//   Z_pre (zeroed BEFORE event — kernel A / Y read these via the per-tile
//          count-vs-target acquire-spin):
//     pool_topk_weight, recv_channel_prefix_matrix, send_head,
//     pool_arrival_count
//
//   N (0xFF = -1, BEFORE event — kernel Y reads via predicate):
//     pool_recv_token, pool_k_slot
//
//   Z_post (zeroed AFTER event — kernel Y atomic-scatter destinations + combine
//          sender state + backward-only scaffolding. With the 3-stream graph
//          all consumers serialize after dispatch via the layer-end exit
//          chain on caller_stream — no race possible):
//     k_local_remaining, y_done_per_token,
//     recv_token_to_slots, k_local_total
//   (`o` is allocated standalone, NOT in the slab — it is forward-only and a
//    standalone allocation can be freed after fwd combine instead of being
//    pinned through bwd by the slab's shared deleter. See allocate_post_poll_bundle.)
struct PostPollBundle {
    // Z_pre + N region views.
    torch::Tensor pool_topk_weight;
    torch::Tensor recv_channel_prefix_matrix;
    torch::Tensor send_head;
    torch::Tensor pool_arrival_count;
    torch::Tensor pool_recv_token;
    torch::Tensor pool_k_slot;

    // Z_post region views (allocated after metadata_done_event is recorded).
    torch::Tensor k_local_remaining;
    torch::Tensor y_done_per_token;
    torch::Tensor o;

    // Backward-pass scaffolding (Z_post; populated by fwd Pass B; read by bwd):
    torch::Tensor recv_token_to_slots;  // [T_recv, num_topk] int32, -1 for non-local k
    torch::Tensor k_local_total;        // [T_recv]            int32, write-once K_local mirror

    // Recorded between Z_pre/N memsets and Z_post memset. Consumer streams
    // wait_event on this to safely read metadata tensors without serializing
    // against the dispatch main kernel.
    EventHandle metadata_done_event{};
};

PostPollBundle allocate_post_poll_bundle(int64_t TK_padded,
                                         int hidden,
                                         int num_recv_tokens,
                                         int num_topk,
                                         int num_ranks,
                                         int num_channels,
                                         int num_tokens,
                                         int total_tiles,
                                         const torch::TensorOptions& x_options,
                                         at::cuda::CUDAStream stream,
                                         int64_t consumer_stream_handle) {
    auto i32_opts = dtype(torch::kInt32).device(torch::kCUDA);
    auto i64_opts = dtype(torch::kInt64).device(torch::kCUDA);
    auto i8_opts  = dtype(torch::kInt8).device(torch::kCUDA);
    auto f32_opts = dtype(torch::kFloat32).device(torch::kCUDA);

    PostPollBundle out;

    int64_t hidden_bytes_per_recv_token = static_cast<int64_t>(hidden) * x_options.dtype().itemsize();

    SlabBuilder b;
    // Z_pre region (zeroed before metadata_done event).
    int64_t off_pool_topk_weight    = b.reserve<float>(TK_padded);
    int64_t off_recv_channel_prefix = b.reserve<int>(static_cast<int64_t>(num_ranks) * num_channels);
    int64_t off_send_head           = b.reserve<int>(static_cast<int64_t>(num_tokens) * num_ranks);
    int64_t off_pool_arrival_count  = b.reserve<int>(total_tiles);
    int64_t z_pre_bytes             = b.total_bytes();

    // N region (-1 fill, before metadata_done).
    int64_t off_pool_recv_token = b.reserve<int>(TK_padded);
    int64_t off_pool_k_slot     = b.reserve<int>(TK_padded);
    int64_t n_end               = b.total_bytes();

    // Z_post region (zeroed after metadata_done).
    int64_t off_k_local_remaining    = b.reserve<int>(num_recv_tokens);
    int64_t off_y_done_per_token = b.reserve<int64_t>(num_recv_tokens);
    int64_t off_recv_token_to_slots    = b.reserve<int>(static_cast<int64_t>(num_recv_tokens) * num_topk);
    int64_t off_k_local_total          = b.reserve<int>(num_recv_tokens);
    // NOTE: `o` is NOT reserved in the slab — it is allocated standalone below
    // so its storage can be freed after fwd combine (forward-only; see below).

    auto bundle = torch::empty({b.total_bytes()}, i8_opts);
    record_consumer_stream(bundle, consumer_stream_handle);
    auto* base = static_cast<int8_t*>(bundle.data_ptr());

    // Z_pre + N memsets (queued BEFORE metadata_done event recording).
    CUDA_CHECK(cudaMemsetAsync(base,                0x00, z_pre_bytes,         stream));
    CUDA_CHECK(cudaMemsetAsync(base + z_pre_bytes,  0xFF, n_end - z_pre_bytes, stream));

    auto keep = [bundle](void*) {};

    out.pool_topk_weight           = at::from_blob(base + off_pool_topk_weight,    {TK_padded},                          keep, f32_opts);
    out.recv_channel_prefix_matrix = at::from_blob(base + off_recv_channel_prefix, {num_ranks, num_channels},            keep, i32_opts);
    out.send_head                  = at::from_blob(base + off_send_head,           {num_tokens, num_ranks},              keep, i32_opts);
    out.pool_arrival_count         = at::from_blob(base + off_pool_arrival_count,  {total_tiles},                        keep, i32_opts);
    out.pool_recv_token            = at::from_blob(base + off_pool_recv_token,     {TK_padded},                          keep, i32_opts);
    out.pool_k_slot                = at::from_blob(base + off_pool_k_slot,         {TK_padded},                          keep, i32_opts);

    // Z_post memset must precede metadata_done event so that consumer kernels
    // gated on the event (potentially on separate HW queues under CDMC>1)
    // cannot race against this memset and have their writes clobbered.
    CUDA_CHECK(cudaMemsetAsync(base + n_end, 0x00, b.total_bytes() - n_end, stream));

    // `o` (kernel-Y atomic-scatter accumulation target) is allocated standalone,
    // NOT as a slab view. The slab's shared `keep` deleter pins ALL its views'
    // storage alive until the last view dies — i.e. through backward, since
    // bwd-needed views (pool_topk_weight, recv_token_to_slots, ...) share the
    // slab. But `o` is forward-only (consumed by combine, never read in bwd),
    // ~T_recv×hidden bytes; a standalone allocation lets the orchestrator free
    // it right after fwd combine (stream_moe.py sets handle.o=None). Zeroed on
    // `stream` BEFORE metadata_done_event so the compute stream observes the
    // zero-init via the same event kernel A/Y wait on. Zero-init is load-bearing:
    // kernel Y accumulates K_local(r)>=1 contributions per row via red.global.add.
    out.o = torch::empty({num_recv_tokens, hidden}, x_options);
    record_consumer_stream(out.o, consumer_stream_handle);
    CUDA_CHECK(cudaMemsetAsync(out.o.data_ptr(), 0x00,
                               static_cast<int64_t>(num_recv_tokens) * hidden_bytes_per_recv_token, stream));

    out.metadata_done_event = EventHandle(stream);

    out.k_local_remaining    = at::from_blob(base + off_k_local_remaining,    {num_recv_tokens},                        keep, i32_opts);
    out.y_done_per_token = at::from_blob(base + off_y_done_per_token, {num_recv_tokens},                        keep, i64_opts);
    out.recv_token_to_slots    = at::from_blob(base + off_recv_token_to_slots,    {num_recv_tokens, num_topk},              keep, i32_opts);
    out.k_local_total          = at::from_blob(base + off_k_local_total,          {num_recv_tokens},                        keep, i32_opts);

    return out;
}

#ifndef DISABLE_NVSHMEM

// Internode counterpart of PrePollBundle. Same one-slab-one-memset rationale;
// the extra fields (rdma/gbl prefix tiers, recv_*_rank_prefix_sum,
// total_tiles_device) are all metadata-kernel outputs and are fully
// overwritten by `internode::streaming_dispatch_metadata`. rank_prefix_matrix
// remains the lone atomicAdd target in the metadata kernel and so requires
// the slab-wide memset(0).
struct PrePollBundleInternode {
    torch::Tensor rdma_channel_prefix_matrix;  // [num_rdma_ranks, num_channels]
    torch::Tensor recv_rdma_rank_prefix_sum;   // [num_rdma_ranks]
    torch::Tensor gbl_channel_prefix_matrix;   // [num_ranks, num_channels]
    torch::Tensor recv_gbl_rank_prefix_sum;    // [num_ranks]
    torch::Tensor expert_frequency;            // [num_local_experts]
    torch::Tensor expert_pool_block_offset;    // [num_local_experts + 1]
    torch::Tensor base_pool;                   // [num_channels, num_ranks, num_local_experts]
    torch::Tensor seen_per_substream;          // [num_channels, num_ranks, num_local_experts]
    torch::Tensor rank_prefix_matrix;          // [num_ranks, num_ranks]
    torch::Tensor tile_id_to_expert;           // [total_tiles_max] (caller narrows)
    torch::Tensor pool_arrival_target;         // [total_tiles_max] (caller narrows)
    torch::Tensor total_tiles_device;          // [1]
};

PrePollBundleInternode allocate_pre_poll_bundle_internode(int64_t total_tiles_max,
                                                          int num_ranks,
                                                          int num_rdma_ranks,
                                                          int num_channels,
                                                          int num_local_experts,
                                                          at::cuda::CUDAStream stream,
                                                          int64_t consumer_stream_handle) {
    auto i32_opts = dtype(torch::kInt32).device(torch::kCUDA);
    auto i8_opts  = dtype(torch::kInt8).device(torch::kCUDA);

    SlabBuilder b;
    int64_t off_rdma_chprefix       = b.reserve<int>(static_cast<int64_t>(num_rdma_ranks) * num_channels);
    int64_t off_recv_rdma_prefix    = b.reserve<int>(num_rdma_ranks);
    int64_t off_gbl_chprefix        = b.reserve<int>(static_cast<int64_t>(num_ranks) * num_channels);
    int64_t off_recv_gbl_prefix     = b.reserve<int>(num_ranks);
    int64_t off_expert_frequency    = b.reserve<int>(num_local_experts);
    int64_t off_expert_pool_offset  = b.reserve<int>(num_local_experts + 1);
    int64_t off_base_pool           = b.reserve<int>(static_cast<int64_t>(num_channels) * num_ranks * num_local_experts);
    int64_t off_seen_per_substream  = b.reserve<int>(static_cast<int64_t>(num_channels) * num_ranks * num_local_experts);
    int64_t off_rank_prefix_matrix  = b.reserve<int>(static_cast<int64_t>(num_ranks) * num_ranks);
    int64_t off_tile_id_to_expert   = b.reserve<int>(total_tiles_max);
    int64_t off_pool_arrival_target = b.reserve<int>(total_tiles_max);
    int64_t off_total_tiles_device  = b.reserve<int>(1);

    auto bundle = torch::empty({b.total_bytes()}, i8_opts);
    record_consumer_stream(bundle, consumer_stream_handle);
    auto* base = static_cast<int8_t*>(bundle.data_ptr());
    CUDA_CHECK(cudaMemsetAsync(base, 0, b.total_bytes(), stream));

    auto keep = [bundle](void*) {};

    return PrePollBundleInternode{
        .rdma_channel_prefix_matrix = at::from_blob(base + off_rdma_chprefix,       {num_rdma_ranks, num_channels},               keep, i32_opts),
        .recv_rdma_rank_prefix_sum  = at::from_blob(base + off_recv_rdma_prefix,    {num_rdma_ranks},                             keep, i32_opts),
        .gbl_channel_prefix_matrix  = at::from_blob(base + off_gbl_chprefix,        {num_ranks, num_channels},                    keep, i32_opts),
        .recv_gbl_rank_prefix_sum   = at::from_blob(base + off_recv_gbl_prefix,     {num_ranks},                                  keep, i32_opts),
        .expert_frequency           = at::from_blob(base + off_expert_frequency,    {num_local_experts},                          keep, i32_opts),
        .expert_pool_block_offset   = at::from_blob(base + off_expert_pool_offset,  {num_local_experts + 1},                      keep, i32_opts),
        .base_pool                  = at::from_blob(base + off_base_pool,           {num_channels, num_ranks, num_local_experts}, keep, i32_opts),
        .seen_per_substream         = at::from_blob(base + off_seen_per_substream,  {num_channels, num_ranks, num_local_experts}, keep, i32_opts),
        .rank_prefix_matrix         = at::from_blob(base + off_rank_prefix_matrix,  {num_ranks, num_ranks},                       keep, i32_opts),
        .tile_id_to_expert          = at::from_blob(base + off_tile_id_to_expert,   {total_tiles_max},                            keep, i32_opts),
        .pool_arrival_target        = at::from_blob(base + off_pool_arrival_target, {total_tiles_max},                            keep, i32_opts),
        .total_tiles_device         = at::from_blob(base + off_total_tiles_device,  {1},                                          keep, i32_opts),
    };
}

// Internode counterpart of PostPollBundle. Same Z_pre/N/Z_post split for
// kernel A streaming overlap (metadata_done_event recorded between the
// pre-event memsets and the Z_post memset). Extra fields vs intranode are
// the dispatch_main outputs read by combine: recv_rdma_channel_prefix_matrix,
// recv_gbl_channel_prefix_matrix, send_rdma_head, send_nvl_head, recv_src_meta
// — these are all write-only by dispatch_main so init value is irrelevant;
// grouped into Z_pre for the single-memset cost. recv_token_to_slots stays
// -1-init (matches the kernel_y_bwd / combine `slot < 0` predicate) → N region.
struct PostPollBundleInternode {
    // Z_pre region (zero-init, before metadata_done_event).
    torch::Tensor pool_topk_weight;
    torch::Tensor pool_arrival_count;
    torch::Tensor recv_rdma_channel_prefix_matrix;
    torch::Tensor recv_gbl_channel_prefix_matrix;
    torch::Tensor send_rdma_head;
    torch::Tensor send_nvl_head;
    torch::Tensor recv_src_meta;

    // N region (-1 byte-fill, before metadata_done_event).
    torch::Tensor pool_recv_token;
    torch::Tensor pool_k_slot;
    torch::Tensor recv_token_to_slots;

    // Z_post region (zero-init, after metadata_done_event).
    torch::Tensor k_local_remaining;
    torch::Tensor y_done_per_token;
    torch::Tensor o;
    torch::Tensor k_local_total;

    EventHandle metadata_done_event{};
};

PostPollBundleInternode allocate_post_poll_bundle_internode(int64_t TK_padded,
                                                            int hidden,
                                                            int num_recv_tokens,
                                                            int num_rdma_recv_tokens,
                                                            int num_topk,
                                                            int num_ranks,
                                                            int num_rdma_ranks,
                                                            int num_channels,
                                                            int num_tokens,
                                                            int total_tiles,
                                                            int source_meta_bytes,
                                                            const torch::TensorOptions& x_options,
                                                            at::cuda::CUDAStream stream,
                                                            int64_t consumer_stream_handle) {
    auto i32_opts = dtype(torch::kInt32).device(torch::kCUDA);
    auto i64_opts = dtype(torch::kInt64).device(torch::kCUDA);
    auto i8_opts  = dtype(torch::kInt8).device(torch::kCUDA);
    auto u8_opts  = dtype(torch::kUInt8).device(torch::kCUDA);
    auto f32_opts = dtype(torch::kFloat32).device(torch::kCUDA);

    int64_t hidden_bytes_per_recv_token = static_cast<int64_t>(hidden) * x_options.dtype().itemsize();

    SlabBuilder b;
    // Z_pre region (zeroed before metadata_done event).
    int64_t off_pool_topk_weight       = b.reserve<float>(TK_padded);
    int64_t off_pool_arrival_count     = b.reserve<int>(total_tiles);
    int64_t off_recv_rdma_chprefix     = b.reserve<int>(static_cast<int64_t>(num_rdma_ranks) * num_channels);
    int64_t off_recv_gbl_chprefix      = b.reserve<int>(static_cast<int64_t>(num_ranks) * num_channels);
    int64_t off_send_rdma_head         = b.reserve<int>(static_cast<int64_t>(num_tokens) * num_rdma_ranks);
    int64_t off_send_nvl_head          = b.reserve<int>(static_cast<int64_t>(num_rdma_recv_tokens) * NUM_MAX_NVL_PEERS);
    int64_t off_recv_src_meta          = b.reserve_bytes(static_cast<int64_t>(num_recv_tokens) * source_meta_bytes);
    int64_t z_pre_bytes                = b.total_bytes();

    // N region (-1 byte-fill, before metadata_done event).
    int64_t off_pool_recv_token        = b.reserve<int>(TK_padded);
    int64_t off_pool_k_slot            = b.reserve<int>(TK_padded);
    int64_t off_recv_token_to_slots    = b.reserve<int>(static_cast<int64_t>(num_recv_tokens) * num_topk);
    int64_t n_end                      = b.total_bytes();

    // Z_post region (zeroed after metadata_done event).
    int64_t off_k_local_remaining    = b.reserve<int>(num_recv_tokens);
    int64_t off_y_done_per_token = b.reserve<int64_t>(num_recv_tokens);
    int64_t off_k_local_total          = b.reserve<int>(num_recv_tokens);
    // NOTE: `o` is NOT reserved in the slab — allocated standalone below
    // (forward-only; freed after fwd combine). See intranode allocator.

    auto bundle = torch::empty({b.total_bytes()}, i8_opts);
    record_consumer_stream(bundle, consumer_stream_handle);
    auto* base = static_cast<int8_t*>(bundle.data_ptr());

    // Z_pre + N memsets (queued BEFORE metadata_done event recording).
    CUDA_CHECK(cudaMemsetAsync(base,                0x00, z_pre_bytes,         stream));
    CUDA_CHECK(cudaMemsetAsync(base + z_pre_bytes,  0xFF, n_end - z_pre_bytes, stream));

    auto keep = [bundle](void*) {};

    PostPollBundleInternode out;
    out.pool_topk_weight                = at::from_blob(base + off_pool_topk_weight,       {TK_padded},                                  keep, f32_opts);
    out.pool_arrival_count              = at::from_blob(base + off_pool_arrival_count,     {total_tiles},                                keep, i32_opts);
    out.recv_rdma_channel_prefix_matrix = at::from_blob(base + off_recv_rdma_chprefix,     {num_rdma_ranks, num_channels},               keep, i32_opts);
    out.recv_gbl_channel_prefix_matrix  = at::from_blob(base + off_recv_gbl_chprefix,      {num_ranks, num_channels},                    keep, i32_opts);
    out.send_rdma_head                  = at::from_blob(base + off_send_rdma_head,         {num_tokens, num_rdma_ranks},                 keep, i32_opts);
    out.send_nvl_head                   = at::from_blob(base + off_send_nvl_head,          {num_rdma_recv_tokens, NUM_MAX_NVL_PEERS},    keep, i32_opts);
    out.recv_src_meta                   = at::from_blob(base + off_recv_src_meta,          {num_recv_tokens, source_meta_bytes},         keep, u8_opts);
    out.pool_recv_token                 = at::from_blob(base + off_pool_recv_token,        {TK_padded},                                  keep, i32_opts);
    out.pool_k_slot                     = at::from_blob(base + off_pool_k_slot,            {TK_padded},                                  keep, i32_opts);
    out.recv_token_to_slots             = at::from_blob(base + off_recv_token_to_slots,    {num_recv_tokens, num_topk},                  keep, i32_opts);

    // Z_post memset must precede metadata_done event (see intranode dispatch
    // for rationale: consumers gated on the event may run on a separate HW
    // queue under CDMC>1 and race the memset).
    CUDA_CHECK(cudaMemsetAsync(base + n_end, 0x00, b.total_bytes() - n_end, stream));

    // `o` allocated standalone (not a slab view) so it can be freed after fwd
    // combine instead of being pinned through bwd by the shared slab deleter;
    // it is forward-only. See intranode allocate_post_poll_bundle for the full
    // rationale. Zeroed on `stream` before the event (atomic-scatter target).
    out.o = torch::empty({num_recv_tokens, hidden}, x_options);
    record_consumer_stream(out.o, consumer_stream_handle);
    CUDA_CHECK(cudaMemsetAsync(out.o.data_ptr(), 0x00,
                               static_cast<int64_t>(num_recv_tokens) * hidden_bytes_per_recv_token, stream));

    out.metadata_done_event = EventHandle(stream);

    out.k_local_remaining             = at::from_blob(base + off_k_local_remaining,    {num_recv_tokens},                            keep, i32_opts);
    out.y_done_per_token          = at::from_blob(base + off_y_done_per_token, {num_recv_tokens},                            keep, i64_opts);
    out.k_local_total                   = at::from_blob(base + off_k_local_total,          {num_recv_tokens},                            keep, i32_opts);

    return out;
}

#endif  // DISABLE_NVSHMEM

}  // namespace

StreamingDispatchOutputs Buffer::intranode_dispatch(
    const torch::Tensor& x,
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
    int num_tokens = static_cast<int>(x.size(0));
    int hidden = static_cast<int>(x.size(1));
    int num_topk = static_cast<int>(topk_idx.size(1));

    validate_dispatch_inputs(x, topk_idx, topk_weights, is_token_in_rank,
                             num_experts, num_ranks, num_local_experts,
                             expert_alignment, tile_m);

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

    // Pre-host-poll bundle: metadata kernel outputs + sender's prefix matrix.
    int64_t total_tiles_max = static_cast<int64_t>(num_tokens) * num_topk * num_ranks / tile_m + num_local_experts + 1;
    auto pre = allocate_pre_poll_bundle(total_tiles_max, num_ranks, num_channels, num_local_experts, stream, compute_stream_handle_);

    intranode::streaming_dispatch_metadata(topk_idx.data_ptr<topk_idx_t>(),
                                           pre.expert_frequency.data_ptr<int>(),
                                           pre.expert_pool_block_offset.data_ptr<int>(),
                                           pre.base_pool.data_ptr<int>(),
                                           pre.seen_per_substream.data_ptr<int>(),
                                           pre.rank_prefix_matrix.data_ptr<int>(),
                                           pre.tile_id_to_expert.data_ptr<int>(),
                                           pre.pool_arrival_target.data_ptr<int>(),
                                           pre.total_tiles_device.data_ptr<int>(),
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

    // Copy rank_prefix_matrix into the IPC slab head. The queue metadata
    // that follows (start_offset, end_offset, head_idx, tail_idx — now
    // 4 × num_channels × num_ranks uint64 genstamped slots) does NOT need
    // a per-iter memset+barrier here: every reader filters by seq match
    // (`utils.cuh::nvl_seq_match`), so stale residue from a prior iter
    // is rejected without zeroing. Iter-0 cleanliness is handled by the
    // once-only ctor memset of the full slab in `Buffer::initialize`.
    EP_HOST_ASSERT((num_ranks * num_ranks * sizeof(int))
                   + (4 * num_channels * num_ranks * sizeof(uint64_t)) <= num_nvl_bytes);
    CUDA_CHECK(cudaMemcpyAsync(buffer_ptrs[nvl_rank],
                               pre.rank_prefix_matrix.data_ptr<int>(),
                               num_ranks * num_ranks * sizeof(int),
                               cudaMemcpyDeviceToDevice,
                               stream));

    auto poll = host_poll_recv_counts(moe_recv_counter, moe_recv_expert_counter,
                                      streaming_total_tiles, num_local_experts);
    int64_t TK_padded = static_cast<int64_t>(poll.total_tiles) * tile_m;

    // pool[TK_padded, hidden] (~290 MB at production) lives outside the
    // post-poll bundle — too big to coalesce into the same caching-allocator
    // size class as the small bundle. Allocate uninitialized: every
    // downstream consumer either predicates on `pool_recv_token >= 0` (fwd
    // kernel A's tile-streaming, fwd combine's gather) or uses quack's
    // `lens_k` to bound the K-tile via TMA's OOB-zero-fill (bwd dW1's
    // grouped GEMM). Padding rows are never read.
    auto pool = torch::empty({TK_padded, hidden}, x.options());
    record_consumer_stream(pool, compute_stream_handle_);

    auto post = allocate_post_poll_bundle(
        TK_padded, hidden, poll.num_recv_tokens, num_topk,
        num_ranks, num_channels, num_tokens, poll.total_tiles,
        x.options(), stream, compute_stream_handle_);

    // Narrow the per-tile arrays from total_tiles_max → total_tiles for the
    // returned views. `narrow` on a from_blob'd tensor returns a view sharing
    // the same storage; the bundle stays alive via the from_blob deleter
    // lambda. Visible size is only known post-poll, so we can't size the
    // arrays correctly at allocate_pre_poll_bundle time.
    auto tile_id_to_expert   = pre.tile_id_to_expert.narrow(0, 0, poll.total_tiles);
    auto pool_arrival_target = pre.pool_arrival_target.narrow(0, 0, poll.total_tiles);

    EP_HOST_ASSERT(
        num_ranks * num_ranks * sizeof(int) +
            num_channels * num_ranks * sizeof(int) +
            num_channels * num_ranks * sizeof(int) +
            num_channels * num_ranks * sizeof(int) * 2 +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * pool.element_size() +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(topk_idx_t) +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float)
        <= num_nvl_bytes);

    intranode::DispatchPoolOut dispatch_pool_out{
        .pool             = reinterpret_cast<int4*>(pool.data_ptr()),
        .pool_topk_weight = post.pool_topk_weight.data_ptr<float>(),
        .pool_recv_token  = post.pool_recv_token.data_ptr<int>(),
        .pool_k_slot      = post.pool_k_slot.data_ptr<int>(),
    };
    intranode::DispatchPerTokenOut dispatch_per_token_out{
        .recv_channel_prefix_matrix = post.recv_channel_prefix_matrix.data_ptr<int>(),
        .send_head                  = post.send_head.data_ptr<int>(),
        .k_local_remaining        = post.k_local_remaining.data_ptr<int>(),
        .recv_token_to_slots        = post.recv_token_to_slots.data_ptr<int>(),
        .k_local_total              = post.k_local_total.data_ptr<int>(),
    };
    intranode::DispatchInputs dispatch_inputs{
        .x                = reinterpret_cast<const int4*>(x.data_ptr()),
        .topk_idx         = topk_idx.data_ptr<topk_idx_t>(),
        .topk_weights     = topk_weights.data_ptr<float>(),
        .is_token_in_rank = is_token_in_rank.data_ptr<bool>(),
    };
    // Increment the host-side issued-count BEFORE launching so subsequent
    // `wait_dispatch_main_started` calls compare against the right target.
    ++dispatch_main_issued_count;
    intranode::DispatchTileSignal dispatch_tile_signal{
        .channel_prefix_matrix = pre.channel_prefix_matrix.data_ptr<int>(),
        .base_pool             = pre.base_pool.data_ptr<int>(),
        .pool_arrival_count    = post.pool_arrival_count.data_ptr<int>(),
        .pool_arrival_target   = pool_arrival_target.data_ptr<int>(),
        .dispatch_seq          = dispatch_seq,
        .started_flag          = dispatch_main_started_flag,
    };
    intranode::DispatchShape dispatch_shape{
        .num_tokens          = num_tokens,
        .hidden_int4         = static_cast<int>(hidden * pool.element_size() / sizeof(int4)),
        .num_topk            = num_topk,
        .num_experts         = num_experts,
        .tile_m              = tile_m,
    };
    intranode::DispatchEnv dispatch_env{
        .buffer_ptrs            = buffer_ptrs_gpu,
        .rank                   = rank,
        .num_max_send_tokens    = config.num_max_nvl_chunked_send_tokens,
        .num_recv_buffer_tokens = config.num_max_nvl_chunked_recv_tokens,
    };
    intranode::launch_dispatch_main(dispatch_pool_out, dispatch_per_token_out,
                                    dispatch_inputs, dispatch_tile_signal,
                                    dispatch_shape, dispatch_env,
                                    num_ranks, stream, config.num_sms);

    return StreamingDispatchOutputs{
        .pool                       = pool,
        .pool_topk_weight           = post.pool_topk_weight,
        .pool_recv_token            = post.pool_recv_token,
        .pool_k_slot                = post.pool_k_slot,
        .send_head                  = post.send_head,
        .rank_prefix_matrix         = pre.rank_prefix_matrix,
        .channel_prefix_matrix      = pre.channel_prefix_matrix,
        .recv_channel_prefix_matrix = post.recv_channel_prefix_matrix,
        .expert_frequency           = pre.expert_frequency,
        .expert_pool_block_offset   = pre.expert_pool_block_offset,
        .base_pool                  = pre.base_pool,
        .seen_per_substream         = pre.seen_per_substream,
        .tile_id_to_expert          = tile_id_to_expert,
        .pool_arrival_target        = pool_arrival_target,
        .pool_arrival_count         = post.pool_arrival_count,
        .k_local_remaining        = post.k_local_remaining,
        .y_done_per_token     = post.y_done_per_token,
        .o                          = post.o,
        .recv_token_to_slots        = post.recv_token_to_slots,
        .k_local_total              = post.k_local_total,
        .total_tiles                = poll.total_tiles,
        .metadata_done_event        = post.metadata_done_event,
    };
}

std::tuple<torch::Tensor, torch::Tensor, EventHandle> Buffer::intranode_dispatch_grads(
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
    const Config& config) {
    EP_HOST_ASSERT(dL_dy.dim() == 2 and dL_dy.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous() and
                   is_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(recv_token_to_slots.dim() == 2 and recv_token_to_slots.is_contiguous() and
                   recv_token_to_slots.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(base_pool.is_contiguous() and base_pool.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(seen_per_substream.is_contiguous() and seen_per_substream.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(pool_arrival_target.dim() == 1 and pool_arrival_target.is_contiguous() and
                   pool_arrival_target.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and
                   rank_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and
                   rank_prefix_matrix.size(1) == num_ranks);

    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    int num_local_experts = num_experts / num_ranks;

    int num_tokens = static_cast<int>(dL_dy.size(0));
    int hidden = static_cast<int>(dL_dy.size(1));
    EP_HOST_ASSERT((hidden * dL_dy.element_size()) % sizeof(int4) == 0);
    int hidden_int4 = hidden * dL_dy.element_size() / sizeof(int4);

    EP_HOST_ASSERT(is_token_in_rank.size(0) == num_tokens and is_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(recv_token_to_slots.size(1) == num_topk);
    EP_HOST_ASSERT(base_pool.numel() == static_cast<int64_t>(num_channels) * num_ranks * num_local_experts);
    EP_HOST_ASSERT(seen_per_substream.numel() == base_pool.numel());

    int total_tiles = static_cast<int>(pool_arrival_target.size(0));
    auto stream = at::cuda::getCurrentCUDAStream();

    // Output: pool-shaped dL_do_pool. Receiver overwrites only the valid
    // (real) slots; padding rows are NOT touched by the receiver. Allocate
    // uninitialized: kernel_y_bwd reads dL_do_pool through its mPaddingMask
    // predicate (zeros (dgate, dup, postact, g) at padding rows BEFORE any
    // store), and dW2's grouped GEMM reads dL_do_pool through quack's
    // `lens_k` which bounds the K-tile via TMA's OOB-zero-fill. Padding
    // rows are never functionally read.
    auto dL_do_pool = torch::empty({TK_padded, hidden}, dL_dy.options());
    record_consumer_stream(dL_do_pool, compute_stream_handle_);

    // Per-tile signal arrays. Both zero-init: bwd_dispatch_arrival_count
    // accumulates Pass 2 release-adds; kernel_y_bwd's scheduler spins on
    // `bwd_dispatch_arrival_count[tile] == pool_arrival_target[tile]`.
    auto i32_opts = dtype(torch::kInt32).device(torch::kCUDA);
    auto bwd_dispatch_arrival_count = torch::zeros({total_tiles}, i32_opts);
    record_consumer_stream(bwd_dispatch_arrival_count, compute_stream_handle_);

    // No per-iter memset+barrier on the IPC ring control bytes — the
    // (start_offset, end_offset, head_idx, tail_idx) slots are 64-bit
    // genstamped and reader-filtered by seq match. See the fwd dispatch
    // launch above for the full reasoning.
    EP_HOST_ASSERT((num_ranks * num_ranks * sizeof(int))
                   + (4 * num_channels * num_ranks * sizeof(uint64_t)) <= num_nvl_bytes);

    intranode::DispatchGradsIO io{
        .dL_do_pool       = reinterpret_cast<int4*>(dL_do_pool.data_ptr()),
        .dL_dy            = reinterpret_cast<const int4*>(dL_dy.data_ptr()),
        .is_token_in_rank = is_token_in_rank.data_ptr<bool>(),
    };
    intranode::DispatchGradsRouting routing{
        .recv_token_to_slots = recv_token_to_slots.data_ptr<int>(),
        .base_pool           = base_pool.data_ptr<int>(),
        .seen_per_substream  = seen_per_substream.data_ptr<int>(),
        .rank_prefix_matrix  = rank_prefix_matrix.data_ptr<int>(),
    };
    // Bump issued-count before launch (see fwd path for rationale).
    ++dispatch_grads_issued_count;
    intranode::DispatchGradsTileSignal tile_signal{
        .bwd_dispatch_arrival_count = bwd_dispatch_arrival_count.data_ptr<int>(),
        .pool_arrival_target        = pool_arrival_target.data_ptr<int>(),
        .dispatch_seq               = dispatch_seq,
        .started_flag               = dispatch_grads_started_flag,
    };
    intranode::DispatchGradsShape shape{
        .num_tokens  = num_tokens,
        .hidden_int4 = hidden_int4,
        .num_topk    = num_topk,
        .num_experts = num_experts,
        .tile_m      = tile_m,
    };
    intranode::DispatchEnv env{
        .buffer_ptrs            = buffer_ptrs_gpu,
        .rank                   = rank,
        .num_max_send_tokens    = config.num_max_nvl_chunked_send_tokens,
        .num_recv_buffer_tokens = config.num_max_nvl_chunked_recv_tokens,
    };
    // Record an event between the channel-control barrier and the main
    // kernel launch — analogous to fwd dispatch's `metadata_done_event`. The
    // bwd orchestrator's compute stream waits on this so kernel_y_bwd can
    // launch CONCURRENT with dispatch_grads_main_kernel; kernel_y_bwd's
    // scheduler then spins on `bwd_dispatch_arrival_count` per-tile for the
    // dispatch_grads ↔ Y_bwd streaming overlap.
    EventHandle grads_started_event(stream);

    intranode::launch_dispatch_grads_main(io, routing, tile_signal, shape, env,
                                          num_ranks, stream, config.num_sms);

    return std::make_tuple(dL_do_pool, bwd_dispatch_arrival_count, grads_started_event);
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> Buffer::intranode_combine(
    const torch::Tensor& x,
    const torch::Tensor& per_slot_weights,
    const torch::Tensor& recv_token_to_slots,
    const torch::Tensor& rank_prefix_matrix,
    const torch::Tensor& channel_prefix_matrix,
    const torch::Tensor& send_head,
    const torch::Tensor& y_done_per_token,
    int64_t combine_seq,
    bool is_fwd,
    const Config& config) {
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(per_slot_weights.dim() == 1 and per_slot_weights.is_contiguous() and
                   per_slot_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(recv_token_to_slots.dim() == 2 and recv_token_to_slots.is_contiguous() and
                   recv_token_to_slots.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and
                   rank_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and
                   channel_prefix_matrix.scalar_type() == torch::kInt32);
    // Phase-D per-token gate: kernel_y forward (fwd combine) or kernel_a_bwd
    // (bwd combine_grads) release-stores `combine_seq` into
    // y_done_per_token[r] once the per-recv-token stripe is fully
    // assembled. Sender's per-warp send loop spins on this before reading x[r].
    EP_HOST_ASSERT(y_done_per_token.dim() == 1 and y_done_per_token.is_contiguous() and
                   y_done_per_token.scalar_type() == torch::kInt64);

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
    // y_done_per_token is indexed by combine sender's iteration variable
    // `token_idx` ∈ [0, num_tokens), which (in combine's naming convention)
    // is the number of input rows of `x` = handle.o.size(0) = T_recv from
    // dispatch's perspective. Note combine's `num_recv_tokens` is dispatch's
    // source-token count (output rows of combine), NOT the size of the gate.
    EP_HOST_ASSERT(y_done_per_token.size(0) == num_tokens);
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(recv_token_to_slots.size(0) == num_tokens);

    int num_topk = static_cast<int>(recv_token_to_slots.size(1));

    // All kernels + allocations run on the caller's current stream.
    auto stream = at::cuda::getCurrentCUDAStream();

    // Fwd combine: no per-K weight output. Kernel Y already pre-multiplies
    // pool_topk_weight per row before the atomic scatter, so `recv_x[t]` =
    // Σ_k w_k·y_k is reconstructed from the data reduce alone. C++ returns
    // `c10::nullopt`; Python surfaces it as `None`.
    auto f32_opts = dtype(torch::kFloat32).device(torch::kCUDA);
    c10::optional<torch::Tensor> recv_topk_weights_out = is_fwd
        ? c10::nullopt
        : c10::optional<torch::Tensor>(torch::empty({num_recv_tokens, num_topk}, f32_opts));
    float* recv_topk_weights_ptr = is_fwd ? nullptr : recv_topk_weights_out->data_ptr<float>();
    if (recv_topk_weights_out.has_value())
        record_consumer_stream(*recv_topk_weights_out, compute_stream_handle_);

    // Combine data. Combine carves its sub-buffer chain from
    // `buffer_ptrs[i] + dispatch_section_bytes`, physically disjoint from
    // dispatch's chain at `buffer_ptrs[i] + 0`. No pre-combine barrier or
    // memset is needed: genstamped meta slots reject stale residue on
    // read, and combine's slots never overlap dispatch's.
    int hidden_int4_x = static_cast<int>(hidden * x.element_size() / sizeof(int4));
    int64_t dispatch_section_bytes = intranode::intranode_get_dispatch_section_bytes(
        num_channels, num_ranks, config.num_max_nvl_chunked_recv_tokens,
        hidden_int4_x, num_topk);
    int64_t combine_section_bytes = intranode::intranode_get_combine_section_bytes(
        num_channels, num_ranks, config.num_max_nvl_chunked_recv_tokens,
        hidden_int4_x, num_topk);
    EP_HOST_ASSERT(dispatch_section_bytes + combine_section_bytes <= num_nvl_bytes);
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    record_consumer_stream(recv_x, compute_stream_handle_);
    intranode::launch_combine_main(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                       recv_x.data_ptr(),
                       recv_topk_weights_ptr,
                       x.data_ptr(),
                       per_slot_weights.data_ptr<float>(),
                       recv_token_to_slots.data_ptr<int>(),
                       rank_prefix_matrix.data_ptr<int>(),
                       channel_prefix_matrix.data_ptr<int>(),
                       send_head.data_ptr<int>(),
                       y_done_per_token.data_ptr<int64_t>(),
                       combine_seq,
                       is_fwd,
                       num_tokens,
                       num_recv_tokens,
                       hidden,
                       num_topk,
                       buffer_ptrs_gpu,
                       dispatch_section_bytes,
                       rank,
                       num_ranks,
                       stream,
                       config.num_sms,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens);

    return {recv_x, recv_topk_weights_out};
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming-MoE consolidated dispatch (internode, pool layout). Mirrors
// `Buffer::intranode_dispatch` shape: one folded metadata kernel +
// host poll on host-mapped recv counters + dispatch_main kernel. The
// metadata kernel is `internode::streaming_dispatch_metadata` (single
// launch — RDMA exchange + NVL aggregation + channel matrices + streaming
// superset). Dispatch_main is `internode::dispatch_main_kernel` — the
// NVL receiver runs Pass A + Pass B + Pass 2 fire (slot allocation +
// per-pool-slot scalar writes + pool_arrival_count release-add), mirroring the
// intranode receiver. RDMA sender + forwarder stages are bulk RDMA→NVL
// staging only, no slot logic.
// ─────────────────────────────────────────────────────────────────────────────
StreamingDispatchOutputs Buffer::internode_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& is_token_in_rank,
    int num_experts,
    int expert_alignment,
    int tile_m,
    int64_t dispatch_seq,
    const Config& config) {
#ifndef DISABLE_NVSHMEM
    pybind11::gil_scoped_release release;

    EP_HOST_ASSERT(num_rdma_ranks > 1);
    EP_HOST_ASSERT(0 < num_rdma_ranks and num_rdma_ranks <= NUM_MAX_RDMA_PEERS);
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    int num_local_experts = num_experts / num_ranks;
    int num_rdma_experts = num_experts / num_rdma_ranks;
    int num_tokens = static_cast<int>(x.size(0));
    int hidden = static_cast<int>(x.size(1));
    int num_topk = static_cast<int>(topk_idx.size(1));
    int hidden_int4 = static_cast<int>(hidden * x.element_size() / sizeof(int4));

    validate_dispatch_inputs(x, topk_idx, topk_weights, is_token_in_rank,
                             num_experts, num_ranks, num_local_experts,
                             expert_alignment, tile_m);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Reset host-mapped sync slots before metadata kernel writes them.
    *moe_recv_counter = -1;
    *moe_recv_rdma_counter = -1;
    *streaming_total_tiles = -1;
    for (int i = 0; i < num_local_experts; ++i)
        moe_recv_expert_counter[i] = -1;

    // Pre-host-poll metadata-kernel outputs bundled into one int8 slab + one
    // memset (see allocate_pre_poll_bundle_internode). Internode-specific
    // shapes: base_pool / seen_per_substream / rank_prefix_matrix span
    // num_world_ranks (= num_ranks for the streaming path on internode);
    // channel prefix matrices come in two tiers (rdma + gbl).
    int64_t total_tiles_max = static_cast<int64_t>(num_tokens) * num_topk * num_ranks / tile_m + num_local_experts + 1;
    auto pre = allocate_pre_poll_bundle_internode(total_tiles_max, num_ranks, num_rdma_ranks,
                                                  num_channels, num_local_experts, stream,
                                                  compute_stream_handle_);

    // Streaming SymBuffer offset within rdma_buffer_ptr — placed AFTER the
    // metadata kernel's count payload. The count-exchange and streaming-
    // superset payloads share `rdma_buffer_ptr`; the fixed offset keeps the
    // kernel's SymBuffer math consistent across them.
    int64_t streaming_rdma_offset =
        2 * static_cast<int64_t>(num_rdma_ranks) *
        (NUM_MAX_NVL_PEERS + num_rdma_experts + 1) * sizeof(int);
    int64_t streaming_rdma_total =
        2 * static_cast<int64_t>(num_rdma_ranks) * num_channels *
        NUM_MAX_NVL_PEERS * num_local_experts * sizeof(int);
    // The metadata kernel's count-exchange + streaming-superset payloads live
    // in a region DISJOINT from the dispatch + combine hot-path SymBuffer
    // chains: the metadata kernel runs at EVERY call, and under cross-rank
    // run-ahead its RDMA puts land while a lagging peer's PRIOR iterations
    // are still draining their rings. Sharing base offset 0 with the dispatch
    // data ring (channel 0 sits first) let a leading rank's metadata puts
    // clobber a lagging rank's live channel-0 ring slots.
    const int64_t metadata_rdma_offset =
        internode::get_dispatch_rdma_region_bytes(
            hidden_int4, num_topk, config.num_max_rdma_chunked_recv_tokens,
            num_channels, num_rdma_ranks) +
        internode::get_combine_rdma_region_bytes(
            hidden_int4, num_topk, config.num_max_rdma_chunked_recv_tokens,
            num_channels, num_rdma_ranks);
    EP_HOST_ASSERT(metadata_rdma_offset + streaming_rdma_offset + streaming_rdma_total
                   <= num_rdma_bytes);

    internode::streaming_dispatch_metadata(
        topk_idx.data_ptr<topk_idx_t>(),
        moe_recv_counter_mapped, moe_recv_rdma_counter_mapped,
        moe_recv_expert_counter_mapped, streaming_total_tiles_mapped,
        pre.rdma_channel_prefix_matrix.data_ptr<int>(),
        pre.recv_rdma_rank_prefix_sum.data_ptr<int>(),
        pre.gbl_channel_prefix_matrix.data_ptr<int>(),
        pre.recv_gbl_rank_prefix_sum.data_ptr<int>(),
        pre.expert_frequency.data_ptr<int>(),
        pre.expert_pool_block_offset.data_ptr<int>(),
        pre.base_pool.data_ptr<int>(),
        pre.seen_per_substream.data_ptr<int>(),
        pre.rank_prefix_matrix.data_ptr<int>(),
        pre.tile_id_to_expert.data_ptr<int>(),
        pre.pool_arrival_target.data_ptr<int>(),
        pre.total_tiles_device.data_ptr<int>(),
        num_tokens, num_topk, num_experts, num_channels,
        hidden_int4, expert_alignment, tile_m,
        streaming_rdma_offset,
        static_cast<void*>(static_cast<uint8_t*>(rdma_buffer_ptr) + metadata_rdma_offset),
        buffer_ptrs_gpu, barrier_signal_ptrs_gpu,
        rank, num_ranks, stream, num_rdma_bytes, num_nvl_bytes);

    // Host-poll all four host-mapped counters before dispatch_main launch
    // (we need num_recv_tokens to allocate `pool` and num_rdma_recv_tokens
    // to size send_nvl_head).
    int num_recv_tokens = -1, num_rdma_recv_tokens = -1, total_tiles = -1;
    auto t0 = std::chrono::high_resolution_clock::now();
    while (true) {
        num_recv_tokens      = static_cast<int>(*moe_recv_counter);
        num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);
        total_tiles          = static_cast<int>(*streaming_total_tiles);
        bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0) and (total_tiles >= 0);
        for (int i = 0; i < num_local_experts and ready; ++i)
            ready &= moe_recv_expert_counter[i] >= 0;
        if (ready) break;
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - t0).count() > NUM_CPU_TIMEOUT_SECS)
            throw std::runtime_error("StreamEP error: internode_dispatch CPU timeout");
    }
    int64_t TK_padded = static_cast<int64_t>(total_tiles) * tile_m;

    // pool[TK_padded, hidden] (~290 MB at production) lives outside the
    // post-poll bundle — too big to coalesce into the same caching-allocator
    // size class. Allocate uninitialized: every downstream consumer either
    // predicates on `pool_recv_token >= 0` (kernel A's tile-streaming,
    // combine's gather) or uses quack's `lens_k` for OOB-zero-fill (bwd dW
    // grouped GEMM). Padding rows are never read.
    auto pool = torch::empty({TK_padded, hidden}, x.options());
    record_consumer_stream(pool, compute_stream_handle_);

    auto post = allocate_post_poll_bundle_internode(
        TK_padded, hidden, num_recv_tokens, num_rdma_recv_tokens, num_topk,
        num_ranks, num_rdma_ranks, num_channels, num_tokens, total_tiles,
        internode::get_source_meta_bytes(), x.options(), stream,
        compute_stream_handle_);

    auto tile_id_to_expert_n   = pre.tile_id_to_expert.narrow(0, 0, total_tiles);
    auto pool_arrival_target_n = pre.pool_arrival_target.narrow(0, 0, total_tiles);

    internode::DispatchPoolOut pool_out{
        .pool             = reinterpret_cast<int4*>(pool.data_ptr()),
        .pool_topk_weight = post.pool_topk_weight.data_ptr<float>(),
        .pool_recv_token  = post.pool_recv_token.data_ptr<int>(),
        .pool_k_slot      = post.pool_k_slot.data_ptr<int>(),
    };
    internode::DispatchPerTokenOut per_token_out{
        .k_local_remaining               = post.k_local_remaining.data_ptr<int>(),
        .recv_token_to_slots               = post.recv_token_to_slots.data_ptr<int>(),
        .k_local_total                     = post.k_local_total.data_ptr<int>(),
        .recv_src_meta                     = post.recv_src_meta.data_ptr(),
        .send_rdma_head                    = post.send_rdma_head.data_ptr<int>(),
        .send_nvl_head                     = post.send_nvl_head.data_ptr<int>(),
        .recv_rdma_channel_prefix_matrix   = post.recv_rdma_channel_prefix_matrix.data_ptr<int>(),
        .recv_gbl_channel_prefix_matrix    = post.recv_gbl_channel_prefix_matrix.data_ptr<int>(),
    };
    internode::DispatchInputs inputs{
        .x                          = reinterpret_cast<const int4*>(x.data_ptr()),
        .topk_idx                   = topk_idx.data_ptr<topk_idx_t>(),
        .topk_weights               = topk_weights.data_ptr<float>(),
        .is_token_in_rank           = is_token_in_rank.data_ptr<bool>(),
        .rdma_channel_prefix_matrix = pre.rdma_channel_prefix_matrix.data_ptr<int>(),
        .recv_rdma_rank_prefix_sum  = pre.recv_rdma_rank_prefix_sum.data_ptr<int>(),
        .gbl_channel_prefix_matrix  = pre.gbl_channel_prefix_matrix.data_ptr<int>(),
        .recv_gbl_rank_prefix_sum   = pre.recv_gbl_rank_prefix_sum.data_ptr<int>(),
    };
    // Bump issued-count before launch (shared Buffer flags across intra/internode).
    ++dispatch_main_issued_count;
    internode::DispatchTileSignal tile_signal{
        .base_pool                 = pre.base_pool.data_ptr<int>(),
        .seen_per_substream        = pre.seen_per_substream.data_ptr<int>(),
        .pool_arrival_count        = post.pool_arrival_count.data_ptr<int>(),
        .pool_arrival_target       = pool_arrival_target_n.data_ptr<int>(),
        .dispatch_seq              = dispatch_seq,
        .started_flag              = dispatch_main_started_flag,
    };
    internode::DispatchShape shape{
        .num_tokens  = num_tokens,
        .hidden_int4 = hidden_int4,
        .num_topk    = num_topk,
        .num_experts = num_experts,
        .tile_m      = tile_m,
    };
    internode::DispatchEnv env{
        .rdma_buffer_ptr                     = rdma_buffer_ptr,
        .buffer_ptrs                         = buffer_ptrs_gpu,
        .rank                                = rank,
        .num_max_rdma_chunked_send_tokens    = config.num_max_rdma_chunked_send_tokens,
        .num_max_rdma_chunked_recv_tokens    = config.num_max_rdma_chunked_recv_tokens,
        .num_max_nvl_chunked_send_tokens     = config.num_max_nvl_chunked_send_tokens,
        .num_max_nvl_chunked_recv_tokens     = config.num_max_nvl_chunked_recv_tokens,
        .reader_prev_head                    = dispatch_reader_prev_head,
        .reader_prev_tail                    = dispatch_reader_prev_tail,
    };

    internode::launch_dispatch_main(pool_out, per_token_out, inputs, tile_signal,
                                    shape, env, num_rdma_ranks, num_channels, stream);

    return StreamingDispatchOutputs{
        .pool                            = pool,
        .pool_topk_weight                = post.pool_topk_weight,
        .pool_recv_token                 = post.pool_recv_token,
        .pool_k_slot                     = post.pool_k_slot,
        .send_head                       = torch::Tensor(),
        .rank_prefix_matrix              = pre.rank_prefix_matrix,
        .channel_prefix_matrix           = torch::Tensor(),
        .recv_channel_prefix_matrix      = torch::Tensor(),
        .expert_frequency                = pre.expert_frequency,
        .expert_pool_block_offset        = pre.expert_pool_block_offset,
        .base_pool                       = pre.base_pool,
        .seen_per_substream              = pre.seen_per_substream,
        .tile_id_to_expert               = tile_id_to_expert_n,
        .pool_arrival_target             = pool_arrival_target_n,
        .pool_arrival_count              = post.pool_arrival_count,
        .k_local_remaining             = post.k_local_remaining,
        .y_done_per_token          = post.y_done_per_token,
        .o                               = post.o,
        .recv_token_to_slots             = post.recv_token_to_slots,
        .k_local_total                   = post.k_local_total,
        .total_tiles                     = total_tiles,
        .metadata_done_event             = post.metadata_done_event,
        .rdma_channel_prefix_matrix      = pre.rdma_channel_prefix_matrix,
        .recv_rdma_rank_prefix_sum       = pre.recv_rdma_rank_prefix_sum,
        .gbl_channel_prefix_matrix       = pre.gbl_channel_prefix_matrix,
        .recv_gbl_rank_prefix_sum        = pre.recv_gbl_rank_prefix_sum,
        .recv_rdma_channel_prefix_matrix = post.recv_rdma_channel_prefix_matrix,
        .recv_gbl_channel_prefix_matrix  = post.recv_gbl_channel_prefix_matrix,
        .send_rdma_head                  = post.send_rdma_head,
        .send_nvl_head                   = post.send_nvl_head,
        .recv_src_meta                   = post.recv_src_meta,
    };
#else
    EP_HOST_ASSERT(false and "internode_dispatch requires NVSHMEM");
    return {};
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming-MoE bwd dispatch (internode, pool layout). Single kernel; no
// metadata kernel, no host poll — routing tensors come from the fwd
// dispatch's StreamingDispatchOutputs (persisted on the StreamingHandle).
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, EventHandle> Buffer::internode_dispatch_grads(
    const torch::Tensor& dL_dy,
    const torch::Tensor& is_token_in_rank,
    const StreamingDispatchOutputs& dispatch_out,
    int64_t TK_padded,
    int64_t dispatch_seq,
    const Config& config) {
#ifndef DISABLE_NVSHMEM
    pybind11::gil_scoped_release release;

    EP_HOST_ASSERT(num_rdma_ranks > 1 and "internode_dispatch_grads requires multi-RDMA-rank world");
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    EP_HOST_ASSERT(dL_dy.dim() == 2 and dL_dy.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous() and is_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(is_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(dispatch_out.recv_token_to_slots.dim() == 2 and dispatch_out.recv_token_to_slots.is_contiguous());

    int num_tokens = static_cast<int>(dL_dy.size(0));
    int hidden = static_cast<int>(dL_dy.size(1));
    int num_topk = static_cast<int>(dispatch_out.recv_token_to_slots.size(1));
    int hidden_int4 = static_cast<int>(hidden * dL_dy.element_size() / sizeof(int4));
    int num_local_experts = static_cast<int>(dispatch_out.expert_pool_block_offset.size(0)) - 1;
    int num_experts = num_local_experts * num_ranks;
    int total_tiles = static_cast<int>(dispatch_out.tile_id_to_expert.size(0));
    int tile_m = (total_tiles == 0) ? 0 : static_cast<int>(TK_padded / total_tiles);
    // total_tiles == 0 (empty-recv rank: this rank's local experts received
    // nothing in fwd, so no bwd pool to fan into) is a legitimate runtime
    // state under extreme expert-skew routings (e.g. test_skewed_experts'
    // all_to_first_K with K <= num_local_experts), NOT an error. We still
    // launch the kernel because:
    //   1. Sender side has work — this rank's local layer has dL/dy for
    //      `num_tokens` source tokens that must be shipped to expert ranks.
    //   2. Receiver / forwarder side must still emit the per-link data-ring
    //      header (N=0) + bump the tail so remote-RDMA-peer forwarders waiting
    //      on the header's arrival don't time out.
    // tile_m is used inside the NVL-receiver path (slot/tile_m, fire_pool_blocks)
    // which is gated by `num_recv_tokens > 0` and never executes on this
    // rank, so tile_m=0 is safe at the kernel level.
    EP_HOST_ASSERT((total_tiles == 0) or (tile_m > 0));

    auto stream = at::cuda::getCurrentCUDAStream();

    auto dL_do_pool = torch::zeros({TK_padded, hidden}, dL_dy.options());
    record_consumer_stream(dL_do_pool, compute_stream_handle_);
    auto i32_opts = at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto bwd_dispatch_arrival_count = torch::zeros({total_tiles}, i32_opts);
    record_consumer_stream(bwd_dispatch_arrival_count, compute_stream_handle_);

    // No pre-bwd-dispatch cleanup: under the cumulative ring-control
    // protocols (RDMA head/tail amo, RDMA meta sentinel-amo, NVL gen-stamp),
    // every polled slot disambiguates this iter's value from prior-iter
    // residue without an inter-iter memset. The dispatch stream sequences
    // straight fwd dispatch → bwd dispatch_grads.

    internode::DispatchGradsIO io{
        .dL_do_pool        = reinterpret_cast<int4*>(dL_do_pool.data_ptr()),
        .dL_dy             = reinterpret_cast<const int4*>(dL_dy.data_ptr()),
        .is_token_in_rank  = is_token_in_rank.data_ptr<bool>(),
    };

    internode::DispatchGradsRouting routing{
        .recv_token_to_slots                = dispatch_out.recv_token_to_slots.data_ptr<int>(),
        .base_pool                          = dispatch_out.base_pool.data_ptr<int>(),
        .seen_per_substream                 = dispatch_out.seen_per_substream.data_ptr<int>(),
        .tile_id_to_expert                  = dispatch_out.tile_id_to_expert.data_ptr<int>(),
        .gbl_channel_prefix_matrix          = dispatch_out.gbl_channel_prefix_matrix.data_ptr<int>(),
        .rdma_channel_prefix_matrix         = dispatch_out.rdma_channel_prefix_matrix.data_ptr<int>(),
        .recv_rdma_rank_prefix_sum          = dispatch_out.recv_rdma_rank_prefix_sum.data_ptr<int>(),
        .recv_gbl_channel_prefix_matrix     = dispatch_out.recv_gbl_channel_prefix_matrix.data_ptr<int>(),
    };
    // Bump issued-count before launch (shared Buffer flags across intra/internode).
    ++dispatch_grads_issued_count;
    internode::DispatchGradsTileSignal tile_signal{
        .bwd_dispatch_arrival_count = bwd_dispatch_arrival_count.data_ptr<int>(),
        .pool_arrival_target        = dispatch_out.pool_arrival_target.data_ptr<int>(),
        .dispatch_seq               = dispatch_seq,
        .started_flag               = dispatch_grads_started_flag,
    };
    internode::DispatchGradsShape shape{
        .num_tokens   = num_tokens,
        .hidden_int4  = hidden_int4,
        .num_topk     = num_topk,
        .num_experts  = num_experts,
        .tile_m       = tile_m,
    };
    internode::DispatchEnv env{
        .rdma_buffer_ptr                     = rdma_buffer_ptr,
        .buffer_ptrs                         = buffer_ptrs_gpu,
        .rank                                = rank,
        .num_max_rdma_chunked_send_tokens    = config.num_max_rdma_chunked_send_tokens,
        .num_max_rdma_chunked_recv_tokens    = config.num_max_rdma_chunked_recv_tokens,
        .num_max_nvl_chunked_send_tokens     = config.num_max_nvl_chunked_send_tokens,
        .num_max_nvl_chunked_recv_tokens     = config.num_max_nvl_chunked_recv_tokens,
        .reader_prev_head                    = dispatch_reader_prev_head,
        .reader_prev_tail                    = dispatch_reader_prev_tail,
    };

    // Recorded before the main kernel launch — bwd orchestrator's compute
    // stream waits on this so kernel_y_bwd can run CONCURRENT with
    // dispatch_grads_main_kernel (per-tile bwd_dispatch_arrival_count
    // release-adds drive the actual streaming overlap). Mirror of fwd's
    // metadata_done_event.
    EventHandle grads_started_event(stream);

    internode::launch_dispatch_grads_main(io, routing, tile_signal, shape, env, num_rdma_ranks, num_channels, stream);

    return std::make_tuple(dL_do_pool, bwd_dispatch_arrival_count, grads_started_event);
#else
    EP_HOST_ASSERT(false and "internode_dispatch_grads requires NVSHMEM");
    return {};
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming-MoE combine (internode, pool layout). Single kernel launch:
// `internode::launch_combine_main` — NVL→RDMA→origin reduction with
// streaming gate at kNVLSender (gated by y_done_per_token). The
// sentinel-encoded `send_rdma_head` / `send_nvl_head` are produced by
// dispatch_main_kernel's sender / forwarder SM tails; stream-order
// causality (same communicate stream) guarantees visibility here.
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, c10::optional<torch::Tensor>> Buffer::internode_combine(
    const torch::Tensor& x,
    const torch::Tensor& per_slot_weights,
    const StreamingDispatchOutputs& dispatch_out,
    const torch::Tensor& y_done_per_token,
    int64_t combine_seq,
    int64_t combine_phase,
    bool is_fwd,
    const Config& config) {
#ifndef DISABLE_NVSHMEM
    pybind11::gil_scoped_release release;

    EP_HOST_ASSERT(num_rdma_ranks > 1 and "internode_combine requires multi-RDMA-rank world");
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(per_slot_weights.dim() == 1 and per_slot_weights.is_contiguous() and
                   per_slot_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(dispatch_out.send_rdma_head.dim() == 2 and dispatch_out.send_rdma_head.is_contiguous());
    EP_HOST_ASSERT(dispatch_out.send_nvl_head.dim() == 2 and dispatch_out.send_nvl_head.is_contiguous());
    EP_HOST_ASSERT(dispatch_out.recv_token_to_slots.dim() == 2 and dispatch_out.recv_token_to_slots.is_contiguous());
    EP_HOST_ASSERT(dispatch_out.send_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(dispatch_out.send_nvl_head.size(1) == NUM_MAX_NVL_PEERS);
    EP_HOST_ASSERT(y_done_per_token.dim() == 1 and y_done_per_token.is_contiguous() and
                   y_done_per_token.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(y_done_per_token.size(0) == x.size(0));

    int num_tokens = static_cast<int>(x.size(0));            // recv-side count (= dispatch's T_recv)
    int num_combined_tokens = static_cast<int>(dispatch_out.send_rdma_head.size(0));
    int hidden = static_cast<int>(x.size(1));
    int num_topk = static_cast<int>(dispatch_out.recv_token_to_slots.size(1));
    int hidden_int4 = static_cast<int>(hidden * x.element_size() / sizeof(int4));

    auto stream = at::cuda::getCurrentCUDAStream();

    // Combine output tensors. Fwd: no per-K weight output (kernel Y already
    // pre-multiplies pool_topk_weight per row); C++ returns `c10::nullopt`
    // and Python surfaces as `None`.
    auto recv_x = torch::empty({num_combined_tokens, hidden}, x.options());
    record_consumer_stream(recv_x, compute_stream_handle_);
    auto f32_opts = at::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    c10::optional<torch::Tensor> recv_topk_weights_out = is_fwd
        ? c10::nullopt
        : c10::optional<torch::Tensor>(torch::empty({num_combined_tokens, num_topk}, f32_opts));
    float* recv_topk_weights_ptr = is_fwd ? nullptr : recv_topk_weights_out->data_ptr<float>();
    if (recv_topk_weights_out.has_value())
        record_consumer_stream(*recv_topk_weights_out, compute_stream_handle_);

    // kNVLSender ships recv-side `x` back to source ranks → it needs the
    // *recv*-side per-(src_world, channel) prefix (`recv_gbl_channel_prefix_matrix`),
    // not the dispatch's send-side `gbl_channel_prefix_matrix`. Same bug class
    // as Phase D's intranode combine fix (see logbook 2026-05-04).
    internode::launch_combine_main(
        at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
        recv_x.data_ptr(), recv_topk_weights_ptr,
        x.data_ptr(), per_slot_weights.data_ptr<float>(),
        dispatch_out.recv_token_to_slots.data_ptr<int>(),
        dispatch_out.send_rdma_head.data_ptr<int>(),       // sentinel-encoded by dispatch_main_kernel tail
        dispatch_out.send_nvl_head.data_ptr<int>(),        // sentinel-encoded by dispatch_main_kernel tail
        dispatch_out.recv_src_meta.data_ptr(),
        dispatch_out.recv_rdma_channel_prefix_matrix.data_ptr<int>(),
        dispatch_out.recv_rdma_rank_prefix_sum.data_ptr<int>(),
        dispatch_out.recv_gbl_channel_prefix_matrix.data_ptr<int>(),
        y_done_per_token.data_ptr<int64_t>(),
        combine_seq,
        static_cast<int>(combine_phase),
        is_fwd,
        num_tokens, num_combined_tokens, hidden, num_topk,
        rdma_buffer_ptr_combine,
        config.num_max_rdma_chunked_send_tokens,
        config.num_max_rdma_chunked_recv_tokens,
        buffer_ptrs_gpu,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens,
        rank, num_ranks, stream, num_channels,
        combine_reader_prev_head, combine_reader_prev_tail);

    return {recv_x, recv_topk_weights_out};
#else
    EP_HOST_ASSERT(false and "internode_combine requires NVSHMEM");
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

}  // namespace stream_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "StreamEP: streaming-tile expert-parallel dispatch / combine";

    pybind11::class_<stream_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6,
             py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6,
             py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &stream_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &stream_ep::Config::get_rdma_buffer_size_hint);

    pybind11::class_<stream_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &stream_ep::EventHandle::current_stream_wait)
        .def("wait", &stream_ep::EventHandle::wait);

    pybind11::class_<stream_ep::StreamingDispatchOutputs>(m, "StreamingDispatchOutputs")
        .def_readonly("pool",                       &stream_ep::StreamingDispatchOutputs::pool)
        .def_readonly("pool_topk_weight",           &stream_ep::StreamingDispatchOutputs::pool_topk_weight)
        .def_readonly("pool_recv_token",            &stream_ep::StreamingDispatchOutputs::pool_recv_token)
        .def_readonly("pool_k_slot",                &stream_ep::StreamingDispatchOutputs::pool_k_slot)
        .def_readonly("send_head",                  &stream_ep::StreamingDispatchOutputs::send_head)
        .def_readonly("rank_prefix_matrix",         &stream_ep::StreamingDispatchOutputs::rank_prefix_matrix)
        .def_readonly("channel_prefix_matrix",      &stream_ep::StreamingDispatchOutputs::channel_prefix_matrix)
        .def_readonly("recv_channel_prefix_matrix", &stream_ep::StreamingDispatchOutputs::recv_channel_prefix_matrix)
        .def_readonly("expert_frequency",           &stream_ep::StreamingDispatchOutputs::expert_frequency)
        .def_readonly("expert_pool_block_offset",   &stream_ep::StreamingDispatchOutputs::expert_pool_block_offset)
        .def_readonly("base_pool",                  &stream_ep::StreamingDispatchOutputs::base_pool)
        .def_readonly("seen_per_substream",         &stream_ep::StreamingDispatchOutputs::seen_per_substream)
        .def_readonly("tile_id_to_expert",          &stream_ep::StreamingDispatchOutputs::tile_id_to_expert)
        .def_readonly("pool_arrival_target",        &stream_ep::StreamingDispatchOutputs::pool_arrival_target)
        .def_readonly("pool_arrival_count",         &stream_ep::StreamingDispatchOutputs::pool_arrival_count)
        .def_readonly("k_local_remaining",        &stream_ep::StreamingDispatchOutputs::k_local_remaining)
        .def_readonly("y_done_per_token",     &stream_ep::StreamingDispatchOutputs::y_done_per_token)
        .def_readonly("o",                          &stream_ep::StreamingDispatchOutputs::o)
        .def("release_o",                           &stream_ep::StreamingDispatchOutputs::release_o)
        .def_readonly("recv_token_to_slots",        &stream_ep::StreamingDispatchOutputs::recv_token_to_slots)
        .def_readonly("k_local_total",                   &stream_ep::StreamingDispatchOutputs::k_local_total)
        .def_readonly("total_tiles",                     &stream_ep::StreamingDispatchOutputs::total_tiles)
        .def_readonly("metadata_done_event",             &stream_ep::StreamingDispatchOutputs::metadata_done_event)
        // Internode-only combine plumbing (empty for intranode):
        .def_readonly("rdma_channel_prefix_matrix",      &stream_ep::StreamingDispatchOutputs::rdma_channel_prefix_matrix)
        .def_readonly("recv_rdma_rank_prefix_sum",       &stream_ep::StreamingDispatchOutputs::recv_rdma_rank_prefix_sum)
        .def_readonly("gbl_channel_prefix_matrix",       &stream_ep::StreamingDispatchOutputs::gbl_channel_prefix_matrix)
        .def_readonly("recv_gbl_rank_prefix_sum",        &stream_ep::StreamingDispatchOutputs::recv_gbl_rank_prefix_sum)
        .def_readonly("recv_rdma_channel_prefix_matrix", &stream_ep::StreamingDispatchOutputs::recv_rdma_channel_prefix_matrix)
        .def_readonly("recv_gbl_channel_prefix_matrix",  &stream_ep::StreamingDispatchOutputs::recv_gbl_channel_prefix_matrix)
        .def_readonly("send_rdma_head",                  &stream_ep::StreamingDispatchOutputs::send_rdma_head)
        .def_readonly("send_nvl_head",                   &stream_ep::StreamingDispatchOutputs::send_nvl_head)
        .def_readonly("recv_src_meta",                   &stream_ep::StreamingDispatchOutputs::recv_src_meta);

    pybind11::class_<stream_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool>())
        .def("is_available", &stream_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &stream_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &stream_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &stream_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &stream_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &stream_ep::Buffer::get_local_ipc_handle)
        .def("get_local_nvshmem_unique_id", &stream_ep::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &stream_ep::Buffer::get_local_buffer_tensor)
        .def("sync", &stream_ep::Buffer::sync)
        .def("destroy", &stream_ep::Buffer::destroy)
        .def("wait_dispatch_main_started", &stream_ep::Buffer::wait_dispatch_main_started)
        .def("wait_dispatch_grads_started", &stream_ep::Buffer::wait_dispatch_grads_started)
        .def("wait_kernel_y_started", &stream_ep::Buffer::wait_kernel_y_started)
        .def("wait_kernel_a_bwd_started", &stream_ep::Buffer::wait_kernel_a_bwd_started)
        .def("bump_kernel_y_issued", &stream_ep::Buffer::bump_kernel_y_issued)
        .def("bump_kernel_a_bwd_issued", &stream_ep::Buffer::bump_kernel_a_bwd_issued)
        .def("kernel_y_started_flag_tensor", &stream_ep::Buffer::kernel_y_started_flag_tensor)
        .def("kernel_a_bwd_started_flag_tensor", &stream_ep::Buffer::kernel_a_bwd_started_flag_tensor)
        .def("set_compute_stream_handle", &stream_ep::Buffer::set_compute_stream_handle)
        .def("intranode_dispatch", &stream_ep::Buffer::intranode_dispatch)
        .def("intranode_dispatch_grads", &stream_ep::Buffer::intranode_dispatch_grads)
        .def("intranode_combine", &stream_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &stream_ep::Buffer::internode_dispatch)
        .def("internode_dispatch_grads", &stream_ep::Buffer::internode_dispatch_grads)
        .def("internode_combine", &stream_ep::Buffer::internode_combine);

    m.def("is_sm90_compiled", stream_ep::is_sm90_compiled);
    m.attr("topk_idx_t") =
        py::reinterpret_borrow<py::object>((PyObject*)torch::getTHPDtype(c10::CppTypeToScalarType<stream_ep::topk_idx_t>::value));
}
