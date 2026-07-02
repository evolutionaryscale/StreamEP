#pragma once

#include "kernels/api.cuh"
#include "kernels/exception.cuh"

namespace stream_ep {

template <typename dtype_t>
dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

template <typename dtype_t>
dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;
}

struct Config {
    int num_sms;
    int num_max_nvl_chunked_send_tokens;
    int num_max_nvl_chunked_recv_tokens;
    int num_max_rdma_chunked_send_tokens;
    int num_max_rdma_chunked_recv_tokens;

    Config(int num_sms,
           int num_max_nvl_chunked_send_tokens,
           int num_max_nvl_chunked_recv_tokens,
           int num_max_rdma_chunked_send_tokens,
           int num_max_rdma_chunked_recv_tokens)
        : num_sms(num_sms),
          num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
          num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
          num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
          num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {
        EP_HOST_ASSERT(num_sms >= 0);
        EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens > 0 and num_max_nvl_chunked_recv_tokens > 0);
        EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens);
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens > 0 and num_max_rdma_chunked_recv_tokens > 0);

        // Ceil up RDMA buffer size
        this->num_max_rdma_chunked_recv_tokens = align_up<int>(num_max_rdma_chunked_recv_tokens, num_max_rdma_chunked_send_tokens);
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens < num_max_rdma_chunked_recv_tokens);
        // NOTES: this assertion is related to RDMA lazy head update, we must ensure senders always have space to push
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens <= num_max_rdma_chunked_recv_tokens / 2);
    }

    size_t get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
        // Below are some assumptions
        // TODO: add assertions
        constexpr int kNumMaxTopK = 128;
        EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(num_ranks <= NUM_MAX_NVL_PEERS or num_sms % 2 == 0);
        const auto num_rdma_ranks = std::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
        const int num_channels = num_sms / 2;
        const int hidden_int4 = static_cast<int>(hidden_bytes / sizeof(int4));

        // Disjoint NVL regions: dispatch's sub-buffer chain occupies bytes
        // [0, D), combine's occupies [D, D + C). Combine kernels offset their
        // base pointers by `get_dispatch_nvl_region_bytes(hidden_int4,
        // num_topk_actual, ...)` at launch time (computed from kernel args).
        // The host upper-bounds with `kNumMaxTopK` so the allocation fits
        // any runtime `num_topk` without reallocation. Disjointness is
        // load-bearing: a shared layout would let iter-N combine writes
        // alias iter-N+1 dispatch read addresses via per-channel stride
        // drift between the two kernels.
        size_t num_bytes = 0;
        num_bytes += internode::get_dispatch_nvl_region_bytes(
            hidden_int4, kNumMaxTopK, num_max_nvl_chunked_recv_tokens,
            num_channels, num_rdma_ranks);
        num_bytes += internode::get_combine_nvl_region_bytes(
            hidden_int4, kNumMaxTopK, num_max_nvl_chunked_recv_tokens,
            num_channels, num_rdma_ranks);
        // NOTE: upstream DeepEP appends a per-slot fp8-scales block here
        // (num_channels * num_nvl_ranks * recv_tokens * kNumMaxScales floats).
        // stream_ep never quantizes dispatched tokens, so no kernel indexes it
        // (see get_dispatch/combine_nvl_region_bytes in api.cuh -- the per-slot
        // layout has no scales sub-buffer, and the block sat past
        // dispatch_region + combine_region where neither kernel reads). Dropped
        // to reclaim the dead padding; do NOT re-add for "upstream compat".
        num_bytes = ((num_bytes + 127) / 128) * 128;
        return num_bytes;
    }

    size_t get_rdma_buffer_size_hint(int64_t hidden_bytes, int num_ranks) const {
#ifndef DISABLE_NVSHMEM
        // Legacy mode
        if (num_ranks <= NUM_MAX_NVL_PEERS)
            return 0;

        // Below are some assumptions
        // TODO: add assertions
        constexpr int kNumMaxTopK = 128;
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(num_sms % 2 == 0);
        const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
        const int num_channels = num_sms / 2;
        const int hidden_int4 = static_cast<int>(hidden_bytes / sizeof(int4));

        // Disjoint RDMA regions: dispatch's sub-buffer chain occupies bytes
        // [0, D_rdma), combine's occupies [D_rdma, D_rdma + C_rdma).
        // combine_main_kernel offsets its RDMA SymBuffer bases by
        // `get_dispatch_rdma_region_bytes(hidden_int4, num_topk_actual, ...)`
        // at launch time (computed from kernel args). The host upper-bounds
        // with `kNumMaxTopK` so the allocation fits any runtime `num_topk`
        // without reallocation. Disjointness is load-bearing for the same
        // reason as the NVL side — see `get_dispatch_rdma_region_bytes` in
        // api.cuh and the analogous NVL block in `get_nvl_buffer_size_hint`
        // above.
        size_t num_bytes = 0;
        num_bytes += internode::get_dispatch_rdma_region_bytes(
            hidden_int4, kNumMaxTopK, num_max_rdma_chunked_recv_tokens,
            num_channels, num_rdma_ranks);
        num_bytes += internode::get_combine_rdma_region_bytes(
            hidden_int4, kNumMaxTopK, num_max_rdma_chunked_recv_tokens,
            num_channels, num_rdma_ranks);
        // NOTE: dropped the unused per-slot fp8-scales padding block here; see
        // the matching note in get_nvl_buffer_size_hint above. stream_ep never
        // quantizes dispatched tokens, and the combine SymBuffer offsets past
        // dispatch_region + combine_region (rdma_buffer_ptr_combine =
        // rdma_buffer_ptr + num_rdma_bytes), so no kernel indexed the padding.
        num_bytes = ((num_bytes + 127) / 128) * 128;
        return num_bytes;
#else
        EP_HOST_ASSERT(false and "NVSHMEM is disable during compilation");
#endif
    }
};

}  // namespace stream_ep
