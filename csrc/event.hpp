#include <ATen/cuda/CUDAContext.h>

#include <memory>

#include "kernels/exception.cuh"

namespace stream_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(at::cuda::getCurrentCUDAStream());
    }

    explicit EventHandle(const at::cuda::CUDAStream& stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const { at::cuda::getCurrentCUDAStream().unwrap().wait(*event); }

    // Make `stream` wait on this event. Mirrors the `event.wait(stream)` shape
    // of `torch.cuda.Event`, so callers can write
    // `compute_a_stream.wait_event(metadata_done_event)` — `Stream.wait_event`
    // is implemented in PyTorch as `event.wait(self)`, so any object exposing
    // a `.wait(stream)` method is duck-compatible.
    void wait(torch::Stream stream) const {
        at::cuda::CUDAStream cuda_stream(stream);
        cuda_stream.unwrap().wait(*event);
    }
};

}  // namespace stream_ep
