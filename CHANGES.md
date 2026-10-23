# Changes from upstream DeepEP

StreamEP forked [DeepSeek's DeepEP](https://github.com/deepseek-ai/DeepEP) at upstream commit `567632d` and is not intended to be upstreamed. Upstream serves drop-in async dispatch with a `Buffer`-owned comm stream; StreamEP serves one integrated, streaming MoE layer. When a cleaner streaming pipeline conflicts with staying close to upstream, the pipeline wins.

This file records the design-level divergence. Mechanism details live in the code and its docstrings.

## Added

- **Pool-layout dispatch.** The receiver writes each `(token, k)` pair into an expert-major, tile-padded pool that feeds the grouped GEMM directly, replacing upstream's token-major output plus a separate permute.
- **Per-tile streaming, forward and backward.** Dispatch signals pool tiles as they fill and the compute kernels consume them as they become ready; combine sends each token once its compute finishes. `dispatch_grads` / `combine_grads` mirror this in the backward pass.
- **Integrated MoE layer** (`stream_ep/stream_moe/`). The `stream_moe_func` autograd function and its four CuTeDSL grouped GEMMs on Quack, with streaming tile schedulers and epilogues, plus expert dropout and activation-checkpoint levels.
- **Routing metadata inside dispatch.** Counts, offsets and per-tile arrival targets are computed by dispatch itself, replacing upstream's separate layout call.
- **Shape-driven tuning.** GEMM tiles and SM counts are picked from the weight shapes and world size, with `TileConfig` overrides.

## Changed

- **Streams are caller-owned.** `Buffer` no longer owns a comm stream; the layer runs on caller-supplied compute and communicate streams.
- **One host sync per layer.** A single poll sizes the dropless pool allocation. Everything else on the streaming path stays on the GPU.
- **The ring protocol** started from upstream's and has been reworked for streaming and for safe reuse across iterations.

## Removed

- The FP8 dispatch path (bf16 only).
- The low-latency inference kernels (high-throughput training only).
- `layout.cu` / `Buffer.get_dispatch_layout`, superseded by in-dispatch metadata.
- The CMake build; `setup.py` inside the pixi env is the only build path.
