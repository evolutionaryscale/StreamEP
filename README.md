# StreamEP

A streaming-tile expert-parallel dispatch / combine library for intranode MoE training on H100 NVLink. Fork of [DeepSeek's DeepEP](https://github.com/deepseek-ai/DeepEP) — see [`NOTICE.md`](NOTICE.md) for fork details.

The headline feature is **per-tile streaming**: dispatch fires release-stamps on expert-major BLOCK_M tiles as soon as their pool slots fill, so a consumer GEMM can spin on `tile_ready[tile_id]` and begin processing while later tokens are still landing. The complementary per-recv-token gate on combine (`y_done_per_token`) lets the combine sender ship the first packet as soon as that recv-token's compute drains, instead of waiting for the whole compute kernel.

## What's in the box

- `stream_ep` — the C++ extension + `Buffer` Python class. Exposes `Buffer.dispatch` (pool-layout receiver, returns a `StreamingHandle`), `Buffer.combine` (consumes the per-recv-token gate), `Buffer.dispatch_grads` / `combine_grads` for the backward symmetry. Internode dispatch / combine inherited from upstream DeepEP.
- `stream_ep.stream_moe` — a complete streaming-MoE pipeline built on top of the buffer:
  - `kernel_a` / `kernel_a_bwd` — first grouped GEMM (`gate * up` SwiGLU, `kFlatten` persistent) with a per-tile `TileReadyRelease` EpiOp. Built on [Quack](https://github.com/Dao-AILab/quack)'s `GemmGatedMixin`.
  - `kernel_y` / `kernel_y_bwd` — second grouped GEMM (`down`) with an `AtomicScatterStore` EpiOp that does per-warp coalesced bf16 atomic-scatter into the output buffer. Spins on `a_ready[tile_id]` from kernel A.
  - `tile_scheduler.StreamingTileScheduler` — Quack `TileScheduler` subclass with a per-tile-ready spin acquire (`tile_ready` for kernel A, `a_ready` for kernel Y) and `cute.lens_k`-based variable-K tile sizes that ride the GEMM's TMA OOB-zero-fill to handle dispatch-padding rows at zero compute cost.
  - `epi_ops` — composable Quack EpiOps (`TileReadyRelease`, `AtomicScatterStore`).
  - `ptx_helpers` — system-scope `st_release_sys_global`, `ld_acquire_sys_global`, `red_add_bf16x2`, etc. Used for cross-stream signaling without Torch event overhead.
  - `stream_moe.StreamMoEFunc` — the autograd boundary that orchestrates dispatch → kernel A → kernel Y → combine on four caller-owned streams (and dispatch_grads → kernel_y_bwd → kernel_a_bwd → combine_grads on the same four streams in backward, with dW1 / dW2 grouped GEMMs landing on a dedicated `grads` stream so they don't contend with the streaming-kernel critical path).

## Streaming dispatch in one paragraph

Dispatch's receiver writes each landed `(token, k)` pair into a stable **pool slot**: pool is laid out expert-major and BLOCK_M-padded, with per-expert blocks at `expert_pool_block_offset[e] * tile_m` rows. Pool slots within an expert can be filled in any order (sender substreams race to fill them), but every slot belongs to exactly one expert and one BLOCK_M tile. As each expert-major substream finishes draining its share of pool slots, dispatch's Pass 2 atomic-adds `pool_arrival_count[tile_id]` and, when it hits the pre-computed `pool_arrival_target[tile_id]`, release-stores `dispatch_seq` into `tile_ready[tile_id]`. A consumer kernel spins on `ld_acquire_sys_global(tile_ready[tile_id]) >= dispatch_seq` from a different stream and proceeds. Padding rows in the pool are never read — `pool_recv_token >= 0` predicates them out, and Quack's `cute.lens_k` bounds the K-tile so the TMA's OOB-zero-fill handles padding columns at zero compute cost.

The dispatch / combine ring protocol underneath (channel send/recv buffers, NVL barrier scheme, RDMA notifier mechanics) is inherited from upstream DeepEP unchanged. See the upstream repo for those details.

## What was removed from upstream

- FP8 dispatch path (~200 LoC). bf16 only.
- Low-latency inference kernels (`internode_ll.cu` + `Buffer::low_latency_*` methods, ~1700 LoC). High-throughput training only.
- `csrc/kernels/layout.cu` and `Buffer::get_dispatch_layout` (~200 LoC). The streaming dispatch synthesizes its own routing metadata via `streaming_dispatch_metadata` in a single kernel launch.
- CMake build path (`csrc/CMakeLists.txt`, `csrc/kernels/CMakeLists.txt`). Builds via `setup.py` driven by [`build_stream_ep.sh`](#build) inside a pixi env.

## Build

StreamEP is built editable inside a consumer pixi environment that already provides torch (cu128), NVSHMEM 3.5.19, CUDA 12.8, and the `nvidia/*` pip wheels with the dev headers (cusparse / cublas / cusolver) the build needs:

```bash
pixi run --manifest-path /path/to/consumer/pixi.toml bash build_stream_ep.sh --clean
```

Produces an editable install of `stream_ep` (Python package) backed by `stream_ep_cpp` (C++ extension). Coexists with upstream `deep_ep` in the same env if both are installed — different module names, different `.so` filenames, so neither shadows the other.

## Tests

DeepEP-style stand-alone scripts (no pytest):

```bash
# Cross-rank streaming-dispatch correctness (8 GPU intranode):
pixi run --manifest-path .../pixi.toml python tests/test_dispatch.py
pixi run --manifest-path .../pixi.toml python tests/test_combine.py
pixi run --manifest-path .../pixi.toml python tests/test_dispatch_grads.py
pixi run --manifest-path .../pixi.toml python tests/test_combine_grads.py
```

The streaming-MoE kernel tests at `stream_ep/stream_moe/tests/` are pytest-driven (single-process, small shapes — they exercise the kernel logic, not multi-rank dispatch):

```bash
pixi run --manifest-path .../pixi.toml pytest stream_ep/stream_moe/tests/ -v
```

## License

MIT. The unmodified `LICENSE` file from upstream DeepEP is preserved as required by the MIT license.
