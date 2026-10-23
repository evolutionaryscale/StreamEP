# StreamEP

Expert-parallel dispatch / combine plus an integrated streaming-MoE layer for MoE training on H100 (SM90), intranode over NVLink and internode over RDMA. Fork of [DeepSeek's DeepEP](https://github.com/deepseek-ai/DeepEP); see [`CHANGES.md`](CHANGES.md) for how it diverges and [`NOTICE.md`](NOTICE.md) for attribution.

It is an internal component of evoscale's MoE training stack, which consumes it as a release wheel.

## How it works

- **Pool-layout dispatch.** Received `(token, k)` pairs land directly in an expert-major, tile-padded pool, so dispatch's output is the grouped GEMM's input with no separate permute.
- **Per-tile streaming.** Dispatch signals each pool tile as it fills, and the compute kernels claim ready tiles while later tokens are still arriving. Combine starts sending a token as soon as its compute is done. The backward pass mirrors both overlaps.
- **Caller-owned streams.** A layer runs on a compute stream and a communicate stream supplied by the caller. The only host sync per layer is one poll that sizes the dropless pool allocation.
- **CuTeDSL compute kernels.** The four grouped GEMMs (forward A / Y, backward A / Y) are built on [Quack](https://github.com/Dao-AILab/quack) with streaming-specific schedulers and epilogues.

```python
from stream_ep import Buffer
from stream_ep.stream_moe.stream_moe import make_streams, stream_moe_func

buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
streams = make_streams()
out = stream_moe_func(
    buffer,
    x,
    topk_idx,
    topk_weights,
    is_token_in_rank,
    w1_local,
    w2_local,
    streams=streams,
    num_experts=num_experts,
)
```

GEMM tiles and SM counts are picked from the weight shapes and world size; `TileConfig` overrides them. `stream_moe_func` is eager-only, so wrap the call in `@torch.compiler.disable` inside a compiled model.

## Layout

- `stream_ep/` — the `Buffer` class over the `stream_ep_cpp` extension built from `csrc/`.
- `stream_ep/stream_moe/` — the MoE layer, its kernels, and the profiling / validation harnesses. The package docstring is the code map.
- `tests/` — multi-rank torchrun scripts. `stream_ep/stream_moe/tests/` — single-process kernel pytest.
- `scripts/` — release wheel build and checks.

## Development

`pixi.toml` defines three environments:

- `default` — the wheel build env, pinned to the versions evoscale runs (python 3.12, torch 2.11.0+cu130, nvcc 13.3, gcc 14.3, `nvidia-nvshmem-cu13==3.4.5`). No GPU needed.
- `test` — `default` plus the runtime deps of the MoE kernels (evoscale's Quack fork, cutlass-dsl, CUDA 13.3 runtime), with StreamEP itself installed editable.
- `lint` — pre-commit only.

```bash
pixi run -e test build-ext                                # in-place build after .cu / .cpp / .cuh edits
pixi run -e test test-kernels                             # kernel pytest, 1 GPU
pixi run -e test test-dist tests/test_dispatch.py         # one torchrun script, 8 GPUs by default
pixi run -e test test-dist tests/test_dispatch.py 2       # ... on 2 GPUs
pixi run -e lint lint                                     # pre-commit hooks; `pixi run -e lint pre-commit install` to run them on commit
```

Internode scripts (`tests/test_*_internode.py`) need two or more nodes. Profiling and multi-iteration validation entry points live in `stream_ep/stream_moe/` (`profile_pipeline`, `validate_multi_iter`, `ncu_microbench`).

## Release

Bump `version` in `pyproject.toml` and merge to `main`. `.github/workflows/wheel.yml` builds and checks the wheel on every PR; on a push to `main` whose version has no release yet, it publishes the wheel as the single asset of release `v<version>`. Release assets are never replaced, because uv caches wheel URLs as immutable; a fix ships as a new version.

To build locally: `pixi run build-wheel && pixi run check-wheel`.

## License

MIT. The unmodified `LICENSE` file from upstream DeepEP is preserved.
