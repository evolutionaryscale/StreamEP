# StreamEP

Fork of [DeepSeek's DeepEP](https://github.com/deepseek-ai/DeepEP) (MIT) specialized for streaming-tile MoE dispatch / combine on intranode H100 NVLink. Diverges from upstream:

- Dispatch is reshaped into a **pool layout** (expert-major, BLOCK_M-padded) so the receiver writes each landed `(token, k)` pair into a stable pool slot. Per-tile `tile_ready` release-stamps fire as expert-major substreams drain, enabling the consumer kernel A to overlap with the tail of dispatch.
- Combine consumes a per-recv-token gate (`compute_done_per_token`) so the sender's per-warp loop can ship the first packet as soon as that recv-token's compute drains, instead of waiting for the whole compute kernel.
- Caller-managed streams (no `comm_stream` ownership inside the buffer); a single sync per layer (host poll on `num_recv` / `total_tiles`) gates allocator sizing.
- FP8 dispatch path, low-latency inference kernels, and `get_dispatch_layout` removed (none used by the streaming-MoE pipeline).

A full design write-up lives in the consumer's `design.md`. This README is a stub; the migration plan rewrites it in Phase 5.

## Build

Built inside the consumer's pixi environment:

```bash
pixi run --manifest-path /path/to/consumer/pixi.toml bash build_stream_ep.sh --clean
```

Produces an editable install of `stream_ep` (Python package) backed by the `stream_ep_cpp` C++ extension. Coexists with upstream `deep_ep` in the same env (different module names, different `.so` filenames).

## License

MIT. The `LICENSE` file is preserved unmodified from upstream DeepEP.
