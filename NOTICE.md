# NOTICE

StreamEP is a fork of [DeepSeek's DeepEP](https://github.com/deepseek-ai/DeepEP) at commit `567632d` ("Update README for mori-EP branch and mori community fork (#578)"), MIT-licensed.

Diverges from upstream in:

- **Streaming-tile dispatch / combine signaling.** Per-tile `tile_ready` release-stamps fire as expert-major substreams drain, so a consumer compute kernel can begin processing tiles while the dispatch receiver is still landing later tokens. Combine consumes a per-recv-token gate (`y_done_per_token`) for symmetric per-token streaming on the return path.
- **Integrated streaming-MoE pipeline** at `stream_ep/stream_moe/` (orchestrator, kernel A / Y forward, kernel A_bwd / Y_bwd backward, EpiOps, `StreamingTileScheduler`, PTX helpers, harnesses, tests). Built around a [Quack](https://github.com/Dao-AILab/quack) grouped-GEMM backbone with `cute.lens_k` for variable-K tiles and an `mPaddingMask` predicate so the GEMM's TMA OOB-zero-fill handles dispatch-padding rows at zero compute cost.
- **Pool-layout dispatch receiver.** Each landed `(token, k)` pair lands at a stable per-expert pool slot (expert-major, BLOCK_M-padded), so `pool` flows into kernel A as a single `[TK_padded, hidden]` tensor with predictable per-expert stride.
- **Caller-managed streams.** No `comm_stream` ownership inside `Buffer`; one host sync per layer (poll on `num_recv` / `total_tiles`) gates allocator sizing.
- **FP8 dispatch path removed** (~200 LoC) — bf16 only.
- **Low-latency inference kernels removed** (`internode_ll.cu` + `Buffer::low_latency_*` methods, ~1700 LoC) — high-throughput training only.
- **`layout.cu` / `Buffer::get_dispatch_layout` removed** — the streaming dispatch synthesizes its own routing metadata via `streaming_dispatch_metadata` (one launch).

The dispatch / combine ring protocol (channel send/recv buffers, NVL barrier scheme, RDMA notifier mechanics) is inherited from upstream DeepEP unchanged.

The unmodified MIT `LICENSE` file from upstream is preserved as required.
