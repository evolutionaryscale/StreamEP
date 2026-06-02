# Changes from upstream DeepEP

StreamEP is a fork of [DeepSeek's DeepEP](https://github.com/deepseek-ai/DeepEP) at upstream commit `567632d`. This document is the high-level rationale for the divergence; for the canonical attribution + license preservation see [`NOTICE.md`](NOTICE.md).

The fork is **not** intended to be upstreamable. DeepEP's surface — drop-in async dispatch with `comm_stream` ownership inside `Buffer` — serves a different audience than ours (caller-managed streams, per-tile signals, integrated MoE compute). When choosing between cleaner streaming pipeline vs. closer to upstream DeepEP, the pipeline wins.

## What's new

### Per-tile streaming signaling

The signature feature. In upstream DeepEP, the compute kernel can't start until `Buffer.dispatch` retires for *all* tokens — the comm-to-compute handoff is a single edge. With per-tile streaming, the compute kernel starts processing as soon as the first tile is ready while the rest of dispatch is still landing later tokens. Dispatch ↔ kernel A becomes a producer-consumer pipeline with tile-granularity overlap rather than two serial phases.

The mechanism: the dispatch receiver writes each landed `(token, k)` pair into a stable **pool slot** (expert-major, BLOCK_M-padded). As each expert-major substream finishes draining its share of pool slots, dispatch's Pass 2 atomic-adds `pool_arrival_count[tile_id]` and, when the count hits `pool_arrival_target[tile_id]`, release-stores `dispatch_seq` into `tile_ready[tile_id]`. The consumer compute kernel on a different stream spins on `ld_acquire_sys_global(tile_ready[tile_id]) >= dispatch_seq` per-tile and proceeds.

Combine has the symmetric per-recv-token gate (`y_done_per_token`): the sender ships the first packet as soon as that recv-token's compute drains, instead of waiting for the whole compute kernel. Kernel Y ↔ combine becomes a producer-consumer pipeline with per-recv-token granularity.

### Pool-layout dispatch receiver

Pool is laid out **expert-major** and **BLOCK_M-padded**, with per-expert blocks at `expert_pool_block_offset[e] * tile_m` rows. Pool slots within an expert can be filled in any order (sender substreams race to fill them), but every slot belongs to exactly one expert and exactly one BLOCK_M tile. The pool tensor flows into the consumer GEMM as a single `[TK_padded, hidden]` tensor with predictable per-expert stride. Padding rows are never read — `pool_recv_token >= 0` predicates them out, and Quack's `cute.lens_k` bounds the K-tile so the GEMM's TMA OOB-zero-fill handles padding columns at zero compute cost.

Upstream DeepEP returns `(recv_x, recv_topk_idx, recv_topk_weights, handle, event)` with token-major layout; consumers must run a separate permute (DeepGEMM `permute_ck` or moe_permute_topK) to land in an expert-major layout suitable for grouped GEMM. The pool layout collapses the permute into dispatch.

### Integrated streaming-MoE pipeline

Everything under `stream_ep/stream_moe/` is new — a complete MoE forward+backward pipeline built on the streaming buffer:

- **`kernel_a` / `kernel_a_bwd`** — first grouped GEMM (`gate * up` SwiGLU, `kFlatten` persistent). Built on Quack's `GemmGatedMixin` with a `TileReadyRelease` EpiOp on the forward (release-stamps tile completion to kernel Y) and a symmetric `wait_kernel_a_bwd_started` gate on the backward.
- **`kernel_y` / `kernel_y_bwd`** — second grouped GEMM (`down`) with an `AtomicScatterStore` EpiOp that does per-warp coalesced `red.global.add.bf16x2` scatter into the output buffer. Spins on `a_ready[tile_id]` from kernel A. The atomic-scatter avoids a separate scatter pass (no trash row, no scatter kernel launch) — sparse contributions land directly at their final addresses.
- **`StreamingTileScheduler`** — Quack `TileScheduler` subclass with a per-tile-ready spin acquire and `cute.lens_k`-based variable-K tile sizes that ride the GEMM's TMA OOB-zero-fill to handle dispatch-padding rows at zero compute cost. Shared between kernel A and kernel Y.
- **`stream_moe_func`** — public layer entry point + `StreamMoEFunc` autograd boundary. Orchestrates dispatch → kernel A → kernel Y → combine on two caller-owned streams (`compute` and `communicate`) and the symmetric backward path.
- **`ptx_helpers`** — system-scope `st_release_sys_global`, `ld_acquire_sys_global`, `red_add_bf16x2`, `pack_bf16x2`. Used for cross-stream signaling without Torch event overhead.

### Two-stream layout and cross-stream synchronization

The layer runs on two caller-owned streams:

- **`communicate`** — forward `dispatch` + forward `combine`; backward `dispatch_grads` + backward `combine_grads`. Same-stream FIFO orders combine after dispatch (and combine_grads after dispatch_grads) — no cross-stream serialization is needed between the comm halves.
- **`compute`** — forward kernel A + kernel Y; backward kernel Y_bwd + kernel A_bwd + dW1 + dW2 grouped GEMMs. Same-stream FIFO covers all intra-compute handoffs.

Within a layer the two streams overlap; across layers they serialize via layer-start `stream.wait_stream(caller)` and layer-end `caller.wait_stream(stream)` back-edges. Real overlap windows:

- **fwd dispatch ↔ kernel A**: dispatch's persistent CTAs drain on the copy engines as kernel A's 132-CTA grid lands. Two GPU-front-end gates (`metadata_done` event + `wait_dispatch_main_started` count-bump) hold kernel A's launch until dispatch_main's block 0 is co-resident on an SM — without this, kernel A's 132 CTAs can grab all the SMs before dispatch_main lands and starve it.
- **fwd kernel Y ↔ combine**: combine's sender warps spin on `y_done_per_token[r] >= dispatch_seq`. The `wait_kernel_y_started` gate holds combine's launch until kernel Y's first CTA is co-resident.
- **bwd dispatch_grads ↔ kernel Y_bwd** and **bwd kernel A_bwd ↔ combine_grads**: mirror of the fwd gates.
- **bwd dW ↔ combine_grads**: dW1/dW2 fill the SMs that combine_grads's CTAs leave idle on per-token gates.

### Backward path

`kernel_a_bwd` and `kernel_y_bwd` mirror the fwd kernel structure with new computations:

- **`kernel_y_bwd`** computes the gradient through the SwiGLU + the (gate, up) matmul. Streams behind `dispatch_grads` like fwd kernel A streams behind dispatch. Emits `dL/dswiglu_in` and `dL/dpostact` to drive dW2 (the down-proj GEMM in transpose).
- **`kernel_a_bwd`** computes the gradient back to `dL/dx` and produces a `dL/dweight` flat fp32 buffer via a `ColVecReduceAtomic` EpiOp (atomic-add into a global accumulator, same pattern as fwd kernel Y's atomic scatter). Streams behind kernel Y_bwd; combine_grads streams behind kernel A_bwd.
- **dW1 / dW2** — two additional grouped GEMMs launched on the compute stream after kernel A_bwd retires. They can't overlap with kernel A_bwd (both saturate 132 SMs) but they DO overlap with combine_grads on the communicate stream, which is in its NVLink-bound sender phase by then.

### Single-launch dispatch metadata

The streaming dispatch synthesizes its own routing metadata (rank counts, channel prefixes, expert frequency offsets, per-tile arrival targets) in a single `streaming_dispatch_metadata` kernel launch, replacing upstream's separate `get_dispatch_layout` call.

### Caller-managed streams

`Buffer` does not own a `comm_stream`. The orchestrator at the autograd boundary supplies two caller-owned streams (`compute` and `communicate`) via a `StreamHolder`. One host sync per layer (poll on `num_recv` / `total_tiles` to size pool allocations) is the floor for dropless variable-routing dispatch — every per-dispatch tensor is freshly allocated, no buffer caching across iters (the freshness invariant is load-bearing for cross-dispatch correctness).

### Internal auto-tuning

`stream_moe_func`'s public surface has no tile / cluster / pingpong / swizzle / num_sms knobs. The 19 grouped-GEMM tuning fields are picked internally from `w1_local.shape` via `_pick_tile_config(I, H)`:

- Prefers bench-tuned defaults (`tile_n_a=192`, `tile_n_y=256`, `tile_n_y_bwd=192`, `tile_n_a_bwd=256`) where they satisfy the kernel's tile_n divisibility constraint.
- Otherwise substitutes the largest power-of-2 ≤ 256 that divides the constraint (e.g., for `I=H=2048`, the defaults all fit; for `I=256`, `tile_n_a` falls back to 128 since 2*256=512 isn't divisible by 192).

`Buffer.num_sms` auto-picks from world size in `Buffer.__init__`: 80 SMs at ≤2 nodes (NVL-dominated, more parallel channels wins), 64 SMs at ≥3 nodes (RDMA tail latency budget rewards fewer larger chunks). Both bench-sweep-tuned. Explicit override is available via `Buffer.set_num_sms`.

The kernel-level wrappers (`streaming_moe_a`, `streaming_moe_y`, ...) still expose all tile/SM args directly — that's the sweep-tuning entry point used by `profile_pipeline.py`.

## What's removed

- **FP8 dispatch path** (~200 LoC). bf16 only.
- **Low-latency inference kernels** (`internode_ll.cu` + `Buffer::low_latency_*` methods, ~1700 LoC). High-throughput training only.
- **`csrc/kernels/layout.cu` and `Buffer::get_dispatch_layout`** (~200 LoC). Replaced by the single-launch `streaming_dispatch_metadata`.
- **CMake build path** (`csrc/CMakeLists.txt`, `csrc/kernels/CMakeLists.txt`). Builds via `setup.py` inside a pixi env that provides torch + the `nvidia/*` pip wheels with dev headers (cusparse / cublas / cusolver).

## What's preserved from upstream

The dispatch / combine **ring protocol** (channel send/recv buffers, NVL barrier scheme, RDMA notifier mechanics) is inherited from upstream DeepEP unchanged. We did not touch:

- Intranode NVL ring sender/receiver coordination (channel-prefix scan, per-rank barrier slots).
- Internode RDMA dispatch / combine: IBGDA notifier mechanics, RDMA → NVL forwarder, NVL → RDMA receiver. (A source-level NVSHMEM v1/v2 device-state layout fix is templated into `ibgda_get_rc` — same fix ebetica's DeepEP fork applies in PR #564 — so we build against unpatched upstream NVSHMEM 3.5.19.)
- The bulk of `csrc/kernels/utils.cuh` (warp-level utilities, `cp.async.bulk` helpers).

## API contract

Public surface is intentionally narrow:

```python
from stream_ep import Buffer
from stream_ep.stream_moe import StreamHolder, make_streams, stream_moe_func

buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)  # num_sms auto-picked
streams = make_streams(device)

# One MoE forward layer (dispatch + kernel A + kernel Y + combine).
out = stream_moe_func(
    buffer, x, topk_idx, topk_weights, is_token_in_rank, w1_local, w2_local,
    streams=streams, num_experts=E,
)
out.sum().backward()
```

That's it. No tile sizes, no cluster shapes, no num_sms — all picked from `w1_local.shape` and `group.size()`. Callers who want to sweep tunings call the kernel-level wrappers directly.

## Compile interaction

`stream_moe_func` is **eager-only**. The streaming-MoE surface launches across two caller-owned streams and runs cross-rank IPC barriers inside the metadata kernel; dynamo has no documented way to enumerate user-managed streams (pytorch/pytorch#92804), and per-rank recompile skew under CDMC>1 would deadlock the `barrier_block` inside the metadata kernel.

Consumers that want `torch.compile` around the outer model must apply `@torch.compiler.disable` at the consumer boundary.

## Testing

Two test surfaces with different scopes:

- **`tests/` — torchrun-style multi-rank integration**. Each script reads `RANK`/`WORLD_SIZE`/`LOCAL_RANK` from env, calls `dist.init_process_group` directly, and exits via `cleanup_dist()` from `tests/utils.py` (barrier + destroy, suppresses TCPStore teardown noise). No pytest harness. Covers cross-rank streaming dispatch + combine correctness (`test_dispatch.py`, `test_combine.py` + grad variants), multi-iteration dispatch_seq reuse (`test_multi_dispatch.py`), routing-metadata generation (`test_metadata.py`), and degenerate routing shapes like all tokens to one expert (`test_skewed_experts.py`). Internode variants (`test_*_internode.py`) test RDMA paths and require ≥2 nodes.
- **`stream_ep/stream_moe/tests/` — pytest single-process kernel logic**. One pytest module per kernel (`test_kernel_a.py`, `test_kernel_a_bwd.py`, `test_kernel_y.py`, `test_kernel_y_bwd.py`). Small shapes, single-GPU, exercises numerical correctness + the producer-consumer spin protocol against an eager-mode reference.

New functionality lands with test coverage in both surfaces as appropriate — new kernel logic gets a pytest case, new dispatch/combine/Buffer behavior gets a torchrun integration test.

## Dependency footprint

StreamEP builds against:

- **PyTorch 2.8** + **CUDA 12.8** + **cp312**.
- **Upstream NVSHMEM 3.5.19** (`nvidia-nvshmem-cu12==3.5.19` pip wheel). The source already includes the v1/v2 device-state layout fix; no patched NVSHMEM needed.
- **[Quack](https://github.com/Dao-AILab/quack) v0.4.1** with two additive hooks (`epi_subtile_store` extract-method on `GemmBase`, `mLensK` plumbing on `VarlenArguments`) needed by the streaming kernel A epilogue and variable-K tile scheduler. Default code paths through `GemmBase.epilogue` are byte-identical to upstream v0.4.1; other Quack consumers are unaffected. Source: [chanzuckerberg/quack](https://github.com/chanzuckerberg/quack) (private), branch `jaime/v0.4.1-port` — 3 commits / +163/-51 LoC vs. upstream v0.4.1.

Coexists with upstream `deep_ep` in the same env — different module names, different `.so` filenames, neither shadows the other.
