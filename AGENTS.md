# AGENTS.md

What anyone changing StreamEP needs to know beyond what the code says. Layout,
environments and commands are in [`README.md`](README.md); how StreamEP diverges
from DeepEP is in [`CHANGES.md`](CHANGES.md). For the code map read the
`stream_ep/stream_moe/__init__.py` docstring, and for the per-layer stream graph
and overlap windows the `stream_ep/stream_moe/stream_moe.py` docstring.

## Scope

StreamEP is an internal component of evoscale's MoE training stack, not a public
release. Evoscale is the single consumer and pins a released wheel, so breaking
API changes are fine as long as the evoscale call site
(`evolutionaryscale/models/moe/stream_moe_wrapper.py`) is updated in the same
change. Target hardware is H100 (SM90), bf16, high-throughput training only.

## Design invariants

### No host syncs in the streaming path

Code in `stream_ep/stream_moe/` and new host code in `csrc/` must not introduce
CPU↔GPU syncs. The only sync per layer is the host poll inside `Buffer.dispatch`
that reads back `num_recv` / `total_tiles` to size the pool; that is the floor
for dropless variable-routing dispatch.

- Disallowed: boolean-mask indexing, `nonzero`, `.item()` / `.tolist()` /
  `.cpu()` followed by host use, D2H copies consumed in the same iteration.
- Allowed: integer-tensor indexing, `where` / `gather` / `scatter` /
  `index_add_`, `cuStreamWaitValue` GPU-side waits, statically known shapes.

If a sync seems necessary, restructure to integer indexing, accept the compute
waste, or fork the kernel forcing it. "Just for now" syncs stay forever.

### Don't propose these

1. **Upper-bound allocation to skip the host poll.** The pipeline is dropless,
   so the worst case is `num_tokens × topk × num_ranks` rows, roughly 16× the
   uniform-routing footprint at production shape, to save a ~150 µs CPU stall
   that does not gate the GPU.
2. **Buffer caching across iterations.** Every per-dispatch tensor is freshly
   allocated. Zero-init and narrowing to the actual `total_tiles` / `T_recv`
   are load-bearing for cross-dispatch correctness: a stale arrival count from
   iteration N lets iteration N+1's CTAs walk past the ready spin. If allocator
   overhead matters, reduce poll latency or aten ops instead.
3. **Cross-layer pipelining.** Layer N+1's routing depends on layer N's output.
   Cached routing in the harnesses is a test convenience, not how production
   runs.

### Pool layout and padding

- The pool is expert-major and padded to `tile_m` per expert. `tile_m` is
  structural: it is the dispatch padding and the per-tile signaling grain shared
  by dispatch and all four compute kernels, not a per-kernel tuning knob.
- Pool placement is a deterministic function of routing. The backward pass
  reuses the forward's saved `StreamingHandle`, so it runs no metadata kernel
  and no host poll.
- Padding rows are never initialized; every consumer is padding-aware instead
  (predicated atomics on `pool_recv_token >= 0`, `seqused_k` for the dW GEMMs,
  the padding mask in `kernel_y_bwd`). A new consumer must gate the same way.
  Use conditional assignment, never multiplication by a 0/1 mask: the GEMM
  through garbage rows can produce inf/NaN, and `0 * inf = NaN`.

### Signaling and streams

- Cross-stream per-tile handoffs use one protocol: the producer
  `red.release.gpu` adds into `arrival_count[tile]` and the consumer spins until
  it equals `arrival_target[tile]`. Per-token handoffs (kernel Y → combine,
  kernel_a_bwd → combine_grads) are single-producer release stamps compared
  against `dispatch_seq`. GEMM → GEMM handoffs on the compute stream rely on
  same-stream FIFO and carry no signal.
- SM starvation between the persistent grids is prevented by the GPU-side
  launch gates (`wait_dispatch_main_started`, `wait_kernel_y_started` and their
  backward mirrors), not by `CUDA_DEVICE_MAX_CONNECTIONS` tuning. A new
  persistent kernel that overlaps another needs the same treatment.

### Communication buffers

- Dispatch, combine and the internode metadata kernel use disjoint regions of
  both the NVL and the RDMA buffers. There is no cross-rank barrier between
  phases, so under load skew one rank's combine (or metadata) writes land in a
  peer's buffer while that peer is still draining dispatch. Never share or
  alias regions; a new carve goes past the existing ones and is covered by the
  `Config::get_*_buffer_size_hint` upper bounds. Two past hangs were exactly
  this.
- Rings persist across iterations with no per-call quiesce, and `put_nbi` is
  fire-and-forget. Slot validity comes from sequence tags and exact cursor
  arithmetic, never from assuming the previous iteration has settled.
- `Buffer` sets `NVSHMEM_IBGDA_NUM_RC_PER_PE >= num_sms` because the forwarder
  coordinator's `qp_id`s span `[num_channels, num_sms)`. Changing `qp_id`
  assignment means revisiting both.

### Allocation

- Per-dispatch tensors that live across the layer are allocated in the
  caller's default-stream pool (`empty_on_default_pool` in `csrc/stream_ep.cpp`)
  without `record_stream`; the layer-end `wait_stream` back-edges make the free
  safe. Allocating them on a side stream strands memory in that stream's pool,
  and `record_stream` piles up pending frees (both cost >10 GB reserved at
  production shape).
- Size per-dispatch arrays from the polled exact counts, never from the local
  token count: ranks may send different numbers of tokens.

## Testing

All suites are pytest, run one session per file by `stream_ep_tests/run.py`
(the `test-*` pixi tasks), so a crash or hang costs one file:

- `stream_ep_tests/kernels/` — single process, one GPU, kernel logic only.
- `stream_ep_tests/dist/` — every rank runs the same session under torchrun;
  passes at any world size, and takes the internode paths on two or more nodes.
- `stream_ep_tests/internode/` — needs two or more nodes (skips below).

Shared pieces: `routing.py` (the routing scenarios the distributed tests
sweep), `reference.py` (torch references built from every rank's gathered
inputs), `conftest.py` (the process group, one `Buffer` and the streams per
session), `kernels/_common.py` (pool layouts and a tile producer). Tests run
at the production shape in `utils.py`.

Rules for the distributed tests, which `conftest.py` cannot enforce:

- Every rank must issue the same collectives in the same order. Do all
  gathers and other collectives first, then assert: a rank that fails between
  two collectives leaves its peers waiting. Skip conditions must evaluate the
  same on every rank. Each test phase ends with a cross-rank verdict, so a
  failure on any rank fails the test everywhere and is reported by rank 0.
- Never reuse a `dispatch_seq` on a `Buffer`; each call takes a fresh one.

Tolerances come from measurement or analysis, never from loosening until a
test passes: exact equality where the operation copies data; bf16 ulp bounds
derived from the rounding steps for reductions; and for GEMM outputs a
per-row relative error set at about twice the worst row measured at the
production shape, against a reference that models only the roundings the
kernels make by design. A known bug gets a `pytest.skip` whose reason names
it, removed by the fix.

`scripts/check_tests.py` (a pre-commit hook) enforces the layout: files sit in
the three suites, distributed tests take the process group from the fixture,
internode modules declare the `internode_only` mark, and imports are
package-qualified.

New functionality lands with tests: a new kernel gets a kernel test, a new
dispatch / combine path or `Buffer` method a distributed one. Assert
numerical contracts, not "did it run", and cover the edge cases the code
handles (padding rows, empty experts and ranks, uneven token counts,
`dispatch_seq` reuse across iterations). A test that fails at production shape
but passes at a smaller config has not passed; find out why instead of
shrinking the config.

## Forks and dependencies

- StreamEP diverges freely from upstream DeepEP; when a cleaner streaming
  pipeline conflicts with staying close to upstream, the pipeline wins.
- The Quack fork is upstream `main` plus the one `seqused_k` commit and must
  stay that way. Anything streaming-specific (subclasses, EpiOps, schedulers)
  lives in `stream_ep/stream_moe/`, and the fork is dropped once `seqused_k`
  lands upstream.
- The pixi environments pin the versions evoscale runs. Bumping torch, CUDA,
  NVSHMEM or cutlass-dsl here goes together with the matching evoscale bump.

## Commits and PRs

Commit titles are imperative, about 70 characters. The body says what changed
and why, and records measured numbers (test counts, timings, memory) rather than
"should pass". PR descriptions follow `.github/pull_request_template.md`.
