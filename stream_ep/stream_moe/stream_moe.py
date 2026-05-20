"""Streaming-MoE layer: dispatch → kernel A → kernel Y → combine on two streams.

Stream graph:

  * ``communicate`` — fwd ``dispatch`` + fwd ``combine``; bwd ``dispatch_grads``
    + bwd ``combine_grads``. Same-stream FIFO orders combine after dispatch
    and combine_grads after dispatch_grads — no cross-stream serialization
    is needed between the comm halves.
  * ``compute`` — fwd kernel_a + kernel_y; bwd kernel_y_bwd + kernel_a_bwd
    + dW1 + dW2. Same-stream FIFO covers all intra-compute handoffs
    (A→Y, Y_bwd→A_bwd, and A_bwd→dW since dW1/dW2 saturate 132 SMs and
    serialize after A_bwd anyway). dW does not need its own stream because
    it cannot overlap with A_bwd; what it CAN overlap with — combine_grads
    on the communicate stream — happens naturally as soon as A_bwd retires.

Real cross-stream overlap windows in this layout:
  * fwd dispatch ↔ kernel_a (communicate ↔ compute): dispatch's 80 persistent
    CTAs drain on the copy engines as kernel_a's 132-CTA grid lands. Per-tile
    handshake is dispatch's ``pool_arrival_count`` release-add chain, which
    kernel_a's scheduler spins on. Communicate → compute hop is
    ``compute.wait_event(metadata_done)`` (event recorded BETWEEN dispatch's
    metadata kernel and dispatch_main, so kernel_a launches concurrent with
    dispatch_main).
  * fwd kernel_y ↔ combine (compute ↔ communicate): combine's sender warps
    spin on ``y_done_per_token[r] >= dispatch_seq`` and grab SMs as kernel_y
    tiles retire. The orchestrator records ``y_started = torch.cuda.Event()``
    between the A and Y launches on ``compute``; the communicate stream waits
    on it before combine — ensuring kernel_y's launch packet is dispatched
    ahead of combine's so kernel_y's 132 CTAs win the SM race.
  * bwd dispatch_grads ↔ kernel_y_bwd: mirror of fwd dispatch ↔ A.
    ``buffer.dispatch_grads`` returns a ``grads_started_event`` recorded
    BEFORE its main kernel launches; compute waits on it so kernel_y_bwd
    launches concurrent with dispatch_grads_main, spinning on
    ``bwd_dispatch_arrival_count`` per-tile.
  * bwd kernel_a_bwd ↔ combine_grads: mirror of fwd Y ↔ combine. An
    ``a_bwd_started`` event between Y_bwd and A_bwd on compute gates
    combine_grads's launch on communicate so A_bwd wins the SM race;
    per-recv-token streaming is driven by ``bwd_a_done_per_token[r]``.
  * bwd dW ↔ combine_grads: dW launches when A_bwd retires (same-stream
    FIFO on compute). combine_grads is still running its NVLink-bound
    sender phase on communicate; dW1/dW2's GEMMs fill the SMs that
    combine_grads's CTAs are leaving idle on per-token gates.

Within a layer the two streams overlap; across layers they serialize via the
layer-start ``stream.wait_stream(caller)`` and layer-end
``caller.wait_stream(stream)`` back-edges.

Public surface
--------------
* :class:`StreamHolder` — dataclass holding the two caller-owned streams.
* :func:`make_streams` — construct a ``StreamHolder`` for a given device.
* :func:`stream_moe_func` — public entry point. Dispatches through the
  registered ``torch.ops.stream_ep.moe`` custom op so the layer is opaque to
  dynamo and inductor inserts the right cross-stream syncs at its boundary.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch
from quack.gemm import gemm
from stream_ep import Buffer as StreamEPBuffer

from stream_ep.stream_moe.kernel_a import (
    streaming_moe_a,
)
from stream_ep.stream_moe.kernel_a_bwd import (
    streaming_moe_a_bwd,
)
from stream_ep.stream_moe.kernel_y import (
    streaming_moe_y,
)
from stream_ep.stream_moe.kernel_y_bwd import (
    streaming_moe_y_bwd,
)


@dataclass(frozen=True)
class StreamHolder:
    """The two caller-owned streams driving one streaming-MoE layer.

    * ``communicate`` — dispatch + combine (fwd) and dispatch_grads +
      combine_grads (bwd). Single comm stream, FIFO-ordered.
    * ``compute`` — kernel_a + kernel_y (fwd) and
      kernel_y_bwd + kernel_a_bwd + dW1 + dW2 (bwd). Single compute stream,
      FIFO-ordered. dW lives on this stream because it SM-contends with
      kernel_a_bwd at the full 132-CTA grid anyway — same-stream FIFO is
      identical to a separate stream gated on compute.
    """

    communicate: torch.cuda.Stream
    compute: torch.cuda.Stream


def make_streams(
    device: torch.device | int | None = None,
    prioritize_communicate: bool = False,
) -> StreamHolder:
    """Allocate the two streams the layer expects.

    Caller creates this once (per training process) and reuses it across all
    layers and iterations — streams are not per-call state.

    When ``prioritize_communicate`` is True, the ``communicate`` stream is
    created at the highest available priority (Megatron-style pattern for
    comm/compute overlap under CUDA_DEVICE_MAX_CONNECTIONS>1). Lower
    priority number = higher priority in PyTorch's Stream API.
    """
    if prioritize_communicate:
        hi = -99
        lo = 0
        return StreamHolder(
            communicate=torch.cuda.Stream(device=device, priority=hi),
            compute=torch.cuda.Stream(device=device, priority=lo),
        )
    return StreamHolder(
        communicate=torch.cuda.Stream(device=device),
        compute=torch.cuda.Stream(device=device),
    )


# ---------------------------------------------------------------------------
# Custom-op registration (stream_ep::moe).
#
# Why: when the caller wraps its outer model in torch.compile, every eager
# function reached via @torch.compiler.disable runs at a graph-break boundary
# where the eager stream context loses visibility into kernels launched by
# inductor on its internal streams. wait_stream(caller_stream) in the body
# below does NOT see those (pytorch/pytorch#92804). Registering this layer as
# an opaque op makes dynamo treat it as a single node; inductor then inserts
# wait_event syncs on the boundary so prior compiled work is drained before
# we enter the op.
#
# Buffer + StreamHolder are Python objects and can't appear in an op schema,
# so the wrapper resolves them to int handles via weak-value dicts. State that
# bwd needs but can't be returned from the op (the StreamingHandle, plus the
# `preact_a` / `pool` tensors fwd consumes) lives in `_INFLIGHT` keyed by
# `dispatch_seq` (monotonic per-Buffer), popped from setup_context onto ctx.
# ---------------------------------------------------------------------------

_BUFFER_REG: weakref.WeakValueDictionary[int, StreamEPBuffer] = weakref.WeakValueDictionary()
_STREAMS_REG: weakref.WeakValueDictionary[int, StreamHolder] = weakref.WeakValueDictionary()

# Per-call StreamingHandle stash keyed by preact_a's underlying storage
# data_ptr. Populated by the fwd kernel at runtime; popped by the bwd kernel
# at runtime via ctx.preact_a_key captured in setup_context.
#
# Storage data_ptr (not id(tensor)) is the right key because under
# torch.compile AOT autograd wraps the kernel's output tensors for autograd
# tracking — setup_context receives a different Python tensor object than the
# kernel returned. id() differs; data_ptr() does not (storage is preserved).
_HANDLE_STASH: dict[int, object] = {}

# NVTX call counters — used to label fwd/bwd nsys ranges with a per-rank
# monotonic id. Set STREAM_EP_NVTX=1 in the env to enable. Off by default
# so we don't pay the nvtx push/pop overhead in production runs.
_NVTX_FWD_COUNT = 0
_NVTX_BWD_COUNT = 0
import os as _os
_NVTX_ENABLED = bool(_os.environ.get("STREAM_EP_NVTX"))


def _nvtx_push(name: str) -> None:
    if _NVTX_ENABLED:
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    if _NVTX_ENABLED:
        torch.cuda.nvtx.range_pop()


def _moe_fwd_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    buf_handle: int,
    streams_handle: int,
    num_experts: int,
    tile_m: int,
    tile_n_a: int,
    tile_n_y: int,
    tile_n_y_bwd: int,
    tile_n_a_bwd: int,
    tile_m_dW1: int | None,
    tile_n_dW1: int | None,
    tile_m_dW2: int | None,
    tile_n_dW2: int | None,
    cluster_m_dW1: int,
    cluster_n_dW1: int,
    cluster_m_dW2: int,
    cluster_n_dW2: int,
    pingpong_dW1: bool,
    pingpong_dW2: bool,
    swizzle_dW1: int,
    swizzle_dW2: int,
    num_sms_a: int | None,
    num_sms_y: int | None,
    num_sms_a_bwd: int | None,
    num_sms_y_bwd: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _NVTX_FWD_COUNT
    _NVTX_FWD_COUNT += 1
    _nvtx_fid = _NVTX_FWD_COUNT
    _nvtx_push(f"moe_fwd_{_nvtx_fid}")

    buffer = _BUFFER_REG[buf_handle]
    streams = _STREAMS_REG[streams_handle]
    caller_stream = torch.cuda.current_stream()

    # Layer-start back-edges. Prevent iter N's persistent kernel A / Y
    # CTAs (and combine sender CTAs) from starving iter N+1's dispatch
    # on shared SMs. We gate both streams the layer will launch work on.
    streams.communicate.wait_stream(caller_stream)
    streams.compute.wait_stream(caller_stream)

    _nvtx_push(f"fwd_dispatch_{_nvtx_fid}")
    with torch.cuda.stream(streams.communicate):
        pool, handle, metadata_done = buffer.dispatch(
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            num_experts,
            tile_m=tile_m,
        )
    _nvtx_pop()
    # Buffer-owned monotonic counter — `handle.dispatch_seq` is the
    # source of truth for the layer's seq. Threaded through kernel Y's
    # `combine_seq` and `buffer.combine` (which itself defaults to
    # ``handle.dispatch_seq`` when no explicit override is passed).
    dispatch_seq = handle.dispatch_seq

    # Cross-stream visibility from communicate → compute. ``metadata_done``
    # is recorded by dispatch *between* the metadata kernel and
    # dispatch_main, so waiting on it lets kernel A launch BEFORE
    # dispatch_main retires — kernel A's scheduler then spins on
    # ``pool_arrival_count`` (which dispatch_main's Pass 2 release-adds
    # fill per-tile) for the actual dispatch ↔ A streaming overlap.
    # The Z_pre + N + Z_post slab regions in ``PostPollBundle`` are all
    # zeroed BEFORE this event records (the CDMC>1 fix in
    # ``allocate_post_poll_bundle``), so every handle tensor is visible
    # the moment ``metadata_done`` fires.
    streams.compute.wait_event(metadata_done)

    # GPU-front-end gate: queue a cuStreamWaitValue on streams.compute so
    # kernel_a's launch cannot grab SMs before dispatch_main's block 0 has
    # entered execution. Without this, under CDMC>1 + torch.compile, kernel_a
    # can race ahead and claim SMs while dispatch_main is still queued —
    # causing dispatch_main to never get its block 0 co-resident. The wait
    # sits at the GPU front-end and does NOT consume an SM.
    buffer.wait_dispatch_main_started(streams.compute)

    # ── Kernel A on `streams.compute` ─────────────────────────────────
    # dispatch_main has retired (wait_stream above), but its per-tile
    # release-add into `pool_arrival_count` runs *before* dispatch_main
    # finishes — kernel A's scheduler spin on
    # `pool_arrival_count[tile] == pool_arrival_target[tile]` re-claims
    # those, terminating immediately on every tile.
    _nvtx_push(f"fwd_compute_{_nvtx_fid}")
    with torch.cuda.stream(streams.compute):
        postact_a = torch.empty(
            handle.total_tiles,
            tile_m,
            w2_local.shape[2],
            dtype=x.dtype,
            device=pool.device,
        )
        # preact_a holds the [2I] pre-SwiGLU gate-up accumulator from kernel
        # A's mD TMA-store path (opt-in via the ``preact_a`` kwarg below).
        # Kept alive across fwd→bwd via ctx.save_for_backward; bwd consumes
        # it for SwiGLU bwd in registers and for postact_a recompute. The
        # cheap thing element-wise-recomputes from preact in bwd, the
        # expensive thing (preact, would otherwise need a GEMM to recover)
        # is the one we save.
        preact_a = torch.empty(
            handle.total_tiles,
            tile_m,
            2 * w2_local.shape[2],
            dtype=x.dtype,
            device=pool.device,
        )
        streaming_moe_a(
            pool,
            w1_local,
            postact_a,
            handle.expert_pool_block_offset,
            handle.pool_arrival_count,
            handle.pool_arrival_target,
            preact_a=preact_a,
            tile_m=tile_m,
            tile_n=tile_n_a,
            cluster_n=2,
            num_sms=num_sms_a,
        )

        # ── Y-started event ────────────────────────────────────────
        # Records the position in the compute stream after kernel A's
        # launch packet and before kernel Y's. When this event fires on
        # the GPU, kernel A has fully retired and kernel Y's launch is
        # next to dispatch from the compute stream's queue. The
        # communicate stream waits on the event before launching
        # combine_main, so kernel_y's 132 persistent CTAs win the SM
        # race against combine's 80.
        y_started = torch.cuda.Event()
        y_started.record()

        # ── Kernel Y on the same compute stream ──────────────────────
        streaming_moe_y(
            postact_a,
            w2_local,
            handle.o,
            handle.pool_recv_token,
            handle.pool_topk_weight,
            handle.k_local_remaining,
            handle.y_done_per_token,
            handle.expert_pool_block_offset,
            handle.pool_arrival_count,
            handle.pool_arrival_target,
            combine_seq=dispatch_seq,
            tile_m=tile_m,
            tile_n=tile_n_y,
            cluster_n=2,
            num_sms=num_sms_y,
        )
    _nvtx_pop()  # fwd_compute_{fid}

    # ── Combine on `streams.communicate` (same stream as dispatch) ────
    # FIFO-ordered after dispatch_main. Waits on the Y-started event so
    # combine's launch doesn't dispatch ahead of kernel_y's on the GPU
    # front-end. Combine's per-warp sender then spins on
    # `y_done_per_token[r] >= dispatch_seq` for per-recv-token streaming.
    _nvtx_push(f"fwd_combine_{_nvtx_fid}")
    streams.communicate.wait_event(y_started)
    with torch.cuda.stream(streams.communicate):
        # combine_seq defaults to ``handle.dispatch_seq`` per Buffer.combine.
        out, _ = buffer.combine(handle.o, handle)
    _nvtx_pop()  # fwd_combine_{fid}

    # Layer-end back-edges. caller_stream waits on both streams so the
    # layer-as-barrier invariant holds across layers.
    caller_stream.wait_stream(streams.communicate)
    caller_stream.wait_stream(streams.compute)

    # Stash the StreamingHandle for the bwd to consume. Keyed by preact_a's
    # storage data_ptr — survives AOT autograd's tensor wrapping. See
    # _HANDLE_STASH comment.
    _HANDLE_STASH[preact_a.untyped_storage().data_ptr()] = handle
    _nvtx_pop()  # moe_fwd_{fid}
    return out, preact_a, pool


@torch.library.custom_op(
    "stream_ep::moe",
    mutates_args=(),
    device_types="cuda",
)
def _moe_op(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    buf_handle: int,
    streams_handle: int,
    num_experts: int,
    tile_m: int,
    tile_n_a: int,
    tile_n_y: int,
    tile_n_y_bwd: int,
    tile_n_a_bwd: int,
    tile_m_dW1: int | None,
    tile_n_dW1: int | None,
    tile_m_dW2: int | None,
    tile_n_dW2: int | None,
    cluster_m_dW1: int,
    cluster_n_dW1: int,
    cluster_m_dW2: int,
    cluster_n_dW2: int,
    pingpong_dW1: bool,
    pingpong_dW2: bool,
    swizzle_dW1: int,
    swizzle_dW2: int,
    num_sms_a: int | None,
    num_sms_y: int | None,
    num_sms_a_bwd: int | None,
    num_sms_y_bwd: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _moe_fwd_impl(
        x, topk_idx, topk_weights, is_token_in_rank, w1_local, w2_local,
        buf_handle, streams_handle, num_experts,
        tile_m, tile_n_a, tile_n_y, tile_n_y_bwd, tile_n_a_bwd,
        tile_m_dW1, tile_n_dW1, tile_m_dW2, tile_n_dW2,
        cluster_m_dW1, cluster_n_dW1, cluster_m_dW2, cluster_n_dW2,
        pingpong_dW1, pingpong_dW2, swizzle_dW1, swizzle_dW2,
        num_sms_a, num_sms_y, num_sms_a_bwd, num_sms_y_bwd,
    )


@_moe_op.register_fake
def _moe_op_fake(
    x, topk_idx, topk_weights, is_token_in_rank, w1_local, w2_local,
    buf_handle, streams_handle, num_experts,
    tile_m, tile_n_a, tile_n_y, tile_n_y_bwd, tile_n_a_bwd,
    tile_m_dW1, tile_n_dW1, tile_m_dW2, tile_n_dW2,
    cluster_m_dW1, cluster_n_dW1, cluster_m_dW2, cluster_n_dW2,
    pingpong_dW1, pingpong_dW2, swizzle_dW1, swizzle_dW2,
    num_sms_a, num_sms_y, num_sms_a_bwd, num_sms_y_bwd,
):
    # `out` matches x: cross-rank-reduced combine output.
    # preact_a / pool shapes depend on routing (total_tiles, TK_padded); use
    # unbacked SymInts so the FX graph remains valid without committing to a
    # static size. They're only consumed by the (opaque) backward, so the
    # symbolic shapes never need to reconcile downstream.
    ctx = torch.library.get_ctx()
    total_tiles = ctx.new_dynamic_size()
    TK_padded = ctx.new_dynamic_size()
    H = x.shape[-1]
    I = w2_local.shape[-1]
    out = torch.empty_like(x)
    preact_a = x.new_empty(total_tiles, tile_m, 2 * I)
    pool = x.new_empty(TK_padded, H)
    return out, preact_a, pool


def _moe_setup_ctx(ctx, inputs, output):
    (
        x, topk_idx, topk_weights, is_token_in_rank, w1_local, w2_local,
        buf_handle, streams_handle, num_experts,
        tile_m, tile_n_a, tile_n_y, tile_n_y_bwd, tile_n_a_bwd,
        tile_m_dW1, tile_n_dW1, tile_m_dW2, tile_n_dW2,
        cluster_m_dW1, cluster_n_dW1, cluster_m_dW2, cluster_n_dW2,
        pingpong_dW1, pingpong_dW2, swizzle_dW1, swizzle_dW2,
        num_sms_a, num_sms_y, num_sms_a_bwd, num_sms_y_bwd,
    ) = inputs
    _, preact_a, pool = output
    # Capture preact_a's storage data_ptr BEFORE save_for_backward — bwd's
    # saved_tensors[0] is a re-materialized Python tensor pointing at the
    # same storage, so its data_ptr matches. id() does not match because
    # SavedVariable + autograd re-wrap the tensor object.
    ctx.preact_a_key = preact_a.untyped_storage().data_ptr()
    # save_for_backward keeps preact_a / pool / w1_local / w2_local alive
    # across the fwd→bwd boundary. Reads only from op inputs/outputs — safe
    # under fake-tensor tracing.
    ctx.save_for_backward(preact_a, pool, w1_local, w2_local)
    ctx.buf_handle = buf_handle
    ctx.streams_handle = streams_handle
    ctx.tile_m = tile_m
    ctx.tile_n_a = tile_n_a
    ctx.tile_n_y_bwd = tile_n_y_bwd
    ctx.tile_n_a_bwd = tile_n_a_bwd
    ctx.tile_m_dW1 = tile_m_dW1
    ctx.tile_n_dW1 = tile_n_dW1
    ctx.tile_m_dW2 = tile_m_dW2
    ctx.tile_n_dW2 = tile_n_dW2
    ctx.cluster_m_dW1 = cluster_m_dW1
    ctx.cluster_n_dW1 = cluster_n_dW1
    ctx.cluster_m_dW2 = cluster_m_dW2
    ctx.cluster_n_dW2 = cluster_n_dW2
    ctx.pingpong_dW1 = pingpong_dW1
    ctx.pingpong_dW2 = pingpong_dW2
    ctx.swizzle_dW1 = swizzle_dW1
    ctx.swizzle_dW2 = swizzle_dW2
    ctx.num_sms_a_bwd = num_sms_a_bwd if num_sms_a_bwd is not None else num_sms_a
    ctx.num_sms_y_bwd = num_sms_y_bwd if num_sms_y_bwd is not None else num_sms_y


def _moe_bwd_impl(ctx, dL_dy, dL_dpreact_a, dL_dpool):
    """Backward on the same two streams as fwd.

    Stages by stream:
      ``streams.communicate``: dispatch_grads → combine_grads (FIFO).
      ``streams.compute``: kernel_y_bwd → kernel_a_bwd → dW2 → dW1 (FIFO).

    `dL_dpreact_a` / `dL_dpool` are gradients flowing into the op's two
    auxiliary outputs (saved for bwd). They have no downstream consumer in
    practice — preact_a and pool are scratch buffers exposed only so
    setup_context can save them under fake-tensor tracing — so they're
    ignored here.
    """
    del dL_dpreact_a, dL_dpool
    global _NVTX_BWD_COUNT
    _NVTX_BWD_COUNT += 1
    _nvtx_bid = _NVTX_BWD_COUNT
    _nvtx_push(f"moe_bwd_{_nvtx_bid}")

    buffer: StreamEPBuffer = _BUFFER_REG[ctx.buf_handle]
    streams: StreamHolder = _STREAMS_REG[ctx.streams_handle]
    preact_a, pool, w1_local, w2_local = ctx.saved_tensors
    handle = _HANDLE_STASH.pop(ctx.preact_a_key)

    tile_m: int = ctx.tile_m
    tile_n_a: int = ctx.tile_n_a
    tile_n_y_bwd: int = ctx.tile_n_y_bwd
    tile_n_a_bwd: int = ctx.tile_n_a_bwd
    tile_m_dW1: int = ctx.tile_m_dW1 if ctx.tile_m_dW1 is not None else tile_m
    tile_n_dW1: int = ctx.tile_n_dW1 if ctx.tile_n_dW1 is not None else tile_n_a
    tile_m_dW2: int = ctx.tile_m_dW2 if ctx.tile_m_dW2 is not None else tile_m
    tile_n_dW2: int = ctx.tile_n_dW2 if ctx.tile_n_dW2 is not None else tile_n_a
    cluster_m_dW1: int = ctx.cluster_m_dW1
    cluster_n_dW1: int = ctx.cluster_n_dW1
    cluster_m_dW2: int = ctx.cluster_m_dW2
    cluster_n_dW2: int = ctx.cluster_n_dW2
    pingpong_dW1: bool = ctx.pingpong_dW1
    pingpong_dW2: bool = ctx.pingpong_dW2
    swizzle_dW1: int = ctx.swizzle_dW1
    swizzle_dW2: int = ctx.swizzle_dW2
    num_sms_a_bwd: int | None = ctx.num_sms_a_bwd
    num_sms_y_bwd: int | None = ctx.num_sms_y_bwd

    # Upstream may pass a non-contiguous grad (e.g. `out.sum().backward()`
    # produces a stride-(0,0) broadcast view); `dispatch_grads` asserts
    # contiguity, so normalise here.
    dL_dy = dL_dy.contiguous()

    caller_stream = torch.cuda.current_stream()
    device = pool.device
    dtype = pool.dtype

    E_local, two_I, H = w1_local.shape
    I = two_I // 2
    total_tiles = handle.total_tiles
    TK_padded = pool.shape[0]
    T_recv = handle.k_local_total.shape[0]

    # Per-stream zero-init / allocation setup — runs before the
    # fan-out gate on caller_stream so it overlaps with the upstream
    # layer's bwd tail.
    with torch.cuda.stream(streams.compute):
        dL_dx_per_r = torch.zeros(T_recv, H, dtype=dtype, device=device)
        bwd_k_local_remaining = torch.empty(
            T_recv, dtype=torch.int32, device=device
        )
        bwd_a_done_per_token = torch.zeros(
            T_recv, dtype=torch.int64, device=device
        )
        dL_dswiglu_in = torch.empty(
            total_tiles, tile_m, two_I, dtype=dtype, device=device
        )
        postact_a_for_dW2 = torch.empty(
            total_tiles, tile_m, I, dtype=dtype, device=device
        )
        # dL_dweight: per-pid_n fp32 atomic-add target; MUST zero-init.
        dL_dweight = torch.zeros(TK_padded, dtype=torch.float32, device=device)
    # Allocated on `streams.compute` (block above) but read by
    # `buffer.combine_grads` on `streams.communicate` below. PyTorch's
    # caching allocator only tracks the allocation stream's events for
    # reuse — without record_stream the storage can be recycled while
    # combine_grads is still reading on `communicate`. Symmetric to the
    # C++-side `record_consumer_stream` plumbing on dispatch slabs (see
    # `Buffer::set_compute_stream_handle`).
    dL_dx_per_r.record_stream(streams.communicate)
    bwd_a_done_per_token.record_stream(streams.communicate)
    dL_dweight.record_stream(streams.communicate)

    with torch.cuda.stream(streams.compute):
        # quack `gemm` with default `add_to_output=False` overwrites D,
        # so neither dW destination needs zero-init.
        dW1_local = torch.empty_like(w1_local)
        dW2_local = torch.empty_like(w2_local)

    # ── Single fan-out gate on caller_stream ───────────────────────────
    streams.communicate.wait_stream(caller_stream)
    streams.compute.wait_stream(caller_stream)

    # Reads of FWD-written handle tensors (expert_pool_block_offset,
    # expert_frequency, k_local_total — all on the fwd post-poll bundle,
    # written on FWD's communicate stream) MUST happen after the fan-out
    # gate above so compute has joined caller_stream's FWD-completion
    # state. Doing these `.to()` / arithmetic before the gate was a real
    # bug at internode: the .to() reads on compute would race against
    # FWD's communicate writes and pull stale / partially-initialized
    # values, producing wrong cu_seqlens_k → wrong K-tile bounds in dW2
    # → reads of garbage padding rows of pool → NaN dW2 grads on
    # specific experts.
    with torch.cuda.stream(streams.compute):
        cu_seqlens_k = (
            handle.expert_pool_block_offset.to(torch.int32) * tile_m
        ).contiguous()
        lens_k_dW = handle.expert_frequency.to(torch.int32)
        bwd_k_local_remaining.copy_(handle.k_local_total, non_blocking=True)

    # ── Stage 1 — dispatch_grads on streams.communicate ────────────────
    with torch.cuda.stream(streams.communicate):
        dL_do_pool, bwd_dispatch_arrival_count, grads_started = (
            buffer.dispatch_grads(
                handle, dL_dy, dispatch_seq=handle.dispatch_seq
            )
        )

    streams.compute.wait_event(grads_started)

    # GPU-front-end gate: queue a cuStreamWaitValue so kernel_y_bwd's launch
    # cannot grab SMs before dispatch_grads_main's block 0 has entered
    # execution. Mirror of the fwd dispatch ↔ kernel_a gate above; same
    # rationale.
    buffer.wait_dispatch_grads_started(streams.compute)

    # ── Stage 2 — kernel_y_bwd on streams.compute ──────────────────────
    with torch.cuda.stream(streams.compute):
        streaming_moe_y_bwd(
            dL_do_pool,
            w2_local,
            dL_dswiglu_in,
            postact_a_for_dW2,
            handle.pool_topk_weight,
            handle.pool_recv_token,
            preact_a,
            dL_dweight,
            handle.expert_pool_block_offset,
            bwd_dispatch_arrival_count,
            handle.pool_arrival_target,
            tile_m=tile_m,
            tile_n=tile_n_y_bwd,
            num_sms=num_sms_y_bwd,
        )

        # ── A_bwd_started event ────────────────────────────────────
        a_bwd_started = torch.cuda.Event()
        a_bwd_started.record()

        # ── Stage 3 — kernel_a_bwd on the SAME compute stream ──────
        streaming_moe_a_bwd(
            dL_dswiglu_in,
            w1_local,
            dL_dx_per_r,
            handle.pool_recv_token,
            bwd_k_local_remaining,
            bwd_a_done_per_token,
            handle.expert_pool_block_offset,
            bwd_dispatch_arrival_count,
            handle.pool_arrival_target,
            dispatch_seq=handle.dispatch_seq,
            tile_m=tile_m,
            tile_n=tile_n_a_bwd,
            num_sms=num_sms_a_bwd,
        )

    streams.communicate.wait_event(a_bwd_started)

    # ── Stage 4 — combine_grads on streams.communicate ─────────────────
    with torch.cuda.stream(streams.communicate):
        dL_dx, dL_dtopk_weights = buffer.combine_grads(
            dL_dx_per_r,
            handle,
            dL_dweight,
            bwd_a_done_per_token,
            dispatch_seq=handle.dispatch_seq,
        )

    # ── Stage 5 — dW1 / dW2 grouped GEMMs on streams.compute ──────────
    with torch.cuda.stream(streams.compute):
        # dW2[e] = postact_a_for_dW2[slot_range_e].T @ dL_do_pool[slot_range_e]
        postact_a_for_dW2_flat = postact_a_for_dW2.view(TK_padded, I)
        gemm(
            dL_do_pool.t(),
            postact_a_for_dW2_flat.t(),
            dW2_local,
            None,
            None,
            tile_M=tile_m_dW2,
            tile_N=tile_n_dW2,
            cluster_M=cluster_m_dW2,
            cluster_N=cluster_n_dW2,
            pingpong=pingpong_dW2,
            max_swizzle_size=swizzle_dW2,
            cu_seqlens_k=cu_seqlens_k,
            lens_k=lens_k_dW,
        )
        # dW1[e] = (dL_dswiglu_in[slot_range_e]).T @ pool[slot_range_e]
        dL_dswiglu_in_flat = dL_dswiglu_in.view(TK_padded, two_I)
        gemm(
            dL_dswiglu_in_flat.t(),
            pool.t(),
            dW1_local,
            None,
            None,
            tile_M=tile_m_dW1,
            tile_N=tile_n_dW1,
            cluster_M=cluster_m_dW1,
            cluster_N=cluster_n_dW1,
            pingpong=pingpong_dW1,
            max_swizzle_size=swizzle_dW1,
            cu_seqlens_k=cu_seqlens_k,
            lens_k=lens_k_dW,
        )

    # ── Exit chain back to caller_stream ───────────────────────────────
    caller_stream.wait_stream(streams.communicate)
    caller_stream.wait_stream(streams.compute)

    # Outputs of combine_grads were allocated on streams.communicate;
    # dW1 / dW2 on streams.compute. Record them on caller_stream so the
    # caching allocator can recycle correctly once the upstream backward
    # consumes them.
    dL_dx.record_stream(caller_stream)
    dL_dtopk_weights.record_stream(caller_stream)
    dW1_local.record_stream(caller_stream)
    dW2_local.record_stream(caller_stream)

    _nvtx_pop()  # moe_bwd_{bid}

    # Gradients match the op's input arity. Non-tensor inputs return None;
    # is_token_in_rank / topk_idx are integer / boolean inputs with no grad.
    return (
        dL_dx,                # x
        None,                 # topk_idx
        dL_dtopk_weights,     # topk_weights
        None,                 # is_token_in_rank
        dW1_local,            # w1_local
        dW2_local,            # w2_local
        None,                 # buf_handle
        None,                 # streams_handle
        None,                 # num_experts
        None,                 # tile_m
        None,                 # tile_n_a
        None,                 # tile_n_y
        None,                 # tile_n_y_bwd
        None,                 # tile_n_a_bwd
        None,                 # tile_m_dW1
        None,                 # tile_n_dW1
        None,                 # tile_m_dW2
        None,                 # tile_n_dW2
        None,                 # cluster_m_dW1
        None,                 # cluster_n_dW1
        None,                 # cluster_m_dW2
        None,                 # cluster_n_dW2
        None,                 # pingpong_dW1
        None,                 # pingpong_dW2
        None,                 # swizzle_dW1
        None,                 # swizzle_dW2
        None,                 # num_sms_a
        None,                 # num_sms_y
        None,                 # num_sms_a_bwd
        None,                 # num_sms_y_bwd
    )


torch.library.register_autograd(
    "stream_ep::moe",
    _moe_bwd_impl,
    setup_context=_moe_setup_ctx,
)


def _register(reg, obj):
    h = id(obj)
    reg[h] = obj
    return h


# `@torch._dynamo.allow_in_graph` tells dynamo: treat this function as a
# leaf in the FX graph. Don't trace into the body — call it eagerly at
# runtime. Without this, dynamo follows the wrapper into the
# `_register(_BUFFER_REG, ...)` / `_register(_STREAMS_REG, ...)` calls,
# trips on `weakref.WeakValueDictionary.__setitem__` (untraceable C-side
# weakref machinery), and graph-breaks ONCE PER LAYER PER ITER. That break
# fragments the compiled outer graph at the MoE boundary, adding a Python
# re-entry round trip per layer. `allow_in_graph` collapses the entire
# wrapper to a single call_function FX node; dynamo continues tracing
# uninterrupted after it. The custom op `torch.ops.stream_ep.moe` is
# already opaque to dynamo (registered via torch.library.custom_op) — this
# decorator just plugs the residual hole around the registry plumbing.
@torch._dynamo.allow_in_graph
def stream_moe_func(
    buffer: StreamEPBuffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    *,
    streams: StreamHolder,
    num_experts: int,
    tile_m: int = 128,
    tile_n_a: int = 192,
    tile_n_y: int = 256,
    tile_n_y_bwd: int = 128,
    tile_n_a_bwd: int = 256,
    tile_m_dW1: int | None = None,
    tile_n_dW1: int | None = 256,
    tile_m_dW2: int | None = None,
    tile_n_dW2: int | None = None,
    cluster_m_dW1: int = 2,
    cluster_n_dW1: int = 2,
    cluster_m_dW2: int = 1,
    cluster_n_dW2: int = 1,
    pingpong_dW1: bool = False,
    pingpong_dW2: bool = False,
    swizzle_dW1: int = 8,
    swizzle_dW2: int = 8,
    num_sms_a: int | None = None,
    num_sms_y: int | None = None,
    num_sms_a_bwd: int | None = None,
    num_sms_y_bwd: int | None = None,
) -> torch.Tensor:
    """One MoE forward layer: dispatch + kernel A + kernel Y + combine.

    Routes through ``torch.ops.stream_ep.moe`` (registered as a custom op
    so it's opaque to dynamo). On top of that, this entry point is itself
    ``@torch.compiler.disable``'d: callers can wrap their outer model in
    ``torch.compile`` without dynamo tracing through any of the streaming-MoE
    surface. The disable is enforced at the library boundary rather than
    asked of every consumer because under CDMC>1 the cross-rank
    ``barrier_block`` in the metadata kernel cannot tolerate per-rank skew
    from compile-side decisions (autotune, specialize/generalize transitions,
    cudagraph capture↔replay) — if one rank takes a recompile path the
    others don't, the barrier deadlocks. The custom op remains the right
    integration primitive for any caller that does manage to reach it
    through a compiled graph.

    The dW1/dW2 tile knobs (``tile_m_dW1``, ``tile_n_dW1``, ``tile_m_dW2``,
    ``tile_n_dW2``) override the dW grouped GEMM tile shape. ``None`` falls
    back to the fwd kernel A tile (``tile_m`` / ``tile_n_a``).

    Returns the cross-rank-reduced output of shape ``[num_tokens, hidden]``.
    """
    # Register the compute stream so the C++ Buffer's per-dispatch slabs
    # (`torch::empty`'d on the communicate stream) are record_stream'd onto
    # `compute` too. Without this, the caching allocator only tracks the
    # communicate stream and can recycle a slab while kernel_y / kernel_a
    # on compute are still reading or writing. Cheap; idempotent stores an
    # int handle, so calling per-layer is fine.
    buffer.runtime.set_compute_stream_handle(streams.compute.cuda_stream)
    out, _preact_a, _pool = torch.ops.stream_ep.moe(
        x,
        topk_idx,
        topk_weights,
        is_token_in_rank,
        w1_local,
        w2_local,
        _register(_BUFFER_REG, buffer),
        _register(_STREAMS_REG, streams),
        num_experts,
        tile_m,
        tile_n_a,
        tile_n_y,
        tile_n_y_bwd,
        tile_n_a_bwd,
        tile_m_dW1,
        tile_n_dW1,
        tile_m_dW2,
        tile_n_dW2,
        cluster_m_dW1,
        cluster_n_dW1,
        cluster_m_dW2,
        cluster_n_dW2,
        pingpong_dW1,
        pingpong_dW2,
        swizzle_dW1,
        swizzle_dW2,
        num_sms_a,
        num_sms_y,
        num_sms_a_bwd,
        num_sms_y_bwd,
    )
    return out
