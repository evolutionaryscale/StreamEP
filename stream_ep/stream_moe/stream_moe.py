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
    dispatch_main). A second GPU-front-end gate
    (``buffer.wait_dispatch_main_started``) holds kernel_a's launch until
    dispatch_main's block 0 has actually entered execution; without it,
    under CDMC>1, kernel_a's 132-CTA grid can grab SMs before dispatch_main
    is co-resident and starve it.
  * fwd kernel_y ↔ combine (compute ↔ communicate): combine's sender warps
    spin on ``y_done_per_token[r] >= dispatch_seq`` and grab SMs as kernel_y
    tiles retire. The orchestrator bumps ``kernel_y_issued`` before
    launching kernel_y, then queues ``buffer.wait_kernel_y_started`` on
    ``communicate`` before combine — a GPU-front-end wait that holds until
    kernel_y's first CTA is co-resident on an SM (the scheduler's
    linear-claim CTA bumps the flag). Combine's 80 sender CTAs cannot
    grab SMs ahead of kernel_y's 132.
  * bwd dispatch_grads ↔ kernel_y_bwd: mirror of fwd dispatch ↔ A.
    ``buffer.dispatch_grads`` returns a ``grads_started_event`` recorded
    BEFORE its main kernel launches; compute waits on it so kernel_y_bwd
    launches concurrent with dispatch_grads_main, spinning on
    ``bwd_dispatch_arrival_count`` per-tile. The mirror of the
    ``wait_dispatch_main_started`` gate
    (``buffer.wait_dispatch_grads_started``) holds kernel_y_bwd's launch
    until dispatch_grads_main's block 0 is co-resident.
  * bwd kernel_a_bwd ↔ combine_grads: mirror of fwd Y ↔ combine. The
    orchestrator bumps ``kernel_a_bwd_issued`` before launching A_bwd, then
    queues ``buffer.wait_kernel_a_bwd_started`` on ``communicate`` before
    combine_grads so A_bwd wins the SM race. Per-recv-token streaming is
    driven by ``bwd_a_done_per_token[r]``.
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
* :class:`StreamMoEFunc` — ``torch.autograd.Function`` running the layer
  forward and backward.
* :func:`stream_moe_func` — thin wrapper around ``StreamMoEFunc.apply`` with
  the public keyword-arg API.

Compile interaction
-------------------
This module is eager-only. Callers that want ``torch.compile`` around the
outer model must apply ``@torch.compiler.disable`` at the consumer boundary
(e.g. evoscale's ``StreamMoEWrapper.forward``). The streaming-MoE surface
launches across two caller-owned streams and runs cross-rank IPC barriers
inside the metadata kernel; dynamo has no documented way to enumerate
user-managed streams (pytorch/pytorch#92804), and per-rank recompile skew
under CDMC>1 would deadlock the ``barrier_block`` inside the metadata
kernel.
"""

from __future__ import annotations

import os
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


# NVTX call counters — used to label fwd/bwd nsys ranges with a per-rank
# monotonic id. Set STREAM_EP_NVTX=1 in the env to enable. Off by default
# so we don't pay the nvtx push/pop overhead in production runs.
_NVTX_FWD_COUNT = 0
_NVTX_BWD_COUNT = 0
_NVTX_ENABLED = bool(os.environ.get("STREAM_EP_NVTX"))


def _nvtx_push(name: str) -> None:
    if _NVTX_ENABLED:
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    if _NVTX_ENABLED:
        torch.cuda.nvtx.range_pop()


class StreamMoEFunc(torch.autograd.Function):
    """Streaming-MoE layer as a differentiable autograd boundary.

    Forward: dispatch → kernel A → kernel Y → combine on the
    ``communicate`` / ``compute`` streams. A GPU-front-end
    ``wait_kernel_y_started`` queued on ``communicate`` before combine
    holds combine's launch until kernel Y's first CTA is co-resident on
    an SM, so kernel Y's 132 CTAs win the SM race against combine's
    80-CTA sender grid.

    Backward: dispatch_grads → kernel_y_bwd → kernel_a_bwd → combine_grads
    on the same ``communicate`` / ``compute`` streams; dW1 / dW2 FIFO-after
    kernel_a_bwd on the same compute stream.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        streams: StreamHolder,
        buffer: StreamEPBuffer,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        is_token_in_rank: torch.Tensor,
        w1_local: torch.Tensor,
        w2_local: torch.Tensor,
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
    ) -> torch.Tensor:
        global _NVTX_FWD_COUNT
        _NVTX_FWD_COUNT += 1
        _fid = _NVTX_FWD_COUNT
        _nvtx_push(f"moe_fwd_{_fid}")

        caller_stream = torch.cuda.current_stream()

        # Layer-start back-edges. Prevent iter N's persistent kernel A / Y
        # CTAs (and combine sender CTAs) from starving iter N+1's dispatch
        # on shared SMs. We gate both streams the layer will launch work on.
        streams.communicate.wait_stream(caller_stream)
        streams.compute.wait_stream(caller_stream)

        _nvtx_push(f"fwd_dispatch_{_fid}")
        with torch.cuda.stream(streams.communicate):
            pool, handle, metadata_done = buffer.dispatch(
                x,
                topk_idx,
                topk_weights,
                is_token_in_rank,
                num_experts,
                tile_m=tile_m,
            )
        _nvtx_pop()  # fwd_dispatch
        # Buffer-owned monotonic counter — ``handle.dispatch_seq`` is the
        # source of truth for the layer's seq. Threaded through kernel Y's
        # ``combine_seq`` (and ``buffer.combine`` defaults to it too).
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

        # GPU-front-end gate: hold kernel_a's launch on ``streams.compute``
        # until dispatch_main_kernel's block 0 has actually entered execution
        # (atomicAdd from block 0 thread 0 → cuStreamWaitValue ge). Without
        # this, under CDMC>1, kernel_a's 132-CTA grid can grab SMs before
        # dispatch_main is co-resident, leaving dispatch_main queued behind
        # kernel_a's CTAs forever. Wait sits at the GPU front-end and does
        # NOT consume an SM.
        buffer.wait_dispatch_main_started(streams.compute)

        # ── Kernel A on `streams.compute` ─────────────────────────────────
        # dispatch_main has retired (wait_stream above), but its per-tile
        # release-add into `pool_arrival_count` runs *before* dispatch_main
        # finishes — kernel A's scheduler spin on
        # `pool_arrival_count[tile] == pool_arrival_target[tile]` re-claims
        # those, terminating immediately on every tile.
        _nvtx_push(f"fwd_compute_{_fid}")
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

            # ── Kernel Y on the same compute stream ──────────────────────
            # Bump the kernel_y issued counter BEFORE launching so a
            # subsequent ``wait_kernel_y_started`` on the communicate stream
            # has the right comparison target. Inside the kernel, the CTA
            # that wins ``linear_idx == 0`` in StreamingTileScheduler
            # atomicAdd's ``kernel_y_started_flag`` — a stronger signal than
            # a torch.cuda.Event between A and Y, which only marks "Y's
            # launch packet is queued" rather than "Y is on an SM".
            buffer.bump_kernel_y_issued()
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
                started_flag=buffer.kernel_y_started_flag_view(),
                tile_m=tile_m,
                tile_n=tile_n_y,
                cluster_n=2,
                num_sms=num_sms_y,
            )
        _nvtx_pop()  # fwd_compute

        # ── Combine on `streams.communicate` (same stream as dispatch) ────
        # FIFO-ordered after dispatch_main. Waits on the
        # ``kernel_y_started_flag`` cross-stream launch gate so combine's
        # 80-CTA sender grid doesn't grab SMs before kernel_y's 132 CTAs
        # have at least one block co-resident. Combine's per-warp sender
        # then spins on ``y_done_per_token[r] >= dispatch_seq`` for
        # per-recv-token streaming.
        _nvtx_push(f"fwd_combine_{_fid}")
        buffer.wait_kernel_y_started(streams.communicate)
        with torch.cuda.stream(streams.communicate):
            # combine_seq defaults to ``handle.dispatch_seq`` per Buffer.combine.
            out, _ = buffer.combine(handle.o, handle)
        _nvtx_pop()  # fwd_combine

        # Layer-end back-edges. caller_stream waits on both streams so the
        # layer-as-barrier invariant holds across layers.
        caller_stream.wait_stream(streams.communicate)
        caller_stream.wait_stream(streams.compute)

        # Save tensors bwd will consume. The contract is "save preact, drop
        # postact" — preact_a is the strictly more useful saved activation:
        # kernel_y_bwd reads it as mC for in-epilogue SwiGLU bwd, and the
        # orchestrator element-wise-recomputes postact_a from it (silu(gate)
        # * up) into a transient buffer just before dW2's grouped GEMM.
        ctx.save_for_backward(preact_a, pool, w1_local, w2_local)
        ctx.streams = streams
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.tile_m = tile_m
        ctx.tile_n_a = tile_n_a
        ctx.tile_n_y = tile_n_y
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
        ctx.num_sms_a = num_sms_a
        ctx.num_sms_y = num_sms_y
        ctx.num_sms_a_bwd = num_sms_a_bwd if num_sms_a_bwd is not None else num_sms_a
        ctx.num_sms_y_bwd = num_sms_y_bwd if num_sms_y_bwd is not None else num_sms_y
        _nvtx_pop()  # moe_fwd
        return out

    @staticmethod
    def backward(ctx, dL_dy):  # type: ignore[override]
        """Backward on the same two streams as fwd.

        Stages by stream:
          ``streams.communicate``: dispatch_grads → combine_grads (FIFO).
          ``streams.compute``: kernel_y_bwd → kernel_a_bwd → dW2 → dW1 (FIFO).

        Cross-stream sync — mirrors fwd's three real overlap windows:

          ``streams.compute.wait_event(grads_started)`` after dispatch_grads
              launches — analogous to fwd's
              ``streams.compute.wait_event(metadata_done)``. The event is
              recorded by ``buffer.dispatch_grads`` between the channel-
              control barrier and ``dispatch_grads_main`` so kernel_y_bwd
              can launch CONCURRENT with dispatch_grads_main; its scheduler
              spins on ``bwd_dispatch_arrival_count`` per-tile.
          ``buffer.wait_dispatch_grads_started`` — GPU-front-end gate on
              ``streams.compute`` that holds kernel_y_bwd's launch until
              dispatch_grads_main's block 0 has entered execution. Same
              rationale as the fwd ``wait_dispatch_main_started`` gate.
          ``buffer.wait_kernel_a_bwd_started`` on ``communicate`` before
              combine_grads — analogous to fwd's
              ``wait_kernel_y_started``. The orchestrator bumps
              ``kernel_a_bwd_issued`` before launching A_bwd on
              ``compute``; the GPU-front-end wait holds combine_grads
              until A_bwd's first CTA is co-resident on an SM, so A_bwd's
              132 CTAs win the SM race against combine_grads's 80. Per-
              recv-token streaming inside combine_grads is driven by
              ``bwd_a_done_per_token[r] >= dispatch_seq`` as kernel_a_bwd
              tiles retire.

        dW1 / dW2 live on the SAME compute stream, FIFO after kernel_a_bwd.
        They cannot overlap with A_bwd (both saturate the 132-CTA grid),
        but they DO overlap with combine_grads on the communicate stream —
        combine_grads's 80 sender CTAs spend much of their duration gated
        on per-token signals / NVLink completion, leaving SMs idle for
        dW's GEMMs. Same-stream FIFO after A_bwd handles dW's inputs from
        kernel_y_bwd (dL_dswiglu_in, postact_a_for_dW2) directly; dW2's
        read of dL_do_pool (written by dispatch_grads on communicate) is
        covered transitively because kernel_y_bwd's per-tile
        ``bwd_dispatch_arrival_count`` acquire-chain already published
        dispatch_grads's writes to the compute stream.

        Within ``streams.compute`` the kernel_y_bwd → kernel_a_bwd handoff
        is implicit same-stream FIFO — no per-tile release-add is needed.

        Returns gradients matching forward's args, in order.
        """
        global _NVTX_BWD_COUNT
        _NVTX_BWD_COUNT += 1
        _bid = _NVTX_BWD_COUNT
        _nvtx_push(f"moe_bwd_{_bid}")

        preact_a, pool, w1_local, w2_local = ctx.saved_tensors
        streams: StreamHolder = ctx.streams
        buffer: StreamEPBuffer = ctx.buffer
        handle = ctx.handle
        # Drop ctx attributes immediately. autograd's grad_fn owns ctx and is
        # held by the output tensor's graph until that output is dropped —
        # in a multi-layer model that means every layer's ctx (and every
        # tensor it references) stays pinned through the iteration. The
        # saved-tensors machinery handles its own release after backward
        # returns; the side-channel attributes do not, so explicitly clear
        # them so handle's tensors (pool_arrival_count et al.) can be freed
        # at this layer's bwd return rather than at the end of the iter.
        del ctx.streams, ctx.buffer, ctx.handle
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

        # Per-stream zero-init / shape-only allocations — runs before the
        # fan-out gate on caller_stream so it overlaps with the upstream
        # layer's bwd tail. These do NOT touch fwd-written handle data, so
        # they don't need the fan-out wait yet.
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
        # Allocated on ``streams.compute`` (block above) but read by
        # ``buffer.combine_grads`` on ``streams.communicate`` below. PyTorch's
        # caching allocator only tracks the allocation stream's events for
        # reuse — without record_stream the storage can be recycled while
        # combine_grads is still reading on ``communicate``. Symmetric to the
        # C++-side ``record_consumer_stream`` plumbing on dispatch slabs (see
        # ``Buffer.set_compute_stream_handle``).
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
        # written on FWD's communicate stream) MUST happen AFTER the fan-out
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
        # Ships dL/dy origin → expert ranks along fwd's routing, K-fans into
        # dL_do_pool[slot]. Pass 2's `red.release.gpu.add.s32` builds
        # bwd_dispatch_arrival_count; kernel_y_bwd's scheduler spins on
        # count == pool_arrival_target. The returned `grads_started_event`
        # is recorded between the channel-control barrier and
        # dispatch_grads_main's launch — analogous to fwd's metadata_done.
        with torch.cuda.stream(streams.communicate):
            dL_do_pool, bwd_dispatch_arrival_count, grads_started = (
                buffer.dispatch_grads(
                    handle, dL_dy, dispatch_seq=handle.dispatch_seq
                )
            )

        # Cross-stream visibility from communicate → compute. Waiting on
        # `grads_started` (recorded BEFORE dispatch_grads_main launches) lets
        # kernel_y_bwd launch concurrent with dispatch_grads_main; the
        # per-tile bwd_dispatch_arrival_count release-adds inside
        # dispatch_grads drive the streaming overlap (analogous to fwd's
        # pool_arrival_count chain).
        streams.compute.wait_event(grads_started)

        # GPU-front-end gate: queue a cuStreamWaitValue so kernel_y_bwd's
        # launch cannot grab SMs before dispatch_grads_main's block 0 has
        # entered execution. Mirror of the fwd dispatch ↔ kernel_a gate
        # above; same rationale.
        buffer.wait_dispatch_grads_started(streams.compute)

        # ── Stage 2 — kernel_y_bwd on streams.compute ──────────────────────
        # Scheduler count-vs-target spin on bwd_dispatch_arrival_count[tile]
        # == pool_arrival_target[tile] terminates immediately because
        # dispatch_grads has fully retired (wait_stream above).
        # Writes dL_dswiglu_in, postact_a_for_dW2, dL_dweight (atomic).
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

            # ── Stage 3 — kernel_a_bwd on the SAME compute stream ──────
            # FIFO-ordered after kernel_y_bwd retires — no cross-stream
            # release-add is needed. Scheduler reuses
            # (bwd_dispatch_arrival_count, pool_arrival_target) — at-target
            # by the time A_bwd runs because Y_bwd already spun on it.
            # Bump the kernel_a_bwd issued counter BEFORE launching so the
            # communicate-stream wait below has the right target. The CTA
            # that wins ``linear_idx == 0`` in StreamingTileScheduler
            # atomicAdd's ``kernel_a_bwd_started_flag``; the
            # ``wait_kernel_a_bwd_started`` GPU-front-end gate on
            # ``communicate`` reads this flag, so combine_grads cannot
            # grab SMs ahead of A_bwd's first co-resident CTA.
            buffer.bump_kernel_a_bwd_issued()
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
                started_flag=buffer.kernel_a_bwd_started_flag_view(),
                tile_m=tile_m,
                tile_n=tile_n_a_bwd,
                num_sms=num_sms_a_bwd,
            )

        # Combine_grads launches when A_bwd has at least one CTA co-resident.
        # Per-recv-token streaming overlap (combine_grads sender ↔ A_bwd
        # tail) is driven by `bwd_a_done_per_token[r] >= dispatch_seq`,
        # fired by A_bwd's last contributor per recv-token.
        buffer.wait_kernel_a_bwd_started(streams.communicate)

        # ── Stage 4 — combine_grads on streams.communicate ─────────────────
        # Sender per-warp loop spins on bwd_a_done_per_token[r] >=
        # dispatch_seq before reading dL_dx_per_r[r] AND dL_dweight[slot]
        # (gathered via recv_token_to_slots[r, k]).
        with torch.cuda.stream(streams.communicate):
            dL_dx, dL_dtopk_weights = buffer.combine_grads(
                dL_dx_per_r,
                handle,
                dL_dweight,
                bwd_a_done_per_token,
                dispatch_seq=handle.dispatch_seq,
            )

        # ── Stage 5 — dW1 / dW2 grouped GEMMs on streams.compute ──────────
        # Both dWs are purely local (no cross-rank). They depend on
        # kernel_y_bwd's outputs (dL_dswiglu_in / postact_a_for_dW2) and
        # dW2 reads dL_do_pool (written by dispatch_grads on communicate).
        # Running on the SAME compute stream after kernel_a_bwd:
        #   * same-stream FIFO covers Y_bwd → dW inputs and the
        #     dispatch_grads → dW2 path (transitively via kernel_y_bwd's
        #     per-tile acquire-chain that already published dispatch_grads's
        #     writes on compute);
        #   * dW cannot overlap with kernel_a_bwd (both saturate 132 SMs),
        #     so FIFO ordering is identical in cost to a separate stream
        #     gated on compute — a dedicated ``dw`` stream would buy nothing;
        #   * dW DOES overlap with combine_grads on the communicate stream
        #     (which is gated on the kernel_a_bwd_started flag and runs
        #     comm-bound), so the useful dW ↔ combine_grads overlap is
        #     preserved.
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
        # dW1 / dW2 on streams.compute. Record them on caller_stream so
        # the caching allocator can recycle correctly once the upstream
        # backward consumes them.
        dL_dx.record_stream(caller_stream)
        dL_dtopk_weights.record_stream(caller_stream)
        dW1_local.record_stream(caller_stream)
        dW2_local.record_stream(caller_stream)

        _nvtx_pop()  # moe_bwd

        return (
            None,  # streams
            None,  # buffer
            dL_dx,  # x
            None,  # topk_idx
            dL_dtopk_weights,  # topk_weights
            None,  # is_token_in_rank
            dW1_local,  # w1_local
            dW2_local,  # w2_local
            None,  # num_experts
            None,  # tile_m
            None,  # tile_n_a
            None,  # tile_n_y
            None,  # tile_n_y_bwd
            None,  # tile_n_a_bwd
            None,  # tile_m_dW1
            None,  # tile_n_dW1
            None,  # tile_m_dW2
            None,  # tile_n_dW2
            None,  # cluster_m_dW1
            None,  # cluster_n_dW1
            None,  # cluster_m_dW2
            None,  # cluster_n_dW2
            None,  # pingpong_dW1
            None,  # pingpong_dW2
            None,  # swizzle_dW1
            None,  # swizzle_dW2
            None,  # num_sms_a
            None,  # num_sms_y
            None,  # num_sms_a_bwd
            None,  # num_sms_y_bwd
        )


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

    The dW1/dW2 tile knobs (``tile_m_dW1``, ``tile_n_dW1``, ``tile_m_dW2``,
    ``tile_n_dW2``) override the dW grouped GEMM tile shape. ``None`` falls
    back to the fwd kernel A tile (``tile_m`` / ``tile_n_a``). The dW2 GEMM
    has output N = ``moe_intermediate_size`` while dW1 has N = ``hidden_size``,
    so optimal tiles are usually different per side.

    Returns the cross-rank-reduced output of shape ``[num_tokens, hidden]``
    produced by the combine receiver — the standard MoE forward output for
    this rank's source tokens.

    Compile interaction: this entry point is eager-only. Callers that want
    ``torch.compile`` around the outer model must apply
    ``@torch.compiler.disable`` at the consumer boundary; see this module's
    docstring for the underlying constraint.
    """
    # Register the compute stream so the C++ Buffer's per-dispatch slabs
    # (``torch::empty``'d on the communicate stream) are record_stream'd onto
    # ``compute`` too. Without this, the caching allocator only tracks the
    # communicate stream and can recycle a slab while kernel_y / kernel_a on
    # compute are still reading or writing. Cheap; idempotent stores an int
    # handle, so calling per-layer is fine.
    buffer.runtime.set_compute_stream_handle(streams.compute.cuda_stream)
    return StreamMoEFunc.apply(
        streams,
        buffer,
        x,
        topk_idx,
        topk_weights,
        is_token_in_rank,
        w1_local,
        w2_local,
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
