"""Streaming-MoE forward layer: dispatch → kernel A → kernel Y → combine on four streams.

Streams are caller-owned — created once via :func:`make_streams` and passed
in as a :class:`StreamHolder`. Per-call inputs vary (routing changes
layer-to-layer in production), and so do the per-call intermediate
allocations: ``postact_a`` is sized by ``total_tiles × tile_M × I`` (varies
with routing). The kernel-Y output ``handle.o`` is allocated by DeepEP inside
``Buffer.dispatch`` at ``[T_recv, hidden]`` zero-init; kernel Y atomic-scatters
into it via PTX-predicated atomics, and the combine sender consumes it on a
separate stream — its per-warp send loop spins on
``handle.compute_done_per_token[r] >= dispatch_seq`` (the per-token gate)
before pushing ``o[r]`` back to ``r``'s origin rank.

Within a layer the four streams overlap (intra-layer); across layers they
serialize via the layer-start ``streams.dispatch.wait_stream(caller)`` and
layer-end ``caller.wait_stream(...)`` back-edges. See design.md §"Cross-stream
sync chain per layer".

Host-side ordering. The setup work (``record_stream`` calls + ``wait_event``)
is split per-stream and interleaved with each stream's kernel launch — kernel
A's path runs first so its launch reaches the GPU before dispatch_main's
persistent CTAs have finished exiting, preserving the per-tile dispatch→A
streaming overlap. Each CUDA API call is ~5 µs of host latency at production
shape; bunching all 19 ``record_stream`` calls + 3 ``wait_event`` calls in
front of kernel A would cost ~100 µs on dispatch_main's tail, which would
collapse the overlap window (the entire dispatch_main is only ~190 µs).

Public surface
--------------
* :class:`StreamHolder` — dataclass holding the four caller-owned streams.
* :func:`make_streams` — construct a ``StreamHolder`` for a given device.
* :class:`StreamMoEFunc` — ``torch.autograd.Function`` running the layer
  forward. ``backward`` returns all-``None`` (the layer is a no-grad boundary).
* :func:`stream_moe_func` — thin wrapper around ``StreamMoEFunc.apply`` with
  the public keyword-arg API.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from deep_ep import Buffer as DeepEPBuffer

from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_a import (
    streaming_moe_a,
)
from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_y import (
    streaming_moe_y,
)


@dataclass(frozen=True)
class StreamHolder:
    """The four caller-owned streams driving one streaming-MoE layer."""

    dispatch: torch.cuda.Stream
    compute_a: torch.cuda.Stream
    compute_y: torch.cuda.Stream
    combine: torch.cuda.Stream


def make_streams(device: torch.device | int | None = None) -> StreamHolder:
    """Allocate the four streams the layer expects.

    Caller creates this once (per training process) and reuses it across all
    layers and iterations — streams are not per-call state.
    """
    return StreamHolder(
        dispatch=torch.cuda.Stream(device=device),
        compute_a=torch.cuda.Stream(device=device),
        compute_y=torch.cuda.Stream(device=device),
        combine=torch.cuda.Stream(device=device),
    )


class StreamMoEFunc(torch.autograd.Function):
    """Streaming-MoE forward as a no-grad autograd boundary.

    All inputs (token activations, routing, expert weights) are treated as
    non-differentiable — ``backward`` returns ``None`` for every forward arg.
    This is the right contract for inference and for the no-grad streaming
    forward path. A separate Function will land alongside backward streaming
    (Phase F) when that path becomes differentiable.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        streams: StreamHolder,
        buffer: DeepEPBuffer,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        is_token_in_rank: torch.Tensor,
        w1_local: torch.Tensor,
        w2_local: torch.Tensor,
        num_experts: int,
        dispatch_seq: int,
        tile_m: int,
        tile_n_a: int,
        tile_n_y: int,
        num_sms_a: int | None,
        num_sms_y: int | None,
    ) -> torch.Tensor:
        caller_stream = torch.cuda.current_stream()

        # Layer-start back-edge — see design.md §"Cross-stream sync chain per
        # layer". Without this, iter N's persistent kernel A / kernel Y CTAs
        # (and the combine sender's per-channel CTAs) starve iter N+1's
        # dispatch_main on shared SMs. We only need to gate ``streams.dispatch``
        # here; the other three consumer streams are gated by ``metadata_done``
        # below, which is recorded downstream of caller_stream →
        # streams.dispatch's serialization.
        streams.dispatch.wait_stream(caller_stream)

        with torch.cuda.stream(streams.dispatch):
            pool, handle, metadata_done = buffer.dispatch(
                x,
                topk_idx,
                topk_weights,
                is_token_in_rank,
                num_experts,
                tile_m=tile_m,
                dispatch_seq=dispatch_seq,
            )

        # ── Kernel A path (queued FIRST so its launch hits the GPU before
        # dispatch_main's persistent CTAs have all exited). dispatch_main runs
        # ~190 µs at production num_sms=132; its CTAs exit gradually as their
        # per-channel work drains, freeing SMs for kernel A's persistent grid
        # to land on. Bunching combine-side host setup before kernel A would
        # push its launch ~50 µs later and collapse the overlap window.
        streams.compute_a.wait_event(metadata_done)
        pool.record_stream(streams.compute_a)
        handle.tile_ready.record_stream(streams.compute_a)
        handle.tile_id_to_expert.record_stream(streams.compute_a)
        handle.expert_pool_block_offset.record_stream(streams.compute_a)
        handle.a_ready.record_stream(streams.compute_a)
        with torch.cuda.stream(streams.compute_a):
            postact_a = torch.empty(
                handle.total_tiles,
                tile_m,
                w2_local.shape[2],
                dtype=x.dtype,
                device=pool.device,
            )
            streaming_moe_a(
                pool,
                w1_local,
                postact_a,
                handle.tile_id_to_expert,
                handle.expert_pool_block_offset,
                handle.tile_ready,
                handle.a_ready,
                dispatch_seq=handle.dispatch_seq,
                compute_seq=handle.dispatch_seq,
                tile_m=tile_m,
                tile_n=tile_n_a,
                num_sms=num_sms_a,
            )

        # ── Kernel Y path. ``postact_a`` is allocated on streams.compute_a
        # where kernel A writes it, and read by kernel Y on streams.compute_y.
        # Per-tile cross-stream data visibility comes from kernel Y's in-kernel
        # ``a_ready`` spin; ``record_stream`` only governs the caching
        # allocator's recycle policy.
        postact_a.record_stream(streams.compute_y)
        streams.compute_y.wait_event(metadata_done)
        handle.tile_id_to_expert.record_stream(streams.compute_y)
        handle.expert_pool_block_offset.record_stream(streams.compute_y)
        handle.a_ready.record_stream(streams.compute_y)
        handle.pool_recv_token.record_stream(streams.compute_y)
        handle.pool_topk_weight.record_stream(streams.compute_y)
        handle.per_token_remaining.record_stream(streams.compute_y)
        handle.compute_done_per_token.record_stream(streams.compute_y)
        handle.o.record_stream(streams.compute_y)
        with torch.cuda.stream(streams.compute_y):
            streaming_moe_y(
                postact_a,
                w2_local,
                handle.o,
                handle.pool_recv_token,
                handle.pool_topk_weight,
                handle.per_token_remaining,
                handle.compute_done_per_token,
                handle.tile_id_to_expert,
                handle.expert_pool_block_offset,
                handle.a_ready,
                compute_seq=handle.dispatch_seq,
                combine_seq=dispatch_seq,
                tile_m=tile_m,
                tile_n=tile_n_y,
                num_sms=num_sms_y,
            )

        # ── Combine sender path. Per-(channel, dst_rank) sender warps spin on
        # ``handle.compute_done_per_token[r] >= dispatch_seq`` before pushing
        # ``o[r]`` to r's origin rank. The per-token gate lets early-completing
        # tokens (K_local=1 first-wave-expert tokens) ship while late-wave
        # tokens are still landing — combine ↔ kernel Y tail overlap. Streaming
        # granularity = NVL chunk size (default 4 tokens per chunk at 8-rank
        # intranode per ``Buffer.get_combine_config``).
        #
        # Cross-stream visibility on ``compute_done_per_token`` is carried by
        # the per-token ``.sys``-scope release/acquire pair the kernels
        # themselves issue; ``streams.combine``'s only cross-stream sync is the
        # ``metadata_done`` event (one-shot, before dispatch main).
        streams.combine.wait_event(metadata_done)
        handle.compute_done_per_token.record_stream(streams.combine)
        handle.o.record_stream(streams.combine)
        handle.rank_prefix_matrix.record_stream(streams.combine)
        handle.channel_prefix_matrix.record_stream(streams.combine)
        handle.send_head.record_stream(streams.combine)
        handle.pool_topk_weight.record_stream(streams.combine)
        handle.recv_token_to_slots.record_stream(streams.combine)
        with torch.cuda.stream(streams.combine):
            out, _ = buffer.combine(handle.o, handle, combine_seq=dispatch_seq)

        # Layer-end back-edges — see design.md §"Cross-stream sync chain per layer".
        caller_stream.wait_stream(streams.dispatch)
        caller_stream.wait_stream(streams.compute_a)
        caller_stream.wait_stream(streams.compute_y)
        caller_stream.wait_stream(streams.combine)
        return out

    @staticmethod
    def backward(ctx, _grad_out):  # type: ignore[override]
        # No-grad boundary: 15 forward args (after ctx) → 15 None gradients.
        return (None,) * 15


def stream_moe_func(
    buffer: DeepEPBuffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    *,
    streams: StreamHolder,
    num_experts: int,
    dispatch_seq: int,
    tile_m: int = 128,
    tile_n_a: int = 256,
    tile_n_y: int = 128,
    num_sms_a: int | None = None,
    num_sms_y: int | None = None,
) -> torch.Tensor:
    """One MoE forward layer: dispatch + kernel A + kernel Y + combine.

    Returns the cross-rank-reduced output of shape ``[num_tokens, hidden]``
    produced by the combine receiver — the standard MoE forward output for
    this rank's source tokens.
    """
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
        dispatch_seq,
        tile_m,
        tile_n_a,
        tile_n_y,
        num_sms_a,
        num_sms_y,
    )
