"""Streaming-MoE forward layer: dispatch → kernel A → kernel Y on three streams.

Streams are caller-owned — created once and passed in. Per-call inputs vary
(routing changes layer-to-layer in production), and so do the per-call
intermediate allocations: ``postact_a`` is sized by ``total_tiles × tile_M × I``
(varies with routing) and ``o_with_trash`` by ``T_recv + 1``. Both are allocated
fresh inside this function and freed when it returns.

Within a layer the three streams overlap (intra-layer); across layers they
serialize via the layer-start ``comm_stream.wait_stream(caller)`` and layer-end
``caller.wait_stream(...)`` back-edges. See design.md §"Cross-stream sync chain
per layer".
"""

from __future__ import annotations

from typing import Tuple

import torch
from deep_ep import Buffer as DeepEPBuffer
from deep_ep.buffer import StreamingHandle

from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_a import (
    streaming_moe_a,
)
from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_y import (
    streaming_moe_y,
)


def streaming_moe_layer(
    buffer: DeepEPBuffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    *,
    comm_stream: torch.cuda.Stream,
    compute_a_stream: torch.cuda.Stream,
    compute_y_stream: torch.cuda.Stream,
    num_experts: int,
    dispatch_seq: int,
    tile_m: int = 128,
    tile_n_a: int = 256,
    tile_n_y: int = 128,
    num_sms_a: int | None = None,
    num_sms_y: int | None = None,
) -> Tuple[torch.Tensor, StreamingHandle]:
    """One MoE forward layer: dispatch + kernel A + kernel Y.

    Returns ``(o, handle)`` where ``o`` has shape ``[T_recv + 1, hidden]``.
    The trailing trash row at index ``T_recv`` absorbs writes from padding
    pool slots; slice ``o[:T_recv]`` for the strict per-recv-token output.
    """
    caller_stream = torch.cuda.current_stream()

    # Layer-start back-edge — see design.md §"Cross-stream sync chain per
    # layer". Without it iter N's persistent kernel A / kernel Y CTAs starve
    # iter N+1's dispatch_main on shared SMs.
    comm_stream.wait_stream(caller_stream)

    with torch.cuda.stream(comm_stream):
        pool, handle, metadata_done = buffer.dispatch(
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            num_experts,
            tile_m=tile_m,
            dispatch_seq=dispatch_seq,
        )
    compute_a_stream.wait_event(metadata_done)
    compute_y_stream.wait_event(metadata_done)

    total_tiles = handle.total_tiles
    T_recv = handle.o.shape[0]
    hidden = w2_local.shape[1]
    intermediate = w2_local.shape[2]

    # `record_stream` only on the stream(s) that actually consume each
    # tensor. Cross-stream data visibility itself comes from the per-tile
    # tile_ready / a_ready release/acquire pairs (`.sys` scope), not from
    # these calls — `record_stream` only governs the caching allocator's
    # recycle policy (don't recycle until each recorded stream's events
    # at free-time have completed).
    #
    # Per-tensor consumers:
    #   pool                       — kernel A reads (compute_a only)
    #   tile_id_to_expert          — kernel A + kernel Y read (both)
    #   expert_pool_block_offset   — kernel A + kernel Y read (both)
    #   tile_ready                 — kernel A spins (compute_a only)
    #   a_ready                    — kernel A writes, kernel Y spins (both)
    #   pool_recv_token            — kernel Y reads (compute_y only)
    #   pool_topk_weight           — kernel Y reads (compute_y only)
    #   per_token_remaining        — kernel Y atomic-decrements (compute_y only)
    #   compute_done_per_token     — kernel Y release-stores (compute_y only;
    #                                Phase D will add combine_send_stream)
    pool.record_stream(compute_a_stream)
    handle.tile_ready.record_stream(compute_a_stream)
    handle.tile_id_to_expert.record_stream(compute_a_stream)
    handle.tile_id_to_expert.record_stream(compute_y_stream)
    handle.expert_pool_block_offset.record_stream(compute_a_stream)
    handle.expert_pool_block_offset.record_stream(compute_y_stream)
    handle.a_ready.record_stream(compute_a_stream)
    handle.a_ready.record_stream(compute_y_stream)
    handle.pool_recv_token.record_stream(compute_y_stream)
    handle.pool_topk_weight.record_stream(compute_y_stream)
    handle.per_token_remaining.record_stream(compute_y_stream)
    handle.compute_done_per_token.record_stream(compute_y_stream)

    with torch.cuda.stream(compute_a_stream):
        postact_a = torch.empty(
            total_tiles, tile_m, intermediate, dtype=x.dtype, device=pool.device
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

    # `postact_a` is allocated on compute_a_stream (where kernel A writes
    # it) and read by kernel Y on compute_y_stream. Per-tile cross-stream
    # data visibility comes from kernel Y's in-kernel `a_ready` spin; this
    # call is only to keep the caching allocator from recycling the block
    # while kernel Y is mid-read.
    postact_a.record_stream(compute_y_stream)
    with torch.cuda.stream(compute_y_stream):
        o_with_trash = torch.zeros(
            T_recv + 1, hidden, dtype=x.dtype, device=pool.device
        )
        streaming_moe_y(
            postact_a,
            w2_local,
            o_with_trash,
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

    # Layer-end back-edge — see design.md §"Cross-stream sync chain per layer".
    caller_stream.wait_stream(comm_stream)
    caller_stream.wait_stream(compute_a_stream)
    caller_stream.wait_stream(compute_y_stream)
    return o_with_trash, handle
