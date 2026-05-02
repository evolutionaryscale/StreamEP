"""Streaming-MoE forward layer: dispatch → kernel A → kernel Y → combine on four streams.

Streams are caller-owned — created once and passed in. Per-call inputs vary
(routing changes layer-to-layer in production), and so do the per-call
intermediate allocations: ``postact_a`` is sized by ``total_tiles × tile_M × I``
(varies with routing). The kernel-Y output ``handle.o`` is allocated by DeepEP
inside ``Buffer.dispatch`` at ``[T_recv, hidden]`` zero-init; kernel Y
atomic-scatters into it via PTX-predicated atomics, and the combine sender
consumes it on a separate stream — its per-warp send loop spins on
``handle.compute_done_per_token[r] >= dispatch_seq`` (Phase D's per-token gate)
before pushing ``o[r]`` back to ``r``'s origin rank.

Within a layer the four streams overlap (intra-layer); across layers they
serialize via the layer-start ``{comm,combine_send}_stream.wait_stream(caller)``
and layer-end ``caller.wait_stream(...)`` back-edges. See design.md
§"Cross-stream sync chain per layer".
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
    combine_send_stream: torch.cuda.Stream,
    num_experts: int,
    dispatch_seq: int,
    tile_m: int = 128,
    tile_n_a: int = 256,
    tile_n_y: int = 128,
    num_sms_a: int | None = None,
    num_sms_y: int | None = None,
) -> Tuple[torch.Tensor, StreamingHandle]:
    """One MoE forward layer: dispatch + kernel A + kernel Y + combine.

    Returns ``(out, handle)`` where ``out`` is the cross-rank-reduced output
    of shape ``[num_tokens, hidden]`` produced by the combine receiver — the
    standard MoE forward output for this rank's source tokens. Kernel Y's
    locally-summed buffer ``handle.o`` is reachable on the handle if needed.
    """
    caller_stream = torch.cuda.current_stream()

    # Layer-start back-edges — see design.md §"Cross-stream sync chain per
    # layer". Without these, iter N's persistent kernel A / kernel Y CTAs
    # (and the combine sender's per-channel CTAs) starve iter N+1's
    # dispatch_main on shared SMs. We only need to gate the streams that
    # launch their first kernel of the layer with the host (comm_stream and
    # combine_send_stream both launch directly from the host); compute_a /
    # compute_y are gated by ``metadata_done`` below, which is recorded
    # downstream of caller_stream → comm_stream's serialization.
    comm_stream.wait_stream(caller_stream)
    combine_send_stream.wait_stream(caller_stream)

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
    # Combine sender reads handle.{recv_src_idx, rank_prefix_matrix,
    # channel_prefix_matrix, send_head, recv_topk_weights} (all metadata
    # outputs of dispatch) plus handle.{compute_done_per_token, o} (which
    # kernel Y populates on compute_y_stream — visibility carried by the
    # per-token release/acquire pair on compute_done_per_token, not by this
    # event). The wait_event here only ensures combine sees metadata-tensor
    # writes without serializing against dispatch main.
    combine_send_stream.wait_event(metadata_done)

    total_tiles = handle.total_tiles
    intermediate = w2_local.shape[2]

    # `record_stream` only on the stream(s) that actually consume each
    # tensor. Cross-stream data visibility itself comes from the per-tile
    # release/acquire pairs (`.sys` scope), not from these calls —
    # `record_stream` only governs the caching allocator's recycle policy
    # (don't recycle until each recorded stream's events at free-time have
    # completed).
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
    #   compute_done_per_token     — kernel Y release-stores (compute_y);
    #                                combine sender acquire-loads (combine_send)
    #   o                          — kernel Y atomic-scatter (compute_y);
    #                                combine sender reads (combine_send)
    #   recv_src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
    #   recv_topk_weights          — combine sender reads (combine_send only)
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
    handle.compute_done_per_token.record_stream(combine_send_stream)
    handle.o.record_stream(compute_y_stream)
    handle.o.record_stream(combine_send_stream)
    handle.recv_src_idx.record_stream(combine_send_stream)
    handle.rank_prefix_matrix.record_stream(combine_send_stream)
    handle.channel_prefix_matrix.record_stream(combine_send_stream)
    handle.send_head.record_stream(combine_send_stream)
    handle.recv_topk_weights.record_stream(combine_send_stream)

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

    # Combine on its own stream: per-(channel, dst_rank) sender warps spin on
    # handle.compute_done_per_token[r] >= dispatch_seq before pushing o[r]
    # to r's origin rank. The per-token gate lets early-completing tokens
    # (K_local=1 first-wave-expert tokens) ship while late-wave tokens are
    # still landing — combine ↔ kernel Y tail overlap. Streaming granularity
    # = NVL chunk size (default 4 tokens per chunk at 8-rank intranode per
    # Buffer.get_combine_config).
    with torch.cuda.stream(combine_send_stream):
        out, _ = buffer.combine(
            handle.o,
            handle,
            topk_weights=handle.recv_topk_weights,
            combine_seq=dispatch_seq,
        )

    # Layer-end back-edges — see design.md §"Cross-stream sync chain per layer".
    caller_stream.wait_stream(comm_stream)
    caller_stream.wait_stream(compute_a_stream)
    caller_stream.wait_stream(compute_y_stream)
    caller_stream.wait_stream(combine_send_stream)
    return out, handle
