"""Phase 2: streaming MoE forward built on DeepEP + SonicMoE.

Baseline (this commit): single dispatch -> full-recv compute -> single combine.
No streaming yet, no autograd. The per-batch streaming loop on top of
`dispatch_done` signals lands next; backward (Phase 5) lands later.
"""

from __future__ import annotations

import torch
from deep_ep import Buffer as DeepEPBuffer
from quack.gemm_interface import gemm, gemm_gated
from sonicmoe.functional import TC_topk_router_metadata_triton


def build_routing_metadata(recv_topk_idx: torch.Tensor, local_num_experts: int):
    """Per-(token, k) routing metadata for sonic-moe-style grouped GEMM.

    Nonlocal slots in `recv_topk_idx` are -1; we remap them to a sentinel expert
    `local_num_experts` so they all bucket into one extra bin. The caller pads
    weights with a zero expert at that index, making the dropped rows compute
    to zero so we can keep the GEMM `A_idx` length fixed at `N * K` (no sync).

    Returns `(x_gather_idx, s_scatter_idx, expert_frequency, expert_frequency_offset)`,
    all int32 device tensors. `s_scatter_idx` and `x_gather_idx` are zero-init —
    sentinel-bin slots are not written by the kernel, and zero is a safe
    fallback because the corresponding `y2` rows are zero anyway.
    """
    device = recv_topk_idx.device
    N, K = recv_topk_idx.shape
    TK = N * K
    sentinel = local_num_experts
    E_sentinel = local_num_experts + 1

    expert_with_sentinel = torch.where(
        recv_topk_idx >= 0,
        recv_topk_idx.to(torch.int32),
        torch.full_like(recv_topk_idx, sentinel, dtype=torch.int32),
    ).contiguous()

    expert_frequency = torch.empty(E_sentinel, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(
        E_sentinel + 1, dtype=torch.int32, device=device
    )
    x_gather_idx = torch.zeros(TK, dtype=torch.int32, device=device)
    s_scatter_idx = torch.zeros(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        expert_with_sentinel,
        E_sentinel,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
    )

    return x_gather_idx, s_scatter_idx, expert_frequency, expert_frequency_offset


def aggregate_topk(
    y2: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    n_recv: int,
) -> torch.Tensor:
    """Scatter `[TK, H]` expert outputs back to `[N_recv, H]`, weighted by the
    per-(token, k) topk weight. The remaining cross-rank reduction happens
    inside `buffer.combine`. Sentinel-bin rows have `y2 == 0` (zero-padded
    weight expert) so their contribution is zero regardless of the (garbage)
    weight read.
    """
    H = y2.size(-1)
    weights_flat = recv_topk_weights.reshape(-1)
    weights_per_slot = weights_flat[s_scatter_idx.long()].to(y2.dtype).unsqueeze(-1)
    out = torch.zeros(n_recv, H, dtype=y2.dtype, device=y2.device)
    out.index_add_(0, x_gather_idx.long(), y2 * weights_per_slot)
    return out


def _pad_zero_expert(w: torch.Tensor) -> torch.Tensor:
    """Append a zero expert at the end of an `[E_local, *, *]` weight tensor.

    Lets the QuACK grouped GEMM process the sentinel-bin rows (which carry
    nonlocal `(token, k)` pairs we don't want compute for) and produce zeros.
    """
    return torch.cat([w, torch.zeros_like(w[:1])], dim=0)


@torch.no_grad()
def streaming_moe_forward(
    x: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buffer: DeepEPBuffer,
    num_experts: int,
) -> torch.Tensor:
    """Non-streaming baseline: dispatch -> compute on full recv_x -> combine.

    Args:
        x: [T, H] bf16, this rank's input tokens.
        w1_q: [E_local, H, 2*I] bf16, QuACK layout (gate+up concat on out dim).
        w2_q: [E_local, I, H] bf16, QuACK layout.
        topk_idx: [T, K] int64, GLOBAL expert ids per token.
        topk_weights: [T, K] float32.
        buffer: DeepEP Buffer constructed for this EP group.
        num_experts: total experts across all ranks.

    Returns:
        out: [T, H] bf16. Topk-weighted expert outputs reduced back to source rank.
    """
    world_size = buffer.group_size
    local_num_experts = num_experts // world_size

    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts, async_finish=False)

    recv_x, recv_topk_idx, recv_topk_weights, _, handle, _ = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        expert_alignment=1,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    assert isinstance(recv_x, torch.Tensor), "FP8 dispatch path not yet supported"

    if recv_x.size(0) == 0:
        empty = torch.zeros_like(recv_x)
        combined, _, _ = buffer.combine(
            empty, handle, async_finish=False, allocate_on_comm_stream=False
        )
        return combined

    N, K = recv_topk_idx.shape
    TK = N * K
    x_gather_idx, s_scatter_idx, _, expert_frequency_offset = build_routing_metadata(
        recv_topk_idx, local_num_experts
    )
    identity_idx = torch.arange(TK, dtype=torch.int32, device=recv_x.device)
    w1_padded = _pad_zero_expert(w1_q)
    w2_padded = _pad_zero_expert(w2_q)

    _, y1 = gemm_gated(
        recv_x,
        w1_padded,
        activation="swiglu",
        cu_seqlens_m=expert_frequency_offset,
        A_idx=x_gather_idx,
        dynamic_scheduler=False,
    )
    y2 = gemm(
        y1,
        w2_padded,
        cu_seqlens_m=expert_frequency_offset,
        A_idx=identity_idx,
        dynamic_scheduler=False,
    )

    o = aggregate_topk(
        y2, x_gather_idx, s_scatter_idx, recv_topk_weights, recv_x.size(0)
    )

    combined, _, _ = buffer.combine(
        o, handle, async_finish=False, allocate_on_comm_stream=False
    )
    return combined
