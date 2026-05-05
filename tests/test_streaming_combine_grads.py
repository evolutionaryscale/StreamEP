"""End-to-end test for the streaming-MoE backward combine_grads.

Exercises ``Buffer.combine_grads`` (intranode). Uses the SAME underlying
``combine_main_kernel`` as fwd ``Buffer.combine`` — the test verifies the
sender's slot-lookup weight payload (loads ``weight_grads[slot]`` via
``recv_token_to_slots[r, k]``, ships 0 for non-local k) and the receiver's
sum-reduction in both halves.

Trick: per-rank tag dL_dx_per_r and weight_grads with ``float(rank)``. Then
analytical reference is:

  - dL_dx[t] = sum over ranks R that t was sent to of float(R) — same as fwd.
  - dL_dtopk_weights[t, k] = float(topk_idx[t, k] / num_local_experts) — the
    one rank that hosts expert topk_idx[t, k] writes float(R), all others 0.

Driver convention matches the rest of the streaming tests:

    torchrun --nproc_per_node=2 DeepEP/tests/test_streaming_combine_grads.py
    torchrun --nproc_per_node=8 DeepEP/tests/test_streaming_combine_grads.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from stream_ep import Buffer


def make_inputs(num_tokens, hidden, num_topk, num_experts, num_ranks, rank, device, seed=123):
    g = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.full((num_tokens, hidden), float(rank), dtype=torch.bfloat16, device=device)
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk),
                             generator=g, device=device, dtype=torch.int64)
    topk_weights = torch.rand((num_tokens, num_topk), generator=g, device=device, dtype=torch.float32)

    num_local_experts = num_experts // num_ranks
    rank_idx = topk_idx // num_local_experts
    is_token_in_rank = torch.zeros((num_tokens, num_ranks), dtype=torch.bool, device=device)
    for r in range(num_ranks):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)
    return x, topk_idx, topk_weights, is_token_in_rank


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    num_sms = 24
    Buffer.set_num_sms(num_sms)
    num_experts = 64
    num_local_experts = num_experts // world_size
    num_topk = 4
    num_tokens = 256
    hidden = 256
    tile_m = 32

    hidden_bytes = hidden * 2
    nvl_bytes = 0
    rdma_bytes = 0
    for cfg in (Buffer.get_dispatch_config(world_size), Buffer.get_combine_config(world_size)):
        nvl_bytes = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, world_size), nvl_bytes)
        rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes)
    buf = Buffer(group, nvl_bytes, rdma_bytes)

    x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
        num_tokens, hidden, num_topk, num_experts, world_size, rank, device)

    pool, handle, _event = buf.dispatch(
        x, topk_idx, topk_weights, is_token_in_rank, num_experts,
        tile_m=tile_m, dispatch_seq=1,
    )
    torch.cuda.synchronize()

    T_recv = handle.o.shape[0]
    TK_padded = handle.pool.shape[0]

    # Per-rank tagged inputs: dL_dx_per_r and weight_grads have value float(rank)
    # everywhere. Receiver's K-way sum then reconstructs an analytical reference.
    dL_dx_per_r = torch.full((T_recv, hidden), float(rank), dtype=torch.bfloat16, device=device)
    weight_grads = torch.full((TK_padded,), float(rank), dtype=torch.float32, device=device)

    # Manually fire the bwd_compute_done_per_token gate (kernel_a_bwd's
    # release-store replacement for this isolated test).
    bwd_compute_done_per_token = torch.full((T_recv,), 1, dtype=torch.int64, device=device)

    # Run combine_grads.
    dL_dx, dL_dtopk_weights = buf.combine_grads(
        dL_dx_per_r, handle, weight_grads, bwd_compute_done_per_token,
        dispatch_seq=1,
    )
    torch.cuda.synchronize()

    assert dL_dx.shape == (num_tokens, hidden)
    assert dL_dx.dtype == torch.bfloat16
    assert dL_dtopk_weights.shape == (num_tokens, num_topk)
    assert dL_dtopk_weights.dtype == torch.float32

    # ── Reference for dL_dx[t] ───────────────────────────────────────────
    # Each rank R that received t contributed float(R). Sum over R-where-t-sent.
    contrib = torch.arange(world_size, dtype=torch.float32, device=device).unsqueeze(0)  # [1, R]
    expected_dL_dx_per_token = (is_token_in_rank.float() * contrib).sum(dim=-1, keepdim=True)
    expected_dL_dx = expected_dL_dx_per_token.expand(num_tokens, hidden).to(torch.bfloat16).contiguous()

    diff_x = (dL_dx.float() - expected_dL_dx.float()).abs().max().item()
    assert diff_x < 1e-2, (
        f"dL_dx mismatch (max abs diff = {diff_x:.4e})\n"
        f"  expected[0:4, 0]: {expected_dL_dx[:4, 0].cpu().tolist()}\n"
        f"  actual[0:4, 0]:   {dL_dx[:4, 0].cpu().tolist()}\n"
        f"  rank={rank}"
    )

    # ── Reference for dL_dtopk_weights[t, k] ──────────────────────────────
    # For each (t, k), exactly one rank R hosts expert topk_idx[t, k]. R's
    # weight_grads[slot] = float(R); other senders' packet[k] = 0. Sum reduces
    # to float(R) = float(topk_idx[t, k] // num_local_experts).
    expected_weights = (topk_idx // num_local_experts).to(torch.float32)
    diff_w = (dL_dtopk_weights - expected_weights).abs().max().item()
    assert diff_w < 1e-3, (
        f"dL_dtopk_weights mismatch (max abs diff = {diff_w:.4e})\n"
        f"  expected[0:4]: {expected_weights[:4].cpu().tolist()}\n"
        f"  actual[0:4]:   {dL_dtopk_weights[:4].cpu().tolist()}\n"
        f"  rank={rank}"
    )

    # Re-run with a different bwd seq to verify per-call statefulness.
    bwd_compute_done_per_token2 = torch.full((T_recv,), 2, dtype=torch.int64, device=device)
    dL_dx2, _ = buf.combine_grads(
        dL_dx_per_r, handle, weight_grads, bwd_compute_done_per_token2,
        dispatch_seq=2,
    )
    torch.cuda.synchronize()
    diff2 = (dL_dx2.float() - expected_dL_dx.float()).abs().max().item()
    assert diff2 < 1e-2, f"second-call dL_dx mismatch (max abs diff = {diff2:.4e})"

    if rank == 0:
        print(f"PASS: rank={rank} world={world_size} T_recv={T_recv} TK_padded={TK_padded}")


if __name__ == "__main__":
    main()
