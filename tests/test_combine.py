"""End-to-end test for the consolidated streaming-MoE combine (intranode).

Exercises ``Buffer.combine`` with the Phase-D per-token gate. Combine sender
spins on ``y_done_per_token[r] >= combine_seq`` before pushing
``handle.o[r]`` back to ``r``'s origin rank; in production the release-store
of ``combine_seq`` is fired by kernel Y. Here we do not run kernel Y — we
manually populate ``handle.o`` with rank-tagged sentinel values and fill
``y_done_per_token`` with ``combine_seq`` so the gate clears
unconditionally, then call ``buf.combine`` and verify the cross-rank
reduction in ``recv_x``.

Two test cases:

  1. ``test_basic_combine`` — single dispatch + populated o + combine; assert
     the output matches an analytical reference (sum over contributing ranks
     of their per-rank sentinel value).
  2. ``test_multi_dispatch_combine`` — three sequential dispatches with
     different routing seeds and per-call manual fills. Verifies no cross-
     dispatch contamination (each combine reads its own dispatch's state
     from the corresponding ``handle``).

Driver convention: torchrun env-driven (``RANK`` / ``WORLD_SIZE`` /
``LOCAL_RANK``), matching ``test_streaming_dispatch.py``. Run as:

    torchrun --nproc_per_node=2 StreamEP/tests/test_combine.py
    torchrun --nproc_per_node=8 StreamEP/tests/test_combine.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from stream_ep import Buffer
from utils import cleanup_dist, make_inputs


def make_buffer(group, num_sms, hidden_bytes):
    Buffer.set_num_sms(num_sms)
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        nvl_bytes = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()), nvl_bytes)
        rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()), rdma_bytes)
    return Buffer(group, nvl_bytes, rdma_bytes)


def expected_combine_output(is_token_in_rank, num_tokens, hidden, num_ranks, dtype, device):
    """Analytical reference: each output[t] = sum over ranks R that hold any of
    t's K experts, of R's per-rank sentinel (= float(R)). With our token data
    set to ``x[r] = rank`` everywhere, the combine receiver atomic-adds these
    values across contributing ranks.

    is_token_in_rank: this rank's [num_tokens, num_ranks] bool routing matrix.
    Returns: [num_tokens, hidden] expected output.
    """
    # is_token_in_rank[t, r] for THIS rank's source tokens t. Reduce across r:
    # output[t] = sum over r where is_token_in_rank[t, r] is True, of r.
    contrib = torch.arange(num_ranks, dtype=torch.float32, device=device).unsqueeze(0)  # [1, R]
    out_per_token = (is_token_in_rank.float() * contrib).sum(dim=-1, keepdim=True)  # [T, 1]
    return out_per_token.expand(num_tokens, hidden).to(dtype).contiguous()


def populate_handle_o_with_rank_tag(handle, rank, num_recv_tokens, hidden, dtype):
    """Set handle.o[r, :] = rank for all valid r. Padding slots stay zero (won't
    be touched by combine sender because they don't exist in [0, T_recv))."""
    handle.o.fill_(float(rank))


def run_one_dispatch_combine(buf, x, topk_idx, topk_weights, is_token_in_rank,
                             num_experts, num_topk, hidden, num_ranks, rank,
                             tile_m, dispatch_seq, dtype, device):
    """One dispatch + manual o/y_done_per_token fill + combine + check."""
    pool, handle, _event = buf.dispatch(
        x, topk_idx, topk_weights, is_token_in_rank, num_experts,
        tile_m=tile_m, dispatch_seq=dispatch_seq,
    )
    torch.cuda.synchronize()

    # Manually populate handle.o with this rank's tag (kernel Y replacement).
    T_recv = handle.o.shape[0]
    populate_handle_o_with_rank_tag(handle, rank, T_recv, hidden, dtype)

    # Manually fire the per-token gate (kernel Y release-store replacement).
    handle.y_done_per_token.fill_(dispatch_seq)

    # Run combine. Use the same int as combine_seq for symmetry with the
    # production layer wrapper.
    out, _recv_topk = buf.combine(
        handle.o, handle,
        combine_seq=dispatch_seq,
    )
    torch.cuda.synchronize()

    # Reference: each output[t, :] = sum over ranks R that hold any of t's experts of R.
    expected = expected_combine_output(is_token_in_rank, x.shape[0], hidden, num_ranks, dtype, device)
    actual = out

    # Compare valid-shape rows. bf16 atol roomy enough for sums up to 64.
    diff = (actual.float() - expected.float()).abs().max().item()
    assert diff < 1e-2, (
        f"combine output mismatch (max abs diff = {diff:.4e})\n"
        f"  expected[0:4, 0]: {expected[:4, 0].cpu().tolist()}\n"
        f"  actual[0:4, 0]:   {actual[:4, 0].cpu().tolist()}\n"
        f"  rank={rank} dispatch_seq={dispatch_seq}"
    )
    return T_recv


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    num_sms = 24
    num_experts = 64
    num_topk = 4
    num_tokens = 256
    hidden = 256
    tile_m = 32
    dtype = torch.bfloat16
    hidden_bytes = hidden * 2

    buf = make_buffer(group, num_sms, hidden_bytes)

    # ─── Test 1: basic combine. Single dispatch + populated o + combine.
    x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
        num_tokens, hidden, num_topk, num_experts, world_size, rank, device, seed=123,
    )
    T_recv = run_one_dispatch_combine(
        buf, x, topk_idx, topk_weights, is_token_in_rank,
        num_experts, num_topk, hidden, world_size, rank,
        tile_m, dispatch_seq=1, dtype=dtype, device=device,
    )
    if rank == 0:
        print(f"PASS test_basic_combine: world={world_size} T_recv={T_recv}")

    # ─── Test 2: multi-dispatch combine. Three back-to-back dispatches with
    # different routing seeds; each combine should produce its own dispatch's
    # expected output (no cross-dispatch contamination).
    for seq, seed in enumerate([456, 789, 1011], start=2):
        x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
            num_tokens, hidden, num_topk, num_experts, world_size, rank, device, seed=seed,
        )
        T_recv = run_one_dispatch_combine(
            buf, x, topk_idx, topk_weights, is_token_in_rank,
            num_experts, num_topk, hidden, world_size, rank,
            tile_m, dispatch_seq=seq, dtype=dtype, device=device,
        )
        if rank == 0:
            print(f"PASS test_multi_dispatch_combine #{seq - 1} (seed={seed}): T_recv={T_recv}")

    if rank == 0:
        print(f"PASS: all combine validations on rank 0 (world={world_size})")

    cleanup_dist()


if __name__ == "__main__":
    main()
