"""End-to-end test for the streaming-MoE internode combine (pool layout).

Mirrors ``tests/test_streaming_combine.py`` (intranode combine) for the
internode topology — 2 RDMA × 8 NVL = 16 GPUs via ``./srun_internode.sh``.
Drives ``Buffer.dispatch`` and ``Buffer.combine`` (which route through the
internode entry points via the topology branch when ``num_rdma_ranks > 1``).

The combine reduction is direction- and topology-uniform (same arg surface
intranode's unified ``combine_main_kernel`` uses); the internode delta is
the three-warp-role wire format on top of the same per-(t, k) reduction
semantics. Test verifies:

  1. Cross-rank reduction at production shape — for the rank-tag input
     ``x[r] = float(rank)``, ``combined_x[t]`` should equal the sum of
     contributing ranks' tags (exactly the contributing set encoded in
     ``is_token_in_rank[t, :]``).
  2. ``recv_topk_weights[t, k]`` should match ``topk_weights[t, k]`` for
     every (t, k) routing to a local expert on any rank, and 0 otherwise.
  3. Multi-iter ``combine_seq`` reuse (1 → 2 → 3) with different routing
     seeds — each combine sees its own dispatch's state, no cross-iter
     contamination.

Note on the streaming gate: this test fills ``y_done_per_token``
with ``combine_seq`` before invoking combine (gate trivially open),
mirroring the intranode combine test. End-to-end gate exercise (kernel-Y
release fires the gate while combine is spinning) requires the production
pipeline with real compute streams — closing the gate then firing it
from the same stream the kernel runs on deadlocks. That coverage lives
in the multi-stream pipeline tests, not here.

Edge cases planted in inputs:
  - ~5% ``-1`` sentinels in ``topk_idx`` (skip branch, no contribution).
  - One expert pinned to receive zero tokens this iter (forces empty-
    expert branch through combine's reverse path).

Driver convention: torchrun env-driven (``RANK`` / ``WORLD_SIZE`` /
``LOCAL_RANK``), matching ``test_streaming_dispatch_internode.py``.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from stream_ep import Buffer

from utils import cleanup_dist, make_inputs


def expected_combine_output(is_token_in_rank: torch.Tensor,
                            num_tokens: int, hidden: int, num_ranks: int,
                            dtype: torch.dtype,
                            device: torch.device) -> torch.Tensor:
    """For ``x[r] = float(rank)``, ``combined_x[t] = sum over r where
    is_token_in_rank[t, r], of float(r)``. Each contributing rank ships
    back its own tag value via combine; receiver sums K_dst contributions.
    """
    contrib = torch.arange(num_ranks, dtype=torch.float32,
                           device=device).unsqueeze(0)
    out_per_token = (is_token_in_rank.float() * contrib).sum(
        dim=-1, keepdim=True)
    return out_per_token.expand(num_tokens, hidden).to(dtype).contiguous()


def expected_recv_topk_weights(topk_idx: torch.Tensor,
                               topk_weights: torch.Tensor,
                               num_local_experts_per_rank: int) -> torch.Tensor:
    """For (t, k) routing to *any* rank's local expert (i.e. topk_idx[t, k]
    >= 0 in our convention; sentinels = -1), recv side should reconstruct
    topk_weights[t, k]. For sentinel slots, expected is 0.

    The combine kernel ships ``per_slot_weights[recv_token_to_slots[r, k]]``
    back to the source per (t, k); for non-local k on a particular sender
    rank the slot is -1 so 0 is shipped. Across the K destination ranks
    that hold (t, k)'s expert, exactly one ships the real weight; the
    sum equals topk_weights[t, k] for valid k, 0 for sentinel.
    """
    expected = torch.where(topk_idx >= 0, topk_weights,
                           torch.zeros_like(topk_weights))
    return expected


def run_one_dispatch_combine(buf: Buffer, x: torch.Tensor,
                             topk_idx: torch.Tensor,
                             topk_weights: torch.Tensor,
                             is_token_in_rank: torch.Tensor,
                             num_experts: int, num_topk: int, hidden: int,
                             world_size: int, rank: int, tile_m: int,
                             dispatch_seq: int, combine_seq: int,
                             *, dtype: torch.dtype,
                             device: torch.device) -> int:
    """Drive one dispatch + manual o-fill + combine cycle and check output."""
    _, handle, _ = buf.dispatch(
        x, topk_idx, topk_weights, is_token_in_rank, num_experts,
        tile_m=tile_m, dispatch_seq=dispatch_seq)
    torch.cuda.synchronize()

    T_recv = handle.o.shape[0]

    handle.o.fill_(float(rank))
    handle.y_done_per_token.fill_(combine_seq)

    recv_x, recv_topk = buf.combine(
        handle.o, handle, combine_seq=combine_seq)

    torch.cuda.synchronize()

    expected = expected_combine_output(
        is_token_in_rank, x.shape[0], hidden, world_size, dtype, device)
    diff = (recv_x.float() - expected.float()).abs().max().item()
    assert diff < 1e-2, (
        f"combine output mismatch (max abs diff = {diff:.4e}); "
        f"rank={rank} dispatch_seq={dispatch_seq} combine_seq={combine_seq}\n"
        f"  expected[0:4, 0]: {expected[:4, 0].cpu().tolist()}\n"
        f"  actual[0:4, 0]:   {recv_x[:4, 0].cpu().tolist()}")

    num_local_experts_per_rank = num_experts // world_size
    expected_w = expected_recv_topk_weights(
        topk_idx, topk_weights, num_local_experts_per_rank)
    diff_w = (recv_topk - expected_w).abs().max().item()
    assert diff_w < 1e-4, (
        f"recv_topk_weights mismatch (max abs diff = {diff_w:.4e}); "
        f"rank={rank} dispatch_seq={dispatch_seq}\n"
        f"  expected[0:4]: {expected_w[:4].cpu().tolist()}\n"
        f"  actual[0:4]:   {recv_topk[:4].cpu().tolist()}")

    return T_recv


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    assert world_size % 8 == 0 and world_size > 8, (
        f"This test requires multi-RDMA (world_size > 8, world_size % 8 == 0); "
        f"got {world_size}")

    num_sms = 24
    Buffer.set_num_sms(num_sms)
    num_experts = 64
    num_topk = 4
    num_tokens = 256
    hidden = 256
    tile_m = 32
    dtype = torch.bfloat16

    hidden_bytes = hidden * 2
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (Buffer.get_dispatch_config(world_size),
                Buffer.get_combine_config(world_size)):
        nvl_bytes  = max(cfg.get_nvl_buffer_size_hint(hidden_bytes,  world_size), nvl_bytes)
        rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes)
    buf = Buffer(group, nvl_bytes, rdma_bytes)

    x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
        num_tokens, hidden, num_topk, num_experts, world_size, rank, device,
        seed=123, plant_sentinels=True)
    T_recv = run_one_dispatch_combine(
        buf, x, topk_idx, topk_weights, is_token_in_rank,
        num_experts, num_topk, hidden, world_size, rank, tile_m,
        dispatch_seq=1, combine_seq=1,
        dtype=dtype, device=device)
    if rank == 0:
        print(f"PASS test_basic_combine_internode: world={world_size} "
              f"T_recv={T_recv}", flush=True)

    for seq, seed in enumerate([456, 789, 1011], start=2):
        x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
            num_tokens, hidden, num_topk, num_experts, world_size, rank, device,
            seed=seed, plant_sentinels=True)
        T_recv = run_one_dispatch_combine(
            buf, x, topk_idx, topk_weights, is_token_in_rank,
            num_experts, num_topk, hidden, world_size, rank, tile_m,
            dispatch_seq=seq, combine_seq=seq,
            dtype=dtype, device=device)
        if rank == 0:
            print(f"PASS test_multi_iter_combine_internode #{seq - 1} "
                  f"(seed={seed}): T_recv={T_recv}", flush=True)

    plant_e = 0
    x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
        num_tokens, hidden, num_topk, num_experts, world_size, rank, device,
        seed=2024, plant_sentinels=True, plant_empty_expert=plant_e)
    T_recv = run_one_dispatch_combine(
        buf, x, topk_idx, topk_weights, is_token_in_rank,
        num_experts, num_topk, hidden, world_size, rank, tile_m,
        dispatch_seq=10, combine_seq=10,
        dtype=dtype, device=device)
    if rank == 0:
        print(f"PASS test_combine_internode_empty_expert (e={plant_e}): "
              f"world={world_size} T_recv={T_recv}", flush=True)

    if rank == 0:
        print(f"PASS: all internode combine validations on rank 0 "
              f"(world={world_size})", flush=True)

    cleanup_dist()


if __name__ == "__main__":
    main()
