"""Targeted test for ``internode::cached_notify_combine``'s sentinel-encoding
half — the in-place reverse-order rewrite of ``send_rdma_head`` and
``send_nvl_head`` that combine's receivers consume to skip past gaps in the
ring buffers.

The kernel does two architectural jobs (see api.cuh):
  1. Buffer cleanup: zero head/tail control regions of this iter's combine
     RDMA + NVL ring buffers (so combine's queues start at 0 each call).
  2. Reverse-order sentinel encoding: in-place rewrite of `send_rdma_head`
     and `send_nvl_head` such that entries `< 0` (no contribution from that
     source for this token) get replaced with `-last_head - 1`, where
     `last_head` is the next real head ahead in reverse traversal order.

This test verifies (2) directly. (1) touches the symmetric NVSHMEM heap and
the rank's NVL IPC slab — awkward to inspect from Python; will be exercised
end-to-end when ``combine_main_kernel`` reads from those buffers.

Driven via ``Buffer.runtime.cached_notify_combine_test`` — a thin C++
wrapper around the kernel that mirrors the production call site (same kernel,
same args). Test wrapper goes away when ``Buffer::internode_combine`` lands.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

import stream_ep
from stream_ep import Buffer

from utils import cleanup_dist


_NUM_MAX_NVL_PEERS = 8


def make_inputs(num_tokens: int, hidden: int, num_topk: int, num_experts: int,
                num_ranks: int, rank: int, device: torch.device,
                *, seed: int = 123):
    g = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn((num_tokens, hidden), generator=g, device=device,
                    dtype=torch.bfloat16)

    idx = torch.randint(0, num_experts, (num_tokens, num_topk),
                        generator=g, device=device, dtype=torch.int64)
    sentinel = torch.rand((num_tokens, num_topk), generator=g,
                          device=device) < 0.05
    idx = torch.where(sentinel, torch.full_like(idx, -1), idx)
    topk_idx = idx.to(stream_ep.topk_idx_t)

    topk_weights = torch.rand((num_tokens, num_topk), generator=g,
                              device=device, dtype=torch.float32)

    num_local_experts = num_experts // num_ranks
    rank_idx = torch.where(topk_idx >= 0, topk_idx // num_local_experts,
                           torch.full_like(topk_idx, -1))
    is_token_in_rank = torch.zeros((num_tokens, num_ranks),
                                   dtype=torch.bool, device=device)
    for r in range(num_ranks):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)
    return x, topk_idx, topk_weights, is_token_in_rank


def channel_task_range(num_tokens: int, num_channels: int, c: int) -> tuple[int, int]:
    """Mirrors ``utils.cuh:get_channel_task_range``: ceiling-partition of
    ``num_tokens`` across ``num_channels``; channel ``c`` owns
    ``[c * ceil_div(N, C), (c+1) * ceil_div(N, C))`` clipped to ``[0, N]``.
    """
    per = (num_tokens + num_channels - 1) // num_channels
    start = min(per * c, num_tokens)
    end = min(start + per, num_tokens)
    return start, end


def encode_rdma_head_reference(send_rdma_head: torch.Tensor,
                               num_channels: int) -> torch.Tensor:
    """Reverse-order sentinel encoding for ``combined_rdma_head``. Per channel,
    per dst_rdma_rank lane, walk ``[token_start, token_end)`` in reverse and
    rewrite ``< 0`` entries with ``-last_head - 1`` (per-lane ``last_head``,
    initialized to ``1 << 25``).
    """
    out = send_rdma_head.clone()
    num_combined_tokens, num_rdma_ranks = out.shape
    out_cpu = out.cpu()
    for c in range(num_channels):
        token_start, token_end = channel_task_range(num_combined_tokens, num_channels, c)
        last_head = [1 << 25] * num_rdma_ranks
        for t in range(token_end - 1, token_start - 1, -1):
            for lane in range(num_rdma_ranks):
                v = int(out_cpu[t, lane].item())
                if v < 0:
                    out_cpu[t, lane] = -last_head[lane] - 1
                else:
                    last_head[lane] = v
    return out_cpu.to(out.device)


def encode_nvl_head_reference(send_nvl_head: torch.Tensor,
                              rdma_channel_prefix_matrix: torch.Tensor,
                              recv_rdma_rank_prefix_sum: torch.Tensor,
                              num_channels: int) -> torch.Tensor:
    """Reverse-order sentinel encoding for ``combined_nvl_head``. Per
    dst_rdma_rank, per channel, the kernel walks
    ``rdma_channel_prefix_matrix[dst_rdma_rank, c-1]..rdma_channel_prefix_matrix[dst_rdma_rank, c]``
    (shifted by ``recv_rdma_rank_prefix_sum[dst_rdma_rank-1]``) in reverse,
    sentinel-encoding per NUM_MAX_NVL_PEERS lane.
    """
    out = send_nvl_head.clone().cpu()
    rcpm = rdma_channel_prefix_matrix.cpu()
    rrps = recv_rdma_rank_prefix_sum.cpu()
    num_rdma_ranks = rcpm.size(0)
    for dst_rdma_rank in range(num_rdma_ranks):
        shift = 0 if dst_rdma_rank == 0 else int(rrps[dst_rdma_rank - 1].item())
        for c in range(num_channels):
            token_start = (0 if c == 0
                           else int(rcpm[dst_rdma_rank, c - 1].item())) + shift
            token_end = int(rcpm[dst_rdma_rank, c].item()) + shift
            last_head = [1 << 25] * _NUM_MAX_NVL_PEERS
            for t in range(token_end - 1, token_start - 1, -1):
                for lane in range(_NUM_MAX_NVL_PEERS):
                    v = int(out[t, lane].item())
                    if v < 0:
                        out[t, lane] = -last_head[lane] - 1
                    else:
                        last_head[lane] = v
    return out.to(send_nvl_head.device)


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
    num_channels = num_sms // 2

    hidden_bytes = hidden * 2
    nvl_bytes, rdma_bytes = 0, 0
    for cfg_ in (Buffer.get_dispatch_config(world_size),
                 Buffer.get_combine_config(world_size)):
        nvl_bytes  = max(cfg_.get_nvl_buffer_size_hint(hidden_bytes,  world_size), nvl_bytes)
        rdma_bytes = max(cfg_.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes)
    buf = Buffer(group, nvl_bytes, rdma_bytes)

    if not hasattr(buf.runtime, 'cached_notify_combine_test'):
        if rank == 0:
            print("[skip] Buffer.runtime.cached_notify_combine_test not built — "
                  "kernel implementation pending.", flush=True)
        cleanup_dist()
        return

    cfg = Buffer.get_combine_config(world_size)

    x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
        num_tokens, hidden, num_topk, num_experts, world_size, rank, device,
        seed=42)

    out = buf.runtime.internode_dispatch(
        x, topk_idx, topk_weights, is_token_in_rank,
        num_experts, 1, tile_m, 1, cfg)
    torch.cuda.synchronize()

    rdma_before = out.send_rdma_head.clone()
    nvl_before = out.send_nvl_head.clone()

    expected_rdma = encode_rdma_head_reference(rdma_before, num_channels)
    expected_nvl = encode_nvl_head_reference(
        nvl_before, out.recv_rdma_channel_prefix_matrix,
        out.recv_rdma_rank_prefix_sum, num_channels)

    rdma_after, nvl_after = buf.runtime.cached_notify_combine_test(out, cfg)
    torch.cuda.synchronize()

    assert torch.equal(rdma_after, expected_rdma), (
        f"rank={rank}: combined_rdma_head sentinel encoding mismatch; "
        f"first deviating index "
        f"{(rdma_after != expected_rdma).nonzero()[:8].cpu().tolist()}\n"
        f"  before[0:4]:   {rdma_before[:4].cpu().tolist()}\n"
        f"  expected[0:4]: {expected_rdma[:4].cpu().tolist()}\n"
        f"  actual[0:4]:   {rdma_after[:4].cpu().tolist()}")

    assert torch.equal(nvl_after, expected_nvl), (
        f"rank={rank}: combined_nvl_head sentinel encoding mismatch; "
        f"first deviating index "
        f"{(nvl_after != expected_nvl).nonzero()[:8].cpu().tolist()}\n"
        f"  before[0:4]:   {nvl_before[:4].cpu().tolist()}\n"
        f"  expected[0:4]: {expected_nvl[:4].cpu().tolist()}\n"
        f"  actual[0:4]:   {nvl_after[:4].cpu().tolist()}")

    if rank == 0:
        T = rdma_before.shape[0]
        T_rdma = nvl_before.shape[0]
        rdma_neg_before = (rdma_before < 0).sum().item()
        nvl_neg_before = (nvl_before < 0).sum().item()
        print(f"PASS: cached_notify_combine sentinel encoding correct "
              f"(world={world_size}, T={T}, T_rdma={T_rdma}, "
              f"rdma neg-entries before/after = {rdma_neg_before}/"
              f"{(rdma_after < 0).sum().item()}, "
              f"nvl neg-entries before/after = {nvl_neg_before}/"
              f"{(nvl_after < 0).sum().item()})", flush=True)

    cleanup_dist()


if __name__ == "__main__":
    main()
