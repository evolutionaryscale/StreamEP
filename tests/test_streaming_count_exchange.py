"""Smoke test for Buffer.streaming_count_exchange (Phase A.1).

Each rank computes its expected per-(channel, src_rank, local_expert) inbox
locally from the global topk_idx, then compares against the kernel result.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from deep_ep import Buffer


def make_topk_idx(num_tokens_per_rank: int, num_topk: int, num_experts: int, rank: int, device):
    g = torch.Generator(device=device).manual_seed(123 + rank)
    return torch.randint(0, num_experts, (num_tokens_per_rank, num_topk),
                         generator=g, device=device, dtype=torch.int64)


def channel_of_token(t, num_tokens, num_channels):
    per_channel = (num_tokens + num_channels - 1) // num_channels
    return t // per_channel


def expected_recv_count(all_topk_idx, num_channels, num_ranks, num_local_experts, my_rank):
    """Compute expected recv_count[c, src, e_local] from gathered topk_idx tensors."""
    out = torch.zeros(num_channels, num_ranks, num_local_experts, dtype=torch.int32)
    expert_lo = my_rank * num_local_experts
    expert_hi = (my_rank + 1) * num_local_experts
    for src, topk in enumerate(all_topk_idx):
        T, K = topk.shape
        for t in range(T):
            c = channel_of_token(t, T, num_channels)
            for k in range(K):
                e = topk[t, k].item()
                if expert_lo <= e < expert_hi:
                    out[c, src, e - expert_lo] += 1
    return out


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    num_sms = 24
    Buffer.set_num_sms(num_sms)
    num_channels = num_sms // 2
    num_experts = 64
    num_local_experts = num_experts // world_size
    num_topk = 4
    num_tokens = 256

    hidden_bytes = 2048 * 2
    nvl_bytes = 0
    rdma_bytes = 0
    for cfg in (Buffer.get_dispatch_config(world_size), Buffer.get_combine_config(world_size)):
        nvl_bytes = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, world_size), nvl_bytes)
        rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes)

    buf = Buffer(group, nvl_bytes, rdma_bytes, num_qps_per_rank=Buffer.num_sms)

    topk_idx = make_topk_idx(num_tokens, num_topk, num_experts, rank, device)

    all_topk = [torch.empty_like(topk_idx) for _ in range(world_size)]
    dist.all_gather(all_topk, topk_idx, group=group)
    all_topk_cpu = [t.cpu() for t in all_topk]

    recv_count = buf.streaming_count_exchange(topk_idx, num_experts)
    torch.cuda.synchronize()

    expected = expected_recv_count(all_topk_cpu, num_channels, world_size, num_local_experts, rank)
    actual = recv_count.cpu().to(torch.int32)

    if not torch.equal(actual, expected):
        diff = (actual - expected)
        n_diff = (diff != 0).sum().item()
        if rank == 0:
            print(f"MISMATCH: {n_diff} cells differ")
            print(f"  actual sum   = {actual.sum().item()}")
            print(f"  expected sum = {expected.sum().item()}")
            mask = diff != 0
            idx = mask.nonzero(as_tuple=False)[:5]
            for r in idx.tolist():
                c, s, e = r
                print(f"  [c={c} src={s} e={e}]: actual={actual[c,s,e]} expected={expected[c,s,e]}")
        raise SystemExit(1)

    tile_m = 128
    md = buf.streaming_metadata_init(recv_count, tile_m=tile_m)

    expert_freq_expected = expected.sum(dim=(0, 1)).to(torch.int32)
    assert torch.equal(md.expert_frequency.cpu(), expert_freq_expected), \
        f"expert_frequency mismatch: {md.expert_frequency.cpu()} vs {expert_freq_expected}"

    cumsum_expected = torch.zeros(num_local_experts + 1, dtype=torch.int32)
    cumsum_expected[1:] = expert_freq_expected.cumsum(0).to(torch.int32)
    assert torch.equal(md.expert_frequency_offset.cpu(), cumsum_expected)

    flat = expected.view(num_channels * world_size, num_local_experts)
    base_flat_exp = torch.zeros_like(flat)
    if num_channels * world_size > 1:
        base_flat_exp[1:] = flat[:-1].cumsum(0).to(torch.int32)
    base_expected = (base_flat_exp.view(num_channels, world_size, num_local_experts)
                     + cumsum_expected[:-1].view(1, 1, num_local_experts))
    assert torch.equal(md.base.cpu(), base_expected), \
        f"base mismatch first cell: actual={md.base[0,0,0].item()} expected={base_expected[0,0,0].item()}"

    tiles_per_expert = ((expert_freq_expected + tile_m - 1) // tile_m).to(torch.int32)
    expected_total_tiles = int(tiles_per_expert.sum().item())
    actual_total_tiles = int(md.total_tiles_device.cpu().item())
    assert actual_total_tiles == expected_total_tiles, \
        f"total_tiles {actual_total_tiles} vs expected {expected_total_tiles}"

    per_sr_expected = expected.sum(dim=2).to(torch.int32)
    assert torch.equal(md.per_source_rank_remaining.cpu(), per_sr_expected)

    cum_expected = torch.zeros(num_local_experts + 1, dtype=torch.int32)
    cum_expected[1:] = tiles_per_expert.cumsum(0).to(torch.int32)
    assert torch.equal(md.cumulative_tiles_before_e.cpu(), cum_expected)

    if rank == 0:
        print(f"PASS: rank={rank} world={world_size} sum={actual.sum().item()} "
              f"total_tiles={actual_total_tiles}")


if __name__ == "__main__":
    main()
