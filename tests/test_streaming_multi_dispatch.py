"""Multi-dispatch correctness test for streaming dispatch (pool layout).

Runs N>1 dispatches on the same Buffer with different routing each time and
validates pool placement on each. Catches state leaks between dispatches in:
  - the IPC count_exchange e_inbox / u_inbox slabs,
  - the per-tile pool_arrival_count counters,
  - any device-resident streaming buffers reused across dispatches.

Validates:
  1. pool[s] data tag matches its source rank (per-pool-slot).
  2. Pool layout — slots fall in the right expert's pool region.
  3. Per-(recv_token, k) coverage — every routed-to-local pair has exactly
     one pool slot.
  4. tile_ready[i] == dispatch_seq for i in [0, total_tiles).
  5. pool_arrival_target sized correctly per tile.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from stream_ep import Buffer


def make_inputs(num_tokens, hidden, num_topk, num_experts, num_ranks, rank, device, seed):
    g = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device=device) * rank
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk),
                             generator=g, device=device, dtype=torch.int64)
    topk_weights = torch.rand((num_tokens, num_topk), generator=g, device=device, dtype=torch.float32)

    num_local_experts = num_experts // num_ranks
    rank_idx = topk_idx // num_local_experts
    is_token_in_rank = torch.zeros((num_tokens, num_ranks), dtype=torch.bool, device=device)
    for r in range(num_ranks):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)
    return x, topk_idx, topk_weights, is_token_in_rank


def validate_dispatch(
    rank, world_size, num_local_experts, tile_m,
    x, topk_idx, is_token_in_rank,
    pool, handle, dispatch_seq, group,
):
    # Gather inputs across ranks to compute expected outputs.
    all_topk = [torch.empty_like(topk_idx) for _ in range(world_size)]
    dist.all_gather(all_topk, topk_idx, group=group)
    all_is_in_rank = [torch.empty_like(is_token_in_rank) for _ in range(world_size)]
    dist.all_gather(all_is_in_rank, is_token_in_rank, group=group)

    # Expected per-recv-token (src_rank, src_token_idx, topk_local) tables.
    e_lo, e_hi = rank * num_local_experts, (rank + 1) * num_local_experts
    expected_src_rank = []
    expected_topk_local = []
    for src in range(world_size):
        src_in = all_is_in_rank[src][:, rank]
        idx_in_src = torch.nonzero(src_in, as_tuple=False).flatten()
        src_topk = all_topk[src][idx_in_src]
        local_mask = (src_topk >= e_lo) & (src_topk < e_hi)
        local_topk = torch.where(local_mask, src_topk - e_lo, torch.full_like(src_topk, -1))
        expected_src_rank.append(torch.full((idx_in_src.numel(),), src, dtype=torch.int32, device=topk_idx.device))
        expected_topk_local.append(local_topk)
    expected_src_rank = torch.cat(expected_src_rank, dim=0).cpu()
    expected_topk_local = torch.cat(expected_topk_local, dim=0).cpu()
    T_recv = expected_src_rank.shape[0]

    pool_cpu = pool.cpu()
    pool_recv_token = handle.pool_recv_token.cpu()
    pool_k_slot = handle.pool_k_slot.cpu()
    expert_pool_block_offset = handle.expert_pool_block_offset.cpu()
    pool_arrival_target = handle.pool_arrival_target.cpu()
    expert_frequency = handle.expert_frequency.cpu()
    total_tiles = handle.total_tiles
    TK_padded = total_tiles * tile_m

    # (1+2+3) Per-pool-slot validation.
    seen_rk = torch.zeros((T_recv, topk_idx.shape[1]), dtype=torch.bool)
    for s in range(TK_padded):
        rt = int(pool_recv_token[s].item())
        k = int(pool_k_slot[s].item())
        if rt < 0:
            continue
        e_local = int(expected_topk_local[rt, k].item())
        assert e_local >= 0, (
            f"dispatch {dispatch_seq}: slot {s} → (rt={rt}, k={k}) but expected_topk_local says nonlocal"
        )
        block_start = int(expert_pool_block_offset[e_local].item()) * tile_m
        block_end = int(expert_pool_block_offset[e_local + 1].item()) * tile_m
        assert block_start <= s < block_end, (
            f"dispatch {dispatch_seq}: slot {s} for expert {e_local} outside [{block_start}, {block_end})"
        )
        src_rank = int(expected_src_rank[rt].item())
        actual = pool_cpu[s, 0].to(torch.int32).item()
        assert actual == src_rank, (
            f"dispatch {dispatch_seq}: slot {s} pool[s,0]={actual} != src_rank {src_rank}"
        )
        assert not seen_rk[rt, k], f"dispatch {dispatch_seq}: (rt={rt}, k={k}) appears multiple times"
        seen_rk[rt, k] = True
    expected_seen = (expected_topk_local >= 0)
    assert torch.equal(seen_rk, expected_seen), (
        f"dispatch {dispatch_seq}: pool covers {seen_rk.sum()} pairs, expected {expected_seen.sum()}"
    )

    # (4) tile_ready
    ready = handle.tile_ready[:total_tiles].cpu()
    assert (ready == dispatch_seq).all(), (
        f"dispatch {dispatch_seq}: tile_ready mismatches at {(ready != dispatch_seq).nonzero().flatten()[:8]}"
    )

    # (5) pool_arrival_target
    for e in range(num_local_experts):
        e_block_start = int(expert_pool_block_offset[e].item())
        e_block_end = int(expert_pool_block_offset[e + 1].item())
        n_e = int(expert_frequency[e].item())
        for tile_id in range(e_block_start, e_block_end):
            tile_in_e = tile_id - e_block_start
            target = int(pool_arrival_target[tile_id].item())
            expected = (n_e - tile_in_e * tile_m) if tile_id == e_block_end - 1 else tile_m
            assert target == expected, (
                f"dispatch {dispatch_seq}: pool_arrival_target[{tile_id}]={target} != {expected}"
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

    seeds = [123, 456, 789]
    for i, seed in enumerate(seeds):
        dispatch_seq = i + 1
        x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
            num_tokens, hidden, num_topk, num_experts, world_size, rank, device, seed=seed)

        pool, handle, _event = buf.dispatch(
            x, topk_idx, topk_weights, is_token_in_rank, num_experts,
            tile_m=tile_m, dispatch_seq=dispatch_seq,
        )
        torch.cuda.synchronize()

        T_recv = validate_dispatch(
            rank, world_size, num_local_experts, tile_m,
            x, topk_idx, is_token_in_rank,
            pool, handle, dispatch_seq, group,
        )

        if rank == 0:
            print(f"PASS dispatch {dispatch_seq} (seed={seed}): "
                  f"T_recv={T_recv} total_tiles={handle.total_tiles}")

    if rank == 0:
        print(f"PASS: all {len(seeds)} multi-dispatch validations on rank 0 (world={world_size})")


if __name__ == "__main__":
    main()
