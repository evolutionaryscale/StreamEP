"""Multi-dispatch correctness test for streaming dispatch.

Runs N>1 dispatches on the same Buffer with different routing each time and
validates correctness on each. Catches state leaks between dispatches in:
  - the IPC count_exchange e_inbox / u_inbox slabs,
  - any device-resident streaming buffers reused across dispatches.

Validates:
  1. recv_x value (source-rank tag matches expected per-row src).
  2. recv_topk_idx — local-id mapping per dispatch.
  3. tile_remaining all-zero post-finalize.
  4. tile_ready[i] == dispatch_seq for i in [0, total_tiles).
  5. tile_records consistency with recv_topk_idx (every valid (r, k) pair
     appears in exactly one tile slot).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from deep_ep import Buffer


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
    recv_x, recv_topk_idx, handle, dispatch_seq, group,
):
    # Gather inputs across ranks to compute expected outputs.
    all_x = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(all_x, x, group=group)
    all_topk = [torch.empty_like(topk_idx) for _ in range(world_size)]
    dist.all_gather(all_topk, topk_idx, group=group)
    all_is_in_rank = [torch.empty_like(is_token_in_rank) for _ in range(world_size)]
    dist.all_gather(all_is_in_rank, is_token_in_rank, group=group)

    expected_recv_x, expected_recv_topk = [], []
    for src in range(world_size):
        src_x = all_x[src]
        src_in = all_is_in_rank[src][:, rank]
        idx_in_src = torch.nonzero(src_in, as_tuple=False).flatten()
        expected_recv_x.append(src_x[idx_in_src])
        expected_recv_topk.append(all_topk[src][idx_in_src])
    expected_recv_x = torch.cat(expected_recv_x, dim=0)
    expected_topk = torch.cat(expected_recv_topk, dim=0)

    # (1) recv_x
    assert recv_x.shape == expected_recv_x.shape, (
        f"dispatch {dispatch_seq}: recv_x shape {recv_x.shape} vs {expected_recv_x.shape}"
    )
    assert torch.equal(recv_x.cpu(), expected_recv_x.cpu()), (
        f"dispatch {dispatch_seq}: recv_x value mismatch"
    )

    # (2) recv_topk_idx local-id mapping
    e_lo, e_hi = rank * num_local_experts, (rank + 1) * num_local_experts
    local_mask = (expected_topk >= e_lo) & (expected_topk < e_hi)
    expected_topk_local = torch.where(
        local_mask, expected_topk - e_lo, torch.full_like(expected_topk, -1)
    )
    actual_topk = recv_topk_idx.cpu()
    assert torch.equal(actual_topk, expected_topk_local.cpu()), (
        f"dispatch {dispatch_seq}: recv_topk_idx mismatch"
    )

    # (3) tile_remaining all-zero
    tr = handle.tile_remaining[:handle.total_tiles].cpu()
    assert (tr == 0).all(), (
        f"dispatch {dispatch_seq}: tile_remaining nonzero at {tr.nonzero().tolist()}"
    )

    # (4) tile_ready full
    ready = handle.tile_ready[:handle.total_tiles].cpu()
    assert (ready == dispatch_seq).all(), (
        f"dispatch {dispatch_seq}: tile_ready not all == dispatch_seq; "
        f"mismatches at {(ready != dispatch_seq).nonzero().tolist()}"
    )

    # (5) tile_records consistency: every valid (r, k) appears exactly once.
    rows = handle.tile_records_recv_x_rows[:handle.total_tiles].cpu()
    kslots = handle.tile_records_k_slots[:handle.total_tiles].cpu()
    eids = handle.tile_records_expert_id[:handle.total_tiles].cpu()
    seen = torch.zeros(actual_topk.shape, dtype=torch.bool)
    for tile_id in range(handle.total_tiles):
        e_local = int(eids[tile_id].item())
        for slot in range(tile_m):
            r = int(rows[tile_id, slot].item())
            k = int(kslots[tile_id, slot].item())
            if r == -1:
                continue
            assert int(actual_topk[r, k].item()) == e_local, (
                f"dispatch {dispatch_seq}: tile {tile_id} slot {slot}: "
                f"r={r} k={k} expert={e_local} but recv_topk_idx={actual_topk[r,k]}"
            )
            assert not seen[r, k], (
                f"dispatch {dispatch_seq}: (r={r}, k={k}) appears in multiple tile slots"
            )
            seen[r, k] = True
    expected_seen = (actual_topk >= 0)
    assert torch.equal(seen, expected_seen), (
        f"dispatch {dispatch_seq}: tile_records covers {seen.sum()} valid pairs but "
        f"recv_topk_idx has {expected_seen.sum()} valid"
    )


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
    buf = Buffer(group, nvl_bytes, rdma_bytes, num_qps_per_rank=Buffer.num_sms)

    # Different inputs per dispatch (different seeds). Confirms no state leak between
    # dispatches in IPC slabs, tile_ready, count_exchange inboxes, etc.
    seeds = [123, 456, 789]
    for i, seed in enumerate(seeds):
        dispatch_seq = i + 1
        x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
            num_tokens, hidden, num_topk, num_experts, world_size, rank, device, seed=seed)

        recv_x, recv_topk_idx, _recv_topk_weights, handle, _event = buf.dispatch(
            x, topk_idx, topk_weights, is_token_in_rank, num_experts,
            tile_m=tile_m, dispatch_seq=dispatch_seq,
        )
        torch.cuda.synchronize()

        validate_dispatch(
            rank, world_size, num_local_experts, tile_m,
            x, topk_idx, is_token_in_rank,
            recv_x, recv_topk_idx, handle, dispatch_seq, group,
        )

        if rank == 0:
            print(f"PASS dispatch {dispatch_seq} (seed={seed}): "
                  f"T_recv={recv_x.size(0)} total_tiles={handle.total_tiles}")

    if rank == 0:
        print(f"PASS: all {len(seeds)} multi-dispatch validations on rank 0 (world={world_size})")


if __name__ == "__main__":
    main()
