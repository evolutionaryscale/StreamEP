"""End-to-end test for the consolidated streaming-MoE dispatch (intranode).

Exercises ``Buffer.dispatch`` (which folds streaming_count_exchange +
streaming_metadata_init + dispatch main + slot_assign into a single host call
with one host sync per layer; the sender-side channel-prefix scan that used to
live in a separate ``notify_dispatch`` kernel is now inline in dispatch's sender
preamble). Verifies:

  1. recv_x correctness — each landed token has the value `x = ones * src_rank`.
  2. recv_topk_idx agrees with the routed-to-this-rank entries from each source.
  3. tile_remaining is all-zero post-slot_assign.
  4. tile_ready[tile_id] == dispatch_seq for every tile_id in [0, total_tiles).
  5. tile_records (recv_x_rows, k_slots, expert_id) agrees with recv_topk_idx for
     every (r, k) pair where recv_topk_idx[r, k] >= 0.
  6. Bit-determinism: re-running dispatch on identical inputs yields identical
     tile_records.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from deep_ep import Buffer


def make_inputs(num_tokens, hidden, num_topk, num_experts, num_ranks, rank, device):
    g = torch.Generator(device=device).manual_seed(123 + rank)
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

    x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
        num_tokens, hidden, num_topk, num_experts, world_size, rank, device)

    recv_x, recv_topk_idx, recv_topk_weights, handle, _event = buf.dispatch(
        x, topk_idx, topk_weights, is_token_in_rank, num_experts,
        tile_m=tile_m, dispatch_seq=1,
    )
    torch.cuda.synchronize()

    # ─── (1) recv_x value check: each row should equal its source rank.
    # Gather all source-rank routings to compute expected per-row src_rank for this rank.
    all_x = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(all_x, x, group=group)
    all_topk = [torch.empty_like(topk_idx) for _ in range(world_size)]
    dist.all_gather(all_topk, topk_idx, group=group)
    all_is_in_rank = [torch.empty_like(is_token_in_rank) for _ in range(world_size)]
    dist.all_gather(all_is_in_rank, is_token_in_rank, group=group)

    # Each source rank contributes (in src_token_idx order) its tokens whose is_token_in_rank[:, rank] is True.
    # That order matches recv_x's layout (source-major intranode).
    expected_recv_x = []
    expected_src_idx = []
    expected_recv_topk = []
    for src in range(world_size):
        src_x = all_x[src]
        src_in = all_is_in_rank[src][:, rank]
        idx_in_src = torch.nonzero(src_in, as_tuple=False).flatten()
        expected_recv_x.append(src_x[idx_in_src])
        expected_src_idx.append(idx_in_src.to(torch.int32))
        expected_recv_topk.append(all_topk[src][idx_in_src])
    expected_recv_x = torch.cat(expected_recv_x, dim=0)
    assert recv_x.shape == expected_recv_x.shape, f"shape {recv_x.shape} vs {expected_recv_x.shape}"
    if not torch.equal(recv_x.cpu(), expected_recv_x.cpu()):
        actual_first_col = recv_x[:, 0].cpu().to(torch.int32)
        expected_first_col = expected_recv_x[:, 0].cpu().to(torch.int32)
        diff_idx = (actual_first_col != expected_first_col).nonzero(as_tuple=False).flatten()[:10]
        if rank == 0:
            print(f"[rank {rank}] recv_x mismatch — first col actual vs expected:")
            for i in diff_idx.tolist():
                print(f"  row {i}: actual={actual_first_col[i].item()} expected={expected_first_col[i].item()}")
            print(f"  recv_channel_prefix_matrix:\n{handle.recv_channel_prefix_matrix.cpu()}")
            print(f"  rank_prefix_matrix col {rank}: {handle.rank_prefix_matrix[:, rank].cpu()}")
        raise AssertionError("recv_x mismatch")

    # ─── (2) recv_topk_idx — local-expert indices in [0, E_local) or -1 sentinel.
    # The dispatch kernel rewrites global expert id → local id (subtracts e_lo) when
    # routed to this rank, else writes -1.
    expected_topk = torch.cat(expected_recv_topk, dim=0)
    e_lo, e_hi = rank * num_local_experts, (rank + 1) * num_local_experts
    local_mask = (expected_topk >= e_lo) & (expected_topk < e_hi)
    expected_topk_local = torch.where(local_mask, expected_topk - e_lo, torch.full_like(expected_topk, -1))
    actual_topk = recv_topk_idx.cpu()
    assert torch.equal(actual_topk, expected_topk_local.cpu()), "recv_topk_idx mismatch"

    # ─── (3) tile_remaining all-zero
    tr = handle.tile_remaining[:handle.total_tiles].cpu()
    assert (tr == 0).all(), f"tile_remaining nonzero: {tr.nonzero()}"

    # ─── (4) Every tile fired exactly once: tile_ready[tile_id] == dispatch_seq
    # for all tile_id in [0, total_tiles). With per-tile ready signal, the
    # firing test is just an equality check on the int64 array.
    ready = handle.tile_ready[:handle.total_tiles].cpu()
    assert (ready == 1).all(), f"tile_ready not all == dispatch_seq (1): {(ready != 1).nonzero()}"

    # ─── (5) tile_records consistency with recv_topk_idx
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
            assert int(actual_topk[r, k].item()) == e_local, \
                f"tile {tile_id} slot {slot}: r={r} k={k} expert={e_local} but recv_topk_idx={actual_topk[r,k]}"
            assert not seen[r, k], f"(r={r}, k={k}) appears in multiple tile slots"
            seen[r, k] = True
    expected_seen = (actual_topk >= 0)
    assert torch.equal(seen, expected_seen), \
        f"tile_records covers {seen.sum()} (r,k) pairs but recv_topk_idx has {expected_seen.sum()} valid"

    # ─── (6) Bit-determinism
    recv_x2, recv_topk_idx2, _, handle2, _ = buf.dispatch(
        x, topk_idx, topk_weights, is_token_in_rank, num_experts,
        tile_m=tile_m, dispatch_seq=2,
    )
    torch.cuda.synchronize()
    rows2 = handle2.tile_records_recv_x_rows[:handle2.total_tiles].cpu()
    kslots2 = handle2.tile_records_k_slots[:handle2.total_tiles].cpu()
    eids2 = handle2.tile_records_expert_id[:handle2.total_tiles].cpu()
    assert torch.equal(rows, rows2), "tile_records_recv_x_rows not deterministic"
    assert torch.equal(kslots, kslots2), "tile_records_k_slots not deterministic"
    assert torch.equal(eids, eids2), "tile_records_expert_id not deterministic"

    if rank == 0:
        print(f"PASS: rank={rank} world={world_size} T_recv={recv_x.size(0)} "
              f"total_tiles={handle.total_tiles}")


if __name__ == "__main__":
    main()
