"""End-to-end test for the consolidated streaming-MoE dispatch (intranode, pool layout).

Exercises ``Buffer.dispatch`` (pool-layout: dispatch's receiver writes each
landed (token, k) pair routing to a local expert into its own pool slot, and
fires tile_ready in expert-major order at substream end). Verifies:

  1. Pool data correctness — each pool slot matches its source rank's token data.
  2. Pool layout — slots fall in their expert's pool region (bounded by
     expert_pool_block_offset[e] * tile_m for each expert e); padding rows
     have pool_recv_token == -1.
  3. Per-(recv_token, k) coverage — every (r, k) with recv-side k routing to a
     local expert is recorded in exactly one pool slot.
  4. tile_ready[tile_id] == dispatch_seq for every tile_id in [0, total_tiles).
  5. tile_id_to_expert agrees with the slot range.
  6. Bit-determinism: re-running dispatch produces identical pool placement.
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

    pool, handle, _event = buf.dispatch(
        x, topk_idx, topk_weights, is_token_in_rank, num_experts,
        tile_m=tile_m, dispatch_seq=1,
    )
    torch.cuda.synchronize()

    total_tiles = handle.total_tiles
    TK_padded = total_tiles * tile_m
    e_lo, e_hi = rank * num_local_experts, (rank + 1) * num_local_experts

    # Build expected (recv_token_id, k_local) → src_rank mapping by gathering.
    all_x = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(all_x, x, group=group)
    all_topk = [torch.empty_like(topk_idx) for _ in range(world_size)]
    dist.all_gather(all_topk, topk_idx, group=group)
    all_is_in_rank = [torch.empty_like(is_token_in_rank) for _ in range(world_size)]
    dist.all_gather(all_is_in_rank, is_token_in_rank, group=group)

    # Recv-tokens are laid out source-major: rank 0's first, then rank 1's, etc.
    # Each source rank's contribution is the ordered set of token indices where
    # is_token_in_rank[:, my_rank] is True.
    expected_recv_token_src_rank = []          # [T_recv] → src_rank
    expected_recv_token_src_idx = []           # [T_recv] → src_token_idx
    expected_recv_topk_local = []              # [T_recv, num_topk] → e_local or -1
    for src in range(world_size):
        src_in = all_is_in_rank[src][:, rank]
        idx_in_src = torch.nonzero(src_in, as_tuple=False).flatten()
        src_topk = all_topk[src][idx_in_src]
        local_mask = (src_topk >= e_lo) & (src_topk < e_hi)
        local_topk = torch.where(local_mask, src_topk - e_lo, torch.full_like(src_topk, -1))
        expected_recv_token_src_rank.append(torch.full((idx_in_src.numel(),), src, dtype=torch.int32, device=device))
        expected_recv_token_src_idx.append(idx_in_src.to(torch.int32))
        expected_recv_topk_local.append(local_topk)
    expected_recv_token_src_rank = torch.cat(expected_recv_token_src_rank, dim=0).cpu()
    expected_recv_token_src_idx = torch.cat(expected_recv_token_src_idx, dim=0).cpu()
    expected_recv_topk_local = torch.cat(expected_recv_topk_local, dim=0).cpu()
    T_recv = expected_recv_token_src_rank.shape[0]

    pool_cpu = pool.cpu()
    pool_recv_token = handle.pool_recv_token.cpu()
    pool_k_slot = handle.pool_k_slot.cpu()
    pool_topk_weight = handle.pool_topk_weight.cpu()
    expert_pool_block_offset = handle.expert_pool_block_offset.cpu()
    tile_id_to_expert = handle.tile_id_to_expert.cpu()
    pool_arrival_target = handle.pool_arrival_target.cpu()
    expert_frequency = handle.expert_frequency.cpu()

    # ─── (1) Per-pool-slot validation. For every slot s with pool_recv_token[s] >= 0:
    #         (a) the slot lies in the right expert's pool region,
    #         (b) pool[s, :] == src_rank's tokens-of-ones value,
    #         (c) (recv_token, k) pair is unique across slots.
    seen_rk = torch.zeros((T_recv, num_topk), dtype=torch.bool)
    for s in range(TK_padded):
        rt = int(pool_recv_token[s].item())
        k = int(pool_k_slot[s].item())
        if rt < 0:
            continue  # padding slot
        assert 0 <= rt < T_recv, f"slot {s} pool_recv_token {rt} out of [0, {T_recv})"
        assert 0 <= k < num_topk, f"slot {s} pool_k_slot {k} out of [0, {num_topk})"
        e_local = int(expected_recv_topk_local[rt, k].item())
        assert e_local >= 0, f"slot {s} maps to (rt={rt}, k={k}) but expected_recv_topk_local[rt,k] = {e_local}"
        # Slot must fall in expert e_local's pool block range.
        block_start = int(expert_pool_block_offset[e_local].item()) * tile_m
        block_end = int(expert_pool_block_offset[e_local + 1].item()) * tile_m
        assert block_start <= s < block_end, (
            f"slot {s} for expert {e_local} outside [{block_start}, {block_end})"
        )
        # Per-pool-slot data: src_rank's value.
        src_rank = int(expected_recv_token_src_rank[rt].item())
        actual = pool_cpu[s, 0].to(torch.int32).item()
        assert actual == src_rank, f"slot {s}: pool[s,0] = {actual} != src_rank {src_rank}"
        # Uniqueness.
        assert not seen_rk[rt, k], f"(recv_token={rt}, k={k}) appears in multiple slots"
        seen_rk[rt, k] = True
        # pool_topk_weight matches the source token's topk_weight for this k.
        # (We don't explicitly gather topk_weights from peers; just sanity-check finiteness.)
        assert torch.isfinite(pool_topk_weight[s])
    expected_seen = (expected_recv_topk_local >= 0)
    assert torch.equal(seen_rk, expected_seen), (
        f"pool covers {seen_rk.sum()} (rt, k) pairs but expected {expected_seen.sum()}"
    )

    # ─── (3) tile_ready[tile_id] == dispatch_seq for all tile_id.
    ready = handle.tile_ready[:total_tiles].cpu()
    assert (ready == 1).all(), f"tile_ready not all == dispatch_seq (1); mismatches at {(ready != 1).nonzero().flatten()[:8]}"

    # ─── (4) tile_id_to_expert agrees with the expert_pool_block_offset partition.
    for tile_id in range(total_tiles):
        e_actual = int(tile_id_to_expert[tile_id].item())
        e_block_start = int(expert_pool_block_offset[e_actual].item())
        e_block_end = int(expert_pool_block_offset[e_actual + 1].item())
        assert e_block_start <= tile_id < e_block_end, (
            f"tile {tile_id}: tile_id_to_expert={e_actual} but tile not in [{e_block_start}, {e_block_end})"
        )

    # ─── (5) pool_arrival_target: BLOCK_M for full tiles, leftover for last per expert.
    for e in range(num_local_experts):
        e_block_start = int(expert_pool_block_offset[e].item())
        e_block_end = int(expert_pool_block_offset[e + 1].item())
        n_e = int(expert_frequency[e].item())
        for tile_id in range(e_block_start, e_block_end):
            tile_in_e = tile_id - e_block_start
            target = int(pool_arrival_target[tile_id].item())
            if tile_id == e_block_end - 1:
                expected = n_e - tile_in_e * tile_m
            else:
                expected = tile_m
            assert target == expected, (
                f"pool_arrival_target[{tile_id}] = {target} != expected {expected} (e={e}, tile_in_e={tile_in_e})"
            )

    # ─── (6) Pipeline buffer initialization. per_token_remaining[r] should equal
    #         K_local(r) (the count of local-expert landings for recv-token r),
    #         which is also the count of pool slots for r. compute_done_per_token
    #         and o (and a_ready) should be zero-init.
    per_token_remaining = handle.per_token_remaining.cpu()
    assert per_token_remaining.shape == (T_recv,), (
        f"per_token_remaining shape {per_token_remaining.shape} != ({T_recv},)"
    )
    expected_k_local = (expected_recv_topk_local >= 0).sum(dim=1).to(torch.int32)
    assert torch.equal(per_token_remaining, expected_k_local), (
        f"per_token_remaining mismatch; first deviating r: "
        f"{(per_token_remaining != expected_k_local).nonzero().flatten()[:8]}"
    )
    compute_done = handle.compute_done_per_token.cpu()
    assert (compute_done == 0).all(), "compute_done_per_token should be zero-init"
    a_ready_t = handle.a_ready[:total_tiles].cpu()
    assert (a_ready_t == 0).all(), "a_ready should be zero-init"
    o_t = handle.o.cpu()
    assert o_t.shape == (T_recv, hidden), f"o shape {o_t.shape} != ({T_recv}, {hidden})"
    assert (o_t == 0).all(), "o should be zero-init"

    # ─── (6.5) Backward scaffolding: recv_token_to_slots[r, k] should equal the
    # slot for (r, k) when k routes to a local expert, and -1 otherwise. The
    # expected mapping is the inverse of the existing per-slot writes:
    # for every slot s with pool_recv_token[s] >= 0, recv_token_to_slots[
    #   pool_recv_token[s], pool_k_slot[s]] == s. k_local_count[r] should
    # equal per_token_remaining[r] (write-once mirror that fwd doesn't decrement).
    rtts = handle.recv_token_to_slots.cpu()
    assert rtts.shape == (T_recv, num_topk), (
        f"recv_token_to_slots shape {rtts.shape} != ({T_recv}, {num_topk})"
    )
    expected_rtts = torch.full((T_recv, num_topk), -1, dtype=torch.int32)
    valid_slots = pool_recv_token >= 0
    for s in valid_slots.nonzero().flatten().tolist():
        rt = int(pool_recv_token[s].item())
        k = int(pool_k_slot[s].item())
        expected_rtts[rt, k] = s
    assert torch.equal(rtts, expected_rtts), (
        f"recv_token_to_slots mismatch; first deviating (r, k): "
        f"{(rtts != expected_rtts).nonzero()[:8]}"
    )
    klc = handle.k_local_count.cpu()
    assert klc.shape == (T_recv,), f"k_local_count shape {klc.shape} != ({T_recv},)"
    assert torch.equal(klc, per_token_remaining), (
        "k_local_count should equal per_token_remaining (both = K_local(r) pre-decrement)"
    )

    # seen_per_substream[c, src, e] = my_e_inbox[c, src, e] from the metadata
    # kernel's IPC count exchange. Invariant: summing over (c, src) gives the
    # per-expert totals, i.e. expert_frequency[e].
    sps = handle.seen_per_substream.cpu()
    expected_sps_shape = (sps.shape[0], world_size, num_local_experts)
    assert sps.shape == expected_sps_shape, (
        f"seen_per_substream shape {sps.shape} != {expected_sps_shape}"
    )
    sps_per_expert = sps.sum(dim=(0, 1))
    expected_sps_per_expert = handle.expert_frequency.cpu()
    assert torch.equal(sps_per_expert, expected_sps_per_expert), (
        f"sum_(c, src) seen_per_substream[c, src, :] should equal expert_frequency; "
        f"got {sps_per_expert} vs {expected_sps_per_expert}"
    )

    # ─── (7) Bit-determinism — re-run with same inputs, expect identical pool layout.
    pool2, handle2, _ = buf.dispatch(
        x, topk_idx, topk_weights, is_token_in_rank, num_experts,
        tile_m=tile_m, dispatch_seq=2,
    )
    torch.cuda.synchronize()
    assert torch.equal(handle.pool_recv_token.cpu(), handle2.pool_recv_token.cpu()), \
        "pool_recv_token not deterministic"
    assert torch.equal(handle.pool_k_slot.cpu(), handle2.pool_k_slot.cpu()), \
        "pool_k_slot not deterministic"
    assert torch.equal(handle.recv_token_to_slots.cpu(), handle2.recv_token_to_slots.cpu()), \
        "recv_token_to_slots not deterministic"
    assert torch.equal(handle.k_local_count.cpu(), handle2.k_local_count.cpu()), \
        "k_local_count not deterministic"
    assert torch.equal(handle.seen_per_substream.cpu(), handle2.seen_per_substream.cpu()), \
        "seen_per_substream not deterministic"
    # `pool` padding rows are now left uninitialized (kernel Y's pool_recv_token
    # predicate is the actual safety mechanism). Compare valid rows only.
    valid = (handle.pool_recv_token.cpu() >= 0)
    assert torch.equal(pool.cpu()[valid], pool2.cpu()[valid]), "pool data (valid rows) not deterministic"
    assert torch.equal(handle.tile_id_to_expert.cpu(), handle2.tile_id_to_expert.cpu()), \
        "tile_id_to_expert not deterministic"

    if rank == 0:
        print(f"PASS: rank={rank} world={world_size} T_recv={T_recv} "
              f"TK_padded={TK_padded} total_tiles={total_tiles}")


if __name__ == "__main__":
    main()
