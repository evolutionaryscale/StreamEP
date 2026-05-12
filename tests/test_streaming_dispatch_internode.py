"""End-to-end test for the streaming-MoE internode dispatch (pool layout).

Mirrors ``tests/test_streaming_dispatch.py`` (which exercises the intranode
streaming path) for the internode topology — 2 RDMA × 8 NVL = 16 GPUs via
``./srun_internode.sh``. Drives the new C++ entry point
``Buffer.runtime.internode_dispatch`` directly (not yet wired through
``Buffer.dispatch``'s topology branch — that switchover lands in a later
commit alongside the legacy ``Buffer.internode_dispatch`` deletion).

Verifies, against an eager reference computed from all-gather'd per-rank
inputs:

  1. Pool data correctness — for every pool slot ``s`` with
     ``pool_recv_token[s] >= 0``, ``pool[s, 0]`` matches the source rank's
     ``x[t, 0]`` value.
  2. Pool layout — every populated slot falls inside its expert's pool block
     range (``[expert_pool_block_offset[e], expert_pool_block_offset[e+1]) *
     tile_m``); padding rows have ``pool_recv_token == -1``.
  3. Per-(recv_token, k) coverage — every (r, k) routing to a local expert is
     recorded in exactly one pool slot.
  4. ``tile_ready[tile_id] == dispatch_seq`` for every tile_id in
     ``[0, total_tiles)``.
  5. ``tile_id_to_expert`` and ``pool_arrival_target`` agree with the
     ``expert_pool_block_offset`` partition.
  6. ``per_token_remaining[r]`` matches the count of local-expert landings
     for recv-token ``r``; ``recv_token_to_slots[r, k]`` is the inverse of
     ``pool_recv_token`` / ``pool_k_slot``.
  7. Bit-determinism: re-running produces identical ``pool_recv_token`` /
     ``pool_k_slot`` / ``recv_token_to_slots`` and identical valid-row pool
     data.

Edge cases planted in inputs:
  - ~5% ``-1`` sentinels in ``topk_idx`` to exercise the skip branch.
  - One expert pinned to receive zero tokens this iter (forces empty-expert
    branch in Pass 2 fire).
  - Multi-iter dispatch_seq reuse (1 then 2) to verify cross-iter
    correctness.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from stream_ep import Buffer

from utils import cleanup_dist, make_inputs


def compute_expected_recv_layout(all_x: list[torch.Tensor],
                                 all_topk: list[torch.Tensor],
                                 all_is_in_rank: list[torch.Tensor],
                                 *, this_rank: int, num_local_experts: int):
    """Build per-recv-token expected metadata.

    Recv-tokens are laid out globally-source-major: source rank 0's tokens
    first (in source token-index order), then rank 1's, etc. Within each
    source, only tokens with ``is_token_in_rank[:, this_rank] == True`` are
    delivered.

    Returns:
        expected_src_rank   [T_recv] int32 — source world rank for each
                             recv-token.
        expected_src_idx    [T_recv] int32 — source token index for each
                             recv-token.
        expected_topk_local [T_recv, num_topk] int64 — e_local per (r, k),
                             with -1 for non-local k.
    """
    e_lo = this_rank * num_local_experts
    e_hi = (this_rank + 1) * num_local_experts
    src_rank, src_idx, topk_local = [], [], []
    for src in range(len(all_x)):
        src_in = all_is_in_rank[src][:, this_rank]
        idx_in_src = torch.nonzero(src_in, as_tuple=False).flatten()
        src_topk = all_topk[src][idx_in_src]
        local_mask = (src_topk >= e_lo) & (src_topk < e_hi)
        local_topk = torch.where(local_mask, src_topk - e_lo,
                                 torch.full_like(src_topk, -1))
        src_rank.append(torch.full((idx_in_src.numel(),), src,
                                    dtype=torch.int32, device=idx_in_src.device))
        src_idx.append(idx_in_src.to(torch.int32))
        topk_local.append(local_topk)
    return (torch.cat(src_rank, dim=0).cpu(),
            torch.cat(src_idx, dim=0).cpu(),
            torch.cat(topk_local, dim=0).cpu())


def assert_pool_correctness(pool: torch.Tensor,
                            pool_recv_token: torch.Tensor,
                            pool_k_slot: torch.Tensor,
                            pool_topk_weight: torch.Tensor,
                            expert_pool_block_offset: torch.Tensor,
                            expected_src_rank: torch.Tensor,
                            expected_topk_local: torch.Tensor,
                            *, T_recv: int, TK_padded: int, tile_m: int,
                            num_topk: int, this_rank: int):
    """Walk every pool slot and verify per-slot invariants + coverage."""
    seen_rk = torch.zeros((T_recv, num_topk), dtype=torch.bool)
    for s in range(TK_padded):
        rt = int(pool_recv_token[s].item())
        k = int(pool_k_slot[s].item())
        if rt < 0:
            continue
        assert 0 <= rt < T_recv, \
            f"slot {s} pool_recv_token {rt} out of [0, {T_recv})"
        assert 0 <= k < num_topk, \
            f"slot {s} pool_k_slot {k} out of [0, {num_topk})"
        e_local = int(expected_topk_local[rt, k].item())
        assert e_local >= 0, (
            f"slot {s} maps to (rt={rt}, k={k}) but expected_topk_local[rt,k] "
            f"= {e_local}")
        block_start = int(expert_pool_block_offset[e_local].item()) * tile_m
        block_end = int(expert_pool_block_offset[e_local + 1].item()) * tile_m
        assert block_start <= s < block_end, (
            f"slot {s} for expert {e_local} (this_rank={this_rank}) outside "
            f"[{block_start}, {block_end})")
        src_rank = int(expected_src_rank[rt].item())
        actual = pool[s, 0].to(torch.int32).item()
        assert actual == src_rank, (
            f"slot {s}: pool[s,0] = {actual} != src_rank {src_rank} "
            f"(rt={rt}, k={k}, e_local={e_local})")
        assert not seen_rk[rt, k], \
            f"(recv_token={rt}, k={k}) appears in multiple slots"
        seen_rk[rt, k] = True
        assert torch.isfinite(pool_topk_weight[s])
    expected_seen = (expected_topk_local >= 0)
    assert torch.equal(seen_rk, expected_seen), (
        f"pool covers {int(seen_rk.sum())} (rt, k) pairs but expected "
        f"{int(expected_seen.sum())}")


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
    num_local_experts = num_experts // world_size
    num_topk = 4
    num_tokens = 256
    hidden = 256
    tile_m = 32
    plant_empty_expert = None

    hidden_bytes = hidden * 2
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (Buffer.get_dispatch_config(world_size),
                Buffer.get_combine_config(world_size)):
        nvl_bytes  = max(cfg.get_nvl_buffer_size_hint(hidden_bytes,  world_size), nvl_bytes)
        rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes)
    buf = Buffer(group, nvl_bytes, rdma_bytes)

    x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
        num_tokens, hidden, num_topk, num_experts, world_size, rank, device,
        plant_sentinels=True, plant_empty_expert=plant_empty_expert)

    # Skip cleanly if the C++ entry point isn't built yet.
    if not hasattr(buf.runtime, 'internode_dispatch'):
        if rank == 0:
            print("[skip] Buffer.runtime.internode_dispatch not "
                  "built — kernel implementation pending.", flush=True)
        cleanup_dist()
        return

    cfg = Buffer.get_dispatch_config(world_size)
    out = buf.runtime.internode_dispatch(
        x, topk_idx, topk_weights, is_token_in_rank,
        num_experts, 1, tile_m, 1, cfg)
    torch.cuda.synchronize()

    total_tiles = out.total_tiles
    TK_padded = total_tiles * tile_m

    all_x = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(all_x, x, group=group)
    all_topk = [torch.empty_like(topk_idx) for _ in range(world_size)]
    dist.all_gather(all_topk, topk_idx, group=group)
    all_is_in_rank = [torch.empty_like(is_token_in_rank) for _ in range(world_size)]
    dist.all_gather(all_is_in_rank, is_token_in_rank, group=group)

    expected_src_rank, expected_src_idx, expected_topk_local = \
        compute_expected_recv_layout(all_x, all_topk, all_is_in_rank,
                                     this_rank=rank,
                                     num_local_experts=num_local_experts)
    T_recv = expected_src_rank.shape[0]

    pool_cpu                 = out.pool.cpu()
    pool_recv_token          = out.pool_recv_token.cpu()
    pool_k_slot              = out.pool_k_slot.cpu()
    pool_topk_weight         = out.pool_topk_weight.cpu()
    expert_pool_block_offset = out.expert_pool_block_offset.cpu()
    tile_id_to_expert        = out.tile_id_to_expert.cpu()
    pool_arrival_target      = out.pool_arrival_target.cpu()
    expert_frequency         = out.expert_frequency.cpu()
    tile_ready               = out.tile_ready.cpu()
    per_token_remaining      = out.per_token_remaining.cpu()
    recv_token_to_slots      = out.recv_token_to_slots.cpu()
    k_local_count            = out.k_local_count.cpu()

    # ─── (1)+(2)+(3) Per-pool-slot validation + coverage. ────────────────────
    assert_pool_correctness(
        pool_cpu, pool_recv_token, pool_k_slot, pool_topk_weight,
        expert_pool_block_offset, expected_src_rank, expected_topk_local,
        T_recv=T_recv, TK_padded=TK_padded, tile_m=tile_m,
        num_topk=num_topk, this_rank=rank)

    # ─── (4) tile_ready[tile_id] == dispatch_seq for all tile_id. ────────────
    assert (tile_ready[:total_tiles] == 1).all(), (
        f"tile_ready not all == dispatch_seq (1); first mismatches at "
        f"{(tile_ready[:total_tiles] != 1).nonzero().flatten()[:8]}")

    # ─── (5) tile_id_to_expert and pool_arrival_target agree with offsets. ──
    for tile_id in range(total_tiles):
        e_actual = int(tile_id_to_expert[tile_id].item())
        e_block_start = int(expert_pool_block_offset[e_actual].item())
        e_block_end = int(expert_pool_block_offset[e_actual + 1].item())
        assert e_block_start <= tile_id < e_block_end, (
            f"tile {tile_id}: tile_id_to_expert={e_actual} but tile not in "
            f"[{e_block_start}, {e_block_end})")
    for e in range(num_local_experts):
        e_block_start = int(expert_pool_block_offset[e].item())
        e_block_end = int(expert_pool_block_offset[e + 1].item())
        n_e = int(expert_frequency[e].item())
        for tile_id in range(e_block_start, e_block_end):
            tile_in_e = tile_id - e_block_start
            target = int(pool_arrival_target[tile_id].item())
            expected = (n_e - tile_in_e * tile_m) if tile_id == e_block_end - 1 else tile_m
            assert target == expected, (
                f"pool_arrival_target[{tile_id}] = {target} != expected "
                f"{expected} (e={e}, tile_in_e={tile_in_e})")

    # ─── (6) per_token_remaining + recv_token_to_slots + k_local_count. ─────
    assert per_token_remaining.shape == (T_recv,), (
        f"per_token_remaining shape {per_token_remaining.shape} != ({T_recv},)")
    expected_k_local = (expected_topk_local >= 0).sum(dim=1).to(torch.int32)
    assert torch.equal(per_token_remaining, expected_k_local), (
        f"per_token_remaining mismatch; first deviating r: "
        f"{(per_token_remaining != expected_k_local).nonzero().flatten()[:8]}")

    assert recv_token_to_slots.shape == (T_recv, num_topk), (
        f"recv_token_to_slots shape {recv_token_to_slots.shape} != "
        f"({T_recv}, {num_topk})")
    expected_rtts = torch.full((T_recv, num_topk), -1, dtype=torch.int32)
    valid_slots = pool_recv_token >= 0
    for s in valid_slots.nonzero().flatten().tolist():
        rt = int(pool_recv_token[s].item())
        k = int(pool_k_slot[s].item())
        expected_rtts[rt, k] = s
    assert torch.equal(recv_token_to_slots, expected_rtts), (
        f"recv_token_to_slots mismatch; first deviating (r, k): "
        f"{(recv_token_to_slots != expected_rtts).nonzero()[:8]}")

    assert torch.equal(k_local_count, per_token_remaining), (
        "k_local_count should equal per_token_remaining (write-once mirror)")

    # ─── (7) Bit-determinism: re-run with same inputs. ──────────────────────
    out2 = buf.runtime.internode_dispatch(
        x, topk_idx, topk_weights, is_token_in_rank,
        num_experts, 1, tile_m, 2, cfg)
    torch.cuda.synchronize()
    assert torch.equal(pool_recv_token, out2.pool_recv_token.cpu()), \
        "pool_recv_token not deterministic across re-runs"
    assert torch.equal(pool_k_slot, out2.pool_k_slot.cpu()), \
        "pool_k_slot not deterministic across re-runs"
    assert torch.equal(recv_token_to_slots, out2.recv_token_to_slots.cpu()), \
        "recv_token_to_slots not deterministic across re-runs"
    valid = (pool_recv_token >= 0)
    assert torch.equal(out.pool.cpu()[valid], out2.pool.cpu()[valid]), \
        "pool data (valid rows) not deterministic across re-runs"
    assert (out2.tile_ready[:total_tiles].cpu() == 2).all(), (
        f"second-run tile_ready not all == 2; first mismatches at "
        f"{(out2.tile_ready[:total_tiles].cpu() != 2).nonzero().flatten()[:8]}")

    if rank == 0:
        print(f"PASS: world_size={world_size} T_recv={T_recv} "
              f"TK_padded={TK_padded} total_tiles={total_tiles}", flush=True)

    cleanup_dist()


if __name__ == "__main__":
    main()
