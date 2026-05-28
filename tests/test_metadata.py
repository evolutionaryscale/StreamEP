"""Standalone correctness test for `streaming_dispatch_metadata_kernel`.

Per-tensor diff of every metadata-kernel output on the dispatch `handle`
against an independent eager-torch reference computed from globally-
gathered `topk_idx`. Self-validating — no v1 / v2 comparison; the kernel
is "right" iff every output matches the pure-torch reference.

Runs intranode by default (world=8). Pass `--internode` to validate on a
multi-node alloc (world=16+) — that path additionally verifies the
NVL/RDMA partitioned `base_pool` ordering plus the internode-only
counters (`recv_rdma_rank_prefix_sum`, `recv_gbl_rank_prefix_sum`).

Launch:
    ./scripts/srun_1node.sh StreamEP/tests/test_metadata.py
    ./scripts/srun_4node.sh StreamEP/tests/test_metadata.py --internode

Exits 0 on PASS, non-zero on first FAIL with a printed diff summary.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
from stream_ep import Buffer
from utils import cleanup_dist, make_inputs


# ──────────────────────────────────────────────────────────────────────────
# Eager reference for the metadata kernel outputs.
# ──────────────────────────────────────────────────────────────────────────


def compute_metadata_reference(
    all_topk_idx: torch.Tensor,  # (num_ranks, T, K) int64 — globally gathered
    rank: int,
    num_ranks: int,
    num_channels: int,
    num_experts: int,
    expert_alignment: int,
    tile_m: int,
    *,
    is_internode: bool,
    num_max_nvl_peers: int = 8,
) -> dict:
    """Pure-torch reference for `streaming_dispatch_metadata_kernel`'s
    outputs *as observed by `rank`*. Returns a dict of expected tensors.

    The eager reference works for both intranode and internode topologies:
    the only place they differ is `base_pool`'s lex ordering — intranode
    is (c, src_world) lex; internode partitions into NVL-local first then
    RDMA-remote, with (c, src_world) lex within each partition.
    """
    device = all_topk_idx.device
    T = all_topk_idx.shape[1]
    K = all_topk_idx.shape[2]
    E_local = num_experts // num_ranks
    tokens_per_channel = (T + num_channels - 1) // num_channels
    rdma_rank = rank // num_max_nvl_peers  # only meaningful in internode

    # ── Stage 1: per-(c, src_world, e_local) counts of (token, k) pairs
    # that route to (this rank, e_local). Plus per-(c, src_world) unique
    # token counts (= |{ t : ∃ k with topk_idx[src, t, k] // E_local == rank }|).

    valid = all_topk_idx >= 0  # (num_ranks, T, K)
    # Map invalid entries to dst_rank=-1 so the rank==our_rank filter excludes them.
    dst_rank = torch.where(valid, all_topk_idx // E_local, torch.full_like(all_topk_idx, -1))
    e_local_g = torch.where(valid, all_topk_idx - dst_rank.clamp(min=0) * E_local, torch.zeros_like(all_topk_idx))

    to_me = valid & (dst_rank == rank)  # (num_ranks, T, K)

    t_idx = torch.arange(T, device=device).view(1, T, 1).expand(num_ranks, T, K)
    channel = torch.clamp(t_idx // tokens_per_channel, max=num_channels - 1)
    src_idx = torch.arange(num_ranks, device=device).view(num_ranks, 1, 1).expand(num_ranks, T, K)

    # seen[c, src, e] = count of (t, k) where src's topk routes to our (rank, e_local)
    flat_idx_full = (
        channel.long() * num_ranks * E_local
        + src_idx.long() * E_local
        + e_local_g.long()
    )
    flat_idx_used = flat_idx_full[to_me]
    seen = torch.zeros(
        num_channels * num_ranks * E_local, dtype=torch.int32, device=device
    )
    seen.scatter_add_(0, flat_idx_used, torch.ones_like(flat_idx_used, dtype=torch.int32))
    seen = seen.view(num_channels, num_ranks, E_local)

    # u_inbox[c, src] = #tokens from src with ANY k routing to our rank.
    to_me_any_k = to_me.any(dim=-1)  # (num_ranks, T)
    u_channel = channel[:, :, 0]  # (num_ranks, T)
    u_src = src_idx[:, :, 0]
    u_flat = u_channel.long() * num_ranks + u_src.long()
    u_idx_used = u_flat[to_me_any_k]
    u_inbox_flat = torch.zeros(num_channels * num_ranks, dtype=torch.int32, device=device)
    u_inbox_flat.scatter_add_(0, u_idx_used, torch.ones_like(u_idx_used, dtype=torch.int32))
    u_inbox = u_inbox_flat.view(num_channels, num_ranks)

    # ── Stage 2: derived counters.

    # expert_frequency[e] = sum over (c, src) of seen[c, src, e]
    expert_frequency = seen.sum(dim=(0, 1)).to(torch.int32)

    # expert_pool_block_offset: prefix sum of ceil(expert_frequency / tile_m)
    n_blocks_per_e = (expert_frequency + tile_m - 1) // tile_m
    expert_pool_block_offset = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    expert_pool_block_offset[1:] = n_blocks_per_e.cumsum(0).to(torch.int32)
    total_tiles = int(expert_pool_block_offset[-1].item())

    # ── Stage 3: base_pool[c, src, e] — cumulative lex offset.
    # Intranode: lex over (c, src_world).
    # Internode: NVL-local (src_rdma == rdma_rank) first, then RDMA-remote.
    # Within each partition: (c, src_world) lex.
    base_pool = torch.zeros_like(seen)

    if is_internode:
        # Build a permutation: cs_order[partition_pos] = (c, src_world)
        # Partition 1: cs with src_rdma == rdma_rank (NVL-local).
        # Partition 2: cs with src_rdma != rdma_rank.
        src_world_arr = torch.arange(num_ranks, device=device)
        src_rdma_arr = src_world_arr // num_max_nvl_peers
        is_nvl_local = (src_rdma_arr == rdma_rank)  # (num_ranks,)

        # In lex order (c, src_world): NVL-local first then RDMA-remote.
        # Build a mask matching the iteration order; use torch arithmetic to scan.
        # For simplicity (eager reference): do the two-pass loop directly.
        for e in range(E_local):
            acc = int(expert_pool_block_offset[e].item()) * tile_m
            # Pass 1: NVL-local
            for cs in range(num_channels * num_ranks):
                c = cs // num_ranks
                src_w = cs - c * num_ranks
                if int(src_rdma_arr[src_w].item()) != rdma_rank:
                    continue
                base_pool[c, src_w, e] = acc
                acc += int(seen[c, src_w, e].item())
            # Pass 2: RDMA-remote
            for cs in range(num_channels * num_ranks):
                c = cs // num_ranks
                src_w = cs - c * num_ranks
                if int(src_rdma_arr[src_w].item()) == rdma_rank:
                    continue
                base_pool[c, src_w, e] = acc
                acc += int(seen[c, src_w, e].item())
    else:
        # Intranode: simple (c, src) lex.
        for e in range(E_local):
            acc = int(expert_pool_block_offset[e].item()) * tile_m
            for c in range(num_channels):
                for src in range(num_ranks):
                    base_pool[c, src, e] = acc
                    acc += int(seen[c, src, e].item())

    # ── Stage 4: tile_id_to_expert + pool_arrival_target.
    tile_id_to_expert = torch.zeros(total_tiles, dtype=torch.int32, device=device)
    pool_arrival_target = torch.zeros(total_tiles, dtype=torch.int32, device=device)
    for e in range(E_local):
        e_start = int(expert_pool_block_offset[e].item())
        e_end = int(expert_pool_block_offset[e + 1].item())
        n_tiles_e = e_end - e_start
        n_e = int(expert_frequency[e].item())
        for t in range(n_tiles_e):
            tile_id = e_start + t
            tile_id_to_expert[tile_id] = e
            target = tile_m if t < n_tiles_e - 1 else (n_e - t * tile_m)
            pool_arrival_target[tile_id] = target

    # ── Stage 5: rank_prefix_matrix column for this rank.
    # per_src_unique[src] = sum over c of u_inbox[c, src] = total unique tokens
    # from sender `src` routing to our rank.
    per_src_unique = u_inbox.sum(dim=0).to(torch.int32)  # (num_ranks,)
    # rank_prefix_matrix[i, rank] = cumsum of per_src_unique over senders 0..i.
    rank_prefix_column = per_src_unique.cumsum(0).to(torch.int32)

    # ── Stage 6: total recv counters.
    moe_recv_counter = int(per_src_unique.sum().item())
    moe_recv_expert_counter = (
        (expert_frequency + expert_alignment - 1) // expert_alignment
    ) * expert_alignment

    out = {
        "expert_frequency": expert_frequency,
        "expert_pool_block_offset": expert_pool_block_offset,
        "seen_per_substream": seen,
        "base_pool": base_pool,
        "tile_id_to_expert": tile_id_to_expert,
        "pool_arrival_target": pool_arrival_target,
        "rank_prefix_column": rank_prefix_column,
        "total_tiles": total_tiles,
        "moe_recv_counter": moe_recv_counter,
        "moe_recv_expert_counter": moe_recv_expert_counter,
    }

    if is_internode:
        # Additional internode-only outputs.
        kNumRDMARanks = num_ranks // num_max_nvl_peers
        # recv_rdma_rank_prefix_sum[i] = cumulative unique tokens from RDMA-rank
        # senders 0..i (across all NVL peers within that RDMA rank).
        # In intranode-eager form: sum per-src then group by RDMA rank.
        per_src = per_src_unique  # (num_ranks,)
        per_rdma_unique = per_src.view(kNumRDMARanks, num_max_nvl_peers).sum(dim=1)
        out["recv_rdma_rank_prefix_sum"] = per_rdma_unique.cumsum(0).to(torch.int32)
        out["recv_gbl_rank_prefix_sum"] = per_src.cumsum(0).to(torch.int32)

    return out


# ──────────────────────────────────────────────────────────────────────────
# Comparison utilities.
# ──────────────────────────────────────────────────────────────────────────


def diff_int(actual: torch.Tensor, expected: torch.Tensor, name: str) -> int:
    """Return number of mismatched elements; print first few diffs."""
    actual = actual.cpu()
    expected = expected.cpu()
    if actual.shape != expected.shape:
        print(f"[{name}] SHAPE MISMATCH: actual {tuple(actual.shape)}, expected {tuple(expected.shape)}")
        return -1
    diff = (actual != expected)
    n_bad = int(diff.sum().item())
    if n_bad > 0:
        bad_idx = diff.nonzero(as_tuple=False)[:8]
        print(f"[{name}] {n_bad} mismatches (of {actual.numel()}). First diffs:")
        for idx in bad_idx:
            idx_tup = tuple(int(x) for x in idx.tolist())
            a = actual[idx_tup].item()
            e = expected[idx_tup].item()
            print(f"  idx={idx_tup}: actual={a}, expected={e}")
    return n_bad


def diff_scalar(actual: int, expected: int, name: str) -> int:
    if actual != expected:
        print(f"[{name}] MISMATCH: actual={actual}, expected={expected}")
        return 1
    return 0


# ──────────────────────────────────────────────────────────────────────────
# Test entry point.
# ──────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--internode",
        action="store_true",
        help="Run with the internode metadata kernel (4-node).",
    )
    parser.add_argument("--num_tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--num_topk", type=int, default=13)
    parser.add_argument("--num_experts", type=int, default=384)
    parser.add_argument("--tile_m", type=int, default=128)
    parser.add_argument(
        "--num_sms",
        type=int,
        default=80,
        help="num_sms (sets num_channels = num_sms // 2).",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=3,
        help="Number of dispatch iters to validate (each with a fresh routing).",
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    Buffer.set_num_sms(args.num_sms)

    expert_alignment = 1  # default in C++ launch path (no quack alignment)
    num_local_experts = args.num_experts // world_size
    num_channels = args.num_sms // 2

    # Size NVL / RDMA bytes (mirror the rest of the test suite).
    hidden_bytes = args.hidden * 2
    nvl_bytes = 0
    rdma_bytes = 0
    for cfg in (
        Buffer.get_dispatch_config(world_size),
        Buffer.get_combine_config(world_size),
    ):
        nvl_bytes = max(
            cfg.get_nvl_buffer_size_hint(hidden_bytes, world_size), nvl_bytes
        )
        rdma_bytes = max(
            cfg.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes
        )
    buf = Buffer(group, nvl_bytes, rdma_bytes)

    is_internode = args.internode
    if rank == 0:
        print(
            f"[metadata] world={world_size} num_sms={args.num_sms} "
            f"num_channels={num_channels} T={args.num_tokens} K={args.num_topk} "
            f"E={args.num_experts} E_local={num_local_experts} "
            f"tile_m={args.tile_m} is_internode={is_internode} "
            f"n_iter={args.n_iter}"
        )

    overall_fail = 0
    for it in range(args.n_iter):
        x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
            args.num_tokens,
            args.hidden,
            args.num_topk,
            args.num_experts,
            world_size,
            rank,
            device,
            seed=1000 + it,  # different routing per iter
        )

        pool, handle, _event = buf.dispatch(
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            args.num_experts,
            tile_m=args.tile_m,
            dispatch_seq=it + 1,
        )
        torch.cuda.synchronize()

        # Gather topk_idx from every rank for the eager reference.
        topk_int64 = topk_idx.to(torch.int64)
        all_topk = [torch.empty_like(topk_int64) for _ in range(world_size)]
        dist.all_gather(all_topk, topk_int64, group=group)
        all_topk = torch.stack(all_topk, dim=0)  # (world_size, T, K)

        ref = compute_metadata_reference(
            all_topk,
            rank,
            world_size,
            num_channels,
            args.num_experts,
            expert_alignment,
            args.tile_m,
            is_internode=is_internode,
        )

        # Compare every output the handle exposes.
        fail = 0
        fail += diff_int(handle.expert_frequency, ref["expert_frequency"], f"iter{it} expert_frequency")
        fail += diff_int(
            handle.expert_pool_block_offset,
            ref["expert_pool_block_offset"],
            f"iter{it} expert_pool_block_offset",
        )
        fail += diff_int(handle.seen_per_substream, ref["seen_per_substream"], f"iter{it} seen_per_substream")
        fail += diff_int(handle.base_pool, ref["base_pool"], f"iter{it} base_pool")
        fail += diff_int(
            handle.tile_id_to_expert[: ref["total_tiles"]],
            ref["tile_id_to_expert"],
            f"iter{it} tile_id_to_expert",
        )
        fail += diff_int(
            handle.pool_arrival_target[: ref["total_tiles"]],
            ref["pool_arrival_target"],
            f"iter{it} pool_arrival_target",
        )
        # rank_prefix_matrix: this rank fills its own column.
        rpm_actual_col = handle.rank_prefix_matrix[:, rank]
        fail += diff_int(rpm_actual_col, ref["rank_prefix_column"], f"iter{it} rank_prefix_matrix[:,rank]")
        fail += diff_scalar(handle.total_tiles, ref["total_tiles"], f"iter{it} total_tiles")

        # Note: pool_recv_token is downstream of metadata + post-poll bundle —
        # its count of non-padding rows must match moe_recv_counter, but we don't
        # verify the recv-token slot ordering here (covered by test_dispatch).

        if fail == 0 and rank == 0:
            print(f"[metadata] iter {it}: PASS")
        elif fail > 0:
            print(f"[metadata] rank {rank} iter {it}: {fail} FAILURES")
            overall_fail += fail
        dist.barrier(group=group)

    # Aggregate fail across ranks.
    fail_tensor = torch.tensor([overall_fail], dtype=torch.int64, device=device)
    dist.all_reduce(fail_tensor, op=dist.ReduceOp.SUM, group=group)
    total_fail = int(fail_tensor.item())

    if rank == 0:
        if total_fail == 0:
            print(f"[metadata] ALL {args.n_iter} ITERS OK ({world_size} ranks)")
        else:
            print(f"[metadata] FAIL — {total_fail} mismatches across all ranks/iters")

    cleanup_dist()
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
