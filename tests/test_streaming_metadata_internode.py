"""Direct test for ``internode::streaming_dispatch_metadata`` (the folded
metadata kernel that replaces ``notify_dispatch`` and absorbs
``streaming_dispatch_metadata``'s streaming-superset phases for the
internode topology).

Drives the kernel via the thin wrapper ``Buffer.runtime.streaming_metadata_test``
on 2 RDMA × 8 NVL = 16 GPUs and asserts every output against an eager-mode
reference computed from all-gather'd ``topk_idx``. No ``dispatch_main``,
no combine — isolates metadata-kernel correctness.

Run via:
  ./srun_internode.sh StreamEP/tests/test_streaming_metadata_internode.py

The eager reference covers:
  - ``seen_per_substream[c, src_world, e_local]`` — receiver-side per-substream
    per-expert (token, k) pair count.
  - ``expert_frequency[e_local]`` — sum over (c, src_world).
  - ``expert_pool_block_offset[e+1]`` — ceil-div prefix sum (tile units).
  - ``base_pool[c, src_world, e_local]`` — slot-start, expert-major then
    (src_world, c) lex order within each expert.
  - ``tile_id_to_expert[tile_id]`` and ``pool_arrival_target[tile_id]``
    — per-tile arrays.
  - ``moe_recv_counter`` (=num_recv), ``moe_recv_rdma_counter``,
    ``moe_recv_expert_counter[e]`` (aligned), ``streaming_total_tiles``.
  - ``rank_prefix_matrix[i, this_rank]`` — cumulative-recv-from-senders-0..i.
  - ``rdma_channel_prefix_matrix`` / ``gbl_channel_prefix_matrix`` —
    per-(dst_rank, channel) sender-side cumulative counts.

Edge cases planted in inputs:
  - ``-1`` sentinels in ``topk_idx`` (~5% via deterministic mask) to exercise
    the skip branch.
  - Some experts with zero tokens this iter (no contributions across all
    senders) to verify ``expert_pool_block_offset`` handles empty experts.
  - Bit-determinism check: re-run the kernel; outputs must be identical.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

import stream_ep
from stream_ep import Buffer

from utils import cleanup_dist


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────

def make_topk_idx(num_tokens: int, num_topk: int, num_experts: int, rank: int,
                  *, device: torch.device, plant_empty_expert: int | None = None):
    """Per-rank ``topk_idx`` with planted edge cases.

    Each rank seeds ``123 + rank`` so all 16 ranks produce different routing.
    A ~5% mask plants ``-1`` sentinels (skip branch). Optionally pin every
    occurrence of ``plant_empty_expert`` to ``-1`` so that expert ends up
    receiving zero tokens this iter (forces the empty-expert branch in
    ``expert_pool_block_offset``).
    """
    g = torch.Generator(device=device).manual_seed(123 + rank)
    idx = torch.randint(0, num_experts, (num_tokens, num_topk),
                        generator=g, device=device, dtype=torch.int64)
    sentinel = torch.rand((num_tokens, num_topk), generator=g, device=device) < 0.05
    idx = torch.where(sentinel, torch.full_like(idx, -1), idx)
    if plant_empty_expert is not None:
        idx = torch.where(idx == plant_empty_expert, torch.full_like(idx, -1), idx)
    return idx.to(stream_ep.topk_idx_t)


# ─────────────────────────────────────────────────────────────────────────────
# Eager reference (vectorized via bincount-on-linear-indices).
# Returns a dict of expected outputs from `this_rank`'s receiver perspective.
# ─────────────────────────────────────────────────────────────────────────────

def compute_reference(topk_idx_all: torch.Tensor,
                      *,
                      num_world_ranks: int,
                      num_experts: int,
                      num_channels: int,
                      tile_m: int,
                      expert_alignment: int,
                      this_rank: int):
    """``topk_idx_all`` is ``[num_world_ranks, num_tokens, num_topk]`` on CPU."""
    NUM_NVL = 8  # NUM_MAX_NVL_PEERS
    num_rdma_ranks = num_world_ranks // NUM_NVL
    E_local = num_experts // num_world_ranks
    num_tokens = int(topk_idx_all.shape[1])
    num_topk = int(topk_idx_all.shape[2])
    tokens_per_channel = (num_tokens + num_channels - 1) // num_channels
    this_rdma = this_rank // NUM_NVL

    S, T, K = num_world_ranks, num_tokens, num_topk

    # ── seen_per_substream[c, src_world, e_local] for this_rank ───────────────
    routes_to_me = (topk_idx_all >= this_rank * E_local) & \
                   (topk_idx_all < (this_rank + 1) * E_local)            # [S,T,K]
    e_local = (topk_idx_all - this_rank * E_local).to(torch.int64)        # valid where mask
    s_idx = torch.arange(S).view(S, 1, 1).expand(S, T, K)
    c_idx = (torch.arange(T) // tokens_per_channel).view(1, T, 1).expand(S, T, K)
    lin = c_idx * (S * E_local) + s_idx * E_local + e_local
    seen_flat = torch.bincount(lin[routes_to_me].flatten(),
                               minlength=num_channels * S * E_local)
    seen = seen_flat.view(num_channels, S, E_local).to(torch.int32)

    # ── expert_frequency / expert_pool_block_offset / num_recv_per_expert ────
    expert_frequency = seen.sum(dim=(0, 1)).to(torch.int32)               # [E_local]
    n_blocks_per_e = ((expert_frequency + tile_m - 1) // tile_m).to(torch.int32)
    expert_pool_block_offset = torch.zeros(E_local + 1, dtype=torch.int32)
    expert_pool_block_offset[1:] = torch.cumsum(n_blocks_per_e, dim=0).to(torch.int32)
    total_tiles = int(expert_pool_block_offset[-1])
    num_recv_per_expert = (((expert_frequency + expert_alignment - 1) //
                            expert_alignment) * expert_alignment).to(torch.int32)

    # ── base_pool[c, src_world, e_local] = expert_block_offset[e]*tile_m
    #     + Σ over (c', s') < (c, s) lex of seen[c', s', e]   ──────────────────
    # Walk each expert's substreams in (c, s) lex order: per-expert running
    # accumulator; at each (c, s) write current acc, then add seen[c, s, e].
    # Vectorized as a per-expert cumsum along the flattened (c, s) axis.
    seen_cs_e = seen.permute(2, 0, 1).reshape(E_local, num_channels * S)   # [E, C*S]
    cs_cumsum = torch.cumsum(seen_cs_e, dim=1).to(torch.int32)             # [E, C*S]
    cs_pre   = cs_cumsum - seen_cs_e                                       # exclusive prefix
    block_off_e = (expert_pool_block_offset[:-1].to(torch.int32) * tile_m) \
                  .view(E_local, 1)
    base_pool_E_CS = block_off_e + cs_pre                                  # [E, C*S]
    base_pool = base_pool_E_CS.view(E_local, num_channels, S) \
                              .permute(1, 2, 0).contiguous()                # [C, S, E]

    # ── tile_id_to_expert / pool_arrival_target ──────────────────────────────
    tile_id_to_expert = torch.zeros(total_tiles, dtype=torch.int32)
    pool_arrival_target = torch.zeros(total_tiles, dtype=torch.int32)
    for e in range(E_local):
        e_start = int(expert_pool_block_offset[e])
        e_end = int(expert_pool_block_offset[e + 1])
        n_tiles = e_end - e_start
        n_e = int(expert_frequency[e])
        if n_tiles == 0:
            continue
        tile_id_to_expert[e_start:e_end] = e
        pool_arrival_target[e_start:e_end - 1] = tile_m
        pool_arrival_target[e_end - 1] = n_e - (n_tiles - 1) * tile_m

    # ── num_recv (host-mapped moe_recv_counter): UNIQUE (src_world, t) pairs
    #     where token routes to this_rank ─────────────────────────────────────
    # A token may route to this_rank via multiple k's; counted once per (s, t).
    routes_to_me_token = routes_to_me.any(dim=2)                           # [S, T]
    num_recv = int(routes_to_me_token.sum())

    # ── num_recv_rdma (moe_recv_rdma_counter) and
    #    recv_rdma_rank_prefix_sum: PER-THIS-NVL-SLOT.
    #
    # Notify_dispatch's RDMA exchange pairs same-NVL-slot ranks: this rank's
    # RDMA inbox slot[s_rdma] = count from sender (s_rdma, this_nvl) to this
    # rank's RDMA. So:
    #   num_recv_rdma             = Σ over s_rdma of (tokens at (s_rdma, this_nvl)
    #                                                  routing to ANY nvl in this_rdma)
    #   recv_rdma_rank_prefix_sum[d] = cumulative the above for s_rdma in 0..d.
    rdma_lo = this_rdma * NUM_NVL * E_local
    rdma_hi = (this_rdma + 1) * NUM_NVL * E_local
    routes_to_my_rdma = (topk_idx_all >= rdma_lo) & (topk_idx_all < rdma_hi)
    routes_to_my_rdma_token = routes_to_my_rdma.any(dim=2)              # [S, T]
    this_nvl = this_rank % NUM_NVL
    # Pick out senders at this_nvl across all src_rdma:
    senders_at_my_nvl = torch.tensor(
        [s_rdma * NUM_NVL + this_nvl for s_rdma in range(num_rdma_ranks)])
    per_src_rdma_at_my_nvl = routes_to_my_rdma_token[senders_at_my_nvl].sum(dim=1).to(torch.int32)
    num_recv_rdma = int(per_src_rdma_at_my_nvl.sum())
    recv_rdma_rank_prefix_sum_ref = torch.cumsum(per_src_rdma_at_my_nvl, dim=0).to(torch.int32)

    # ── rank_prefix_matrix[i, this_rank] = cumulative-recv-from-senders-0..i ─
    per_src_unique = routes_to_me_token.sum(dim=1).to(torch.int32)         # [S]
    rank_prefix_col = torch.cumsum(per_src_unique, dim=0).to(torch.int32)  # [S]

    # ── recv_gbl_rank_prefix_sum[s] = cumulative-recv-from-senders-0..s
    #     (same as rank_prefix_col here — both count unique recv-tokens
    #     prefix-summed over src_world). ──────────────────────────────────────
    recv_gbl_rank_prefix_sum = rank_prefix_col.clone()

    # recv_rdma_rank_prefix_sum was computed above (per-this-NVL-slot semantic).
    recv_rdma_rank_prefix_sum = recv_rdma_rank_prefix_sum_ref

    # ── gbl_channel_prefix_matrix[dst_world, c] (sender-side at THIS rank):
    #     cumulative count of tokens sent from THIS rank to dst_world up
    #     through and including channel c. ────────────────────────────────────
    my_topk = topk_idx_all[this_rank]                                      # [T, K]
    routes_my_to_d = (my_topk.view(T, K, 1) >= 0) & \
                     ((my_topk.view(T, K, 1) // E_local) ==
                      torch.arange(num_world_ranks).view(1, 1, num_world_ranks))
    # is_token_in_rank for this rank as sender: any-k routes to dst_world.
    in_dst_token = routes_my_to_d.any(dim=1)                                # [T, num_world]
    c_t = torch.arange(T) // tokens_per_channel                             # [T]
    # per-(dst_world, c) count = sum over t of (in_dst_token & c_t==c).
    onehot_c = torch.nn.functional.one_hot(c_t, num_classes=num_channels).to(torch.int32)
    per_dst_per_c = (in_dst_token.to(torch.int32).t() @ onehot_c)           # [num_world, num_channels]
    gbl_channel_prefix_matrix = torch.cumsum(per_dst_per_c, dim=1).to(torch.int32)

    # ── rdma_channel_prefix_matrix[dst_rdma, c] (sender-side at THIS rank):
    #     cumulative count of tokens sent from THIS rank to ANY rank in
    #     dst_rdma up through and including channel c. ───────────────────────
    in_dst_rdma_token = in_dst_token.view(T, num_rdma_ranks, NUM_NVL).any(dim=2)  # [T, num_rdma]
    per_rdma_per_c = (in_dst_rdma_token.to(torch.int32).t() @ onehot_c)
    rdma_channel_prefix_matrix = torch.cumsum(per_rdma_per_c, dim=1).to(torch.int32)

    return {
        'seen_per_substream':         seen,
        'expert_frequency':           expert_frequency,
        'expert_pool_block_offset':   expert_pool_block_offset,
        'base_pool':                  base_pool,
        'tile_id_to_expert':          tile_id_to_expert,
        'pool_arrival_target':        pool_arrival_target,
        'num_recv':                   num_recv,
        'num_recv_rdma':              num_recv_rdma,
        'num_recv_per_expert':        num_recv_per_expert,
        'total_tiles':                total_tiles,
        'rank_prefix_col':            rank_prefix_col,
        'recv_gbl_rank_prefix_sum':   recv_gbl_rank_prefix_sum,
        'recv_rdma_rank_prefix_sum':  recv_rdma_rank_prefix_sum,
        'gbl_channel_prefix_matrix':  gbl_channel_prefix_matrix,
        'rdma_channel_prefix_matrix': rdma_channel_prefix_matrix,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def assert_eq(name: str, got: torch.Tensor, expected: torch.Tensor, *, rank: int):
    got_cpu = got.detach().cpu()
    if got_cpu.shape != expected.shape:
        raise AssertionError(
            f"rank {rank}: {name} shape mismatch (got {tuple(got_cpu.shape)}, "
            f"expected {tuple(expected.shape)})")
    if not torch.equal(got_cpu, expected):
        diff = (got_cpu != expected).nonzero(as_tuple=False)
        first = diff[0].tolist() if diff.numel() else None
        raise AssertionError(
            f"rank {rank}: {name} value mismatch at {diff.shape[0]} positions; "
            f"first idx {first}: got={got_cpu[tuple(first)] if first else None}, "
            f"expected={expected[tuple(first)] if first else None}")


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    assert world_size % 8 == 0 and world_size > 8, \
        f"This test requires multi-RDMA (world_size > 8, world_size % 8 == 0); got {world_size}"

    num_sms = 24
    Buffer.set_num_sms(num_sms)
    num_channels = num_sms // 2
    num_experts = 64
    num_local_experts = num_experts // world_size
    num_topk = 4
    num_tokens = 256
    hidden = 256
    tile_m = 32
    expert_alignment = 1

    # Plant an empty-expert edge case at expert 0 (all senders get -1 there).
    plant_empty_expert = 0

    # Buffer sizing — mirror test_streaming_dispatch.py shape (rdma + nvl
    # buffers sized via Config hints for this world_size).
    hidden_bytes = hidden * 2
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (Buffer.get_dispatch_config(world_size),
                Buffer.get_combine_config(world_size)):
        nvl_bytes  = max(cfg.get_nvl_buffer_size_hint(hidden_bytes,  world_size), nvl_bytes)
        rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes)
    buf = Buffer(group, nvl_bytes, rdma_bytes)

    # Per-rank input.
    topk_idx = make_topk_idx(num_tokens, num_topk, num_experts, rank,
                             device=device, plant_empty_expert=plant_empty_expert)

    # All-gather everyone's topk_idx for the eager reference. Reference is
    # computed on CPU to keep memory pressure low at world_size=16.
    topk_idx_all_list = [torch.empty_like(topk_idx) for _ in range(world_size)]
    dist.all_gather(topk_idx_all_list, topk_idx, group=group)
    topk_idx_all = torch.stack(topk_idx_all_list, dim=0).cpu()

    ref = compute_reference(topk_idx_all,
                            num_world_ranks=world_size,
                            num_experts=num_experts,
                            num_channels=num_channels,
                            tile_m=tile_m,
                            expert_alignment=expert_alignment,
                            this_rank=rank)

    # Skip cleanly if the C++ entry point isn't built yet (kernel impl pending).
    if not hasattr(buf.runtime, 'streaming_metadata_test'):
        if rank == 0:
            print("[skip] Buffer.runtime.streaming_metadata_test not built — "
                  "kernel implementation pending. Reference computation OK.",
                  flush=True)
        cleanup_dist()
        return

    # Drive the kernel.
    cfg = Buffer.get_dispatch_config(world_size)
    out = buf.runtime.streaming_metadata_test(
        topk_idx, num_experts, expert_alignment, tile_m, cfg)
    torch.cuda.synchronize()

    # ── Asserts ──────────────────────────────────────────────────────────────
    assert_eq('expert_frequency',           out.expert_frequency,           ref['expert_frequency'],           rank=rank)
    assert_eq('expert_pool_block_offset',   out.expert_pool_block_offset,   ref['expert_pool_block_offset'],   rank=rank)
    assert_eq('seen_per_substream',         out.seen_per_substream,         ref['seen_per_substream'],         rank=rank)
    assert_eq('base_pool',                  out.base_pool,                  ref['base_pool'],                  rank=rank)
    assert_eq('tile_id_to_expert',          out.tile_id_to_expert,          ref['tile_id_to_expert'],          rank=rank)
    assert_eq('pool_arrival_target',        out.pool_arrival_target,        ref['pool_arrival_target'],        rank=rank)
    assert_eq('rdma_channel_prefix_matrix', out.rdma_channel_prefix_matrix, ref['rdma_channel_prefix_matrix'], rank=rank)
    assert_eq('gbl_channel_prefix_matrix',  out.gbl_channel_prefix_matrix,  ref['gbl_channel_prefix_matrix'],  rank=rank)
    assert_eq('recv_rdma_rank_prefix_sum',  out.recv_rdma_rank_prefix_sum,  ref['recv_rdma_rank_prefix_sum'],  rank=rank)
    assert_eq('recv_gbl_rank_prefix_sum',   out.recv_gbl_rank_prefix_sum,   ref['recv_gbl_rank_prefix_sum'],   rank=rank)
    # rank_prefix_matrix: only this rank's column is populated by the metadata
    # kernel (mirrors intranode); we check that column.
    assert_eq('rank_prefix_matrix[:, rank]',
              out.rank_prefix_matrix[:, rank], ref['rank_prefix_col'], rank=rank)

    # Host-mapped scalar / per-expert counters.
    assert int(out.num_recv) == int(ref['num_recv']), \
        f"rank {rank}: num_recv mismatch (got {int(out.num_recv)}, expected {int(ref['num_recv'])})"
    assert int(out.num_recv_rdma) == int(ref['num_recv_rdma']), \
        f"rank {rank}: num_recv_rdma mismatch (got {int(out.num_recv_rdma)}, expected {int(ref['num_recv_rdma'])})"
    assert int(out.total_tiles) == int(ref['total_tiles']), \
        f"rank {rank}: total_tiles mismatch (got {int(out.total_tiles)}, expected {int(ref['total_tiles'])})"
    assert_eq('num_recv_per_expert',
              out.num_recv_per_expert, ref['num_recv_per_expert'], rank=rank)

    # ── Bit-determinism: re-run with same inputs, same outputs ──────────────
    out2 = buf.runtime.streaming_metadata_test(
        topk_idx, num_experts, expert_alignment, tile_m, cfg)
    torch.cuda.synchronize()
    for fld in ('expert_frequency', 'base_pool', 'seen_per_substream',
                'tile_id_to_expert', 'pool_arrival_target'):
        assert torch.equal(getattr(out, fld), getattr(out2, fld)), \
            f"rank {rank}: {fld} not bit-deterministic across re-runs"

    if rank == 0:
        print(f"PASS — internode metadata kernel matches eager reference at "
              f"world_size={world_size}, num_experts={num_experts}, "
              f"num_topk={num_topk}, num_tokens={num_tokens}", flush=True)

    cleanup_dist()


if __name__ == "__main__":
    main()
