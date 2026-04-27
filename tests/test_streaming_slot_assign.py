"""Phase A.5 signal-correctness test for the streaming-MoE pipeline.

Runs a real intranode dispatch + ``Buffer.streaming_dispatch_finalize`` and
verifies:

  1. Every (r, k) pair with recv_topk_idx[r, k] >= 0 lands in exactly one tile
     slot, and the slot's metadata agrees with recv_topk_idx.
  2. tile_remaining is all-zero post-finalize.
  3. tile_ready_queue_seq[i] == dispatch_seq for i in [0, total_tiles), and
     tile_ready_queue_head == total_tiles.
  4. Bit-determinism: re-running the slot-assign step on a fresh tile_remaining
     reproduces tile_records_* exactly.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from deep_ep import Buffer


def make_topk_idx(num_tokens, num_topk, num_experts, rank, device):
    g = torch.Generator(device=device).manual_seed(123 + rank)
    return torch.randint(0, num_experts, (num_tokens, num_topk),
                         generator=g, device=device, dtype=torch.int64)


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

    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    topk_idx = make_topk_idx(num_tokens, num_topk, num_experts, rank, device)
    topk_weights = torch.full((num_tokens, num_topk), 1.0 / num_topk,
                              dtype=torch.float32, device=device)

    n_tpr, n_tprdma, n_tpe, is_tir, _ = buf.get_dispatch_layout(
        topk_idx, num_experts, async_finish=False)
    (recv_x, recv_topk_idx, _recv_topk_weights, _, handle, _) = buf.dispatch(
        x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_tokens_per_rank=n_tpr, num_tokens_per_rdma_rank=n_tprdma,
        is_token_in_rank=is_tir, num_tokens_per_expert=n_tpe,
        expert_alignment=1, async_finish=False, allocate_on_comm_stream=False,
    )
    torch.cuda.synchronize()
    T_recv = recv_x.size(0)

    sh = buf.streaming_dispatch_finalize(
        topk_idx, num_experts, recv_topk_idx, handle,
        tile_m=tile_m, dispatch_seq=1,
    )
    torch.cuda.synchronize()

    if sh.total_tiles == 0:
        if rank == 0:
            print(f"PASS: rank={rank} empty dispatch")
        return

    assert sh.tile_remaining.eq(0).all().item(), \
        f"tile_remaining nonzero: {sh.tile_remaining.ne(0).sum().item()} entries"
    assert sh.tile_ready_queue_head.cpu().item() == sh.total_tiles, \
        f"queue head {sh.tile_ready_queue_head.cpu().item()} != total_tiles {sh.total_tiles}"
    seq = sh.tile_ready_queue_seq.cpu()
    assert torch.all(seq == sh.dispatch_seq), \
        f"queue seq mismatch: min={seq.min().item()} max={seq.max().item()}"

    rows = sh.tile_records_recv_x_rows.cpu()
    kslots = sh.tile_records_k_slots.cpu()
    expert_id = sh.tile_records_expert_id.cpu()
    cum = sh.cumulative_tiles_before_e.cpu()
    freq = sh.expert_frequency.cpu()
    rti_cpu = recv_topk_idx.cpu()

    valid_pairs = set()
    for tile_id in range(sh.total_tiles):
        e = expert_id[tile_id].item()
        e_first = cum[e].item()
        e_last_excl = cum[e + 1].item()
        assert e_first <= tile_id < e_last_excl, \
            f"tile {tile_id} expert {e} not in [{e_first}, {e_last_excl})"
        local_tile = tile_id - e_first
        tiles_for_e = (freq[e].item() + tile_m - 1) // tile_m
        partial = freq[e].item() - (tiles_for_e - 1) * tile_m if local_tile == tiles_for_e - 1 else tile_m
        for row_in_tile in range(partial):
            r = rows[tile_id, row_in_tile].item()
            k = kslots[tile_id, row_in_tile].item()
            assert rti_cpu[r, k].item() == e, \
                f"tile {tile_id} row {row_in_tile}: recv_topk_idx[{r},{k}]={rti_cpu[r,k].item()} != {e}"
            assert (r, k) not in valid_pairs, f"duplicate (r, k) = ({r}, {k})"
            valid_pairs.add((r, k))

    expected_pairs = set()
    for r in range(T_recv):
        for k in range(num_topk):
            e = rti_cpu[r, k].item()
            if 0 <= e < num_local_experts:
                expected_pairs.add((r, k))
    assert valid_pairs == expected_pairs, \
        f"set mismatch: extra={valid_pairs - expected_pairs} missing={expected_pairs - valid_pairs}"

    saved_rows = sh.tile_records_recv_x_rows.clone()
    saved_kslots = sh.tile_records_k_slots.clone()
    saved_expert_id = sh.tile_records_expert_id.clone()

    rerun_rows = torch.full_like(sh.tile_records_recv_x_rows, -1)
    rerun_kslots = torch.full_like(sh.tile_records_k_slots, -1)
    rerun_expert_id = torch.full_like(sh.tile_records_expert_id, -1)
    rerun_remaining = torch.full((sh.total_tiles,), tile_m, dtype=torch.int32, device=device)
    freq = sh.expert_frequency
    cum = sh.cumulative_tiles_before_e
    tiles_per_expert = (freq + tile_m - 1) // tile_m
    last_tile_count = freq - (tiles_per_expert - 1) * tile_m
    last_tile_idx = (cum[1:] - 1).to(torch.int64)
    valid = freq > 0
    if valid.any():
        rerun_remaining[last_tile_idx[valid]] = last_tile_count[valid].to(torch.int32)
    rerun_queue = torch.full_like(sh.tile_ready_queue, -1)
    rerun_queue_seq = torch.zeros_like(sh.tile_ready_queue_seq)
    rerun_queue_head = torch.zeros_like(sh.tile_ready_queue_head)

    rank_prefix_matrix, _, recv_channel_prefix_matrix, *_ = handle
    buf.runtime.streaming_slot_assign(
        recv_topk_idx, recv_channel_prefix_matrix, rank_prefix_matrix,
        sh.base, sh.expert_frequency_offset, sh.cumulative_tiles_before_e,
        sh.per_source_rank_remaining,
        rerun_rows, rerun_kslots, rerun_expert_id, rerun_remaining,
        rerun_queue, rerun_queue_seq, rerun_queue_head,
        Buffer.num_sms // 2, tile_m, sh.dispatch_seq,
    )
    torch.cuda.synchronize()

    if not torch.equal(saved_rows, rerun_rows):
        raise SystemExit(f"DETERMINISM FAIL: tile_records_recv_x_rows differs ({(saved_rows != rerun_rows).sum().item()} elements)")
    if not torch.equal(saved_kslots, rerun_kslots):
        raise SystemExit(f"DETERMINISM FAIL: tile_records_k_slots differs ({(saved_kslots != rerun_kslots).sum().item()} elements)")
    if not torch.equal(saved_expert_id, rerun_expert_id):
        raise SystemExit(f"DETERMINISM FAIL: tile_records_expert_id differs ({(saved_expert_id != rerun_expert_id).sum().item()} elements)")

    if rank == 0:
        print(f"PASS: rank={rank} world={world_size} T_recv={T_recv} "
              f"total_tiles={sh.total_tiles} valid_pairs={len(valid_pairs)} "
              f"determinism=ok")


if __name__ == "__main__":
    main()
