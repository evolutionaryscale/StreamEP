"""End-to-end test for the streaming-MoE internode dispatch_grads (bwd dispatch).

Mirrors ``tests/test_dispatch_grads.py`` (intranode bwd dispatch) for the
16-rank internode topology. Drives ``Buffer.dispatch`` and
``Buffer.dispatch_grads`` (which route through the internode entry
points via the topology branch when ``num_rdma_ranks > 1``).

Verifies:
  1. Shape + dtype of ``dL_do_pool`` and ``bwd_dispatch_arrival_count``.
  2. Per pool slot: ``dL_do_pool[s]`` equals the source rank's
     ``dL_dy[t_src]`` for the recv-token ``r`` mapped to slot ``s``
     by fwd dispatch's pool layout. Each rank tags ``dL_dy`` with
     a unique per-rank offset (``rank * 100.0``) so source identification
     is unambiguous.
  3. ``bwd_dispatch_arrival_count[tile_id]`` reaches the per-tile target
     for every tile.
  4. Determinism: re-running with the same handle + input produces
     identical ``dL_do_pool`` (valid rows) and refires the arrival count.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from stream_ep import Buffer

from utils import cleanup_dist, make_inputs


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

    num_experts = 64
    num_topk = 4
    num_tokens = 256
    hidden = 256
    tile_m = 32

    hidden_bytes = hidden * 2
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (Buffer.get_dispatch_config(world_size),
                Buffer.get_combine_config(world_size)):
        nvl_bytes  = max(cfg.get_nvl_buffer_size_hint(hidden_bytes,  world_size), nvl_bytes)
        rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes)
    buf = Buffer(group, nvl_bytes, rdma_bytes)

    x, topk_idx, topk_weights, is_token_in_rank = make_inputs(
        num_tokens, hidden, num_topk, num_experts, world_size, rank, device,
        seed=123, x_kind="randn", plant_sentinels=True)

    _, handle, _ = buf.dispatch(
        x, topk_idx, topk_weights, is_token_in_rank, num_experts,
        tile_m=tile_m, dispatch_seq=1)
    torch.cuda.synchronize()

    total_tiles = handle.total_tiles
    TK_padded = total_tiles * tile_m
    T_recv = handle.o.shape[0]

    g_dy = torch.Generator(device=device).manual_seed(7919 + rank * 31)
    dL_dy = (torch.randn((num_tokens, hidden), dtype=torch.bfloat16,
                         device=device, generator=g_dy)
             + rank * 100.0)
    dL_dy = dL_dy.contiguous()

    dL_do_pool, bwd_arrival_count, _grads_started = buf.dispatch_grads(handle, dL_dy, dispatch_seq=1)
    torch.cuda.synchronize()

    assert dL_do_pool.shape == (TK_padded, hidden), \
        f"dL_do_pool shape {dL_do_pool.shape} != ({TK_padded}, {hidden})"
    assert dL_do_pool.dtype == torch.bfloat16
    assert bwd_arrival_count.shape == (total_tiles,), \
        f"bwd_arrival_count shape {bwd_arrival_count.shape} != ({total_tiles},)"
    assert bwd_arrival_count.dtype == torch.int32

    all_dL_dy = [torch.empty_like(dL_dy) for _ in range(world_size)]
    dist.all_gather(all_dL_dy, dL_dy, group=group)
    all_is_in_rank = [torch.empty_like(is_token_in_rank) for _ in range(world_size)]
    dist.all_gather(all_is_in_rank, is_token_in_rank, group=group)

    recv_token_src_rank = torch.empty((T_recv,), dtype=torch.int32, device="cpu")
    recv_token_src_idx = torch.empty((T_recv,), dtype=torch.int32, device="cpu")
    cur = 0
    for src in range(world_size):
        src_in = all_is_in_rank[src][:, rank].cpu()
        idx_in_src = torch.nonzero(src_in, as_tuple=False).flatten().to(torch.int32)
        n = idx_in_src.numel()
        recv_token_src_rank[cur:cur + n] = src
        recv_token_src_idx[cur:cur + n] = idx_in_src
        cur += n
    assert cur == T_recv, f"recv_token mapping size {cur} != T_recv {T_recv}"

    pool_recv_token = handle.pool_recv_token.cpu()
    pool_k_slot = handle.pool_k_slot.cpu()
    dL_do_pool_cpu = dL_do_pool.cpu()
    all_dL_dy_cpu = [t.cpu() for t in all_dL_dy]

    mismatches = 0
    for s in range(TK_padded):
        r = int(pool_recv_token[s].item())
        if r < 0:
            continue
        src = int(recv_token_src_rank[r].item())
        t_src = int(recv_token_src_idx[r].item())
        expected = all_dL_dy_cpu[src][t_src]
        got = dL_do_pool_cpu[s]
        if not torch.equal(expected, got):
            mismatches += 1
            if mismatches <= 4:
                k = int(pool_k_slot[s].item())
                print(f"[rank {rank}] slot {s} (r={r}, k={k}, src={src}, t_src={t_src}): "
                      f"expected {expected[:4]}... got {got[:4]}...")
    assert mismatches == 0, f"[rank {rank}] {mismatches} slot mismatches in dL_do_pool"

    pool_arrival_target_cpu = handle.pool_arrival_target[:total_tiles].cpu()
    bwd_arrival_count_cpu = bwd_arrival_count.cpu()
    assert torch.equal(bwd_arrival_count_cpu, pool_arrival_target_cpu), (
        f"bwd_arrival_count not fully fired; first deviating tile_id: "
        f"{(bwd_arrival_count_cpu != pool_arrival_target_cpu).nonzero().flatten()[:8]}")

    dL_do_pool2, bwd_arrival_count2, _grads_started2 = buf.dispatch_grads(handle, dL_dy, dispatch_seq=2)
    torch.cuda.synchronize()
    valid = (handle.pool_recv_token.cpu() >= 0)
    assert torch.equal(dL_do_pool.cpu()[valid], dL_do_pool2.cpu()[valid]), \
        "dL_do_pool (valid rows) not deterministic across re-runs"
    assert torch.equal(bwd_arrival_count2.cpu(), pool_arrival_target_cpu), \
        "bwd_arrival_count did not refire to pool_arrival_target on second call"

    if rank == 0:
        print(f"PASS: rank={rank} world={world_size} T_recv={T_recv} "
              f"TK_padded={TK_padded} total_tiles={total_tiles}", flush=True)

    cleanup_dist()


if __name__ == "__main__":
    main()
