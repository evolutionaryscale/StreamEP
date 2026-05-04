"""End-to-end test for the streaming-MoE backward dispatch_grads kernel.

Exercises ``Buffer.dispatch_grads`` (intranode, pool layout). Same routing
as ``Buffer.dispatch`` — the kernel ships ``dL/dy[t]`` from origin → expert
ranks, K-fans into pool slots looked up from ``handle.recv_token_to_slots``.
Verifies:

  1. dL_do_pool[slot] equals the source rank's dL_dy[recv_src_idx[r]] for
     every (recv_token r, k_local k) pair with slot >= 0.
  2. bwd_y_ready[tile_id] == dispatch_seq for every tile_id in [0, total_tiles).
  3. Bit-determinism across re-runs.

Mirror of test_streaming_dispatch.py for the bwd path.
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
    T_recv = handle.o.shape[0]

    # Build a deterministic dL_dy on each rank where the value of
    # dL_dy[t, h] uniquely identifies (rank, t, h). Receiver should land
    # source-rank's value into dL_do_pool — exactly what we want to check.
    g_dy = torch.Generator(device=device).manual_seed(7919 + rank * 31)
    dL_dy = (torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device, generator=g_dy)
             + rank * 100.0)
    dL_dy = dL_dy.contiguous()

    # Run dispatch_grads.
    dL_do_pool, bwd_y_ready = buf.dispatch_grads(handle, dL_dy, dispatch_seq=1)
    torch.cuda.synchronize()

    assert dL_do_pool.shape == (TK_padded, hidden), \
        f"dL_do_pool shape {dL_do_pool.shape} != ({TK_padded}, {hidden})"
    assert dL_do_pool.dtype == torch.bfloat16
    assert bwd_y_ready.shape == (total_tiles,), \
        f"bwd_y_ready shape {bwd_y_ready.shape} != ({total_tiles},)"
    assert bwd_y_ready.dtype == torch.int64

    # Build expected: for each rank S, gather S's dL_dy. Then for each pool slot
    # with pool_recv_token[s] >= 0, determine sender_rank from recv-token's
    # position in the rank-major recv layout, and check
    # dL_do_pool[s] == all_dL_dy[sender_rank][recv_src_idx[s_recv]].
    all_dL_dy = [torch.empty_like(dL_dy) for _ in range(world_size)]
    dist.all_gather(all_dL_dy, dL_dy, group=group)
    all_is_in_rank = [torch.empty_like(is_token_in_rank) for _ in range(world_size)]
    dist.all_gather(all_is_in_rank, is_token_in_rank, group=group)

    # For each recv-token r ∈ [0, T_recv), determine sender rank via the same
    # rank-major layout fwd uses (rank 0's tokens-to-me first, then rank 1's, ...).
    recv_token_src_rank = torch.empty((T_recv,), dtype=torch.int32, device="cpu")
    cur = 0
    for src in range(world_size):
        n = int(all_is_in_rank[src][:, rank].sum().item())
        recv_token_src_rank[cur:cur + n] = src
        cur += n
    assert cur == T_recv

    pool_recv_token = handle.pool_recv_token.cpu()
    pool_k_slot = handle.pool_k_slot.cpu()
    recv_src_idx = handle.recv_src_idx.cpu()
    dL_do_pool_cpu = dL_do_pool.cpu()
    all_dL_dy_cpu = [t.cpu() for t in all_dL_dy]

    mismatches = 0
    for s in range(TK_padded):
        r = int(pool_recv_token[s].item())
        if r < 0:
            continue  # padding
        src = int(recv_token_src_rank[r].item())
        t_src = int(recv_src_idx[r].item())
        expected = all_dL_dy_cpu[src][t_src]
        got = dL_do_pool_cpu[s]
        if not torch.equal(expected, got):
            mismatches += 1
            if mismatches <= 4:
                k = int(pool_k_slot[s].item())
                print(f"[rank {rank}] slot {s} (r={r}, k={k}, src={src}, t_src={t_src}): "
                      f"expected {expected[:4]}... got {got[:4]}...")
    assert mismatches == 0, f"[rank {rank}] {mismatches} slot mismatches in dL_do_pool"

    # bwd_y_ready[tile_id] should equal dispatch_seq for every tile.
    bwd_y_ready_cpu = bwd_y_ready.cpu()
    expected_seq = 1
    assert (bwd_y_ready_cpu == expected_seq).all(), (
        f"bwd_y_ready not fully fired; first deviating tile_id: "
        f"{(bwd_y_ready_cpu != expected_seq).nonzero().flatten()[:8]}"
    )

    # Determinism: re-run dispatch_grads with same handle, same input → same output.
    dL_do_pool2, bwd_y_ready2 = buf.dispatch_grads(handle, dL_dy, dispatch_seq=2)
    torch.cuda.synchronize()
    valid = (handle.pool_recv_token.cpu() >= 0)
    assert torch.equal(dL_do_pool.cpu()[valid], dL_do_pool2.cpu()[valid]), \
        "dL_do_pool (valid rows) not deterministic"
    assert (bwd_y_ready2.cpu() == 2).all(), \
        "bwd_y_ready did not fire to dispatch_seq=2 on second call"

    if rank == 0:
        print(f"PASS: rank={rank} world={world_size} T_recv={T_recv} "
              f"TK_padded={TK_padded} total_tiles={total_tiles}")


if __name__ == "__main__":
    main()
