"""Cross-layer correctness check for the streaming pipeline.

Runs `streaming_moe_layer` for N iterations with the SAME inputs and asserts
each iteration's output matches a torch-eager reference. This catches the
class of cross-layer race the historical kernel-A had, where some CTAs of
iter N+1 saw stale "all done" state from iter N (or vice versa) and produced
wrong outputs in some iterations but not others.

Symptom signature for that bug class: iter 0 correct, iter K (K >= 1) silently
wrong. The bench_pipeline.py timing harness wouldn't catch it (timing-only),
and the per-kernel test suite uses a single iter.

Launch
------
    torchrun --nproc_per_node=8 \\
        -m evolutionaryscale.models.moe.streaming_moe.validate_multi_iter \\
        [--n_iter 20] [--rtol 5e-2] [--atol 5e-2]

Outputs PASS / FAIL per iter on rank 0.
"""

import argparse
import os

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F

from evolutionaryscale.models.moe.streaming_moe.profile_pipeline import (
    DTYPE,
    NUM_EXPERTS,
    NUM_SMS,
    SEQ_LEN_PER_RANK,
    TILE_M,
    TILE_N_A,
    TILE_N_Y,
    TOPK,
    H,
    I,
    make_buffer,
    make_uniform_topk_idx,
)
from evolutionaryscale.models.moe.streaming_moe.streaming_moe import streaming_moe_layer


def torch_reference_recv(
    pool: torch.Tensor,
    pool_recv_token: torch.Tensor,
    pool_topk_weight: torch.Tensor,
    tile_id_to_expert: torch.Tensor,
    expert_pool_block_offset: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    T_recv: int,
    tile_m: int,
) -> torch.Tensor:
    """Eager torch reproduction of kernel A + kernel Y on the local-rank
    receive side. Returns o_ref[T_recv, H].

    For each pool slot s:
      a = SwiGLU(pool[s] @ W1[expert(s)])    # tile_m × I → I
      y = a @ W2[expert(s)]                   # I → H
      o[recv_token(s)] += topk_weight(s) * y
    """
    H_dim = pool.shape[1]
    o_ref = torch.zeros(T_recv, H_dim, dtype=torch.float32, device=pool.device)
    E_local = expert_pool_block_offset.shape[0] - 1

    # W1: (E_local, 2I, H), W2: (E_local, H, I).
    for e in range(E_local):
        slot_lo = expert_pool_block_offset[e].item() * tile_m
        slot_hi = expert_pool_block_offset[e + 1].item() * tile_m
        if slot_lo == slot_hi:
            continue
        # pool slice for this expert: (slot_hi - slot_lo) × H
        x_e = pool[slot_lo:slot_hi].to(torch.float32)
        # h = x_e @ W1[e].T → (n_slots, 2I)
        h = x_e @ w1_local[e].to(torch.float32).T
        # SwiGLU: split 2I into [2I/2, 2I/2], silu(first) * second
        u, v = h.chunk(2, dim=-1)
        a = F.silu(u) * v  # (n_slots, I)
        # y = a @ W2[e].T → (n_slots, H)
        y = a @ w2_local[e].to(torch.float32).T
        # weighted scatter
        weights = pool_topk_weight[slot_lo:slot_hi].to(torch.float32)
        recv_tokens = pool_recv_token[slot_lo:slot_hi]
        valid = recv_tokens >= 0
        weighted = weights[:, None] * y
        o_ref.index_add_(0, recv_tokens[valid].to(torch.int64), weighted[valid])
    return o_ref.to(DTYPE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_sms", type=int, default=NUM_SMS)
    p.add_argument("--seq_len", type=int, default=SEQ_LEN_PER_RANK)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument("--rtol", type=float, default=5e-2)
    p.add_argument("--atol", type=float, default=5e-2)
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    torch_dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = torch_dist.group.WORLD
    device = torch.device(f"cuda:{local_rank}")
    local_E = NUM_EXPERTS // world_size

    if rank == 0:
        print(
            f"[validate] world={world_size} num_sms={args.num_sms} "
            f"H={H} I={I} E={NUM_EXPERTS} K={TOPK} T={args.seq_len} "
            f"n_warmup={args.n_warmup} n_iter={args.n_iter} "
            f"rtol={args.rtol} atol={args.atol}",
            flush=True,
        )

    buffer = make_buffer(group, args.num_sms)

    g = torch.Generator(device=device).manual_seed(42)
    w1_full = (
        torch.randn(NUM_EXPERTS, 2 * I, H, dtype=DTYPE, device=device, generator=g)
        * 0.02
    ).contiguous()
    w2_full = (
        torch.randn(NUM_EXPERTS, H, I, dtype=DTYPE, device=device, generator=g) * 0.02
    ).contiguous()
    w1_local = w1_full[rank * local_E : (rank + 1) * local_E].contiguous()
    w2_local = w2_full[rank * local_E : (rank + 1) * local_E].contiguous()

    torch.manual_seed(100 + rank)
    x = (torch.randn(args.seq_len, H, dtype=DTYPE, device=device) * 0.1).contiguous()
    topk_idx = make_uniform_topk_idx(args.seq_len, TOPK, NUM_EXPERTS, rank, device)
    topk_weights = torch.softmax(
        torch.randn(args.seq_len, TOPK, dtype=torch.float32, device=device), dim=-1
    ).contiguous()

    rank_idx = topk_idx // local_E
    is_token_in_rank = torch.zeros(
        (args.seq_len, world_size), dtype=torch.bool, device=device
    )
    for r in range(world_size):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)

    comm_stream = torch.cuda.Stream()
    compute_a_stream = torch.cuda.Stream()
    compute_y_stream = torch.cuda.Stream()
    torch_dist.barrier(group=group)

    # Warmup (no validation).
    for warm_seq in range(1, args.n_warmup + 1):
        streaming_moe_layer(
            buffer,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1_local,
            w2_local,
            comm_stream=comm_stream,
            compute_a_stream=compute_a_stream,
            compute_y_stream=compute_y_stream,
            num_experts=NUM_EXPERTS,
            dispatch_seq=warm_seq,
            tile_m=TILE_M,
            tile_n_a=TILE_N_A,
            tile_n_y=TILE_N_Y,
        )
    torch.cuda.synchronize()
    torch_dist.barrier(group=group)

    # Validated iters.
    fail_count = 0
    for step in range(args.n_iter):
        seq = 100 + step
        o_actual, handle = streaming_moe_layer(
            buffer,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1_local,
            w2_local,
            comm_stream=comm_stream,
            compute_a_stream=compute_a_stream,
            compute_y_stream=compute_y_stream,
            num_experts=NUM_EXPERTS,
            dispatch_seq=seq,
            tile_m=TILE_M,
            tile_n_a=TILE_N_A,
            tile_n_y=TILE_N_Y,
        )
        torch.cuda.synchronize()

        # streaming_moe_layer returns (o_actual, handle) but not pool —
        # re-dispatch privately with a distinct seq to grab a pool snapshot
        # for the reference. Cross-checking infrastructure, not on the perf
        # path; cheap relative to the validation cost.
        T_recv = handle.o.shape[0]
        with torch.cuda.stream(comm_stream):
            pool_check, handle_check, _ = buffer.dispatch(
                x,
                topk_idx,
                topk_weights,
                is_token_in_rank,
                NUM_EXPERTS,
                tile_m=TILE_M,
                dispatch_seq=10000 + step,
            )
        torch.cuda.current_stream().wait_stream(comm_stream)
        torch.cuda.synchronize()

        o_ref = torch_reference_recv(
            pool_check,
            handle_check.pool_recv_token,
            handle_check.pool_topk_weight,
            handle_check.tile_id_to_expert,
            handle_check.expert_pool_block_offset,
            w1_local,
            w2_local,
            T_recv=T_recv,
            tile_m=TILE_M,
        )

        o_actual = o_actual.to(torch.float32)
        o_ref_f = o_ref.to(torch.float32)
        diff = (o_actual - o_ref_f).abs()
        rel = diff / (o_ref_f.abs() + 1e-3)
        bad = (diff > args.atol) & (rel > args.rtol)
        max_abs = diff.max().item()
        max_rel = rel.max().item()
        n_bad = bad.sum().item()
        ok = n_bad == 0

        # All-reduce ok across ranks so all ranks see the same outcome.
        ok_t = torch.tensor([1 if ok else 0], device=device, dtype=torch.int32)
        torch_dist.all_reduce(ok_t, op=torch_dist.ReduceOp.MIN)
        all_ok = ok_t.item() == 1

        if rank == 0:
            tag = "PASS" if all_ok else "FAIL"
            print(
                f"[validate] iter {step:3d} seq={seq}: {tag}  "
                f"max_abs={max_abs:.4f} max_rel={max_rel:.4f} "
                f"n_bad={n_bad} (this rank)",
                flush=True,
            )
        if not all_ok:
            fail_count += 1

    torch_dist.barrier(group=group)
    if rank == 0:
        if fail_count == 0:
            print(f"[validate] ALL {args.n_iter} ITERS OK", flush=True)
        else:
            print(
                f"[validate] {fail_count} / {args.n_iter} iters FAILED — "
                "cross-layer race or correctness regression suspected",
                flush=True,
            )

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
