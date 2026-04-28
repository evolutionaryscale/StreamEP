"""Smoke test for streaming kernel A at any world size.

Runs the same dispatch + kernel A pipeline as profile_kernel_a.py but without
torch.profiler. Times a few warmup iterations and a few timed iterations,
prints per-rank timings. Useful to bisect whether 8-GPU slowness comes from
the profiler or from the pipeline itself.

Launch:
    torchrun --nproc_per_node=8 \
        -m evolutionaryscale.models.moe.streaming_moe.smoke_kernel_a
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as torch_dist

from evolutionaryscale.models.moe.streaming_moe.profile_kernel_a import (
    DTYPE,
    NUM_EXPERTS,
    NUM_SMS,
    SEQ_LEN_PER_RANK,
    TOPK,
    H,
    I,
    make_buffer,
    make_uniform_topk_idx,
    one_step,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_sms", type=int, default=NUM_SMS)
    p.add_argument("--seq_len", type=int, default=SEQ_LEN_PER_RANK)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_iter", type=int, default=5)
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
            f"[smoke] config: world={world_size} num_sms={args.num_sms} "
            f"H={H} I={I} E={NUM_EXPERTS} K={TOPK} T={args.seq_len} "
            f"n_warmup={args.n_warmup} n_iter={args.n_iter}",
            flush=True,
        )

    t0 = time.time()
    buffer = make_buffer(group, args.num_sms)
    if rank == 0:
        print(f"[smoke] buffer init: {time.time()-t0:.2f}s", flush=True)

    g = torch.Generator(device=device).manual_seed(42)
    w1_full = (
        torch.randn(NUM_EXPERTS, 2 * I, H, dtype=DTYPE, device=device, generator=g)
        * 0.02
    ).contiguous()
    w1_local = w1_full[rank * local_E : (rank + 1) * local_E].contiguous()

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

    compute_a_stream = torch.cuda.Stream()
    torch_dist.barrier(group=group)

    t0 = time.time()
    for warm_seq in range(1, args.n_warmup + 1):
        one_step(
            buffer,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1_local,
            compute_a_stream,
            dispatch_seq=warm_seq,
        )
    torch.cuda.synchronize()
    torch_dist.barrier(group=group)
    t_warmup = time.time() - t0
    if rank == 0:
        print(
            f"[smoke] warmup ({args.n_warmup} iters, includes JIT): {t_warmup:.2f}s "
            f"({t_warmup / args.n_warmup * 1e3:.1f} ms/iter avg)",
            flush=True,
        )

    t0 = time.time()
    for step in range(args.n_iter):
        one_step(
            buffer,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1_local,
            compute_a_stream,
            dispatch_seq=100 + step,
        )
    torch.cuda.synchronize()
    torch_dist.barrier(group=group)
    t_iter = time.time() - t0
    if rank == 0:
        print(
            f"[smoke] timed   ({args.n_iter} iters):              {t_iter:.2f}s "
            f"({t_iter / args.n_iter * 1e3:.1f} ms/iter avg)",
            flush=True,
        )
        print("[smoke] OK", flush=True)

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
