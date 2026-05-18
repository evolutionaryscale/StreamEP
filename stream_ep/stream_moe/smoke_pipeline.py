"""Smoke test for the streaming pipeline up to kernel Y.

Runs the same dispatch + kernel A + kernel Y pipeline as profile_pipeline.py
but without torch.profiler. Times warmup + timed iterations and prints
per-rank timings. Useful to bisect whether multi-GPU slowness comes from the
profiler vs. the pipeline itself, or to sanity-check that the full pipeline
runs end-to-end without hanging or producing invalid results.

Launch:
    torchrun --nproc_per_node=8 \\
        -m stream_ep.stream_moe.smoke_pipeline
"""

import argparse
import time

import torch
import torch.distributed as torch_dist

from stream_ep.stream_moe.profile_pipeline import (
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
from stream_ep.stream_moe.stream_moe import (
    make_streams,
    stream_moe_func,
)
from stream_ep.stream_moe.profile_pipeline import (
    barrier,
    get_global_rank,
    get_world_size,
    init_distributed,
    rank_zero_print,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_sms", type=int, default=NUM_SMS)
    p.add_argument("--seq_len", type=int, default=SEQ_LEN_PER_RANK)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_iter", type=int, default=5)
    args = p.parse_args()

    device = init_distributed()
    rank, world_size = get_global_rank(), get_world_size()
    group = torch_dist.group.WORLD
    local_E = NUM_EXPERTS // world_size

    rank_zero_print(
        f"[smoke] config: world={world_size} num_sms={args.num_sms} "
        f"H={H} I={I} E={NUM_EXPERTS} K={TOPK} T={args.seq_len} "
        f"n_warmup={args.n_warmup} n_iter={args.n_iter}"
    )

    t0 = time.time()
    buffer = make_buffer(group, args.num_sms)
    rank_zero_print(f"[smoke] buffer init: {time.time() - t0:.2f}s")

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
    w1_local.requires_grad_(True)
    w2_local.requires_grad_(True)

    torch.manual_seed(100 + rank)
    x = (torch.randn(args.seq_len, H, dtype=DTYPE, device=device) * 0.1).contiguous()
    x.requires_grad_(True)
    topk_idx = make_uniform_topk_idx(args.seq_len, TOPK, NUM_EXPERTS, rank, device)
    topk_weights = torch.softmax(
        torch.randn(args.seq_len, TOPK, dtype=torch.float32, device=device), dim=-1
    ).contiguous()
    topk_weights.requires_grad_(True)

    rank_idx = topk_idx // local_E
    is_token_in_rank = torch.zeros(
        (args.seq_len, world_size), dtype=torch.bool, device=device
    )
    for r in range(world_size):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)

    streams = make_streams()
    barrier(group)

    t0 = time.time()
    for warm_seq in range(1, args.n_warmup + 1):
        out = stream_moe_func(
            buffer,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1_local,
            w2_local,
            streams=streams,
            num_experts=NUM_EXPERTS,
            tile_m=TILE_M,
            tile_n_a=TILE_N_A,
            tile_n_y=TILE_N_Y,
        )
        out.sum().backward()
    torch.cuda.synchronize()
    barrier(group)
    t_warmup = time.time() - t0
    rank_zero_print(
        f"[smoke] warmup ({args.n_warmup} iters, includes JIT): "
        f"{t_warmup:.2f}s ({t_warmup / args.n_warmup * 1e3:.1f} ms/iter avg)"
    )

    t0 = time.time()
    for step in range(args.n_iter):
        out = stream_moe_func(
            buffer,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1_local,
            w2_local,
            streams=streams,
            num_experts=NUM_EXPERTS,
            tile_m=TILE_M,
            tile_n_a=TILE_N_A,
            tile_n_y=TILE_N_Y,
        )
        out.sum().backward()
    torch.cuda.synchronize()
    barrier(group)
    t_iter = time.time() - t0
    rank_zero_print(
        f"[smoke] timed   ({args.n_iter} iters):              "
        f"{t_iter:.2f}s ({t_iter / args.n_iter * 1e3:.1f} ms/iter avg)"
    )
    rank_zero_print("[smoke] OK")

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
