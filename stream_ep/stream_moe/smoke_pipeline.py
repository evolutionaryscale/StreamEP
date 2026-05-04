"""Smoke test for the streaming pipeline up to kernel Y.

Runs the same dispatch + kernel A + kernel Y pipeline as profile_pipeline.py
but without torch.profiler. Times warmup + timed iterations and prints
per-rank timings. Useful to bisect whether multi-GPU slowness comes from the
profiler vs. the pipeline itself, or to sanity-check that the full pipeline
runs end-to-end without hanging or producing invalid results.

Launch:
    torchrun --nproc_per_node=8 \\
        -m evolutionaryscale.models.moe.streaming_moe.smoke_pipeline
"""

import argparse
import time

import torch
import torch.distributed as torch_dist

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
from evolutionaryscale.utils.distributed import (
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
        f"n_warmup={args.n_warmup} n_iter={args.n_iter}",
        flush=True,
    )

    t0 = time.time()
    buffer = make_buffer(group, args.num_sms)
    rank_zero_print(f"[smoke] buffer init: {time.time() - t0:.2f}s", flush=True)

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

    dispatch_stream = torch.cuda.Stream()
    compute_a_stream = torch.cuda.Stream()
    compute_y_stream = torch.cuda.Stream()
    combine_stream = torch.cuda.Stream()
    barrier(group)

    t0 = time.time()
    for warm_seq in range(1, args.n_warmup + 1):
        streaming_moe_layer(
            buffer,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1_local,
            w2_local,
            dispatch_stream=dispatch_stream,
            compute_a_stream=compute_a_stream,
            compute_y_stream=compute_y_stream,
            combine_stream=combine_stream,
            num_experts=NUM_EXPERTS,
            dispatch_seq=warm_seq,
            tile_m=TILE_M,
            tile_n_a=TILE_N_A,
            tile_n_y=TILE_N_Y,
        )
    torch.cuda.synchronize()
    barrier(group)
    t_warmup = time.time() - t0
    rank_zero_print(
        f"[smoke] warmup ({args.n_warmup} iters, includes JIT): "
        f"{t_warmup:.2f}s ({t_warmup / args.n_warmup * 1e3:.1f} ms/iter avg)",
        flush=True,
    )

    t0 = time.time()
    for step in range(args.n_iter):
        streaming_moe_layer(
            buffer,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1_local,
            w2_local,
            dispatch_stream=dispatch_stream,
            compute_a_stream=compute_a_stream,
            compute_y_stream=compute_y_stream,
            combine_stream=combine_stream,
            num_experts=NUM_EXPERTS,
            dispatch_seq=100 + step,
            tile_m=TILE_M,
            tile_n_a=TILE_N_A,
            tile_n_y=TILE_N_Y,
        )
    torch.cuda.synchronize()
    barrier(group)
    t_iter = time.time() - t0
    rank_zero_print(
        f"[smoke] timed   ({args.n_iter} iters):              "
        f"{t_iter:.2f}s ({t_iter / args.n_iter * 1e3:.1f} ms/iter avg)",
        flush=True,
    )
    rank_zero_print("[smoke] OK", flush=True)

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
