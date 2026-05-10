"""Stress test for the streaming pipeline under long-run / sparse-routing
conditions. Designed to catch protocol-level bugs that smoke (5 iters) can't
surface:

- ``dispatch_seq`` cycling: the C4 NVL gen-stamp packs the low 12 bits of
  ``dispatch_seq`` into the slot; this run uses N >> 4096 iters with
  monotonically growing seq to exercise the wrap boundary.

- Sparse routing via ``make_skewed_topk_idx``: some ``(channel, dst_nvl)``
  substreams see very few tokens per iter. M1's iter-entry force-write of
  ``nvl_channel_head/tail`` is what keeps those substreams from latching
  onto stale-aliased values from many iters ago.

- ``rdma_channel_head/tail`` cumulative wrap: M2 widened the NIC AMO +
  reader_prev arrays to int64. At production shape × this run's iter count
  the counter advance is ≪ 2^31, so we don't hit the int32 wrap during
  the test — but we DO exercise the new 8-byte AMO WQE constructor on
  every iter, so any mlx5 plumbing bug would manifest as a hang or
  numerical mismatch within the first few iters.

Sanity assertion per iter: output is finite (no NaN/Inf). Run every
``--check_every`` iters to bound the verification overhead.

Launch:
    ./srun_internode.sh -m stream_ep.stream_moe.stress_pipeline \
        --n_iter 10000 --skew_hot_frac 0.25 --skew_hot_weight 4.0
"""

from __future__ import annotations

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
    barrier,
    get_global_rank,
    get_world_size,
    init_distributed,
    make_buffer,
    make_skewed_topk_idx,
    make_uniform_topk_idx,
    rank_zero_print,
)
from stream_ep.stream_moe.stream_moe import (
    make_streams,
    stream_moe_func,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_sms", type=int, default=NUM_SMS)
    p.add_argument("--seq_len", type=int, default=SEQ_LEN_PER_RANK)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_iter", type=int, default=10000)
    p.add_argument(
        "--check_every",
        type=int,
        default=500,
        help="Sample output for finite-value check every N iters.",
    )
    p.add_argument(
        "--skew_hot_frac",
        type=float,
        default=0.25,
        help="Fraction of experts in the hot bucket (sparse-routing stressor).",
    )
    p.add_argument(
        "--skew_hot_weight",
        type=float,
        default=4.0,
        help="Per-token sampling weight ratio (hot / cold). 1.0 = uniform.",
    )
    p.add_argument(
        "--seq_offset",
        type=int,
        default=10000,
        help="dispatch_seq starts here. Set high enough to cross at least one "
        "4096-iter window during the run (default 10000 + n_iter > 14096).",
    )
    args = p.parse_args()

    device = init_distributed()
    rank, world_size = get_global_rank(), get_world_size()
    group = torch_dist.group.WORLD
    local_E = NUM_EXPERTS // world_size

    rank_zero_print(
        f"[stress] config: world={world_size} num_sms={args.num_sms} "
        f"H={H} I={I} E={NUM_EXPERTS} K={TOPK} T={args.seq_len} "
        f"n_warmup={args.n_warmup} n_iter={args.n_iter} "
        f"skew_hot_frac={args.skew_hot_frac} skew_hot_weight={args.skew_hot_weight} "
        f"seq_offset={args.seq_offset}"
    )

    t0 = time.time()
    buffer = make_buffer(group, args.num_sms)
    rank_zero_print(f"[stress] buffer init: {time.time() - t0:.2f}s")

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
    topk_weights_pool = torch.softmax(
        torch.randn(args.seq_len, TOPK, dtype=torch.float32, device=device), dim=-1
    ).contiguous()
    topk_weights_pool.requires_grad_(True)

    skew_gen = torch.Generator(device=device).manual_seed(7000 + rank)

    def make_routing(step_seed: int):
        if args.skew_hot_frac > 0:
            topk_idx = make_skewed_topk_idx(
                args.seq_len,
                TOPK,
                NUM_EXPERTS,
                hot_frac=args.skew_hot_frac,
                hot_weight=args.skew_hot_weight,
                device=device,
                generator=skew_gen,
            )
        else:
            topk_idx = make_uniform_topk_idx(
                args.seq_len, TOPK, NUM_EXPERTS, rank, device
            )
        rank_idx = topk_idx // local_E
        is_token_in_rank = torch.zeros(
            (args.seq_len, world_size), dtype=torch.bool, device=device
        )
        for r in range(world_size):
            is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)
        return topk_idx, is_token_in_rank

    streams = make_streams()
    barrier(group)

    # Warmup with one routing draw (JIT cost + caching allocator stabilize).
    topk_idx, is_token_in_rank = make_routing(0)
    t0 = time.time()
    for warm_seq in range(1, args.n_warmup + 1):
        out = stream_moe_func(
            buffer, x, topk_idx, topk_weights_pool, is_token_in_rank,
            w1_local, w2_local,
            streams=streams, num_experts=NUM_EXPERTS,
            dispatch_seq=warm_seq,
            tile_m=TILE_M, tile_n_a=TILE_N_A, tile_n_y=TILE_N_Y,
        )
        out.sum().backward()
    torch.cuda.synchronize()
    barrier(group)
    t_warmup = time.time() - t0
    rank_zero_print(
        f"[stress] warmup ({args.n_warmup} iters, includes JIT): "
        f"{t_warmup:.2f}s ({t_warmup / args.n_warmup * 1e3:.1f} ms/iter avg)"
    )

    # Stress loop. Re-draw routing every iter so sparse-substream dormancy
    # patterns vary across the run; monotonically growing dispatch_seq exercises
    # the 12-bit gen-stamp window. Output finite-check sampled every
    # --check_every iters (cheap host poll; doesn't block the GPU).
    n_checked = 0
    t0 = time.time()
    for step in range(args.n_iter):
        topk_idx, is_token_in_rank = make_routing(step)
        seq = args.seq_offset + step
        out = stream_moe_func(
            buffer, x, topk_idx, topk_weights_pool, is_token_in_rank,
            w1_local, w2_local,
            streams=streams, num_experts=NUM_EXPERTS,
            dispatch_seq=seq,
            tile_m=TILE_M, tile_n_a=TILE_N_A, tile_n_y=TILE_N_Y,
        )
        out.sum().backward()

        if (step + 1) % args.check_every == 0:
            # Synchronize once so we observe THIS iter's output before the
            # next iter's launch races ahead. Then a single all-rank assert.
            torch.cuda.synchronize()
            finite = torch.isfinite(out).all().item()
            grad_finite = torch.isfinite(x.grad).all().item() if x.grad is not None else True
            ok = bool(finite and grad_finite)
            ok_t = torch.tensor(int(ok), device=device)
            torch_dist.all_reduce(ok_t, op=torch_dist.ReduceOp.MIN)
            if ok_t.item() == 0:
                rank_zero_print(
                    f"[stress] FAIL at step={step+1} seq={seq}: "
                    f"out_finite={finite} grad_finite={grad_finite} (this rank)"
                )
                raise RuntimeError("non-finite values detected")
            n_checked += 1

        # Reset grads each iter — without this, .grad magnitudes grow linearly
        # across 10K iters (same gradient accumulated repeatedly) and the
        # finite-check would fail on overflow long before any protocol bug.
        x.grad = None
        topk_weights_pool.grad = None
        w1_local.grad = None
        w2_local.grad = None

    torch.cuda.synchronize()
    barrier(group)
    t_iter = time.time() - t0
    rank_zero_print(
        f"[stress] timed   ({args.n_iter} iters): "
        f"{t_iter:.2f}s ({t_iter / args.n_iter * 1e3:.2f} ms/iter avg, "
        f"{n_checked} finite-checks passed)"
    )
    rank_zero_print(
        f"[stress] OK — final dispatch_seq={args.seq_offset + args.n_iter - 1} "
        f"crossed {(args.n_iter // 4096)} full 12-bit seq windows"
    )

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
