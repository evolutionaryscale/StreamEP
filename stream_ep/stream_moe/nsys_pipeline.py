"""nsys target: the full streaming MoE pipeline (fwd+bwd) at the 82ba5b shape,
with warmup OUTSIDE a cudaProfilerStart/Stop fence so an nsys run with
`--capture-range=cudaProfilerApi` captures only a handful of steady iterations
(no JIT/warmup bloat).

This is the comm/overlap counterpart to ncu_microbench.py: ncu gives per-GEMM
SM internals on one GPU; this gives the multi-rank picture nsys is built for —
achieved NVLink/DRAM/SM/Tensor throughput over time (gpu-metrics), dispatch ↔
compute overlap geometry, exposed comm, and per-rank/per-node imbalance. Run
the FULL pipeline (not isolated kernels) because overlap only exists across the
4 concurrent streams.

Shape defaults to the 82ba5b prod MoE (inherited from profile_pipeline globals:
H=3072, I=768, E=256, K=8, T=8192/rank). World size = however many ranks
torchrun launches (8 = 1 node intranode/NVL; 16/32 = 2/4 nodes internode/RDMA).

Launch (via scripts/srun_nsys_pipeline.sh, which sets the nsys env):
    NSYS_CAPTURE=cudaProfilerApi STREAM_EP_NVTX=1 \
    torchrun --nproc-per-node=8 --no-python scripts/nsys_wrap.sh \
        -m stream_ep.stream_moe.nsys_pipeline --n_iter 12

Each profiled iter is wrapped in an `iter_<k>` NVTX range; stream_moe_func's
own fwd_dispatch/fwd_compute/fwd_combine/moe_bwd ranges fire when
STREAM_EP_NVTX=1, so the analyzer can attribute gpu-metrics to each phase.
"""

import argparse

import torch
import torch.distributed as torch_dist

from stream_ep.stream_moe.profile_pipeline import (
    DTYPE,
    NUM_EXPERTS,
    SEQ_LEN_PER_RANK,
    TOPK,
    H,
    I,
    barrier,
    get_global_rank,
    get_world_size,
    init_distributed,
    make_skewed_topk_idx,
    make_uniform_topk_idx,
    make_buffer,
    rank_zero_print,
)
from stream_ep.stream_moe.stream_moe import make_streams, stream_moe_func


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_sms", type=int, default=None, help="StreamEP num_sms; default = Buffer auto-pick")
    p.add_argument("--seq_len", type=int, default=SEQ_LEN_PER_RANK)
    p.add_argument("--n_warmup", type=int, default=5, help="warmup iters OUTSIDE the profiler fence (JIT + caches)")
    p.add_argument("--n_iter", type=int, default=12, help="steady iters INSIDE the profiler fence")
    p.add_argument("--skew_hot_frac", type=float, default=0.0, help=">0 uses biased routing (heavy-comm regime)")
    p.add_argument("--skew_hot_weight", type=float, default=4.0)
    args = p.parse_args()

    device = init_distributed()
    rank, world_size = get_global_rank(), get_world_size()
    group = torch_dist.group.WORLD
    local_E = NUM_EXPERTS // world_size

    buffer = make_buffer(group, args.num_sms)
    rank_zero_print(
        f"[nsys_pipeline] world={world_size} num_sms={buffer.num_sms} "
        f"H={H} I={I} E={NUM_EXPERTS} K={TOPK} T={args.seq_len} local_E={local_E} "
        f"n_warmup={args.n_warmup} n_iter={args.n_iter} skew={args.skew_hot_frac}"
    )

    g = torch.Generator(device=device).manual_seed(42)
    w1_full = (torch.randn(NUM_EXPERTS, 2 * I, H, dtype=DTYPE, device=device, generator=g) * 0.02).contiguous()
    w2_full = (torch.randn(NUM_EXPERTS, H, I, dtype=DTYPE, device=device, generator=g) * 0.02).contiguous()
    w1_local = w1_full[rank * local_E : (rank + 1) * local_E].contiguous().requires_grad_(True)
    w2_local = w2_full[rank * local_E : (rank + 1) * local_E].contiguous().requires_grad_(True)

    torch.manual_seed(100 + rank)
    x = (torch.randn(args.seq_len, H, dtype=DTYPE, device=device) * 0.1).contiguous().requires_grad_(True)
    if args.skew_hot_frac > 0:
        skew_gen = torch.Generator(device=device).manual_seed(7000 + rank)
        topk_idx = make_skewed_topk_idx(
            args.seq_len, TOPK, NUM_EXPERTS, hot_frac=args.skew_hot_frac,
            hot_weight=args.skew_hot_weight, device=device, generator=skew_gen,
        )
    else:
        topk_idx = make_uniform_topk_idx(args.seq_len, TOPK, NUM_EXPERTS, rank, device)
    topk_weights = torch.softmax(
        torch.randn(args.seq_len, TOPK, dtype=torch.float32, device=device), dim=-1
    ).contiguous().requires_grad_(True)

    rank_idx = topk_idx // local_E
    is_token_in_rank = torch.zeros((args.seq_len, world_size), dtype=torch.bool, device=device)
    for r in range(world_size):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)

    streams = make_streams()

    def one_iter():
        out = stream_moe_func(
            buffer, x, topk_idx, topk_weights, is_token_in_rank,
            w1_local, w2_local, streams=streams, num_experts=NUM_EXPERTS,
        )
        out.sum().backward()

    # Warmup OUTSIDE the fence: JIT compile (CuTeDSL kernels), kernel cache,
    # allocator, comm-buffer priming. Excluded from the nsys capture.
    for _ in range(args.n_warmup):
        one_iter()
    torch.cuda.synchronize()
    barrier(group)

    # Steady region. With nsys --capture-range=cudaProfilerApi, capture begins
    # here and ends at profiler.stop(). Per-iter NVTX range so the analyzer can
    # bucket gpu-metrics samples by iteration / drop any transient first iter.
    torch.cuda.profiler.start()
    for k in range(args.n_iter):
        torch.cuda.nvtx.range_push(f"iter_{k}")
        one_iter()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    barrier(group)

    rank_zero_print("[nsys_pipeline] OK")
    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
