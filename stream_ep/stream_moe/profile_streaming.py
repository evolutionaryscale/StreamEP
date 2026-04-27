"""Torch-profiler trace of streaming_moe_forward at the production point.

Single config (num_sms=48, topk=12, world=8). Standard torch.profiler schedule:
  wait=1 → skip 1 step (warm allocator),
  warmup=2 → 2 steps (warm JIT cache, kernel cache),
  active=5 → 5 steps recorded,
  repeat=1 → only do this once.

Trace files land in `{repo root}/profiles/` per-rank. Open with
chrome://tracing or TensorBoard.

Launch:
    torchrun --nproc_per_node=8 \
        -m evolutionaryscale.models.moe.streaming_moe.profile_streaming \
        [--out_dir /path/to/profiles]
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as torch_dist
import torch.profiler
from deep_ep import Buffer as DeepEPBuffer

from evolutionaryscale.models.moe.streaming_moe.streaming_moe import (
    streaming_moe_forward,
)
from evolutionaryscale.utils.distributed import (
    Axes,
    ParallelConfig,
    get_parallelism_group,
    init_distributed,
    parallel_context,
)

D_MODEL = 2048
INTERMEDIATE_SIZE = 384
NUM_EXPERTS = 384
SEQ_LEN_PER_RANK = 8192
DTYPE = torch.bfloat16
NUM_SMS = 48
TOPK = 12


def make_uniform_topk_idx(n_tokens, topk, num_experts, rank, device):
    base = (torch.arange(n_tokens, device=device) + rank * n_tokens) * topk
    offsets = torch.arange(topk, device=device).unsqueeze(0)
    return ((base.unsqueeze(1) + offsets) % num_experts).to(torch.int64)


def make_buffer(group, num_sms):
    DeepEPBuffer.set_num_sms(num_sms)
    hidden_bytes = D_MODEL * 2
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (
        DeepEPBuffer.get_dispatch_config(group.size()),
        DeepEPBuffer.get_combine_config(group.size()),
    ):
        nvl_bytes = max(
            cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()), nvl_bytes
        )
        rdma_bytes = max(
            cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()), rdma_bytes
        )
    return DeepEPBuffer(
        group, nvl_bytes, rdma_bytes, num_qps_per_rank=DeepEPBuffer.num_sms
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(
            os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "..", "..", "profiles"
                )
            )
        ),
    )
    args = p.parse_args()

    init_distributed()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 8 and rank == 0:
        print(f"WARNING: expected 8 ranks, got {world_size}")

    pc = ParallelConfig(dp_replicate_degree=1, ep_degree=world_size)
    with parallel_context(parallel_config=pc):
        ep_group = get_parallelism_group(Axes.EP)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        local_E = NUM_EXPERTS // world_size

        buffer = make_buffer(ep_group, NUM_SMS)
        w1 = torch.randn(
            local_E, 2 * INTERMEDIATE_SIZE, D_MODEL, dtype=DTYPE, device=device
        ).mul_(0.02)
        w2 = torch.randn(
            local_E, D_MODEL, INTERMEDIATE_SIZE, dtype=DTYPE, device=device
        ).mul_(0.02)
        w1_q = w1.permute(0, 2, 1).contiguous()
        w2_q = w2.permute(0, 2, 1).contiguous()
        x = torch.randn(SEQ_LEN_PER_RANK, D_MODEL, dtype=DTYPE, device=device) * 0.1
        topk_idx = make_uniform_topk_idx(
            SEQ_LEN_PER_RANK, TOPK, NUM_EXPERTS, rank, device
        )
        topk_weights = torch.full(
            (SEQ_LEN_PER_RANK, TOPK), 1.0 / TOPK, dtype=torch.float32, device=device
        )

        compute_done_pool = torch.zeros(
            NUM_SMS // 2, 8, dtype=torch.int64, device=device
        )
        compute_tile_counter_pool = torch.zeros(8, dtype=torch.int32, device=device)

        os.makedirs(args.out_dir, exist_ok=True)
        if rank == 0:
            print(f"writing traces to {args.out_dir}/")

        # Allow jit / kernel-cache warmups before profiler arms — keeps the
        # first profiled step from being all compile.
        for _ in range(5):
            streaming_moe_forward(
                x,
                w1_q,
                w2_q,
                topk_idx,
                topk_weights,
                buffer,
                NUM_EXPERTS,
                compute_done=compute_done_pool,
                compute_tile_counter=compute_tile_counter_pool,
            )
        torch.cuda.synchronize()
        torch_dist.barrier(group=ep_group)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.out_dir),
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
        ) as prof:
            for _ in range(1 + 2 + 5):
                streaming_moe_forward(
                    x,
                    w1_q,
                    w2_q,
                    topk_idx,
                    topk_weights,
                    buffer,
                    NUM_EXPERTS,
                    compute_done=compute_done_pool,
                    compute_tile_counter=compute_tile_counter_pool,
                )
                prof.step()

        torch.cuda.synchronize()
        torch_dist.barrier(group=ep_group)
        if rank == 0:
            print("done; trace files (one per rank) in:", args.out_dir)


if __name__ == "__main__":
    main()
