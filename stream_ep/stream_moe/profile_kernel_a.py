"""Torch-profiler trace of DeepEP dispatch → streaming kernel A end-to-end (pool layout).

Validates the streaming property visually: kernel A's CTAs should be on the
GPU timeline concurrently with DeepEP's dispatch tail (Pass 2 firing per
substream), not strictly after.

What you should see in the resulting chrome trace
-------------------------------------------------
- `comm_stream`: streaming_dispatch_metadata → (single host poll) →
  tile_arrays_init → dispatch main kernel. The dispatch kernel's receiver
  block does Pass 1 (per-batch slot allocation + data copy into pool) and
  Pass 2 (substream-end expert-major tile_ready firing) inline.
- `compute_a_stream`: streaming_moe_a (kernel A) launches early (overlapped
  with the tail of dispatch). Its CTAs spin on tile_ready[tile_id] until
  dispatch's Pass 2 fires that tile, then process it. The first tiles should
  start computing before dispatch finishes.

Launch
------
    torchrun --nproc_per_node=2 \
        -m evolutionaryscale.models.moe.streaming_moe.profile_kernel_a \
        [--out_dir /path/to/profiles]

The profile is taken with the standard torch.profiler schedule
(wait=1, warmup=2, active=5, repeat=1). Trace files land per-rank in
`{repo_root}/profiles/`. Open with chrome://tracing or TensorBoard.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as torch_dist
import torch.profiler
from deep_ep import Buffer as DeepEPBuffer
from quack.moe_streaming_sm90 import streaming_moe_a

# Compact-but-meaningful config. Smaller than production but large enough that
# kernel A has nontrivial compute time and total_tiles > num_persistent_ctas
# so the queue-pull dispatch is exercised.
H = 2048
I = 2048
NUM_EXPERTS = 64
SEQ_LEN_PER_RANK = 8192
TOPK = 4
DTYPE = torch.bfloat16
NUM_SMS = 24
TILE_M = 128
TILE_N = 256


def make_uniform_topk_idx(n_tokens, topk, num_experts, rank, device):
    base = (torch.arange(n_tokens, device=device) + rank * n_tokens) * topk
    offsets = torch.arange(topk, device=device).unsqueeze(0)
    return ((base.unsqueeze(1) + offsets) % num_experts).to(torch.int64)


def make_buffer(group, num_sms):
    DeepEPBuffer.set_num_sms(num_sms)
    hidden_bytes = H * 2
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


def one_step(
    buffer,
    x,
    topk_idx,
    topk_weights,
    is_token_in_rank,
    w1_local,
    compute_a_stream,
    dispatch_seq,
):
    """A single dispatch + kernel A iteration. Returns the postact_a tensor and
    StreamingHandle so the caller can drop them after the profiler step.
    """
    pool, handle, _event = buffer.dispatch(
        x,
        topk_idx,
        topk_weights,
        is_token_in_rank,
        NUM_EXPERTS,
        tile_m=TILE_M,
        dispatch_seq=dispatch_seq,
    )

    total_tiles = handle.total_tiles
    postact_a = torch.empty(total_tiles, TILE_M, I, dtype=DTYPE, device=pool.device)
    consumer_head = torch.zeros(1, dtype=torch.int32, device=pool.device)

    # Tensors allocated on the comm/default stream are about to be read+written
    # on compute_a_stream. record_stream tells the caching allocator not to
    # recycle them while kernel A is still in flight.
    for t in (
        pool,
        postact_a,
        consumer_head,
        handle.tile_id_to_expert,
        handle.expert_pool_block_offset,
        handle.tile_ready,
    ):
        t.record_stream(compute_a_stream)

    # Kernel A on its own stream so it overlaps with dispatch's tail.
    with torch.cuda.stream(compute_a_stream):
        streaming_moe_a(
            pool,
            w1_local,
            postact_a,
            handle.tile_id_to_expert,
            handle.expert_pool_block_offset,
            handle.tile_ready,
            consumer_head,
            dispatch_seq=handle.dispatch_seq,
            tile_m=TILE_M,
            tile_n=TILE_N,
        )
    return postact_a, handle


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
    p.add_argument("--num_sms", type=int, default=NUM_SMS)
    p.add_argument("--seq_len", type=int, default=SEQ_LEN_PER_RANK)
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    torch_dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = torch_dist.group.WORLD

    device = torch.device(f"cuda:{local_rank}")
    local_E = NUM_EXPERTS // world_size

    buffer = make_buffer(group, args.num_sms)

    # Replicated W1 across ranks (each rank slices its E_local share).
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

    os.makedirs(args.out_dir, exist_ok=True)
    if rank == 0:
        print(f"writing traces to {args.out_dir}/", flush=True)
        print(
            f"config: world={world_size} num_sms={args.num_sms} "
            f"H={H} I={I} E={NUM_EXPERTS} K={TOPK} T={args.seq_len} "
            f"tile_m={TILE_M} tile_n={TILE_N}",
            flush=True,
        )

    # Warm: dispatch + kernel A JIT, kernel cache, allocator.
    for warm_seq in range(1, 6):
        _post, _handle = one_step(
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

    schedule = torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.out_dir),
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        seq = 100  # bumped past warmup seqs so it's clearly distinct in the trace
        for step in range(1 + 2 + 5):
            with torch.profiler.record_function(f"step_{step}_dispatch+kernelA"):
                _post, _handle = one_step(
                    buffer,
                    x,
                    topk_idx,
                    topk_weights,
                    is_token_in_rank,
                    w1_local,
                    compute_a_stream,
                    dispatch_seq=seq + step,
                )
            prof.step()

    torch.cuda.synchronize()
    torch_dist.barrier(group=group)
    if rank == 0:
        print("done; trace files (one per rank) in:", args.out_dir, flush=True)
    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
