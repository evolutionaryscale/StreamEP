"""Torch-profiler trace of the streaming pipeline up to kernel Y
(dispatch → kernel A → kernel Y), the slice that's fully implemented today.

Validates the streaming property visually: kernel A's CTAs should overlap
with the tail of dispatch (Pass 2 firing per substream), and kernel Y's
CTAs should overlap with the tail of kernel A (per-tile a_ready firing).

What you should see in the resulting chrome trace
-------------------------------------------------
- `dispatch_stream`: streaming_dispatch_metadata → host poll → tile_arrays_init
  → dispatch main kernel. The dispatch kernel's receiver does Pass 1
  (per-batch slot allocation + data copy into pool) and Pass 2
  (substream-end expert-major tile_ready firing) inline.
- `compute_a_stream`: streaming_moe_a launches early, overlapped with the
  tail of dispatch. Its CTAs spin on tile_ready[tile_id] then process.
  Per-tile a_ready[tile_id] release-stores happen at the end of each tile.
- `compute_y_stream`: streaming_moe_y launches early, overlapped with the
  tail of kernel A. Its CTAs spin on a_ready[tile_id] then GEMM + per-warp
  coalesced atomic-scatter into o[T_recv, H], finishing with per-token
  bookkeeping (per_token_remaining decrement + compute_done release).

Launch
------
    torchrun --nproc_per_node=2 \\
        -m evolutionaryscale.models.moe.streaming_moe.profile_pipeline \\
        [--out_dir /path/to/profiles]

Trace files land per-rank in `{repo_root}/profiles/`. Open with
chrome://tracing or TensorBoard.
"""

import argparse
import os

import torch
import torch.distributed as torch_dist
import torch.profiler
from stream_ep import Buffer as StreamEPBuffer

from evolutionaryscale.models.moe.streaming_moe.streaming_moe import (
    make_streams,
    stream_moe_func,
)
from evolutionaryscale.utils.distributed import (
    barrier,
    get_global_rank,
    get_world_size,
    init_distributed,
    rank_zero_print,
)

H = 2048
I = 2048
NUM_EXPERTS = 64
SEQ_LEN_PER_RANK = 8192
TOPK = 4
DTYPE = torch.bfloat16
NUM_SMS = 80  # See bench_pipeline.py for sweep justification.
TILE_M = 128
TILE_N_A = 256
TILE_N_Y = 128


def make_uniform_topk_idx(n_tokens, topk, num_experts, rank, device):
    base = (torch.arange(n_tokens, device=device) + rank * n_tokens) * topk
    offsets = torch.arange(topk, device=device).unsqueeze(0)
    return ((base.unsqueeze(1) + offsets) % num_experts).to(torch.int64)


def make_skewed_topk_idx(
    n_tokens, topk, num_experts, hot_frac, hot_weight, device, generator
):
    """Biased multinomial routing: the first ``hot_frac`` of experts get
    ``hot_weight`` × the per-token sampling weight of the rest, sampled without
    replacement so each token still gets ``topk`` distinct experts. Used to
    exercise the heavy-comm regime — under hot_frac=0.25, hot_weight=4.0 the
    expected token share for the hot 25% of experts is ~57% (vs the 25%
    uniform), which roughly doubles dispatch / combine traffic on the
    hot ranks and surfaces the streaming-overlap behavior that's invisible
    at uniform routing.
    """
    n_hot = max(1, int(num_experts * hot_frac))
    weights = torch.ones(num_experts, device=device)
    weights[:n_hot] *= hot_weight
    probs = weights.expand(n_tokens, -1).contiguous()
    return torch.multinomial(probs, topk, replacement=False, generator=generator).to(
        torch.int64
    )


def make_buffer(group, num_sms):
    StreamEPBuffer.set_num_sms(num_sms)
    hidden_bytes = H * 2
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (
        StreamEPBuffer.get_dispatch_config(group.size()),
        StreamEPBuffer.get_combine_config(group.size()),
    ):
        nvl_bytes = max(
            cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()), nvl_bytes
        )
        rdma_bytes = max(
            cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()), rdma_bytes
        )
    return StreamEPBuffer(group, nvl_bytes, rdma_bytes)


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
    p.add_argument("--num_sms_a", type=int, default=None)
    p.add_argument("--num_sms_y", type=int, default=None)
    p.add_argument("--tile_m", type=int, default=TILE_M)
    p.add_argument("--tile_n_a", type=int, default=TILE_N_A)
    p.add_argument("--tile_n_y", type=int, default=TILE_N_Y)
    p.add_argument(
        "--skew_hot_frac",
        type=float,
        default=0.0,
        help="If >0, use biased routing instead of uniform: the first hot_frac "
        "of experts get hot_weight× sampling weight. 0 (default) = uniform.",
    )
    p.add_argument("--skew_hot_weight", type=float, default=4.0)
    args = p.parse_args()

    device = init_distributed()
    rank, world_size = get_global_rank(), get_world_size()
    group = torch_dist.group.WORLD
    local_E = NUM_EXPERTS // world_size

    buffer = make_buffer(group, args.num_sms)

    # Replicated W1 / W2 across ranks (each rank slices its E_local share).
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
    if args.skew_hot_frac > 0:
        skew_gen = torch.Generator(device=device).manual_seed(7000 + rank)
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

    os.makedirs(args.out_dir, exist_ok=True)
    if rank == 0:
        print(f"writing traces to {args.out_dir}/", flush=True)
        print(
            f"config: world={world_size} num_sms={args.num_sms} "
            f"H={H} I={I} E={NUM_EXPERTS} K={TOPK} T={args.seq_len} "
            f"tile_m={args.tile_m} tile_n_a={args.tile_n_a} tile_n_y={args.tile_n_y}",
            flush=True,
        )

    # Warm: dispatch + kernel A + kernel Y + combine JIT, kernel cache, allocator.
    # Includes the bwd path so its kernels JIT during warmup too.
    for warm_seq in range(1, 6):
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
            dispatch_seq=warm_seq,
            tile_m=args.tile_m,
            tile_n_a=args.tile_n_a,
            tile_n_y=args.tile_n_y,
            num_sms_a=args.num_sms_a,
            num_sms_y=args.num_sms_y,
        )
        out.sum().backward()
    torch.cuda.synchronize()
    barrier(group)

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
            with torch.profiler.record_function(f"step_{step}_fwd"):
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
                    dispatch_seq=seq + step,
                    tile_m=args.tile_m,
                    tile_n_a=args.tile_n_a,
                    tile_n_y=args.tile_n_y,
                    num_sms_a=args.num_sms_a,
                    num_sms_y=args.num_sms_y,
                )
            with torch.profiler.record_function(f"step_{step}_bwd"):
                out.sum().backward()
            prof.step()

    torch.cuda.synchronize()
    barrier(group)
    rank_zero_print("done; trace files (one per rank) in:", args.out_dir)
    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
