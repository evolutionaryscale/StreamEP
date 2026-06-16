"""Full-pipeline depth stress repro for the internode dispatch_grads hang.

The pure-comm repro (``test_dispatch_grads_stress.py``) does NOT hang — bare
``dispatch``+``dispatch_grads`` rapid-fire at production shape/depth is clean.
So the trigger is in the *interaction* the pure-comm loop omits: the compute
kernels between collectives, combine/combine_grads, and the 2-stream overlap +
launch gates. This repro exercises the FULL ``stream_moe_func`` pipeline
(dispatch -> kernel_a -> kernel_y -> combine ; bwd: dispatch_grads ->
kernel_y_bwd -> kernel_a_bwd -> combine_grads, on the two caller streams),
chained ``num_layers`` deep then a single ``backward()`` — i.e. the benchmark's
MoE stack with attention/FSDP/optimizer/compile stripped away.

The benchmark hangs at config=82ba5b, n_layers>=32, dp_shard=32 (4-node) in the
backward dispatch_grads (``NVL receiver timeout (prefix)`` / ``forwarder
timeout``), around warmup step ~4. This repro reproduces that regime: each
"step" chains ``num_layers`` stream_moe_func calls (so backward runs
``num_layers`` dispatch_grads back-to-back, exactly like a deep model's bwd),
repeated ``num_steps`` times.

Progress is printed + flushed per (step, fwd-layer) and around backward, so a
hang pinpoints the stalling step/phase. Clean run prints PASS.

Driver (4 nodes / world=32):
    ./scripts/srun_4node.sh StreamEP/tests/test_pipeline_depth_stress.py \
        [--num_layers 44] [--num_steps 6]

Defaults = the 82B shape/depth that hung. preact ckpt OFF (default).
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from stream_ep import Buffer
from stream_ep.stream_moe.stream_moe import make_streams, stream_moe_func

from utils import cleanup_dist


def uniform_topk_idx(n_tokens, topk, num_experts, rank, device):
    base = (torch.arange(n_tokens, device=device) + rank * n_tokens) * topk
    offsets = torch.arange(topk, device=device).unsqueeze(0)
    return ((base.unsqueeze(1) + offsets) % num_experts).to(torch.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_layers", type=int, default=44)   # 82ba5b depth
    p.add_argument("--num_steps", type=int, default=6)     # bench hung ~step 4
    p.add_argument("--hidden", type=int, default=3072)     # 82ba5b d_model
    p.add_argument("--intermediate", type=int, default=768)  # 82ba5b moe_intermediate
    p.add_argument("--num_experts", type=int, default=256)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num_tokens", type=int, default=8192)
    p.add_argument("--rand_routing", action="store_true",
                   help="Draw a FRESH RANDOM topk_idx per layer via torch.rand "
                        "(default: fixed uniform round-robin). Random routing = "
                        "top-K of uniform scores (K distinct experts/token), "
                        "seeded per rank, so per-rank recv volumes are uneven "
                        "and vary layer-to-layer — the realistic, asymmetric "
                        "stress (vs the light, balanced round-robin). Drive "
                        "memory toward OOM with --num_layers / --num_tokens; "
                        "no artificial ballast or side-collectives here.")
    p.add_argument("--activation_checkpoint", action="store_true",
                   help="Pass activation_checkpoint=True to stream_moe_func "
                        "(matches the bench's model.moe_activation_checkpoint). "
                        "Drops preact_a from fwd retention and RECOMPUTES it in "
                        "bwd — shifts the memory peak from forward into backward "
                        "(where the comm forwarders spin), so the near-OOM "
                        "reclaim storm lands in dispatch_grads, not forward.")
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    assert world_size % 8 == 0 and world_size > 8, (
        f"internode repro needs world_size > 8 and % 8 == 0; got {world_size}")
    assert args.num_experts % world_size == 0

    H, I, E, K, T = (args.hidden, args.intermediate, args.num_experts,
                     args.topk, args.num_tokens)
    E_local = E // world_size

    hidden_bytes = H * 2
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (Buffer.get_dispatch_config(world_size),
                Buffer.get_combine_config(world_size)):
        nvl_bytes = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, world_size), nvl_bytes)
        rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes)
    buf = Buffer(group, nvl_bytes, rdma_bytes)
    streams = make_streams()

    g = torch.Generator(device=device).manual_seed(42 + rank)
    w1 = (torch.randn(E_local, 2 * I, H, dtype=torch.bfloat16, device=device, generator=g)
          * 0.02).contiguous().requires_grad_(True)
    w2 = (torch.randn(E_local, H, I, dtype=torch.bfloat16, device=device, generator=g)
          * 0.02).contiguous().requires_grad_(True)

    torch.manual_seed(100 + rank)
    x0 = (torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    topk_weights = torch.softmax(
        torch.randn(T, K, dtype=torch.float32, device=device), dim=-1).contiguous()
    num_local_experts = E // world_size

    def is_token_in_rank_of(idx):
        rank_idx = idx // num_local_experts
        tir = torch.zeros((T, world_size), dtype=torch.bool, device=device)
        for r in range(world_size):
            tir[:, r] = (rank_idx == r).any(dim=-1)
        return tir

    # Routing. Default: fixed uniform round-robin, precomputed once. With
    # --rand_routing: a fresh random draw PER LAYER (top-K of torch.rand scores
    # = K distinct experts/token), seeded per rank so recv volumes are uneven
    # across ranks and vary layer-to-layer.
    route_gen = torch.Generator(device=device).manual_seed(7000 + rank)

    def draw_routing():
        idx = (
            torch.rand(T, E, device=device, generator=route_gen)
            .topk(K, dim=-1).indices.to(torch.int64)
            if args.rand_routing
            else uniform_topk_idx(T, K, E, rank, device)
        )
        return idx, is_token_in_rank_of(idx)

    fixed_routing = None if args.rand_routing else draw_routing()

    def log(m):
        if rank == 0:
            print(m, flush=True)

    GiB = 1 << 30

    def meminfo():  # observability only — cheap driver/bookkeeping queries, no sync
        free_b, total_b = torch.cuda.mem_get_info()
        return (f"alloc={torch.cuda.memory_allocated() / GiB:.1f} "
                f"reserved={torch.cuda.memory_reserved() / GiB:.1f} "
                f"free={free_b / GiB:.1f}/{total_b / GiB:.1f} GB")

    log(f"[repro] world={world_size} H={H} I={I} E={E} K={K} T={T} "
        f"num_layers={args.num_layers} num_steps={args.num_steps} E_local={E_local}")
    log(f"[repro] PYTORCH_CUDA_ALLOC_CONF="
        f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')} "
        f"rand_routing={args.rand_routing} activation_checkpoint={args.activation_checkpoint}")

    for step in range(args.num_steps):
        if w1.grad is not None:
            w1.grad = None
        if w2.grad is not None:
            w2.grad = None
        h = x0.clone().requires_grad_(True)
        for layer in range(args.num_layers):
            log(f"[repro] step {step} fwd layer {layer}")
            layer_topk_idx, layer_is_token_in_rank = (
                draw_routing() if args.rand_routing else fixed_routing)
            h = stream_moe_func(
                buf, h, layer_topk_idx, topk_weights, layer_is_token_in_rank,
                w1, w2, streams=streams, num_experts=E,
                activation_checkpoint=args.activation_checkpoint)
        log(f"[repro] step {step} end-fwd | {meminfo()} | backward starting "
            f"({args.num_layers} dispatch_grads back-to-back)")
        h.sum().backward()
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])
        log(f"[repro] step {step} COMPLETE | {meminfo()}")

    if rank == 0:
        print(f"PASS: {args.num_steps} steps x {args.num_layers}-layer "
              f"stream_moe_func fwd+bwd completed without hang "
              f"(world={world_size}, H={H}, I={I}, E={E}, K={K}, T={T})", flush=True)

    cleanup_dist()


if __name__ == "__main__":
    main()
