"""Repetition stress repro for the internode dispatch_grads hang.

The single-shot ``test_dispatch_grads_internode`` passes, but the full
benchmark hangs in the BACKWARD internode ``dispatch_grads`` path at depth
(``StreamEP dispatch_grads NVL receiver timeout (prefix)`` / ``forwarder
timeout``), only after ~100+ back-to-back dispatch_grads calls
(N layers x several steps). `profile_pipeline.py`'s docstring notes the
collectives need "natural per-iter slack to satisfy the single-slot
``rdma_channel_meta`` protocol, which the full fwd+bwd pipeline provides but a
rapid-fire bench loop does not" — so the hypothesis is that many rapid
back-to-back dispatches overwhelm the single-slot meta protocol.

This test reproduces that **rapid-fire** regime with PURE COMM (no compute
kernels, so no JIT and the dispatch_grads calls are maximally back-to-back —
even less slack than the real pipeline, which only helps trigger the race).
Per "step" it mirrors the benchmark's per-training-step comm structure:

  * issue ``num_layers`` forward ``dispatch`` calls, KEEPING all handles alive
    (as autograd keeps every layer's saved state until backward);
  * then issue ``num_layers`` ``dispatch_grads`` calls in REVERSE order
    (as autograd runs layer backwards in reverse) — back-to-back, no slack.

Progress is printed per (step, layer) and flushed, so when it hangs the log
pinpoints the exact step/layer/op that stalled. A clean run prints PASS.

Driver (4 nodes / world=32 — internode is where the bug lives):
    ./scripts/srun_4node.sh StreamEP/tests/test_dispatch_grads_stress.py \
        [--num_layers 44] [--num_steps 8] [--hidden 3072] [--num_experts 256] \
        [--topk 8] [--num_tokens 8192] [--tile_m 128]

Defaults match the 82B shape / depth that hung (config=82ba5b, dp_shard=32).
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from stream_ep import Buffer

from utils import cleanup_dist


def random_routing(n_tokens, topk, num_experts, num_local_experts, world_size, device, gen):
    """K random distinct experts per token, RE-RANDOMIZED each call (gen advances).

    Matches benchmark.py's force_uniform_routing (`logits*0 + randn*1e-3`, then
    topk) — the routing varies every forward, so num_recv / total_tiles /
    per-(channel,src,expert) counts vary every dispatch. That per-dispatch size
    variation is the suspected trigger the earlier FIXED-routing repro missed.
    Returns (topk_idx, topk_weights, is_token_in_rank)."""
    logits = torch.randn(n_tokens, num_experts, device=device, generator=gen)
    topk_idx = torch.topk(logits, topk, dim=-1).indices.to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(n_tokens, topk, dtype=torch.float32, device=device, generator=gen),
        dim=-1).contiguous()
    rank_idx = topk_idx // num_local_experts
    is_token_in_rank = torch.zeros((n_tokens, world_size), dtype=torch.bool, device=device)
    for r in range(world_size):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)
    return topk_idx.contiguous(), topk_weights, is_token_in_rank


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_layers", type=int, default=44,
                   help="forward dispatches kept per step before the dispatch_grads burst")
    p.add_argument("--num_steps", type=int, default=8)
    p.add_argument("--hidden", type=int, default=3072)       # 82ba5b d_model
    p.add_argument("--num_experts", type=int, default=256)   # 82ba5b
    p.add_argument("--topk", type=int, default=8)            # 82ba5b
    p.add_argument("--num_tokens", type=int, default=8192)   # per-rank T
    p.add_argument("--tile_m", type=int, default=128)
    p.add_argument("--num_sms", type=int, default=None)
    p.add_argument("--slack", action="store_true",
                   help="Insert a cross-rank barrier after each dispatch_grads. "
                        "Tests the single-slot rdma_channel_meta race: if --slack "
                        "makes the hang vanish, the bug is iter N+1's remote meta "
                        "put overwriting the slot before iter N's forwarder reads it.")
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    assert world_size % 8 == 0 and world_size > 8, (
        f"internode repro needs world_size > 8 and % 8 == 0; got {world_size}")
    assert args.num_experts % world_size == 0, (
        f"num_experts {args.num_experts} must divide world_size {world_size}")

    if args.num_sms is not None:
        Buffer.set_num_sms(args.num_sms)

    H, E, K, T = args.hidden, args.num_experts, args.topk, args.num_tokens
    num_local_experts = E // world_size

    hidden_bytes = H * 2
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (Buffer.get_dispatch_config(world_size),
                Buffer.get_combine_config(world_size)):
        nvl_bytes = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, world_size), nvl_bytes)
        rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes)
    buf = Buffer(group, nvl_bytes, rdma_bytes)

    # x / dL_dy fixed (values irrelevant — we stress comm). Routing is
    # RE-RANDOMIZED per dispatch (see random_routing) to match the benchmark.
    torch.manual_seed(100 + rank)
    x = (torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    dL_dy = (torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    route_gen = torch.Generator(device=device).manual_seed(7000 + rank)

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f"[repro] world={world_size} H={H} E={E} K={K} T={T} "
        f"num_layers={args.num_layers} num_steps={args.num_steps} "
        f"local_experts={num_local_experts}")

    for step in range(args.num_steps):
        # --- forward: num_layers dispatches, keep all handles alive ---
        handles = []
        for layer in range(args.num_layers):
            topk_idx, topk_weights, is_token_in_rank = random_routing(
                T, K, E, num_local_experts, world_size, device, route_gen)
            log(f"[repro] step {step} fwd-dispatch layer {layer}")
            _pool, handle, _ev = buf.dispatch(
                x, topk_idx, topk_weights, is_token_in_rank, E, tile_m=args.tile_m)
            handles.append(handle)
        # --- backward: dispatch_grads in reverse, back-to-back (no slack) ---
        for layer, handle in enumerate(reversed(handles)):
            log(f"[repro] step {step} bwd-dispatch_grads layer {args.num_layers - 1 - layer} "
                f"(seq={handle.dispatch_seq})")
            _dl_do_pool, _cnt, _ev = buf.dispatch_grads(
                handle, dL_dy, dispatch_seq=handle.dispatch_seq)
            if args.slack:
                torch.cuda.synchronize()
                dist.barrier(device_ids=[torch.cuda.current_device()])
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])
        log(f"[repro] step {step} COMPLETE")

    if rank == 0:
        print(f"PASS: {args.num_steps} steps x {args.num_layers} layers of "
              f"dispatch+dispatch_grads completed without hang "
              f"(world={world_size}, H={H}, E={E}, K={K}, T={T})", flush=True)

    cleanup_dist()


if __name__ == "__main__":
    main()
