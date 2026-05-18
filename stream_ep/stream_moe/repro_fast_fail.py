"""Stripped-down fast-fail repro for the CDMC=8 + torch.compile dispatch hang.

Reproduces the ``moe_recv_counter == -1`` deadlock seen in the full evoscale
moe_benchmark (`logs/bench_default_stream_moe_diag.log`,
`logs/record_stream_probe_2026-05-18.log`) without the ~30 s evoscale model
build / FSDP / DTensor / optimizer overhead. A tiny module wraps
``stream_moe_func`` between two ``nn.Linear`` layers; the whole module is
compiled with ``torch.compile``; the loop runs forward + backward without an
optimizer step. The point is to A/B cheaply against debug probes in
``stream_moe.py``.

Launch:
    ./scripts/srun_1node.sh -m stream_ep.stream_moe.repro_fast_fail \\
        [--n-iter 200] [--no-compile]
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from stream_ep.stream_moe.profile_pipeline import (
    DTYPE,
    H,
    I,
    NUM_EXPERTS,
    NUM_SMS,
    SEQ_LEN_PER_RANK,
    TILE_M,
    TILE_N_A,
    TILE_N_Y,
    TOPK,
    barrier,
    get_global_rank,
    get_world_size,
    init_distributed,
    make_buffer,
    rank_zero_print,
)
from stream_ep.stream_moe.stream_moe import (
    make_streams,
    stream_moe_func,
)
import torch.distributed as torch_dist


class TinyMoEModel(nn.Module):
    def __init__(
        self,
        group,
        num_sms: int,
        ep_size: int,
        local_E: int,
        hidden: int,
        intermediate: int,
        num_experts: int,
        topk: int,
        n_layers: int,
        lazy_runtime: bool,
    ):
        super().__init__()
        self.group = group
        self.num_sms = num_sms
        self.ep_size = ep_size
        self.local_E = local_E
        self.num_experts = num_experts
        self.topk = topk
        self.n_layers = n_layers

        self.pre = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False, dtype=DTYPE) for _ in range(n_layers)]
        )
        self.post = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False, dtype=DTYPE) for _ in range(n_layers)]
        )
        self.router = nn.ModuleList(
            [nn.Linear(hidden, num_experts, bias=False, dtype=DTYPE) for _ in range(n_layers)]
        )

        # Shared expert weights across layers (keeps memory bounded; each layer
        # still gets its own stream_moe_func call with a fresh dispatch_seq).
        self.w1_local = nn.Parameter(
            torch.randn(local_E, 2 * intermediate, hidden, dtype=DTYPE) * 0.02
        )
        self.w2_local = nn.Parameter(
            torch.randn(local_E, hidden, intermediate, dtype=DTYPE) * 0.02
        )

        self.buffer = None
        self.streams = None
        if not lazy_runtime:
            self.buffer = make_buffer(group, num_sms)
            self.streams = make_streams()

    def _ensure_runtime(self, device: torch.device) -> None:
        if self.buffer is not None:
            return
        self.buffer = make_buffer(self.group, self.num_sms)
        self.streams = make_streams(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_runtime(x.device)
        h = x
        for layer_idx in range(self.n_layers):
            h_in = h
            h = self.pre[layer_idx](h)
            router_logits = self.router[layer_idx](h).float()
            topk_weights, topk_idx = router_logits.topk(self.topk, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1).contiguous()
            topk_idx = topk_idx.to(torch.int64).contiguous()

            rank_idx = topk_idx // self.local_E
            ranks = torch.arange(self.ep_size, device=x.device).view(1, 1, -1)
            is_token_in_rank = (rank_idx.unsqueeze(-1) == ranks).any(dim=1)

            out = stream_moe_func(
                self.buffer,
                h.contiguous(),
                topk_idx,
                topk_weights,
                is_token_in_rank,
                self.w1_local,
                self.w2_local,
                streams=self.streams,
                num_experts=self.num_experts,
                tile_m=TILE_M,
                tile_n_a=TILE_N_A,
                tile_n_y=TILE_N_Y,
            )
            h = self.post[layer_idx](out) + h_in
        return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_sms", type=int, default=NUM_SMS)
    p.add_argument("--seq_len", type=int, default=SEQ_LEN_PER_RANK)
    p.add_argument("--n_iter", type=int, default=200)
    p.add_argument("--n_layers", type=int, default=20,
                   help="MoE layers per forward (production default is 20).")
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument("--compile", dest="compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument(
        "--lazy-runtime", dest="lazy_runtime", action="store_true", default=True,
        help="Construct Buffer + StreamHolder inside forward() on the first call "
        "(default; mirrors StreamMoEWrapper). Triggers the iter-2 recompile that "
        "may be the source of the CDMC=8 barrier_block deadlock.",
    )
    p.add_argument("--eager-runtime", dest="lazy_runtime", action="store_false")
    args = p.parse_args()

    device = init_distributed()
    rank, world_size = get_global_rank(), get_world_size()
    group = torch_dist.group.WORLD
    local_E = NUM_EXPERTS // world_size

    rank_zero_print(
        f"[repro] world={world_size} compile={args.compile} "
        f"lazy_runtime={args.lazy_runtime} num_sms={args.num_sms} "
        f"H={H} I={I} E={NUM_EXPERTS} K={TOPK} T={args.seq_len} "
        f"n_layers={args.n_layers} n_iter={args.n_iter}"
    )

    t0 = time.time()
    model = TinyMoEModel(
        group=group,
        num_sms=args.num_sms,
        ep_size=world_size,
        local_E=local_E,
        hidden=H,
        intermediate=I,
        num_experts=NUM_EXPERTS,
        topk=TOPK,
        n_layers=args.n_layers,
        lazy_runtime=args.lazy_runtime,
    ).to(device)
    rank_zero_print(f"[repro] model build (lazy_runtime={args.lazy_runtime}): "
                    f"{time.time() - t0:.2f}s")

    if args.compile:
        model = torch.compile(model)

    torch.manual_seed(100 + rank)
    x = (
        torch.randn(args.seq_len, H, dtype=DTYPE, device=device) * 0.1
    ).contiguous().requires_grad_(True)

    barrier(group)
    t0 = time.time()
    for i in range(args.n_iter):
        out = model(x)
        out.sum().backward()
        if rank == 0 and (i + 1) % args.print_every == 0:
            print(f"[repro] iter {i + 1}/{args.n_iter}", flush=True)
    torch.cuda.synchronize()
    barrier(group)
    t = time.time() - t0
    rank_zero_print(
        f"[repro] OK — {args.n_iter} iters in {t:.2f}s "
        f"({t / args.n_iter * 1e3:.1f} ms/iter avg)"
    )

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
