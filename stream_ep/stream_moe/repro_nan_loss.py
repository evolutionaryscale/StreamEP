"""Loss-based NaN repro for stream_moe.

Builds a tiny classification model around stream_moe layers and runs
SGD with cross-entropy loss against random integer targets. Reports
the first iter where any rank sees NaN/Inf in the loss or in any
gradient, plus per-iter loss for rank 0.

Why a separate repro: ``repro_fast_fail.py`` confirms the dispatch /
combine pipeline doesn't hang at 4-node, but it uses ``out.sum().backward()``
which is degenerate enough to never surface numerical issues — gradients
are trivially uniform and small. ``moe_benchmark`` at 4-node observed
NaN loss from step 2 onward with the streaming-MoE backend (both with
default CDMC=8 and CDMC=1). This script narrows that down to the
stream-MoE compute pipeline + a real loss + an optimizer step, with
``n_layers`` and ``n_iter`` exposed so we can sweep whether NaN-onset
depends on depth or iteration count.

Launch:
    ./scripts/srun_4node.sh -m stream_ep.stream_moe.repro_nan_loss \\
        [--n_iter N] [--n_layers L] [--lr LR] [--vocab V] [--no-compile]
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as torch_dist

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


class TinyMoEClassifier(nn.Module):
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
        vocab_size: int,
    ):
        super().__init__()
        self.group = group
        self.num_sms = num_sms
        self.ep_size = ep_size
        self.local_E = local_E
        self.num_experts = num_experts
        self.topk = topk
        self.n_layers = n_layers

        # Same residual block shape as repro_fast_fail (pre Linear + MoE +
        # post Linear), repeated n_layers times. Adds a classification head
        # at the end mapping H → vocab so cross-entropy has somewhere to land.
        self.pre = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False, dtype=DTYPE) for _ in range(n_layers)]
        )
        self.post = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False, dtype=DTYPE) for _ in range(n_layers)]
        )
        self.router = nn.ModuleList(
            [nn.Linear(hidden, num_experts, bias=False, dtype=DTYPE) for _ in range(n_layers)]
        )
        self.head = nn.Linear(hidden, vocab_size, bias=False, dtype=DTYPE)

        # Shared expert weights across layers (matches repro_fast_fail);
        # each layer still gets its own stream_moe_func call.
        self.w1_local = nn.Parameter(
            torch.randn(local_E, 2 * intermediate, hidden, dtype=DTYPE) * 0.02
        )
        self.w2_local = nn.Parameter(
            torch.randn(local_E, hidden, intermediate, dtype=DTYPE) * 0.02
        )

        self.buffer = None
        self.streams = None

    def _ensure_runtime(self, device: torch.device) -> None:
        if self.buffer is not None:
            return
        self.buffer = make_buffer(self.group, self.num_sms)
        self.streams = make_streams(device=device)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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

        logits = self.head(h).float()  # [T, vocab], compute loss in fp32
        loss = F.cross_entropy(logits, labels)
        return loss


def _any_nan_grad(model: nn.Module) -> tuple[bool, str]:
    """Return (has_nan, first_param_name_with_nan)."""
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return True, name
    return False, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_sms", type=int, default=NUM_SMS)
    p.add_argument("--seq_len", type=int, default=SEQ_LEN_PER_RANK)
    p.add_argument("--n_iter", type=int, default=50)
    p.add_argument("--n_layers", type=int, default=20,
                   help="MoE layers per forward (production default is 20).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument("--compile", dest="compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--check-grads", dest="check_grads", action="store_true", default=True,
                   help="After each backward, scan params for NaN/Inf grads.")
    p.add_argument("--no-check-grads", dest="check_grads", action="store_false")
    args = p.parse_args()

    device = init_distributed()
    rank, world_size = get_global_rank(), get_world_size()
    group = torch_dist.group.WORLD
    local_E = NUM_EXPERTS // world_size

    rank_zero_print(
        f"[nan-repro] world={world_size} compile={args.compile} "
        f"num_sms={args.num_sms} H={H} I={I} E={NUM_EXPERTS} K={TOPK} "
        f"T={args.seq_len} n_layers={args.n_layers} n_iter={args.n_iter} "
        f"lr={args.lr} vocab={args.vocab}"
    )

    torch.manual_seed(100 + rank)

    t0 = time.time()
    model = TinyMoEClassifier(
        group=group,
        num_sms=args.num_sms,
        ep_size=world_size,
        local_E=local_E,
        hidden=H,
        intermediate=I,
        num_experts=NUM_EXPERTS,
        topk=TOPK,
        n_layers=args.n_layers,
        vocab_size=args.vocab,
    ).to(device)
    rank_zero_print(f"[nan-repro] model build: {time.time() - t0:.2f}s")

    if args.compile:
        model = torch.compile(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    x = (torch.randn(args.seq_len, H, dtype=DTYPE, device=device) * 0.1).contiguous()
    x.requires_grad_(True)

    first_nan_loss_iter = -1
    first_nan_grad_iter = -1

    barrier(group)
    t0 = time.time()
    for i in range(args.n_iter):
        labels = torch.randint(0, args.vocab, (args.seq_len,), device=device)

        optimizer.zero_grad(set_to_none=True)
        loss = model(x, labels)

        loss_is_nan = not torch.isfinite(loss).item()
        if loss_is_nan and first_nan_loss_iter < 0:
            first_nan_loss_iter = i

        loss.backward()

        nan_grad = False
        nan_grad_name = ""
        if args.check_grads:
            nan_grad, nan_grad_name = _any_nan_grad(
                model._orig_mod if hasattr(model, "_orig_mod") else model
            )
            if nan_grad and first_nan_grad_iter < 0:
                first_nan_grad_iter = i

        optimizer.step()

        if rank == 0 and (i + 1) % args.print_every == 0:
            tag = ""
            if loss_is_nan:
                tag += " [LOSS_NAN]"
            if nan_grad:
                tag += f" [GRAD_NAN: {nan_grad_name}]"
            print(
                f"[nan-repro] iter {i + 1}/{args.n_iter} "
                f"loss={loss.item():.4f}{tag}",
                flush=True,
            )

    torch.cuda.synchronize()
    barrier(group)
    t = time.time() - t0
    rank_zero_print(
        f"[nan-repro] DONE — {args.n_iter} iters in {t:.2f}s "
        f"({t / args.n_iter * 1e3:.1f} ms/iter avg)"
    )
    rank_zero_print(
        f"[nan-repro] first_nan_loss_iter={first_nan_loss_iter} "
        f"first_nan_grad_iter={first_nan_grad_iter}"
    )

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
