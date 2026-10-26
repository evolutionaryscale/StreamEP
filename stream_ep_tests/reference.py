"""Torch references for the distributed tests, built from every rank's inputs.

Inputs are gathered rank-major, so global token ``g`` is row ``g`` of the
concatenation of every rank's tokens. Dispatch lays a rank's received tokens
out in that same order, so the expected receive layout is a filter of the
global token list.
"""

import itertools
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F

from stream_ep.stream_moe.expert_dropout import keep_scale_ref
from stream_ep_tests.routing import Routing


def all_gather_rows(t: torch.Tensor, group) -> tuple[torch.Tensor, list[int]]:
    """Concatenate every rank's ``t`` along dim 0 (row counts may differ).
    Returns the concatenation and the row offset of each rank, plus the total."""
    world = group.size()
    count = torch.tensor([t.shape[0]], device=t.device)
    counts = [torch.empty_like(count) for _ in range(world)]
    dist.all_gather(counts, count, group=group)
    counts = [int(c) for c in counts]
    padded = t.new_zeros((max(counts), *t.shape[1:]))
    padded[: t.shape[0]] = t
    chunks = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(chunks, padded, group=group)
    offsets = [sum(counts[:r]) for r in range(world + 1)]
    return torch.cat([c[:n] for c, n in zip(chunks, counts, strict=True)]), offsets


@dataclass(frozen=True)
class Global:
    """Every rank's tokens and routing, concatenated rank-major."""

    x: torch.Tensor  # [N, H]
    topk_idx: torch.Tensor  # [N, K] int64, -1 = dropped slot
    topk_weights: torch.Tensor  # [N, K] fp32
    offsets: list[int]  # rank r owns global rows [offsets[r], offsets[r + 1])
    num_experts: int
    group: dist.ProcessGroup

    @classmethod
    def gather(cls, x: torch.Tensor, routing: Routing, group) -> "Global":
        x_all, offsets = all_gather_rows(x, group)
        idx_all, _ = all_gather_rows(routing.topk_idx.to(torch.int64), group)
        w_all, _ = all_gather_rows(routing.topk_weights, group)
        return cls(x_all, idx_all, w_all, offsets, routing.num_experts, group)

    @property
    def rank(self) -> int:
        return self.group.rank()

    @property
    def expert_lo(self) -> int:
        return self.rank * (self.num_experts // self.group.size())

    @property
    def num_local_experts(self) -> int:
        return self.num_experts // self.group.size()

    def local_rows(self, t: torch.Tensor) -> torch.Tensor:
        """This rank's rows of a global ``[N, ...]`` tensor."""
        return t[self.offsets[self.rank] : self.offsets[self.rank + 1]]

    def local_topk(self) -> torch.Tensor:
        """[N, K] local expert of each global (token, k), -1 if not on this rank."""
        e = self.topk_idx - self.expert_lo
        return torch.where((self.topk_idx >= 0) & (e >= 0) & (e < self.num_local_experts), e, -1)

    def recv_tokens(self) -> torch.Tensor:
        """Global ids of the tokens this rank receives, in receive order."""
        return (self.local_topk() >= 0).any(dim=1).nonzero().flatten()

    def to_sources(self, per_recv: torch.Tensor) -> torch.Tensor:
        """What combine returns: each source token's rows from every rank that
        received it, summed (fp32). ``per_recv`` is this rank's ``[T_recv, ...]``."""
        full = per_recv.new_zeros((self.x.shape[0], *per_recv.shape[1:]), dtype=torch.float32)
        full[self.recv_tokens()] = per_recv.float()
        dist.all_reduce(full, group=self.group)
        return self.local_rows(full)


def _gather_rows_differentiable(t: torch.Tensor, group) -> tuple[torch.Tensor, list[int]]:
    """``all_gather_rows`` that autograd can backpropagate through."""
    _, offsets = all_gather_rows(t.new_zeros(t.shape[0], 0), group)
    counts = [b - a for a, b in itertools.pairwise(offsets)]
    padded = torch.cat([t, t.new_zeros(max(counts) - t.shape[0], *t.shape[1:])])
    chunks = dist_nn.all_gather(padded, group=group)
    return torch.cat([c[:n] for c, n in zip(chunks, counts, strict=True)]), offsets


@dataclass(frozen=True)
class Dropout:
    p: float
    seed: int
    recv_token_to_slots: torch.Tensor  # [T_recv, K] pool slot of each local (token, k), from a dispatch of the same routing


def swiglu(h: torch.Tensor) -> torch.Tensor:
    """Kernel A's gated activation: gate and up are adjacent columns of ``x @ W1.T``."""
    return F.silu(h[..., 0::2]) * h[..., 1::2]


def moe(x, routing: Routing, w1_local, w2_local, group, *, dropout: Dropout | None = None) -> torch.Tensor:
    """fp32 MoE layer output for this rank's tokens, differentiable in ``x``,
    the topk weights and both local weight tensors. The one bf16 rounding the
    kernels make by design (the stored pre-activation) is modelled; the rest of
    the kernels' error is what the tests bound.

    Each rank applies its local experts to every global token routed to them,
    so dW sums over all ranks' tokens; the partial outputs are then summed
    across ranks and each rank keeps its own rows."""
    x_all, offsets = _gather_rows_differentiable(x.float(), group)
    weights_all, _ = _gather_rows_differentiable(routing.topk_weights, group)
    inputs = Global(x_all, all_gather_rows(routing.topk_idx.long(), group)[0], weights_all, offsets, routing.num_experts, group)
    local_topk = inputs.local_topk()

    if dropout is not None:
        recv_ids = inputs.recv_tokens()
        recv_index = torch.full((x_all.shape[0],), -1, dtype=torch.long, device=x.device)
        recv_index[recv_ids] = torch.arange(recv_ids.numel(), device=x.device)

    # Every gathered input stays in the graph even when no token reaches this
    # rank's experts, so every rank runs the same backward collectives.
    out = x_all * 0 + weights_all[:, :1] * 0
    for e in range(w1_local.shape[0]):
        token, k = (local_topk == e).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        # Kernel A stores the pre-activation in bf16 and the backward reads it back, so
        # the reference rounds it the same way; everything else stays fp32.
        a = swiglu((x_all[token] @ w1_local[e].float().T).to(torch.bfloat16).float())
        if dropout is not None:
            slots = dropout.recv_token_to_slots[recv_index[token], k]
            a = a * keep_scale_ref(dropout.seed, slots, torch.arange(a.shape[1], device=a.device), dropout.p)
        out = out.index_add(0, token, weights_all[token, k, None] * (a @ w2_local[e].float().T))
    return inputs.local_rows(dist_nn.all_reduce(out, group=group))


def dtopk_scale(x, routing: Routing, w1_local, w2_local, grad_out, group) -> torch.Tensor:
    """[T, K] sum of the absolute terms of each topk-weight gradient,
    ``sum_i |a_i * g_i|`` with ``a`` the expert activation and ``g = grad_out @ W2``:
    the scale a dot product's rounding error is relative to."""
    x_all, offsets = all_gather_rows(x.float(), group)
    grad_all, _ = all_gather_rows(grad_out.float(), group)
    inputs = Global(x_all, all_gather_rows(routing.topk_idx.long(), group)[0], routing.topk_weights, offsets, routing.num_experts, group)
    local_topk = inputs.local_topk()
    scale = torch.zeros(local_topk.shape, device=x.device)
    for e in range(w1_local.shape[0]):
        token, k = (local_topk == e).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        a = swiglu((x_all[token] @ w1_local[e].float().T).to(torch.bfloat16).float())
        scale[token, k] = (a * (grad_all[token] @ w2_local[e].float())).abs().sum(1)
    dist.all_reduce(scale, group=group)
    return inputs.local_rows(scale)
