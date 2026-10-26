"""Routing scenarios for the distributed tests and bench harnesses.

Each scenario builds one rank's ``Routing`` for a ``Shape`` from ``(rank,
num_ranks, seed)`` alone, so a failure reproduces exactly and the same scenario
runs at any world size. ``SCENARIOS`` is the set the distributed tests sweep.
"""

from dataclasses import dataclass

import torch

import stream_ep
from stream_ep_tests.utils import Shape


@dataclass(frozen=True)
class Routing:
    topk_idx: torch.Tensor  # [T, K] topk_idx_t, -1 = dropped slot
    topk_weights: torch.Tensor  # [T, K] fp32
    num_experts: int
    num_ranks: int

    @property
    def num_tokens(self) -> int:
        return self.topk_idx.shape[0]

    @property
    def is_token_in_rank(self) -> torch.Tensor:
        """[T, num_ranks] bool: token t has at least one expert on rank r."""
        owner = torch.where(self.topk_idx >= 0, self.topk_idx // (self.num_experts // self.num_ranks), -1)
        ranks = torch.arange(self.num_ranks, device=self.topk_idx.device)
        return (owner.unsqueeze(-1) == ranks).any(dim=1)

    def dispatch_args(self) -> tuple:
        """Positional routing arguments of ``Buffer.dispatch`` / ``stream_moe_func`` after ``x``."""
        return self.topk_idx, self.topk_weights, self.is_token_in_rank


def _distinct(scores: torch.Tensor, k: int) -> torch.Tensor:
    return scores.topk(k, dim=-1).indices


def _generator(rank: int, seed: int, device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed * 1_000_003 + rank)


def _routing(topk_idx: torch.Tensor, g: torch.Generator, shape: Shape, num_ranks: int) -> Routing:
    weights = torch.softmax(torch.randn(topk_idx.shape, generator=g, device=topk_idx.device), dim=-1)
    return Routing(topk_idx.to(stream_ep.topk_idx_t).contiguous(), weights, shape.num_experts, num_ranks)


def uniform(shape: Shape, rank, num_ranks, device, *, seed=0, num_tokens=None, sentinel_p=0.0, empty_expert=None) -> Routing:
    """K distinct uniformly random experts per token, as a top-k router picks.

    ``sentinel_p`` drops that fraction of slots to ``-1``; ``empty_expert``
    drops every slot routed to that expert so it receives nothing."""
    g = _generator(rank, seed, device)
    scores = torch.rand(num_tokens or shape.tokens, shape.num_experts, generator=g, device=device)
    topk_idx = _distinct(scores, shape.topk)
    if sentinel_p > 0:
        topk_idx = torch.where(torch.rand(topk_idx.shape, generator=g, device=device) < sentinel_p, -1, topk_idx)
    if empty_expert is not None:
        topk_idx = torch.where(topk_idx == empty_expert, -1, topk_idx)
    return _routing(topk_idx, g, shape, num_ranks)


def uneven_tokens(shape: Shape, rank, num_ranks, device, *, seed=0) -> Routing:
    """Rank 0 sends 1/64 of its peers' tokens: its receive budget sized from its
    own token count would be far below what it receives."""
    return uniform(shape, rank, num_ranks, device, seed=seed, num_tokens=max(1, shape.tokens // 64) if rank == 0 else shape.tokens)


def first_k(shape: Shape, rank, num_ranks, device, *, seed=0) -> Routing:
    """Every token picks experts 0..K-1: their ranks get everything, the rest nothing."""
    topk_idx = torch.arange(shape.topk, device=device).expand(shape.tokens, shape.topk)
    return _routing(topk_idx, _generator(rank, seed, device), shape, num_ranks)


def even_experts(shape: Shape, rank, num_ranks, device, *, seed=0) -> Routing:
    """Random routing over even experts only: every odd expert is empty."""
    g = _generator(rank, seed, device)
    topk_idx = 2 * _distinct(torch.rand(shape.tokens, shape.num_experts // 2, generator=g, device=device), shape.topk)
    return _routing(topk_idx, g, shape, num_ranks)


def hot_rank(shape: Shape, rank, num_ranks, device, *, seed=0, hot=1, frac=0.9) -> Routing:
    """``frac`` of tokens route only to the experts starting at rank ``hot``'s
    first one (spilling onto the next rank when it has fewer than K), the rest uniformly."""
    g = _generator(rank, seed, device)
    local = shape.num_experts // num_ranks
    window = max(local, shape.topk)
    spread = _distinct(torch.rand(shape.tokens, shape.num_experts, generator=g, device=device), shape.topk)
    on_hot = (
        (hot % num_ranks) * local + _distinct(torch.rand(shape.tokens, window, generator=g, device=device), shape.topk)
    ) % shape.num_experts
    is_hot = torch.rand(shape.tokens, 1, generator=g, device=device) < frac
    return _routing(torch.where(is_hot, on_hot, spread), g, shape, num_ranks)


def power_law(shape: Shape, rank, num_ranks, device, *, seed=0, alpha=1.2) -> Routing:
    """Expert popularity ~ Zipf(alpha): a few hot experts and a long tail."""
    g = _generator(rank, seed, device)
    popularity = torch.arange(1, shape.num_experts + 1, device=device, dtype=torch.float32) ** -alpha
    topk_idx = torch.multinomial(popularity.expand(shape.tokens, -1), shape.topk, generator=g)
    return _routing(topk_idx, g, shape, num_ranks)


SCENARIOS = {
    "uniform": uniform,
    "sentinels": lambda *args, **kw: uniform(*args, sentinel_p=0.05, **kw),
    "empty_expert": lambda *args, **kw: uniform(*args, sentinel_p=0.05, empty_expert=1, **kw),
    "uneven_tokens": uneven_tokens,
    "hot_rank": hot_rank,
    "power_law": power_law,
    "even_experts": even_experts,
    "first_k": first_k,
}
