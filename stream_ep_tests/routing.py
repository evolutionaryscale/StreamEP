"""Routing scenarios for the distributed tests and bench harnesses.

Each scenario builds one rank's ``Routing`` from ``(rank, num_ranks, seed)``
alone, so a failure reproduces exactly and the same scenario runs at any world
size. ``SCENARIOS`` is the set the distributed tests sweep.
"""

from dataclasses import dataclass

import torch

import stream_ep
from stream_ep_tests.utils import NUM_EXPERTS, SEQ_LEN_PER_RANK, TOPK


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


def uniform(
    rank,
    num_ranks,
    device,
    *,
    seed=0,
    num_tokens=SEQ_LEN_PER_RANK,
    num_topk=TOPK,
    num_experts=NUM_EXPERTS,
    sentinel_p=0.0,
    empty_expert=None,
) -> Routing:
    """K distinct uniformly random experts per token, as a top-k router picks.

    ``sentinel_p`` drops that fraction of slots to ``-1``; ``empty_expert``
    drops every slot routed to that expert so it receives nothing."""
    g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + rank)
    topk_idx = _distinct(torch.rand(num_tokens, num_experts, generator=g, device=device), num_topk)
    if sentinel_p > 0:
        topk_idx = torch.where(torch.rand(topk_idx.shape, generator=g, device=device) < sentinel_p, -1, topk_idx)
    if empty_expert is not None:
        topk_idx = torch.where(topk_idx == empty_expert, -1, topk_idx)
    return _routing(topk_idx, g, num_experts, num_ranks)


def first_k(rank, num_ranks, device, *, seed=0, num_tokens=SEQ_LEN_PER_RANK, num_topk=TOPK, num_experts=NUM_EXPERTS) -> Routing:
    """Every token picks experts 0..K-1: their ranks get everything, the rest nothing."""
    g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + rank)
    topk_idx = torch.arange(num_topk, device=device).expand(num_tokens, num_topk)
    return _routing(topk_idx, g, num_experts, num_ranks)


def even_experts(rank, num_ranks, device, *, seed=0, num_tokens=SEQ_LEN_PER_RANK, num_topk=TOPK, num_experts=NUM_EXPERTS) -> Routing:
    """Random routing over even experts only: every odd expert is empty."""
    g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + rank)
    topk_idx = 2 * _distinct(torch.rand(num_tokens, num_experts // 2, generator=g, device=device), num_topk)
    return _routing(topk_idx, g, num_experts, num_ranks)


def hot_rank(
    rank, num_ranks, device, *, seed=0, num_tokens=SEQ_LEN_PER_RANK, num_topk=TOPK, num_experts=NUM_EXPERTS, hot=1, frac=0.9
) -> Routing:
    """``frac`` of tokens route only to rank ``hot``'s experts, the rest uniformly."""
    g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + rank)
    local = num_experts // num_ranks
    spread = _distinct(torch.rand(num_tokens, num_experts, generator=g, device=device), num_topk)
    on_hot = (hot % num_ranks) * local + _distinct(torch.rand(num_tokens, local, generator=g, device=device), num_topk)
    is_hot = torch.rand(num_tokens, 1, generator=g, device=device) < frac
    return _routing(torch.where(is_hot, on_hot, spread), g, num_experts, num_ranks)


def power_law(
    rank, num_ranks, device, *, seed=0, num_tokens=SEQ_LEN_PER_RANK, num_topk=TOPK, num_experts=NUM_EXPERTS, alpha=1.2
) -> Routing:
    """Expert popularity ~ Zipf(alpha): a few hot experts and a long tail."""
    g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + rank)
    popularity = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32) ** -alpha
    topk_idx = torch.multinomial(popularity.expand(num_tokens, -1), num_topk, generator=g)
    return _routing(topk_idx, g, num_experts, num_ranks)


def _routing(topk_idx: torch.Tensor, g: torch.Generator, num_experts: int, num_ranks: int) -> Routing:
    weights = torch.softmax(torch.randn(topk_idx.shape, generator=g, device=topk_idx.device), dim=-1)
    return Routing(topk_idx.to(stream_ep.topk_idx_t).contiguous(), weights, num_experts, num_ranks)


def _uneven_tokens(rank, num_ranks, device, **kw) -> Routing:
    # Rank 0 sends 1/64 of its peers' tokens: its receive budget sized from its
    # own token count would be far below what it receives.
    num_tokens = kw.pop("num_tokens", SEQ_LEN_PER_RANK)
    return uniform(rank, num_ranks, device, num_tokens=num_tokens // 64 if rank == 0 else num_tokens, **kw)


SCENARIOS = {
    "uniform": uniform,
    "sentinels": lambda rank, num_ranks, device, **kw: uniform(rank, num_ranks, device, sentinel_p=0.05, **kw),
    "empty_expert": lambda rank, num_ranks, device, **kw: uniform(rank, num_ranks, device, sentinel_p=0.05, empty_expert=1, **kw),
    "uneven_tokens": _uneven_tokens,
    "hot_rank": hot_rank,
    "power_law": power_law,
    "even_experts": even_experts,
    "first_k": first_k,
}
