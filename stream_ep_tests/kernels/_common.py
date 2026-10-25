"""Shared setup for the single-GPU kernel tests: pool layouts, arrival signals,
and a producer that streams pool tiles in the way dispatch does.

Kernels run at each distinct (hidden, intermediate) pair of the test shapes.
``LAYOUTS["small"]`` starts with an empty expert and has a partial tile and
full tiles; ``LAYOUTS["many_tiles"]`` has 40 experts, several empty, and enough
tiles that every CTA loops over many of them. Tile configs come from
``default_tile_config`` for the shape (with the 2-CTA clusters stream_moe_func
launches A and Y with) plus an alternate.
"""

from dataclasses import dataclass

import torch

from stream_ep.stream_moe.stream_moe import default_tile_config
from stream_ep_tests.utils import DTYPE, SHAPES

TILE_M = 128


@dataclass(frozen=True)
class PoolLayout:
    """An expert-major, ``TILE_M``-padded pool with ``rows[e]`` real rows for expert ``e``."""

    rows: tuple[int, ...]

    @property
    def num_experts(self) -> int:
        return len(self.rows)

    @property
    def tiles(self) -> torch.Tensor:
        return (torch.tensor(self.rows, device="cuda") + TILE_M - 1) // TILE_M

    @property
    def total_tiles(self) -> int:
        return int(self.tiles.sum())

    @property
    def num_slots(self) -> int:
        return self.total_tiles * TILE_M

    def block_offset(self) -> torch.Tensor:
        """``expert_pool_block_offset``: [E_local + 1] prefix sum of tiles per expert."""
        return torch.cat([self.tiles.new_zeros(1), self.tiles.cumsum(0)]).to(torch.int32)

    def slot_expert(self) -> torch.Tensor:
        """[num_slots] expert owning each pool slot."""
        return torch.repeat_interleave(torch.arange(self.num_experts, device="cuda"), self.tiles * TILE_M)

    def real(self) -> torch.Tensor:
        """[num_slots] bool: slot holds a real (token, k); the rest is per-expert padding."""
        start = torch.repeat_interleave(self.block_offset()[:-1].long() * TILE_M, self.tiles * TILE_M)
        count = torch.repeat_interleave(torch.tensor(self.rows, device="cuda"), self.tiles * TILE_M)
        return torch.arange(self.num_slots, device="cuda") - start < count

    def arrival_target(self) -> torch.Tensor:
        """Real rows per tile, as dispatch's metadata kernel sets it."""
        return self.real().view(-1, TILE_M).sum(1).to(torch.int32)

    def recv_tokens(self, num_recv_tokens: int, seed: int = 0) -> torch.Tensor:
        """``pool_recv_token``: each real slot maps to a receive token, several
        slots per token (a token routed to several local experts), padding -1."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        tokens = torch.randint(0, num_recv_tokens, (self.num_slots,), generator=g, device="cuda", dtype=torch.int32)
        return torch.where(self.real(), tokens, -1)


def _random_rows(num_experts=40, max_rows=700, empty=(0, 7, 23, 39), seed=0) -> tuple[int, ...]:
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(1, max_rows, (num_experts,), generator=g)
    rows[list(empty)] = 0
    return tuple(rows.tolist())


LAYOUTS = {
    "small": PoolLayout((0, 200, 0, 129, 384)),
    "many_tiles": PoolLayout(_random_rows()),
}

# (hidden, intermediate) pairs the kernel tests run at.
KERNEL_SHAPES = {f"{s.hidden}x{s.intermediate}": (s.hidden, s.intermediate) for s in SHAPES.values()}


def configs(kernel: str, hidden: int, intermediate: int) -> dict[str, dict]:
    """``default`` is what stream_moe_func launches ``kernel`` with at this shape; ``alternate`` an unclustered 128-wide tile."""
    cfg = default_tile_config(intermediate, hidden)
    default = {
        "a": {"tile_n": cfg.tile_n_a, "cluster_n": 2},
        "y": {"tile_n": cfg.tile_n_y, "cluster_n": 2},
        "a_bwd": {"tile_n": cfg.tile_n_a_bwd},
        "y_bwd": {"tile_n": cfg.tile_n_y_bwd},
    }[kernel]
    return {"default": default, "alternate": {"tile_n": 128}}


def randn(*shape, scale=1.0, seed=0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32).mul_(scale).to(DTYPE)


def expert_weights(layout: PoolLayout, out_features: int, in_features: int, seed: int) -> torch.Tensor:
    """[E_local, out, in] bf16, scaled so a GEMM over ``in_features`` keeps unit-order outputs."""
    return randn(layout.num_experts, out_features, in_features, scale=in_features**-0.5, seed=seed)


def per_expert(layout: PoolLayout, rows: torch.Tensor, weights: torch.Tensor, fn) -> torch.Tensor:
    """Apply ``fn(rows_of_expert_e, weights[e])`` to every expert's real slots (fp32).
    Returns [num_slots, ...] with padding rows zero."""
    real, expert = layout.real(), layout.slot_expert()
    out = None
    for e in range(layout.num_experts):
        idx = (real & (expert == e)).nonzero().flatten()
        if idx.numel() == 0:
            continue
        y = fn(rows[idx].float(), weights[e].float())
        if out is None:
            out = y.new_zeros(layout.num_slots, *y.shape[1:])
        out[idx] = y
    return out


def fired(target: torch.Tensor) -> torch.Tensor:
    """An arrival count already at its target: the consumer's spin passes at once."""
    return target.clone()


def warm_up_producer(dst: torch.Tensor, src: torch.Tensor, count: torch.Tensor, target: torch.Tensor) -> None:
    """Run ``stream_tiles`` once on scratch copies before the consumer launches.

    With lazy module loading, the first launch of a kernel from a not-yet-loaded
    module waits for the device to go idle, which never happens while a
    persistent consumer spins on the arrival counts."""
    stream_tiles(dst.clone(), src, count.clone(), target)
    torch.cuda.synchronize()


def stream_tiles(
    dst: torch.Tensor, src: torch.Tensor, count: torch.Tensor, target: torch.Tensor, *, chunks: int = 2, sleep_cycles: int = 20_000
):
    """On the current stream, copy ``src`` into ``dst`` tile by tile in reverse
    tile order, each tile in ``chunks`` row ranges, and after each range add its
    share of the tile's target to the arrival count, as dispatch's receivers do.
    Until a tile's count reaches its target its rows in ``dst`` still hold
    whatever was there, so a consumer that does not wait reads them."""
    rows_per_chunk = TILE_M // chunks
    for t in reversed(range(count.shape[0])):
        share = target[t : t + 1] // chunks
        for c in range(chunks):
            lo = t * TILE_M + c * rows_per_chunk
            torch.cuda._sleep(sleep_cycles)
            dst[lo : lo + rows_per_chunk].copy_(src[lo : lo + rows_per_chunk])
            count[t : t + 1].add_(share if c < chunks - 1 else target[t : t + 1] - (chunks - 1) * share)
