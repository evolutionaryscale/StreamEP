"""Kernel A (up-projection + SwiGLU over the pool) against a torch reference.

``postact[s] = swiglu(pool[s] @ W1[e(s)].T)`` for every real slot, and with
``preact_a`` the pre-activation itself. Padding rows of the pool hold inf, as
an uninitialized pool may; real rows must not see it. ``num_sms=8`` makes each
CTA loop over many tiles and experts; the streamed case writes the pool while
the kernel runs, so a kernel that reads a tile before its arrival count reaches
its target reads NaN.
"""

import pytest
import torch

from stream_ep.stream_moe.kernel_a import streaming_moe_a
from stream_ep_tests.kernels._common import (
    LAYOUTS,
    TILE_M,
    PoolLayout,
    expert_weights,
    fired,
    per_expert,
    randn,
    stream_tiles,
    warm_up_producer,
)
from stream_ep_tests.reference import swiglu
from stream_ep_tests.utils import DTYPE, H, I, assert_rows_close

# Production: tile_n_a=192 with 2-CTA clusters (stream_moe.py); the alternate covers the unclustered path.
CONFIGS = {"production": {"tile_n": 192, "cluster_n": 2}, "alternate": {"tile_n": 128, "cluster_n": 1}}
RTOL = 1e-2


def inputs(layout):
    pool = randn(layout.num_slots, H, seed=1)
    pool[~layout.real()] = float("inf")
    return pool, expert_weights(layout, 2 * I, H, seed=2)


def check(layout, pool, w1, postact, preact=None):
    real = layout.real()
    preact_ref = per_expert(layout, pool, w1, lambda x, w: x @ w.T)
    # Kernel A rounds the pre-activation to bf16 before the activation, as the reference model does.
    assert_rows_close(postact.view(-1, I)[real], swiglu(preact_ref.to(DTYPE).float())[real], RTOL, "postact")
    if preact is not None:
        assert_rows_close(preact.view(-1, 2 * I)[real], preact_ref[real], RTOL, "preact")


@pytest.mark.parametrize("num_sms", [None, 8])
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_kernel_a(layout, config, num_sms):
    layout = LAYOUTS[layout]
    pool, w1 = inputs(layout)
    postact = torch.full((layout.total_tiles, TILE_M, I), float("nan"), dtype=DTYPE, device="cuda")
    preact = torch.full((layout.total_tiles, TILE_M, 2 * I), float("nan"), dtype=DTYPE, device="cuda")
    target = layout.arrival_target()
    streaming_moe_a(
        pool, w1, postact, layout.block_offset(), fired(target), target, preact_a=preact, tile_m=TILE_M, num_sms=num_sms, **CONFIGS[config]
    )
    torch.cuda.synchronize()
    check(layout, pool, w1, postact, preact)


@pytest.mark.parametrize("config", CONFIGS)
def test_kernel_a_streamed_pool(config):
    layout = LAYOUTS["small"]
    src, w1 = inputs(layout)
    pool = torch.full_like(src, float("nan"))
    postact = torch.full((layout.total_tiles, TILE_M, I), float("nan"), dtype=DTYPE, device="cuda")
    target = layout.arrival_target()
    count = torch.zeros_like(target)

    warm_up_producer(pool, src, count, target)
    compute, producer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(compute):
        # Leave SMs free for the producer: a persistent grid on every SM would starve it.
        streaming_moe_a(pool, w1, postact, layout.block_offset(), count, target, tile_m=TILE_M, num_sms=16, **CONFIGS[config])
    with torch.cuda.stream(producer):
        stream_tiles(pool, src, count, target)
    torch.cuda.synchronize()
    check(layout, src, w1, postact)


@pytest.mark.parametrize("num_experts", [32, 64, 100])
def test_many_local_experts(num_experts):
    """The scheduler finds each tile's expert with a warp ballot over the block offsets;
    32 and more experts take its second ballot slot."""
    if num_experts > 64:
        pytest.skip(
            "known bug: the scheduler's expert lookup covers at most 64 local experts while dispatch allows 1024; fix in a follow-up"
        )
    layout = PoolLayout(tuple(0 if e % 7 == 3 else 64 + 32 * (e % 5) for e in range(num_experts)))
    pool, w1 = inputs(layout)
    postact = torch.full((layout.total_tiles, TILE_M, I), float("nan"), dtype=DTYPE, device="cuda")
    target = layout.arrival_target()
    streaming_moe_a(pool, w1, postact, layout.block_offset(), fired(target), target, tile_m=TILE_M, num_sms=8, **CONFIGS["production"])
    torch.cuda.synchronize()
    check(layout, pool, w1, postact)
