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
from stream_ep.stream_moe.tile_scheduler import MAX_LOCAL_EXPERTS
from stream_ep_tests.kernels._common import (
    KERNEL_SHAPES,
    LAYOUTS,
    TILE_M,
    PoolLayout,
    configs,
    expert_weights,
    fired,
    per_expert,
    randn,
    stream_tiles,
    warm_up_producer,
)
from stream_ep_tests.reference import swiglu
from stream_ep_tests.utils import DTYPE, assert_rows_close

RTOL = 1e-2


def inputs(layout, H, I):
    pool = randn(layout.num_slots, H, seed=1)
    pool[~layout.real()] = float("inf")
    return pool, expert_weights(layout, 2 * I, H, seed=2)


def check(layout, pool, w1, postact, preact=None):
    real, I = layout.real(), postact.shape[-1]
    preact_ref = per_expert(layout, pool, w1, lambda x, w: x @ w.T)
    # Kernel A rounds the pre-activation to bf16 before the activation, as the reference model does.
    assert_rows_close(postact.view(-1, I)[real], swiglu(preact_ref.to(DTYPE).float())[real], RTOL, "postact")
    if preact is not None:
        assert_rows_close(preact.view(-1, 2 * I)[real], preact_ref[real], RTOL, "preact")


def launch(layout, pool, w1, count, target, *, preact=None, **kwargs):
    I = w1.shape[1] // 2
    postact = torch.full((layout.total_tiles, TILE_M, I), float("nan"), dtype=DTYPE, device="cuda")
    streaming_moe_a(pool, w1, postact, layout.block_offset(), count, target, preact_a=preact, tile_m=TILE_M, **kwargs)
    return postact


@pytest.mark.parametrize("num_sms", [None, 8])
@pytest.mark.parametrize("config", ["default", "alternate"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", KERNEL_SHAPES)
def test_kernel_a(shape, layout, config, num_sms):
    (H, I), layout = KERNEL_SHAPES[shape], LAYOUTS[layout]
    pool, w1 = inputs(layout, H, I)
    preact = torch.full((layout.total_tiles, TILE_M, 2 * I), float("nan"), dtype=DTYPE, device="cuda")
    target = layout.arrival_target()
    postact = launch(layout, pool, w1, fired(target), target, preact=preact, num_sms=num_sms, **configs("a", H, I)[config])
    torch.cuda.synchronize()
    check(layout, pool, w1, postact, preact)


@pytest.mark.parametrize("config", ["default", "alternate"])
@pytest.mark.parametrize("shape", KERNEL_SHAPES)
def test_kernel_a_streamed_pool(shape, config):
    (H, I), layout = KERNEL_SHAPES[shape], LAYOUTS["small"]
    src, w1 = inputs(layout, H, I)
    pool = torch.full_like(src, float("nan"))
    target = layout.arrival_target()
    count = torch.zeros_like(target)

    warm_up_producer(pool, src, count, target)
    compute, producer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(compute):
        # Leave SMs free for the producer: a persistent grid on every SM would starve it.
        postact = launch(layout, pool, w1, count, target, num_sms=16, **configs("a", H, I)[config])
    with torch.cuda.stream(producer):
        stream_tiles(pool, src, count, target)
    torch.cuda.synchronize()
    check(layout, src, w1, postact)


def many_experts_layout(num_experts: int) -> PoolLayout:
    return PoolLayout(tuple(0 if e % 7 == 3 else 64 + 32 * (e % 5) for e in range(num_experts)))


@pytest.mark.parametrize("num_experts", [32, 64])
def test_many_local_experts(num_experts):
    """The scheduler finds each tile's expert with a warp ballot over the block offsets;
    32 and more experts take its second ballot slot, and 64 fill it."""
    H, I = KERNEL_SHAPES["512x256"]
    layout = many_experts_layout(num_experts)
    pool, w1 = inputs(layout, H, I)
    target = layout.arrival_target()
    postact = launch(layout, pool, w1, fired(target), target, num_sms=8, **configs("a", H, I)["default"])
    torch.cuda.synchronize()
    check(layout, pool, w1, postact)


def test_too_many_local_experts_raises():
    H, I = KERNEL_SHAPES["512x256"]
    layout = many_experts_layout(MAX_LOCAL_EXPERTS + 1)
    pool, w1 = inputs(layout, H, I)
    target = layout.arrival_target()
    with pytest.raises(AssertionError, match="local experts exceed"):
        launch(layout, pool, w1, fired(target), target, **configs("a", H, I)["default"])
