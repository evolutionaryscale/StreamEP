"""Kernel Y backward (down-projection data gradient with the SwiGLU backward and
the weight gradient in its epilogue) against torch autograd.

Per real slot ``s`` of expert ``e``, with ``a = swiglu(preact[s])`` and
``g = dL_do_pool[s] @ W2[e]``:

- ``dL_dswiglu_in[s]`` is the gradient of ``w[s] * a`` w.r.t. ``preact[s]``, given ``g``;
- ``postact_a_for_dW2[s] = w[s] * a``;
- ``dL_dweight[s] = a . g``, checked against the sum of its terms' magnitudes.

Padding rows of ``dL_do_pool`` hold inf, as an uninitialized pool may: the
GEMM through them produces inf and the SwiGLU backward NaN, so the padding
mask must assign zeros there rather than multiply. Both pool-shaped outputs
start as NaN, so a row the kernel skips also fails.
"""

import pytest
import torch

from stream_ep.stream_moe.kernel_y_bwd import streaming_moe_y_bwd
from stream_ep_tests.kernels._common import LAYOUTS, TILE_M, expert_weights, fired, per_expert, randn, stream_tiles, warm_up_producer
from stream_ep_tests.reference import swiglu
from stream_ep_tests.utils import DTYPE, H, I, assert_rows_close, report

# Production: tile_n_y_bwd=192 without clusters (default_tile_config).
CONFIGS = {"production": {"tile_n": 192}, "alternate": {"tile_n": 128}}
RTOL = 1e-2
# dL_dweight is an fp32 dot product over I terms of bf16 inputs: worst-case
# accumulation error I * 2**-24 of the terms' magnitudes.
DOT_RTOL = I * 2**-24


class Case:
    def __init__(self, layout):
        self.layout = layout
        self.real = layout.real()
        self.dL_do = randn(layout.num_slots, H, seed=7)
        self.dL_do[~self.real] = float("inf")
        self.w2 = expert_weights(layout, H, I, seed=8)
        self.preact = randn(layout.num_slots, 2 * I, seed=9)
        self.weight = torch.where(self.real, torch.rand(layout.num_slots, device="cuda") * 2 - 0.5, 0.0)
        self.recv = layout.recv_tokens(layout.num_slots // 3, seed=2)
        self.target = layout.arrival_target()
        self.dswiglu = torch.full((layout.total_tiles, TILE_M, 2 * I), float("nan"), dtype=DTYPE, device="cuda")
        self.postact = torch.full((layout.total_tiles, TILE_M, I), float("nan"), dtype=DTYPE, device="cuda")
        self.dweight = torch.zeros(layout.num_slots, dtype=torch.float32, device="cuda")

    def launch(self, dL_do, count, **kwargs):
        t = self.layout.total_tiles
        streaming_moe_y_bwd(dL_do, self.w2, self.dswiglu, self.postact, self.weight, self.recv, self.preact.view(t, TILE_M, 2 * I), self.dweight,
                            self.layout.block_offset(), count, self.target, tile_m=TILE_M, **kwargs)  # fmt: skip

    def check(self):
        real = self.real
        g = per_expert(self.layout, self.dL_do, self.w2, lambda d, w: d @ w)[real]
        preact = self.preact[real].float().requires_grad_()
        a = swiglu(preact)
        w = self.weight[real, None]
        (dswiglu_ref,) = torch.autograd.grad(a, preact, w * g)
        assert_rows_close(self.dswiglu.view(-1, 2 * I)[real], dswiglu_ref, RTOL, "dL_dswiglu_in")
        assert_rows_close(self.postact.view(-1, I)[real], (w * a).detach(), RTOL, "postact_a_for_dW2")
        # A dot product's rounding error scales with the sum of its terms' magnitudes,
        # not with its (possibly cancelling) value.
        terms = (a * g).detach()
        dot_ratio = (self.dweight[real] - terms.sum(1)).abs() / terms.abs().sum(1)
        report("dL_dweight", dot_ratio.max().item())
        assert dot_ratio.max() <= DOT_RTOL, f"dL_dweight: slot error {dot_ratio.max():.3e} of its terms' magnitude > {DOT_RTOL:.1e}"
        assert not self.dswiglu.view(-1, 2 * I)[~real].any(), "dL_dswiglu_in not zero at padding rows"
        assert not self.postact.view(-1, I)[~real].any(), "postact_a_for_dW2 not zero at padding rows"


@pytest.mark.parametrize("num_sms", [None, 8])
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_kernel_y_bwd(layout, config, num_sms):
    case = Case(LAYOUTS[layout])
    case.launch(case.dL_do, fired(case.target), num_sms=num_sms, **CONFIGS[config])
    torch.cuda.synchronize()
    case.check()


def test_kernel_y_bwd_streamed_gradients():
    """dispatch_grads fills ``dL_do_pool`` while kernel_y_bwd runs, as in the backward pass."""
    case = Case(LAYOUTS["small"])
    dL_do = torch.full_like(case.dL_do, float("nan"))
    count = torch.zeros_like(case.target)
    warm_up_producer(dL_do, case.dL_do, count, case.target)
    compute, producer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(compute):
        case.launch(dL_do, count, num_sms=16, **CONFIGS["production"])
    with torch.cuda.stream(producer):
        stream_tiles(dL_do, case.dL_do, count, case.target)
    torch.cuda.synchronize()
    case.check()


def test_oversized_tile_raises():
    """A tile too large for y_bwd's four-stage mainloop fails at compile time
    with an actionable message, not as an invalid launch."""
    E, I_big, tiles = 1, 1024, 1
    slots = tiles * TILE_M
    z = lambda *s, dtype=DTYPE: torch.zeros(*s, dtype=dtype, device="cuda")  # noqa: E731
    target = torch.zeros(tiles, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="ab_stage=4.*shrink tile_n_y_bwd"):
        streaming_moe_y_bwd(z(slots, H), z(E, H, I_big), z(tiles, TILE_M, 2 * I_big), z(tiles, TILE_M, I_big), z(slots, dtype=torch.float32),
                            torch.full((slots,), -1, dtype=torch.int32, device="cuda"), z(tiles, TILE_M, 2 * I_big), z(slots, dtype=torch.float32),
                            torch.tensor([0, tiles], dtype=torch.int32, device="cuda"), target, target, tile_m=TILE_M, tile_n=256)  # fmt: skip
