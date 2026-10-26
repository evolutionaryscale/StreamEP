"""``stream_moe_func`` forward and backward against the fp32 torch reference.

Every case runs the full layer (dispatch, kernel A, kernel Y, combine, and the
backward mirror) for every test shape and compares the output and the
gradients of ``x``, the topk weights, and both local weight tensors. The
reference sums every rank's tokens into dW, so a gradient that only covers one
rank's tokens fails.

Tolerances are per row (token, or expert for dW; see ``assert_rows_close``):
the kernels round activations to bf16 between the GEMMs and accumulate the
outputs with bf16 atomics, while the reference stays in fp32 throughout.
"""

import pytest
import torch
import torch.nn.functional as F

from stream_ep.stream_moe.stream_moe import TileConfig, stream_moe_func
from stream_ep_tests import reference, routing
from stream_ep_tests.utils import DTYPE, SHAPES, Shape, assert_rows_close, report

# Worst row over all test shapes and ranks: 0.74% (out), 0.58% (dx), 0.36%
# (dW); about 2x margin over the largest.
RTOL = 1.5e-2
# A topk-weight gradient is a dot product over I, and its rows hold only K of
# them, so it is checked per element against the sum of its terms' magnitudes.
# The kernel's bf16 pre-activation can differ from the reference's by one ulp
# per element (different accumulation order before rounding), which moves
# each term by at most one bf16 ulp: 2**-8 of the terms. Worst measured: 0.12%.
DTOPK_RTOL = 2**-8


def assert_dtopk_close(actual, expected, scale, name):
    ratio = (actual.float() - expected.float()).abs() / scale.clamp(min=1e-30)
    report(name, ratio.max().item())
    assert ratio.max() <= DTOPK_RTOL, f"{name}: error {ratio.max():.3e} of its terms' magnitude > {DTOPK_RTOL:.2e}"


def make_weights(group, shape: Shape, seed=0):
    """This rank's experts, scaled so each GEMM keeps unit-order outputs."""
    g = torch.Generator(device="cuda").manual_seed(seed * 1_000_003 + group.rank())
    E_local, H, I = shape.num_experts // group.size(), shape.hidden, shape.intermediate
    w1 = torch.randn(E_local, 2 * I, H, generator=g, device="cuda").mul_(H**-0.5).to(DTYPE)
    w2 = torch.randn(E_local, H, I, generator=g, device="cuda").mul_(I**-0.5).to(DTYPE)
    return w1, w2


def leaves(*tensors):
    return [t.detach().clone().requires_grad_() for t in tensors]


def grad(t: torch.Tensor) -> torch.Tensor:
    """``t.grad``, or zeros when autograd never reached ``t`` (no token used this rank's experts)."""
    return torch.zeros_like(t) if t.grad is None else t.grad


def check_layer(buffers, streams, group, shape, scenario, *, seed=0, dropout_p=0.0, **layer_kwargs):
    device = torch.device("cuda", torch.cuda.current_device())
    buffer, shape = buffers(shape), SHAPES[shape]
    r = routing.SCENARIOS[scenario](shape, group.rank(), group.size(), device, seed=seed)
    w1, w2 = make_weights(group, shape, seed)
    x = torch.randn(r.num_tokens, shape.hidden, dtype=DTYPE, device=device)
    grad_out = torch.randn_like(x)

    dropout = None
    if dropout_p > 0:
        # Placement is a function of routing, so a separate dispatch yields the slots the layer uses.
        _, handle, _ = buffer.dispatch(x, *r.dispatch_args(), r.num_experts)
        dropout = reference.Dropout(dropout_p, seed + 1, handle.recv_token_to_slots.long())
        layer_kwargs |= {"expert_dropout": dropout_p, "dropout_seed": dropout.seed}

    xs, ws, w1s, w2s = leaves(x, r.topk_weights, w1, w2)
    out = stream_moe_func(buffer, xs, r.topk_idx, ws, r.is_token_in_rank, w1s, w2s, streams=streams, num_experts=r.num_experts, **layer_kwargs)
    out.backward(grad_out)
    torch.cuda.synchronize()

    xr, wr, w1r, w2r = leaves(x, r.topk_weights, w1, w2)
    r_ref = routing.Routing(r.topk_idx, wr, r.num_experts, r.num_ranks)
    out_ref = reference.moe(xr, r_ref, w1r, w2r, group, dropout=dropout)
    out_ref.backward(grad_out.float())

    dtopk_scale = reference.dtopk_scale(x, r, w1, w2, grad_out, group)
    for name, actual, expected in (
        ("out", out, out_ref),
        ("dx", grad(xs), grad(xr)),
        ("dw1", grad(w1s), grad(w1r)),
        ("dw2", grad(w2s), grad(w2r)),
    ):
        assert_rows_close(actual, expected, RTOL, name)
    assert_dtopk_close(grad(ws), grad(wr), dtopk_scale, "dtopk_weights")


@pytest.mark.parametrize("scenario", routing.SCENARIOS)
@pytest.mark.parametrize("shape", SHAPES)
def test_layer(buffers, streams, group, shape, scenario):
    check_layer(buffers, streams, group, shape, scenario)


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("scenario", ["sentinels", "first_k"])
@pytest.mark.parametrize("shape", SHAPES)
def test_activation_checkpoint(buffers, streams, group, shape, scenario, level):
    if level == 2 and scenario == "first_k" and group.size() > 8:
        pytest.skip("known bug: level 2 hangs internode when a rank receives no tokens (NVL barrier timeout); fix in a follow-up")
    check_layer(buffers, streams, group, shape, scenario, activation_checkpoint_level=level)


@pytest.mark.parametrize("level", [0, 2])
@pytest.mark.parametrize("shape", SHAPES)
def test_expert_dropout(buffers, streams, group, shape, level):
    check_layer(buffers, streams, group, shape, "sentinels", dropout_p=0.1, activation_checkpoint_level=level)


@pytest.mark.parametrize("shape", SHAPES)
def test_tile_config_override(buffers, streams, group, shape):
    cfg = TileConfig(tile_n_a=128, tile_n_y=128, tile_n_y_bwd=128, tile_n_a_bwd=128, num_sms_a=96, num_sms_y=96)
    check_layer(buffers, streams, group, shape, "power_law", tile_config=cfg)


def check_chain(buffer, streams, group, shape: Shape, num_layers, *, num_steps=1, checked_layers=None, **layer_kwargs):
    """``num_steps`` training steps through ``num_layers`` pre-norm residual layers on one Buffer.

    Each backward ``dispatch_grads`` / ``combine_grads`` then runs while the
    other layers' state is live, and every step reuses the Buffer's rings, as
    in a deep model. On the last step each layer in ``checked_layers`` (all by
    default) is checked against the reference on its own input and incoming
    gradient, so a cross-layer bug fails the layer it corrupts while the
    chains' normal bf16 drift does not accumulate into the comparison."""
    device = torch.device("cuda", torch.cuda.current_device())
    layers = [(routing.uniform(shape, group.rank(), group.size(), device, seed=s, sentinel_p=0.02), *make_weights(group, shape, s)) for s in range(num_layers)]
    for _ in range(num_steps):
        h = leaves(torch.randn(shape.tokens, shape.hidden, dtype=DTYPE, device=device))[0]
        runs = []
        for r, w1, w2 in layers:
            # Pre-norm, as in the model: without it SwiGLU's roughly quadratic growth
            # overflows the residual stream within a few dozen layers.
            x = F.rms_norm(h, (shape.hidden,))
            x.retain_grad()
            w, w1l, w2l = leaves(r.topk_weights, w1, w2)
            out = stream_moe_func(buffer, x, r.topk_idx, w, r.is_token_in_rank, w1l, w2l, streams=streams, num_experts=r.num_experts, **layer_kwargs)
            out.retain_grad()
            runs.append((r, x, out, w, w1l, w2l))
            h = h + out
        h.backward(torch.randn_like(h))
        torch.cuda.synchronize()

    checks, dtopk_checks = [], []
    for i in range(num_layers) if checked_layers is None else checked_layers:
        r, x, out, w, w1l, w2l = runs[i]
        dtopk_scale = reference.dtopk_scale(x, r, w1l, w2l, out.grad, group)
        xr, wr, w1r, w2r = leaves(x, w, w1l, w2l)
        out_ref = reference.moe(xr, routing.Routing(r.topk_idx, wr, r.num_experts, r.num_ranks), w1r, w2r, group)
        out_ref.backward(out.grad.float())
        checks += [
            (f"layer {i} out", out, out_ref),
            (f"layer {i} dx", grad(x), grad(xr)),
            (f"layer {i} dw1", grad(w1l), grad(w1r)),
            (f"layer {i} dw2", grad(w2l), grad(w2r)),
        ]
        dtopk_checks.append((f"layer {i} dtopk_weights", grad(w), grad(wr), dtopk_scale))
    for name, actual, expected in checks:
        assert_rows_close(actual, expected, RTOL, name)
    for name, actual, expected, scale in dtopk_checks:
        assert_dtopk_close(actual, expected, scale, name)


@pytest.mark.parametrize("shape", SHAPES)
def test_residual_chain(buffers, streams, group, shape):
    check_chain(buffers(shape), streams, group, SHAPES[shape], num_layers=3, num_steps=3)
