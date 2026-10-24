"""Expert dropout in kernel A (forward) and kernel_y_bwd against a torch
reference that regenerates the mask with ``keep_scale_ref``.

The mask is keyed on ``(seed, pool row, column)``. A dropped unit must be
exactly zero in every masked output and a kept one scaled by ``1 / (1 - p)``;
``preact_a`` stays unmasked, and a different seed gives a different mask.
"""

import torch

from stream_ep.stream_moe.expert_dropout import keep_scale_ref
from stream_ep.stream_moe.kernel_a import streaming_moe_a
from stream_ep.stream_moe.kernel_y_bwd import streaming_moe_y_bwd
from stream_ep_tests.kernels._common import LAYOUTS, TILE_M, expert_weights, fired, per_expert, randn
from stream_ep_tests.reference import swiglu
from stream_ep_tests.utils import DTYPE, H, I, assert_rows_close

P = 0.2
SEED = 1234
RTOL = 1e-2
LAYOUT = LAYOUTS["small"]


def mask(seed: int) -> torch.Tensor:
    """[num_slots, I] keep scale (``1 / (1 - P)`` or 0) of every pool slot."""
    return keep_scale_ref(seed, torch.arange(LAYOUT.num_slots, device="cuda"), torch.arange(I, device="cuda"), P)


def run_kernel_a(seed):
    pool, w1 = randn(LAYOUT.num_slots, H, seed=1), expert_weights(LAYOUT, 2 * I, H, seed=2)
    postact = torch.empty(LAYOUT.total_tiles, TILE_M, I, dtype=DTYPE, device="cuda")
    preact = torch.empty(LAYOUT.total_tiles, TILE_M, 2 * I, dtype=DTYPE, device="cuda")
    target = LAYOUT.arrival_target()
    streaming_moe_a(pool, w1, postact, LAYOUT.block_offset(), fired(target), target, preact_a=preact, tile_m=TILE_M, tile_n=192, cluster_n=2,
                    dropout_p=P, dropout_seed=seed)  # fmt: skip
    torch.cuda.synchronize()
    return pool, w1, postact.view(-1, I), preact.view(-1, 2 * I)


def test_kernel_a_expert_dropout():
    real = LAYOUT.real()
    pool, w1, postact, preact = run_kernel_a(SEED)
    _, _, other_seed, _ = run_kernel_a(SEED + 1)

    preact_ref = per_expert(LAYOUT, pool, w1, lambda x, w: x @ w.T)
    keep = mask(SEED)
    expected = swiglu(preact_ref.to(DTYPE).float()) * keep
    assert_rows_close(preact[real], preact_ref[real], RTOL, "preact (unmasked)")
    assert_rows_close(postact[real], expected[real], RTOL, "postact")
    assert torch.equal(postact[real] == 0, expected[real] == 0), "dropped units differ from the reference mask"
    assert abs((keep[real] > 0).float().mean().item() - (1 - P)) < 5e-3
    assert not torch.equal(postact[real] == 0, other_seed[real] == 0), "a different seed must change the mask"


def test_kernel_y_bwd_expert_dropout():
    real = LAYOUT.real()
    dL_do = randn(LAYOUT.num_slots, H, seed=7)
    w2 = expert_weights(LAYOUT, H, I, seed=8)
    preact = randn(LAYOUT.num_slots, 2 * I, seed=9)
    weight = torch.where(real, torch.rand(LAYOUT.num_slots, device="cuda"), 0.0)
    recv = LAYOUT.recv_tokens(LAYOUT.num_slots // 3)
    dswiglu = torch.empty(LAYOUT.total_tiles, TILE_M, 2 * I, dtype=DTYPE, device="cuda")
    postact = torch.empty(LAYOUT.total_tiles, TILE_M, I, dtype=DTYPE, device="cuda")
    dweight = torch.zeros(LAYOUT.num_slots, dtype=torch.float32, device="cuda")
    target = LAYOUT.arrival_target()
    streaming_moe_y_bwd(dL_do, w2, dswiglu, postact, weight, recv, preact.view(-1, TILE_M, 2 * I), dweight, LAYOUT.block_offset(), fired(target), target,
                        tile_m=TILE_M, tile_n=192, dropout_p=P, dropout_seed=SEED)  # fmt: skip
    torch.cuda.synchronize()

    g = per_expert(LAYOUT, dL_do, w2, lambda d, w: d @ w)[real]
    pre = preact[real].float().requires_grad_()
    a = swiglu(pre) * mask(SEED)[real]
    w = weight[real, None]
    (dswiglu_ref,) = torch.autograd.grad(a, pre, w * g)
    assert_rows_close(dswiglu.view(-1, 2 * I)[real], dswiglu_ref, RTOL, "dL_dswiglu_in")
    assert_rows_close(postact.view(-1, I)[real], (w * a).detach(), RTOL, "postact_a_for_dW2")
    dropped = mask(SEED)[real] == 0
    assert not postact.view(-1, I)[real][dropped].any(), "postact_a_for_dW2 nonzero at dropped units"
    terms = (a * g).detach()
    assert ((dweight[real] - terms.sum(1)).abs() <= I * 2**-24 * terms.abs().sum(1)).all(), (
        "dL_dweight off beyond fp32 dot-product rounding"
    )
