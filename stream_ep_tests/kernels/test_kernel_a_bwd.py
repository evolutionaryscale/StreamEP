"""Kernel A backward (data gradient + atomic scatter into ``dL_dx_per_r``) against a torch reference.

``dL_dx_per_r[r] = sum over slots s with pool_recv_token[s] == r of dL_dswiglu_in[s] @ W1[e(s)]``,
then ``bwd_a_done_per_token[r] = dispatch_seq`` once a token's last slot lands.
Padding rows of ``dL_dswiglu_in`` hold NaN; the predicated scatter must skip
them. Guard rows around the output catch a scatter to ``r = -1``, ``r = T_recv``,
or past column ``H`` of the last row, including an add of zeros.
"""

import pytest
import torch

from stream_ep.stream_moe.kernel_a_bwd import streaming_moe_a_bwd
from stream_ep_tests.kernels._common import (
    KERNEL_SHAPES,
    LAYOUTS,
    TILE_M,
    assert_guards_untouched,
    configs,
    expert_weights,
    fired,
    guarded_output,
    per_expert,
    randn,
)
from stream_ep_tests.utils import assert_rows_close

RTOL = 1e-2
SEQ = 9


@pytest.mark.parametrize("num_sms", [None, 8])
@pytest.mark.parametrize("config", ["default", "alternate"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", KERNEL_SHAPES)
def test_kernel_a_bwd(shape, layout, config, num_sms):
    (H, I), layout = KERNEL_SHAPES[shape], LAYOUTS[layout]
    num_recv = layout.num_slots // 3
    real = layout.real()
    grad_in = randn(layout.num_slots, 2 * I, seed=5)
    grad_in[~real] = float("nan")
    w1 = expert_weights(layout, 2 * I, H, seed=6)
    recv = layout.recv_tokens(num_recv, seed=1)

    guarded, dL_dx = guarded_output(num_recv, H)
    k_local = torch.bincount(recv[real].long(), minlength=num_recv).to(torch.int32)
    remaining, done = k_local.clone(), torch.zeros(num_recv, dtype=torch.int64, device="cuda")
    target = layout.arrival_target()
    tiles = grad_in.view(layout.total_tiles, TILE_M, 2 * I)
    streaming_moe_a_bwd(
        tiles,
        w1,
        dL_dx,
        recv,
        remaining,
        done,
        layout.block_offset(),
        fired(target),
        target,
        SEQ,
        tile_m=TILE_M,
        num_sms=num_sms,
        **configs("a_bwd", H, I)[config],
    )
    torch.cuda.synchronize()

    per_slot = per_expert(layout, grad_in, w1, lambda g, w: g @ w)
    expected = torch.zeros(num_recv, H, device="cuda").index_add_(0, recv[real].long(), per_slot[real])
    assert_rows_close(dL_dx, expected, RTOL, "dL_dx_per_r")
    assert_guards_untouched(guarded, "dL_dx_per_r")
    assert not remaining.any(), "a token's remaining count did not reach zero"
    assert torch.equal(done, torch.where(k_local > 0, SEQ, 0)), "done stamps differ from tokens with slots"
