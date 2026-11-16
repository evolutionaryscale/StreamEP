"""Kernel Y (down-projection + weighted atomic scatter into ``o``) against a torch reference.

``o[r] = sum over slots s with pool_recv_token[s] == r of w[s] * (postact[s] @ W2[e(s)].T)``.
Each receive token gets several slots, and once a token's last slot lands the
kernel stamps ``y_done_per_token[r] = combine_seq`` for combine. Padding rows
of ``postact`` hold NaN with weight 0, as an uninitialized pool would; the
predicated scatter must skip them rather than add ``0 * NaN``. Guard rows
around ``o`` catch a scatter to ``r = -1``, ``r = T_recv``, or past column ``H``
of the last row, including an add of zeros.
"""

import pytest
import torch

from stream_ep.stream_moe.kernel_y import scatter_num_pid_n, streaming_moe_y
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
SEQ = 7


@pytest.mark.parametrize("num_sms", [None, 8])
@pytest.mark.parametrize("config", ["default", "alternate"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", KERNEL_SHAPES)
def test_kernel_y(shape, layout, config, num_sms):
    (H, I), layout = KERNEL_SHAPES[shape], LAYOUTS[layout]
    num_recv = layout.num_slots // 3
    real = layout.real()
    postact = randn(layout.num_slots, I, seed=3)
    postact[~real] = float("nan")
    w2 = expert_weights(layout, H, I, seed=4)
    recv = layout.recv_tokens(num_recv)
    weight = torch.where(real, torch.rand(layout.num_slots, device="cuda"), 0.0)

    guarded, o = guarded_output(num_recv, H)
    k_local = torch.bincount(recv[real].long(), minlength=num_recv).to(torch.int32)
    remaining, done = k_local.clone(), torch.zeros(num_recv, dtype=torch.int64, device="cuda")
    target = layout.arrival_target()
    tiles = postact.view(layout.total_tiles, TILE_M, I)
    streaming_moe_y(
        tiles,
        w2,
        o,
        recv,
        weight,
        remaining,
        done,
        layout.block_offset(),
        fired(target),
        target,
        combine_seq=SEQ,
        tile_m=TILE_M,
        num_sms=num_sms,
        **configs("y", H, I)[config],
    )
    torch.cuda.synchronize()

    y = per_expert(layout, postact, w2, lambda a, w: a @ w.T)
    expected = torch.zeros(num_recv, H, device="cuda").index_add_(0, recv[real].long(), weight[real, None] * y[real])
    assert_rows_close(o, expected, RTOL, "o")
    assert_guards_untouched(guarded, "o")
    assert not remaining.any(), "a token's remaining count did not reach zero"
    assert torch.equal(done, torch.where(k_local > 0, SEQ, 0)), "done stamps differ from tokens with slots"


def test_scatter_rejects_stripes_past_h():
    # H = 768 at tile_n 256 is 3 stripes; a 2-CTA cluster would add a 4th past H.
    assert scatter_num_pid_n(768, 256, 1) == 3
    assert scatter_num_pid_n(768, 128, 2) == 6
    with pytest.raises(AssertionError, match="cluster_n"):
        scatter_num_pid_n(768, 256, 2)
    with pytest.raises(AssertionError, match="tile_n"):
        scatter_num_pid_n(768, 512, 1)
