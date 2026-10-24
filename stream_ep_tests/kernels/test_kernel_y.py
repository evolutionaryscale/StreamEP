"""Kernel Y (down-projection + weighted atomic scatter into ``o``) against a torch reference.

``o[r] = sum over slots s with pool_recv_token[s] == r of w[s] * (postact[s] @ W2[e(s)].T)``.
Each receive token gets several slots, and once a token's last slot lands the
kernel stamps ``y_done_per_token[r] = combine_seq`` for combine. Padding rows
of ``postact`` hold NaN with weight 0, as an uninitialized pool would; the
predicated scatter must skip them rather than add ``0 * NaN``. Guard rows
around ``o`` catch a scatter to ``r = -1`` or ``r = T_recv``.
"""

import pytest
import torch

from stream_ep.stream_moe.kernel_y import streaming_moe_y
from stream_ep_tests.kernels._common import LAYOUTS, TILE_M, expert_weights, fired, per_expert, randn
from stream_ep_tests.utils import DTYPE, H, I, assert_rows_close

# Production: tile_n_y=256 with 2-CTA clusters (stream_moe.py).
CONFIGS = {"production": {"tile_n": 256, "cluster_n": 2}, "alternate": {"tile_n": 128, "cluster_n": 1}}
RTOL = 1e-2
SEQ = 7


def run(layout, num_recv, config, num_sms=None):
    real = layout.real()
    postact = randn(layout.num_slots, I, seed=3)
    postact[~real] = float("nan")
    w2 = expert_weights(layout, H, I, seed=4)
    recv = layout.recv_tokens(num_recv)
    weight = torch.where(real, torch.rand(layout.num_slots, device="cuda"), 0.0)

    guarded = torch.zeros(num_recv + 2, H, dtype=DTYPE, device="cuda")
    o = guarded[1:-1]
    k_local = torch.bincount(recv[real].long(), minlength=num_recv).to(torch.int32)
    remaining, done = k_local.clone(), torch.zeros(num_recv, dtype=torch.int64, device="cuda")
    target = layout.arrival_target()
    tiles = postact.view(layout.total_tiles, TILE_M, I)
    block_offset = layout.block_offset()
    streaming_moe_y(
        tiles,
        w2,
        o,
        recv,
        weight,
        remaining,
        done,
        block_offset,
        fired(target),
        target,
        combine_seq=SEQ,
        tile_m=TILE_M,
        num_sms=num_sms,
        **CONFIGS[config],
    )
    torch.cuda.synchronize()

    y = per_expert(layout, postact, w2, lambda a, w: a @ w.T)
    expected = torch.zeros(num_recv, H, device="cuda").index_add_(0, recv[real].long(), weight[real, None] * y[real])
    assert_rows_close(o, expected, RTOL, "o")
    assert not guarded[0].any() and not guarded[-1].any(), "scatter wrote outside o"
    assert not remaining.any(), "a token's remaining count did not reach zero"
    assert torch.equal(done, torch.where(k_local > 0, SEQ, 0)), "done stamps differ from tokens with slots"


@pytest.mark.parametrize("num_sms", [None, 8])
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_kernel_y(layout, config, num_sms):
    layout = LAYOUTS[layout]
    run(layout, num_recv=layout.num_slots // 3, config=config, num_sms=num_sms)
