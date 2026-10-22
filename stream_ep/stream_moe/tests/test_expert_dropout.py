"""Expert dropout in kernel A (fwd) and kernel_y_bwd against a torch reference
that regenerates the same mask with ``keep_scale_ref``.

Zero patterns are checked exactly (a dropped unit is exactly 0 in every
masked output, a kept one never is); values to bf16 tolerance.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from stream_ep.stream_moe.expert_dropout import keep_scale_ref
from stream_ep.stream_moe.tests.test_kernel_a import (
    _make_pool_arrival,
    _make_tile_metadata,
)

P = 0.2
H, I, E_LOCAL, TILE_M = 512, 512, 3, 128
TILES = [0, 0, 1, 2, 2]
TK = len(TILES) * TILE_M


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def _meta(device):
    _, block_offset = _make_tile_metadata(TILES, E_LOCAL, device)
    count, target = _make_pool_arrival(len(TILES), device=device)
    return block_offset, count, target


def _mask(seed, device):
    rows = torch.arange(TK, device=device)
    return keep_scale_ref(seed, rows, torch.arange(I, device=device), P)


def _assert_close(out, ref, what, rtol=3e-2, atol=3e-3):
    bad = (out.float() - ref.float()).abs() > atol + rtol * ref.float().abs()
    assert not bad.any(), f"{what}: {bad.sum().item()} / {bad.numel()} out of tolerance"


def test_kernel_a_expert_dropout(device):
    from stream_ep.stream_moe.kernel_a import streaming_moe_a

    torch.manual_seed(0)
    pool = torch.randn(TK, H, dtype=torch.bfloat16, device=device)
    W1 = torch.randn(E_LOCAL, 2 * I, H, dtype=torch.bfloat16, device=device) * 0.05

    def run(p, seed=None):
        postact = torch.empty(len(TILES), TILE_M, I, dtype=torch.bfloat16, device=device)
        preact = torch.empty(len(TILES), TILE_M, 2 * I, dtype=torch.bfloat16, device=device)
        streaming_moe_a(
            pool, W1, postact, *_meta(device), preact_a=preact, tile_m=TILE_M,
            tile_n=256, cluster_n=2, dropout_p=p, dropout_seed=seed,
        )
        torch.cuda.synchronize()
        return postact.view(TK, I), preact.view(TK, 2 * I)

    post0, pre0 = run(0.0)
    post1, pre1 = run(P, seed=1234)
    post2, _ = run(P, seed=1235)
    assert torch.equal(pre1, pre0), "preact must stay unmasked"

    d = _mask(1234, device)
    kept = d > 0
    assert abs(kept.float().mean().item() - (1 - P)) < 5e-3
    assert torch.equal(post1 != 0, kept & (post0 != 0)), "mask mismatch vs reference"
    assert not torch.equal(post1 != 0, post2 != 0), "seed must change the mask"
    ref = F.silu(pre0[:, 0::2].float()) * pre0[:, 1::2].float()
    _assert_close(post0, ref, "postact p=0")
    _assert_close(post1, ref * d, "postact p>0")


def test_kernel_y_bwd_expert_dropout(device):
    from stream_ep.stream_moe.kernel_y_bwd import streaming_moe_y_bwd

    torch.manual_seed(1)
    dL_do_pool = torch.randn(TK, H, dtype=torch.bfloat16, device=device)
    W2 = torch.randn(E_LOCAL, H, I, dtype=torch.bfloat16, device=device) * 0.05
    preact = torch.randn(len(TILES), TILE_M, 2 * I, dtype=torch.bfloat16, device=device)
    w = torch.rand(TK, dtype=torch.float32, device=device)
    recv = torch.arange(TK, dtype=torch.int32, device=device)
    recv[TILE_M + 90 : 2 * TILE_M] = -1  # expert 0's padding tail
    recv[-17:] = -1

    def run(p, seed=None):
        dswiglu_in = torch.empty_like(preact)
        post_dW2 = torch.empty(len(TILES), TILE_M, I, dtype=torch.bfloat16, device=device)
        dweight = torch.zeros(TK, dtype=torch.float32, device=device)
        streaming_moe_y_bwd(
            dL_do_pool, W2, dswiglu_in, post_dW2, w, recv, preact, dweight,
            *_meta(device), tile_m=TILE_M, tile_n=128, dropout_p=p, dropout_seed=seed,
        )
        torch.cuda.synchronize()
        return dswiglu_in.view(TK, 2 * I), post_dW2.view(TK, I), dweight

    expert = torch.tensor(TILES, device=device).repeat_interleave(TILE_M)
    g = torch.bmm(dL_do_pool.float().unsqueeze(1), W2.float()[expert]).squeeze(1)
    gate, up = preact.view(TK, 2 * I)[:, 0::2].float(), preact.view(TK, 2 * I)[:, 1::2].float()
    h = F.silu(gate) * up
    active = (recv >= 0).float()[:, None]

    def ref(d):
        dout = w[:, None] * d * g
        sig = torch.sigmoid(gate)
        dgate = sig * (1 + gate * (1 - sig)) * up * dout
        dup = F.silu(gate) * dout
        dswiglu_in = torch.stack([dgate, dup], dim=-1).view(TK, 2 * I)
        return dswiglu_in * active, w[:, None] * d * h * active, (d * h * g).sum(-1) * active[:, 0]

    for p, seed in ((0.0, None), (P, 99)):
        d = _mask(seed, device) if p > 0 else torch.ones(TK, I, device=device)
        out, out_ref = run(p, seed), ref(d)
        for o, r, name in zip(out, out_ref, ("dswiglu_in", "postact_for_dW2", "dweight")):
            _assert_close(o, r, f"{name} p={p}", atol=3e-2 if name == "dweight" else 3e-3)
        if p > 0:
            dropped = (d == 0) | (active == 0)
            assert (out[1][dropped] == 0).all() and (out[0][:, 0::2][dropped] == 0).all()
            assert (out[1][~dropped] != 0).float().mean() > 0.99
