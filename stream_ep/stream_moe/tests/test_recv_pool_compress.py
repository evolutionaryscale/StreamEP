"""Round-trip tests for the activation_checkpoint_level=2 pool <-> x_recv triton
kernels (``recv_pool_compress``).

Builds the layout level 2 actually sees: a *broadcast* pool (all of a token's
slots hold the identical row) + a shuffled slot->token map with duplicate slots
per token and -1 padding slots. Verifies:
  - compact(pool) -> x_recv reproduces every token's row bitwise (the benign
    identical-duplicate scatter),
  - reexpand(x_recv) -> pool reproduces the real (non-padding) rows bitwise and
    zeros the padding rows,
  - H both a multiple and a non-multiple of BLOCK_H (mask path).
"""
from __future__ import annotations

import pytest
import torch

from stream_ep.stream_moe.recv_pool_compress import (
    compress_pool_to_recv,
    reexpand_recv_to_pool,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def _build(T_recv, slots_per_token, n_pad, H, device, dtype=torch.bfloat16):
    """(pool, pool_recv_token, base, real_mask) with the broadcast invariant.

    ``slots_per_token[t]`` >= 1 real slots for token t; ``n_pad`` -1 padding
    slots; the whole slot order is shuffled. pool[s] = base[token(s)] on real
    slots, garbage on padding (to prove padding never leaks into x_recv).
    """
    assert len(slots_per_token) == T_recv and all(k >= 1 for k in slots_per_token)
    base = torch.randn(T_recv, H, dtype=dtype, device=device)
    tokens = []
    for t, k in enumerate(slots_per_token):
        tokens += [t] * k
    tokens += [-1] * n_pad
    perm = torch.randperm(len(tokens), device=device)
    idx = torch.tensor(tokens, dtype=torch.int32, device=device)[perm].contiguous()
    real = idx >= 0
    pool = torch.empty(idx.numel(), H, dtype=dtype, device=device)
    pool[real] = base[idx[real].long()]
    pool[~real] = torch.randn(int((~real).sum()), H, dtype=dtype, device=device)
    return pool, idx, base, real


@pytest.mark.parametrize("H", [512, 2048])
def test_compact_reexpand_roundtrip(device, H):
    torch.manual_seed(0)
    T_recv = 200
    slots = [1 + (t % 3) for t in range(T_recv)]  # 1..3 slots/token (duplicates)
    pool, idx, base, real = _build(T_recv, slots, n_pad=137, H=H, device=device)

    x_recv = torch.empty(T_recv, H, dtype=pool.dtype, device=device)
    compress_pool_to_recv(pool, idx, x_recv)
    assert torch.equal(x_recv, base), "compact did not reproduce per-token rows"

    pool2 = torch.empty_like(pool)
    reexpand_recv_to_pool(x_recv, idx, pool2)
    assert torch.equal(pool2[real], pool[real]), "reexpand real rows differ"
    assert (pool2[~real] == 0).all(), "reexpand padding rows not zeroed"


def test_identity_no_padding(device):
    # Single slot per token, no padding -> round-trip is the identity on all rows.
    torch.manual_seed(1)
    T_recv, H = 64, 512
    pool, idx, base, real = _build(T_recv, [1] * T_recv, n_pad=0, H=H, device=device)
    x_recv = torch.empty(T_recv, H, dtype=pool.dtype, device=device)
    compress_pool_to_recv(pool, idx, x_recv)
    assert torch.equal(x_recv, base)
    pool2 = torch.empty_like(pool)
    reexpand_recv_to_pool(x_recv, idx, pool2)
    assert torch.equal(pool2, pool)


def test_H_not_multiple_of_block(device):
    # H % BLOCK_H != 0 exercises the partial-block mask.
    torch.manual_seed(2)
    T_recv, H = 50, 513
    pool, idx, base, real = _build(T_recv, [2] * T_recv, n_pad=20, H=H, device=device)
    x_recv = torch.empty(T_recv, H, dtype=pool.dtype, device=device)
    compress_pool_to_recv(pool, idx, x_recv)
    assert torch.equal(x_recv, base)
    pool2 = torch.empty_like(pool)
    reexpand_recv_to_pool(x_recv, idx, pool2)
    assert torch.equal(pool2[real], pool[real])
    assert (pool2[~real] == 0).all()
