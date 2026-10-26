"""Activation-checkpoint level 2's pool <-> x_recv kernels (``recv_pool_compress``).

Level 2 saves each received token once (``x_recv``) instead of the expanded
pool, where a token's row repeats in every slot it routes to, and re-expands
it in backward. The round trip must reproduce every real slot bit for bit,
never read padding into ``x_recv``, and leave padding slots of the re-expanded
pool exactly zero whatever ``x_recv`` holds.
"""

import pytest
import torch

from stream_ep.stream_moe.recv_pool_compress import compress_pool_to_recv, reexpand_recv_to_pool
from stream_ep_tests.utils import DTYPE


def broadcast_pool(num_tokens, slots_per_token, num_padding, H, seed=0):
    """(pool, pool_recv_token, x_recv): slot order shuffled, padding slots garbage."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x_recv = torch.randn(num_tokens, H, generator=g, device="cuda").to(DTYPE)
    tokens = torch.repeat_interleave(torch.arange(num_tokens, device="cuda"), torch.tensor(slots_per_token, device="cuda"))
    recv = torch.cat([tokens, tokens.new_full((num_padding,), -1)])
    recv = recv[torch.randperm(recv.numel(), generator=g, device="cuda")].to(torch.int32)
    pool = torch.randn(recv.numel(), H, generator=g, device="cuda").to(DTYPE)
    pool[recv >= 0] = x_recv[recv[recv >= 0].long()]
    return pool, recv, x_recv


CASES = {
    "duplicates_and_padding": (200, [1 + t % 3 for t in range(200)], 137, 1024),
    "identity": (64, [1] * 64, 0, 512),
    "partial_block": (50, [2] * 50, 20, 513),  # H % BLOCK_H != 0
}


@pytest.mark.parametrize("case", CASES)
def test_round_trip(case):
    pool, recv, x_recv = broadcast_pool(*CASES[case])
    real = recv >= 0

    compressed = torch.full_like(x_recv, float("nan"))
    compress_pool_to_recv(pool, recv, compressed)
    expanded = torch.full_like(pool, float("nan"))
    reexpand_recv_to_pool(compressed, recv, expanded)

    assert torch.equal(compressed, x_recv), "compress did not reproduce each token's row"
    assert torch.equal(expanded[real], pool[real]), "re-expanded real slots differ"
    assert not expanded[~real].any(), "re-expanded padding slots not zero"


def test_padding_zero_with_inf_in_x_recv():
    pool, recv, x_recv = broadcast_pool(*CASES["duplicates_and_padding"])
    x_recv[0] = float("inf")
    expanded = torch.full_like(pool, float("nan"))
    reexpand_recv_to_pool(x_recv, recv, expanded)
    assert not expanded[recv < 0].any(), "re-expanded padding slots not zero"
