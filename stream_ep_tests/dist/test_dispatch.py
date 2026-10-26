"""``Buffer.dispatch`` against the layout the gathered routing implies.

Tokens are random rows, so a pool row equal to its source row pins down
exactly which token landed in which slot. Every routing scenario checks:

- each real slot holds its (token, k)'s data and topk weight, inside its
  expert's tile-aligned block, and every local (token, k) has exactly one slot;
- padding rows are marked ``pool_recv_token == -1``;
- the per-expert counts, block offsets, per-tile expert and arrival targets,
  and the per-token maps (``recv_token_to_slots``, ``k_local_total`` /
  ``k_local_remaining``, ``seen_per_substream``);
- every tile's arrival count reached its target, and the combine-side
  buffers start at zero.
"""

import pytest
import torch

from stream_ep_tests import routing
from stream_ep_tests.reference import Global
from stream_ep_tests.utils import DTYPE, SHAPES

TILE_M = 128


def dispatch(buffer, group, shape, scenario, seed=0, config=None):
    device = torch.device("cuda", torch.cuda.current_device())
    r = routing.SCENARIOS[scenario](SHAPES[shape], group.rank(), group.size(), device, seed=seed)
    x = torch.randn(r.num_tokens, SHAPES[shape].hidden, dtype=DTYPE, device=device)
    pool, handle, _ = buffer.dispatch(x, *r.dispatch_args(), r.num_experts, tile_m=TILE_M, config=config)
    torch.cuda.synchronize()
    return pool, handle, Global.gather(x, r, group)


def check_layout(pool, handle, inputs: Global):
    recv_ids = inputs.recv_tokens()
    local_topk = inputs.local_topk()[recv_ids]  # [T_recv, K]
    T_recv, E_local = recv_ids.numel(), inputs.num_local_experts
    assert handle.o.shape[0] == T_recv, f"T_recv {handle.o.shape[0]} != expected {T_recv}"

    freq = torch.bincount(local_topk[local_topk >= 0], minlength=E_local)
    assert torch.equal(handle.expert_frequency.long(), freq)
    blocks = torch.cat([freq.new_zeros(1), torch.cumsum((freq + TILE_M - 1) // TILE_M, 0)])
    assert torch.equal(handle.expert_pool_block_offset.long(), blocks)
    assert handle.total_tiles == blocks[-1].item()
    assert pool.shape[0] == handle.TK_padded == handle.total_tiles * TILE_M

    slot_expert = torch.repeat_interleave(torch.arange(E_local, device=pool.device), blocks.diff() * TILE_M)
    real = (handle.pool_recv_token >= 0).nonzero().flatten()
    r, k = handle.pool_recv_token[real].long(), handle.pool_k_slot[real].long()
    assert torch.equal(local_topk[r, k], slot_expert[real]), "slot outside its expert's block"
    assert torch.equal(pool[real], inputs.x[recv_ids[r]]), "pool rows differ from their source tokens"
    assert torch.equal(handle.pool_topk_weight[real], inputs.topk_weights[recv_ids[r], k])

    expected_slots = torch.full_like(local_topk, -1)
    expected_slots[r, k] = real
    assert torch.equal(handle.recv_token_to_slots.long(), expected_slots), "not exactly one slot per local (token, k)"
    assert torch.equal(handle.k_local_total.long(), (local_topk >= 0).sum(1))
    assert torch.equal(handle.k_local_remaining, handle.k_local_total)
    assert torch.equal(handle.seen_per_substream.sum(dim=(0, 1)).long(), freq)

    tile_expert = torch.repeat_interleave(torch.arange(E_local, device=pool.device), blocks.diff())
    assert torch.equal(handle.tile_id_to_expert.long(), tile_expert)
    per_tile = torch.bincount(real // TILE_M, minlength=handle.total_tiles)
    assert torch.equal(handle.pool_arrival_target.long(), per_tile)
    assert torch.equal(handle.pool_arrival_count[: handle.total_tiles], handle.pool_arrival_target)
    assert not handle.o.any() and not handle.y_done_per_token.any()


@pytest.mark.parametrize("scenario", routing.SCENARIOS)
@pytest.mark.parametrize("shape", SHAPES)
def test_layout(buffers, group, shape, scenario):
    check_layout(*dispatch(buffers(shape), group, shape, scenario))


@pytest.mark.parametrize("shape", SHAPES)
def test_back_to_back_dispatches(buffers, group, shape):
    """Fresh routing each call on one Buffer: no state leaks between dispatches."""
    for seed in range(4):
        check_layout(*dispatch(buffers(shape), group, shape, "sentinels", seed=seed))


def test_placement_is_deterministic(buffers, group):
    _, a, _ = dispatch(buffers("odd_topk"), group, "odd_topk", "power_law", seed=7)
    _, b, _ = dispatch(buffers("odd_topk"), group, "odd_topk", "power_law", seed=7)
    assert torch.equal(a.pool_recv_token, b.pool_recv_token)
    assert torch.equal(a.pool_k_slot, b.pool_k_slot)
    assert torch.equal(a.recv_token_to_slots, b.recv_token_to_slots)
