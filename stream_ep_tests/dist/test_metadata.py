"""The metadata kernel's outputs against a torch reference from the gathered routing.

Covers what ``test_dispatch`` does not: the per-(channel, source, expert)
counts and pool offsets the receivers allocate slots from (``base_pool``,
NVL-local sources before RDMA-remote ones on the internode path), the
receive-side rank prefix, and the internode channel and rank prefix sums.

On the internode path the metadata kernel's NVL slabs must not alias the
dispatch data ring: a leading peer's dispatch would overwrite a lagging
peer's metadata inbox mid-read and corrupt ``total_tiles``. This file delays
metadata Phase B on one rank per node (``STREAMEP_DEBUG_PB_DELAY_US``, read
once per process, so it applies to this file's session only) and issues
dispatches back to back, which opens that race window on every dispatch.
"""

import os

import pytest
import torch

from stream_ep import Buffer
from stream_ep_tests import routing
from stream_ep_tests.reference import Global
from stream_ep_tests.utils import DTYPE, H

os.environ.setdefault("STREAMEP_DEBUG_PB_DELAY_US", "20000")

TILE_M = 128
NUM_NVL_PEERS = 8


def expected_metadata(inputs: Global, num_channels: int, internode: bool) -> dict[str, torch.Tensor]:
    W, E_local = inputs.group.size(), inputs.num_local_experts
    device = inputs.x.device
    counts = torch.tensor(inputs.offsets, device=device).diff()
    src = torch.repeat_interleave(torch.arange(W, device=device), counts)  # [N] source rank
    t = torch.arange(inputs.x.shape[0], device=device) - torch.tensor(inputs.offsets[:-1], device=device)[src]
    channel = torch.minimum(t // ((counts[src] + num_channels - 1) // num_channels), torch.tensor(num_channels - 1))

    local_topk = inputs.local_topk()
    token, k = (local_topk >= 0).nonzero(as_tuple=True)
    seen = torch.zeros(num_channels, W, E_local, dtype=torch.long, device=device)
    seen.index_put_((channel[token], src[token], local_topk[token, k]), torch.ones_like(token), accumulate=True)

    freq = seen.sum(dim=(0, 1))
    blocks = torch.cat([freq.new_zeros(1), torch.cumsum((freq + TILE_M - 1) // TILE_M, 0)])
    # Receivers take each expert's slots substream by substream in (channel, source) order;
    # internode, the NVL-local sources come first.
    c_idx, s_idx = torch.meshgrid(torch.arange(num_channels, device=device), torch.arange(W, device=device), indexing="ij")
    remote = (s_idx // NUM_NVL_PEERS != inputs.rank // NUM_NVL_PEERS) if internode else torch.zeros_like(s_idx, dtype=torch.bool)
    order = (remote.long() * num_channels * W + c_idx * W + s_idx).flatten().argsort()
    flat = seen.view(num_channels * W, E_local)[order]
    base = torch.empty_like(flat)
    base[order] = blocks[:-1] * TILE_M + flat.cumsum(0) - flat

    unique_per_src = torch.zeros(W, dtype=torch.long, device=device).index_add_(0, src, (local_topk >= 0).any(1).long())
    out = {
        "seen_per_substream": seen,
        "base_pool": base.view(num_channels, W, E_local),
        "rank_prefix_column": unique_per_src.cumsum(0),
    }
    if internode:
        num_nodes = W // NUM_NVL_PEERS
        owner = torch.where(inputs.topk_idx >= 0, inputs.topk_idx // E_local, -1)
        in_rank = (owner.unsqueeze(-1) == torch.arange(W, device=device)).any(1)  # [N, W]
        in_node = in_rank.view(-1, num_nodes, NUM_NVL_PEERS).any(-1)  # [N, nodes]
        mine = inputs.local_rows(torch.arange(inputs.x.shape[0], device=device))
        per_channel = torch.zeros(num_channels, W, dtype=torch.long, device=device).index_add_(0, channel[mine], in_rank[mine].long())
        per_channel_node = torch.zeros(num_channels, num_nodes, dtype=torch.long, device=device).index_add_(0, channel[mine], in_node[mine].long())
        # Over RDMA this rank receives, from the rank with its NVL index on each
        # node, every token bound for any rank on this node.
        node, nvl = divmod(inputs.rank, NUM_NVL_PEERS)
        from_peers = torch.stack(
            [in_node[inputs.offsets[n * NUM_NVL_PEERS + nvl] : inputs.offsets[n * NUM_NVL_PEERS + nvl + 1], node].sum() for n in range(num_nodes)]
        )
        out |= {
            "gbl_channel_prefix_matrix": per_channel.cumsum(0).T,
            "rdma_channel_prefix_matrix": per_channel_node.cumsum(0).T,
            "recv_gbl_rank_prefix_sum": unique_per_src.cumsum(0),
            "recv_rdma_rank_prefix_sum": from_peers.cumsum(0),
        }
    return out


def actual_metadata(handle, rank: int, internode: bool) -> dict[str, torch.Tensor]:
    out = {
        "seen_per_substream": handle.seen_per_substream,
        "base_pool": handle.base_pool,
        "rank_prefix_column": handle.rank_prefix_matrix[:, rank],
    }
    if internode:
        d = handle._dispatch_out
        out |= {name: getattr(d, name) for name in ("gbl_channel_prefix_matrix", "rdma_channel_prefix_matrix", "recv_gbl_rank_prefix_sum", "recv_rdma_rank_prefix_sum")}
    return out


def check(handle, inputs: Global, internode: bool):
    expected = expected_metadata(inputs, Buffer.num_sms // 2, internode)
    for name, actual in actual_metadata(handle, inputs.rank, internode).items():
        assert actual.shape == expected[name].shape, f"{name}: shape {tuple(actual.shape)} != {tuple(expected[name].shape)}"
        assert torch.equal(actual.long(), expected[name]), f"{name}: {(actual.long() != expected[name]).sum().item()} mismatches"


@pytest.mark.parametrize("scenario", routing.SCENARIOS)
@pytest.mark.parametrize("num_sms", [None, 64])
def test_metadata(buffer, group, scenario, num_sms):
    """Back-to-back dispatches with fresh routing, checked after the last one."""
    device = torch.device("cuda", torch.cuda.current_device())
    saved = Buffer.num_sms
    Buffer.num_sms = num_sms or saved
    try:
        runs = []
        for seed in range(4):
            r = routing.SCENARIOS[scenario](group.rank(), group.size(), device, seed=seed)
            x = torch.randn(r.num_tokens, H, dtype=DTYPE, device=device)
            _, handle, _ = buffer.dispatch(x, *r.dispatch_args(), r.num_experts, tile_m=TILE_M)
            runs.append((handle, x, r))
        torch.cuda.synchronize()
        # Gather everything before checking: a rank that fails a check must not skip a later gather.
        gathered = [(handle, Global.gather(x, r, group)) for handle, x, r in runs]
        for handle, inputs in gathered:
            check(handle, inputs, buffer._is_internode())
    finally:
        Buffer.num_sms = saved
