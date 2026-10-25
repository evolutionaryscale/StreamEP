"""``Buffer.combine`` and ``Buffer.combine_grads`` against a gathered reference.

Kernel Y and kernel_a_bwd are not run: their outputs (``handle.o``,
``dL_dx_per_r``, the per-slot weight gradients) are filled with random rows and
their per-token release stamps are pre-set, so only the reduction back to the
source ranks is checked. Random rows make every contribution distinguishable,
so a row read from the wrong receive slot or rank shows up as a mismatch.
"""

import pytest
import torch

from stream_ep_tests import routing
from stream_ep_tests.dist.test_dispatch import dispatch
from stream_ep_tests.utils import SHAPES, assert_sums_close



def rounding_ulps(group) -> float:
    """Combine sums each token's rows in fp32 and rounds once to bf16. Internode,
    each node's forwarder first rounds its partial sum, so the receiver adds one
    rounding per node. Each rounding is at most half an ulp of a partial sum,
    which the sum of absolute values bounds. A row taken from the wrong slot or
    rank is off by O(1), hundreds of ulps."""
    num_nodes = (group.size() + 7) // 8
    return 0.5 if num_nodes == 1 else 0.5 * (num_nodes + 1)


@pytest.mark.parametrize("scenario", routing.SCENARIOS)
@pytest.mark.parametrize("shape", SHAPES)
def test_combine(buffers, group, shape, scenario):
    buffer = buffers(shape)
    _, handle, inputs = dispatch(buffer, group, shape, scenario)
    handle.o.copy_(torch.randn_like(handle.o))
    handle.y_done_per_token.fill_(handle.dispatch_seq)

    out, weights_out = buffer.combine(handle.o, handle)
    torch.cuda.synchronize()

    expected, abs_sum = inputs.to_sources(handle.o), inputs.to_sources(handle.o.abs())
    assert weights_out is None
    assert_sums_close(out, expected, abs_sum, rounding_ulps(group), "out")


@pytest.mark.parametrize("scenario", routing.SCENARIOS)
@pytest.mark.parametrize("shape", SHAPES)
def test_combine_grads(buffers, group, shape, scenario):
    buffer = buffers(shape)
    _, handle, inputs = dispatch(buffer, group, shape, scenario)
    dL_dx_per_r = torch.randn_like(handle.o)
    weight_grads = torch.randn(handle.TK_padded, dtype=torch.float32, device=handle.o.device)

    done = torch.full((handle.o.shape[0],), handle.dispatch_seq, dtype=torch.int64, device=handle.o.device)

    dL_dx, dL_dtopk = buffer.combine_grads(dL_dx_per_r, handle, weight_grads, done)
    torch.cuda.synchronize()

    # Each (token, k) lives on exactly one rank, which sends its slot's gradient.
    slots = handle.recv_token_to_slots.long()
    expected_dtopk = inputs.to_sources(torch.where(slots >= 0, weight_grads[slots.clamp(min=0)], 0.0))
    expected, abs_sum = inputs.to_sources(dL_dx_per_r), inputs.to_sources(dL_dx_per_r.abs())

    assert_sums_close(dL_dx, expected, abs_sum, rounding_ulps(group), "dL_dx")
    assert torch.equal(dL_dtopk, expected_dtopk)
