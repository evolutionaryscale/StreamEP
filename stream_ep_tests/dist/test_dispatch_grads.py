"""``Buffer.dispatch_grads`` (backward dispatch) against the forward layout.

``dL/dy`` rows are random, so each pool slot of ``dL_do_pool`` must hold
exactly its source token's row: dispatch_grads ships ``dL/dy[t]`` along the
forward routing and fans it into every slot the forward dispatch gave
``(t, k)``. Every tile's backward arrival count must reach the forward target.
"""

import pytest
import torch

from stream_ep_tests import routing
from stream_ep_tests.dist.test_dispatch import dispatch
from stream_ep_tests.reference import all_gather_rows
from stream_ep_tests.utils import SHAPES


@pytest.mark.parametrize("scenario", routing.SCENARIOS)
@pytest.mark.parametrize("shape", SHAPES)
def test_dispatch_grads(buffers, group, shape, scenario):
    buffer = buffers(shape)
    _, handle, inputs = dispatch(buffer, group, shape, scenario)
    dL_dy = torch.randn(inputs.local_rows(inputs.x).shape, dtype=inputs.x.dtype, device=inputs.x.device)

    dL_do_pool, arrival_count, _ = buffer.dispatch_grads(handle, dL_dy)
    torch.cuda.synchronize()

    assert dL_do_pool.shape == (handle.TK_padded, dL_dy.shape[1])
    real = (handle.pool_recv_token >= 0).nonzero().flatten()
    source = inputs.recv_tokens()[handle.pool_recv_token[real].long()]
    dL_dy_all, _ = all_gather_rows(dL_dy, group)
    assert torch.equal(dL_do_pool[real], dL_dy_all[source]), "dL_do_pool rows differ from their source gradients"
    assert torch.equal(arrival_count[: handle.total_tiles], handle.pool_arrival_target)
