"""Back-to-back internode dispatches and dispatch_grads, communication only.

Each step issues 44 forward dispatches with fresh routing, keeping every
handle alive as autograd would, then 44 ``dispatch_grads`` in reverse with no
compute in between: less slack between collectives than the real pipeline
gives. The first, middle and last layers of the last step must deliver each
source token's gradient row to every slot forward dispatch gave it.
"""

import pytest
import torch

from stream_ep_tests import routing
from stream_ep_tests.reference import Global
from stream_ep_tests.utils import DTYPE, H

pytestmark = pytest.mark.usefixtures("internode_only")

NUM_LAYERS = 44
NUM_STEPS = 3
CHECKED = (0, NUM_LAYERS // 2, NUM_LAYERS - 1)


def test_dispatch_grads_stress(buffer, group):
    device = torch.device("cuda", torch.cuda.current_device())
    for step in range(NUM_STEPS):
        layers = []
        for layer in range(NUM_LAYERS):
            r = routing.uniform(group.rank(), group.size(), device, seed=step * NUM_LAYERS + layer)
            x = torch.randn(r.num_tokens, H, dtype=DTYPE, device=device)
            _, handle, _ = buffer.dispatch(x, *r.dispatch_args(), r.num_experts)
            layers.append((r, handle))
        grads = {}
        for layer in reversed(range(NUM_LAYERS)):
            r, handle = layers[layer]
            dL_dy = torch.randn(r.num_tokens, H, dtype=DTYPE, device=device)
            dL_do_pool, count, _ = buffer.dispatch_grads(handle, dL_dy)
            if layer in CHECKED:  # keep only what is checked: 44 live gradient pools exceed memory
                grads[layer] = (r, handle, dL_dy, dL_do_pool, count)
        torch.cuda.synchronize()

    # Every gather before any check, so a rank that fails cannot skip a later gather.
    failures = []
    for layer, (r, handle, dL_dy, dL_do_pool, count) in sorted(grads.items()):
        sources = Global.gather(dL_dy, r, group)  # gathers the gradient rows in place of tokens
        real = (handle.pool_recv_token >= 0).nonzero().flatten()
        rows = sources.x[sources.recv_tokens()[handle.pool_recv_token[real].long()]]
        if not torch.equal(dL_do_pool[real], rows):
            failures.append(f"layer {layer}: dL_do_pool rows differ from their source gradients")
        if not torch.equal(count[: handle.total_tiles], handle.pool_arrival_target):
            failures.append(f"layer {layer}: arrival counts short of target")
    assert not failures, "\n".join(failures)
