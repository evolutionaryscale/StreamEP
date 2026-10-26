"""The full layer chained deep across nodes.

Each step runs 24 residual layers forward and backward on one Buffer, so 24
``dispatch_grads`` / ``combine_grads`` run back to back between the backward
compute kernels, and every step reuses the rings, as in a deep model. The
first, middle and last layers of the final step are checked against the
reference on their own inputs.
"""

import pytest

from stream_ep_tests.dist.test_pipeline import check_chain
from stream_ep_tests.utils import SHAPES

pytestmark = pytest.mark.usefixtures("internode_only")

NUM_LAYERS = 24


@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("shape", ["small", "odd_topk"])
def test_pipeline_depth(buffers, streams, group, shape, level):
    checked = [0, NUM_LAYERS // 2, NUM_LAYERS - 1]
    check_chain(
        buffers(shape), streams, group, SHAPES[shape], NUM_LAYERS, num_steps=3, checked_layers=checked, activation_checkpoint_level=level
    )
