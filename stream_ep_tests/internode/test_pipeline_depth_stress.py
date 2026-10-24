"""The full layer chained at production depth across nodes.

Each step runs 44 residual layers forward and backward on one Buffer, so 44
``dispatch_grads`` / ``combine_grads`` run back to back between the backward
compute kernels, and every step reuses the rings, as in a deep model. The
first, middle and last layers of the final step are checked against the
reference on their own inputs.
"""

import pytest

from stream_ep_tests.dist.test_pipeline import check_chain

pytestmark = pytest.mark.usefixtures("internode_only")

NUM_LAYERS = 44


def test_pipeline_depth(buffer, streams, group):
    check_chain(
        buffer, streams, group, NUM_LAYERS, num_steps=3, checked_layers=[0, NUM_LAYERS // 2, NUM_LAYERS - 1], activation_checkpoint_level=1
    )
