"""A ``Buffer`` keeps the token geometry and ring configs of its first call.

The internode rings' head/tail cursors persist across calls at offsets set by
the bytes per token, the top-k and the config, so a call that changes any of
them must raise before it touches the rings, and leave the Buffer usable at
its original geometry.
"""

import pytest
import torch

from stream_ep import Buffer, Config
from stream_ep_tests.dist.test_dispatch import check_layout, dispatch


def test_dispatch_rejects_new_token_geometry(buffers, group):
    dispatch(buffers("small"), group, "small", "uniform")
    with pytest.raises(ValueError, match="token geometry changed"):
        dispatch(buffers("small"), group, "odd_topk", "uniform")


def test_dispatch_rejects_new_config(buffers, group):
    first = Buffer.get_dispatch_config(group.size())
    fewer_sms = Config(
        first.num_sms - 2,
        first.num_max_nvl_chunked_send_tokens,
        first.num_max_nvl_chunked_recv_tokens,
        first.num_max_rdma_chunked_send_tokens,
        first.num_max_rdma_chunked_recv_tokens,
    )
    dispatch(buffers("small"), group, "small", "uniform")
    with pytest.raises(ValueError, match="dispatch config changed"):
        dispatch(buffers("small"), group, "small", "uniform", config=fewer_sms)


def test_combine_rejects_new_hidden(buffers, group):
    _, handle, _ = dispatch(buffers("small"), group, "small", "uniform")
    with pytest.raises(ValueError, match="token geometry changed"):
        buffers("small").combine(torch.cat([handle.o, handle.o], dim=1), handle)


def test_original_geometry_still_works(buffers, group):
    check_layout(*dispatch(buffers("small"), group, "small", "uniform"))
