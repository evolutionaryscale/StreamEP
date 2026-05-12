from pathlib import Path

import torch

# Register stream_ep sources with quack's persistent .o cache fingerprint so
# that edits to stream_ep python files (kernel_*.py, tile_scheduler.py,
# epi_ops.py, ptx_helpers.py) invalidate the cache automatically. Without
# this, quack only hashes its own sources and stale .o files would mask
# stream_ep behavioural changes. Must run before any `jit_cache`-decorated
# function is called.
import quack.cache_utils as _quack_cache_utils

_STREAM_EP_PKG_ROOT = Path(__file__).resolve().parent
if _STREAM_EP_PKG_ROOT not in _quack_cache_utils.EXTRA_SOURCE_DIRS:
    _quack_cache_utils.EXTRA_SOURCE_DIRS.append(_STREAM_EP_PKG_ROOT)

from .buffer import Buffer

# noinspection PyUnresolvedReferences
from stream_ep_cpp import Config, EventHandle, topk_idx_t
