"""Profiling harnesses for the streaming-MoE pipeline (not shipped in the wheel).

Run as modules from the repo root, e.g.
``torchrun --standalone --nproc-per-node=8 -m stream_ep_bench.profile_pipeline``.
"""

import torch

# Default profiling shape: a large training MoE layer, so the GEMMs and the
# communication are realistically sized. Every harness can override it.
H = 3072
I = 768
NUM_EXPERTS = 256
TOPK = 8
SEQ_LEN_PER_RANK = 8192
DTYPE = torch.bfloat16
