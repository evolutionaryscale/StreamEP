"""Tests for streaming_moe.

2a.1 — smoke test of `streaming_moe_forward` on 2 GPUs. Asserts shape/dtype/finiteness.
2a.2 — numerics vs eager reference. Reference uses replicated global weights and
       computes the topk-weighted expert output token-by-(token, k); streaming
       version shards weights across ranks and relies on `combine` for cross-rank
       reduction. Both should produce the same per-rank output up to bf16 noise.
"""

import pytest
import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
from deep_ep import Buffer as DeepEPBuffer

from evolutionaryscale.models.moe.streaming_moe.streaming_moe import (
    streaming_moe_forward,
)
from evolutionaryscale.utils.testing_utils import (
    requires_gpus,
    run_distributed_test,
    setup_distributed_for_test,
)


def _make_buffer(
    group, num_sms: int, hidden_size: int, dtype: torch.dtype
) -> DeepEPBuffer:
    DeepEPBuffer.set_num_sms(num_sms)
    hidden_bytes = hidden_size * max(torch.tensor([], dtype=dtype).element_size(), 2)
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (
        DeepEPBuffer.get_dispatch_config(group.size()),
        DeepEPBuffer.get_combine_config(group.size()),
    ):
        nvl_bytes = max(
            cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()), nvl_bytes
        )
        rdma_bytes = max(
            cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()), rdma_bytes
        )
    return DeepEPBuffer(
        group, nvl_bytes, rdma_bytes, num_qps_per_rank=DeepEPBuffer.num_sms
    )


def _make_uniform_topk_idx(
    n_tokens: int, topk: int, num_experts: int, rank: int, device: torch.device
) -> torch.Tensor:
    base = (torch.arange(n_tokens, device=device) + rank * n_tokens) * topk
    offsets = torch.arange(topk, device=device).unsqueeze(0)
    return ((base.unsqueeze(1) + offsets) % num_experts).to(torch.int64)


def _make_global_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    device,
):
    """Replicated global weights via fixed seed. Each rank slices its local
    portion. QuACK gate/up convention: interleaved on the 2*I output dim
    (gate = preact[..., ::2], up = preact[..., 1::2]).
    """
    g = torch.Generator(device=device).manual_seed(42)
    w1 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
            generator=g,
        )
        * 0.02
    ).contiguous()
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=dtype,
            device=device,
            generator=g,
        )
        * 0.02
    ).contiguous()
    return w1, w2


def eager_moe_reference(
    x: torch.Tensor,
    w1_full_q: torch.Tensor,
    w2_full_q: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Plain (vectorized) reference: for each (token, k), run the routed expert
    and topk-weighted-sum. Convention matches `quack.gemm_interface.gemm_gated`
    with `activation="swiglu"`: gate/up are interleaved on the last weight dim
    (`gate = preact[..., ::2]`, `up = preact[..., 1::2]`).
    """
    T, H = x.shape
    K = topk_idx.size(1)
    expert_ids = topk_idx.reshape(-1)
    weights_flat = topk_weights.reshape(-1).to(x.dtype)
    token_ids = (
        torch.arange(T, device=x.device, dtype=torch.long)
        .unsqueeze(1)
        .expand(T, K)
        .reshape(-1)
    )

    x_expanded = x[token_ids]
    w1_e = w1_full_q[expert_ids]
    w2_e = w2_full_q[expert_ids]

    preact = torch.einsum("th,thi->ti", x_expanded, w1_e)
    gate = preact[..., ::2]
    up = preact[..., 1::2]
    h = F.silu(gate) * up
    o = torch.einsum("ti,tih->th", h, w2_e)

    out = torch.zeros_like(x)
    out.index_add_(0, token_ids, weights_flat.unsqueeze(-1) * o)
    return out


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """1 - cosine-similarity-like metric, matching DeepEP's tests/utils.py."""
    a, b = x.double() + 1, y.double() + 1
    denom = (a * a + b * b).sum()
    return float((1 - 2 * (a * b).sum() / denom).item())


def streaming_moe_smoke(rank, world_size, master_port):
    setup_distributed_for_test(rank, world_size, master_port)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    T = 64
    H = 128
    I = 64
    K = 2
    num_experts = 8
    local_E = num_experts // world_size

    torch.manual_seed(0)
    x = (torch.randn(T, H, dtype=dtype, device=device) * 0.1).contiguous()
    w1 = (
        torch.randn(local_E, 2 * I, H, dtype=dtype, device=device) * 0.02
    ).contiguous()
    w2 = (torch.randn(local_E, H, I, dtype=dtype, device=device) * 0.02).contiguous()
    w1_q = w1.permute(0, 2, 1).contiguous()
    w2_q = w2.permute(0, 2, 1).contiguous()

    topk_idx = _make_uniform_topk_idx(T, K, num_experts, rank, device)
    topk_weights = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device)

    assert torch_dist.group.WORLD is not None
    buffer = _make_buffer(
        torch_dist.group.WORLD, num_sms=16, hidden_size=H, dtype=dtype
    )

    out = streaming_moe_forward(
        x, w1_q, w2_q, topk_idx, topk_weights, buffer, num_experts
    )

    assert out.shape == (T, H), f"shape {out.shape}, expected ({T}, {H})"
    assert out.dtype == dtype
    assert torch.isfinite(out).all(), "non-finite values in output"

    torch_dist.destroy_process_group()


def streaming_moe_numerics(rank, world_size, master_port):
    setup_distributed_for_test(rank, world_size, master_port)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    T = 64
    H = 128
    I = 64
    K = 2
    num_experts = 8
    local_E = num_experts // world_size

    w1_full, w2_full = _make_global_weights(num_experts, H, I, dtype, device)
    w1_full_q = w1_full.permute(0, 2, 1).contiguous()
    w2_full_q = w2_full.permute(0, 2, 1).contiguous()

    w1_q = w1_full_q[rank * local_E : (rank + 1) * local_E].contiguous()
    w2_q = w2_full_q[rank * local_E : (rank + 1) * local_E].contiguous()

    torch.manual_seed(100 + rank)
    x = (torch.randn(T, H, dtype=dtype, device=device) * 0.1).contiguous()

    topk_idx = _make_uniform_topk_idx(T, K, num_experts, rank, device)
    topk_weights = torch.softmax(
        torch.randn(T, K, dtype=torch.float32, device=device), dim=-1
    ).contiguous()

    assert torch_dist.group.WORLD is not None
    buffer = _make_buffer(
        torch_dist.group.WORLD, num_sms=16, hidden_size=H, dtype=dtype
    )

    out_streaming = streaming_moe_forward(
        x, w1_q, w2_q, topk_idx, topk_weights, buffer, num_experts
    )
    out_ref = eager_moe_reference(x, w1_full_q, w2_full_q, topk_idx, topk_weights)

    diff = calc_diff(out_streaming, out_ref)
    max_abs = (out_streaming - out_ref).abs().max().item()
    print(f"[rank {rank}] calc_diff={diff:.3e} max_abs={max_abs:.3e}", flush=True)
    assert diff < 1e-3, f"rank {rank}: calc_diff {diff:.3e} exceeds tolerance"

    torch_dist.destroy_process_group()


@pytest.mark.nightly
@requires_gpus(2)
def test_streaming_moe_smoke_2gpu():
    run_distributed_test(streaming_moe_smoke, world_size=2)


@pytest.mark.nightly
@requires_gpus(2)
def test_streaming_moe_numerics_2gpu():
    run_distributed_test(streaming_moe_numerics, world_size=2)
