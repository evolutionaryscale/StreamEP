"""Tests for the streaming-MoE kernel Y (fused atomic-scatter epilogue).

Mirrors the kernel-A 4-case suite plus an extra padding-row case.

  * test_compile: kernel compiles for a representative shape (compile-only).
  * test_single_tile: total_tiles=1, a_ready pre-set, no padding rows. Validates
    GEMM + per-row weight multiply + atomic-scatter against a torch reference.
  * test_multi_tile_static: total_tiles=N spread across multiple experts,
    multiple recv-tokens per tile, K_local>1 per recv-token (so o[r] gets
    multiple contributions). Validates per-token accumulation and end-of-tile
    bookkeeping (k_local_remaining decrement + y_done_per_token release).
  * test_padding_rows: tile contains some pool slots with pool_recv_token == -1.
    Validates PTX-predicated atomic-add: padding lanes skip the atomic at
    issue time so no contributions land in valid rows from padding slots.
  * test_producer_consumer: kernel Y on compute_y_stream spins on a_ready while
    a producer kernel on a separate stream fires slot-by-slot.

All cases assert:
  - o[r, :] equals sum over slots s mapping to r of
    (pool_topk_weight[s] * (postact_a[s] @ W2[expert_id(s)].T))
  - k_local_remaining[r] hits 0 (each contribution decremented exactly once)
  - y_done_per_token[r] == combine_seq (release fired exactly once)
  - No trash row needed: predicated atomic skips padding lanes entirely
    (verified by k_local_remaining[r] == 0 for all valid r).
"""

from __future__ import annotations

import pytest
import torch


def _make_tile_metadata(tile_to_expert_list, E_local, device):
    """Build (tile_id_to_expert, expert_pool_block_offset). Same as kernel A test."""
    total_tiles = len(tile_to_expert_list)
    tile_id_to_expert = torch.tensor(
        tile_to_expert_list, dtype=torch.int32, device=device
    )
    expert_pool_block_offset = torch.zeros(
        E_local + 1, dtype=torch.int32, device=device
    )
    counts = [0] * E_local
    for e in tile_to_expert_list:
        counts[e] += 1
    cum = 0
    for e in range(E_local):
        expert_pool_block_offset[e] = cum
        cum += counts[e]
    expert_pool_block_offset[E_local] = cum
    assert cum == total_tiles
    return tile_id_to_expert, expert_pool_block_offset


def _make_a_ready(total_tiles, compute_seq, device, fired=True):
    val = compute_seq if fired else 0
    return torch.full((total_tiles,), val, dtype=torch.int64, device=device)


def _eager_y_reference(
    postact_a: torch.Tensor,  # (total_tiles, tile_m, I)
    W2: torch.Tensor,  # (E_local, H, I)
    pool_recv_token: torch.Tensor,  # (TK_padded,) int32
    pool_topk_weight: torch.Tensor,  # (TK_padded,) float32
    tile_id_to_expert: torch.Tensor,  # (total_tiles,) int32
    T_recv: int,
):
    """Compute o_ref[r, :] = sum over s mapping to r of
    (w[s] * (postact_a_flat[s] @ W2[expert_id(s)].T)) in fp32.
    Returns (o_ref bf16, k_local_per_r int32) for assertion comparisons.
    """
    total_tiles, tile_m, I = postact_a.shape
    E_local, H, _ = W2.shape
    device = postact_a.device

    o_ref = torch.zeros(T_recv, H, dtype=torch.float32, device=device)
    k_local = torch.zeros(T_recv, dtype=torch.int32, device=device)

    postact_flat = postact_a.view(total_tiles * tile_m, I).float()
    pool_topk_weight_f = pool_topk_weight.float()
    for s in range(total_tiles * tile_m):
        r = int(pool_recv_token[s].item())
        if r < 0 or r >= T_recv:
            continue
        t = s // tile_m
        e = int(tile_id_to_expert[t].item())
        y = postact_flat[s] @ W2[e].float().t()  # (H,)
        o_ref[r] += pool_topk_weight_f[s] * y
        k_local[r] += 1
    return o_ref.bfloat16(), k_local


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def test_streaming_moe_y_compiles(device):
    """JIT-compile only (no launch) for a representative production-shape config."""
    from stream_ep.stream_moe.kernel_y import (
        streaming_moe_y,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 64
    total_tiles = 4
    TK_padded = total_tiles * tile_m
    T_recv = 32

    dtype = torch.bfloat16
    postact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)
    W2 = torch.zeros(E_local, H, I, dtype=dtype, device=device)
    o = torch.zeros(T_recv, H, dtype=dtype, device=device)
    pool_recv_token = torch.zeros(TK_padded, dtype=torch.int32, device=device)
    pool_topk_weight = torch.zeros(TK_padded, dtype=torch.float32, device=device)
    k_local_remaining = torch.zeros(T_recv, dtype=torch.int32, device=device)
    y_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        [0, 0, 0, 0], E_local, device
    )
    a_ready = _make_a_ready(total_tiles, compute_seq=1, device=device)

    import quack.cache_utils as cu

    orig = cu.COMPILE_ONLY
    cu.COMPILE_ONLY = True
    try:
        streaming_moe_y(
            postact_a,
            W2,
            o,
            pool_recv_token,
            pool_topk_weight,
            k_local_remaining,
            y_done_per_token,
            expert_pool_block_offset,
            # placeholder pool_arrival_count / pool_arrival_target (elided by
            # kernel Y's spin_kind=ACQUIRE_VS_SEQ; any int32 [total_tiles] tensors
            # of the right shape are fine — reuse a_ready's int32 sibling).
            torch.zeros(total_tiles, dtype=torch.int32, device=device),
            torch.zeros(total_tiles, dtype=torch.int32, device=device),
            a_ready,
            compute_seq=1,
            combine_seq=1,
            tile_m=tile_m,
            tile_n=tile_n,
        )
    finally:
        cu.COMPILE_ONLY = orig


def test_streaming_moe_y_single_tile(device):
    """total_tiles=1, all rows map to distinct recv-tokens (K_local=1).
    Validates GEMM + weight multiply + atomic-scatter into o[r].
    """
    from stream_ep.stream_moe.kernel_y import (
        streaming_moe_y,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 64
    total_tiles = 1
    TK_padded = total_tiles * tile_m
    T_recv = tile_m  # one recv-token per slot
    chosen_expert = 2

    dtype = torch.bfloat16
    torch.manual_seed(0)
    postact_a = torch.randn(total_tiles, tile_m, I, dtype=dtype, device=device).mul_(
        0.1
    )
    W2 = torch.randn(E_local, H, I, dtype=dtype, device=device).mul_(0.02)

    o = torch.zeros(T_recv, H, dtype=dtype, device=device)
    # pool_recv_token: slot s → recv-token s (1:1 mapping, K_local=1 each)
    pool_recv_token = torch.arange(TK_padded, dtype=torch.int32, device=device)
    pool_topk_weight = torch.rand(TK_padded, dtype=torch.float32, device=device)
    k_local_remaining = torch.ones(T_recv, dtype=torch.int32, device=device)
    y_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        [chosen_expert], E_local, device
    )
    a_ready = _make_a_ready(total_tiles, compute_seq=1, device=device)

    streaming_moe_y(
        postact_a,
        W2,
        o,
        pool_recv_token,
        pool_topk_weight,
        k_local_remaining,
        y_done_per_token,
        expert_pool_block_offset,
        # placeholder pool_arrival_count / pool_arrival_target (elided by
        # kernel Y's spin_kind=ACQUIRE_VS_SEQ).
        torch.zeros(total_tiles, dtype=torch.int32, device=device),
        torch.zeros(total_tiles, dtype=torch.int32, device=device),
        a_ready,
        compute_seq=1,
        combine_seq=42,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    o_ref, k_local = _eager_y_reference(
        postact_a, W2, pool_recv_token, pool_topk_weight, tile_id_to_expert, T_recv
    )

    o_kernel = o[:T_recv]
    diff = (o_kernel.float() - o_ref.float()).abs()
    abs_thresh = 1e-2  # bf16 atomic-add accumulates per contribution (lossier
    # than fp32-then-cast); allow ~few ULP at typical scale.
    rel_thresh = 5e-2
    # Fail if any element exceeds BOTH absolute and relative tolerances.
    rel = diff / (o_ref.float().abs() + 1e-3)
    bad = (diff > abs_thresh) & (rel > rel_thresh)
    assert not bad.any(), (
        f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
        f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
        f"max rel diff {rel.max().item():.4f}"
    )

    assert (
        k_local_remaining == 0
    ).all(), f"k_local_remaining not zeroed: {k_local_remaining}"
    assert (
        y_done_per_token == 42
    ).all(), f"y_done_per_token not all 42: {y_done_per_token}"


def test_streaming_moe_y_multi_tile_static(device):
    """6 tiles spread across 4 experts with multiple slots → same recv-token
    (K_local=3 for some recv-tokens). Validates per-token accumulation across
    tiles + bookkeeping decrement to zero.
    """
    from stream_ep.stream_moe.kernel_y import (
        streaming_moe_y,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 64
    tile_to_expert_list = [0, 0, 1, 2, 2, 3]
    total_tiles = len(tile_to_expert_list)
    TK_padded = total_tiles * tile_m
    # 256 recv-tokens; each appears in 3 distinct pool slots (K_local=3 each).
    T_recv = 256

    dtype = torch.bfloat16
    torch.manual_seed(7)
    postact_a = torch.randn(total_tiles, tile_m, I, dtype=dtype, device=device).mul_(
        0.1
    )
    W2 = torch.randn(E_local, H, I, dtype=dtype, device=device).mul_(0.02)

    o = torch.zeros(T_recv, H, dtype=dtype, device=device)

    # Build pool_recv_token so each recv-token r appears in exactly 3 slots
    # (TK_padded / T_recv = 768 / 256 = 3). Simple deterministic distribution.
    assert TK_padded % T_recv == 0
    k_local_each = TK_padded // T_recv
    pool_recv_token_list = []
    for s in range(TK_padded):
        pool_recv_token_list.append(s % T_recv)
    pool_recv_token = torch.tensor(
        pool_recv_token_list, dtype=torch.int32, device=device
    )
    pool_topk_weight = torch.rand(TK_padded, dtype=torch.float32, device=device)
    k_local_remaining = torch.full(
        (T_recv,), k_local_each, dtype=torch.int32, device=device
    )
    y_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    a_ready = _make_a_ready(total_tiles, compute_seq=1, device=device)

    streaming_moe_y(
        postact_a,
        W2,
        o,
        pool_recv_token,
        pool_topk_weight,
        k_local_remaining,
        y_done_per_token,
        expert_pool_block_offset,
        # placeholder pool_arrival_count / pool_arrival_target (elided by
        # kernel Y's spin_kind=ACQUIRE_VS_SEQ).
        torch.zeros(total_tiles, dtype=torch.int32, device=device),
        torch.zeros(total_tiles, dtype=torch.int32, device=device),
        a_ready,
        compute_seq=1,
        combine_seq=99,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    o_ref, k_ref = _eager_y_reference(
        postact_a, W2, pool_recv_token, pool_topk_weight, tile_id_to_expert, T_recv
    )

    assert (
        k_ref == k_local_each
    ).all(), (
        f"reference K_local mismatch: got {k_ref.unique()}, expected {k_local_each}"
    )

    o_kernel = o[:T_recv]
    diff = (o_kernel.float() - o_ref.float()).abs()
    abs_thresh = 1e-2  # bf16 atomic-add accumulates per contribution (lossier
    # than fp32-then-cast); allow ~few ULP at typical scale.
    rel_thresh = 5e-2
    # Fail if any element exceeds BOTH absolute and relative tolerances.
    rel = diff / (o_ref.float().abs() + 1e-3)
    bad = (diff > abs_thresh) & (rel > rel_thresh)
    assert not bad.any(), (
        f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
        f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
        f"max rel diff {rel.max().item():.4f}"
    )

    assert (k_local_remaining == 0).all(), (
        f"k_local_remaining not zeroed: bad indices "
        f"{(k_local_remaining != 0).nonzero().squeeze().tolist()}"
    )
    assert (y_done_per_token == 99).all(), (
        f"y_done_per_token not all 99: bad indices "
        f"{(y_done_per_token != 99).nonzero().squeeze().tolist()}"
    )


def test_streaming_moe_y_padding_rows(device):
    """Some pool slots have pool_recv_token == -1 (padding rows from BLOCK_M
    padding in dispatch). Validates PTX-predicated atomic-add: padding lanes
    skip `red.global.add.noftz.v4.bf16x2` at issue time, so no padding
    contribution lands in any valid o[r, :].
    """
    from stream_ep.stream_moe.kernel_y import (
        streaming_moe_y,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 64
    total_tiles = 2
    TK_padded = total_tiles * tile_m
    T_recv = 100

    dtype = torch.bfloat16
    torch.manual_seed(13)
    postact_a = torch.randn(total_tiles, tile_m, I, dtype=dtype, device=device).mul_(
        0.1
    )
    W2 = torch.randn(E_local, H, I, dtype=dtype, device=device).mul_(0.02)

    o = torch.zeros(T_recv, H, dtype=dtype, device=device)

    # First tile: slots 0..99 → recv-tokens 0..99 (valid), slots 100..127 padding.
    # Second tile: slots 128..227 → recv-tokens 0..99 (K_local=2), slots 228..255 padding.
    pool_recv_token_list = []
    for t in range(total_tiles):
        for m in range(tile_m):
            s = t * tile_m + m
            if m < T_recv:
                pool_recv_token_list.append(s % T_recv)
            else:
                pool_recv_token_list.append(-1)
    pool_recv_token = torch.tensor(
        pool_recv_token_list, dtype=torch.int32, device=device
    )

    # Padding rows: weight = 0 (matches design's zero-init pad semantics).
    pool_topk_weight = torch.rand(TK_padded, dtype=torch.float32, device=device)
    pool_topk_weight[pool_recv_token < 0] = 0.0

    k_local_remaining = torch.full((T_recv,), 2, dtype=torch.int32, device=device)
    y_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        [0, 1], E_local, device
    )
    a_ready = _make_a_ready(total_tiles, compute_seq=1, device=device)

    streaming_moe_y(
        postact_a,
        W2,
        o,
        pool_recv_token,
        pool_topk_weight,
        k_local_remaining,
        y_done_per_token,
        expert_pool_block_offset,
        # placeholder pool_arrival_count / pool_arrival_target (elided by
        # kernel Y's spin_kind=ACQUIRE_VS_SEQ).
        torch.zeros(total_tiles, dtype=torch.int32, device=device),
        torch.zeros(total_tiles, dtype=torch.int32, device=device),
        a_ready,
        compute_seq=1,
        combine_seq=7,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    o_ref, _ = _eager_y_reference(
        postact_a, W2, pool_recv_token, pool_topk_weight, tile_id_to_expert, T_recv
    )

    o_kernel = o[:T_recv]
    diff = (o_kernel.float() - o_ref.float()).abs()
    abs_thresh = 1e-2  # bf16 atomic-add accumulates per contribution (lossier
    # than fp32-then-cast); allow ~few ULP at typical scale.
    rel_thresh = 5e-2
    # Fail if any element exceeds BOTH absolute and relative tolerances.
    rel = diff / (o_ref.float().abs() + 1e-3)
    bad = (diff > abs_thresh) & (rel > rel_thresh)
    assert not bad.any(), (
        f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
        f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
        f"max rel diff {rel.max().item():.4f}"
    )

    assert (k_local_remaining == 0).all()
    assert (y_done_per_token == 7).all()


def test_streaming_moe_y_producer_consumer(device):
    """Kernel Y on compute_y_stream spins on a_ready while a producer fires
    slot-by-slot from a separate stream.
    """
    from stream_ep.stream_moe.kernel_y import (
        fire_a_ready_with_delay,
        streaming_moe_y,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 64
    tile_to_expert_list = [0, 0, 1, 2, 2, 3]
    total_tiles = len(tile_to_expert_list)
    TK_padded = total_tiles * tile_m
    T_recv = TK_padded  # 1:1 mapping for simplicity

    dtype = torch.bfloat16
    torch.manual_seed(11)
    postact_a = torch.randn(total_tiles, tile_m, I, dtype=dtype, device=device).mul_(
        0.1
    )
    W2 = torch.randn(E_local, H, I, dtype=dtype, device=device).mul_(0.02)

    o = torch.zeros(T_recv, H, dtype=dtype, device=device)
    pool_recv_token = torch.arange(TK_padded, dtype=torch.int32, device=device)
    pool_topk_weight = torch.rand(TK_padded, dtype=torch.float32, device=device)
    k_local_remaining = torch.ones(T_recv, dtype=torch.int32, device=device)
    y_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    a_ready = _make_a_ready(total_tiles, compute_seq=1, device=device, fired=False)

    # Pre-warm the producer JIT.
    fire_a_ready_with_delay(a_ready, compute_seq=999, delay_us=0)
    torch.cuda.synchronize()
    a_ready.zero_()
    torch.cuda.synchronize()

    compute_y_stream = torch.cuda.Stream()
    producer_stream = torch.cuda.Stream()

    with torch.cuda.stream(compute_y_stream):
        streaming_moe_y(
            postact_a,
            W2,
            o,
            pool_recv_token,
            pool_topk_weight,
            k_local_remaining,
            y_done_per_token,
            expert_pool_block_offset,
            # placeholder pool_arrival_count / pool_arrival_target (elided by
            # kernel Y's spin_kind=ACQUIRE_VS_SEQ; any int32 [total_tiles] tensors
            # of the right shape are fine — reuse a_ready's int32 sibling).
            torch.zeros(total_tiles, dtype=torch.int32, device=device),
            torch.zeros(total_tiles, dtype=torch.int32, device=device),
            a_ready,
            compute_seq=1,
            combine_seq=5,
            tile_m=tile_m,
            tile_n=tile_n,
        )
    with torch.cuda.stream(producer_stream):
        fire_a_ready_with_delay(a_ready, compute_seq=1, delay_us=50)

    torch.cuda.synchronize()

    o_ref, _ = _eager_y_reference(
        postact_a, W2, pool_recv_token, pool_topk_weight, tile_id_to_expert, T_recv
    )

    o_kernel = o[:T_recv]
    diff = (o_kernel.float() - o_ref.float()).abs()
    abs_thresh = 1e-2  # bf16 atomic-add accumulates per contribution (lossier
    # than fp32-then-cast); allow ~few ULP at typical scale.
    rel_thresh = 5e-2
    # Fail if any element exceeds BOTH absolute and relative tolerances.
    rel = diff / (o_ref.float().abs() + 1e-3)
    bad = (diff > abs_thresh) & (rel > rel_thresh)
    assert not bad.any(), (
        f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
        f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
        f"max rel diff {rel.max().item():.4f}"
    )

    assert (k_local_remaining == 0).all()
    assert (y_done_per_token == 5).all()


if __name__ == "__main__":
    dev = torch.device("cuda")
    test_streaming_moe_y_compiles(dev)
    print("compile OK")
    test_streaming_moe_y_single_tile(dev)
    print("single-tile PASS")
    test_streaming_moe_y_multi_tile_static(dev)
    print("multi-tile-static PASS")
    test_streaming_moe_y_padding_rows(dev)
    print("padding-rows PASS")
    test_streaming_moe_y_producer_consumer(dev)
    print("producer-consumer PASS")
