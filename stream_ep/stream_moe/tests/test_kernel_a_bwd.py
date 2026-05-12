"""Tests for the streaming-MoE kernel A bwd (NN GEMM + atomic-scatter epilogue).

Mirrors the kernel-Y suite — the bwd kernel is a strict subset of fwd
kernel_y's epilogue (no per-slot weight multiply), so the test cases
parallel `test_kernel_y.py` 1:1.

  * test_compile: kernel compiles for a representative shape (compile-only).
  * test_single_tile: total_tiles=1, bwd_a_ready pre-set, no padding rows.
    Validates GEMM `dL_dswiglu_in @ W1[e]` + atomic-scatter against a torch
    reference.
  * test_multi_tile_static: total_tiles=N spread across multiple experts,
    multiple recv-tokens per tile, K_local>1 per recv-token (so dL_dx_per_r[r]
    gets multiple contributions). Validates per-token accumulation and
    end-of-tile bookkeeping (bwd_k_local_remaining decrement +
    bwd_a_done_per_token release).
  * test_padding_rows: tile contains some pool slots with pool_recv_token == -1.
    Validates PTX-predicated atomic-add: padding lanes skip the atomic at
    issue time so no contributions land in valid rows from padding slots.
  * test_producer_consumer: kernel A bwd on compute_a_stream spins on
    bwd_a_ready while a producer kernel on a separate stream fires
    slot-by-slot.

All cases assert:
  - dL_dx_per_r[r, :] equals sum over slots s mapping to r of
    (dL_dswiglu_in[s] @ W1[expert_id(s)])
  - bwd_k_local_remaining[r] hits 0 (each contribution decremented once)
  - bwd_a_done_per_token[r] == dispatch_seq (release fired exactly once)
  - No trash row needed: predicated atomic skips padding lanes entirely.
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


def _make_bwd_a_ready(total_tiles, dispatch_seq, device, fired=True):
    val = dispatch_seq if fired else 0
    return torch.full((total_tiles,), val, dtype=torch.int64, device=device)


def _eager_a_bwd_reference(
    dL_dswiglu_in: torch.Tensor,  # (total_tiles, tile_m, 2I)
    W1: torch.Tensor,  # (E_local, 2I, H)
    pool_recv_token: torch.Tensor,  # (TK_padded,) int32
    tile_id_to_expert: torch.Tensor,  # (total_tiles,) int32
    T_recv: int,
):
    """Compute dL_dx_per_r_ref[r, :] = sum over s mapping to r of
    (dL_dswiglu_in[s] @ W1[expert_id(s)]) in fp32, then cast to bf16.
    Returns (dL_dx_per_r_ref bf16, k_local_per_r int32).
    """
    total_tiles, tile_m, two_I = dL_dswiglu_in.shape
    E_local, _, H = W1.shape
    device = dL_dswiglu_in.device

    ref = torch.zeros(T_recv, H, dtype=torch.float32, device=device)
    k_local = torch.zeros(T_recv, dtype=torch.int32, device=device)

    flat = dL_dswiglu_in.view(total_tiles * tile_m, two_I).float()
    for s in range(total_tiles * tile_m):
        r = int(pool_recv_token[s].item())
        if r < 0 or r >= T_recv:
            continue
        t = s // tile_m
        e = int(tile_id_to_expert[t].item())
        # (2I,) @ (2I, H) → (H,)
        ref[r] += flat[s] @ W1[e].float()
        k_local[r] += 1
    return ref.bfloat16(), k_local


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def test_streaming_moe_a_bwd_compiles(device):
    """JIT-compile only (no launch) for a representative production-shape config."""
    from stream_ep.stream_moe.kernel_a_bwd import (
        streaming_moe_a_bwd,
    )

    H, I, E_local = 128, 256, 4
    two_I = 2 * I
    tile_m, tile_n = 128, 64
    total_tiles = 4
    TK_padded = total_tiles * tile_m
    T_recv = 32

    dtype = torch.bfloat16
    dL_dswiglu_in = torch.zeros(total_tiles, tile_m, two_I, dtype=dtype, device=device)
    W1 = torch.zeros(E_local, two_I, H, dtype=dtype, device=device)
    dL_dx_per_r = torch.zeros(T_recv, H, dtype=dtype, device=device)
    pool_recv_token = torch.zeros(TK_padded, dtype=torch.int32, device=device)
    bwd_k_local_remaining = torch.zeros(T_recv, dtype=torch.int32, device=device)
    bwd_a_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        [0, 0, 0, 0], E_local, device
    )
    bwd_a_ready = _make_bwd_a_ready(total_tiles, dispatch_seq=1, device=device)

    import quack.cache_utils as cu

    orig = cu.COMPILE_ONLY
    cu.COMPILE_ONLY = True
    try:
        streaming_moe_a_bwd(
            dL_dswiglu_in,
            W1,
            dL_dx_per_r,
            pool_recv_token,
            bwd_k_local_remaining,
            bwd_a_done_per_token,
            expert_pool_block_offset,
            bwd_a_ready,
            dispatch_seq=1,
            tile_m=tile_m,
            tile_n=tile_n,
        )
    finally:
        cu.COMPILE_ONLY = orig


def test_streaming_moe_a_bwd_single_tile(device):
    """total_tiles=1, all rows map to distinct recv-tokens (K_local=1).
    Validates GEMM + atomic-scatter into dL_dx_per_r[r].
    """
    from stream_ep.stream_moe.kernel_a_bwd import (
        streaming_moe_a_bwd,
    )

    H, I, E_local = 128, 256, 4
    two_I = 2 * I
    tile_m, tile_n = 128, 64
    total_tiles = 1
    TK_padded = total_tiles * tile_m
    T_recv = tile_m
    chosen_expert = 2

    dtype = torch.bfloat16
    torch.manual_seed(0)
    dL_dswiglu_in = torch.randn(
        total_tiles, tile_m, two_I, dtype=dtype, device=device
    ).mul_(0.05)
    W1 = torch.randn(E_local, two_I, H, dtype=dtype, device=device).mul_(0.02)

    dL_dx_per_r = torch.zeros(T_recv, H, dtype=dtype, device=device)
    pool_recv_token = torch.arange(TK_padded, dtype=torch.int32, device=device)
    bwd_k_local_remaining = torch.ones(T_recv, dtype=torch.int32, device=device)
    bwd_a_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        [chosen_expert], E_local, device
    )
    dispatch_seq = 42
    bwd_a_ready = _make_bwd_a_ready(
        total_tiles, dispatch_seq=dispatch_seq, device=device
    )

    streaming_moe_a_bwd(
        dL_dswiglu_in,
        W1,
        dL_dx_per_r,
        pool_recv_token,
        bwd_k_local_remaining,
        bwd_a_done_per_token,
        expert_pool_block_offset,
        bwd_a_ready,
        dispatch_seq=dispatch_seq,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    ref, _ = _eager_a_bwd_reference(
        dL_dswiglu_in, W1, pool_recv_token, tile_id_to_expert, T_recv
    )

    got = dL_dx_per_r[:T_recv]
    diff = (got.float() - ref.float()).abs()
    abs_thresh = 1e-2  # bf16 atomic-add accumulates per contribution.
    rel_thresh = 5e-2
    rel = diff / (ref.float().abs() + 1e-3)
    bad = (diff > abs_thresh) & (rel > rel_thresh)
    assert not bad.any(), (
        f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
        f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
        f"max rel diff {rel.max().item():.4f}"
    )

    assert (
        bwd_k_local_remaining == 0
    ).all(), f"bwd_k_local_remaining not zeroed: {bwd_k_local_remaining}"
    assert (bwd_a_done_per_token == dispatch_seq).all(), (
        f"bwd_a_done_per_token not all {dispatch_seq}: "
        f"{bwd_a_done_per_token}"
    )


def test_streaming_moe_a_bwd_multi_tile_static(device):
    """6 tiles spread across 4 experts with multiple slots → same recv-token
    (K_local=3 for some recv-tokens). Validates per-token accumulation across
    tiles + bookkeeping decrement to zero.
    """
    from stream_ep.stream_moe.kernel_a_bwd import (
        streaming_moe_a_bwd,
    )

    H, I, E_local = 128, 256, 4
    two_I = 2 * I
    tile_m, tile_n = 128, 64
    tile_to_expert_list = [0, 0, 1, 2, 2, 3]
    total_tiles = len(tile_to_expert_list)
    TK_padded = total_tiles * tile_m
    T_recv = 256

    dtype = torch.bfloat16
    torch.manual_seed(7)
    dL_dswiglu_in = torch.randn(
        total_tiles, tile_m, two_I, dtype=dtype, device=device
    ).mul_(0.05)
    W1 = torch.randn(E_local, two_I, H, dtype=dtype, device=device).mul_(0.02)

    dL_dx_per_r = torch.zeros(T_recv, H, dtype=dtype, device=device)

    # Each recv-token r appears in exactly TK_padded // T_recv = 3 slots.
    assert TK_padded % T_recv == 0
    k_local_each = TK_padded // T_recv
    pool_recv_token = torch.tensor(
        [s % T_recv for s in range(TK_padded)], dtype=torch.int32, device=device
    )
    bwd_k_local_remaining = torch.full(
        (T_recv,), k_local_each, dtype=torch.int32, device=device
    )
    bwd_a_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    dispatch_seq = 99
    bwd_a_ready = _make_bwd_a_ready(
        total_tiles, dispatch_seq=dispatch_seq, device=device
    )

    streaming_moe_a_bwd(
        dL_dswiglu_in,
        W1,
        dL_dx_per_r,
        pool_recv_token,
        bwd_k_local_remaining,
        bwd_a_done_per_token,
        expert_pool_block_offset,
        bwd_a_ready,
        dispatch_seq=dispatch_seq,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    ref, k_ref = _eager_a_bwd_reference(
        dL_dswiglu_in, W1, pool_recv_token, tile_id_to_expert, T_recv
    )

    assert (
        k_ref == k_local_each
    ).all(), (
        f"reference K_local mismatch: got {k_ref.unique()}, expected {k_local_each}"
    )

    got = dL_dx_per_r[:T_recv]
    diff = (got.float() - ref.float()).abs()
    abs_thresh = 1e-2
    rel_thresh = 5e-2
    rel = diff / (ref.float().abs() + 1e-3)
    bad = (diff > abs_thresh) & (rel > rel_thresh)
    assert not bad.any(), (
        f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
        f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
        f"max rel diff {rel.max().item():.4f}"
    )

    assert (bwd_k_local_remaining == 0).all(), (
        f"bwd_k_local_remaining not zeroed: bad indices "
        f"{(bwd_k_local_remaining != 0).nonzero().squeeze().tolist()}"
    )
    assert (bwd_a_done_per_token == dispatch_seq).all(), (
        f"bwd_a_done_per_token not all {dispatch_seq}: bad indices "
        f"{(bwd_a_done_per_token != dispatch_seq).nonzero().squeeze().tolist()}"
    )


def test_streaming_moe_a_bwd_padding_rows(device):
    """Some pool slots have pool_recv_token == -1 (padding rows from BLOCK_M
    padding in dispatch). Validates PTX-predicated atomic-add: padding lanes
    skip `red.global.add.noftz.v4.bf16x2` at issue time, so no padding
    contribution lands in any valid dL_dx_per_r[r].
    """
    from stream_ep.stream_moe.kernel_a_bwd import (
        streaming_moe_a_bwd,
    )

    H, I, E_local = 128, 256, 4
    two_I = 2 * I
    tile_m, tile_n = 128, 64
    total_tiles = 2
    T_recv = 100

    dtype = torch.bfloat16
    torch.manual_seed(13)
    dL_dswiglu_in = torch.randn(
        total_tiles, tile_m, two_I, dtype=dtype, device=device
    ).mul_(0.05)
    W1 = torch.randn(E_local, two_I, H, dtype=dtype, device=device).mul_(0.02)

    dL_dx_per_r = torch.zeros(T_recv, H, dtype=dtype, device=device)

    # Each tile: slots 0..99 → recv-tokens 0..99 (valid),
    # slots 100..127 padding (-1). K_local per recv-token = 2.
    pool_recv_token_list = []
    for _ in range(total_tiles):
        for m in range(tile_m):
            if m < T_recv:
                pool_recv_token_list.append(m)
            else:
                pool_recv_token_list.append(-1)
    pool_recv_token = torch.tensor(
        pool_recv_token_list, dtype=torch.int32, device=device
    )

    bwd_k_local_remaining = torch.full((T_recv,), 2, dtype=torch.int32, device=device)
    bwd_a_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        [0, 1], E_local, device
    )
    dispatch_seq = 7
    bwd_a_ready = _make_bwd_a_ready(
        total_tiles, dispatch_seq=dispatch_seq, device=device
    )

    streaming_moe_a_bwd(
        dL_dswiglu_in,
        W1,
        dL_dx_per_r,
        pool_recv_token,
        bwd_k_local_remaining,
        bwd_a_done_per_token,
        expert_pool_block_offset,
        bwd_a_ready,
        dispatch_seq=dispatch_seq,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    ref, _ = _eager_a_bwd_reference(
        dL_dswiglu_in, W1, pool_recv_token, tile_id_to_expert, T_recv
    )

    got = dL_dx_per_r[:T_recv]
    diff = (got.float() - ref.float()).abs()
    abs_thresh = 1e-2
    rel_thresh = 5e-2
    rel = diff / (ref.float().abs() + 1e-3)
    bad = (diff > abs_thresh) & (rel > rel_thresh)
    assert not bad.any(), (
        f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
        f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
        f"max rel diff {rel.max().item():.4f}"
    )

    assert (bwd_k_local_remaining == 0).all()
    assert (bwd_a_done_per_token == dispatch_seq).all()


def test_streaming_moe_a_bwd_producer_consumer(device):
    """Kernel A bwd on compute_a_stream spins on bwd_a_ready while a producer
    fires slot-by-slot from a separate stream. Reuses the kernel-A producer
    helper (`fire_tiles_with_delay`) since the streaming-handshake shape is
    identical (per-tile int64 release-store with system scope).
    """
    from stream_ep.stream_moe.kernel_a import (
        fire_tiles_with_delay,
    )
    from stream_ep.stream_moe.kernel_a_bwd import (
        streaming_moe_a_bwd,
    )

    H, I, E_local = 128, 256, 4
    two_I = 2 * I
    tile_m, tile_n = 128, 64
    tile_to_expert_list = [0, 0, 1, 2, 2, 3]
    total_tiles = len(tile_to_expert_list)
    TK_padded = total_tiles * tile_m
    T_recv = TK_padded  # 1:1 mapping for simplicity

    dtype = torch.bfloat16
    torch.manual_seed(11)
    dL_dswiglu_in = torch.randn(
        total_tiles, tile_m, two_I, dtype=dtype, device=device
    ).mul_(0.05)
    W1 = torch.randn(E_local, two_I, H, dtype=dtype, device=device).mul_(0.02)

    dL_dx_per_r = torch.zeros(T_recv, H, dtype=dtype, device=device)
    pool_recv_token = torch.arange(TK_padded, dtype=torch.int32, device=device)
    bwd_k_local_remaining = torch.ones(T_recv, dtype=torch.int32, device=device)
    bwd_a_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    bwd_a_ready = _make_bwd_a_ready(
        total_tiles, dispatch_seq=1, device=device, fired=False
    )

    # Pre-warm the producer JIT.
    fire_tiles_with_delay(bwd_a_ready, dispatch_seq=999, delay_us=0)
    torch.cuda.synchronize()
    bwd_a_ready.zero_()
    torch.cuda.synchronize()

    compute_a_stream = torch.cuda.Stream()
    producer_stream = torch.cuda.Stream()

    with torch.cuda.stream(compute_a_stream):
        streaming_moe_a_bwd(
            dL_dswiglu_in,
            W1,
            dL_dx_per_r,
            pool_recv_token,
            bwd_k_local_remaining,
            bwd_a_done_per_token,
            expert_pool_block_offset,
            bwd_a_ready,
            dispatch_seq=5,
            tile_m=tile_m,
            tile_n=tile_n,
        )
    with torch.cuda.stream(producer_stream):
        fire_tiles_with_delay(bwd_a_ready, dispatch_seq=5, delay_us=50)

    torch.cuda.synchronize()

    ref, _ = _eager_a_bwd_reference(
        dL_dswiglu_in, W1, pool_recv_token, tile_id_to_expert, T_recv
    )

    got = dL_dx_per_r[:T_recv]
    diff = (got.float() - ref.float()).abs()
    abs_thresh = 1e-2
    rel_thresh = 5e-2
    rel = diff / (ref.float().abs() + 1e-3)
    bad = (diff > abs_thresh) & (rel > rel_thresh)
    assert not bad.any(), (
        f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
        f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
        f"max rel diff {rel.max().item():.4f}"
    )

    assert (bwd_k_local_remaining == 0).all()
    assert (bwd_a_done_per_token == 5).all()


if __name__ == "__main__":
    dev = torch.device("cuda")
    test_streaming_moe_a_bwd_compiles(dev)
    print("compile OK")
    test_streaming_moe_a_bwd_single_tile(dev)
    print("single-tile PASS")
    test_streaming_moe_a_bwd_multi_tile_static(dev)
    print("multi-tile-static PASS")
    test_streaming_moe_a_bwd_padding_rows(dev)
    print("padding-rows PASS")
    test_streaming_moe_a_bwd_producer_consumer(dev)
    print("producer-consumer PASS")
