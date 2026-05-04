"""Tests for the streaming-MoE kernel Y bwd (pool layout).

Mirrors `test_streaming_kernel_a.py`'s structure since kernel_y_bwd uses the
same streaming machinery (linear-claim scheduler, per-tile acquire-spin,
per-tile release-store after multi-pid_n gating).

**One sequence number, not two.** Backward shares a single `dispatch_seq`
across acquire-and-release (per `bwd.md` § "Sequence counter") — fwd's
two-value pattern (`dispatch_seq=1` for the producer, `compute_seq=N` for the
release) collapses to one `dispatch_seq` in bwd because the orchestrator
reuses `handle.dispatch_seq` end-to-end. So tests pre-set
`bwd_y_ready = seq` and pass `dispatch_seq = seq` and assert
`bwd_a_ready == seq` — same value end-to-end. Mismatched values (the trap
this comment is here to flag) deadlock the kernel's per-tile acquire-spin.

Reference math: `dL_dpostact_a[slot, :] = pool_topk_weight[slot] *
(dL_do_pool[slot, :] @ W2[expert_for_slot])`. The kernel runs an NN GEMM
(`dL_do_pool @ W2`) by passing W2 permuted to (I, H, E_local) with I
contiguous (n-major B); the reference computes the same product in plain
torch using `W2[e]` directly (the bf16 → fp32 → matmul → bf16 round-trip
matches what the kernel's Float32 accumulator + bf16 epi_convert produces up
to the standard MMA-ordering tolerance).
"""

from __future__ import annotations

import pytest
import torch


def _make_ready(
    total_tiles: int, dispatch_seq: int, device, fired: bool = True
) -> torch.Tensor:
    """Allocate a [total_tiles] int64 ready array. If fired=True, pre-set to
    dispatch_seq (all tiles ready at launch); else zero (test producer fires)."""
    val = dispatch_seq if fired else 0
    return torch.full((total_tiles,), val, dtype=torch.int64, device=device)


def _make_tile_metadata(tile_to_expert_list, E_local, device):
    """Build (tile_id_to_expert, expert_pool_block_offset) from a list mapping
    each tile_id to its expert. Tiles must already be in expert-major order
    (i.e. all tiles for expert e come before any tile for expert e+1).
    """
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
    assert (
        cum == total_tiles
    ), f"tile_to_expert_list must be expert-major and contiguous; got {tile_to_expert_list}"
    return tile_id_to_expert, expert_pool_block_offset


def _ref_dL_dpostact_a(
    dL_do_pool: torch.Tensor,  # (TK_padded, H) bf16
    W2: torch.Tensor,  # (E_local, H, I) bf16
    pool_topk_weight: torch.Tensor,  # (TK_padded,) fp32
    tile_to_expert_list: list[int],
    tile_m: int,
) -> torch.Tensor:
    """Eager torch reference: per-tile NN matmul `dL_do_pool[tile] @ W2[e]`,
    then per-row weight multiply. Returns bf16 (total_tiles, tile_m, I)."""
    total_tiles = len(tile_to_expert_list)
    I = W2.shape[2]
    out = torch.zeros(
        total_tiles, tile_m, I, dtype=dL_do_pool.dtype, device=dL_do_pool.device
    )
    for t in range(total_tiles):
        e = tile_to_expert_list[t]
        rows = slice(t * tile_m, (t + 1) * tile_m)
        # NN matmul in fp32 to match the kernel's Float32 accumulator.
        g = dL_do_pool[rows].float() @ W2[e].float()
        # Per-row weight broadcast (fp32 multiply, bf16 output).
        w = pool_topk_weight[rows].view(tile_m, 1)
        out[t] = (g * w).to(dL_do_pool.dtype)
    return out


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def test_streaming_moe_y_bwd_compiles(device):
    """JIT-compile only (no launch) for a representative production-shape config."""
    from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_y_bwd import (
        streaming_moe_y_bwd,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 128
    total_tiles = 4
    TK_padded = total_tiles * tile_m
    seq = 1

    dtype = torch.bfloat16
    dL_do_pool = torch.randn(TK_padded, H, dtype=dtype, device=device)
    W2 = torch.randn(E_local, H, I, dtype=dtype, device=device).mul_(0.02)
    dL_dpostact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)
    pool_topk_weight = torch.ones(TK_padded, dtype=torch.float32, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        [0, 0, 0, 0], E_local, device
    )
    bwd_y_ready = _make_ready(total_tiles, dispatch_seq=seq, device=device)
    bwd_a_ready = _make_ready(total_tiles, dispatch_seq=0, device=device, fired=False)

    import quack.cache_utils as cu

    orig = cu.COMPILE_ONLY
    cu.COMPILE_ONLY = True
    try:
        streaming_moe_y_bwd(
            dL_do_pool,
            W2,
            dL_dpostact_a,
            pool_topk_weight,
            tile_id_to_expert,
            expert_pool_block_offset,
            bwd_y_ready,
            bwd_a_ready,
            dispatch_seq=seq,
            tile_m=tile_m,
            tile_n=tile_n,
        )
    finally:
        cu.COMPILE_ONLY = orig


def test_streaming_moe_y_bwd_single_tile(device):
    """total_tiles=1, bwd_y_ready pre-set. Validates the full kernel path:
    linear claim, scheduler payload, NN GEMM dL_do_pool @ W2[e], per-row
    pool_topk_weight multiply, pool-layout TMA store, per-tile bwd_a_ready
    release.
    """
    from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_y_bwd import (
        streaming_moe_y_bwd,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 128
    total_tiles = 1
    TK_padded = total_tiles * tile_m
    chosen_expert = 2
    seq = 7

    dtype = torch.bfloat16
    torch.manual_seed(0)
    dL_do_pool = torch.randn(TK_padded, H, dtype=dtype, device=device)
    W2 = torch.randn(E_local, H, I, dtype=dtype, device=device).mul_(0.02)
    dL_dpostact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)
    # Mix of weights (some negative, some > 1) to exercise the multiplicative
    # ColVec broadcast — `*=` vs `+=` would silently pass under all-ones.
    pool_topk_weight = torch.linspace(
        -0.5, 1.5, TK_padded, dtype=torch.float32, device=device
    )

    tile_to_expert_list = [chosen_expert]
    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    bwd_y_ready = _make_ready(total_tiles, dispatch_seq=seq, device=device)
    bwd_a_ready = _make_ready(total_tiles, dispatch_seq=0, device=device, fired=False)

    streaming_moe_y_bwd(
        dL_do_pool,
        W2,
        dL_dpostact_a,
        pool_topk_weight,
        tile_id_to_expert,
        expert_pool_block_offset,
        bwd_y_ready,
        bwd_a_ready,
        dispatch_seq=seq,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    ref = _ref_dL_dpostact_a(
        dL_do_pool, W2, pool_topk_weight, tile_to_expert_list, tile_m
    )
    # bf16 GEMM ordering noise + bf16 cast on the per-row weight multiply.
    # Achieved at this shape: max_abs ≈ 0, max_rel ≈ 0 (bit-identical for
    # single-tile). Use the same "fail iff BOTH atol AND rtol violated"
    # pattern fwd kernel Y uses for room around the few-ULP outliers
    # multi-tile pulls in.
    diff = (dL_dpostact_a.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-3)
    abs_thresh, rel_thresh = 1e-2, 5e-2
    bad = (diff > abs_thresh) & (rel > rel_thresh)
    assert not bad.any(), (
        f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
        f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
        f"max rel diff {rel.max().item():.4f}"
    )
    assert (bwd_a_ready == seq).all(), (
        f"bwd_a_ready not all set to dispatch_seq={seq} (per-tile release "
        f"didn't fire); got unique values {bwd_a_ready.unique().tolist()}"
    )


def test_streaming_moe_y_bwd_multi_tile_static(device):
    """total_tiles=N>1 spread across multiple experts. Validates per-tile
    expert selection (W2[expert_id] varies via tile_id_to_expert) and
    persistent kernel termination via the linear-claim bounds check.
    """
    from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_y_bwd import (
        streaming_moe_y_bwd,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 128
    tile_to_expert_list = [0, 0, 1, 2, 2, 3]
    total_tiles = len(tile_to_expert_list)
    TK_padded = total_tiles * tile_m
    seq = 11

    dtype = torch.bfloat16
    torch.manual_seed(7)
    dL_do_pool = torch.randn(TK_padded, H, dtype=dtype, device=device)
    W2 = torch.randn(E_local, H, I, dtype=dtype, device=device).mul_(0.02)
    dL_dpostact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)
    pool_topk_weight = torch.linspace(
        -0.5, 1.5, TK_padded, dtype=torch.float32, device=device
    )

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    bwd_y_ready = _make_ready(total_tiles, dispatch_seq=seq, device=device)
    bwd_a_ready = _make_ready(total_tiles, dispatch_seq=0, device=device, fired=False)

    streaming_moe_y_bwd(
        dL_do_pool,
        W2,
        dL_dpostact_a,
        pool_topk_weight,
        tile_id_to_expert,
        expert_pool_block_offset,
        bwd_y_ready,
        bwd_a_ready,
        dispatch_seq=seq,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    ref = _ref_dL_dpostact_a(
        dL_do_pool, W2, pool_topk_weight, tile_to_expert_list, tile_m
    )
    abs_thresh, rel_thresh = 1e-2, 5e-2
    for t in range(total_tiles):
        diff = (dL_dpostact_a[t].float() - ref[t].float()).abs()
        rel = diff / (ref[t].float().abs() + 1e-3)
        bad = (diff > abs_thresh) & (rel > rel_thresh)
        assert not bad.any(), (
            f"tile {t}: expert={tile_to_expert_list[t]}, "
            f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
            f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
            f"max rel diff {rel.max().item():.4f}"
        )
    assert (bwd_a_ready == seq).all(), (
        f"bwd_a_ready not all set; bad indices "
        f"{(bwd_a_ready != seq).nonzero().squeeze().tolist()}"
    )


def test_streaming_moe_y_bwd_producer_consumer(device):
    """Kernel Y bwd on its own stream spins on bwd_y_ready while a producer
    kernel on a separate stream release-stores dispatch_seq slot by slot
    with delays between fires. Mirrors fwd kernel A's producer-consumer test;
    fire_tiles_with_delay is generic over int64 ready arrays so we reuse it.
    """
    from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_a import (
        fire_tiles_with_delay,
    )
    from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_y_bwd import (
        streaming_moe_y_bwd,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 128
    tile_to_expert_list = [0, 0, 1, 2, 2, 3]
    total_tiles = len(tile_to_expert_list)
    TK_padded = total_tiles * tile_m
    seq = 13

    dtype = torch.bfloat16
    torch.manual_seed(11)
    dL_do_pool = torch.randn(TK_padded, H, dtype=dtype, device=device)
    W2 = torch.randn(E_local, H, I, dtype=dtype, device=device).mul_(0.02)
    dL_dpostact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)
    pool_topk_weight = torch.linspace(
        -0.5, 1.5, TK_padded, dtype=torch.float32, device=device
    )

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    bwd_y_ready = _make_ready(total_tiles, dispatch_seq=0, device=device, fired=False)
    bwd_a_ready = _make_ready(total_tiles, dispatch_seq=0, device=device, fired=False)

    # Pre-warm the producer JIT compile so the host doesn't block during the
    # concurrent launch (use dispatch_seq=999 then reset).
    fire_tiles_with_delay(bwd_y_ready, dispatch_seq=999, delay_us=0)
    torch.cuda.synchronize()
    bwd_y_ready.zero_()
    torch.cuda.synchronize()

    consumer_stream = torch.cuda.Stream()
    producer_stream = torch.cuda.Stream()

    with torch.cuda.stream(consumer_stream):
        streaming_moe_y_bwd(
            dL_do_pool,
            W2,
            dL_dpostact_a,
            pool_topk_weight,
            tile_id_to_expert,
            expert_pool_block_offset,
            bwd_y_ready,
            bwd_a_ready,
            dispatch_seq=seq,
            tile_m=tile_m,
            tile_n=tile_n,
        )
    with torch.cuda.stream(producer_stream):
        fire_tiles_with_delay(bwd_y_ready, dispatch_seq=seq, delay_us=50)

    torch.cuda.synchronize()

    ref = _ref_dL_dpostact_a(
        dL_do_pool, W2, pool_topk_weight, tile_to_expert_list, tile_m
    )
    abs_thresh, rel_thresh = 1e-2, 5e-2
    for t in range(total_tiles):
        diff = (dL_dpostact_a[t].float() - ref[t].float()).abs()
        rel = diff / (ref[t].float().abs() + 1e-3)
        bad = (diff > abs_thresh) & (rel > rel_thresh)
        assert not bad.any(), (
            f"tile {t}: expert={tile_to_expert_list[t]}, "
            f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
            f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
            f"max rel diff {rel.max().item():.4f}"
        )
    assert (bwd_a_ready == seq).all(), (
        f"bwd_a_ready not all set under producer-consumer; bad indices "
        f"{(bwd_a_ready != seq).nonzero().squeeze().tolist()}"
    )


if __name__ == "__main__":
    dev = torch.device("cuda")
    test_streaming_moe_y_bwd_compiles(dev)
    print("compile OK")
    test_streaming_moe_y_bwd_single_tile(dev)
    print("single-tile PASS")
    test_streaming_moe_y_bwd_multi_tile_static(dev)
    print("multi-tile-static PASS")
    test_streaming_moe_y_bwd_producer_consumer(dev)
    print("producer-consumer PASS")
