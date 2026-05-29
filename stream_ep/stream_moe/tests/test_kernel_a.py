"""Tests for the streaming-MoE kernel A (pool layout).

These tests exercise the streaming-design properties directly:
  - test_compile: kernel compiles for a representative shape.
  - test_single_tile: total_tiles=1, arrival pre-fired. Numerics: matmul +
    SwiGLU on pool[0:tile_m, :] vs an eager pytorch reference.
  - test_multi_tile_static: total_tiles=N spread across multiple experts via
    expert_pool_block_offset; all per-tile arrivals pre-fired, persistent
    CTAs absorb all tiles.
  - test_producer_consumer: producer kernel on a different stream fires
    pool_arrival_count entries with delay; kernel A spins then drains.

Per-tile ready handshake:
  * (pool_arrival_count, pool_arrival_target) int32 pair. Consumer spins
    until ``pool_arrival_count[tile] == pool_arrival_target[tile]``.
  * Internal consumer_head[1] int32 (allocated inside `streaming_moe_a`) —
    single global atomic-add counter for linear claims.

Pool layout:
  * pool[total_tiles * tile_m, H] is expert-major and BLOCK_M-padded; tile t
    occupies rows [t*tile_m, (t+1)*tile_m) and belongs to
    expert_id = tile_id_to_expert[t].
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def _swiglu_ref(h_two_I: torch.Tensor) -> torch.Tensor:
    """QuACK's gated epilogue pairs ADJACENT N-elements: gate = h[..., ::2],
    up = h[..., 1::2], output = silu(gate) * up. See gemm_act.py
    GemmGatedMixin.epi_visit_subtile.
    """
    gate = h_two_I[..., 0::2]
    up = h_two_I[..., 1::2]
    return F.silu(gate) * up


def _make_pool_arrival(
    total_tiles: int, device, fired: bool = True, target_val: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate (pool_arrival_count, pool_arrival_target) int32 pair.

    pool_arrival_target is set to ``target_val`` (per-tile firing target).
    pool_arrival_count is ``target_val`` if fired=True (scheduler spin
    passes immediately) or zero otherwise (test producer fires by
    `red.release.gpu.global.add.s32`'ing ``target_val`` into each slot
    with a delay between fires — see ``fire_tiles_with_delay``).
    """
    pool_arrival_target = torch.full(
        (total_tiles,), target_val, dtype=torch.int32, device=device
    )
    init = target_val if fired else 0
    pool_arrival_count = torch.full(
        (total_tiles,), init, dtype=torch.int32, device=device
    )
    return pool_arrival_count, pool_arrival_target


def _make_tile_metadata(tile_to_expert_list, E_local, device):
    """Build (tile_id_to_expert, expert_pool_block_offset) from a list mapping
    each tile_id to its expert. Tiles must already be in expert-major order
    (i.e. all tiles for expert e come before any tile for expert e+1)."""
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


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def test_streaming_moe_a_compiles(device):
    """JIT-compile only (no launch) for a representative production-shape config."""
    from stream_ep.stream_moe.kernel_a import (
        streaming_moe_a,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 256
    total_tiles = 4
    TK_padded = total_tiles * tile_m

    dtype = torch.bfloat16
    pool = torch.randn(TK_padded, H, dtype=dtype, device=device)
    W1 = torch.randn(E_local, 2 * I, H, dtype=dtype, device=device).mul_(0.02)
    postact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)

    # All 4 tiles belong to expert 0.
    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        [0, 0, 0, 0], E_local, device
    )
    pool_arrival_count, pool_arrival_target = _make_pool_arrival(
        total_tiles, device=device
    )
    import quack.cache_utils as cu

    orig = cu.COMPILE_ONLY
    cu.COMPILE_ONLY = True
    try:
        streaming_moe_a(
            pool,
            W1,
            postact_a,
            expert_pool_block_offset,
            pool_arrival_count,
            pool_arrival_target,
            tile_m=tile_m,
            tile_n=tile_n,
        )
    finally:
        cu.COMPILE_ONLY = orig


def test_streaming_moe_a_single_tile(device):
    """total_tiles=1, arrival pre-fired. Validates the full kernel path:
    linear claim, scheduler 5-int payload, strided pool read, per-tile postact,
    expert lookup via tile_id_to_expert.
    """
    from stream_ep.stream_moe.kernel_a import (
        streaming_moe_a,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 256
    total_tiles = 1
    TK_padded = total_tiles * tile_m
    chosen_expert = 2

    dtype = torch.bfloat16
    torch.manual_seed(0)
    pool = torch.randn(TK_padded, H, dtype=dtype, device=device)
    W1 = torch.randn(E_local, 2 * I, H, dtype=dtype, device=device).mul_(0.02)
    postact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        [chosen_expert], E_local, device
    )
    pool_arrival_count, pool_arrival_target = _make_pool_arrival(
        total_tiles, device=device
    )

    streaming_moe_a(
        pool,
        W1,
        postact_a,
        expert_pool_block_offset,
        pool_arrival_count,
        pool_arrival_target,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    x_tile = pool[0:tile_m, :]
    h = x_tile.float() @ W1[chosen_expert].float().t()
    a_ref = _swiglu_ref(h).to(dtype)

    a_kernel = postact_a[0]
    diff = (a_kernel.float() - a_ref.float()).abs()
    rel = diff / (a_ref.float().abs() + 1e-3)
    assert (
        rel.max().item() < 5e-2
    ), f"max rel diff {rel.max().item():.4f}, max abs diff {diff.max().item():.4f}"


def test_streaming_moe_a_multi_tile_static(device):
    """total_tiles=N>1 spread across multiple experts via expert_pool_block_offset.
    Validates per-tile expert selection (W1[expert_id] varies via
    tile_id_to_expert) and persistent kernel termination via the linear-claim
    bounds check.
    """
    from stream_ep.stream_moe.kernel_a import (
        streaming_moe_a,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 256
    # Expert-major distribution: expert 0: 2 tiles, expert 1: 1 tile,
    # expert 2: 2 tiles, expert 3: 1 tile.  total_tiles = 6.
    tile_to_expert_list = [0, 0, 1, 2, 2, 3]
    total_tiles = len(tile_to_expert_list)
    TK_padded = total_tiles * tile_m

    dtype = torch.bfloat16
    torch.manual_seed(7)
    pool = torch.randn(TK_padded, H, dtype=dtype, device=device)
    W1 = torch.randn(E_local, 2 * I, H, dtype=dtype, device=device).mul_(0.02)
    postact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    pool_arrival_count, pool_arrival_target = _make_pool_arrival(
        total_tiles, device=device
    )

    streaming_moe_a(
        pool,
        W1,
        postact_a,
        expert_pool_block_offset,
        pool_arrival_count,
        pool_arrival_target,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    for t in range(total_tiles):
        e = tile_to_expert_list[t]
        x_tile = pool[t * tile_m : (t + 1) * tile_m, :]
        h = x_tile.float() @ W1[e].float().t()
        a_ref = _swiglu_ref(h).to(dtype)
        diff = (postact_a[t].float() - a_ref.float()).abs()
        rel = diff / (a_ref.float().abs() + 1e-3)
        assert rel.max().item() < 5e-2, (
            f"tile {t}: expert={e}, max rel diff {rel.max().item():.4f}, "
            f"max abs diff {diff.max().item():.4f}"
        )


def test_streaming_moe_a_with_preact(device):
    """``preact_a`` opt-in path: kernel A's standard mD TMA-store writes the
    pre-SwiGLU [2I] gate-up accumulator alongside the postact_a [I] postact
    write. This is the bwd-savings path — saving preact lets ``kernel_a_bwd``
    skip a recompute GEMM (~370 µs/layer at production) in exchange for the
    extra TMA-store traffic (overlaps with the GEMM mainloop).

    Multi-tile spread across multiple experts (mirrors the no-preact
    multi_tile test) so we hit per-expert W1 slabs and validate that the
    preact write lands at the right pool row for each tile_id.
    """
    from stream_ep.stream_moe.kernel_a import (
        streaming_moe_a,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 256
    tile_to_expert_list = [0, 0, 1, 2, 2, 3]
    total_tiles = len(tile_to_expert_list)
    TK_padded = total_tiles * tile_m

    dtype = torch.bfloat16
    torch.manual_seed(13)
    pool = torch.randn(TK_padded, H, dtype=dtype, device=device)
    W1 = torch.randn(E_local, 2 * I, H, dtype=dtype, device=device).mul_(0.02)
    postact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)
    preact_a = torch.zeros(total_tiles, tile_m, 2 * I, dtype=dtype, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    pool_arrival_count, pool_arrival_target = _make_pool_arrival(
        total_tiles, device=device
    )

    streaming_moe_a(
        pool,
        W1,
        postact_a,
        expert_pool_block_offset,
        pool_arrival_count,
        pool_arrival_target,
        preact_a=preact_a,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    # Per-tile postact + preact references.
    for t in range(total_tiles):
        e = tile_to_expert_list[t]
        x_tile = pool[t * tile_m : (t + 1) * tile_m, :]
        h_ref = (x_tile.float() @ W1[e].float().t()).to(dtype)  # [tile_m, 2I]
        a_ref = _swiglu_ref(h_ref).to(dtype)  # [tile_m, I]

        # Postact: same as the no-preact path — sanity check the SwiGLU output
        # didn't change just because we added the mD store.
        diff_a = (postact_a[t].float() - a_ref.float()).abs()
        rel_a = diff_a / (a_ref.float().abs() + 1e-3)
        assert (
            rel_a.max().item() < 5e-2
        ), f"tile {t}: postact mismatch, max rel diff {rel_a.max().item():.4f}"

        # Preact: the [2I] pre-SwiGLU accumulator. The eager reference is the
        # raw `pool @ W1[e].T` GEMM in fp32 cast to bf16 — exactly what kernel
        # A's mD TMA-store should land (no alpha/beta/RowVec/ColVec applied
        # since callers don't pass them).
        diff_h = (preact_a[t].float() - h_ref.float()).abs()
        rel_h = diff_h / (h_ref.float().abs() + 1e-3)
        assert rel_h.max().item() < 5e-2, (
            f"tile {t}: preact mismatch, expert={e}, "
            f"max rel diff {rel_h.max().item():.4f}, "
            f"max abs diff {diff_h.max().item():.4f}"
        )



def test_streaming_moe_a_producer_consumer(device):
    """Kernel A on compute_a_stream spins on pool_arrival_count while a
    producer kernel on a separate stream release-adds into each slot with
    delays between fires.
    """
    from stream_ep.stream_moe.kernel_a import (
        fire_tiles_with_delay,
        streaming_moe_a,
    )

    H, I, E_local = 128, 256, 4
    tile_m, tile_n = 128, 256
    tile_to_expert_list = [0, 0, 1, 2, 2, 3]
    total_tiles = len(tile_to_expert_list)
    TK_padded = total_tiles * tile_m

    dtype = torch.bfloat16
    torch.manual_seed(11)
    pool = torch.randn(TK_padded, H, dtype=dtype, device=device)
    W1 = torch.randn(E_local, 2 * I, H, dtype=dtype, device=device).mul_(0.02)
    postact_a = torch.zeros(total_tiles, tile_m, I, dtype=dtype, device=device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    pool_arrival_count, pool_arrival_target = _make_pool_arrival(
        total_tiles, device=device, fired=False
    )

    # Pre-warm the producer JIT compile so the host doesn't block during the
    # concurrent launch (one-shot pre-fire then reset).
    fire_tiles_with_delay(pool_arrival_count, pool_arrival_target, delay_us=0)
    torch.cuda.synchronize()
    pool_arrival_count.zero_()
    torch.cuda.synchronize()

    compute_a_stream = torch.cuda.Stream()
    producer_stream = torch.cuda.Stream()

    with torch.cuda.stream(compute_a_stream):
        streaming_moe_a(
            pool,
            W1,
            postact_a,
            expert_pool_block_offset,
            pool_arrival_count,
            pool_arrival_target,
            tile_m=tile_m,
            tile_n=tile_n,
        )
    with torch.cuda.stream(producer_stream):
        fire_tiles_with_delay(pool_arrival_count, pool_arrival_target, delay_us=50)

    torch.cuda.synchronize()

    for t in range(total_tiles):
        e = tile_to_expert_list[t]
        x_tile = pool[t * tile_m : (t + 1) * tile_m, :]
        h = x_tile.float() @ W1[e].float().t()
        a_ref = _swiglu_ref(h).to(dtype)
        diff = (postact_a[t].float() - a_ref.float()).abs()
        rel = diff / (a_ref.float().abs() + 1e-3)
        assert rel.max().item() < 5e-2, (
            f"tile {t}: expert={e}, max rel diff {rel.max().item():.4f}, "
            f"max abs diff {diff.max().item():.4f}"
        )


if __name__ == "__main__":
    dev = torch.device("cuda")
    test_streaming_moe_a_compiles(dev)
    print("compile OK")
    test_streaming_moe_a_single_tile(dev)
    print("single-tile PASS")
    test_streaming_moe_a_multi_tile_static(dev)
    print("multi-tile-static PASS")
    test_streaming_moe_a_with_preact(dev)
    print("with-preact PASS")
    test_streaming_moe_a_producer_consumer(dev)
    print("producer-consumer PASS")
