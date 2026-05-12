"""Tests for the streaming-MoE kernel Y bwd (pool layout).

Mirrors `test_kernel_a.py`'s structure since kernel_y_bwd uses the
same streaming machinery (linear-claim scheduler, per-tile acquire-spin,
per-tile release-store after multi-pid_n gating).

**One sequence number, not two.** Backward shares a single `dispatch_seq`
across acquire-and-release — fwd's
two-value pattern (`dispatch_seq=1` for the producer, `compute_seq=N` for the
release) collapses to one `dispatch_seq` in bwd because the orchestrator
reuses `handle.dispatch_seq` end-to-end. So tests pre-set
`bwd_y_ready = seq` and pass `dispatch_seq = seq` and assert
`bwd_a_ready == seq` — same value end-to-end. Mismatched values (the trap
this comment is here to flag) deadlock the kernel's per-tile acquire-spin.

Reference math (SwiGLU bwd folded into the epilogue):
  g[slot, :]                = dL_do_pool[slot] @ W2[e]            (unweighted)
  dpostact[slot, :I]        = pool_topk_weight[slot] * g[slot, :]
  (dgate, dup, postact)     = dswiglu(gate, up, dpostact)
  dL_dswiglu_in[slot, 2n]   = dgate;  dL_dswiglu_in[slot, 2n+1] = dup
  dL_dweight[slot]          = Σ_n postact[slot, n] * g[slot, n]   (UNWEIGHTED g)

The kernel runs an NN GEMM (`dL_do_pool @ W2`) by passing W2 permuted to
(I, H, E_local) with I contiguous (n-major B); the reference computes the
same product in plain torch using `W2[e]` directly. The bf16 → fp32 → matmul
→ bf16 round-trip matches what the kernel's Float32 accumulator + bf16
recast (via the f32-recast trick on mD) produces up to the standard
MMA-ordering tolerance.
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


def _ref_dL_dswiglu_in(
    dL_do_pool: torch.Tensor,  # (TK_padded, H) bf16
    W2: torch.Tensor,  # (E_local, H, I) bf16
    pool_topk_weight: torch.Tensor,  # (TK_padded,) fp32
    preact_a: torch.Tensor,  # (total_tiles, tile_m, 2*I) bf16
    tile_to_expert_list: list[int],
    tile_m: int,
) -> torch.Tensor:
    """Eager torch reference for `dL_dswiglu_in`:

      g          = dL_do_pool @ W2[e]                       (unweighted)
      dpostact   = pool_topk_weight * g                     (per-row scale)
      dgate      = silu_grad(gate) * up * dpostact
      dup        = silu(gate) * dpostact
      dL_dswiglu_in[slot, 2n]   = dgate
      dL_dswiglu_in[slot, 2n+1] = dup

    Returns bf16 (total_tiles, tile_m, 2*I) — same shape as preact_a.
    """
    total_tiles = len(tile_to_expert_list)
    I = W2.shape[2]
    two_I = 2 * I
    out = torch.zeros(
        total_tiles, tile_m, two_I, dtype=dL_do_pool.dtype, device=dL_do_pool.device
    )
    for t in range(total_tiles):
        e = tile_to_expert_list[t]
        rows = slice(t * tile_m, (t + 1) * tile_m)
        # NN matmul in fp32 to match the kernel's Float32 accumulator.
        g = dL_do_pool[rows].float() @ W2[e].float()  # (tile_m, I)
        w = pool_topk_weight[rows].view(tile_m, 1)
        dpostact = w * g  # (tile_m, I)
        gate = preact_a[t, :, 0::2].float()
        up = preact_a[t, :, 1::2].float()
        # SwiGLU bwd: same formulas as quack.activation.dswiglu.
        sigmoid_x = torch.sigmoid(gate)
        silu_x = gate * sigmoid_x
        d_silu_x_dout = (sigmoid_x - silu_x * sigmoid_x) * dpostact + silu_x * dpostact
        dgate = d_silu_x_dout * up
        dup = silu_x * dpostact
        out[t, :, 0::2] = dgate.to(dL_do_pool.dtype)
        out[t, :, 1::2] = dup.to(dL_do_pool.dtype)
    return out


def _ref_dL_dweight(
    dL_do_pool: torch.Tensor,  # (TK_padded, H) bf16
    W2: torch.Tensor,  # (E_local, H, I) bf16
    preact_a: torch.Tensor,  # (total_tiles, tile_m, 2*I) bf16
    tile_to_expert_list: list[int],
    tile_m: int,
) -> torch.Tensor:
    """Eager torch reference for per-slot dL/dweight = postact · g (sum over I).

    g  = dL_do_pool[tile] @ W2[e]               (unweighted; matches kernel)
    postact[m, i] = silu(preact[m, 2i]) * preact[m, 2i+1]
                                                (paired-N, mirrors fwd kernel A)
    dL/dweight[slot] = Σ_i postact[m, i] * g[m, i]
    Returns fp32 (TK_padded,).
    """
    total_tiles = len(tile_to_expert_list)
    TK_padded = total_tiles * tile_m
    out = torch.zeros(TK_padded, dtype=torch.float32, device=dL_do_pool.device)
    for t in range(total_tiles):
        e = tile_to_expert_list[t]
        rows = slice(t * tile_m, (t + 1) * tile_m)
        g = dL_do_pool[rows].float() @ W2[e].float()  # (tile_m, I)
        # Recompute postact from preact, paired-N: gate at 0::2, up at 1::2.
        gate = preact_a[t, :, 0::2].float()
        up = preact_a[t, :, 1::2].float()
        postact = torch.nn.functional.silu(gate) * up  # (tile_m, I)
        out[rows] = (postact * g).sum(dim=-1)
    return out


def _alloc_preact(total_tiles, tile_m, I, dtype, device, seed):
    """Allocate a (total_tiles, tile_m, 2*I) bf16 preact buffer with reproducible
    randn values."""
    g = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(total_tiles, tile_m, 2 * I, dtype=dtype, device=device, generator=g)
        * 0.5
    )


def _alloc_dL_dweight(TK_padded, device):
    """ColVecReduceAtomic destination — flat (TK_padded,) fp32 zero-init.

    Kernel atomic-adds per-pid_n stripe partials into this buffer via
    ``red.global.add.f32``; collapsed in-kernel (no post-hoc .sum() needed).
    """
    return torch.zeros(TK_padded, dtype=torch.float32, device=device)


def _alloc_postact_a_for_dW2(total_tiles, tile_m, I, dtype, device):
    """mPostAct destination — bf16 (total_tiles, tile_m, I) empty buffer.

    Kernel writes weighted postact (= postact * pool_topk_weight) into
    each pid_n CTA's (tile_M, tile_N) slab via TMA. Plain bf16 store, no
    f32-recast — same path GemmActMixin uses for fwd postact.
    """
    return torch.empty(total_tiles, tile_m, I, dtype=dtype, device=device)


def _ref_postact_a_for_dW2(
    pool_topk_weight: torch.Tensor,  # (TK_padded,) fp32
    preact_a: torch.Tensor,  # (total_tiles, tile_m, 2*I) bf16
    tile_m: int,
) -> torch.Tensor:
    """Eager torch reference for postact_a_for_dW2 = silu(gate)*up*w in bf16.

    Mirrors what kernel_y_bwd's epilogue computes (postact byproduct of
    dswiglu, multiplied by per-row pool_topk_weight, cast to bf16).
    """
    total_tiles, _, two_I = preact_a.shape
    I = two_I // 2
    TK_padded = total_tiles * tile_m
    out = torch.empty(
        total_tiles, tile_m, I, dtype=preact_a.dtype, device=preact_a.device
    )
    w = pool_topk_weight.view(total_tiles, tile_m, 1)
    gate = preact_a[..., 0::2].float()
    up = preact_a[..., 1::2].float()
    postact = torch.nn.functional.silu(gate) * up
    out.copy_((postact * w).to(preact_a.dtype))
    _ = TK_padded  # silence unused-name warning; helper documents shape
    return out


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def test_streaming_moe_y_bwd_compiles(device):
    """JIT-compile only (no launch) for a representative production-shape config."""
    from stream_ep.stream_moe.kernel_y_bwd import (
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
    dL_dswiglu_in = torch.zeros(total_tiles, tile_m, 2 * I, dtype=dtype, device=device)
    pool_topk_weight = torch.ones(TK_padded, dtype=torch.float32, device=device)
    pool_recv_token = torch.arange(TK_padded, dtype=torch.int32, device=device)
    preact_a = _alloc_preact(total_tiles, tile_m, I, dtype, device, seed=42)
    dL_dweight = _alloc_dL_dweight(TK_padded, device)
    postact_a_for_dW2 = _alloc_postact_a_for_dW2(total_tiles, tile_m, I, dtype, device)

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
            dL_dswiglu_in,
            postact_a_for_dW2,
            pool_topk_weight,
            pool_recv_token,
            preact_a,
            dL_dweight,
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
    from stream_ep.stream_moe.kernel_y_bwd import (
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
    dL_dswiglu_in = torch.zeros(total_tiles, tile_m, 2 * I, dtype=dtype, device=device)
    # Mix of weights (some negative, some > 1) to exercise the multiplicative
    # ColVec broadcast — `*=` vs `+=` would silently pass under all-ones.
    pool_topk_weight = torch.linspace(
        -0.5, 1.5, TK_padded, dtype=torch.float32, device=device
    )
    pool_recv_token = torch.arange(TK_padded, dtype=torch.int32, device=device)
    preact_a = _alloc_preact(total_tiles, tile_m, I, dtype, device, seed=42)
    dL_dweight = _alloc_dL_dweight(TK_padded, device)
    postact_a_for_dW2 = _alloc_postact_a_for_dW2(total_tiles, tile_m, I, dtype, device)

    tile_to_expert_list = [chosen_expert]
    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    bwd_y_ready = _make_ready(total_tiles, dispatch_seq=seq, device=device)
    bwd_a_ready = _make_ready(total_tiles, dispatch_seq=0, device=device, fired=False)

    streaming_moe_y_bwd(
        dL_do_pool,
        W2,
        dL_dswiglu_in,
        postact_a_for_dW2,
        pool_topk_weight,
        pool_recv_token,
        preact_a,
        dL_dweight,
        tile_id_to_expert,
        expert_pool_block_offset,
        bwd_y_ready,
        bwd_a_ready,
        dispatch_seq=seq,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    ref = _ref_dL_dswiglu_in(
        dL_do_pool, W2, pool_topk_weight, preact_a, tile_to_expert_list, tile_m
    )
    # bf16 GEMM ordering noise + bf16 cast through dswiglu's intermediate
    # fp32 ops + bf16x2 packing on output. Use the same "fail iff BOTH atol
    # AND rtol violated" pattern fwd kernel Y uses.
    diff = (dL_dswiglu_in.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-3)
    abs_thresh, rel_thresh = 1e-2, 5e-2
    bad = (diff > abs_thresh) & (rel > rel_thresh)
    assert not bad.any(), (
        f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
        f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
        f"max rel diff {rel.max().item():.4f}"
    )
    # dL/dweight: kernel-side per-pid_n red.global.add.f32 atomic-add
    # collapsed across N-stripes (no post-hoc .sum()). Compare against
    # eager `Σ_i postact[i] * g[i]` reference. Tolerance scales with I
    # (the reduction length): per-element noise multiplied by ~sqrt(I)
    # for sum-of-squares-like patterns.
    dL_dweight_ref = _ref_dL_dweight(
        dL_do_pool, W2, preact_a, tile_to_expert_list, tile_m
    )
    diff_w = (dL_dweight - dL_dweight_ref).abs()
    rel_w = diff_w / (dL_dweight_ref.abs() + 1e-3)
    abs_thresh_w, rel_thresh_w = 5e-2, 5e-2
    bad_w = (diff_w > abs_thresh_w) & (rel_w > rel_thresh_w)
    assert not bad_w.any(), (
        f"dL/dweight: {bad_w.sum().item()} elements exceed both "
        f"rtol={rel_thresh_w} and atol={abs_thresh_w}; "
        f"max abs diff {diff_w.max().item():.4f}, "
        f"max rel diff {rel_w.max().item():.4f}"
    )
    # postact_a_for_dW2: kernel-side mPostAct TMA store (= postact * w in bf16).
    postact_ref = _ref_postact_a_for_dW2(pool_topk_weight, preact_a, tile_m)
    diff_p = (postact_a_for_dW2.float() - postact_ref.float()).abs()
    rel_p = diff_p / (postact_ref.float().abs() + 1e-3)
    bad_p = (diff_p > 1e-2) & (rel_p > 5e-2)
    assert not bad_p.any(), (
        f"postact_a_for_dW2: {bad_p.sum().item()} elements exceed both "
        f"rtol=5e-2 and atol=1e-2; max abs {diff_p.max().item():.4f}, "
        f"max rel {rel_p.max().item():.4f}"
    )
    assert (bwd_a_ready == seq).all(), (
        f"bwd_a_ready not all set to dispatch_seq={seq} (per-tile release "
        f"didn't fire); got unique values {bwd_a_ready.unique().tolist()}"
    )


def _assert_dL_dswiglu_in_per_tile(
    dL_dswiglu_in, ref, tile_to_expert_list, abs_thresh=1e-2, rel_thresh=5e-2
):
    for t in range(len(tile_to_expert_list)):
        diff = (dL_dswiglu_in[t].float() - ref[t].float()).abs()
        rel = diff / (ref[t].float().abs() + 1e-3)
        bad = (diff > abs_thresh) & (rel > rel_thresh)
        assert not bad.any(), (
            f"tile {t}: expert={tile_to_expert_list[t]}, "
            f"{bad.sum().item()} elements exceed both rtol={rel_thresh} and "
            f"atol={abs_thresh}; max abs diff {diff.max().item():.4f}, "
            f"max rel diff {rel.max().item():.4f}"
        )


def test_streaming_moe_y_bwd_multi_tile_static(device):
    """total_tiles=N>1 spread across multiple experts. Validates per-tile
    expert selection (W2[expert_id] varies via tile_id_to_expert) and
    persistent kernel termination via the linear-claim bounds check.
    """
    from stream_ep.stream_moe.kernel_y_bwd import (
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
    dL_dswiglu_in = torch.zeros(total_tiles, tile_m, 2 * I, dtype=dtype, device=device)
    pool_topk_weight = torch.linspace(
        -0.5, 1.5, TK_padded, dtype=torch.float32, device=device
    )
    pool_recv_token = torch.arange(TK_padded, dtype=torch.int32, device=device)
    preact_a = _alloc_preact(total_tiles, tile_m, I, dtype, device, seed=43)
    dL_dweight = _alloc_dL_dweight(TK_padded, device)
    postact_a_for_dW2 = _alloc_postact_a_for_dW2(total_tiles, tile_m, I, dtype, device)

    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    bwd_y_ready = _make_ready(total_tiles, dispatch_seq=seq, device=device)
    bwd_a_ready = _make_ready(total_tiles, dispatch_seq=0, device=device, fired=False)

    streaming_moe_y_bwd(
        dL_do_pool,
        W2,
        dL_dswiglu_in,
        postact_a_for_dW2,
        pool_topk_weight,
        pool_recv_token,
        preact_a,
        dL_dweight,
        tile_id_to_expert,
        expert_pool_block_offset,
        bwd_y_ready,
        bwd_a_ready,
        dispatch_seq=seq,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    ref = _ref_dL_dswiglu_in(
        dL_do_pool, W2, pool_topk_weight, preact_a, tile_to_expert_list, tile_m
    )
    _assert_dL_dswiglu_in_per_tile(dL_dswiglu_in, ref, tile_to_expert_list)

    # dL/dweight: per-stripe partials summed across the n-stripe dim.
    dL_dweight_ref = _ref_dL_dweight(
        dL_do_pool, W2, preact_a, tile_to_expert_list, tile_m
    )
    diff_w = (dL_dweight - dL_dweight_ref).abs()
    rel_w = diff_w / (dL_dweight_ref.abs() + 1e-3)
    abs_thresh_w, rel_thresh_w = 5e-2, 5e-2
    bad_w = (diff_w > abs_thresh_w) & (rel_w > rel_thresh_w)
    assert not bad_w.any(), (
        f"dL/dweight: {bad_w.sum().item()} elements exceed both "
        f"rtol={rel_thresh_w} and atol={abs_thresh_w}; "
        f"max abs diff {diff_w.max().item():.4f}, "
        f"max rel diff {rel_w.max().item():.4f}"
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
    from stream_ep.stream_moe.kernel_a import (
        fire_tiles_with_delay,
    )
    from stream_ep.stream_moe.kernel_y_bwd import (
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
    dL_dswiglu_in = torch.zeros(total_tiles, tile_m, 2 * I, dtype=dtype, device=device)
    pool_topk_weight = torch.linspace(
        -0.5, 1.5, TK_padded, dtype=torch.float32, device=device
    )
    pool_recv_token = torch.arange(TK_padded, dtype=torch.int32, device=device)
    preact_a = _alloc_preact(total_tiles, tile_m, I, dtype, device, seed=44)
    dL_dweight = _alloc_dL_dweight(TK_padded, device)
    postact_a_for_dW2 = _alloc_postact_a_for_dW2(total_tiles, tile_m, I, dtype, device)

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
            dL_dswiglu_in,
            postact_a_for_dW2,
            pool_topk_weight,
            pool_recv_token,
            preact_a,
            dL_dweight,
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

    ref = _ref_dL_dswiglu_in(
        dL_do_pool, W2, pool_topk_weight, preact_a, tile_to_expert_list, tile_m
    )
    _assert_dL_dswiglu_in_per_tile(dL_dswiglu_in, ref, tile_to_expert_list)

    dL_dweight_ref = _ref_dL_dweight(
        dL_do_pool, W2, preact_a, tile_to_expert_list, tile_m
    )
    diff_w = (dL_dweight - dL_dweight_ref).abs()
    rel_w = diff_w / (dL_dweight_ref.abs() + 1e-3)
    abs_thresh_w, rel_thresh_w = 5e-2, 5e-2
    bad_w = (diff_w > abs_thresh_w) & (rel_w > rel_thresh_w)
    assert not bad_w.any(), (
        f"dL/dweight: {bad_w.sum().item()} elements exceed both "
        f"rtol={rel_thresh_w} and atol={abs_thresh_w}; "
        f"max abs diff {diff_w.max().item():.4f}, "
        f"max rel diff {rel_w.max().item():.4f}"
    )

    assert (bwd_a_ready == seq).all(), (
        f"bwd_a_ready not all set under producer-consumer; bad indices "
        f"{(bwd_a_ready != seq).nonzero().squeeze().tolist()}"
    )


def _assert_dL_dweight_real_slots(
    dL_dweight,
    dL_dweight_ref,
    real_mask,
    *,
    abs_thresh: float = 5e-2,
    rel_thresh: float = 5e-2,
    msg_prefix: str = "",
):
    """Compare per-slot dL/dweight ONLY at real (non-padding) slots."""
    diff = (dL_dweight - dL_dweight_ref).abs()
    rel = diff / (dL_dweight_ref.abs() + 1e-3)
    bad = (diff > abs_thresh) & (rel > rel_thresh) & real_mask
    assert not bad.any(), (
        f"{msg_prefix}dL/dweight (real slots only): "
        f"{bad.sum().item()} bad / {real_mask.sum().item()} real slots; "
        f"max abs diff {diff[real_mask].max().item():.4f}, "
        f"max rel diff {rel[real_mask].max().item():.4f}"
    )


def test_streaming_moe_y_bwd_dense_padding(device):
    """Orchestrator-shaped config: dense layout (1 tile per expert across
    E_local experts), tile_n=256 (matches `tile_n_a` orchestrator default),
    production-shape H/I, and PADDING ROWS WITH GARBAGE preact / dL_do_pool —
    mirrors what fwd kernel A and dispatch_grads leave for slots where
    `pool_recv_token == -1`.

    The reference dL/dweight is checked ONLY at real slots; padding-slot
    values are unused downstream (combine_grads only reads
    `weight_grads[recv_token_to_slots[r, k]]` which is always a real slot or
    -1 → 0).

    The pool_topk_weight is zero for padding (matching the C++ runtime's
    Z_pre memset of the dispatch bundle); real slots get random nonzero
    weights. This exercises the SAME numerical path the orchestrator hits.
    """
    from stream_ep.stream_moe.kernel_y_bwd import (
        streaming_moe_y_bwd,
    )

    # Production-ish shape (matches profile_pipeline.py defaults that the
    # orchestrator's validate_multi_iter exercises).
    H, I, E_local = 2048, 2048, 8
    tile_m, tile_n = 128, 256  # tile_n=256 matches tile_n_a default
    tile_to_expert_list = list(range(E_local))  # dense: 1 tile per expert
    total_tiles = len(tile_to_expert_list)
    TK_padded = total_tiles * tile_m
    real_per_tile = 32  # mirrors seq_len=64 single-hot routing on rank 0
    seq = 17

    dtype = torch.bfloat16
    g = torch.Generator(device=device).manual_seed(123)

    # Build a real-slot mask: first `real_per_tile` rows of each tile are real,
    # the rest are padding. pool_recv_token = -1 for padding (matches fwd
    # Pass B's N-region memset).
    real_mask_3d = torch.zeros(total_tiles, tile_m, dtype=torch.bool, device=device)
    real_mask_3d[:, :real_per_tile] = True
    real_mask = real_mask_3d.reshape(TK_padded)

    # Garbage data over the WHOLE pool (mirrors what fwd kernel A's mD store
    # and dispatch_grads's TMA writes leave: real rows get valid values, padding
    # rows are torch::empty() garbage). We populate ALL rows with random data —
    # the kernel will compute on padding rows too, but real-slot output is the
    # only thing we check.
    dL_do_pool = (
        torch.randn(TK_padded, H, dtype=dtype, device=device, generator=g) * 0.1
    )
    # Plant inf at padding rows of dL_do_pool so the GEMM through a padding row
    # produces inf and dswiglu(*,*,inf) → NaN; if the mPaddingMask predicate is
    # broken, the NaNs will leak into mD / mPostAct via the TMA store and the
    # padding-row asserts below will fail.
    dL_do_pool[~real_mask] = float("inf")
    W2 = torch.randn(E_local, H, I, dtype=dtype, device=device, generator=g).mul_(0.02)
    # Pre-fill the kernel outputs with a sentinel so we can detect "kernel
    # didn't write" (sentinel survives) vs "kernel wrote zero" (predicate
    # fired) vs "kernel wrote garbage" (predicate broken). The TMA store
    # overwrites the whole tile regardless, so any sentinel surviving means
    # the kernel didn't run this row.
    dL_dswiglu_in = torch.full(
        (total_tiles, tile_m, 2 * I), float("nan"), dtype=dtype, device=device
    )
    preact_a = _alloc_preact(total_tiles, tile_m, I, dtype, device, seed=124)

    # pool_topk_weight: real-slot rows have nonzero values (matching Pass B),
    # padding rows are 0 (matching the Z_pre memset in stream_ep.cpp).
    pool_topk_weight = torch.zeros(TK_padded, dtype=torch.float32, device=device)
    pool_topk_weight[real_mask] = torch.linspace(
        -0.5, 1.5, real_mask.sum().item(), dtype=torch.float32, device=device
    )
    # pool_recv_token: real slots get an arbitrary nonneg id; padding slots
    # are -1 (matching fwd Pass B's N-region 0xFF memset). The kernel's
    # mPaddingMask predicate keys on `recv_token < 0` to zero out (dgate, dup,
    # postact, g) at padding rows BEFORE the colvec_reduce_accumulate / weight
    # multiply / mD / mPostAct stores.
    pool_recv_token = torch.full((TK_padded,), -1, dtype=torch.int32, device=device)
    pool_recv_token[real_mask] = torch.arange(
        real_mask.sum().item(), dtype=torch.int32, device=device
    )

    dL_dweight = _alloc_dL_dweight(TK_padded, device)
    # NaN sentinel so the post-kernel padding-row asserts can distinguish
    # "predicate zeroed the padding rows" from "kernel skipped these rows".
    postact_a_for_dW2 = torch.full(
        (total_tiles, tile_m, I), float("nan"), dtype=dtype, device=device
    )
    tile_id_to_expert, expert_pool_block_offset = _make_tile_metadata(
        tile_to_expert_list, E_local, device
    )
    bwd_y_ready = _make_ready(total_tiles, dispatch_seq=seq, device=device)
    bwd_a_ready = _make_ready(total_tiles, dispatch_seq=0, device=device, fired=False)

    streaming_moe_y_bwd(
        dL_do_pool,
        W2,
        dL_dswiglu_in,
        postact_a_for_dW2,
        pool_topk_weight,
        pool_recv_token,
        preact_a,
        dL_dweight,
        tile_id_to_expert,
        expert_pool_block_offset,
        bwd_y_ready,
        bwd_a_ready,
        dispatch_seq=seq,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    torch.cuda.synchronize()

    # Reference dL_dweight per slot (computed on ALL rows; we filter to real
    # later). Uses the SAME garbage values the kernel sees, so any per-slot
    # disagreement is a kernel bug, not a missing-input bug.
    dL_dweight_ref = _ref_dL_dweight(
        dL_do_pool, W2, preact_a, tile_to_expert_list, tile_m
    )
    _assert_dL_dweight_real_slots(
        dL_dweight, dL_dweight_ref, real_mask, msg_prefix="dense_padding "
    )

    # dL/dswiglu_in for real slots — same dual-threshold rule as the
    # multi-tile-static test, but iterate per tile and slice the real-row
    # subrange.
    ref_dswiglu = _ref_dL_dswiglu_in(
        dL_do_pool, W2, pool_topk_weight, preact_a, tile_to_expert_list, tile_m
    )
    for t in range(total_tiles):
        diff = (
            dL_dswiglu_in[t, :real_per_tile].float()
            - ref_dswiglu[t, :real_per_tile].float()
        ).abs()
        rel = diff / (ref_dswiglu[t, :real_per_tile].float().abs() + 1e-3)
        bad = (diff > 1e-2) & (rel > 5e-2)
        assert not bad.any(), (
            f"dense_padding tile {t}: dL/dswiglu_in (real rows only) "
            f"{bad.sum().item()} bad / {real_per_tile * 2 * I}; "
            f"max_abs={diff.max().item():.4f} max_rel={rel.max().item():.4f}"
        )

    assert (bwd_a_ready == seq).all(), (
        f"dense_padding bwd_a_ready not all set; bad indices "
        f"{(bwd_a_ready != seq).nonzero().squeeze().tolist()}"
    )

    # Padding-row predicate check: with the mPaddingMask predicate firing at
    # `pool_recv_token < 0`, mD (`dL_dswiglu_in`) and mPostAct
    # (`postact_a_for_dW2`) must be exactly zero at padding rows — even though
    # `dL_do_pool[padding] = +inf` planted upstream would, without the
    # predicate, drive `dswiglu(*,*,inf) → NaN` into both outputs via the TMA
    # store. Both buffers were pre-filled with NaN so a kernel that skips these
    # rows entirely (rather than writing zero) would still fail this check.
    pad_mask = ~real_mask
    pad_dswiglu = dL_dswiglu_in.view(TK_padded, 2 * I)[pad_mask]
    assert (pad_dswiglu == 0).all(), (
        f"dense_padding: dL_dswiglu_in not zero at padding rows — predicate "
        f"didn't fire. {(pad_dswiglu != 0).sum().item()} nonzero / "
        f"{pad_dswiglu.numel()}"
    )
    pad_postact = postact_a_for_dW2.view(TK_padded, I)[pad_mask]
    assert (pad_postact == 0).all(), (
        f"dense_padding: postact_a_for_dW2 not zero at padding rows — "
        f"predicate didn't fire. {(pad_postact != 0).sum().item()} nonzero "
        f"/ {pad_postact.numel()}"
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
    test_streaming_moe_y_bwd_dense_padding(dev)
    print("dense-padding PASS")
