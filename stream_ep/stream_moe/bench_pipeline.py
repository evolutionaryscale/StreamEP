"""Bench the streaming pipeline up to kernel Y (dispatch → A → Y).

Setup
-----
- Run Buffer.dispatch once to produce a StreamingHandle (multi-GPU required
  for DeepEP).
- Pool layout puts data into expert-major BLOCK_M-padded tiles. Both kernel A
  (gemm_gated) and kernel Y (gemm + atomic-scatter into o[T_recv, H]) read
  from the pool via the standard varlen_m strided TMA path.
- Time three things, all on a single GPU:
    * dispatch only (dispatch_stream)
    * each kernel in isolation with its ready signal pre-set (compute_a/y stream)
    * end-to-end pipeline (4 streams: dispatch, kernel A, kernel Y, combine —
      real producer-consumer release/acquire across them)
- Reference baselines:
    * kernel A vs `quack.gemm_act.gemm_gated` (same per-tile compute, no
      streaming scheduler).
    * kernel Y vs `quack.gemm.gemm` + a separate `index_add_` scatter (so
      `gemm` produces y_per_tile[T_recv, H] and we accumulate o[r] += w * y[r];
      this is a strict per-token equivalent at the ~ms scale and a fair
      timing baseline for the fused atomic-scatter).

Launch:
    torchrun --nproc_per_node=2 \\
        -m stream_ep.stream_moe.bench_pipeline
"""

import argparse

import torch
import torch.distributed as torch_dist
from quack.gemm import gemm
from quack.gemm_act import gemm_act

from stream_ep.stream_moe.profile_pipeline import (
    make_buffer,
    make_uniform_topk_idx,
)
from stream_ep.stream_moe.kernel_a import (
    streaming_moe_a,
)
from stream_ep.stream_moe.kernel_a_bwd import (
    streaming_moe_a_bwd,
)
from stream_ep.stream_moe.kernel_y import (
    streaming_moe_y,
)
from stream_ep.stream_moe.kernel_y_bwd import (
    streaming_moe_y_bwd,
)
from stream_ep.stream_moe.stream_moe import (
    make_streams,
    stream_moe_func,
)
from stream_ep.stream_moe.profile_pipeline import (
    barrier,
    get_global_rank,
    get_world_size,
    init_distributed,
    rank_zero_only,
    rank_zero_print,
)

H = 2048
I = 2048
NUM_EXPERTS = 64
SEQ_LEN_PER_RANK = 8192
TOPK = 4
DTYPE = torch.bfloat16
NUM_SMS = 80  # DeepEP num_sms (channels = num_sms / 2; max = num_device_sms,
# i.e. 132 on H100). Sweep at the current pipeline's full
# fwd+bwd footprint (kernel-bounded measurement, profile traces,
# 8×H100 production shape) shows fwd_e2e and fwd+bwd_e2e cluster
# within ~20 µs across {80, 96, 112}, with 80 the slight winner
# (-52 µs fwd, -70 µs fwd+bwd vs the 132 ceiling). Below 80,
# bwd combine_grads bloats faster than fwd improves; above 80,
# the dispatch grid is too wide to leave SMs for kernel A's
# CTAs to land mid-dispatch (gap_dispatch_to_a +6 µs at 132
# vs −22 µs at 80 — i.e. kernel A overlaps 22 µs of dispatch's
# tail). 80 is the sweet spot that maximises streaming overlap
# without starving combine_grads of channels.
TILE_M = 128
TILE_N_A = 256
TILE_N_Y = 256  # After tile_n_a_bwd was decoupled to 256, fwd kernel Y itself
# also peaks at 256 — re-sweep at production: 128→4811, 160→4727, 192→4745,
# 224→4766, 256→4687 (single runs); 6-run mean 4714 vs 4757 at 160 (-43 μs).
TILE_N_Y_BWD = 128  # Decoupled from TILE_N_A: kernel_y_bwd is faster at smaller
# tile_n (epilogue ColVecReduceAtomic + dswiglu pressure) than fwd kernel A.
TILE_N_A_BWD = 256  # Decoupled from TILE_N_Y: kernel_a_bwd does a much larger
# data-grad GEMM than fwd kernel Y (M=TK, K=2I, N=H — same N=H but K=2I, double
# the K-axis work) and saturates better at tile_N=256 than at fwd-Y's optimum
# of 160. Sweep at production: 128→4884, 160→4832, 192→4939, 256→4800 (single
# runs); 5-run mean at 256 is ~4757 vs 4905 at 160 → ~148 μs saved.


def time_kernel(fn, *, warmup=10, iters=50) -> float:
    """Median wall-clock time of `fn` over `iters` runs (μs), warmup excluded."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_us = sorted(starts[i].elapsed_time(ends[i]) * 1e3 for i in range(iters))
    return times_us[len(times_us) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--num_sms_a",
        type=int,
        default=None,
        help="Cap on kernel A persistent grid CTA count. None = full GPU.",
    )
    p.add_argument(
        "--num_sms_y",
        type=int,
        default=None,
        help="Cap on kernel Y persistent grid CTA count. None = full GPU.",
    )
    p.add_argument(
        "--num_sms_dispatch",
        type=int,
        default=NUM_SMS,
        help="DeepEP num_sms (channel count; dispatch grid uses ~2×).",
    )
    p.add_argument("--tile_m", type=int, default=TILE_M)
    p.add_argument("--tile_n_a", type=int, default=TILE_N_A)
    p.add_argument("--tile_n_y", type=int, default=TILE_N_Y)
    p.add_argument("--tile_n_y_bwd", type=int, default=TILE_N_Y_BWD)
    p.add_argument("--tile_n_a_bwd", type=int, default=TILE_N_A_BWD)
    args, _ = p.parse_known_args()

    device = init_distributed()
    rank, world_size = get_global_rank(), get_world_size()
    group = torch_dist.group.WORLD
    local_E = NUM_EXPERTS // world_size

    buffer = make_buffer(group, args.num_sms_dispatch)

    g = torch.Generator(device=device).manual_seed(42)
    w1_full = (
        torch.randn(NUM_EXPERTS, 2 * I, H, dtype=DTYPE, device=device, generator=g)
        * 0.02
    ).contiguous()
    w2_full = (
        torch.randn(NUM_EXPERTS, H, I, dtype=DTYPE, device=device, generator=g) * 0.02
    ).contiguous()
    w1_local = w1_full[rank * local_E : (rank + 1) * local_E].contiguous()
    w2_local = w2_full[rank * local_E : (rank + 1) * local_E].contiguous()

    torch.manual_seed(100 + rank)
    x = (
        torch.randn(SEQ_LEN_PER_RANK, H, dtype=DTYPE, device=device) * 0.1
    ).contiguous()
    topk_idx = make_uniform_topk_idx(SEQ_LEN_PER_RANK, TOPK, NUM_EXPERTS, rank, device)
    topk_weights = torch.softmax(
        torch.randn(SEQ_LEN_PER_RANK, TOPK, dtype=torch.float32, device=device), dim=-1
    ).contiguous()
    rank_idx = topk_idx // local_E
    is_token_in_rank = torch.zeros(
        (SEQ_LEN_PER_RANK, world_size), dtype=torch.bool, device=device
    )
    for r in range(world_size):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)

    # Four caller-managed streams. The metadata-done event records between the
    # metadata kernel and the dispatch main kernel — consumer streams
    # `wait_event` on it to safely read metadata tensors without serializing
    # against dispatch main.
    streams = make_streams(device)

    # Run dispatch once to get a StreamingHandle. Reuse this for all isolated
    # kernel timing runs (kernel A / kernel Y individually).
    with torch.cuda.stream(streams.dispatch):
        pool, handle, metadata_done = buffer.dispatch(
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            NUM_EXPERTS,
            tile_m=args.tile_m,
            dispatch_seq=1,
        )
    # For isolated kernel timing below we need dispatch fully done; sync the
    # default stream against streams.dispatch so subsequent default-stream work
    # (the isolated runs) sees dispatch's writes.
    torch.cuda.current_stream().wait_stream(streams.dispatch)
    torch.cuda.synchronize()
    total_tiles = handle.total_tiles
    TK_padded = pool.shape[0]
    T_recv = handle.o.shape[0]

    expert_pool_block_offset = handle.expert_pool_block_offset
    cu_seqlens_m = (expert_pool_block_offset.to(torch.int32) * args.tile_m).contiguous()
    expert_frequency = handle.expert_frequency
    TK = int(expert_frequency.sum().item())  # actual (token, k) count, no padding

    # Pre-fired ready signals so kernel A / kernel Y can be timed in isolation.
    tile_ready_fired = torch.full_like(handle.tile_ready, handle.dispatch_seq)
    a_ready_fired = torch.full_like(handle.a_ready, handle.dispatch_seq)
    # Fresh a_ready for kernel A's own release-store (one isolated kernel A
    # run writes here; we don't read it). zero-init so the per-tile multi-pid_n
    # gating's `tile_n_stripes_done` dance fires cleanly.
    a_ready_for_a = torch.zeros_like(handle.a_ready)

    # Postact_a (kernel A → kernel Y intermediate). Sized for per-tile slabs.
    postact_a = torch.empty(total_tiles, args.tile_m, I, dtype=DTYPE, device=device)
    # Preact_a (kernel A's pre-SwiGLU bf16 [2I] gate-up accumulator). Saved
    # in fwd via ctx; bwd kernel_y_bwd reads it as mC (f32-recast).
    preact_a = torch.zeros(total_tiles, args.tile_m, 2 * I, dtype=DTYPE, device=device)
    # Reference postact buffers for the gemm_gated baseline.
    postact_flat_ref = torch.empty(TK_padded, I, dtype=DTYPE, device=device)
    preact_flat_ref = torch.empty(TK_padded, 2 * I, dtype=DTYPE, device=device)
    # Reference y buffer for kernel Y baseline (TK_padded rows, then scattered).
    y_ref = torch.empty(TK_padded, H, dtype=DTYPE, device=device)

    # `o` is `handle.o` from DeepEP, sized [T_recv, H], zero-init. Kernel Y
    # writes via PTX-predicated atomic-scatter (no trash row).
    o_buf = handle.o

    # ── Bwd-side buffers for isolated bwd-kernel timing ───────────────────
    # The bwd path doesn't allocate via the same cached handle — each
    # `dispatch_grads` call returns fresh `dL_do_pool` / `bwd_y_ready`.
    # For isolated kernel_y_bwd / kernel_a_bwd / combine_grads timing we
    # capture one set from a single dispatch_grads call below and reuse it
    # across timed iters.
    dL_dy_in = (torch.randn_like(x) * 0.01).contiguous()
    dL_dswiglu_in = torch.empty(
        total_tiles, args.tile_m, 2 * I, dtype=DTYPE, device=device
    )
    # postact_a_for_dW2: kernel_y_bwd's mPostAct TMA-store output (weighted
    # postact = postact * pool_topk_weight, bf16 (M, I)). Reused as input
    # to dW2's grouped GEMM in the orchestrator.
    postact_a_for_dW2 = torch.empty(
        total_tiles, args.tile_m, I, dtype=DTYPE, device=device
    )
    dL_dweight = torch.zeros(TK_padded, dtype=torch.float32, device=device)
    dL_dx_per_r = torch.zeros(T_recv, H, dtype=DTYPE, device=device)
    # Pre-fired bwd ready signals so each bwd kernel can be timed in isolation.
    bwd_y_ready_fired = torch.full(
        (total_tiles,), handle.dispatch_seq, dtype=torch.int64, device=device
    )
    bwd_a_ready_fired = torch.full(
        (total_tiles,), handle.dispatch_seq, dtype=torch.int64, device=device
    )
    # bwd_a_ready_for_y: kernel_y_bwd writes its release-store here. Must
    # zero-init before each call so multi-pid_n stripe-done gating fires
    # cleanly. Reset inside run_streaming_y_bwd.
    bwd_a_ready_for_y = torch.zeros(total_tiles, dtype=torch.int64, device=device)
    bwd_per_token_remaining = torch.empty(T_recv, dtype=torch.int32, device=device)
    bwd_per_token_remaining_init = handle.k_local_count.to(torch.int32).clone()
    bwd_compute_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)

    if rank == 0:
        # MoE forward FLOPs (kernel A + kernel Y bf16 matmuls).
        flops_a = 2 * 2 * TK * H * I  # gated: 2*M*N*K with N = 2I
        flops_y = 2 * TK * H * I
        flops_total = flops_a + flops_y
        print(
            f"\nconfig: H={H} I={I} E_total={NUM_EXPERTS} E_local={local_E} "
            f"K={TOPK} T_per_rank={SEQ_LEN_PER_RANK} world={world_size} "
            f"num_sms_dispatch={args.num_sms_dispatch} "
            f"num_sms_a={args.num_sms_a} num_sms_y={args.num_sms_y} "
            f"tile_m={args.tile_m} tile_n_a={args.tile_n_a} tile_n_y={args.tile_n_y}"
        )
        print(
            f"        TK_padded={TK_padded} T_recv={T_recv} total_tiles={total_tiles} "
            f"TK={TK} padding_rows={TK_padded - TK}"
        )
        print(
            f"        FLOPs/forward: {flops_total / 1e9:.2f} GFLOPs "
            f"(kernel A {flops_a / 1e9:.2f} + kernel Y {flops_y / 1e9:.2f})"
        )
        print()

    # ────────────────────────────────────────────────────────────────────
    # Per-stage isolated timing.
    # ────────────────────────────────────────────────────────────────────

    def run_streaming_a():
        streaming_moe_a(
            pool,
            w1_local,
            postact_a,
            handle.tile_id_to_expert,
            handle.expert_pool_block_offset,
            tile_ready_fired,
            a_ready_for_a,
            dispatch_seq=handle.dispatch_seq,
            compute_seq=handle.dispatch_seq,
            tile_m=args.tile_m,
            tile_n=args.tile_n_a,
            num_sms=args.num_sms_a,
        )

    def run_gemm_gated_ref():
        gemm_act(
            pool,
            w1_local,
            preact_flat_ref,
            None,
            postact_flat_ref,
            None,
            "swiglu",
            tile_M=args.tile_m,
            tile_N=args.tile_n_a,
            cluster_M=1,
            cluster_N=1,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=None,
        )

    def run_streaming_y_only():
        # Reset per-call accumulator state. (Production pipeline: DeepEP's
        # dispatch zeros these; here we time kernel Y in isolation across many
        # calls so we reset by hand.)
        handle.per_token_remaining.copy_(_per_token_remaining_init)
        handle.compute_done_per_token.zero_()
        o_buf.zero_()
        streaming_moe_y(
            postact_a,
            w2_local,
            o_buf,
            handle.pool_recv_token,
            handle.pool_topk_weight,
            handle.per_token_remaining,
            handle.compute_done_per_token,
            handle.tile_id_to_expert,
            handle.expert_pool_block_offset,
            a_ready_fired,
            compute_seq=handle.dispatch_seq,
            combine_seq=1,
            tile_m=args.tile_m,
            tile_n=args.tile_n_y,
            num_sms=args.num_sms_y,
        )

    def run_gemm_y_ref():
        # Plain GEMM on the same per-tile pool layout: y_ref[s, :] = postact_a_flat[s] @ W2[expert_id(s)].T.
        # Uses the same varlen_m path as kernel A (cu_seqlens_m = expert_pool_block_offset * tile_m).
        # Then a separate index_add_ scatter into o_ref accumulates topk-weighted y_ref by recv-token.
        # We time JUST the GEMM here as a fair baseline against kernel Y's GEMM portion.
        # The atomic-scatter portion's cost is captured by the difference between
        # streaming_moe_y total and run_gemm_y_ref total in the summary.
        postact_flat_in = postact_a.view(TK_padded, I)
        gemm(
            postact_flat_in,
            w2_local,
            y_ref,
            None,
            None,
            tile_M=args.tile_m,
            tile_N=args.tile_n_y,
            cluster_M=1,
            cluster_N=1,
            cu_seqlens_m=cu_seqlens_m,
        )

    def run_combine_only():
        # Combine sender's per-token gate spins on
        # `compute_done_per_token[r] >= combine_seq`. Kernel Y populated
        # `compute_done_per_token` to `handle.dispatch_seq` for every recv-token
        # in the preceding `run_streaming_y_only()` warmup; refresh defensively
        # in case run_streaming_y_only zeroed it before the gate was supposed to
        # fire (it doesn't, but the explicit fill_ keeps the isolated timing
        # decoupled from kernel-Y call ordering).
        handle.compute_done_per_token.fill_(handle.dispatch_seq)
        buffer.combine(handle.o, handle, combine_seq=handle.dispatch_seq)

    # ── Bwd-side isolated runners ──────────────────────────────────────────
    # `dispatch_grads` allocates a fresh `dL_do_pool` / `bwd_y_ready` each
    # call; the kernel_y_bwd runner reuses one captured pair below.
    _dL_do_pool_captured: torch.Tensor | None = None
    _bwd_y_ready_captured: torch.Tensor | None = None

    def run_dispatch_grads_only():
        nonlocal _dL_do_pool_captured, _bwd_y_ready_captured
        with torch.cuda.stream(streams.dispatch):
            _dL_do_pool_captured, _bwd_y_ready_captured = buffer.dispatch_grads(
                handle, dL_dy_in, dispatch_seq=handle.dispatch_seq
            )
        torch.cuda.current_stream().wait_stream(streams.dispatch)

    def run_streaming_y_bwd():
        # Reset bwd_a_ready (kernel_y_bwd's release target) so the
        # multi-pid_n stripe-done gating sees zero on the first stripe.
        # Zero dL_dweight too — kernel atomic-adds into it; non-zero would
        # double-count across iterations.
        bwd_a_ready_for_y.zero_()
        dL_dweight.zero_()
        streaming_moe_y_bwd(
            _dL_do_pool_captured,
            w2_local,
            dL_dswiglu_in,
            postact_a_for_dW2,
            handle.pool_topk_weight,
            handle.pool_recv_token,
            preact_a,
            dL_dweight,
            handle.tile_id_to_expert,
            handle.expert_pool_block_offset,
            bwd_y_ready_fired,
            bwd_a_ready_for_y,
            dispatch_seq=handle.dispatch_seq,
            tile_m=args.tile_m,
            tile_n=args.tile_n_y_bwd,
            num_sms=args.num_sms_a,
        )

    # NN-form W2 / W1 for the bwd-ref GEMMs: kernel_y_bwd does
    # `g = dL_do_pool @ W2`  (M=TK_padded varlen-m, K=H, N=I) — the ref needs
    # an N-major (k-contig) B operand of shape (E_local, K=H, N=I), i.e.
    # W2.permute(0, 2, 1). Same idea for W1: kernel_a_bwd does
    # `dL/dpool = dL_dswiglu_in @ W1` (M=TK_padded, K=2I, N=H), N-major B is
    # W1.permute(0, 2, 1). Materialised once outside the timing loop so the
    # cost of the permute doesn't pollute the GEMM measurement.
    w2_nmajor_ref = w2_local.permute(0, 2, 1).contiguous()
    w1_nmajor_ref = w1_local.permute(0, 2, 1).contiguous()

    def run_gemm_y_bwd_ref():
        # Plain streaming GEMM mirroring kernel_y_bwd's data-grad shape:
        # M=TK_padded varlen-m, K=H, N=I. (Kernel_y_bwd's full output is
        # (M, 2I) bf16 = (M, I) fp32 view via the dswiglu pack trick; the
        # GEMM mainloop itself is N=I — so the fair baseline is N=I.) No
        # epilogue (no SwiGLU bwd, no ColVecReduce). Reuses postact_flat_ref
        # as the output sink.
        gemm(
            _dL_do_pool_captured,
            w2_nmajor_ref,
            postact_flat_ref,
            None,
            None,
            tile_M=args.tile_m,
            tile_N=args.tile_n_y_bwd,
            cluster_M=1,
            cluster_N=1,
            cu_seqlens_m=cu_seqlens_m,
        )

    def run_streaming_a_bwd():
        bwd_per_token_remaining.copy_(bwd_per_token_remaining_init)
        bwd_compute_done_per_token.zero_()
        dL_dx_per_r.zero_()
        streaming_moe_a_bwd(
            dL_dswiglu_in,
            w1_local,
            dL_dx_per_r,
            handle.pool_recv_token,
            bwd_per_token_remaining,
            bwd_compute_done_per_token,
            handle.tile_id_to_expert,
            handle.expert_pool_block_offset,
            bwd_a_ready_fired,
            dispatch_seq=handle.dispatch_seq,
            tile_m=args.tile_m,
            tile_n=args.tile_n_a_bwd,
            num_sms=args.num_sms_y,
        )

    def run_gemm_a_bwd_ref():
        # Plain streaming GEMM mirroring kernel_a_bwd's data-grad shape:
        # M=TK_padded varlen-m, K=2*I, N=H. No atomic-scatter epilogue.
        dL_dswiglu_in_flat = dL_dswiglu_in.view(TK_padded, 2 * I)
        gemm(
            dL_dswiglu_in_flat,
            w1_nmajor_ref,
            y_ref,  # reuse: same (TK_padded, H) shape as kernel Y baseline
            None,
            None,
            tile_M=args.tile_m,
            tile_N=args.tile_n_a_bwd,
            cluster_M=1,
            cluster_N=1,
            cu_seqlens_m=cu_seqlens_m,
        )

    def run_combine_grads_only():
        # combine_grads's sender gate is `bwd_compute_done_per_token`; pre-fire
        # the same way run_combine_only fires `compute_done_per_token`. The
        # kernel runs to completion using captured `dL_dx_per_r` / `dL_dweight`.
        bwd_compute_done_per_token.fill_(handle.dispatch_seq)
        buffer.combine_grads(
            dL_dx_per_r,
            handle,
            dL_dweight,
            bwd_compute_done_per_token,
            dispatch_seq=handle.dispatch_seq,
        )

    # ── dW1 / dW2 grouped-GEMM tail timing (varlen-K). Same calls the bwd
    # orchestrator runs after kernel_y_bwd / kernel_a_bwd. Quack `gemm` with
    # `cu_seqlens_k`. ──
    cu_seqlens_k = (
        handle.expert_pool_block_offset.to(torch.int32) * args.tile_m
    ).contiguous()
    dW2_local_buf = torch.empty_like(w2_local)
    dW1_local_buf = torch.empty_like(w1_local)
    postact_a_for_dW2_flat = postact_a_for_dW2.view(TK_padded, I)

    def run_dW2_grouped_gemm():
        # dW2[e] = postact_a[slot_range_e].T @ dL_do_pool[slot_range_e]
        gemm(
            _dL_do_pool_captured.t(),
            postact_a_for_dW2_flat.t(),
            dW2_local_buf,
            None,
            None,
            tile_M=args.tile_m,
            tile_N=args.tile_n_a,
            cluster_M=2,
            cluster_N=1,
            cu_seqlens_k=cu_seqlens_k,
        )

    def run_dW1_grouped_gemm():
        # dW1[e] = dL_dswiglu_in[slot_range_e].T @ pool[slot_range_e]
        dL_dswiglu_in_flat = dL_dswiglu_in.view(TK_padded, 2 * I)
        gemm(
            dL_dswiglu_in_flat.t(),
            pool.t(),
            dW1_local_buf,
            None,
            None,
            tile_M=args.tile_m,
            tile_N=args.tile_n_a,
            cluster_M=2,
            cluster_N=1,
            cu_seqlens_k=cu_seqlens_k,
        )

    # Capture initial per_token_remaining so isolated y timing can reset it.
    _per_token_remaining_init = handle.per_token_remaining.clone()

    # Materialize one dispatch_grads result for kernel_y_bwd / a_bwd / combine_grads inputs.
    run_dispatch_grads_only()
    torch.cuda.synchronize()
    assert _dL_do_pool_captured is not None

    # Warm everything up: A, gemm_gated, Y, gemm, combine, and the bwd-side
    # kernels (kernel_y_bwd / kernel_a_bwd / combine_grads + dW1/dW2 grouped
    # GEMMs). dispatch_grads was already warmed by the capture above.
    run_streaming_a()
    run_gemm_gated_ref()
    run_streaming_y_only()
    run_gemm_y_ref()
    run_combine_only()
    run_streaming_y_bwd()
    run_gemm_y_bwd_ref()
    run_streaming_a_bwd()
    run_gemm_a_bwd_ref()
    run_combine_grads_only()
    run_dW2_grouped_gemm()
    run_dW1_grouped_gemm()
    torch.cuda.synchronize()
    barrier(group)

    rank_zero_print("=== per-stage isolated timing ===")
    rank_zero_print(
        f"{'kernel':>26s}  {'tile_M':>6s}  {'tile_N':>6s}  {'time (μs)':>10s}  {'TFLOPs/s':>10s}"
    )
    rank_zero_print(f"{'-' * 26}  {'-' * 6}  {'-' * 6}  {'-' * 10}  {'-' * 10}")

    @rank_zero_only
    def fmt_row(name, tm, tn, t_us, flops):
        tflops = flops / (t_us * 1e-6) / 1e12
        print(f"{name:>26s}  {tm:>6d}  {tn:>6d}  {t_us:>10.1f}  {tflops:>10.2f}")

    streaming_a_us = time_kernel(run_streaming_a)
    fmt_row(
        "streaming_moe_a",
        args.tile_m,
        args.tile_n_a,
        streaming_a_us,
        2 * 2 * TK * H * I,
    )

    gated_us = time_kernel(run_gemm_gated_ref)
    fmt_row(
        "gemm_gated (ref)", args.tile_m, args.tile_n_a, gated_us, 2 * 2 * TK * H * I
    )

    streaming_y_us = time_kernel(run_streaming_y_only)
    fmt_row(
        "streaming_moe_y", args.tile_m, args.tile_n_y, streaming_y_us, 2 * TK * H * I
    )

    gemm_y_us = time_kernel(run_gemm_y_ref)
    fmt_row(
        "gemm (ref, no scatter)", args.tile_m, args.tile_n_y, gemm_y_us, 2 * TK * H * I
    )

    combine_us = time_kernel(run_combine_only)
    if rank == 0:
        # Combine isn't a matmul — print as raw µs without TFLOPs.
        print(
            f"{'buffer.combine':>26s}  {'-':>6s}  {'-':>6s}  {combine_us:>10.1f}  {'-':>10s}"
        )

    # ── Bwd-side isolated timing rows. FLOPs match the per-kernel data-grad
    # GEMM (kernel_y_bwd: dL_do_pool @ W2 → 2*TK*H*I; kernel_a_bwd:
    # dL_dswiglu_in @ W1 → 2*2*TK*H*I; dW2 / dW1 grouped GEMMs are the
    # transposed varlen-K versions of the same arithmetic intensity). ──
    dispatch_grads_us = time_kernel(run_dispatch_grads_only)
    if rank == 0:
        print(
            f"{'buffer.dispatch_grads':>26s}  {'-':>6s}  {'-':>6s}  "
            f"{dispatch_grads_us:>10.1f}  {'-':>10s}"
        )

    streaming_y_bwd_us = time_kernel(run_streaming_y_bwd)
    fmt_row(
        "streaming_moe_y_bwd",
        args.tile_m,
        args.tile_n_y_bwd,
        streaming_y_bwd_us,
        2 * TK * H * I,
    )

    gemm_y_bwd_us = time_kernel(run_gemm_y_bwd_ref)
    fmt_row(
        "gemm (ref, y_bwd shape)",
        args.tile_m,
        args.tile_n_y_bwd,
        gemm_y_bwd_us,
        2 * TK * H * I,
    )

    streaming_a_bwd_us = time_kernel(run_streaming_a_bwd)
    fmt_row(
        "streaming_moe_a_bwd",
        args.tile_m,
        args.tile_n_y,
        streaming_a_bwd_us,
        2 * 2 * TK * H * I,
    )

    gemm_a_bwd_us = time_kernel(run_gemm_a_bwd_ref)
    fmt_row(
        "gemm (ref, a_bwd shape)",
        args.tile_m,
        args.tile_n_y,
        gemm_a_bwd_us,
        2 * 2 * TK * H * I,
    )

    combine_grads_us = time_kernel(run_combine_grads_only)
    if rank == 0:
        print(
            f"{'buffer.combine_grads':>26s}  {'-':>6s}  {'-':>6s}  "
            f"{combine_grads_us:>10.1f}  {'-':>10s}"
        )

    dW2_us = time_kernel(run_dW2_grouped_gemm)
    fmt_row(
        "gemm_grouped dW2 (cu_K)", args.tile_m, args.tile_n_a, dW2_us, 2 * TK * H * I
    )

    dW1_us = time_kernel(run_dW1_grouped_gemm)
    fmt_row(
        "gemm_grouped dW1 (cu_K)",
        args.tile_m,
        args.tile_n_a,
        dW1_us,
        2 * 2 * TK * H * I,
    )

    rank_zero_print()

    # ────────────────────────────────────────────────────────────────────
    # End-to-end pipeline timing: dispatch + A + Y + combine on four streams.
    # ────────────────────────────────────────────────────────────────────
    # Each iteration runs a fresh dispatch (so DeepEP allocates fresh per-token
    # state) followed by kernel A on compute_a_stream, kernel Y on
    # compute_y_stream, and combine sender on combine_stream.
    # Cross-stream visibility:
    # (a) `metadata_done` event recorded by dispatch between the metadata
    #     kernel and the dispatch main kernel — consumer streams wait_event on
    #     this to safely read metadata tensors without serializing against
    #     dispatch main, preserving per-tile streaming overlap.
    # (b) per-tile `tile_ready` (dispatch→A) / `a_ready` (A→Y) release/acquire
    #     pairs and per-token `compute_done_per_token` (Y→combine sender) gate.

    def run_pipeline_step(seq):
        stream_moe_func(
            buffer,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1_local,
            w2_local,
            streams=streams,
            num_experts=NUM_EXPERTS,
            dispatch_seq=seq,
            tile_m=args.tile_m,
            tile_n_a=args.tile_n_a,
            tile_n_y=args.tile_n_y,
            tile_n_y_bwd=args.tile_n_y_bwd,
            tile_n_a_bwd=args.tile_n_a_bwd,
            num_sms_a=args.num_sms_a,
            num_sms_y=args.num_sms_y,
        )

    # End-to-end timing uses real CUDA events around the whole pipeline.
    pipe_warmup = 5
    pipe_iters = 30

    seq_counter = [10]  # mutable seq, bumped each iter

    def step():
        run_pipeline_step(seq_counter[0])
        seq_counter[0] += 1

    pipeline_us = time_kernel(step, warmup=pipe_warmup, iters=pipe_iters)

    # Dispatch-only timing for context (we already paid for it once above).
    def dispatch_only():
        seq = seq_counter[0]
        seq_counter[0] += 1
        with torch.cuda.stream(streams.dispatch):
            buffer.dispatch(
                x,
                topk_idx,
                topk_weights,
                is_token_in_rank,
                NUM_EXPERTS,
                tile_m=args.tile_m,
                dispatch_seq=seq,
            )
        # Layer-end barrier: default stream waits on streams.dispatch.
        torch.cuda.current_stream().wait_stream(streams.dispatch)

    dispatch_us = time_kernel(dispatch_only, warmup=pipe_warmup, iters=pipe_iters)

    # ── Fwd+bwd e2e (training-iter cost). Separate `step` because we need
    # leaves with requires_grad=True for autograd to track. We toggle the
    # flag here so the fwd-only run above stays representative of inference. ──
    x.requires_grad_(True)
    topk_weights.requires_grad_(True)
    w1_local.requires_grad_(True)
    w2_local.requires_grad_(True)

    def step_fwd_bwd():
        seq = seq_counter[0]
        seq_counter[0] += 1
        out = stream_moe_func(
            buffer,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1_local,
            w2_local,
            streams=streams,
            num_experts=NUM_EXPERTS,
            dispatch_seq=seq,
            tile_m=args.tile_m,
            tile_n_a=args.tile_n_a,
            tile_n_y=args.tile_n_y,
            tile_n_y_bwd=args.tile_n_y_bwd,
            tile_n_a_bwd=args.tile_n_a_bwd,
            num_sms_a=args.num_sms_a,
            num_sms_y=args.num_sms_y,
        )
        out.sum().backward()
        x.grad = None
        topk_weights.grad = None
        w1_local.grad = None
        w2_local.grad = None

    pipeline_fwd_bwd_us = time_kernel(
        step_fwd_bwd, warmup=pipe_warmup, iters=pipe_iters
    )

    if rank == 0:
        print("=== end-to-end pipeline (dispatch + A + Y + combine, 4 streams) ===")
        print(f"  dispatch (alone, dispatch_stream):           {dispatch_us:7.1f} μs")
        print(f"  streaming_moe_a (alone, compute_a):      {streaming_a_us:7.1f} μs")
        print(f"  streaming_moe_y (alone, compute_y):      {streaming_y_us:7.1f} μs")
        print(f"  buffer.combine (alone, combine_send):    {combine_us:7.1f} μs")
        sequential_sum = dispatch_us + streaming_a_us + streaming_y_us + combine_us
        print(f"  serial sum of stages:                    {sequential_sum:7.1f} μs")
        print(f"  fwd-only e2e (4 streams, real overlap):  {pipeline_us:7.1f} μs")
        if pipeline_us < sequential_sum:
            saved = sequential_sum - pipeline_us
            pct = 100.0 * saved / sequential_sum
            print(f"  fwd overlap saved: {saved:7.1f} μs ({pct:.1f}% of serial sum)")
        else:
            print("  (no overlap savings observed; pipeline > serial sum)")
        # Bwd serial budget: 4 bwd-stage isolated timings + 2 dW grouped GEMMs.
        bwd_stages_sum = (
            dispatch_grads_us
            + streaming_y_bwd_us
            + streaming_a_bwd_us
            + combine_grads_us
        )
        bwd_grouped_sum = dW2_us + dW1_us
        full_serial_sum = sequential_sum + bwd_stages_sum + bwd_grouped_sum
        print(f"  dispatch_grads (alone):                  {dispatch_grads_us:7.1f} μs")
        print(
            f"  streaming_moe_y_bwd (alone):             {streaming_y_bwd_us:7.1f} μs"
        )
        print(
            f"  streaming_moe_a_bwd (alone):             {streaming_a_bwd_us:7.1f} μs"
        )
        print(f"  buffer.combine_grads (alone):            {combine_grads_us:7.1f} μs")
        print(f"  gemm_grouped dW2 (alone):                {dW2_us:7.1f} μs")
        print(f"  gemm_grouped dW1 (alone):                {dW1_us:7.1f} μs")
        print(f"  serial sum (fwd + bwd + dW1/dW2):        {full_serial_sum:7.1f} μs")
        print(
            f"  fwd+bwd e2e (training iter):             {pipeline_fwd_bwd_us:7.1f} μs"
        )
        if pipeline_fwd_bwd_us < full_serial_sum:
            saved_fb = full_serial_sum - pipeline_fwd_bwd_us
            pct_fb = 100.0 * saved_fb / full_serial_sum
            print(
                f"  fwd+bwd overlap saved: {saved_fb:7.1f} μs ({pct_fb:.1f}% of serial sum)"
            )

        print()
        print("=== summary ===")
        print(f"  streaming_moe_a:           {streaming_a_us:7.1f} μs")
        print(f"  gemm_gated (ref):          {gated_us:7.1f} μs")
        if streaming_a_us > 0 and gated_us > 0:
            ra = streaming_a_us / gated_us
            print(f"    streaming_a / gemm_gated: {ra:.3f}×")
        print(f"  streaming_moe_y:           {streaming_y_us:7.1f} μs")
        print(f"  gemm (ref, no scatter):    {gemm_y_us:7.1f} μs")
        if streaming_y_us > 0 and gemm_y_us > 0:
            ry = streaming_y_us / gemm_y_us
            atomic_overhead_us = streaming_y_us - gemm_y_us
            print(
                f"    streaming_y / gemm:       {ry:.3f}× "
                f"(atomic-scatter overhead: {atomic_overhead_us:+.1f} μs)"
            )
        print(f"  streaming_moe_y_bwd:       {streaming_y_bwd_us:7.1f} μs")
        print(f"  gemm (ref, y_bwd shape):   {gemm_y_bwd_us:7.1f} μs")
        if streaming_y_bwd_us > 0 and gemm_y_bwd_us > 0:
            ryb = streaming_y_bwd_us / gemm_y_bwd_us
            yb_overhead = streaming_y_bwd_us - gemm_y_bwd_us
            print(
                f"    streaming_y_bwd / gemm:   {ryb:.3f}× "
                f"(SwiGLU+ColVecReduce overhead: {yb_overhead:+.1f} μs)"
            )
        print(f"  streaming_moe_a_bwd:       {streaming_a_bwd_us:7.1f} μs")
        print(f"  gemm (ref, a_bwd shape):   {gemm_a_bwd_us:7.1f} μs")
        if streaming_a_bwd_us > 0 and gemm_a_bwd_us > 0:
            rab = streaming_a_bwd_us / gemm_a_bwd_us
            ab_overhead = streaming_a_bwd_us - gemm_a_bwd_us
            print(
                f"    streaming_a_bwd / gemm:   {rab:.3f}× "
                f"(atomic-scatter overhead: {ab_overhead:+.1f} μs)"
            )

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
