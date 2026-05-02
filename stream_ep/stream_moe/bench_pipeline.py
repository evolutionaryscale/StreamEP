"""Bench the streaming pipeline up to kernel Y (dispatch → A → Y).

Setup
-----
- Run Buffer.dispatch once to produce a StreamingHandle (multi-GPU required
  for DeepEP).
- Pool layout puts data into expert-major BLOCK_M-padded tiles. Both kernel A
  (gemm_gated) and kernel Y (gemm + atomic-scatter into o[T_recv, H]) read
  from the pool via the standard varlen_m strided TMA path.
- Time three things, all on a single GPU:
    * dispatch only (comm_stream)
    * each kernel in isolation with its ready signal pre-set (compute_a/y stream)
    * end-to-end pipeline (3 streams, real producer-consumer release/acquire)
- Reference baselines:
    * kernel A vs `quack.gemm_act.gemm_gated` (same per-tile compute, no
      streaming scheduler).
    * kernel Y vs `quack.gemm.gemm` + a separate `index_add_` scatter (so
      `gemm` produces y_per_tile[T_recv, H] and we accumulate o[r] += w * y[r];
      this is a strict per-token equivalent at the ~ms scale and a fair
      timing baseline for the fused atomic-scatter).

Launch:
    torchrun --nproc_per_node=2 \\
        -m evolutionaryscale.models.moe.streaming_moe.bench_pipeline
"""

import argparse
import os

import torch
import torch.distributed as torch_dist
from deep_ep import Buffer as DeepEPBuffer
from quack.gemm import gemm
from quack.gemm_act import gemm_act

from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_a import (
    streaming_moe_a,
)
from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_y import (
    streaming_moe_y,
)
from evolutionaryscale.models.moe.streaming_moe.streaming_moe import streaming_moe_layer

H = 2048
I = 2048
NUM_EXPERTS = 64
SEQ_LEN_PER_RANK = 8192
TOPK = 4
DTYPE = torch.bfloat16
NUM_SMS = 132  # DeepEP num_sms (channels = num_sms / 2; max = num_device_sms,
# i.e. 132 on H100). Sweep (logs/sweep/) shows pipeline e2e
# plateaus past num_sms=64 at ~1645–1665 µs (vs 1816 µs at the
# old 24 default). Past the plateau the dispatch tail finishes
# well before kernel A (~376 µs vs ~794 µs), so any further
# dispatch speedup just enlarges an already-overlapping gap;
# the critical path is kernel A. Default to the ceiling rather
# than picking a mid-range sweet spot that's within noise.
TILE_M = 128
TILE_N_A = 256
TILE_N_Y = 128


def make_uniform_topk_idx(n_tokens, topk, num_experts, rank, device):
    base = (torch.arange(n_tokens, device=device) + rank * n_tokens) * topk
    offsets = torch.arange(topk, device=device).unsqueeze(0)
    return ((base.unsqueeze(1) + offsets) % num_experts).to(torch.int64)


def make_buffer(group, num_sms):
    DeepEPBuffer.set_num_sms(num_sms)
    hidden_bytes = H * 2
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
    args, _ = p.parse_known_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    torch_dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = torch_dist.group.WORLD

    device = torch.device(f"cuda:{local_rank}")
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

    # Four caller-managed streams. `comm_stream` for dispatch; `compute_a_stream`
    # for kernel A; `compute_y_stream` for kernel Y; `combine_send_stream` for
    # the combine sender. The metadata-done event records between the metadata
    # kernel and the dispatch main kernel — consumer streams `wait_event` on it
    # to safely read metadata tensors without serializing against dispatch main.
    comm_stream = torch.cuda.Stream(device=device)
    compute_a_stream = torch.cuda.Stream(device=device)
    compute_y_stream = torch.cuda.Stream(device=device)
    combine_send_stream = torch.cuda.Stream(device=device)

    # Run dispatch once to get a StreamingHandle. Reuse this for all isolated
    # kernel timing runs (kernel A / kernel Y individually).
    with torch.cuda.stream(comm_stream):
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
    # default stream against comm_stream so subsequent default-stream work
    # (the isolated runs) sees dispatch's writes.
    torch.cuda.current_stream().wait_stream(comm_stream)
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
    # Reference postact buffers for the gemm_gated baseline.
    postact_flat_ref = torch.empty(TK_padded, I, dtype=DTYPE, device=device)
    preact_flat_ref = torch.empty(TK_padded, 2 * I, dtype=DTYPE, device=device)
    # Reference y buffer for kernel Y baseline (TK_padded rows, then scattered).
    y_ref = torch.empty(TK_padded, H, dtype=DTYPE, device=device)

    # `o` is `handle.o` from DeepEP, sized [T_recv, H], zero-init. Kernel Y
    # writes via PTX-predicated atomic-scatter (no trash row).
    o_buf = handle.o

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
        buffer.combine(
            handle.o,
            handle,
            topk_weights=handle.recv_topk_weights,
            combine_seq=handle.dispatch_seq,
        )

    # Capture initial per_token_remaining so isolated y timing can reset it.
    _per_token_remaining_init = handle.per_token_remaining.clone()

    # Warm everything up: A, gemm_gated, Y, gemm, combine.
    run_streaming_a()
    run_gemm_gated_ref()
    run_streaming_y_only()
    run_gemm_y_ref()
    run_combine_only()
    torch.cuda.synchronize()
    torch_dist.barrier(group=group)

    if rank == 0:
        print("=== per-stage isolated timing ===")
        print(
            f"{'kernel':>26s}  {'tile_M':>6s}  {'tile_N':>6s}  {'time (μs)':>10s}  {'TFLOPs/s':>10s}"
        )
        print(f"{'-' * 26}  {'-' * 6}  {'-' * 6}  {'-' * 10}  {'-' * 10}")

    def fmt_row(name, tm, tn, t_us, flops):
        if rank != 0:
            return
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

    if rank == 0:
        print()

    # ────────────────────────────────────────────────────────────────────
    # End-to-end pipeline timing: dispatch + A + Y + combine on four streams.
    # ────────────────────────────────────────────────────────────────────
    # Each iteration runs a fresh dispatch (so DeepEP allocates fresh per-token
    # state) followed by kernel A on compute_a_stream, kernel Y on
    # compute_y_stream, and combine sender on combine_send_stream.
    # Cross-stream visibility:
    # (a) `metadata_done` event recorded by dispatch between the metadata
    #     kernel and the dispatch main kernel — consumer streams wait_event on
    #     this to safely read metadata tensors without serializing against
    #     dispatch main, preserving per-tile streaming overlap.
    # (b) per-tile `tile_ready` (dispatch→A) / `a_ready` (A→Y) release/acquire
    #     pairs and per-token `compute_done_per_token` (Y→combine sender) gate.

    def run_pipeline_step(seq):
        streaming_moe_layer(
            buffer,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1_local,
            w2_local,
            comm_stream=comm_stream,
            compute_a_stream=compute_a_stream,
            compute_y_stream=compute_y_stream,
            combine_send_stream=combine_send_stream,
            num_experts=NUM_EXPERTS,
            dispatch_seq=seq,
            tile_m=args.tile_m,
            tile_n_a=args.tile_n_a,
            tile_n_y=args.tile_n_y,
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
        nonlocal_seq = seq_counter[0]
        seq_counter[0] += 1
        with torch.cuda.stream(comm_stream):
            buffer.dispatch(
                x,
                topk_idx,
                topk_weights,
                is_token_in_rank,
                NUM_EXPERTS,
                tile_m=args.tile_m,
                dispatch_seq=nonlocal_seq,
            )
        # Layer-end barrier: default stream waits on comm_stream.
        torch.cuda.current_stream().wait_stream(comm_stream)

    dispatch_us = time_kernel(dispatch_only, warmup=pipe_warmup, iters=pipe_iters)

    if rank == 0:
        print("=== end-to-end pipeline (dispatch + A + Y + combine, 4 streams) ===")
        print(f"  dispatch (alone, comm_stream):           {dispatch_us:7.1f} μs")
        print(f"  streaming_moe_a (alone, compute_a):      {streaming_a_us:7.1f} μs")
        print(f"  streaming_moe_y (alone, compute_y):      {streaming_y_us:7.1f} μs")
        print(f"  buffer.combine (alone, combine_send):    {combine_us:7.1f} μs")
        sequential_sum = dispatch_us + streaming_a_us + streaming_y_us + combine_us
        print(f"  serial sum of stages:                    {sequential_sum:7.1f} μs")
        print(f"  pipeline end-to-end (4 streams, real overlap): {pipeline_us:7.1f} μs")
        if pipeline_us < sequential_sum:
            saved = sequential_sum - pipeline_us
            pct = 100.0 * saved / sequential_sum
            print(f"  overlap saved: {saved:7.1f} μs ({pct:.1f}% of serial sum)")
        else:
            print("  (no overlap savings observed; pipeline > serial sum)")

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

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
