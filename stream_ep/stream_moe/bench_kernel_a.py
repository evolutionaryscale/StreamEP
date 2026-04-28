"""Bench streaming kernel A vs the existing non-streaming gemm_gated on the
same per-tile work. Both run on a single GPU with a Buffer.dispatch'd
StreamingHandle as the source of truth for what gets computed.

Setup
-----
- Run Buffer.dispatch once (2 GPUs are still required because DeepEP needs
  a real distributed buffer to drive count_exchange + slot_assign).
- After dispatch, build the equivalent input for the non-streaming reference:
    * A_idx_flat: tile_records_recv_x_rows concatenated in expert order
    * cu_seqlens_m: expert_frequency_offset (length E_local + 1)
    * W1: same per-expert weights
    * postact: (TK, I) flat output
- Time both kernels (CUDA-event timed, repeated, median).
- Sweep tile_n for each variant.

The two kernels do the same total work (gather + matmul + SwiGLU on TK rows
× expert weights). The non-streaming variant is the existing well-tuned QuACK
gemm_gated path — its number is the floor we should be aiming for in the
streaming kernel.

Launch:
    torchrun --nproc_per_node=2 \
        -m evolutionaryscale.models.moe.streaming_moe.bench_kernel_a
"""

from __future__ import annotations

import os

import torch
import torch.distributed as torch_dist
from deep_ep import Buffer as DeepEPBuffer
from quack.gemm_act import gemm_act
from quack.moe_streaming_sm90 import streaming_moe_a

H = 2048
I = 2048
NUM_EXPERTS = 64
SEQ_LEN_PER_RANK = 8192
TOPK = 4
DTYPE = torch.bfloat16
NUM_SMS = 24
TILE_M = 128


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


def time_kernel(fn, *, warmup=5, iters=20) -> float:
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
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    torch_dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = torch_dist.group.WORLD

    device = torch.device(f"cuda:{local_rank}")
    local_E = NUM_EXPERTS // world_size

    buffer = make_buffer(group, NUM_SMS)

    g = torch.Generator(device=device).manual_seed(42)
    w1_full = (
        torch.randn(NUM_EXPERTS, 2 * I, H, dtype=DTYPE, device=device, generator=g)
        * 0.02
    ).contiguous()
    w1_local = w1_full[rank * local_E : (rank + 1) * local_E].contiguous()

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

    # Run dispatch once to get a StreamingHandle. Reuse this for all timing.
    recv_x, _, _, handle, _ = buffer.dispatch(
        x,
        topk_idx,
        topk_weights,
        is_token_in_rank,
        NUM_EXPERTS,
        tile_m=TILE_M,
        dispatch_seq=1,
    )
    torch.cuda.synchronize()
    total_tiles = handle.total_tiles
    T_recv = recv_x.shape[0]

    # ── Build the equivalent flat input for non-streaming gemm_gated.
    # tile_records_recv_x_rows is laid out in expert-then-tile order; the only
    # complication is that the LAST tile of each expert may have -1 sentinel
    # rows (if expert_frequency[e] is not divisible by tile_M). For the
    # non-streaming reference we drop those sentinels and use the per-expert
    # cu_seqlens_m derived from expert_frequency.
    expert_frequency = handle.expert_frequency  # (E_local,)
    expert_frequency_offset = handle.expert_frequency_offset  # (E_local+1,)
    rows = handle.tile_records_recv_x_rows[:total_tiles].view(
        -1
    )  # (total_tiles*tile_m,)
    valid_mask = rows >= 0
    A_idx_flat = rows[valid_mask].contiguous()
    TK = int(A_idx_flat.shape[0])
    assert int(expert_frequency.sum().item()) == TK, "expert_frequency should sum to TK"

    cu_seqlens_m = expert_frequency_offset.to(torch.int32).contiguous()  # (E_local+1,)

    # Outputs.
    postact_flat = torch.empty(TK, I, dtype=DTYPE, device=device)
    preact_flat = torch.empty(TK, 2 * I, dtype=DTYPE, device=device)

    # W1 layout for gemm_gated: (l=E_local, n=2I, k=H), k-major. We already have
    # w1_local of shape (E_local, 2I, H) contiguous → that IS k-major per expert.
    # gemm_gated handles this directly.
    W1_for_gated = w1_local

    # Streaming-side W1: (E_local, 2I, H) → (2I, H, E_local) view (k-major slab
    # per expert) — this is what streaming_moe_a does internally via permute.
    # Pre-permute outside the timed call to remove that ~25 μs/launch.
    # streaming_moe_a still applies the permute internally; we'd need a separate
    # path to skip it. For this bench, just leave it — its overhead is ~25 μs
    # and the streaming kernel takes thousands of μs, so it's noise.

    if rank == 0:
        print(
            f"\nconfig: H={H} I={I} E_total={NUM_EXPERTS} E_local={local_E} "
            f"K={TOPK} T_per_rank={SEQ_LEN_PER_RANK} world={world_size}"
        )
        print(
            f"        T_recv={T_recv} total_tiles={total_tiles} "
            f"TK={TK} sentinels_dropped={int(rows.numel() - TK)}"
        )
        flops = (
            2 * 2 * TK * H * I
        )  # gated → 2I output halved → I; both halves computed → 2*I; matmul 2*M*N*K
        print(f"        FLOPs/launch: {flops / 1e9:.2f} GFLOPs (bf16 matmul)")
        print()
        print(
            f"{'kernel':>22s}  {'tile_M':>6s}  {'tile_N':>6s}  {'time (μs)':>10s}  {'TFLOPs/s':>10s}"
        )
        print(f"{'-'*22}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}")

    def fmt_row(name, tm, tn, t_us):
        if rank != 0:
            return
        flops = 2 * 2 * TK * H * I
        tflops = flops / (t_us * 1e-6) / 1e12
        print(f"{name:>22s}  {tm:>6d}  {tn:>6d}  {t_us:>10.1f}  {tflops:>10.2f}")

    # ────────────────────────────────────────────────────────────────────
    # Controlled experiment: force all tiles to expert 0 (and gemm_gated to a
    # single batch). If L2 reuse on W1[e] is the cause of the streaming-vs-
    # non-streaming gap, both should now run at the same speed (since every
    # tile reads the same W1[0] and L2 is always warm).
    # ────────────────────────────────────────────────────────────────────
    handle_expert_orig = handle.tile_records_expert_id.clone()
    handle.tile_records_expert_id[:total_tiles] = 0  # in-place override
    cu_seqlens_m_single = torch.tensor([0, TK], dtype=torch.int32, device=device)
    if rank == 0:
        print()
        print("=== single-expert control (all tiles → expert 0) ===")
        print(
            f"{'kernel':>22s}  {'tile_M':>6s}  {'tile_N':>6s}  {'time (μs)':>10s}  {'TFLOPs/s':>10s}"
        )
        print(f"{'-'*22}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}")

    for tile_n in (128, 256):
        consumer_head = torch.zeros(1, dtype=torch.int32, device=device)
        postact_se = torch.empty(total_tiles, TILE_M, I, dtype=DTYPE, device=device)

        def run_streaming_se():
            consumer_head.zero_()
            streaming_moe_a(
                recv_x,
                w1_local,
                postact_se,
                handle.tile_records_recv_x_rows,
                handle.tile_records_expert_id,
                handle.tile_ready,
                consumer_head,
                dispatch_seq=handle.dispatch_seq,
                tile_m=TILE_M,
                tile_n=tile_n,
            )

        try:
            t = time_kernel(run_streaming_se, warmup=3, iters=10)
            fmt_row("streaming_moe_a [SE]", TILE_M, tile_n, t)
        except Exception as e:
            if rank == 0:
                print(
                    f"streaming_moe_a SE tile_n={tile_n}: FAILED — {type(e).__name__}: {e}"
                )

    for tile_M, tile_N in [(128, 256)]:

        def run_gated_se():
            gemm_act(
                recv_x,
                w1_local[:1].contiguous(),  # only expert 0's slab
                preact_flat,
                None,
                postact_flat,
                None,
                "swiglu",
                tile_M=tile_M,
                tile_N=tile_N,
                cluster_M=1,
                cluster_N=1,
                cu_seqlens_m=cu_seqlens_m_single,
                A_idx=A_idx_flat,
            )

        try:
            t = time_kernel(run_gated_se, warmup=3, iters=10)
            fmt_row("gemm_gated [SE]", tile_M, tile_N, t)
        except Exception as e:
            if rank == 0:
                print(
                    f"gemm_gated SE tile_M={tile_M} tile_N={tile_N}: FAILED — {type(e).__name__}: {e}"
                )

    # Restore original expert ids for the cross-check below.
    handle.tile_records_expert_id[:total_tiles] = handle_expert_orig[:total_tiles]
    if rank == 0:
        print()
        print("=== production-shape (multi-expert) ===")
        print(
            f"{'kernel':>22s}  {'tile_M':>6s}  {'tile_N':>6s}  {'time (μs)':>10s}  {'TFLOPs/s':>10s}"
        )
        print(f"{'-'*22}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}")

    # ── Streaming kernel A across tile_n sweep.
    for tile_n in (128, 256, 512):
        consumer_head = torch.zeros(1, dtype=torch.int32, device=device)
        postact_streaming = torch.empty(
            total_tiles, TILE_M, I, dtype=DTYPE, device=device
        )

        def run_streaming():
            consumer_head.zero_()
            streaming_moe_a(
                recv_x,
                w1_local,
                postact_streaming,
                handle.tile_records_recv_x_rows,
                handle.tile_records_expert_id,
                handle.tile_ready,
                consumer_head,
                dispatch_seq=handle.dispatch_seq,
                tile_m=TILE_M,
                tile_n=tile_n,
            )

        try:
            t = time_kernel(run_streaming, warmup=3, iters=10)
            fmt_row("streaming_moe_a", TILE_M, tile_n, t)
        except Exception as e:
            if rank == 0:
                print(
                    f"streaming_moe_a tile_n={tile_n}: FAILED — {type(e).__name__}: {e}"
                )

    # ── Non-streaming gemm_gated across (tile_M, tile_N) sweep.
    # gemm_gated is the reference: same total work, no streaming overhead, no
    # gather-overhead beyond the standard varlen_m + gather_A path.
    for tile_M, tile_N in [(128, 256), (128, 512), (256, 256), (256, 512)]:

        def run_gated():
            gemm_act(
                recv_x,  # A (T_recv, H), bf16, k-major
                W1_for_gated,  # B (E_local, 2I, H)
                preact_flat,  # D (TK, 2I)
                None,  # C
                postact_flat,  # PostAct (TK, I)
                None,  # tile_count_semaphore
                "swiglu",
                tile_M=tile_M,
                tile_N=tile_N,
                cluster_M=1,
                cluster_N=1,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx_flat,
            )

        try:
            t = time_kernel(run_gated, warmup=3, iters=10)
            fmt_row("gemm_gated (varlen_m)", tile_M, tile_N, t)
        except Exception as e:
            if rank == 0:
                print(
                    f"gemm_gated tile_M={tile_M} tile_N={tile_N}: FAILED — {type(e).__name__}: {e}"
                )

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
