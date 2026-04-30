"""Bench streaming kernel A vs the non-streaming gemm_gated on the same
per-tile work (pool layout). Both run on a single GPU with a Buffer.dispatch'd
StreamingHandle as the source of truth for what gets computed.

Setup
-----
- Run Buffer.dispatch once (multi-GPU required for DeepEP).
- Pool layout puts data into expert-major order with BLOCK_M-padded tiles, so
  both kernels read the SAME pool tensor:
    * streaming_moe_a uses tile_id_to_expert + expert_pool_block_offset and
      strided TMA via the standard varlen_m path.
    * gemm_gated uses cu_seqlens_m = expert_pool_block_offset * tile_m and
      reads pool dense (no A_idx needed — pool is already expert-grouped).
- Time both kernels (CUDA-event timed, repeated, median).
- Sweep tile_n for each variant.

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
    pool, handle, _ = buffer.dispatch(
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
    TK_padded = pool.shape[0]

    # Pool layout: cu_seqlens_m for both kernels is expert_pool_block_offset *
    # tile_m. Pool is already expert-grouped, so neither kernel needs A_idx.
    expert_frequency = handle.expert_frequency
    expert_pool_block_offset = handle.expert_pool_block_offset
    cu_seqlens_m = (expert_pool_block_offset.to(torch.int32) * TILE_M).contiguous()
    TK = int(expert_frequency.sum().item())  # actual (token, k) pair count (no padding)

    # Outputs sized to match the pool layout's per-expert padded blocks.
    postact_flat = torch.empty(TK_padded, I, dtype=DTYPE, device=device)
    preact_flat = torch.empty(TK_padded, 2 * I, dtype=DTYPE, device=device)

    if rank == 0:
        print(
            f"\nconfig: H={H} I={I} E_total={NUM_EXPERTS} E_local={local_E} "
            f"K={TOPK} T_per_rank={SEQ_LEN_PER_RANK} world={world_size}"
        )
        print(
            f"        TK_padded={TK_padded} total_tiles={total_tiles} "
            f"TK={TK} padding_rows={TK_padded - TK}"
        )
        flops = 2 * 2 * TK * H * I  # gated: 2*M*N*K with N = 2I → halved → I
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
    # Single-expert control: force all tiles to expert 0 (override
    # tile_id_to_expert and use a single-batch cu_seqlens_m). Tests whether
    # multi-expert L2 thrashing on W1[e] is what slows the multi-expert run.
    # ────────────────────────────────────────────────────────────────────
    handle_expert_orig = handle.tile_id_to_expert.clone()
    handle.tile_id_to_expert[:total_tiles] = 0  # force all tiles to expert 0
    cu_seqlens_m_single = torch.tensor([0, TK_padded], dtype=torch.int32, device=device)
    expert_pool_block_offset_se = torch.tensor(
        [0, total_tiles] + [total_tiles] * (local_E - 1),
        dtype=torch.int32,
        device=device,
    )
    if rank == 0:
        print()
        print("=== single-expert control (all tiles → expert 0) ===")
        print(
            f"{'kernel':>22s}  {'tile_M':>6s}  {'tile_N':>6s}  {'time (μs)':>10s}  {'TFLOPs/s':>10s}"
        )
        print(f"{'-'*22}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}")

    se_streaming_us = None
    se_gated_us = None
    for tile_n in (128, 256):
        consumer_head = torch.zeros(1, dtype=torch.int32, device=device)
        postact_se = torch.empty(total_tiles, TILE_M, I, dtype=DTYPE, device=device)

        def run_streaming_se():
            consumer_head.zero_()
            streaming_moe_a(
                pool,
                w1_local,
                postact_se,
                handle.tile_id_to_expert,
                expert_pool_block_offset_se,
                handle.tile_ready,
                consumer_head,
                dispatch_seq=handle.dispatch_seq,
                tile_m=TILE_M,
                tile_n=tile_n,
            )

        try:
            t = time_kernel(run_streaming_se)
            fmt_row("streaming_moe_a [SE]", TILE_M, tile_n, t)
            if tile_n == 256:
                se_streaming_us = t
        except Exception as e:
            if rank == 0:
                print(
                    f"streaming_moe_a SE tile_n={tile_n}: FAILED — {type(e).__name__}: {e}"
                )

    def run_gated_se():
        gemm_act(
            pool,
            w1_local[:1].contiguous(),  # only expert 0's slab
            preact_flat,
            None,
            postact_flat,
            None,
            "swiglu",
            tile_M=128,
            tile_N=256,
            cluster_M=1,
            cluster_N=1,
            cu_seqlens_m=cu_seqlens_m_single,
            A_idx=None,
        )

    try:
        t = time_kernel(run_gated_se)
        fmt_row("gemm_gated [SE]", 128, 256, t)
        se_gated_us = t
    except Exception as e:
        if rank == 0:
            print(f"gemm_gated SE: FAILED — {type(e).__name__}: {e}")

    # Restore original expert ids for the multi-expert run.
    handle.tile_id_to_expert[:total_tiles] = handle_expert_orig[:total_tiles]
    if rank == 0:
        print()
        print("=== production-shape (multi-expert) ===")
        print(
            f"{'kernel':>22s}  {'tile_M':>6s}  {'tile_N':>6s}  {'time (μs)':>10s}  {'TFLOPs/s':>10s}"
        )
        print(f"{'-'*22}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}")

    streaming_us = None
    for tile_n in (128, 256):
        consumer_head = torch.zeros(1, dtype=torch.int32, device=device)
        postact_streaming = torch.empty(
            total_tiles, TILE_M, I, dtype=DTYPE, device=device
        )

        def run_streaming():
            consumer_head.zero_()
            streaming_moe_a(
                pool,
                w1_local,
                postact_streaming,
                handle.tile_id_to_expert,
                handle.expert_pool_block_offset,
                handle.tile_ready,
                consumer_head,
                dispatch_seq=handle.dispatch_seq,
                tile_m=TILE_M,
                tile_n=tile_n,
            )

        try:
            t = time_kernel(run_streaming)
            fmt_row("streaming_moe_a", TILE_M, tile_n, t)
            if tile_n == 256:
                streaming_us = t
        except Exception as e:
            if rank == 0:
                print(
                    f"streaming_moe_a tile_n={tile_n}: FAILED — {type(e).__name__}: {e}"
                )

    gated_us = None

    def run_gated():
        gemm_act(
            pool,
            w1_local,  # B (E_local, 2I, H)
            preact_flat,  # D (TK_padded, 2I)
            None,
            postact_flat,  # PostAct (TK_padded, I)
            None,
            "swiglu",
            tile_M=128,
            tile_N=256,
            cluster_M=1,
            cluster_N=1,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=None,
        )

    try:
        t = time_kernel(run_gated)
        fmt_row("gemm_gated (varlen_m)", 128, 256, t)
        gated_us = t
    except Exception as e:
        if rank == 0:
            print(f"gemm_gated: FAILED — {type(e).__name__}: {e}")

    if rank == 0:
        print()
        print("=== summary (multi-expert, tile_M=128, tile_N=256) ===")
        if streaming_us is not None and gated_us is not None:
            ratio = streaming_us / gated_us
            verdict = "FASTER" if ratio < 1.0 else "slower"
            print(f"  streaming_moe_a:       {streaming_us:7.1f} μs")
            print(f"  gemm_gated (varlen_m): {gated_us:7.1f} μs")
            print(
                f"  streaming / gemm_gated: {ratio:.3f}x ({verdict} than non-streaming reference)"
            )
        if se_streaming_us is not None and se_gated_us is not None:
            print(
                f"  [single-expert control] streaming {se_streaming_us:7.1f} μs vs "
                f"gemm_gated {se_gated_us:7.1f} μs ({se_streaming_us/se_gated_us:.3f}x)"
            )

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
