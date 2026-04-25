"""Phase 0 validation: streaming-overlap ceiling for DeepEP + SonicMoE.

Launch with:
    torchrun --nproc_per_node=8 -m evolutionaryscale.models.moe.streaming_moe.validate_phase0

Measures, per (num_sms, topk) sweep config, the isolated CUDA time of:
    t_dispatch  - DeepEPBuffer.dispatch
    t_compute   - SonicMoE up + SwiGLU + down on the dispatched tokens
    t_combine   - DeepEPBuffer.combine

and computes the streaming ceiling
    ceiling = (t_disp + t_comp + t_comb - max(...)) / (t_disp + t_comp + t_comb)

which upper-bounds the step-time savings if all three stages overlapped perfectly.

8 GPUs single node => DeepEP intranode kernel path. t_dispatch / t_combine are
NVL-only and represent a lower bound on multi-node values; multi-node ceilings
will be at least as large.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import torch
import torch.distributed as torch_dist
from deep_ep import Buffer as DeepEPBuffer
from quack.gemm_interface import gemm, gemm_gated

from evolutionaryscale.utils.distributed import (
    Axes,
    ParallelConfig,
    get_parallelism_group,
    init_distributed,
    parallel_context,
)

# Dimensions cribbed from moe_17BA840M / glmalpha2_18b_moe (~18B MoE config).
D_MODEL = 2048
INTERMEDIATE_SIZE = 384
NUM_EXPERTS = 384
SEQ_LEN_PER_RANK = 8192
DTYPE = torch.bfloat16

NUM_SMS_SWEEP = (16, 24, 32, 48, 56)
TOPK_SWEEP = (1, 4, 8, 12, 16)


@dataclass
class TimingResult:
    num_sms: int
    topk: int
    t_dispatch_ms: float
    t_compute_ms: float
    t_combine_ms: float

    @property
    def total(self) -> float:
        return self.t_dispatch_ms + self.t_compute_ms + self.t_combine_ms

    @property
    def ceiling_pct(self) -> float:
        m = max(self.t_dispatch_ms, self.t_compute_ms, self.t_combine_ms)
        return 100.0 * (self.total - m) / self.total


def make_uniform_topk_idx(
    n_tokens: int, topk: int, num_experts: int, rank: int, device: torch.device
) -> torch.Tensor:
    """Deterministic uniform routing: every expert gets the same global token count."""
    base = (torch.arange(n_tokens, device=device) + rank * n_tokens) * topk
    offsets = torch.arange(topk, device=device).unsqueeze(0)
    return ((base.unsqueeze(1) + offsets) % num_experts).to(torch.int64)


def make_buffer(group, num_sms: int) -> DeepEPBuffer:
    DeepEPBuffer.set_num_sms(num_sms)
    hidden_bytes = D_MODEL * max(torch.tensor([], dtype=DTYPE).element_size(), 2)
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
        group,
        nvl_bytes,
        rdma_bytes,
        num_qps_per_rank=DeepEPBuffer.num_sms,
        use_default_stream_as_comm_stream=False,
    )


def time_fn(fn, n_warmup: int, n_iter: int) -> float:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    torch_dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def build_expanded_metadata(
    recv_topk_idx: torch.Tensor,  # [N_recv, K] local expert ids (-1 = padding)
    local_num_experts: int,
):
    """Expand DeepEP's compact (per-unique-token) recv layout into the per-(token,
    expert) sorted layout that grouped-GEMM consumes.

    Returns:
        x_gather_idx: [TK] int32, x_gather_idx[r] = source row in recv_x for
            the r-th expanded slot, sorted by destination expert.
        expert_frequency_offset: [E_local + 1] int32 cumsum, gives [start, end)
            row range in expanded space for each local expert.
        TK: total number of valid (token, expert) pairs on this rank.
    """
    device = recv_topk_idx.device
    N, K = recv_topk_idx.shape
    flat_expert = recv_topk_idx.reshape(-1)
    flat_token = (
        torch.arange(N, device=device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(N, K)
        .reshape(-1)
    )
    valid = flat_expert >= 0
    expert = flat_expert[valid]
    token = flat_token[valid]
    order = expert.argsort(stable=True)
    x_gather_idx = token[order].contiguous()
    sorted_expert = expert[order]
    counts = torch.bincount(sorted_expert, minlength=local_num_experts)
    expert_frequency_offset = torch.zeros(
        local_num_experts + 1, dtype=torch.int32, device=device
    )
    expert_frequency_offset[1:] = counts.to(torch.int32).cumsum(0).to(torch.int32)
    return x_gather_idx, expert_frequency_offset, int(x_gather_idx.numel())


def run_sweep(
    rank: int, world_size: int, group, n_warmup: int, n_iter: int
) -> list[TimingResult]:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    local_num_experts = NUM_EXPERTS // world_size

    # QuACK grouped GEMM expects B as (E, K, N) with the contraction dim K
    # contiguous. We mirror what sonicmoe.functional does: store as (E, *, *)
    # and pass `w.permute(2, 1, 0)` for the up-proj contraction shape and
    # `w.permute(2, 1, 0)` for down-proj.
    # gemm_gated: A=[TK, H], B=[E, H, 2*I] (contracts H), output [TK, 2*I] split into z,y1
    # gemm:       A=[TK, I], B=[E, I, H]  (contracts I), output [TK, H]
    w1 = torch.randn(
        local_num_experts, 2 * INTERMEDIATE_SIZE, D_MODEL, dtype=DTYPE, device=device
    ).mul_(0.02)
    w2 = torch.randn(
        local_num_experts, D_MODEL, INTERMEDIATE_SIZE, dtype=DTYPE, device=device
    ).mul_(0.02)
    # Pre-permute to QuACK's expected (E, contract_dim, out_dim) layout.
    w1_q = w1.permute(0, 2, 1).contiguous()  # (E, H, 2*I)
    w2_q = w2.permute(0, 2, 1).contiguous()  # (E, I, H)

    x = torch.randn(SEQ_LEN_PER_RANK, D_MODEL, dtype=DTYPE, device=device) * 0.1

    results: list[TimingResult] = []

    for num_sms in NUM_SMS_SWEEP:
        buffer = make_buffer(group, num_sms)
        for topk in TOPK_SWEEP:
            topk_idx = make_uniform_topk_idx(
                SEQ_LEN_PER_RANK, topk, NUM_EXPERTS, rank, device
            )
            topk_weights = torch.full(
                (SEQ_LEN_PER_RANK, topk), 1.0 / topk, dtype=torch.float32, device=device
            )

            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                _layout_evt,
            ) = buffer.get_dispatch_layout(topk_idx, NUM_EXPERTS, async_finish=False)

            def do_dispatch(buffer=buffer):
                return buffer.dispatch(
                    x,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    num_tokens_per_rank=num_tokens_per_rank,
                    num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                    is_token_in_rank=is_token_in_rank,
                    num_tokens_per_expert=num_tokens_per_expert,
                    expert_alignment=1,
                    async_finish=False,
                    allocate_on_comm_stream=False,
                    num_recv_tokens_per_expert_as_cuda=True,
                )

            t_dispatch = time_fn(do_dispatch, n_warmup, n_iter)

            (
                recv_x,
                _recv_topk_idx,
                _recv_topk_weights,
                num_recv_per_expert,
                handle,
                _,
            ) = do_dispatch()
            assert isinstance(recv_x, torch.Tensor), "FP8 path not exercised here"

            if recv_x.size(0) == 0:
                if rank == 0:
                    print(
                        f"[skip] num_sms={num_sms} topk={topk}: empty dispatch on this rank"
                    )
                results.append(TimingResult(num_sms, topk, t_dispatch, 0.0, 0.0))
                continue

            x_gather_idx, expert_frequency_offset, tk_total = build_expanded_metadata(
                _recv_topk_idx, local_num_experts
            )

            identity_idx = torch.arange(tk_total, dtype=torch.int32, device=device)

            def do_compute():
                # Up + SwiGLU fused: gather recv_x[x_gather_idx] then grouped-GEMM
                # returns (preact z [TK, 2*I], postact y1 [TK, I]).
                _z, y1 = gemm_gated(
                    recv_x,
                    w1_q,
                    activation="swiglu",
                    cu_seqlens_m=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    dynamic_scheduler=False,
                )
                # Down: y1 [TK, I] @ w2[E, I, H] -> y2 [TK, H]
                y2 = gemm(
                    y1,
                    w2_q,
                    cu_seqlens_m=expert_frequency_offset,
                    A_idx=identity_idx,
                    dynamic_scheduler=False,
                )
                return y2

            t_compute = time_fn(do_compute, n_warmup, n_iter)

            # Combine input has the compact [N_recv, H] layout (one row per
            # unique source token), not the expanded [TK, H]. We time combine
            # on a properly-shaped tensor; the exact contents don't affect
            # kernel time.
            combine_in = recv_x.clone()

            def do_combine(buffer=buffer):
                return buffer.combine(
                    combine_in,
                    handle,
                    async_finish=False,
                    allocate_on_comm_stream=False,
                )

            t_combine = time_fn(do_combine, n_warmup, n_iter)

            results.append(
                TimingResult(num_sms, topk, t_dispatch, t_compute, t_combine)
            )
            if rank == 0:
                r = results[-1]
                print(
                    f"num_sms={num_sms:3d} topk={topk:2d}  "
                    f"disp={r.t_dispatch_ms:7.3f}ms  "
                    f"comp={r.t_compute_ms:7.3f}ms  "
                    f"comb={r.t_combine_ms:7.3f}ms  "
                    f"ceiling={r.ceiling_pct:5.1f}%"
                )

        del buffer
        torch.cuda.empty_cache()

    return results


def reduce_across_ranks(values: list[float], op: str, group) -> list[float]:
    """Reduce per-config timings across ranks. op in {'mean', 'max'}."""
    t = torch.tensor(values, dtype=torch.float64, device="cuda")
    if op == "mean":
        torch_dist.all_reduce(t, op=torch_dist.ReduceOp.SUM, group=group)
        t /= group.size()
    elif op == "max":
        torch_dist.all_reduce(t, op=torch_dist.ReduceOp.MAX, group=group)
    else:
        raise ValueError(op)
    return t.cpu().tolist()


def print_summary(results: list[TimingResult], step_time_ms: float | None) -> None:
    print("\n" + "=" * 90)
    print("PHASE 0 STREAMING CEILING (rank 0 view; per-rank values)")
    print("=" * 90)
    header = (
        f"{'num_sms':>8} {'topk':>5} {'sparsity':>9} {'dispatch':>10} {'compute':>10} "
        f"{'combine':>10} {'sum':>10} {'max':>10} {'ceiling':>9}"
    )
    print(header)
    print("-" * len(header))
    best = None
    for r in results:
        sparsity = r.topk / NUM_EXPERTS
        m = max(r.t_dispatch_ms, r.t_compute_ms, r.t_combine_ms)
        print(
            f"{r.num_sms:>8d} {r.topk:>5d} {sparsity:>9.4f} "
            f"{r.t_dispatch_ms:>10.3f} {r.t_compute_ms:>10.3f} {r.t_combine_ms:>10.3f} "
            f"{r.total:>10.3f} {m:>10.3f} {r.ceiling_pct:>8.1f}%"
        )
        if best is None or r.ceiling_pct > best.ceiling_pct:
            best = r
    print("-" * len(header))
    if best is not None:
        print(
            f"max ceiling: {best.ceiling_pct:.1f}% at num_sms={best.num_sms}, topk={best.topk} "
            f"(disp/comp/comb = {best.t_dispatch_ms:.2f}/{best.t_compute_ms:.2f}/{best.t_combine_ms:.2f} ms)"
        )
    if step_time_ms is not None:
        print(f"\nUsing user-supplied step time = {step_time_ms:.2f} ms/step:")
        for r in results:
            sparsity = r.topk / NUM_EXPERTS
            m = max(r.t_dispatch_ms, r.t_compute_ms, r.t_combine_ms)
            saving = (r.total - m) / step_time_ms * 100.0
            print(
                f"  num_sms={r.num_sms:3d} topk={r.topk:2d} sparsity={sparsity:.4f}: "
                f"projected step-time saving <= {saving:5.2f}%"
            )
    print(
        "\nReminder: 8-GPU single node => DeepEP intranode kernel; t_dispatch / t_combine are "
        "NVL-only.\nMulti-node t_dispatch is typically larger (RDMA), so the ceiling is a lower "
        "bound on real gains."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument(
        "--step_time_ms",
        type=float,
        default=None,
        help="Optional: full training step time (ms). If supplied, prints projected step-time savings.",
    )
    p.add_argument(
        "--out_json",
        type=str,
        default="/tmp/phase0_results.json",
        help="Where to dump per-config timings (rank 0 only).",
    )
    args = p.parse_args()

    init_distributed()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 8:
        if rank == 0:
            print(f"WARNING: expected 8 ranks (one node), got {world_size}")

    pc = ParallelConfig(dp_replicate_degree=1, ep_degree=world_size)
    with parallel_context(parallel_config=pc):
        ep_group = get_parallelism_group(Axes.EP)
        assert ep_group is not None

        results = run_sweep(rank, world_size, ep_group, args.n_warmup, args.n_iter)

        disp = reduce_across_ranks([r.t_dispatch_ms for r in results], "max", ep_group)
        comp = reduce_across_ranks([r.t_compute_ms for r in results], "max", ep_group)
        comb = reduce_across_ranks([r.t_combine_ms for r in results], "max", ep_group)
        results_max = [
            TimingResult(r.num_sms, r.topk, d, c, b)
            for r, d, c, b in zip(results, disp, comp, comb, strict=True)
        ]

        if rank == 0:
            print_summary(results_max, args.step_time_ms)
            with open(args.out_json, "w") as f:
                json.dump(
                    {
                        "config": {
                            "d_model": D_MODEL,
                            "intermediate_size": INTERMEDIATE_SIZE,
                            "num_experts": NUM_EXPERTS,
                            "seq_len_per_rank": SEQ_LEN_PER_RANK,
                            "world_size": world_size,
                            "step_time_ms": args.step_time_ms,
                        },
                        "results": [
                            {
                                "num_sms": r.num_sms,
                                "topk": r.topk,
                                "sparsity": r.topk / NUM_EXPERTS,
                                "t_dispatch_ms": r.t_dispatch_ms,
                                "t_compute_ms": r.t_compute_ms,
                                "t_combine_ms": r.t_combine_ms,
                                "ceiling_pct": r.ceiling_pct,
                            }
                            for r in results_max
                        ],
                    },
                    f,
                    indent=2,
                )
            print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
