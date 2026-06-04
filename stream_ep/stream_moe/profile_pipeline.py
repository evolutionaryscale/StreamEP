"""Torch-profiler trace of the streaming pipeline (fwd + bwd) plus a
per-kernel summary table parsed from the profiler's in-memory event
aggregator.

This is the source of truth for collective-kernel per-call timing
(dispatch / dispatch_grads / combine / combine_grads): the collective ops
need natural per-iter slack to satisfy the single-slot `rdma_channel_meta`
protocol, which the full fwd+bwd pipeline provides but a rapid-fire bench
loop does not. It also reports end-to-end overlap savings and peak memory.

What you should see in the resulting chrome trace
-------------------------------------------------
- `communicate stream`: dispatch_metadata → host poll → tile_arrays_init
  → dispatch main → combine. Combine FIFOs after dispatch, but waits on
  a torch.cuda.Event recorded on the compute stream so kernel Y wins the
  SM race.
- `compute stream`: streaming_moe_a → streaming_moe_y. Same-stream FIFO
  covers the A → Y handoff (no per-tile A→Y signal). Kernel A's
  CTAs spin on pool_arrival_count[tile] == pool_arrival_target[tile] for
  the dispatch ↔ A overlap window. Combine's per-warp sender then picks
  up Y tiles as they retire via the per-token y_done_per_token gate.

Launch
------
    torchrun --nproc_per_node=2 \\
        -m stream_ep.stream_moe.profile_pipeline \\
        [--profile_dir /path/to/profiles]

Default --profile_dir is `/tmp/stream_moe_profile_<SLURM_JOB_ID-or-pid>/`.
Trace files land per-rank in that directory. Open the .json with
chrome://tracing or point TensorBoard at the dir.
"""

import argparse
import datetime as _datetime
import os

import torch
import torch.distributed as torch_dist
import torch.profiler
from stream_ep import Buffer as StreamEPBuffer

from stream_ep.stream_moe.stream_moe import (
    make_streams,
    stream_moe_func,
)


# torchrun-driven distributed helpers. Kept here so smoke_pipeline /
# validate_multi_iter can reuse them by importing from
# profile_pipeline.
def init_distributed() -> torch.device:
    """Init NCCL process group from torchrun env vars; return the local cuda device."""
    local_rank = int(os.environ["LOCAL_RANK"])

    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend="nccl", device_id=local_rank)
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def get_global_rank() -> int:
    return torch_dist.get_rank()


def get_world_size() -> int:
    return torch_dist.get_world_size()


def barrier(group=None) -> None:
    torch_dist.barrier(group=group)


def rank_zero_print(*args, **kwargs) -> None:
    if torch_dist.is_initialized() and torch_dist.get_rank() != 0:
        return
    print(*args, **kwargs)


def rank_zero_only(fn):
    """Decorator: only rank 0 actually executes; other ranks return None."""
    def wrapper(*args, **kwargs):
        if torch_dist.is_initialized() and torch_dist.get_rank() != 0:
            return None
        return fn(*args, **kwargs)
    return wrapper

H = 2048
I = 384
NUM_EXPERTS = 384
SEQ_LEN_PER_RANK = 8192
TOPK = 13
DTYPE = torch.bfloat16
NUM_SMS = 64  # 64 SMs (32 channels): tuned for 4+ node RDMA-dominated runs.
TILE_M = 128
TILE_N_A = 192
TILE_N_Y = 256


def make_uniform_topk_idx(n_tokens, topk, num_experts, rank, device):
    base = (torch.arange(n_tokens, device=device) + rank * n_tokens) * topk
    offsets = torch.arange(topk, device=device).unsqueeze(0)
    return ((base.unsqueeze(1) + offsets) % num_experts).to(torch.int64)


def make_skewed_topk_idx(
    n_tokens, topk, num_experts, hot_frac, hot_weight, device, generator
):
    """Biased multinomial routing: the first ``hot_frac`` of experts get
    ``hot_weight`` × the per-token sampling weight of the rest, sampled without
    replacement so each token still gets ``topk`` distinct experts. Used to
    exercise the heavy-comm regime — under hot_frac=0.25, hot_weight=4.0 the
    expected token share for the hot 25% of experts is ~57% (vs the 25%
    uniform), which roughly doubles dispatch / combine traffic on the
    hot ranks and surfaces the streaming-overlap behavior that's invisible
    at uniform routing.
    """
    n_hot = max(1, int(num_experts * hot_frac))
    weights = torch.ones(num_experts, device=device)
    weights[:n_hot] *= hot_weight
    probs = weights.expand(n_tokens, -1).contiguous()
    return torch.multinomial(probs, topk, replacement=False, generator=generator).to(
        torch.int64
    )


def make_buffer(group, num_sms=None):
    if num_sms is not None:
        StreamEPBuffer.set_num_sms(num_sms)
    hidden_bytes = H * 2
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (
        StreamEPBuffer.get_dispatch_config(group.size()),
        StreamEPBuffer.get_combine_config(group.size()),
    ):
        nvl_bytes = max(
            cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()), nvl_bytes
        )
        rdma_bytes = max(
            cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()), rdma_bytes
        )
    return StreamEPBuffer(group, nvl_bytes, rdma_bytes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--profile_dir",
        type=str,
        default=None,
        help="Directory for chrome traces + per-rank profile artifacts. "
        "Default: /tmp/stream_moe_profile_<YYYYmmdd_HHMMSS>/ (rank-0 picked, "
        "broadcast to peers so all ranks write to the same directory).",
    )
    p.add_argument("--num_sms", type=int, default=None,
                   help="StreamEP num_sms override; default = Buffer auto-pick.")
    p.add_argument("--seq_len", type=int, default=SEQ_LEN_PER_RANK)
    p.add_argument("--num_sms_a", type=int, default=None)
    p.add_argument("--num_sms_y", type=int, default=None)
    p.add_argument("--num_sms_a_bwd", type=int, default=None)
    p.add_argument("--num_sms_y_bwd", type=int, default=None)
    p.add_argument("--prioritize_communicate", action="store_true")
    p.add_argument(
        "--recompute_preact",
        action="store_true",
        help="Activation-checkpoint preact_a: skip the fwd [2I] save and "
        "recompute it from pool @ W1 in bwd. Measures the recompute's e2e "
        "cost and peak-memory saving vs the default (save preact_a).",
    )
    p.add_argument("--tile_m", type=int, default=TILE_M)
    p.add_argument("--tile_n_a", type=int, default=TILE_N_A)
    p.add_argument("--tile_n_y", type=int, default=TILE_N_Y)
    p.add_argument("--tile_n_a_bwd", type=int, default=256)
    p.add_argument("--tile_n_y_bwd", type=int, default=128)
    # dW grouped-GEMM tile knobs. None → fall back to (tile_m, tile_n_a) at the
    # bwd call site, matching pre-decouple behaviour.
    p.add_argument("--tile_m_dW1", type=int, default=None)
    p.add_argument("--tile_n_dW1", type=int, default=256)
    p.add_argument("--tile_m_dW2", type=int, default=None)
    p.add_argument("--tile_n_dW2", type=int, default=None)
    p.add_argument("--cluster_m_dW1", type=int, default=2)
    p.add_argument("--cluster_n_dW1", type=int, default=2)
    p.add_argument("--cluster_m_dW2", type=int, default=1)
    p.add_argument("--cluster_n_dW2", type=int, default=1)
    p.add_argument("--pingpong_dW1", action="store_true")
    p.add_argument("--pingpong_dW2", action="store_true")
    p.add_argument("--swizzle_dW1", type=int, default=8)
    p.add_argument("--swizzle_dW2", type=int, default=8)
    p.add_argument(
        "--skew_hot_frac",
        type=float,
        default=0.0,
        help="If >0, use biased routing instead of uniform: the first hot_frac "
        "of experts get hot_weight× sampling weight. 0 (default) = uniform.",
    )
    p.add_argument("--skew_hot_weight", type=float, default=4.0)
    args = p.parse_args()

    device = init_distributed()
    rank, world_size = get_global_rank(), get_world_size()
    group = torch_dist.group.WORLD
    local_E = NUM_EXPERTS // world_size

    # Default profile_dir is a per-run timestamped /tmp directory. Rank 0
    # picks the stamp and broadcasts it so all ranks agree (otherwise
    # torchrun's parallel python spawn can straddle a second boundary and
    # leave ranks writing to different dirs).
    if args.profile_dir is None:
        if rank == 0:
            stamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.profile_dir = f"/tmp/stream_moe_profile_{stamp}"
        obj_list = [args.profile_dir]
        torch_dist.broadcast_object_list(obj_list, src=0)
        args.profile_dir = obj_list[0]

    buffer = make_buffer(group, args.num_sms)

    # Replicated W1 / W2 across ranks (each rank slices its E_local share).
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
    w1_local.requires_grad_(True)
    w2_local.requires_grad_(True)

    torch.manual_seed(100 + rank)
    x = (torch.randn(args.seq_len, H, dtype=DTYPE, device=device) * 0.1).contiguous()
    x.requires_grad_(True)
    if args.skew_hot_frac > 0:
        skew_gen = torch.Generator(device=device).manual_seed(7000 + rank)
        topk_idx = make_skewed_topk_idx(
            args.seq_len,
            TOPK,
            NUM_EXPERTS,
            hot_frac=args.skew_hot_frac,
            hot_weight=args.skew_hot_weight,
            device=device,
            generator=skew_gen,
        )
    else:
        topk_idx = make_uniform_topk_idx(args.seq_len, TOPK, NUM_EXPERTS, rank, device)
    topk_weights = torch.softmax(
        torch.randn(args.seq_len, TOPK, dtype=torch.float32, device=device), dim=-1
    ).contiguous()
    topk_weights.requires_grad_(True)

    rank_idx = topk_idx // local_E
    is_token_in_rank = torch.zeros(
        (args.seq_len, world_size), dtype=torch.bool, device=device
    )
    for r in range(world_size):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)

    streams = make_streams(prioritize_communicate=args.prioritize_communicate)

    os.makedirs(args.profile_dir, exist_ok=True)
    if rank == 0:
        print(f"writing traces to {args.profile_dir}/", flush=True)
        print(
            f"config: world={world_size} num_sms={StreamEPBuffer.num_sms} "
            f"H={H} I={I} E={NUM_EXPERTS} K={TOPK} T={args.seq_len} "
            f"tile_m={args.tile_m} tile_n_a={args.tile_n_a} tile_n_y={args.tile_n_y} "
            f"recompute_preact={args.recompute_preact}",
            flush=True,
        )

    # Warm: dispatch + kernel A + kernel Y + combine JIT, kernel cache, allocator.
    # Includes the bwd path so its kernels JIT during warmup too.
    for warm_seq in range(1, 6):
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
            activation_checkpoint=args.recompute_preact,
        )
        out.sum().backward()
    torch.cuda.synchronize()
    barrier(group)

    # [mem] reset peak counters after warmup so the reported peak reflects the
    # steady-state profiled iters, not one-time JIT/compile allocations. Note:
    # this tracks torch's caching allocator only (pool / activations / weights /
    # GEMM workspaces). The NVL + RDMA comm buffers are allocated outside torch
    # (cuMem / NVSHMEM symmetric heap), so the ring-depth knob does NOT show up
    # here — see the comm-buffer sizes printed below for that.
    torch.cuda.reset_peak_memory_stats()

    n_wait, n_warmup, n_active = 1, 2, 5
    n_steps = n_wait + n_warmup + n_active
    schedule = torch.profiler.schedule(
        wait=n_wait, warmup=n_warmup, active=n_active, repeat=1
    )
    # Wall-time CUDA events around fwd and bwd for every step. We use the
    # active-region medians as the e2e numbers. cuda_time_total from the
    # profiler is summed-across-streams (would double-count overlap), so
    # we rely on plain events for wall time. Per-kernel times still come
    # from the profiler aggregator.
    fwd_starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    fwd_ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    bwd_starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    bwd_ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_dir),
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        seq = 100  # bumped past warmup seqs so it's clearly distinct in the trace
        for step in range(n_steps):
            fwd_starts[step].record()
            with torch.profiler.record_function(f"step_{step}_fwd"):
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
                    activation_checkpoint=args.recompute_preact,
                )
            fwd_ends[step].record()
            bwd_starts[step].record()
            with torch.profiler.record_function(f"step_{step}_bwd"):
                out.sum().backward()
            bwd_ends[step].record()
            prof.step()

    torch.cuda.synchronize()
    barrier(group)

    # Active-region medians for the fwd / bwd wall-time numbers. Profiler
    # wait+warmup steps are excluded — those caches and seqs are noisy.
    active_range = range(n_wait + n_warmup, n_steps)
    fwd_times_us = sorted(
        fwd_starts[i].elapsed_time(fwd_ends[i]) * 1e3 for i in active_range
    )
    bwd_times_us = sorted(
        bwd_starts[i].elapsed_time(bwd_ends[i]) * 1e3 for i in active_range
    )
    fwd_e2e_us = fwd_times_us[len(fwd_times_us) // 2]
    bwd_e2e_us = bwd_times_us[len(bwd_times_us) // 2]

    if rank == 0:
        events = prof.key_averages()
        # Torch renamed `cuda_time_total` → `device_time_total` on
        # FunctionEventAvg (device-agnostic); old name still works as a
        # `table(sort_by=...)` key alias but no longer exists as an attribute.
        def _dev_time_total(ev) -> float:
            v = getattr(ev, "device_time_total", None)
            if v is None:
                v = getattr(ev, "cuda_time_total", 0.0)
            return float(v)

        full_table = events.table(
            sort_by="cuda_time_total",
            row_limit=40,
            max_name_column_width=80,
        )
        full_table_path = os.path.join(args.profile_dir, "full_kernel_table.txt")
        with open(full_table_path, "w") as f:
            f.write(full_table)

        # Logical roles we attribute each profiler event to. C++ kernels match
        # by their natural symbol substring; CuTeDSL-emitted kernels (kernel A,
        # Y, A_bwd, Y_bwd, dW grouped GEMM) get mangled to names like
        # `kernel_cutlass_kernel_stream_epstream_moekernel_aStreamingMoeA_
        # object_at__TiledMM...`, so we normalize via a case-insensitive,
        # underscore-stripped probe of CamelCase class fragments.
        #
        # Order matters: the more-specific patterns must be tried first so
        # `streaming_moe_a_bwd` claims its events before `streaming_moe_a`,
        # and the streaming-moe kernels claim their events before the `gemm`
        # catch-all (which then collects only the dW1 / dW2 quack.gemm calls).
        target_kernels = [
            # Collective ops — this script is their timing home.
            "streaming_dispatch_metadata_kernel",
            "dispatch_main_kernel",
            "dispatch_grads_main_kernel",
            "combine_main_kernel",
            # Compute kernels.
            "streaming_moe_a_bwd",
            "streaming_moe_y_bwd",
            "streaming_moe_a",
            "streaming_moe_y",
            # dW1 / dW2 grouped GEMMs (varlen-K quack.gemm calls in bwd).
            "gemm",
        ]

        def _classify(ev_key: str) -> str | None:
            # C++ kernels: direct substring match on the natural name.
            for key in (
                "streaming_dispatch_metadata_kernel",
                "dispatch_grads_main_kernel",
                "dispatch_main_kernel",
                "combine_main_kernel",
            ):
                if key in ev_key:
                    return key
            # CuTeDSL-emitted kernels: probe a normalized form so the mangled
            # `kernel_cutlass_kernel_..._StreamingMoeABwd_...` symbols
            # resolve back to their logical role. Case + underscore stripped
            # so future name shuffles in the JIT layer don't break the match.
            norm = ev_key.lower().replace("_", "")
            for probe, key in (
                ("streamingmoeabwd", "streaming_moe_a_bwd"),
                ("streamingmoeybwd", "streaming_moe_y_bwd"),
                ("streamingmoea", "streaming_moe_a"),
                ("streamingmoey", "streaming_moe_y"),
                ("quackgemm", "gemm"),
            ):
                if probe in norm:
                    return key
            return None

        # Aggregate cuda_time_total and count by target key (a single key may
        # match multiple event-list entries if torch splits by minor variation).
        total_us_by_key: dict[str, float] = {k: 0.0 for k in target_kernels}
        count_by_key: dict[str, int] = {k: 0 for k in target_kernels}
        sample_event_key: dict[str, str] = {}
        for ev in events:
            key = _classify(ev.key)
            if key is None:
                continue
            total_us_by_key[key] += _dev_time_total(ev)
            count_by_key[key] += ev.count
            sample_event_key.setdefault(key, ev.key)

        def per_call(key: str) -> float:
            n = count_by_key.get(key, 0)
            return total_us_by_key.get(key, 0.0) / n if n else 0.0

        # Per-call avg time for each role. combine_main_kernel is invoked
        # twice per training iter (once for fwd combine, once for bwd
        # combine_grads) but the per-call avg is the same, so we use it for
        # both serial sums.
        dispatch_meta_us = per_call("streaming_dispatch_metadata_kernel")
        dispatch_us = per_call("dispatch_main_kernel")
        dispatch_grads_us = per_call("dispatch_grads_main_kernel")
        combine_kernel_us = per_call("combine_main_kernel")
        streaming_a_us = per_call("streaming_moe_a")
        streaming_y_us = per_call("streaming_moe_y")
        streaming_a_bwd_us = per_call("streaming_moe_a_bwd")
        streaming_y_bwd_us = per_call("streaming_moe_y_bwd")
        # `gemm` matches both dW1 and dW2 invocations (and any other quack.gemm
        # calls in the bwd path). per_call gives the avg of a single dW gemm;
        # total dW serial contribution is 2× that (dW1 + dW2).
        gemm_per_call_us = per_call("gemm")
        gemm_count_per_iter = count_by_key.get("gemm", 0) / max(n_active, 1)

        # Serial-sum-vs-overlap analysis. By construction in `stream_moe_func`,
        # fwd and bwd don't overlap with each other (autograd boundary +
        # `out.sum().backward()`), so total e2e = fwd_e2e + bwd_e2e.
        # Within fwd, dispatch/A/Y/combine theoretically overlap across 4
        # streams. Within bwd, dispatch_grads/y_bwd/a_bwd/combine_grads/dW1/dW2
        # theoretically overlap across the bwd streams.
        # buffer.combine fires a single combine_main_kernel; gap-sentinel
        # encoding is folded into dispatch_main_kernel's tail.
        fwd_combine_us = combine_kernel_us
        bwd_combine_grads_us = combine_kernel_us

        fwd_serial_us = (
            dispatch_meta_us
            + dispatch_us
            + streaming_a_us
            + streaming_y_us
            + fwd_combine_us
        )
        bwd_serial_us = (
            dispatch_grads_us
            + streaming_y_bwd_us
            + streaming_a_bwd_us
            + bwd_combine_grads_us
            + 2 * gemm_per_call_us
        )
        total_serial_us = fwd_serial_us + bwd_serial_us
        total_e2e_us = fwd_e2e_us + bwd_e2e_us

        def pct_saved(serial: float, actual: float) -> str:
            if serial <= 0:
                return "n/a"
            saved = serial - actual
            return f"{saved:+7.1f} μs ({100 * saved / serial:+5.1f}% of serial)"

        print()
        print(
            "=== end-to-end pipeline analysis (from profiler key_averages + CUDA events) ==="
        )
        print(f"  buffer.dispatch (metadata + main):       {dispatch_meta_us + dispatch_us:7.1f} μs")
        print(f"    streaming_dispatch_metadata_kernel:    {dispatch_meta_us:7.1f} μs")
        print(f"    dispatch_main_kernel:                  {dispatch_us:7.1f} μs")
        print(f"  streaming_moe_a:                         {streaming_a_us:7.1f} μs")
        print(f"  streaming_moe_y:                         {streaming_y_us:7.1f} μs")
        print(f"  buffer.combine:                          {fwd_combine_us:7.1f} μs")
        print(f"    combine_main_kernel:                   {combine_kernel_us:7.1f} μs")
        print(f"  fwd serial sum of stages:                {fwd_serial_us:7.1f} μs")
        print(f"  fwd e2e (4 streams, real overlap):       {fwd_e2e_us:7.1f} μs")
        print(f"  fwd overlap saved:                       {pct_saved(fwd_serial_us, fwd_e2e_us)}")
        print()
        print(f"  buffer.dispatch_grads:                   {dispatch_grads_us:7.1f} μs")
        print(f"  streaming_moe_y_bwd:                     {streaming_y_bwd_us:7.1f} μs")
        print(f"  streaming_moe_a_bwd:                     {streaming_a_bwd_us:7.1f} μs")
        print(
            f"  buffer.combine_grads:                    {bwd_combine_grads_us:7.1f} μs"
        )
        print(
            f"  gemm_grouped dW1 + dW2 ({gemm_count_per_iter:.0f}× quack.gemm/iter): {2 * gemm_per_call_us:7.1f} μs"
        )
        print(f"  bwd serial sum of stages:                {bwd_serial_us:7.1f} μs")
        print(f"  bwd e2e (real overlap):                  {bwd_e2e_us:7.1f} μs")
        print(f"  bwd overlap saved:                       {pct_saved(bwd_serial_us, bwd_e2e_us)}")
        print()
        print(f"  total serial sum (fwd + bwd):            {total_serial_us:7.1f} μs")
        print(
            f"  total e2e per training iter (fwd + bwd): {total_e2e_us:7.1f} μs"
        )
        print(f"  total overlap saved:                     {pct_saved(total_serial_us, total_e2e_us)}")
        print(
            "  (fwd and bwd don't overlap with each other by construction — "
            "autograd boundary at out.sum().backward())"
        )
        print()
        print("=== memory (rank 0) ===")
        print(f"  peak allocated (torch caching allocator):  {torch.cuda.max_memory_allocated() / 2**30:7.2f} GiB")
        print(f"  peak reserved  (torch caching allocator):  {torch.cuda.max_memory_reserved() / 2**30:7.2f} GiB")
        # Comm buffers live outside torch (cuMem NVL + NVSHMEM RDMA symmetric
        # heap), so they don't appear above. Report their sizes explicitly —
        # the RDMA size scales with num_max_rdma_chunked_recv_tokens (the ring
        # depth), so this is the number that moves under a 128-vs-256 change.
        _hb = H * 2
        _dc = StreamEPBuffer.get_dispatch_config(world_size)
        _cc = StreamEPBuffer.get_combine_config(world_size)
        _rdma = max(_dc.get_rdma_buffer_size_hint(_hb, world_size),
                    _cc.get_rdma_buffer_size_hint(_hb, world_size))
        _nvl = max(_dc.get_nvl_buffer_size_hint(_hb, world_size),
                   _cc.get_nvl_buffer_size_hint(_hb, world_size))
        print(f"  comm buffers (outside torch): NVL {_nvl / 2**30:6.2f} GiB   "
              f"RDMA {_rdma / 2**30:6.2f} GiB  (RDMA scales with ring depth)")
        print()
        # Per-kernel table — useful for cross-referencing the symbols that
        # got matched to each role above.
        print("=== matched kernel symbols (per-call avg) ===")
        print(
            f"{'role':>40s}  {'symbol':>50s}  {'count':>5s}  {'avg (μs)':>10s}"
        )
        print(f"{'-' * 40}  {'-' * 50}  {'-' * 5}  {'-' * 10}")
        any_matched = False
        for key in target_kernels:
            if count_by_key[key] == 0:
                continue
            any_matched = True
            avg = per_call(key)
            print(
                f"{key:>40s}  {sample_event_key.get(key, '')[:50]:>50s}  "
                f"{count_by_key[key]:>5d}  {avg:>10.1f}"
            )
        if not any_matched:
            print(
                "  (no matching kernels found — check full_kernel_table.txt "
                "and update target_kernels in profile_pipeline.py if names changed)"
            )
        print()
        print(f"  Full table (top 40 by cuda_time_total): {full_table_path}")
        print(f"  Chrome traces: {args.profile_dir}/")
    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
