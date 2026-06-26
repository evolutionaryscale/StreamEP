"""Single-GPU ncu microbench for the four streaming-MoE CuTeDSL kernels.

Why this exists
---------------
`profile_pipeline.py` profiles the *whole* fwd+bwd pipeline under torch's
profiler, on N nodes, with the real dispatch/combine comm. That is the right
tool for the overlap story, but it is the *wrong* target for Nsight Compute
(ncu): ncu's default per-kernel replay relaunches each kernel in isolation
many times, which (a) deadlocks on the NVSHMEM dispatch/combine comm kernels
(cross-rank dependency) and (b) deadlocks even on kernels A/Y in-situ, because
those spin on `pool_arrival_count[tile] == pool_arrival_target[tile]` signalled
by a *concurrent* dispatch kernel on another stream that ncu pauses during
replay.

This harness sidesteps both problems. It drives each compute kernel
standalone, single-GPU, with NO comm and the per-tile arrival gate
**pre-fired** (`pool_arrival_count == pool_arrival_target`), exactly like the
per-kernel unit tests (`tests/test_kernel_a.py` etc.). With the gate already
satisfied the kernel never spins, so default kernel replay works perfectly and
you get full SM / tensor-core / memory / occupancy / roofline counters per
kernel.

Shape defaults to the **real 4-node 82ba5b** point so the GEMM tiles match
production: world=32 -> E_local = 256/32 = 8 experts/rank; H=3072, I=768;
uniform routing gives ~T*topk*world/E = 8192 tokens/expert -> ~64 tiles/expert
at tile_m=128 -> total_tiles=512 (M_recv=65536). Override --world / --tiles /
--hidden / --intermediate to profile a different operating point. tile_n per
kernel is pulled from the production `_resolve_tile_config(I, H)` so you
measure the bench-tuned tiles unless you pass --tile_n_*.

Launch (under ncu; see scripts/ncu_microbench.sh which wraps this)
------------------------------------------------------------------
    ncu --profile-from-start off --launch-count 1 \
        --kernel-name 'regex:StreamingMoe' --set full \
        --clock-control none -f -o report \
        python -m stream_ep.stream_moe.ncu_microbench --kernel a

Or run bare (no ncu) to sanity-check shapes / timing:
    python -m stream_ep.stream_moe.ncu_microbench --kernel a --bare

The single profiled launch is wrapped in `cudaProfilerStart/Stop`
(torch.cuda.profiler.start/stop), so with `--profile-from-start off` ncu
captures exactly that one kernel and skips warmup + the internal zero-init
memsets.

NOTE on the scatter epilogue: kernels Y / A_bwd / Y_bwd atomic-scatter into a
recv-token destination. This harness uses an identity slot->token map
(no top-k fan-in), so the *GEMM* is exactly production-shaped but the scatter
has no atomic contention. The GEMM dominates; treat the epilogue numbers as a
lower bound on scatter cost.
"""

from __future__ import annotations

import argparse

import torch

from stream_ep.stream_moe.stream_moe import TileConfig, _resolve_tile_config

# Module-level default shape = 82ba5b prod MoE (per-expert), same as
# profile_pipeline's globals.
H_DEFAULT = 3072
I_DEFAULT = 768
NUM_EXPERTS = 256
TOPK = 8
SEQ_LEN_PER_RANK = 8192
DTYPE = torch.bfloat16


def _expert_pool_block_offset(total_tiles: int, E_local: int, device) -> torch.Tensor:
    """Evenly split total_tiles across E_local experts, expert-major. Returns
    the (E_local+1,) int32 prefix-sum the kernels index by tile position."""
    base = total_tiles // E_local
    rem = total_tiles % E_local
    counts = [base + (1 if e < rem else 0) for e in range(E_local)]
    off = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    cum = 0
    for e in range(E_local):
        off[e] = cum
        cum += counts[e]
    off[E_local] = cum
    assert cum == total_tiles
    return off


def _fired_arrival(total_tiles: int, device, target: int = 1):
    """(arrival_count, arrival_target) int32 pair with count == target so the
    scheduler spin passes immediately (gate-free standalone launch)."""
    tgt = torch.full((total_tiles,), target, dtype=torch.int32, device=device)
    return tgt.clone(), tgt


def _shape(args) -> dict:
    H = args.hidden
    I = args.intermediate
    E_local = max(1, NUM_EXPERTS // args.world)
    cfg = _resolve_tile_config(
        TileConfig(
            tile_m=args.tile_m,
            tile_n_a=args.tile_n_a,
            tile_n_y=args.tile_n_y,
            tile_n_a_bwd=args.tile_n_a_bwd,
            tile_n_y_bwd=args.tile_n_y_bwd,
        ),
        I,
        H,
    )
    tile_m = cfg.tile_m or 128
    if args.tiles is not None:
        total_tiles = args.tiles
    else:
        # uniform routing: tokens/expert = T * topk * world / E ; tiles =
        # ceil(tokens/expert / tile_m) * E_local.
        tokens_per_expert = SEQ_LEN_PER_RANK * TOPK * args.world // NUM_EXPERTS
        tiles_per_expert = max(1, -(-tokens_per_expert // tile_m))
        total_tiles = tiles_per_expert * E_local
    return dict(H=H, I=I, E_local=E_local, tile_m=tile_m, total_tiles=total_tiles, cfg=cfg)


def build(kernel: str, s: dict, device):
    """Build the standalone call closure for one kernel. Returns (fn, label)
    where fn() launches the kernel once on the current stream."""
    H, I, E_local = s["H"], s["I"], s["E_local"]
    tile_m, total_tiles, cfg = s["tile_m"], s["total_tiles"], s["cfg"]
    two_I = 2 * I
    TK = total_tiles * tile_m
    g = torch.Generator(device=device).manual_seed(0)

    epbo = _expert_pool_block_offset(total_tiles, E_local, device)
    W1 = (torch.randn(E_local, two_I, H, dtype=DTYPE, device=device, generator=g) * 0.02).contiguous()
    W2 = (torch.randn(E_local, H, I, dtype=DTYPE, device=device, generator=g) * 0.02).contiguous()

    if kernel == "a":
        from stream_ep.stream_moe.kernel_a import streaming_moe_a

        pool = (torch.randn(TK, H, dtype=DTYPE, device=device, generator=g) * 0.1).contiguous()
        postact_a = torch.empty(total_tiles, tile_m, I, dtype=DTYPE, device=device)
        preact_a = torch.empty(total_tiles, tile_m, two_I, dtype=DTYPE, device=device)
        cnt, tgt = _fired_arrival(total_tiles, device)

        def fn():
            streaming_moe_a(
                pool, W1, postact_a, epbo, cnt, tgt,
                preact_a=preact_a, tile_m=tile_m, tile_n=cfg.tile_n_a,
            )

        return fn, f"kernel_a  GEMM (TK={TK},H={H})@W1(2I={two_I}) tile_m={tile_m} tile_n_a={cfg.tile_n_a}"

    if kernel == "y":
        from stream_ep.stream_moe.kernel_y import streaming_moe_y

        T_recv = TK
        postact_a = (torch.randn(total_tiles, tile_m, I, dtype=DTYPE, device=device, generator=g) * 0.1).contiguous()
        o = torch.zeros(T_recv, H, dtype=DTYPE, device=device)
        pool_recv_token = torch.arange(TK, dtype=torch.int32, device=device)
        pool_topk_weight = torch.rand(TK, dtype=torch.float32, device=device)
        k_local_remaining = torch.ones(T_recv, dtype=torch.int32, device=device)
        y_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)
        cnt, tgt = _fired_arrival(total_tiles, device)

        def fn():
            streaming_moe_y(
                postact_a, W2, o, pool_recv_token, pool_topk_weight,
                k_local_remaining, y_done_per_token, epbo, cnt, tgt,
                combine_seq=1, tile_m=tile_m, tile_n=cfg.tile_n_y,
            )

        return fn, f"kernel_y  GEMM (TK={TK},I={I})@W2(H={H}) tile_m={tile_m} tile_n_y={cfg.tile_n_y}"

    if kernel == "a_bwd":
        from stream_ep.stream_moe.kernel_a_bwd import streaming_moe_a_bwd

        T_recv = TK
        dL_dswiglu_in = (torch.randn(total_tiles, tile_m, two_I, dtype=DTYPE, device=device, generator=g) * 0.1).contiguous()
        dL_dx_per_r = torch.zeros(T_recv, H, dtype=DTYPE, device=device)
        pool_recv_token = torch.arange(TK, dtype=torch.int32, device=device)
        bwd_k_local_remaining = torch.ones(T_recv, dtype=torch.int32, device=device)
        bwd_a_done_per_token = torch.zeros(T_recv, dtype=torch.int64, device=device)
        cnt, tgt = _fired_arrival(total_tiles, device)

        def fn():
            streaming_moe_a_bwd(
                dL_dswiglu_in, W1, dL_dx_per_r, pool_recv_token,
                bwd_k_local_remaining, bwd_a_done_per_token, epbo, cnt, tgt,
                dispatch_seq=1, tile_m=tile_m, tile_n=cfg.tile_n_a_bwd,
            )

        return fn, f"kernel_a_bwd  NN GEMM (TK={TK},2I={two_I})@W1(H={H}) tile_m={tile_m} tile_n_a_bwd={cfg.tile_n_a_bwd}"

    if kernel == "y_bwd":
        from stream_ep.stream_moe.kernel_y_bwd import streaming_moe_y_bwd

        dL_do_pool = (torch.randn(TK, H, dtype=DTYPE, device=device, generator=g) * 0.1).contiguous()
        dL_dswiglu_in = torch.empty(total_tiles, tile_m, two_I, dtype=DTYPE, device=device)
        postact_a_for_dW2 = torch.empty(total_tiles, tile_m, I, dtype=DTYPE, device=device)
        pool_topk_weight = torch.rand(TK, dtype=torch.float32, device=device)
        pool_recv_token = torch.arange(TK, dtype=torch.int32, device=device)
        preact_a = (torch.randn(total_tiles, tile_m, two_I, dtype=DTYPE, device=device, generator=g) * 0.1).contiguous()
        dL_dweight = torch.zeros(TK, dtype=torch.float32, device=device)
        cnt, tgt = _fired_arrival(total_tiles, device)

        def fn():
            streaming_moe_y_bwd(
                dL_do_pool, W2, dL_dswiglu_in, postact_a_for_dW2,
                pool_topk_weight, pool_recv_token, preact_a, dL_dweight,
                epbo, cnt, tgt, tile_m=tile_m, tile_n=cfg.tile_n_y_bwd,
            )

        return fn, f"kernel_y_bwd  NN GEMM (TK={TK},H={H})@W2(I={I}) tile_m={tile_m} tile_n_y_bwd={cfg.tile_n_y_bwd}"

    raise ValueError(f"unknown kernel {kernel!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernel", required=True, choices=["a", "y", "a_bwd", "y_bwd"])
    p.add_argument("--world", type=int, default=32, help="EP world size the shape mimics (sets E_local=256/world; default 32 = 4 nodes)")
    p.add_argument("--tiles", type=int, default=None, help="total padded tiles; default = uniform-routing estimate for --world")
    p.add_argument("--hidden", type=int, default=H_DEFAULT)
    p.add_argument("--intermediate", type=int, default=I_DEFAULT)
    p.add_argument("--tile_m", type=int, default=None)
    p.add_argument("--tile_n_a", type=int, default=None)
    p.add_argument("--tile_n_y", type=int, default=None)
    p.add_argument("--tile_n_a_bwd", type=int, default=None)
    p.add_argument("--tile_n_y_bwd", type=int, default=None)
    p.add_argument("--warmup", type=int, default=5, help="warmup launches before the profiled one (JIT + cache)")
    p.add_argument("--bare", action="store_true", help="no profiler region; just warm + time the kernel (sanity check)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    s = _shape(args)
    fn, label = build(args.kernel, s, device)
    print(f"[ncu_microbench] {label}", flush=True)
    print(
        f"[ncu_microbench] world={args.world} E_local={s['E_local']} "
        f"total_tiles={s['total_tiles']} M_recv={s['total_tiles'] * s['tile_m']}",
        flush=True,
    )

    # Warmup: JIT compile + kernel cache + allocator. The profiled launch then
    # hits the steady-state compiled kernel.
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    if args.bare:
        n = 20
        start = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            start[i].record()
            fn()
            end[i].record()
        torch.cuda.synchronize()
        times = sorted(start[i].elapsed_time(end[i]) * 1e3 for i in range(n))
        print(f"[ncu_microbench] median {times[n // 2]:.1f} us  (min {times[0]:.1f}, max {times[-1]:.1f})", flush=True)
        return

    # One profiled launch, fenced by cudaProfilerStart/Stop so ncu
    # (--profile-from-start off) captures exactly this kernel.
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    fn()
    torch.cuda.profiler.stop()
    torch.cuda.synchronize()
    print("[ncu_microbench] profiled launch done", flush=True)


if __name__ == "__main__":
    main()
