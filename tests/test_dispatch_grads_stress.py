"""Repetition stress repro for the internode dispatch_grads hang.

The single-shot ``test_dispatch_grads_internode`` passes, but the full
benchmark hangs in the BACKWARD internode ``dispatch_grads`` path at depth
(``StreamEP dispatch_grads NVL receiver timeout (prefix)`` / ``forwarder
timeout``), only after ~100+ back-to-back dispatch_grads calls
(N layers x several steps). `profile_pipeline.py`'s docstring notes the
collectives need "natural per-iter slack to satisfy the single-slot
``rdma_channel_meta`` protocol, which the full fwd+bwd pipeline provides but a
rapid-fire bench loop does not" — so the hypothesis is that many rapid
back-to-back dispatches overwhelm the single-slot meta protocol.

This test reproduces that **rapid-fire** regime with PURE COMM (no compute
kernels, so no JIT and the dispatch_grads calls are maximally back-to-back —
even less slack than the real pipeline, which only helps trigger the race).
Per "step" it mirrors the benchmark's per-training-step comm structure:

  * issue ``num_layers`` forward ``dispatch`` calls, KEEPING all handles alive
    (as autograd keeps every layer's saved state until backward);
  * then issue ``num_layers`` ``dispatch_grads`` calls in REVERSE order
    (as autograd runs layer backwards in reverse) — back-to-back, no slack.

Progress is printed per (step, layer) and flushed, so when it hangs the log
pinpoints the exact step/layer/op that stalled. A clean run prints PASS.

Driver (4 nodes / world=32 — internode is where the bug lives):
    ./scripts/srun_4node.sh StreamEP/tests/test_dispatch_grads_stress.py \
        [--num_layers 44] [--num_steps 8] [--hidden 3072] [--num_experts 256] \
        [--topk 8] [--num_tokens 8192] [--tile_m 128]

Defaults match the 82B shape / depth that hung (config=82ba5b, dp_shard=32).
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import socket
import subprocess
import sys
import threading
import time

import torch
import torch.distributed as dist

from stream_ep import Buffer

from utils import cleanup_dist


# ── stall diagnostic ─────────────────────────────────────────────────────────
# Per-rank watchdog. If the main thread makes no heartbeat progress for
# `--diag_timeout` seconds (fired BEFORE the ~100 s GPU watchdog trap), this
# rank dumps:
#   1. all-thread Python stacks (faulthandler; works even while the main thread
#      is in a C++ call — `internode_dispatch_grads` releases the GIL at
#      stream_ep.cpp:1709), and
#   2. per-thread kernel wait-channel + state from /proc (privilege-free), and
#   3. a best-effort native stack (py-spy --native / gdb; needs ptrace perms).
# The DISCRIMINATOR (see markdowns/internode_dispatch_hang.md §3): is the main
# thread parked INSIDE `buf.dispatch_grads(...)` -> the launch never happened
# because the host is stuck in torch::zeros/cudaMalloc (HOST launch-stall), or
# parked at `torch.cuda.synchronize()` / `dist.barrier()` with every
# dispatch_grads already returned (GPU-side wedge, all ranks launched)?
_hb = {"t": 0.0, "what": "init", "mem": "n/a"}
_hb_lock = threading.Lock()


def _beat(what, mem=None):
    """Record main-thread progress; the watchdog reports the last `what` on stall."""
    with _hb_lock:
        _hb["t"] = time.monotonic()
        _hb["what"] = what
        if mem is not None:
            _hb["mem"] = mem


def _proc_thread_states():
    """Privilege-free per-thread kernel wait-channel + run state from /proc.

    `wchan` names the kernel function the thread is blocked in — an nvidia
    `ioctl` / `os_*` symbol points at a CUDA-driver call (cudaMalloc/cuMemMap);
    `state` D = uninterruptible (in a syscall), S = sleeping, R = running."""
    out = []
    try:
        for tid in sorted(os.listdir("/proc/self/task")):
            d = f"/proc/self/task/{tid}"
            try:
                comm = open(f"{d}/comm").read().strip()
                wchan = open(f"{d}/wchan").read().strip() or "0"
                state = open(f"{d}/stat").read().rsplit(") ", 1)[1][0]
            except OSError:
                continue
            out.append(f"    tid {tid} [{comm}] state={state} wchan={wchan}")
    except OSError as e:
        out.append(f"    (/proc unavailable: {e})")
    return "\n".join(out)


def start_stall_watchdog(rank, timeout_s):
    if timeout_s <= 0:
        return
    faulthandler.enable()
    _beat("watchdog armed")
    host, pid = socket.gethostname(), os.getpid()
    main_tid = threading.main_thread().ident

    # yama ptrace_scope=1 blocks a child (gdb/py-spy we spawn) from tracing its
    # parent (us). PR_SET_PTRACER_ANY opts this process into being traceable by
    # anyone, so the best-effort native-stack dump below can attach. Harmless if
    # it fails (scope=0 doesn't need it; the Python stack + wchan don't either).
    try:
        import ctypes
        PR_SET_PTRACER, PR_SET_PTRACER_ANY = 0x59616D61, ctypes.c_ulong(-1).value
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[diag][rank {rank}] PR_SET_PTRACER failed: {e}\n")

    def _run():
        last_fire = -1e9
        n = 0
        while True:
            time.sleep(min(5.0, max(1.0, timeout_s / 2.0)))
            with _hb_lock:
                idle = time.monotonic() - _hb["t"]
                what, mem = _hb["what"], _hb["mem"]
            now = time.monotonic()
            if idle < timeout_s or (now - last_fire) < timeout_s:
                continue
            last_fire = now
            n += 1
            # Per-rank file: no cross-rank interleaving (the merged srun stream
            # garbles concurrent dumps), and flushed before any abort so even a
            # truncated gdb run lands. stderr gets only a one-line pointer.
            path = f"logs/diag_stall/rank{rank:02d}_pid{pid}.txt"
            try:
                os.makedirs("logs/diag_stall", exist_ok=True)
                f = open(path, "a")
            except OSError:
                f = sys.stderr
            sys.stderr.write(f"[diag][rank {rank}] STALL #{n} ({idle:.0f}s, '{what}') -> {path}\n")
            sys.stderr.flush()
            f.write(f"\n========== [rank {rank}] STALL #{n} ({idle:.0f}s no progress) ==========\n")
            f.write(f"host={host} pid={pid} main_tid={main_tid}\n")
            f.write(f"last activity : {what}\n")
            f.write(f"last mem (cached at step start) : {mem}\n")
            f.write(f"per-thread kernel state:\n{_proc_thread_states()}\n")
            f.write("python stacks (all threads):\n")
            f.flush()
            faulthandler.dump_traceback(all_threads=True, file=f)
            f.flush()
            # MAIN-thread native stack only (thread 1 = initial LWP). Main-only
            # `bt` skips the all-thread symbol load, so it completes well before
            # the ~100 s GPU-watchdog abort kills us. py-spy first if present.
            for tool in (["py-spy", "dump", "--native", "--nonblocking", "--pid", str(pid)],
                         ["gdb", "-p", str(pid), "-batch", "-nx",
                          "-ex", "set pagination off", "-ex", "thread 1", "-ex", "bt"]):
                try:
                    r = subprocess.run(tool, capture_output=True, text=True, timeout=40)
                    f.write(f"\n[{tool[0]}] rc={r.returncode} native stack:\n{r.stdout}\n")
                    if r.returncode != 0 and r.stderr.strip():
                        f.write(f"[{tool[0]}] stderr: {r.stderr.strip()[:500]}\n")
                    if r.returncode == 0 and r.stdout.strip():
                        break
                except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                    f.write(f"[{tool[0]}] unavailable: {e}\n")
            f.write(f"========== [rank {rank}] END STALL #{n} ==========\n")
            f.flush()
            if f is not sys.stderr:
                f.close()

    threading.Thread(target=_run, name="stall-watchdog", daemon=True).start()


def random_routing(n_tokens, topk, num_experts, num_local_experts, world_size, device, gen):
    """K random distinct experts per token, RE-RANDOMIZED each call (gen advances).

    Matches benchmark.py's force_uniform_routing (`logits*0 + randn*1e-3`, then
    topk) — the routing varies every forward, so num_recv / total_tiles /
    per-(channel,src,expert) counts vary every dispatch. That per-dispatch size
    variation is the suspected trigger the earlier FIXED-routing repro missed.
    Returns (topk_idx, topk_weights, is_token_in_rank)."""
    logits = torch.randn(n_tokens, num_experts, device=device, generator=gen)
    topk_idx = torch.topk(logits, topk, dim=-1).indices.to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(n_tokens, topk, dtype=torch.float32, device=device, generator=gen),
        dim=-1).contiguous()
    rank_idx = topk_idx // num_local_experts
    is_token_in_rank = torch.zeros((n_tokens, world_size), dtype=torch.bool, device=device)
    for r in range(world_size):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)
    return topk_idx.contiguous(), topk_weights, is_token_in_rank


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_layers", type=int, default=44,
                   help="forward dispatches kept per step before the dispatch_grads burst")
    p.add_argument("--num_steps", type=int, default=8)
    p.add_argument("--hidden", type=int, default=3072)       # 82ba5b d_model
    p.add_argument("--num_experts", type=int, default=256)   # 82ba5b
    p.add_argument("--topk", type=int, default=8)            # 82ba5b
    p.add_argument("--num_tokens", type=int, default=8192)   # per-rank T
    p.add_argument("--tile_m", type=int, default=128)
    p.add_argument("--num_sms", type=int, default=None)
    p.add_argument("--slack", action="store_true",
                   help="Insert a cross-rank barrier after each dispatch_grads. "
                        "Tests the single-slot rdma_channel_meta race: if --slack "
                        "makes the hang vanish, the bug is iter N+1's remote meta "
                        "put overwriting the slot before iter N's forwarder reads it.")
    p.add_argument("--leave_free_gb", type=float, default=None,
                   help="Memory-stress: allocate ballast so only ~this many GB "
                        "remain free, pushing the caching allocator near capacity. "
                        "The dropless per-dispatch pool (size varies every iter via "
                        "random_routing) then forces cudaMalloc/cudaFree mid-burst "
                        "-- the suspected stall that lets a peer lap the "
                        "kRdmaMetaRingDepth ring and clobber a meta slab.")
    p.add_argument("--frag", action="store_true",
                   help="With --leave_free_gb: also fragment the remaining headroom "
                        "(alloc varied blocks, free alternating -> cached holes, NO "
                        "empty_cache) so variable-size pool allocs can't reuse a hole.")
    p.add_argument("--lag_ms", type=float, default=0.0,
                   help="NVL-hop repro knob: enqueue a GPU-side spin kernel "
                        "(torch.cuda._sleep, no host sync) of this many ms before "
                        "each dispatch_grads on the ranks selected by "
                        "--lag_nvl_ranks. Emulates the benchmark's compute-gated "
                        "receivers: the lagging rank's NVL receiver falls a full "
                        "generation behind its same-node forwarders, exposing the "
                        "NVL rings' per-gen state reuse (zero-seeded room check, "
                        "slot restart, seq-tagged single-register head/tail). "
                        "0 = off.")
    p.add_argument("--lag_nvl_ranks", type=str, default="4",
                   help="Comma-separated nvl-rank indices (rank %% 8) that lag "
                        "when --lag_ms > 0. One rank per node lags symmetrically.")
    p.add_argument("--lag_asym", action="store_true",
                   help="Apply --lag_ms/--lag_once_s/--side_load_mm on NODE-0 "
                        "ranks only (rank < 8) instead of one rank per node "
                        "symmetrically. A symmetric lag cannot park anyone at "
                        "the RDMA meta-wait: the only ranks that consume each "
                        "other's RDMA metas are the same-nvl-position lane "
                        "peers — the lagged set itself — so they sleep and "
                        "wake together and the park lands on same-node NVL "
                        "prefix waits instead (hang doc §8.13 run 2's 'zero "
                        "meta probes'). An ASYMMETRIC lag makes the un-lagged "
                        "lane peer spin at the RDMA meta-wait for the full "
                        "lag BEFORE the put lands — the §8.24 reader-side "
                        "visibility-wedge precondition (poll-installed stale "
                        "L2 line vs inbound RDMA write).")
    p.add_argument("--lag_fwd", action="store_true",
                   help="With --lag_ms: also lag the forward dispatch calls "
                        "(default: backward dispatch_grads burst only).")
    p.add_argument("--lag_once_s", type=float, default=0.0,
                   help="Launch-partition repro (the bench wedge shape, doc "
                        "§8.12): ONE long GPU-side spin (this many seconds, keep "
                        "below the ~100 s GPU watchdog) before the mid-burst "
                        "dispatch_grads (bwd layer num_layers//2) of every step, "
                        "on the --lag_nvl_ranks ranks. The un-lagged ranks run a "
                        "full generation ahead and park (expect 'stuck nvl-room' "
                        "used~ring, head 0 at >10 s); when the laggard arrives "
                        "the run either recovers (PASS => parking is benign, the "
                        "wedge needs the pipeline's gating cycle) or hangs "
                        "(=> in-kernel cross-gen NVL bug, repro'd). 0 = off.")
    p.add_argument("--side_load_mm", type=int, default=0,
                   help="Mid-kernel SM-contention repro: enqueue this many "
                        "8192^2 bf16 matmuls on a SIDE stream of the "
                        "--lag_nvl_ranks ranks before each dispatch_grads, "
                        "concurrent with the comm kernel (no sync). Emulates the "
                        "bench's six-stream compute competing for SM issue slots "
                        "and HBM bandwidth, slowing the victim's NVL receiver "
                        "warps MID-KERNEL (a state inter-kernel lag cannot "
                        "create). 0 = off.")
    p.add_argument("--side_load_once_mm", type=int, default=0,
                   help="With --lag_once_s: enqueue this many 8192^2 bf16 "
                        "matmuls on a side stream at the lag_once layer "
                        "(~2.7 ms each — size to span the whole lag), so the "
                        "parked comm kernels run under sustained L2/HBM/SM "
                        "load for the full park window. 0 = off.")
    p.add_argument("--side_load_all", action="store_true",
                   help="Apply --side_load_mm/--side_load_once_mm on ALL "
                        "ranks (bench-like: every GPU computes), not just "
                        "the --lag_nvl_ranks/--lag_asym lag set. The rank "
                        "that matters for the §8.24 visibility question is "
                        "the PARKED READER (the un-lagged lane peer), which "
                        "the lag-set gating excludes by construction.")
    p.add_argument("--side_ar_once_mb", type=int, default=0,
                   help="With --lag_once_s: enqueue a burst of NCCL "
                        "all-reduces (this many MB each, x --side_ar_once_n) "
                        "on a side stream at the lag_once layer on ALL ranks "
                        "(collective), sized to span the lag — FSDP-like "
                        "NIC+PCIe ingress traffic concurrent with the parked "
                        "comm kernels and the inbound meta put. 0 = off.")
    p.add_argument("--side_ar_once_n", type=int, default=300,
                   help="Number of all-reduces in the --side_ar_once_mb "
                        "burst (each ~MB/bw; size n*per-AR-time ~ lag).")
    p.add_argument("--diag_timeout", type=float, default=0.0,
                   help="Per-rank stall watchdog: if the main thread makes no "
                        "progress for this many seconds, dump that rank's thread "
                        "stacks + /proc wait-channels + best-effort native stack. "
                        "Set below the ~100 s GPU watchdog (e.g. 45). 0 = off.")
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    start_stall_watchdog(rank, args.diag_timeout)

    assert world_size % 8 == 0 and world_size > 8, (
        f"internode repro needs world_size > 8 and % 8 == 0; got {world_size}")
    assert args.num_experts % world_size == 0, (
        f"num_experts {args.num_experts} must divide world_size {world_size}")

    if args.num_sms is not None:
        Buffer.set_num_sms(args.num_sms)

    H, E, K, T = args.hidden, args.num_experts, args.topk, args.num_tokens
    num_local_experts = E // world_size

    hidden_bytes = H * 2
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (Buffer.get_dispatch_config(world_size),
                Buffer.get_combine_config(world_size)):
        nvl_bytes = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, world_size), nvl_bytes)
        rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes)
    buf = Buffer(group, nvl_bytes, rdma_bytes)

    # NVL-receiver lag (see --lag_ms). ~1.8e6 cycles/ms on H100; precision is
    # irrelevant — only cumulative skew vs the un-lagged ranks matters.
    lag_nvl = {int(s) for s in args.lag_nvl_ranks.split(",") if s}
    lag_cycles = int(args.lag_ms * 1.8e6)
    on_lag_node = (rank < 8) if args.lag_asym else True
    do_lag = args.lag_ms > 0 and (rank % 8) in lag_nvl and on_lag_node

    do_lag_once = args.lag_once_s > 0 and (rank % 8) in lag_nvl and on_lag_node
    lag_once_layer = args.num_layers // 2

    def lag():
        if do_lag:
            torch.cuda._sleep(lag_cycles)

    def lag_once(bwd_layer):
        if do_lag_once and bwd_layer == lag_once_layer:
            torch.cuda._sleep(int(args.lag_once_s * 1.8e9))

    side_sel = (rank % 8) in lag_nvl and on_lag_node
    if args.side_load_all:
        side_sel = True
    do_side = args.side_load_mm > 0 and side_sel
    do_side_once = args.side_load_once_mm > 0 and side_sel
    if do_side or do_side_once:
        side_stream = torch.cuda.Stream()
        side_a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
        side_b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
        side_c = torch.empty(8192, 8192, dtype=torch.bfloat16, device=device)
        with torch.cuda.stream(side_stream):
            torch.mm(side_a, side_b, out=side_c)  # warmup: cublas workspace alloc
        torch.cuda.synchronize()

    def side_load():
        if do_side:
            with torch.cuda.stream(side_stream):
                for _ in range(args.side_load_mm):
                    torch.mm(side_a, side_b, out=side_c)

    def side_load_once(bwd_layer):
        # Bench-emulation companion to lag_once: a ~lag-length burst of L2/HBM/
        # SM traffic on the side stream at the SAME bwd layer, so the parked
        # comm kernel (incl. the meta-wait spin of the un-lagged lane peer
        # under --lag_asym) coexists with compute for the whole park window —
        # the bench's two-stream + FSDP condition the per-layer side_load
        # (~ms burst, lag-set ranks only) cannot create. ~2.7 ms per 8192^3
        # bf16 matmul on H100: size --side_load_once_mm to cover --lag_once_s.
        if do_side_once and bwd_layer == lag_once_layer:
            with torch.cuda.stream(side_stream):
                for _ in range(args.side_load_once_mm):
                    torch.mm(side_a, side_b, out=side_c)

    # NCCL-AR burst at the lag layer (see --side_ar_once_mb). Collective —
    # every rank participates regardless of lag gating; the lagged rank's
    # main-stream sleep does not block its side stream.
    do_ar_once = args.side_ar_once_mb > 0 and args.lag_once_s > 0
    if do_ar_once:
        ar_stream = torch.cuda.Stream()
        ar_buf = torch.empty(args.side_ar_once_mb * (1 << 20) // 2,
                             dtype=torch.bfloat16, device=device)
        with torch.cuda.stream(ar_stream):
            dist.all_reduce(ar_buf, async_op=True)  # warmup: comm bootstrap
        torch.cuda.synchronize()

    def side_ar_once(bwd_layer):
        if do_ar_once and bwd_layer == lag_once_layer:
            with torch.cuda.stream(ar_stream):
                for _ in range(args.side_ar_once_n):
                    dist.all_reduce(ar_buf, async_op=True)

    # x / dL_dy fixed (values irrelevant — we stress comm). Routing is
    # RE-RANDOMIZED per dispatch (see random_routing) to match the benchmark.
    torch.manual_seed(100 + rank)
    x = (torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    dL_dy = (torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    route_gen = torch.Generator(device=device).manual_seed(7000 + rank)

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f"[repro] world={world_size} H={H} E={E} K={K} T={T} "
        f"num_layers={args.num_layers} num_steps={args.num_steps} "
        f"local_experts={num_local_experts}")
    if args.lag_ms > 0:
        log(f"[repro] lag: {args.lag_ms} ms x nvl_ranks {sorted(lag_nvl)} "
            f"({lag_cycles} cycles), fwd={args.lag_fwd}")

    GiB = 1 << 30

    def mem_str():
        free, total = torch.cuda.mem_get_info()
        return (f"alloc={torch.cuda.memory_allocated() / GiB:.2f} "
                f"peak_alloc={torch.cuda.max_memory_allocated() / GiB:.2f} "
                f"reserved={torch.cuda.memory_reserved() / GiB:.2f} "
                f"peak_reserved={torch.cuda.max_memory_reserved() / GiB:.2f} "
                f"gap={(torch.cuda.max_memory_reserved() - torch.cuda.max_memory_allocated()) / GiB:.2f} "
                f"free={free / GiB:.1f} of {total / GiB:.1f} GB")

    # --- optional memory stress (fragmentation hypothesis) ---
    # Every rank ballasts independently so the pressure is symmetric. Held alive
    # for the whole run.
    ballast = []
    if args.leave_free_gb is not None:
        torch.cuda.synchronize()
        free0, total = torch.cuda.mem_get_info()
        consume = max(0, free0 - int(args.leave_free_gb * GiB))
        if consume > 0:
            ballast.append(torch.empty(consume, dtype=torch.uint8, device=device))
        if args.frag:
            # Carve the headroom into varied blocks, drop alternating ones so the
            # caching allocator holds fragmented free holes (NOT returned to the
            # driver -- no empty_cache). 32..256 MB blocks.
            blocks = []
            try:
                for i in range(256):
                    blocks.append(torch.empty(((i % 8) + 1) * (32 << 20),
                                              dtype=torch.uint8, device=device))
            except RuntimeError:
                pass
            for j in range(0, len(blocks), 2):
                blocks[j] = None
        torch.cuda.synchronize()
        log(f"[repro] mem-stress (leave_free_gb={args.leave_free_gb} frag={args.frag}): {mem_str()}")

    log(f"[repro] initial mem: {mem_str()}")

    for step in range(args.num_steps):
        _beat(f"step {step} forward start", mem=mem_str())
        # --- forward: num_layers dispatches, keep all handles alive ---
        handles = []
        for layer in range(args.num_layers):
            topk_idx, topk_weights, is_token_in_rank = random_routing(
                T, K, E, num_local_experts, world_size, device, route_gen)
            log(f"[repro] step {step} fwd-dispatch layer {layer}")
            _beat(f"PRE fwd-dispatch step {step} layer {layer}")
            if args.lag_fwd:
                lag()
            _pool, handle, _ev = buf.dispatch(
                x, topk_idx, topk_weights, is_token_in_rank, E, tile_m=args.tile_m)
            _beat(f"POST fwd-dispatch step {step} layer {layer}")
            handles.append(handle)
        # --- backward: dispatch_grads in reverse, back-to-back (no slack) ---
        for layer, handle in enumerate(reversed(handles)):
            L = args.num_layers - 1 - layer
            log(f"[repro] step {step} bwd-dispatch_grads layer {L} (seq={handle.dispatch_seq})")
            _beat(f"PRE bwd-dispatch_grads step {step} layer {L} seq={handle.dispatch_seq}")
            lag()
            side_load_once(L)  # enqueue BEFORE the lag sleep: side stream runs through the park
            side_ar_once(L)
            lag_once(L)
            side_load()
            _dl_do_pool, _cnt, _ev = buf.dispatch_grads(
                handle, dL_dy, dispatch_seq=handle.dispatch_seq)
            _beat(f"POST bwd-dispatch_grads step {step} layer {L} seq={handle.dispatch_seq}")
            if args.slack:
                torch.cuda.synchronize()
                dist.barrier(device_ids=[torch.cuda.current_device()])
        _beat(f"PRE end-of-step sync+barrier step {step}")
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])
        _beat(f"POST step {step} COMPLETE")
        log(f"[repro] step {step} COMPLETE | mem {mem_str()}")

    if rank == 0:
        print(f"PASS: {args.num_steps} steps x {args.num_layers} layers of "
              f"dispatch+dispatch_grads completed without hang "
              f"(world={world_size}, H={H}, E={E}, K={K}, T={T})", flush=True)

    cleanup_dist()


if __name__ == "__main__":
    main()
