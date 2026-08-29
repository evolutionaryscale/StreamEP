"""Regression test: internode NVL metadata <-> dispatch-data ring aliasing.

Bug (root-caused 2026-08-29, prod failures 2026-08-27): the internode
`streaming_dispatch_metadata` kernels carve their NVL slabs (count slabs +
streaming inbox) from `buffer_ptrs[*]` at byte offset 0 -- the same bytes as
the dispatch data kernel's `nvl_channel_x` (channel 0, writer nvl_rank 0)
ring slice. The metadata Phase B kernel reads the streaming inbox AFTER the
last cross-rank barrier (Phase A's trailing `barrier_block`), in a separate,
purely-local cooperative kernel. A leading NVL peer -- whose own host poll
finished -- launches its SAME-iteration `dispatch_main_kernel`, whose
channel-0 forwarder writes ring slots 0..~60 into the victim's NVL buffer
(the NVL ring restarts slot indexing at 0 every generation), overwriting the
victim's streaming inbox with bf16 token payload while the victim's Phase B
is still pending. The payload bytes, read back as int32 substream counts,
produce garbage `expert_frequency` -> garbage `streaming_total_tiles`
(negative -> internode_dispatch CPU timeout; huge positive -> multi-TiB
`pool` alloc / OOM). Only the victim's Phase-B-derived outputs corrupt; the
Phase-A count exchange (recv_tokens / rdma_recv_tokens / per-expert
counters) stays sane -- the exact production signature.

In production the reader wins the race by only the ~O(100us) poll+launch
pipeline latency, so the bug fires rarely (victim's cooperative Phase B
delayed by SM contention from the overlapped GEMM/dense streams). This test
makes the race deterministic through the permanent, default-off
`STREAMEP_DEBUG_PB_DELAY_US` hook: a device-side spin at Phase B entry on
the nvl_rank==7 rank of each node, emulating a legal scheduling delay. Under
the delay, UNFIXED code corrupts within the first few dispatches, every run.

Detection is layered:
  * the host poll's always-on fail-fast raises
    `RuntimeError: StreamEP-CORRUPT(host)` on an insane `total_tiles`
    (negative or > 2^20);
  * a garbage-positive `total_tiles` big enough to OOM the `pool` alloc
    raises `torch.OutOfMemoryError` before any kernel touches it;
  * this test additionally bounds the returned pool's tile count by the
    routing-independent maximum (world*T*K/tile_m + E_local).

PASS/FAIL semantics: the test PASSES iff NO corruption occurs on any rank.
The fix is always-on (unconditional): the metadata NVL slabs are carved at
nvl_metadata_offset = dispatch + combine NVL region bytes, disjoint from
both data rings. This test passes on correct code and FAILS if that fix
ever regresses. There is deliberately NO runtime way to select the buggy
offset-0 layout -- reproducing red requires building the pre-fix source
(metadata NVL carve at buffer offset 0), not flipping a flag.

Launch (torchrun-style, internode: world > 8 and world % 8 == 0; prod-ish
shapes H=512 latent bf16, E=512, K=8, T=8192, num_sms=64 -> 32 channels):

    ./scripts/srun_4node.sh StreamEP/tests/test_metadata_nvl_aliasing.py

This test intentionally relies on two permanent, zero-overhead-when-off
pieces of the C++ side: the `STREAMEP_DEBUG_PB_DELAY_US` Phase-B entry spin
and the host-poll `total_tiles` sanity fail-fast. Do not remove them.
"""

from __future__ import annotations

import os

# The C++ launcher caches STREAMEP_DEBUG_PB_DELAY_US on its first call; set it
# before anything can trigger a dispatch so the test is self-contained and does
# NOT depend on the caller. 20 ms comfortably exceeds the natural reader margin
# (~O(100us)) while staying far below every device/host watchdog. The unset
# _RANK selects the default victim set: nvl_rank == 7 (one rank per node).
os.environ["STREAMEP_DEBUG_PB_DELAY_US"] = os.environ.get(
    "STREAMEP_TEST_ALIASING_DELAY_US", "20000")
os.environ.pop("STREAMEP_DEBUG_PB_DELAY_RANK", None)

import argparse
import sys

import torch
import torch.distributed as dist

from stream_ep import Buffer

from utils import cleanup_dist


def random_routing(n_tokens, topk, num_experts, num_local_experts, world_size,
                   device, gen):
    """K random distinct experts per token, re-randomized each call.

    Mirrors test_dispatch_grads_stress.random_routing: uniform routing is
    sufficient -- the clobbering writer is the (channel 0, writer nvl_rank 0)
    forwarder slice, which carries traffic on every dispatch at these shapes.
    """
    logits = torch.randn(n_tokens, num_experts, device=device, generator=gen)
    topk_idx = torch.topk(logits, topk, dim=-1).indices.to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(n_tokens, topk, dtype=torch.float32, device=device,
                    generator=gen), dim=-1).contiguous()
    rank_idx = topk_idx // num_local_experts
    is_token_in_rank = torch.zeros((n_tokens, world_size), dtype=torch.bool,
                                   device=device)
    for r in range(world_size):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)
    return topk_idx.contiguous(), topk_weights, is_token_in_rank


def _is_corruption_error(e: RuntimeError) -> bool:
    """Classify exceptions that are faces of the aliasing corruption."""
    msg = str(e)
    return ("StreamEP-CORRUPT" in msg                  # host poll fail-fast
            or "internode_dispatch CPU timeout" in msg  # garbage-negative, pre-fail-fast builds
            or "out of memory" in msg                   # garbage-positive pool alloc
            or "Tried to allocate" in msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_dispatches", type=int, default=8,
                   help="Forward dispatches to run. Under the Phase-B delay the "
                        "unfixed code corrupts by dispatch #2-3 (dispatch #1 is "
                        "shielded by first-call host warmup); 8 gives margin.")
    p.add_argument("--num_tokens", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=512)      # LatentMoE latent dim
    p.add_argument("--num_experts", type=int, default=512)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--tile_m", type=int, default=128)
    p.add_argument("--num_sms", type=int, default=64)      # -> 32 channels (prod)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    assert world_size % 8 == 0 and world_size > 8, (
        f"internode aliasing test needs world_size > 8 and % 8 == 0; got {world_size}")
    assert args.num_experts % world_size == 0

    Buffer.set_num_sms(args.num_sms)

    H, E, K, T = args.hidden, args.num_experts, args.topk, args.num_tokens
    num_local_experts = E // world_size

    hidden_bytes = H * 2  # bf16
    nvl_bytes, rdma_bytes = 0, 0
    for cfg in (Buffer.get_dispatch_config(world_size),
                Buffer.get_combine_config(world_size)):
        nvl_bytes = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, world_size), nvl_bytes)
        rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes)
    buf = Buffer(group, nvl_bytes, rdma_bytes)

    torch.manual_seed(100 + rank)
    x = (torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    route_gen = torch.Generator(device=device).manual_seed(7000 + rank)

    # Routing-independent hard bound: total (token, k) pairs any rank can
    # receive is world*T*K, so total_tiles <= world*T*K/tile_m + E_local.
    max_sane_tiles = (world_size * T * K) // args.tile_m + num_local_experts

    def fail(why: str):
        print(f"[nvl-aliasing] FAIL rank={rank}: {why}\n"
              f"[nvl-aliasing] This is the NVL metadata<->dispatch-data ring "
              f"aliasing regression (metadata NVL slabs overlap dispatch's "
              f"channel-0 nvl_channel_x slice; see test docstring).",
              flush=True)
        sys.exit(1)

    if rank == 0:
        print(f"[nvl-aliasing] world={world_size} H={H} E={E} K={K} T={T} "
              f"num_sms={args.num_sms} dispatches={args.num_dispatches} "
              f"delay_us={os.environ['STREAMEP_DEBUG_PB_DELAY_US']} "
              f"(victims: nvl_rank==7)",
              flush=True)

    # Keep every (pool, handle) alive until the end-of-run sync: the data
    # kernels write `pool` asynchronously on the comm stream.
    keep = []
    for i in range(args.num_dispatches):
        topk_idx, topk_weights, is_token_in_rank = random_routing(
            T, K, E, num_local_experts, world_size, device, route_gen)
        if rank == 0:
            print(f"[nvl-aliasing] dispatch {i}", flush=True)
        try:
            pool, handle, _ev = buf.dispatch(
                x, topk_idx, topk_weights, is_token_in_rank, E,
                tile_m=args.tile_m)
        except RuntimeError as e:  # torch.OutOfMemoryError subclasses RuntimeError
            if _is_corruption_error(e):
                fail(f"corrupt streaming_total_tiles at dispatch {i}: {e}")
            raise
        tiles = pool.shape[0] // args.tile_m
        if pool.shape[0] % args.tile_m != 0 or not (0 <= tiles <= max_sane_tiles):
            fail(f"insane pool geometry at dispatch {i}: pool.shape={tuple(pool.shape)} "
                 f"tiles={tiles} (sane: 0..{max_sane_tiles})")
        keep.append((pool, handle))

    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    if rank == 0:
        print(f"PASS: {args.num_dispatches} internode dispatches with a "
              f"{os.environ['STREAMEP_DEBUG_PB_DELAY_US']}us metadata Phase-B "
              f"delay on nvl_rank==7 ranks completed with sane total_tiles on "
              f"all ranks (world={world_size}, H={H}, E={E}, K={K}, T={T})",
              flush=True)

    cleanup_dist()


if __name__ == "__main__":
    main()
