"""Reproduce the uneven-tokens-per-rank dispatch overflow (StreamEP).

Bug: `intranode_dispatch` / `internode_dispatch` size the two pre-poll per-tile
metadata arrays (`tile_id_to_expert`, `pool_arrival_target`) from a budget
computed off the LOCAL rank's token count assuming EVERY rank sent the same
number:

    total_tiles_max = num_tokens(local) * num_topk * num_ranks / tile_m + ...
                      (csrc/stream_ep.cpp:1164 intranode, :1610 internode)

In eval / inference each rank has a DIFFERENT local token count. A rank with
few local tokens under-budgets `total_tiles_max`, while the host-polled ACTUAL
`total_tiles` (summed cross-rank) reflects what it really receives. When a peer
sent more tokens than this rank has locally, actual > budget and the narrow

    tile_id_to_expert.narrow(0, 0, poll.total_tiles)   (csrc/stream_ep.cpp:1237-38 / :1705-06)

throws PyTorch's **"start+length exceeds dimension size"**.

Why the existing `test_skewed_experts.py` misses it: it gives every rank the
SAME T, so the uniform-per-rank assumption holds with equality even under
maximal expert skew. This test instead gives rank 0 far fewer tokens than its
peers (uneven per-rank T) under the CONTROL `uniform_rotating` routing — so a
failure here is unambiguously the uneven token count, not routing skew.

World-agnostic (reproduces on both the intranode and internode paths):
    ./scripts/srun_1node.sh StreamEP/tests/test_uneven_tokens_per_rank.py   # 8 GPU, intranode (:1164/:1237)
    ./scripts/srun_2node.sh StreamEP/tests/test_uneven_tokens_per_rank.py   # 16 GPU, internode (:1610/:1705)

Expected BEFORE the fix: rank 0 raises "start+length exceeds dimension size"
(the run fails). AFTER sizing total_tiles_max from the GLOBAL token count:
dispatch succeeds on every rank and the test passes.

Overflow is guaranteed when t_large > t_small * (world_size + 1) (uniform
routing): the underfull rank still receives ~1/world_size of the global pool,
which exceeds its local-uniform budget. Defaults (128 vs 8192) hold through
~60 ranks.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from stream_ep.stream_moe.profile_pipeline import (
    DTYPE,
    H,
    NUM_EXPERTS,
    SEQ_LEN_PER_RANK,
    TOPK,
    make_buffer,
)

# Reuse the control routing from the skewed-experts stress test (same-dir on
# sys.path when launched as a torchrun script). uniform_rotating spreads each
# rank's tokens evenly across all experts -> it PASSES with uniform T, so any
# failure here is caused purely by the uneven per-rank token counts below.
from test_skewed_experts import scenario_uniform_rotating

from utils import cleanup_dist


def local_token_count(rank: int, t_small: int, t_large: int) -> int:
    """Rank 0 is the underfull rank (few local tokens); every other rank is
    full. Deterministic — the eval/inference regime of uneven per-rank T."""
    return t_small if rank == 0 else t_large


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t_small", type=int, default=128,
                   help="rank-0 local token count (the underfull rank)")
    p.add_argument("--t_large", type=int, default=SEQ_LEN_PER_RANK,
                   help="every other rank's local token count")
    p.add_argument("--num_sms", type=int, default=None)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    assert NUM_EXPERTS % world_size == 0, f"E={NUM_EXPERTS} not divisible by world_size={world_size}"
    E, K, tile_m = NUM_EXPERTS, TOPK, 128
    T = local_token_count(rank, args.t_small, args.t_large)

    if rank == 0:
        import stream_ep
        print(f"[uneven-toks] stream_ep from: {stream_ep.__file__}", flush=True)
        print(f"[uneven-toks] world={world_size} E={E} K={K} H={H} tile_m={tile_m} | "
              f"uneven T: rank0={args.t_small}, others={args.t_large} | routing=uniform_rotating (control)",
              flush=True)
        if not (args.t_large > args.t_small * (world_size + 1)):
            print(f"[uneven-toks] WARNING: t_large={args.t_large} <= t_small*(W+1)="
                  f"{args.t_small * (world_size + 1)} — overflow may not trigger at this world size", flush=True)

    buffer = make_buffer(group, args.num_sms)
    x = (torch.randn(T, H, dtype=DTYPE, device=device) * 0.1).contiguous()
    topk_idx, topk_weights, is_token_in_rank = scenario_uniform_rotating(T, K, E, world_size, rank, device)

    # The pre-poll budget this rank's dispatch will allocate (mirrors stream_ep.cpp:1164).
    total_tiles_max = T * K * world_size // tile_m + (E // world_size) + 1

    dist.barrier(group=group, device_ids=[local_rank])

    try:
        _pool, handle, _ev = buffer.dispatch(x, topk_idx, topk_weights, is_token_in_rank, E)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 — reproduction: report whatever dispatch raised
        is_overflow = "exceeds dimension size" in str(e)
        tag = "REPRODUCED the uneven-tokens overflow" if is_overflow else "FAILED (unexpected error)"
        print(f"[uneven-toks] rank{rank} (T={T}, budget total_tiles_max={total_tiles_max}) "
              f"{tag}: {type(e).__name__}: {e}", flush=True)
        # Do NOT try to coordinate across ranks here: a rank that passed the
        # narrow is now blocked inside dispatch_main waiting on this rank's
        # (never-launched) participation. Fail the process; torchrun tears the
        # rest down.
        raise SystemExit(1)

    got_tiles = int(handle.total_tiles)
    print(f"[uneven-toks] rank{rank} (T={T}) OK: actual total_tiles={got_tiles}, "
          f"budget total_tiles_max={total_tiles_max}"
          + ("  <-- actual EXCEEDS old-formula budget (fix is working)" if got_tiles > total_tiles_max else ""),
          flush=True)

    # All ranks reached here => none overflowed. (Only reachable post-fix, or
    # on shapes that don't trigger the bug.)
    ok = torch.tensor([1], dtype=torch.int32, device=device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    dist.barrier(group=group, device_ids=[local_rank])
    if rank == 0:
        print(f"[uneven-toks] SUMMARY world={world_size}: PASS — dispatch handled uneven "
              f"tokens-per-rank on all {world_size} ranks", flush=True)
    cleanup_dist()


if __name__ == "__main__":
    main()
