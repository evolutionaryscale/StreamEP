"""Dispatch when ranks contribute different numbers of tokens.

Rank 0 sends far fewer tokens than its peers under balanced routing. Dispatch
must size its pre-poll per-tile metadata (``tile_id_to_expert``,
``pool_arrival_target``) from the global token count: sized from the local
count, the underfull rank's budget is smaller than the ``total_tiles`` it
actually receives and narrowing to it fails. The overflow condition holds when
``t_large > t_small * (world_size + 1)``; the defaults (128 vs 8192) hold up
to about 60 ranks.
"""

import argparse
import os

import torch
import torch.distributed as dist

from stream_ep import Buffer
from stream_ep_tests.utils import DTYPE, H, NUM_EXPERTS, SEQ_LEN_PER_RANK, TOPK

# The control routing from the skewed-experts test: uniform_rotating spreads each
# rank's tokens evenly across all experts and passes with uniform T, so any
# failure here comes from the uneven per-rank token counts below.
from stream_ep_tests.dist.test_skewed_experts import scenario_uniform_rotating
from stream_ep_tests.utils import cleanup_dist


def local_token_count(rank: int, t_small: int, t_large: int) -> int:
    """Rank 0 is the underfull rank (few local tokens); every other rank is
    full. Deterministic — the eval/inference regime of uneven per-rank T."""
    return t_small if rank == 0 else t_large


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t_small", type=int, default=128, help="rank-0 local token count (the underfull rank)")
    p.add_argument("--t_large", type=int, default=SEQ_LEN_PER_RANK, help="every other rank's local token count")
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
        print(
            f"[uneven-toks] world={world_size} E={E} K={K} H={H} tile_m={tile_m} | "
            f"uneven T: rank0={args.t_small}, others={args.t_large} | routing=uniform_rotating (control)",
            flush=True,
        )
        if not (args.t_large > args.t_small * (world_size + 1)):
            print(
                f"[uneven-toks] WARNING: t_large={args.t_large} <= t_small*(W+1)="
                f"{args.t_small * (world_size + 1)} — overflow may not trigger at this world size",
                flush=True,
            )

    if args.num_sms is not None:
        Buffer.set_num_sms(args.num_sms)
    buffer = Buffer.sized_for(group, H * DTYPE.itemsize)
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
        print(f"[uneven-toks] rank{rank} (T={T}, budget total_tiles_max={total_tiles_max}) {tag}: {type(e).__name__}: {e}", flush=True)
        # Do NOT try to coordinate across ranks here: a rank that passed the
        # narrow is now blocked inside dispatch_main waiting on this rank's
        # (never-launched) participation. Fail the process; torchrun tears the
        # rest down.
        raise SystemExit(1) from e

    got_tiles = int(handle.total_tiles)
    print(
        f"[uneven-toks] rank{rank} (T={T}) OK: actual total_tiles={got_tiles}, "
        f"budget total_tiles_max={total_tiles_max}"
        + ("  <-- actual EXCEEDS old-formula budget (fix is working)" if got_tiles > total_tiles_max else ""),
        flush=True,
    )

    # All ranks reached here => none overflowed. (Only reachable post-fix, or
    # on shapes that don't trigger the bug.)
    ok = torch.tensor([1], dtype=torch.int32, device=device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    dist.barrier(group=group, device_ids=[local_rank])
    if rank == 0:
        print(
            f"[uneven-toks] SUMMARY world={world_size}: PASS — dispatch handled uneven tokens-per-rank on all {world_size} ranks",
            flush=True,
        )
    cleanup_dist()


if __name__ == "__main__":
    main()
