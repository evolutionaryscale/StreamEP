"""
Streaming-compute dispatch signal tests (intranode).

All tests in this file are expected to FAIL until the corresponding Phase 1
milestone in ../../testing_plan.md lands. They drive the new `Buffer` surface:

  - dispatch_seq: int        -- monotonic per-call sequence number
  - dispatch_done: Tensor    -- shape [num_channels, NUM_MAX_NVL_PEERS], int64
                                  slot[ch, src_rank] >= seq  iff  the recv_x
                                  sub-range for (ch, src_rank) is fully written

Conventions match tests/test_intranode.py (script + torch.multiprocessing.spawn).
NOT pytest — see DeepEP/CLAUDE.md.
"""
import argparse
import time

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, calc_diff, inplace_unique


NUM_MAX_NVL_PEERS = 8


def build_inputs(num_tokens: int, hidden: int, num_topk: int, num_experts: int,
                 num_ranks: int, rank: int):
    """Mirror of tests/test_intranode.py:26-58 — produces dispatch inputs with
    the `x = ones * rank` trick so received sub-ranges are identifiable."""
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='cuda')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0

    return x, topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank


def get_dispatch_done(buffer):
    # NOTE(jaimec00): the actual getter name is a Phase 1 design choice; this is
    # the contract the tests assume. Replace once decided.
    return buffer.get_dispatch_done()


def get_dispatch_seq(buffer):
    return buffer.get_dispatch_seq()


def test_signal_reaches_target(buffer, config, inputs, num_channels, num_ranks,
                               local_rank):
    # NOTE(jaimec00): expected to fail until Phase 1.1 lands
    x, topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = inputs
    buffer.dispatch(
        x=x, num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx, config=config, async_finish=False,
    )
    seq = get_dispatch_seq(buffer)
    done = get_dispatch_done(buffer).cpu()
    assert done.shape == (num_channels, NUM_MAX_NVL_PEERS), \
        f'dispatch_done shape {tuple(done.shape)} != ({num_channels}, {NUM_MAX_NVL_PEERS})'
    assert done.dtype == torch.int64, f'dispatch_done dtype {done.dtype} != int64'
    valid = done[:, :num_ranks]
    assert (valid >= seq).all().item(), \
        f'rank {local_rank}: some slots below seq={seq}: {valid}'


def test_sequence_monotonicity(buffer, config, inputs, num_channels, num_ranks,
                               local_rank, n_iters: int = 5):
    # NOTE(jaimec00): expected to fail until Phase 1.2 lands
    x, topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = inputs
    seqs = []
    for _ in range(n_iters):
        buffer.dispatch(
            x=x, num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx, config=config, async_finish=False,
        )
        seq_k = get_dispatch_seq(buffer)
        done_k = get_dispatch_done(buffer).cpu()[:, :num_ranks]
        assert (done_k >= seq_k).all().item(), \
            f'iter {len(seqs)}: slots below seq={seq_k}: {done_k}'
        seqs.append(seq_k)
    assert all(seqs[i] < seqs[i + 1] for i in range(len(seqs) - 1)), \
        f'rank {local_rank}: dispatch_seq not strictly increasing: {seqs}'


def test_data_before_flag(buffer, config, inputs, num_channels, num_ranks, rank,
                          local_rank, num_tokens: int):
    # NOTE(jaimec00): expected to fail until Phase 1.3 lands
    #
    # Launch dispatch on stream A; on stream B (or host), spin on a single
    # (channel, source_rank) slot. Once observed, read that sub-range of recv_x
    # and verify it matches the source rank's expected payload (x = ones * rank).
    x, topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = inputs
    stream_a = torch.cuda.Stream()
    with torch.cuda.stream(stream_a):
        recv_x, _, _, _, handle, event = buffer.dispatch(
            x=x, num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx, config=config, async_finish=True,
        )
    seq = get_dispatch_seq(buffer)
    done = get_dispatch_done(buffer)

    # Sub-range mapping per intranode.cu:764-773 — handle[0] is rank_prefix_matrix.
    rank_prefix_matrix = handle[0]
    target_channel = 0
    target_src_rank = (rank + 1) % num_ranks

    # Host-side spin (slow but correct for a harness).
    deadline = time.time() + 10.0
    while True:
        observed = done[target_channel, target_src_rank].item()
        if observed >= seq:
            break
        if time.time() > deadline:
            raise AssertionError(
                f'rank {local_rank}: slot ({target_channel},{target_src_rank}) '
                f'never reached seq={seq} (last observed={observed})'
            )

    # NOTE(jaimec00): Phase 1.3 must finalize how to translate
    # (channel, source_rank) -> (start, end) into recv_x. The natural source is
    # recv_channel_offset combined with rank_prefix_matrix; expose via handle or
    # a getter. Stub asserts contiguity of the expected per-source data once
    # the mapping lands.
    event.current_stream_wait()
    src_total = rank_prefix_matrix[target_src_rank][rank].item() - (
        rank_prefix_matrix[target_src_rank - 1][rank].item() if target_src_rank > 0 else 0
    )
    if src_total > 0:
        # Final-state check: the full per-source slice equals target_src_rank.
        start = rank_prefix_matrix[target_src_rank - 1][rank].item() if target_src_rank > 0 else 0
        end = rank_prefix_matrix[target_src_rank][rank].item()
        slice_ = recv_x[start:end].float()
        assert (slice_ == target_src_rank).all().item(), \
            f'rank {local_rank}: per-source slice mismatch for src={target_src_rank}'


def test_cached_mode_idempotence(buffer, config, inputs, num_channels, num_ranks,
                                 local_rank):
    # NOTE(jaimec00): expected to fail until Phase 1.4 lands
    x, topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = inputs
    _, _, _, _, handle, _ = buffer.dispatch(
        x=x, num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config, async_finish=False,
    )
    seq_first = get_dispatch_seq(buffer)
    buffer.dispatch(x=x, handle=handle, config=config, async_finish=False)
    seq_second = get_dispatch_seq(buffer)
    assert seq_second > seq_first, f'cached dispatch did not advance seq: {seq_first} -> {seq_second}'
    done = get_dispatch_done(buffer).cpu()[:, :num_ranks]
    assert (done >= seq_second).all().item(), \
        f'rank {local_rank}: cached-mode slots stuck at seq < {seq_second}: {done}'


def test_zero_count_source(buffer, config, num_channels, num_ranks, rank,
                           local_rank, hidden: int, num_topk: int, num_experts: int):
    # NOTE(jaimec00): expected to fail until Phase 1.5 lands
    #
    # Construct topk_idx that routes EVERYTHING away from one chosen target
    # rank, so for that target rank there is at least one (channel, source_rank)
    # pair with zero tokens.
    num_tokens = 64
    skip_rank = (rank + 1) % num_ranks
    experts_per_rank = num_experts // num_ranks
    allowed_experts = torch.tensor(
        [e for e in range(num_experts) if (e // experts_per_rank) != skip_rank],
        device='cuda',
    )
    sel = torch.randint(0, allowed_experts.numel(), (num_tokens, num_topk), device='cuda')
    topk_idx = allowed_experts[sel].to(deep_ep.topk_idx_t)

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    rank_idx = topk_idx // experts_per_rank
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='cuda')
    is_token_in_rank_acc = torch.zeros((num_tokens, num_ranks), dtype=torch.bool, device='cuda')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        is_token_in_rank_acc[:, i] = (rank_idx == i).any(dim=-1)

    assert num_tokens_per_rank[skip_rank].item() == 0, 'test setup: skip_rank still got tokens'

    buffer.dispatch(
        x=x, num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank_acc,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx, config=config, async_finish=False,
    )
    seq = get_dispatch_seq(buffer)
    done = get_dispatch_done(buffer).cpu()[:, :num_ranks]
    # Every slot — including those whose source had zero tokens for some
    # channel — must still reach seq.
    assert (done >= seq).all().item(), \
        f'rank {local_rank}: zero-count slots failed to fire: {done}'


def test_main(args, num_sms: int, local_rank: int, num_ranks: int, rank: int,
              buffer: 'deep_ep.Buffer', group: dist.ProcessGroup):
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    assert num_experts % num_ranks == 0

    nvl_buffer_size = 256
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size)
    num_channels = num_sms // 2

    inputs = build_inputs(num_tokens, hidden, num_topk, num_experts, num_ranks, rank)

    cases = [
        ('signal_reaches_target',
         lambda: test_signal_reaches_target(buffer, config, inputs, num_channels, num_ranks, local_rank)),
        ('sequence_monotonicity',
         lambda: test_sequence_monotonicity(buffer, config, inputs, num_channels, num_ranks, local_rank)),
        ('data_before_flag',
         lambda: test_data_before_flag(buffer, config, inputs, num_channels, num_ranks, rank, local_rank, num_tokens)),
        ('cached_mode_idempotence',
         lambda: test_cached_mode_idempotence(buffer, config, inputs, num_channels, num_ranks, local_rank)),
        ('zero_count_source',
         lambda: test_zero_count_source(buffer, config, num_channels, num_ranks, rank, local_rank, hidden, num_topk, num_experts)),
    ]

    results = []
    for name, fn in cases:
        if local_rank == 0:
            print(f'[streaming] {name} ... ', flush=True, end='')
        try:
            fn()
            results.append((name, True, None))
            if local_rank == 0:
                print('passed', flush=True)
        except Exception as e:
            results.append((name, False, repr(e)))
            if local_rank == 0:
                print(f'failed ({type(e).__name__})', flush=True)
        group.barrier()

    if local_rank == 0:
        n_pass = sum(1 for _, ok, _ in results if ok)
        print(f'\n[streaming] {n_pass}/{len(results)} passed', flush=True)


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.Buffer(group, int(2e9), 0, low_latency_mode=False,
                            num_qps_per_rank=1, explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl, use_fabric=args.use_fabric)
    torch.manual_seed(rank)
    test_main(args, args.num_sms, local_rank, num_ranks, rank, buffer, group)
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streaming dispatch signal tests (intranode)')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-tokens', type=int, default=4096)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-sms', type=int, default=24)
    parser.add_argument('--allow-mnnvl', action='store_true')
    parser.add_argument('--use-fabric', action='store_true')
    args = parser.parse_args()

    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
