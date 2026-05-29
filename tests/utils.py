import torch
import torch.distributed as dist
from typing import Optional


def cleanup_dist():
    # Sync all ranks then tear down so the TCPStore client closes in lockstep.
    # Without the barrier, rank 0 exits first and the others' heartbeat monitor
    # logs a (harmless but loud) "recvBytes ... Connection was likely closed".
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
        dist.destroy_process_group()


def make_inputs(num_tokens: int, hidden: int, num_topk: int, num_experts: int,
                num_ranks: int, rank: int, device: torch.device, *,
                seed: int = 123,
                x_kind: str = "rank_tag",
                plant_sentinels: bool = False,
                sentinel_p: float = 0.05,
                plant_empty_expert: Optional[int] = None):
    """Standard per-rank dispatch/combine test inputs.

    ``x_kind``:
      - ``"rank_tag"`` (default): ``x[t, :] = float(rank)`` — lets dispatch
        and combine tests verify pool / reduced output in closed form.
      - ``"randn"``: random normal — for gradient correctness tests where
        the input distribution matters.

    Sentinel plant (``plant_sentinels=True``, default p=5%) flips a random
    ``topk_idx`` subset to ``-1`` to exercise the skip branch. The empty-
    expert plant (``plant_empty_expert=e``) additionally rewrites every
    ``topk_idx == e`` to ``-1`` so expert ``e`` receives zero tokens this
    iter (exercises the empty-expert branch on receivers).
    """
    import stream_ep  # late import — tests/utils.py is imported by harnesses pre-Buffer-init

    g = torch.Generator(device=device).manual_seed(seed + rank)
    if x_kind == "rank_tag":
        x = torch.full((num_tokens, hidden), float(rank),
                       dtype=torch.bfloat16, device=device)
    elif x_kind == "randn":
        x = torch.randn((num_tokens, hidden), generator=g, device=device,
                        dtype=torch.bfloat16)
    else:
        raise ValueError(f"unknown x_kind {x_kind!r}")

    idx = torch.randint(0, num_experts, (num_tokens, num_topk),
                        generator=g, device=device, dtype=torch.int64)
    if plant_sentinels:
        sentinel = torch.rand((num_tokens, num_topk),
                              generator=g, device=device) < sentinel_p
        idx = torch.where(sentinel, torch.full_like(idx, -1), idx)
    if plant_empty_expert is not None:
        idx = torch.where(idx == plant_empty_expert,
                          torch.full_like(idx, -1), idx)
    topk_idx = idx.to(stream_ep.topk_idx_t)

    topk_weights = torch.rand((num_tokens, num_topk), generator=g,
                              device=device, dtype=torch.float32)

    num_local_experts = num_experts // num_ranks
    rank_idx = torch.where(topk_idx >= 0,
                           topk_idx // num_local_experts,
                           torch.full_like(topk_idx, -1))
    is_token_in_rank = torch.zeros((num_tokens, num_ranks),
                                   dtype=torch.bool, device=device)
    for r in range(num_ranks):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)
    return x, topk_idx, topk_weights, is_token_in_rank
