"""Streaming-MoE FWD+BWD under deterministic skewed routing distributions.

Stresses the dispatch / combine NVL-region protocol under routing
distributions that production randomness rarely hits: runs the full
streaming-MoE fwd+bwd pipeline (dispatch + kernel_a + kernel_y + combine
+ bwd chain) under a set of explicitly-constructed routing scenarios,
each exercising a different edge case of the dispatch protocol.

Why "explicitly-constructed": `torch.randint`-based routing makes
failures luck-dependent. Deterministic scenarios make pass/fail
reproducible across runs and let us bisect / iterate without
re-rolling the routing seed.

Why "world-size agnostic": same script under
``./scripts/srun_1node.sh -m stream_ep.tests.test_skewed_experts`` (1-node
intranode, NVL only — fast iteration baseline) and
``./scripts/srun_4node.sh -m stream_ep.tests.test_skewed_experts`` (4-node
internode — the bug catcher). The scenarios are parameterized only by
``world_size`` / ``rank``, no per-topology branching.

Scenarios (deterministic per ``(world_size, rank)``):

  ``uniform_rotating``     — every (token, k) picks distinct experts so the
                             total load is exactly balanced. Control;
                             expected to pass even when the others fail.
  ``all_to_first_K``       — every token picks experts ``[0..K-1]``.
                             Maximal expert-dim skew toward the first K
                             experts; ranks owning those experts get all
                             tokens, others get none.
  ``half_empty_experts``   — tokens only pick even-numbered experts. Half
                             of E receive zero tokens. Exercises the
                             empty-substream / cold-expert edge case.
  ``per_rank_imbalance``   — 90% of tokens are routed exclusively to
                             experts owned by rank 1; 10% spread.
  ``power_law``            — geometric decay: expert 0 gets the largest
                             share, expert 1 half of that, etc. Most
                             realistic of training-time routing skew.

For each scenario the test:

  1. Builds ``topk_idx`` / ``topk_weights`` / ``is_token_in_rank``.
  2. Allocates random ``w1_local`` / ``w2_local`` per rank (small N(0, 0.02)).
  3. Runs ``stream_moe_func`` forward.
  4. Runs ``out.sum().backward()`` to produce gradients on the expert
     weights via the autograd chain through ``stream_moe_func``.
  5. Asserts ``isfinite(out).all()`` AND ``isfinite(w*.grad).all()``.

Pass / fail summary printed at end. Non-zero exit if any scenario
fails.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

# Reuse the production-shape constants + buffer construction utilities.
from stream_ep.stream_moe.profile_pipeline import (
    DTYPE,
    H,
    I,
    NUM_EXPERTS,
    NUM_SMS,
    SEQ_LEN_PER_RANK,
    TOPK,
    make_buffer,
)
from stream_ep.stream_moe.stream_moe import (
    make_streams,
    stream_moe_func,
)

from utils import cleanup_dist


# ---------------------------------------------------------------------------
# Scenario constructors. Each returns (topk_idx, topk_weights,
# is_token_in_rank) for the given (T, K, E, world_size, rank). All values
# deterministic in those parameters — no RNG.
# ---------------------------------------------------------------------------
def _topk_weights_uniform(T: int, K: int, device: torch.device) -> torch.Tensor:
    """1/K weight for every (t, k) — never NaN, never imbalanced."""
    return torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device).contiguous()


def _is_token_in_rank(topk_idx: torch.Tensor, world_size: int, num_local_experts: int) -> torch.Tensor:
    """Standard derivation: token t goes to rank r iff any topk_idx[t, :] // E_local == r."""
    T, K = topk_idx.shape
    rank_idx = topk_idx // num_local_experts            # [T, K]
    ranks = torch.arange(world_size, device=topk_idx.device).view(1, 1, -1)
    return (rank_idx.unsqueeze(-1) == ranks).any(dim=1)   # [T, world_size] bool


def scenario_uniform_rotating(T: int, K: int, E: int, world_size: int, rank: int,
                              device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Control: every token picks K *distinct* experts, rotating through E.

    Token t picks experts [(t*K + 0) % E, (t*K + 1) % E, ..., (t*K + K-1) % E].
    Total load per expert is exactly T*K / E across all tokens. Each token's
    K choices are distinct (no duplicate experts in a single token's topk).
    """
    base = torch.arange(T, device=device, dtype=torch.int64).unsqueeze(1) * K
    offsets = torch.arange(K, device=device, dtype=torch.int64).unsqueeze(0)
    topk_idx = ((base + offsets) % E).contiguous()
    num_local_experts = E // world_size
    return topk_idx, _topk_weights_uniform(T, K, device), _is_token_in_rank(topk_idx, world_size, num_local_experts)


def scenario_all_to_first_K(T: int, K: int, E: int, world_size: int, rank: int,
                             device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every token picks experts [0, 1, ..., K-1]. Maximal expert-dim skew."""
    topk_idx = torch.arange(K, device=device, dtype=torch.int64).unsqueeze(0).expand(T, K).contiguous()
    num_local_experts = E // world_size
    return topk_idx, _topk_weights_uniform(T, K, device), _is_token_in_rank(topk_idx, world_size, num_local_experts)


def scenario_half_empty_experts(T: int, K: int, E: int, world_size: int, rank: int,
                                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokens only pick even-numbered experts. Odd experts receive 0 tokens."""
    even = torch.arange(0, E, 2, device=device, dtype=torch.int64)   # [E/2]
    base = torch.arange(T, device=device, dtype=torch.int64).unsqueeze(1) * K
    offsets = torch.arange(K, device=device, dtype=torch.int64).unsqueeze(0)
    idx_in_even = (base + offsets) % even.numel()                     # [T, K]
    topk_idx = even[idx_in_even].contiguous()
    num_local_experts = E // world_size
    return topk_idx, _topk_weights_uniform(T, K, device), _is_token_in_rank(topk_idx, world_size, num_local_experts)


def make_per_rank_imbalance(skew_pct: int, target_rank: int = 1):
    """Factory: returns a scenario that routes ``skew_pct``% of tokens exclusively
    to experts owned by ``target_rank``. The remaining ``100 - skew_pct``% rotate
    through all experts (uniform fallback). At skew_pct=0 it's effectively uniform;
    at skew_pct=100 every token goes to target_rank.
    """
    skew_pct = max(0, min(100, int(skew_pct)))

    def scenario(T: int, K: int, E: int, world_size: int, rank: int,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_local_experts = E // world_size
        clamped_target = min(target_rank, world_size - 1)
        target_e_lo = clamped_target * num_local_experts
        t_ix = torch.arange(T, device=device, dtype=torch.int64).unsqueeze(1)
        k_ix = torch.arange(K, device=device, dtype=torch.int64).unsqueeze(0)
        hot_idx = target_e_lo + ((t_ix * K + k_ix) % num_local_experts)
        cold_idx = (t_ix * K + k_ix) % E
        # Hot iff t * 100 < skew_pct * T (deterministic first-N-fraction split).
        is_hot = (t_ix * 100 < skew_pct * T).expand(T, K)
        topk_idx = torch.where(is_hot, hot_idx, cold_idx).contiguous()
        return topk_idx, _topk_weights_uniform(T, K, device), _is_token_in_rank(topk_idx, world_size, num_local_experts)
    return scenario


def scenario_per_rank_imbalance(T, K, E, world_size, rank, device):
    """90% of tokens to rank 1 — default skew, kept as the canonical scenario name
    when the test is invoked without --per-rank-skew."""
    return make_per_rank_imbalance(90)(T, K, E, world_size, rank, device)


def scenario_power_law(T: int, K: int, E: int, world_size: int, rank: int,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Heavy-tailed: expert e's share decays exponentially in e, but tail
    covers ALL experts (so every rank gets some local work).

    Token t's primary expert is ``floor(E ** (t/T))``, which is a
    deterministic inverse-CDF parameterization of a power-law over the
    full ``[0, E)`` range. Concretely at T=8192, E=384:

      t=0       primary=0       (expert 0 — hot)
      t=T/4     primary~3       (expert 3 — still hot)
      t=T/2     primary~19      (expert 19 — moderate)
      t=3T/4    primary~87      (expert 87)
      t=T-1     primary=E-1     (expert 383 — tail)

    Roughly half the tokens go to the first ~20 experts, the other half
    spread over the rest. Every rank gets some traffic.

    Secondary K-1 slots: rotate through E starting at ``primary + 1`` so
    each token's K experts are distinct.
    """
    t_norm = torch.arange(T, device=device, dtype=torch.float64) / T
    primary_expert = (float(E) ** t_norm).clamp_(0.0, float(E - 1)).to(torch.int64)
    k_ix = torch.arange(K, device=device, dtype=torch.int64).unsqueeze(0)
    topk_idx = ((primary_expert.unsqueeze(1) + k_ix) % E).contiguous()
    num_local_experts = E // world_size
    return topk_idx, _topk_weights_uniform(T, K, device), _is_token_in_rank(topk_idx, world_size, num_local_experts)


# Order: control → milder skews → realistic skew → extreme degenerate. Bail
# on first failure (a CUDA-side trap poisons the context — subsequent
# scenarios in the same process can't get clean signal). The first failing
# scenario is the meaningful data point; the extreme degenerate scenarios
# at the tail of the list are known to fail at small world sizes (most
# ranks have no work, barrier_block times out) and are kept for diagnostic
# completeness, not as expected-pass tests.
SCENARIOS = [
    ("uniform_rotating",   scenario_uniform_rotating),   # CONTROL — must pass
    ("half_empty_experts", scenario_half_empty_experts), # cold-substream
    ("per_rank_imbalance", scenario_per_rank_imbalance), # NVL imbalance
    ("power_law",          scenario_power_law),          # realistic skew (Bug B at 4-node)
    ("all_to_first_K",     scenario_all_to_first_K),     # extreme; may
                                                          # legitimately crash
                                                          # at small world
                                                          # sizes
]


# ---------------------------------------------------------------------------
# Per-scenario runner. Each scenario reuses the same Buffer / streams /
# w1_local / w2_local — only the routing changes. Asserts finite outputs and
# grads. Returns (passed: bool, msg: str).
# ---------------------------------------------------------------------------
def run_scenario(name: str, builder, *,
                 T: int, K: int, E: int, world_size: int, rank: int,
                 device: torch.device, buffer, streams,
                 w1_local: torch.Tensor, w2_local: torch.Tensor,
                 x: torch.Tensor) -> tuple[bool, str]:
    # Zero accumulated grads from prior scenarios so each scenario's grads
    # reflect only its own backward pass.
    if w1_local.grad is not None:
        w1_local.grad.zero_()
    if w2_local.grad is not None:
        w2_local.grad.zero_()

    topk_idx, topk_weights, is_token_in_rank = builder(T, K, E, world_size, rank, device)

    # Sanity asserts on routing tensors before passing to dispatch.
    assert topk_idx.shape == (T, K) and topk_idx.dtype == torch.int64, (
        f"{name}: topk_idx shape/dtype mismatch: got {tuple(topk_idx.shape)} / {topk_idx.dtype}")
    assert topk_weights.shape == (T, K) and topk_weights.dtype == torch.float32, (
        f"{name}: topk_weights shape/dtype mismatch")
    assert is_token_in_rank.shape == (T, world_size) and is_token_in_rank.dtype == torch.bool, (
        f"{name}: is_token_in_rank shape/dtype mismatch")
    assert (topk_idx >= 0).all() and (topk_idx < E).all(), (
        f"{name}: topk_idx out of range [0, {E})")

    try:
        out = stream_moe_func(
            buffer,
            x.contiguous(),
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1_local,
            w2_local,
            streams=streams,
            num_experts=E,
        )
        out_sum = out.sum()
        out_sum.backward()
        torch.cuda.synchronize()
    except Exception as e:
        return False, f"{name}: exception {type(e).__name__}: {e}"

    # Finite-check the rank-local output + grads. Each rank only sees its own
    # slice but a NaN on any rank will fire the assert and surface the
    # scenario as failing.
    if not torch.isfinite(out).all():
        nbad = int((~torch.isfinite(out)).sum().item())
        return False, f"{name}: out has {nbad}/{out.numel()} non-finite"
    for pname, p in [("w1_local.grad", w1_local.grad), ("w2_local.grad", w2_local.grad)]:
        if p is None:
            return False, f"{name}: {pname} is None (autograd did not flow)"
        if not torch.isfinite(p).all():
            nbad = int((~torch.isfinite(p)).sum().item())
            return False, f"{name}: {pname} has {nbad}/{p.numel()} non-finite"
    return True, f"{name}: OK"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", action="append", default=None,
                   help="Run only the named scenarios (may repeat). Default: all SCENARIOS in order.")
    p.add_argument("--per-rank-skew", type=int, default=None,
                   help="Override skew_pct on the per_rank_imbalance scenario (0-100). "
                        "Default 90 (the SCENARIOS default).")
    p.add_argument("--T", type=int, default=SEQ_LEN_PER_RANK,
                   help="Per-rank token count (default = production SEQ_LEN_PER_RANK).")
    p.add_argument("--num_sms", type=int, default=None,
                   help="SMs override; default = Buffer auto-pick by world size.")
    p.add_argument("--per-rank-target", type=int, default=1,
                   help="Target rank for per_rank_imbalance scenario (default 1).")
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    assert NUM_EXPERTS % world_size == 0, (
        f"NUM_EXPERTS={NUM_EXPERTS} must be divisible by world_size={world_size}")
    num_local_experts = NUM_EXPERTS // world_size

    T = args.T
    H_dim = H
    K = TOPK
    E = NUM_EXPERTS

    # Apply --per-rank-skew / --per-rank-target override by rebuilding the scenario.
    scenarios = list(SCENARIOS)
    if args.per_rank_skew is not None or args.per_rank_target != 1:
        skew = args.per_rank_skew if args.per_rank_skew is not None else 90
        tgt = args.per_rank_target
        scenarios = [
            (f"per_rank_imbalance_skew{skew}_tgt{tgt}",
             make_per_rank_imbalance(skew, target_rank=tgt))
            if name == "per_rank_imbalance" else (name, builder)
            for name, builder in scenarios
        ]
    # Filter via --scenario if given. List-based (preserves order + duplicates
    # so [skew, skew] actually re-runs the same scenario twice — needed when
    # a flake reproduces only after a re-issue of the same routing).
    if args.scenario:
        # Allow the user's name "per_rank_imbalance" to match the renamed
        # scenario (per_rank_imbalance_skewN_tgtR) when --per-rank-skew or
        # --per-rank-target was supplied.
        scenarios_by_name = {n: b for n, b in scenarios}
        def _resolve(name):
            if name in scenarios_by_name:
                return (name, scenarios_by_name[name])
            for n, b in scenarios_by_name.items():
                if n.startswith("per_rank_imbalance") and name == "per_rank_imbalance":
                    return (n, b)
            return None
        scenarios = [r for name in args.scenario for r in [_resolve(name)] if r is not None]
        if not scenarios:
            if rank == 0:
                print(f"[skewed-experts] no scenarios matched {args.scenario}", flush=True)
            dist.destroy_process_group()
            return

    if rank == 0:
        print(f"[skewed-experts] world_size={world_size} T={T} H={H_dim} K={K} E={E} "
              f"local_E={num_local_experts} per_rank_skew={args.per_rank_skew or 'default(90)'}", flush=True)
        print(f"[skewed-experts] scenarios: {[n for n, _ in scenarios]}", flush=True)

    # Buffer + streams: shared across scenarios.
    buffer = make_buffer(group, args.num_sms)
    streams = make_streams(device=device)

    # Per-rank expert weights (small N(0, 0.02) so output magnitudes are stable).
    torch.manual_seed(100 + rank)
    w1_local = torch.nn.Parameter(torch.randn(
        num_local_experts, 2 * I, H_dim, dtype=DTYPE, device=device) * 0.02)
    w2_local = torch.nn.Parameter(torch.randn(
        num_local_experts, H_dim, I, dtype=DTYPE, device=device) * 0.02)
    x = (torch.randn(T, H_dim, dtype=DTYPE, device=device) * 0.1).contiguous()

    # Run scenarios sequentially. Bail on first failure — a kernel-side trap
    # poisons the CUDA context, so subsequent scenarios in the same process
    # produce noise rather than signal. Mark unrun scenarios SKIPPED. Each
    # scenario barriers before so a hang on one rank doesn't desync rank-0
    # reporting on the next one.
    results: list[tuple[str, str, str]] = []  # (name, status, msg) where status in {PASS, FAIL, SKIPPED}
    bailed = False
    for name, builder in scenarios:
        if bailed:
            results.append((name, "SKIPPED", f"{name}: skipped (prior scenario poisoned CUDA context)"))
            if rank == 0:
                print(f"[skewed-experts] SKIP: {name} (post-poisoning)", flush=True)
            continue

        dist.barrier(group=group, device_ids=[local_rank])
        if rank == 0:
            print(f"[skewed-experts] running {name} ...", flush=True)
        passed, msg = run_scenario(
            name, builder,
            T=T, K=K, E=E, world_size=world_size, rank=rank,
            device=device, buffer=buffer, streams=streams,
            w1_local=w1_local, w2_local=w2_local, x=x)
        # Aggregate pass/fail across ranks: a scenario passes globally iff every
        # rank reports pass. Use all_reduce on an int32 (1=pass, 0=fail).
        my_pass = torch.tensor([1 if passed else 0], dtype=torch.int32, device=device)
        dist.all_reduce(my_pass, op=dist.ReduceOp.MIN, group=group)
        global_pass = bool(my_pass.item() == 1)
        status = "PASS" if global_pass else "FAIL"
        results.append((name, status, msg))
        if rank == 0:
            print(f"[skewed-experts] {status}: {msg}", flush=True)
        if not global_pass:
            bailed = True

    dist.barrier(group=group, device_ids=[local_rank])

    if rank == 0:
        n_pass = sum(1 for _, s, _ in results if s == "PASS")
        n_fail = sum(1 for _, s, _ in results if s == "FAIL")
        n_skip = sum(1 for _, s, _ in results if s == "SKIPPED")
        print(f"[skewed-experts] SUMMARY world_size={world_size}: "
              f"{n_pass} pass / {n_fail} fail / {n_skip} skipped", flush=True)
        for name, status, msg in results:
            print(f"  {status}: {msg}", flush=True)

    cleanup_dist()

    # Exit non-zero if any scenario explicitly failed.
    if rank == 0 and any(s == "FAIL" for _, s, _ in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
