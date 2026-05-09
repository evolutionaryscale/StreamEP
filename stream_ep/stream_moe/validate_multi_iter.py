"""End-to-end correctness check for the streaming pipeline (fwd + bwd).

Runs `stream_moe_func` for N iterations with the SAME inputs and asserts each
iteration's output AND four bwd gradients (`dL/dx`, `dL/dtopk_weights`,
`dL/dW1_local`, `dL/dW2_local`) match a torch-eager full MoE reference. The
reference is computed in plain torch using all ranks' weights (gathered across
the group) so that for each source token ``t`` on this rank,

    output[t, :] = Σ over k of topk_weights[t, k] * SwiGLU(x[t] @ W1[topk_idx[t, k]]) @ W2[topk_idx[t, k]]

and the four gradients are produced by `torch.autograd.grad` against the same
graph (NOT finite differences — bf16 noise floor makes FD useless).

This catches both (a) the cross-layer-race class the historical kernel-A had
(iter 0 correct, iter K silently wrong because some CTAs of iter N+1 saw stale
"all done" state from iter N), now also for the bwd's `bwd_y_ready` /
`bwd_a_ready` / `bwd_compute_done_per_token` cross-iter signals, and (b) any
regression in the fwd / bwd pipeline as a whole — including the per-token
gate's interaction with the release-store path on either direction.

Symptom signature for the cross-layer race: iter 0 PASS, iter K (K >= 1) FAIL,
on the side (fwd or bwd) that holds stale state. The bench_pipeline.py timing
harness wouldn't catch it (timing-only), and the per-kernel test suite uses a
single iter.

Launch
------
    torchrun --nproc_per_node=8 \\
        -m stream_ep.stream_moe.validate_multi_iter \\
        [--n_iter 20]

Outputs PASS / FAIL per iter on rank 0, with per-tensor max-abs / max-rel
diffs for the failing iters.

Per-tensor thresholds
---------------------
``out`` / ``dL/dx`` / ``dL/dtopk_weights`` are checked tightly (atol=1e-3,
rtol=1e-2) — they sum over at most K_local ≤ TOPK contributions per token (or
H per (t, k) for dL/dweight), so the bf16 chain noise floor is well below
1e-3 at production magnitudes. ``dL/dW1_local`` / ``dL/dW2_local`` use a
looser (atol=8e-2, rtol=8e-2) because their grouped GEMM sums over the per-
expert K-axis (= expert_frequency, scales with world size — 8192 at 16-rank
internode, 4096 at 8-rank intranode). bf16 storage of preact_a (the largest-
magnitude intermediate, |max| ~ 1.5 → bf16 ULP ~0.012) propagates through
swiglu_bwd into dL_dswiglu_in / postact_a_for_dW2; when summed over K=8192
in the dW gemm the noise lands at ~0.06 max-abs. This matches sonic-moe's
bf16 grad tolerance (atol=2e-2 at x_scale=0.02) once scaled to our x_scale=0.1
and our larger K. Storing preact in fp32 would tighten this 5-10× at the
cost of ~256 MB / layer; see logbook.
"""

import argparse

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F


# Per-tensor (atol, rtol) thresholds. Matched to the bf16 chain noise floor
# at production shape — see module docstring.
TOL_DEFAULT = {
    "out":              (1e-3, 1e-2),
    "dL/dx":            (1e-3, 1e-2),
    "dL/dtopk_weights": (1e-3, 1e-2),
    "dL/dW1_local":     (8e-2, 8e-2),
    "dL/dW2_local":     (8e-2, 8e-2),
}

from stream_ep.stream_moe.profile_pipeline import (
    DTYPE,
    NUM_EXPERTS,
    NUM_SMS,
    SEQ_LEN_PER_RANK,
    TILE_M,
    TILE_N_A,
    TILE_N_Y,
    TOPK,
    H,
    I,
    make_buffer,
    make_uniform_topk_idx,
)
from stream_ep.stream_moe.stream_moe import (
    make_streams,
    stream_moe_func,
)
from stream_ep.stream_moe.profile_pipeline import (
    barrier,
    get_global_rank,
    get_world_size,
    init_distributed,
    rank_zero_print,
)


def torch_reference_full_moe(
    x: torch.Tensor,  # (T, H) — this rank's source tokens
    topk_idx: torch.Tensor,  # (T, K) global expert indices
    topk_weights: torch.Tensor,  # (T, K) float32
    w1_full: torch.Tensor,  # (E_total, 2I, H) all ranks' W1
    w2_full: torch.Tensor,  # (E_total, H, I) all ranks' W2
) -> torch.Tensor:
    """Eager full MoE forward: for each source token t, compute the topk-weighted
    sum of SwiGLU(x[t] @ W1[topk_idx[t, k]]) @ W2[topk_idx[t, k]] over k. Returns
    [T, H] in the same dtype as x.

    This is the production-output target for ``stream_moe_func``: dispatch
    sends each (t, k) pair to its expert's rank, kernel A + Y compute the
    weighted contribution locally, and combine reduces them back into out[t]
    on the source rank.

    Implementation: group-by-expert. For each expert e, gather the (token,
    k-slot) pairs that route to e, run one expert forward per pair-batch, then
    weighted-scatter back into out. Avoids expanding to (T, 2I, H) per K
    (~8 GB per K at production shape) which OOMs at 80GB.

    Autograd-compatible: uses functional ``Tensor.index_add`` (out-of-place)
    instead of ``index_add_``, so the reference can be re-used to gather grads
    via ``torch.autograd.grad`` for the bwd validation. Each loop iter rebinds
    ``out`` to a new tensor; the autograd graph holds all E_total intermediates
    (~64 × 16 MB at production = ~1 GB on H100, acceptable).
    """
    T, H_dim = x.shape
    E_total, two_I, _ = w1_full.shape
    out = torch.zeros(T, H_dim, dtype=torch.float32, device=x.device)
    x_f = x.to(torch.float32)
    w_f = topk_weights.to(torch.float32)

    # Flatten (token, k) pairs and bucket by expert.
    flat_e = topk_idx.flatten()  # (T*K,)
    flat_t = (
        torch.arange(T, device=x.device)
        .unsqueeze(1)
        .expand(T, topk_idx.shape[1])
        .flatten()
    )  # (T*K,)
    flat_w = w_f.flatten()  # (T*K,)

    for e in range(E_total):
        mask_e = flat_e == e
        if not mask_e.any():
            continue
        idx_e = mask_e.nonzero(as_tuple=False).flatten()  # (n_e,)
        t_e = flat_t[idx_e]  # (n_e,) — token indices
        w_e = flat_w[idx_e]  # (n_e,) — weights for these (t, k) pairs
        x_e = x_f[t_e]  # (n_e, H)

        h = x_e @ w1_full[e].to(torch.float32).T  # (n_e, 2I)
        # Paired-N gate/up split: gate at even indices, up at odd.
        # Quack's gated epilogue (`gemm_act.py:235-236`) pairs adjacent N-elements,
        # so the streaming kernel A's W1 layout puts gate weights at row 2i and
        # up weights at row 2i+1. `h.chunk(2, dim=-1)` (concat-row split) would
        # mis-interpret that layout and drift the reference numerically off the
        # streaming output — the divergence ate ~50% of (t, k) gradients before
        # this fix, despite forward `out` happening to slip under the dual-
        # threshold rule because the rtol/atol pair didn't both fire.
        u = h[..., 0::2]
        v = h[..., 1::2]
        a = F.silu(u) * v  # (n_e, I)
        y = a @ w2_full[e].to(torch.float32).T  # (n_e, H)

        out = out.index_add(0, t_e.to(torch.int64), w_e[:, None] * y)

    return out.to(x.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_sms", type=int, default=NUM_SMS)
    p.add_argument("--seq_len", type=int, default=SEQ_LEN_PER_RANK)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_iter", type=int, default=20)
    args = p.parse_args()

    device = init_distributed()
    rank, world_size = get_global_rank(), get_world_size()
    group = torch_dist.group.WORLD
    local_E = NUM_EXPERTS // world_size

    rank_zero_print(
        f"[validate] world={world_size} num_sms={args.num_sms} "
        f"H={H} I={I} E={NUM_EXPERTS} K={TOPK} T={args.seq_len} "
        f"n_warmup={args.n_warmup} n_iter={args.n_iter}"
    )
    rank_zero_print(
        "[validate] per-tensor thresholds (atol, rtol): "
        + ", ".join(f"{n}={a:.0e}/{r:.0e}" for n, (a, r) in TOL_DEFAULT.items())
    )

    buffer = make_buffer(group, args.num_sms)

    # Build GLOBAL weights (same across ranks via shared seed). w1_local /
    # w2_local are the rank's slice for the streaming pipeline; the eager
    # reference uses the FULL weights to compute each token's expert
    # contributions regardless of which rank owns them.
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

    torch.manual_seed(100 + rank)
    x = (torch.randn(args.seq_len, H, dtype=DTYPE, device=device) * 0.1).contiguous()
    topk_idx = make_uniform_topk_idx(args.seq_len, TOPK, NUM_EXPERTS, rank, device)
    topk_weights = torch.softmax(
        torch.randn(args.seq_len, TOPK, dtype=torch.float32, device=device), dim=-1
    ).contiguous()

    rank_idx = topk_idx // local_E
    is_token_in_rank = torch.zeros(
        (args.seq_len, world_size), dtype=torch.bool, device=device
    )
    for r in range(world_size):
        is_token_in_rank[:, r] = (rank_idx == r).any(dim=-1)

    # Compute the eager full-MoE reference + reference grads once (all inputs
    # are fixed across iters). Reference grads target every differentiable
    # forward arg; we slice the global w1/w2 grads to the per-rank local
    # window for direct comparison against the streaming bwd's per-rank
    # outputs.
    #
    # Use detach().clone() to make fresh leaves (independent of the streaming
    # path's tensors) and a fixed grad_out so cross-iter comparisons stay
    # meaningful.
    torch.manual_seed(200 + rank)
    grad_out = (
        torch.randn(args.seq_len, H, dtype=DTYPE, device=device) * 0.1
    ).contiguous()

    x_ref = x.detach().clone().requires_grad_(True)
    topk_w_ref = topk_weights.detach().clone().requires_grad_(True)
    w1_full_ref = w1_full.detach().clone().requires_grad_(True)
    w2_full_ref = w2_full.detach().clone().requires_grad_(True)
    out_ref = torch_reference_full_moe(
        x_ref, topk_idx, topk_w_ref, w1_full_ref, w2_full_ref
    )
    dL_dx_ref, dL_dtopk_w_ref, dL_dW1_full_ref, dL_dW2_full_ref = torch.autograd.grad(
        out_ref, [x_ref, topk_w_ref, w1_full_ref, w2_full_ref], grad_outputs=grad_out
    )
    out_ref = out_ref.detach()
    # `.clone()` (not `.contiguous()` — contiguous on an already-contiguous
    # slice returns the SAME view, keeping the global tensor's storage pinned)
    # so the (world_size - 1) / world_size of the global dW gradients can be
    # released after we cache the per-rank slabs.
    dL_dW1_local_ref = dL_dW1_full_ref[rank * local_E : (rank + 1) * local_E].clone()
    dL_dW2_local_ref = dL_dW2_full_ref[rank * local_E : (rank + 1) * local_E].clone()
    del dL_dW1_full_ref, dL_dW2_full_ref
    del x_ref, topk_w_ref, w1_full_ref, w2_full_ref

    streams = make_streams()
    barrier(group)

    # Warmup (no validation). Detach + clone per warmup iter so the warmup's
    # autograd graph doesn't leak into the validated iters' state.
    for warm_seq in range(1, args.n_warmup + 1):
        x_warm = x.detach().clone().requires_grad_(True)
        topk_w_warm = topk_weights.detach().clone().requires_grad_(True)
        w1_warm = w1_local.detach().clone().requires_grad_(True)
        w2_warm = w2_local.detach().clone().requires_grad_(True)
        out_warm = stream_moe_func(
            buffer,
            x_warm,
            topk_idx,
            topk_w_warm,
            is_token_in_rank,
            w1_warm,
            w2_warm,
            streams=streams,
            num_experts=NUM_EXPERTS,
            dispatch_seq=warm_seq,
            tile_m=TILE_M,
            tile_n_a=TILE_N_A,
            tile_n_y=TILE_N_Y,
        )
        # Drive the bwd path during warmup too — exercises the bwd kernels'
        # JIT cache and the cross-iter signal-clear hygiene.
        torch.autograd.grad(
            out_warm, [x_warm, topk_w_warm, w1_warm, w2_warm], grad_outputs=grad_out
        )
    torch.cuda.synchronize()
    barrier(group)

    # Validated iters.
    fail_count = 0
    for step in range(args.n_iter):
        seq = 100 + step
        # Fresh leaves per iter so the autograd graph from one iter never
        # leaks into the next; this also matches how a real training loop
        # creates per-iter tensors via ``Parameter`` / activation forwards.
        x_iter = x.detach().clone().requires_grad_(True)
        topk_w_iter = topk_weights.detach().clone().requires_grad_(True)
        w1_iter = w1_local.detach().clone().requires_grad_(True)
        w2_iter = w2_local.detach().clone().requires_grad_(True)

        out_actual = stream_moe_func(
            buffer,
            x_iter,
            topk_idx,
            topk_w_iter,
            is_token_in_rank,
            w1_iter,
            w2_iter,
            streams=streams,
            num_experts=NUM_EXPERTS,
            dispatch_seq=seq,
            tile_m=TILE_M,
            tile_n_a=TILE_N_A,
            tile_n_y=TILE_N_Y,
        )
        # Drive StreamMoEFunc.backward via autograd.grad — yields all four
        # per-rank gradients in one call. Doing this via .backward() would
        # accumulate into .grad slots; .grad is cleaner and never depends on
        # the user's optimizer-state hygiene.
        dL_dx_actual, dL_dtopk_w_actual, dL_dW1_local_actual, dL_dW2_local_actual = (
            torch.autograd.grad(
                out_actual,
                [x_iter, topk_w_iter, w1_iter, w2_iter],
                grad_outputs=grad_out,
            )
        )
        torch.cuda.synchronize()

        # Compare every (actual, ref) pair under per-tensor (atol, rtol).
        # Fail iff BOTH absolute and relative diff thresholds are violated.
        diagnostics = []
        ok = True
        for name, actual, ref in [
            ("out", out_actual, out_ref),
            ("dL/dx", dL_dx_actual, dL_dx_ref),
            ("dL/dtopk_weights", dL_dtopk_w_actual, dL_dtopk_w_ref),
            ("dL/dW1_local", dL_dW1_local_actual, dL_dW1_local_ref),
            ("dL/dW2_local", dL_dW2_local_actual, dL_dW2_local_ref),
        ]:
            atol, rtol = TOL_DEFAULT[name]
            actual_f = actual.to(torch.float32)
            ref_f = ref.to(torch.float32)
            diff = (actual_f - ref_f).abs()
            rel = diff / (ref_f.abs() + 1e-3)
            bad = (diff > atol) & (rel > rtol)
            n_bad_t = bad.sum().item()
            max_abs_t = diff.max().item()
            max_rel_t = rel.max().item()
            diagnostics.append(
                (name, max_abs_t, max_rel_t, n_bad_t, ref_f.abs().max().item())
            )
            if n_bad_t > 0:
                ok = False

        # All-reduce ok across ranks so all ranks see the same outcome.
        ok_t = torch.tensor([1 if ok else 0], device=device, dtype=torch.int32)
        torch_dist.all_reduce(ok_t, op=torch_dist.ReduceOp.MIN)
        all_ok = ok_t.item() == 1

        tag = "PASS" if all_ok else "FAIL"
        rank_zero_print(f"[validate] iter {step:3d} seq={seq}: {tag}")
        if not all_ok:
            for name, max_abs_t, max_rel_t, n_bad_t, ref_max in diagnostics:
                marker = "  " if n_bad_t == 0 else "!!"
                rank_zero_print(
                    f"  {marker} {name:20s}  max_abs={max_abs_t:.4g} "
                    f"max_rel={max_rel_t:.4g} n_bad={n_bad_t} "
                    f"(ref_max={ref_max:.4g}) (this rank)"
                )
            fail_count += 1

    barrier(group)
    if rank == 0:
        if fail_count == 0:
            print(f"[validate] ALL {args.n_iter} ITERS OK (fwd + bwd)", flush=True)
        else:
            print(
                f"[validate] {fail_count} / {args.n_iter} iters FAILED — "
                "cross-layer race or correctness regression suspected",
                flush=True,
            )

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
