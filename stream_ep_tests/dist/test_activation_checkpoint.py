"""Gradient equivalence of ``activation_checkpoint_level`` 1 and 2 against 0.

Level 1 skips saving ``preact_a`` and recomputes it in backward from
``pool @ w1_local`` with a plain grouped-M GEMM; level 2 additionally saves
only the unexpanded recv tokens and re-expands ``pool`` in backward. The
recompute uses a different kernel than forward's gated kernel A, so
``preact_a`` matches within bf16 accumulation tolerance, not bit-exactly.

Two runs with the same level already differ: the backward accumulates
``dL_dweight`` / ``dL_dx_per_r`` and forward scatters ``y`` with atomics in
nondeterministic order. So the test measures the level-0 run-to-run noise
floor, allows a bf16-relative term on top, and asserts each checkpointed
level stays within that bound; a wrong recompute misses it by orders of
magnitude.
"""

import os

import torch
import torch.distributed as dist
from stream_ep import Buffer
from stream_ep.stream_moe.stream_moe import make_streams, stream_moe_func
from stream_ep_tests.utils import cleanup_dist, make_inputs


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    # Small but kernel-valid shapes (2I % tile_n and I % tile_n hold for the
    # auto-picked tiles at I=256/H=512; see default_tile_config). randn inputs so
    # gradients are non-degenerate.
    num_experts = 64
    num_local_experts = num_experts // world_size
    num_topk = 4
    num_tokens = 1024
    H = 512
    I = 256

    hidden_bytes = H * 2
    buf = Buffer.sized_for(group, hidden_bytes)

    # Identical, deterministic inputs reused across all three runs. Same routing
    # ⟹ bit-identical pool placement ⟹ the only off-vs-on difference is the
    # atomic-reassociation noise (which off-vs-off also has).
    x0, topk_idx, topk_weights0, is_token_in_rank = make_inputs(
        num_tokens,
        H,
        num_topk,
        num_experts,
        world_size,
        rank,
        device,
        x_kind="randn",
    )
    g = torch.Generator(device=device).manual_seed(2024 + rank)
    w1_local = (torch.randn(num_local_experts, 2 * I, H, dtype=torch.bfloat16, device=device, generator=g) * 0.02).contiguous()
    w2_local = (torch.randn(num_local_experts, H, I, dtype=torch.bfloat16, device=device, generator=g) * 0.02).contiguous()

    streams = make_streams()

    def run(level: int):
        x = x0.clone().requires_grad_(True)
        topk_weights = topk_weights0.clone().requires_grad_(True)
        w1 = w1_local.clone().requires_grad_(True)
        w2 = w2_local.clone().requires_grad_(True)
        out = stream_moe_func(
            buf,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1,
            w2,
            streams=streams,
            num_experts=num_experts,
            activation_checkpoint_level=level,
        )
        out.sum().backward()
        torch.cuda.synchronize()
        return {
            "out": out.detach().float(),
            "dx": x.grad.float(),
            "dtopk": topk_weights.grad.float(),
            "dw1": w1.grad.float(),
            "dw2": w2.grad.float(),
        }

    # Two baseline runs (level 0, noise floor) + recompute runs at level 1
    # (recompute preact_a) and level 2 (compressed recv-token ckpt: re-expand
    # pool, then recompute preact_a). Both must match level 0 within tolerance.
    base_a = run(0)
    base_b = run(0)
    recomps = {1: run(1), 2: run(2)}

    def max_abs_diff(p, q):
        return (p - q).abs().max().item()

    # Per-tensor: off-vs-ON diff must be within max(atomic-noise floor,
    # bf16-GEMM tolerance). ``slack`` lets the recompute sit at a different
    # (still valid) point in the atomic-reassociation distribution; ``rtol``
    # covers the bf16 accumulation-order difference of the recompute's separate
    # grouped-M GEMM kernel — which scales with tensor magnitude, so a fixed
    # atol alone would get tight at production shapes. A real bug (wrong
    # preact_a, or a wrong pool re-expansion at level 2) blows past both by
    # orders of magnitude / O(1) relative.
    slack = 4.0
    atol = 1e-3  # absolute floor for tensors whose values + noise are ~0
    rtol = 2e-2  # bf16 GEMM-accumulation tolerance (separate recompute kernel)
    failures = []
    report = []
    for level, recomp in recomps.items():
        # Sanity: no NaN/Inf anywhere.
        for name, t in recomp.items():
            assert torch.isfinite(t).all(), f"level {level} produced non-finite {name} (rank={rank})"
        # Shapes must match the non-checkpointed path exactly.
        for k in recomp:
            assert recomp[k].shape == base_a[k].shape, f"level {level}: {k} shape changed: {base_a[k].shape} vs {recomp[k].shape}"
        for k in ("out", "dx", "dtopk", "dw1", "dw2"):
            noise = max_abs_diff(base_a[k], base_b[k])
            test = max_abs_diff(base_a[k], recomp[k])
            scale = base_a[k].abs().max().item()
            bound = max(slack * noise, atol, rtol * scale)
            report.append(
                f"L{level} {k}: off-vs-on={test:.3e}  off-vs-off(noise)={noise:.3e}  rtol*scale={rtol * scale:.3e}  bound={bound:.3e}"
            )
            if test > bound:
                failures.append(
                    f"level {level} {k}: off-vs-on diff {test:.4e} exceeds bound "
                    f"{bound:.4e} (noise {noise:.4e}, rtol*scale {rtol * scale:.4e}) "
                    f"— recompute is NOT equivalent"
                )

    if rank == 0:
        print(f"[rank0 world={world_size}] activation_checkpoint_level grad-equivalence:")
        for line in report:
            print("  " + line)

    assert not failures, "activation_checkpoint_level mismatch:\n" + "\n".join(failures)

    if rank == 0:
        print(
            f"PASS: activation_checkpoint grads within atomic-noise floor "
            f"(world={world_size}, T={num_tokens}, H={H}, I={I}, E={num_experts}, K={num_topk})"
        )

    cleanup_dist()


if __name__ == "__main__":
    main()
