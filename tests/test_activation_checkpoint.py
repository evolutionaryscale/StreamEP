"""Gradient-equivalence test for the ``activation_checkpoint`` flag.

``stream_moe_func(..., activation_checkpoint=True)`` skips saving ``preact_a``
in forward and recomputes it from ``pool @ w1_local`` in backward with a plain
grouped-M GEMM (``quack.gemm`` with ``cu_seqlens_m``). Because that is a
DIFFERENT kernel than forward's gated kernel A, the reconstructed ``preact_a``
matches forward's within bf16 GEMM-accumulation tolerance (not bit-identical) —
same math, different tile/accumulation order. Every downstream gradient must
therefore match the non-checkpointed path within that tolerance.

What "match" means here has two components: (1) the streaming bwd accumulates
``dL_dweight`` (fp32 atomic-add), ``dL_dx_per_r`` (bf16x2 atomic-add) and the
fwd kernel-Y scatter in non-deterministic order, so two runs with the SAME flag
already differ at the floating-point-reassociation level (`markdowns/design.md`
§Determinism: pool *placement* is deterministic, atomic-add *order* is not);
(2) the recompute's different-kernel GEMM adds a bf16-accumulation difference on
top, which scales with tensor magnitude. So a bit-exact ``torch.equal`` would
fail even off-vs-off. We measure the off-vs-off run-to-run noise floor, allow a
bf16-relative term, and assert the off-vs-ON difference is within that bound —
a real recompute bug (wrong preact) blows past it by orders of magnitude.

The recompute lives entirely in the compute kernels, which are identical
intranode vs internode (`design.md` §Internode-specific notes), so 1-node
coverage exercises the full feature. Driver convention matches the rest of
the streaming tests:

    torchrun --nproc_per_node=8 StreamEP/tests/test_activation_checkpoint.py
    torchrun --nproc_per_node=2 StreamEP/tests/test_activation_checkpoint.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from stream_ep import Buffer
from stream_ep.stream_moe.stream_moe import make_streams, stream_moe_func
from utils import cleanup_dist, make_inputs


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    # Small but kernel-valid shapes (2I % tile_n and I % tile_n hold for the
    # auto-picked tiles at I=256/H=512; see _pick_tile_config). randn inputs so
    # gradients are non-degenerate.
    num_experts = 64
    num_local_experts = num_experts // world_size
    num_topk = 4
    num_tokens = 1024
    H = 512
    I = 256

    hidden_bytes = H * 2
    nvl_bytes = rdma_bytes = 0
    for cfg in (Buffer.get_dispatch_config(world_size), Buffer.get_combine_config(world_size)):
        nvl_bytes = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, world_size), nvl_bytes)
        rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, world_size), rdma_bytes)
    buf = Buffer(group, nvl_bytes, rdma_bytes)

    # Identical, deterministic inputs reused across all three runs. Same routing
    # ⟹ bit-identical pool placement ⟹ the only off-vs-on difference is the
    # atomic-reassociation noise (which off-vs-off also has).
    x0, topk_idx, topk_weights0, is_token_in_rank = make_inputs(
        num_tokens, H, num_topk, num_experts, world_size, rank, device,
        x_kind="randn",
    )
    g = torch.Generator(device=device).manual_seed(2024 + rank)
    w1_local = (torch.randn(num_local_experts, 2 * I, H, dtype=torch.bfloat16,
                            device=device, generator=g) * 0.02).contiguous()
    w2_local = (torch.randn(num_local_experts, H, I, dtype=torch.bfloat16,
                            device=device, generator=g) * 0.02).contiguous()

    streams = make_streams()

    def run(activation_checkpoint: bool):
        x = x0.clone().requires_grad_(True)
        topk_weights = topk_weights0.clone().requires_grad_(True)
        w1 = w1_local.clone().requires_grad_(True)
        w2 = w2_local.clone().requires_grad_(True)
        out = stream_moe_func(
            buf, x, topk_idx, topk_weights, is_token_in_rank, w1, w2,
            streams=streams, num_experts=num_experts,
            activation_checkpoint=activation_checkpoint,
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

    # Two baseline runs (noise floor) + one recompute run.
    base_a = run(False)
    base_b = run(False)
    recomp = run(True)

    def max_abs_diff(p, q):
        return (p - q).abs().max().item()

    # Sanity: no NaN/Inf anywhere.
    for name, t in recomp.items():
        assert torch.isfinite(t).all(), f"recompute produced non-finite {name} (rank={rank})"

    # Shapes must match the non-checkpointed path exactly.
    for k in recomp:
        assert recomp[k].shape == base_a[k].shape, (
            f"{k} shape changed under activation_checkpoint: "
            f"{base_a[k].shape} vs {recomp[k].shape}"
        )

    # Per-tensor: off-vs-ON diff must be within max(atomic-noise floor,
    # bf16-GEMM tolerance). ``slack`` lets the recompute sit at a different
    # (still valid) point in the atomic-reassociation distribution; ``rtol``
    # covers the bf16 accumulation-order difference of the recompute's separate
    # grouped-M GEMM kernel — which scales with tensor magnitude, so a fixed
    # atol alone would get tight at production shapes. A real bug (wrong
    # preact_a) blows past both by orders of magnitude / O(1) relative.
    slack = 4.0
    atol = 1e-3   # absolute floor for tensors whose values + noise are ~0
    rtol = 2e-2   # bf16 GEMM-accumulation tolerance (separate recompute kernel)
    failures = []
    report = []
    for k in ("out", "dx", "dtopk", "dw1", "dw2"):
        noise = max_abs_diff(base_a[k], base_b[k])
        test = max_abs_diff(base_a[k], recomp[k])
        scale = base_a[k].abs().max().item()
        bound = max(slack * noise, atol, rtol * scale)
        report.append(
            f"{k}: off-vs-on={test:.3e}  off-vs-off(noise)={noise:.3e}  "
            f"rtol*scale={rtol * scale:.3e}  bound={bound:.3e}"
        )
        if test > bound:
            failures.append(
                f"{k}: off-vs-on diff {test:.4e} exceeds bound {bound:.4e} "
                f"(noise {noise:.4e}, rtol*scale {rtol * scale:.4e}) "
                f"— recompute is NOT equivalent"
            )

    if rank == 0:
        print(f"[rank0 world={world_size}] activation_checkpoint grad-equivalence:")
        for line in report:
            print("  " + line)

    assert not failures, "activation_checkpoint mismatch:\n" + "\n".join(failures)

    if rank == 0:
        print(f"PASS: activation_checkpoint grads within atomic-noise floor "
              f"(world={world_size}, T={num_tokens}, H={H}, I={I}, E={num_experts}, K={num_topk})")

    cleanup_dist()


if __name__ == "__main__":
    main()
