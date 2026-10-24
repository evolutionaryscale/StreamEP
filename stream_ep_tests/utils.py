import os

import torch

# The 82ba5b production MoE shape (82B total / 4.3B active): d_model H, per-expert
# intermediate I, E experts, top-K routing, tokens per rank. Shared by the
# production-shape tests and the stream_ep_bench harnesses.
H = 3072
I = 768
NUM_EXPERTS = 256
SEQ_LEN_PER_RANK = 8192
TOPK = 8
DTYPE = torch.bfloat16


def report(name: str, ratio: float) -> None:
    """With ``REPORT_ERR=1`` (and ``pytest -s``), print each check's worst error
    ratio: the data tolerances are calibrated from. Show every rank's output
    (torchrun without ``--local-ranks-filter``) to see the worst over ranks."""
    if os.environ.get("REPORT_ERR"):
        print(f"ERRRATIO {name} {ratio:.3e}", flush=True)


def assert_rows_close(actual: torch.Tensor, expected: torch.Tensor, rtol: float, name: str) -> None:
    """Every row's error is small relative to that row: ``||a_r - e_r|| <= rtol * ||e_r||``.

    Rows are the first dimension (tokens, or experts for weight gradients).
    Normalizing per row means one wrong token or expert fails even when its
    values are small next to the rest of the tensor. A row whose expected value
    is zero must come out (near) zero."""
    assert actual.shape == expected.shape, f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert torch.isfinite(actual).all(), f"{name}: non-finite values"
    a, e = actual.float().flatten(1), expected.float().flatten(1)
    err = (a - e).norm(dim=1)
    ref = e.norm(dim=1)
    ratio = err / ref.clamp(min=1e-6 * ref.max().clamp(min=1e-30))
    worst = int(ratio.argmax())
    report(name, ratio[worst].item())
    assert ratio[worst] <= rtol, f"{name}: row {worst} relative error {ratio[worst]:.3e} > {rtol} (row norm {ref[worst]:.3e})"


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of bf16 values at magnitude ``|x|`` (0 where x is 0)."""
    exponent = torch.floor(torch.log2(x.abs().clamp(min=torch.finfo(torch.float32).tiny)))
    return torch.where(x == 0, 0.0, torch.exp2(exponent - 7))


def assert_sums_close(actual: torch.Tensor, expected: torch.Tensor, abs_sum: torch.Tensor, ulps: float, name: str) -> None:
    """A bf16 reduction against its fp32 sum: each element within ``ulps`` bf16
    ulps of the sum of the absolute values of its terms, which bounds every
    partial sum the reduction may round."""
    assert actual.shape == expected.shape, f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    err = (actual.float() - expected.float()).abs()
    bound = ulps * bf16_ulp(abs_sum.float())
    excess = torch.where(bound > 0, err / bound.clamp(min=1e-30), torch.where(err > 0, torch.inf, 0.0))
    report(name, excess.max().item() * ulps)
    bad = (err > bound).nonzero()
    assert bad.numel() == 0, f"{name}: {bad.shape[0]} elements off by more than {ulps} ulp, first at {bad[0].tolist()}"
