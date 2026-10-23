"""Expert dropout: inverted dropout on the expert hidden activation (the SwiGLU
postact between W1 and W2), as in Fedus et al. 2022 / ST-MoE.

The mask is never stored. It is a counter-based hash of
``(seed, pool row, postact column)``, regenerated in registers by kernel A's
forward epilogue and kernel_y_bwd's epilogue. Pool rows are a deterministic
function of the routing (dispatch Pass A), so a whole-block activation
checkpoint recompute reproduces the mask as long as it replays the seed:
``draw_seed`` pulls it from torch's CPU RNG, which ``torch.utils.checkpoint``
restores before recomputing.

``keep_scale_ref`` is the bit-exact torch mirror of the device hash, for tests
and unfused references.
"""

import cutlass
import torch
import torch.distributed as dist
from cutlass import Float32, Int32, Uint32, const_expr
from quack.epilogue.ops import Scalar

# Odd multiplicative constants (golden ratio / murmur3 fmix32).
_ROW_MUL = 0x9E3779B1
_COL_MUL = 0x85EBCA77
_FMIX_A = 0x85EBCA6B
_FMIX_B = 0xC2B2AE35
_U32 = 0xFFFFFFFF


def keep_threshold(p: float) -> int:
    """A unit is kept iff its uint32 hash >= this, i.e. with probability 1 - p."""
    if not 0.0 <= p < 1.0:
        raise ValueError(f"expert dropout p must be in [0, 1); got {p}")
    return min(round(p * 2**32), _U32)


def draw_seed() -> int:
    """Per-forward Int32 seed from torch's CPU RNG, decorrelated across ranks."""
    seed = int(torch.randint(0, 2**31, (), dtype=torch.int64))
    rank = dist.get_rank() if dist.is_initialized() else 0
    return ((seed ^ (rank * _ROW_MUL)) & _U32) >> 1


class DropoutSeed(Scalar):
    """Int32 seed plus each register's global ``(row, col)`` in the output tile.

    ``begin_loop`` returns ``(seed, tDcD, row0, col0)``: ``tDcD[i]`` is the
    CTA-tile coordinate of ``tRS_rD[i]`` and ``(row0, col0)`` the tile origin in
    the flat pool (varlen_m: ``cu_seqlens_m[expert] + pid_m * tile_M``)."""

    fn_port = None

    def __init__(self, name: str = "dropout_seed"):
        super().__init__(name, dtype=Int32)

    @cutlass.cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        seed = Scalar.begin(self, gemm, param, smem_tensor, ctx)
        tDcD = ctx.partition_for_epilogue_fn(
            cutlass.cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
        )
        row0 = ctx.tile_coord_mnkl[0] * ctx.tile_M
        if const_expr(ctx.varlen_manager.varlen_m):
            row0 = row0 + ctx.varlen_manager.params.cu_seqlens_m[ctx.batch_idx]
        return (seed, tDcD, row0, ctx.tile_coord_mnkl[1] * ctx.tile_N)

    @cutlass.cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        seed, tDcD, row0, col0 = state
        return (seed, tDcD[None, None, None, epi_coord[0], epi_coord[1]], row0, col0)


@cutlass.cute.jit
def keep_scale(seed, row, col, threshold, scale):
    """``scale`` if unit ``(row, col)`` is kept, else 0.

    The row mix is shared by every element of a row (CSE'd); the per-element
    part is one murmur3-style finalizer minus its last xorshift, which cannot
    change the high bits the threshold compare reads."""
    h = (Uint32(row) * Uint32(_ROW_MUL)) ^ Uint32(seed)
    h = (h ^ (h >> 16)) * Uint32(_FMIX_A)
    h = (h ^ (h >> 13)) * Uint32(_FMIX_B)
    h = h ^ (h >> 16)
    h = h ^ (Uint32(col) * Uint32(_COL_MUL))
    h = (h ^ (h >> 16)) * Uint32(_FMIX_A)
    h = (h ^ (h >> 13)) * Uint32(_FMIX_B)
    return Float32(scale) if h >= Uint32(threshold) else Float32(0.0)


def _mul32(a: torch.Tensor, c: int) -> torch.Tensor:
    # Split the constant so int64 partial products never overflow.
    return (a * (c & 0xFFFF) + (((a * (c >> 16)) & 0xFFFF) << 16)) & _U32


def keep_scale_ref(seed: int, rows: torch.Tensor, cols: torch.Tensor, p: float) -> torch.Tensor:
    """fp32 ``[len(rows), len(cols)]`` of ``1/(1-p)`` (kept) or 0, bit-matching
    ``keep_scale``."""
    rows = rows.to(torch.int64)[:, None]
    cols = cols.to(torch.int64)[None, :]
    h = _mul32(rows, _ROW_MUL) ^ (seed & _U32)
    h = _mul32(h ^ (h >> 16), _FMIX_A)
    h = _mul32(h ^ (h >> 13), _FMIX_B)
    h = h ^ (h >> 16)
    h = h ^ _mul32(cols, _COL_MUL)
    h = _mul32(h ^ (h >> 16), _FMIX_A)
    h = _mul32(h ^ (h >> 13), _FMIX_B)
    scale = torch.tensor(1.0 / (1.0 - p), dtype=torch.float32, device=rows.device)
    return torch.where(h >= keep_threshold(p), scale, torch.zeros_like(scale))
