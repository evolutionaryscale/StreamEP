"""Triton kernels for activation_checkpoint_level=2 pool <-> x_recv row
compaction / re-expansion (markdowns/act_ckpt_level2_plan.md sec 5).

One ``@triton.jit`` kernel, two ``GATHER`` specializations:
  * expand  (GATHER=True):  pool[slot]                = x_recv[pool_recv_token[slot]]  (0 if -1 pad)
  * compact (GATHER=False): x_recv[pool_recv_token[slot]] = pool[slot]                 (skip -1 pad)

Both are driven by ``pool_recv_token`` (slot -> recv-token id, -1 = padding), so
compact needs no per-token scan.

Compile-once for ALL lengths: the row count ``L`` (= TK_padded) is the grid
dimension, NOT a kernel argument, so it never enters triton's compile key — one
compile per direction handles every per-layer length. ``H`` is a runtime arg
(one variant per model). No autotune; fixed ``BLOCK_H`` (2-D grid: one program
per (slot, H-block), contiguous block load/store that triton auto-vectorizes).

int32 offsets: ``slot*H + h <= TK_padded*H < 2**31`` (asserted in the wrappers),
so no int64 widening. Compaction is a scatter-by-slot whose only write
collisions are the ``K_local`` slots of a token writing the *identical*
broadcast row to the same x_recv row — benign (no atomics, no tearing).
"""
import torch
import triton
import triton.language as tl

_INT32_MAX = 2**31


@triton.jit
def _index_copy_rows(
    out_ptr, in_ptr, idx_ptr, H,
    GATHER: tl.constexpr, BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)                       # slot (both directions)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h < H
    tok = tl.load(idx_ptr + row)                 # int32 slot -> recv-token, -1 = pad
    valid = tok >= 0
    tokc = tl.maximum(tok, 0)                     # clamp pad to row 0 (masked/zeroed out)
    if GATHER:                                   # expand: out[row] = in[tok], pad -> 0
        v = tl.load(in_ptr + tokc * H + h, mask=hmask, other=0.0)
        v = v * valid.to(v.dtype)                # pad row -> 0 (1.0/0.0 multiply, exact)
        tl.store(out_ptr + row * H + h, v, mask=hmask)
    else:                                        # compact: out[tok] = in[row], skip pad
        v = tl.load(in_ptr + row * H + h, mask=hmask)
        tl.store(out_ptr + tokc * H + h, v, mask=hmask & valid)


def _check(pool: torch.Tensor, x_recv: torch.Tensor, pool_recv_token: torch.Tensor):
    assert pool.is_contiguous() and x_recv.is_contiguous() and pool_recv_token.is_contiguous()
    TK_padded, H = pool.shape
    assert pool_recv_token.shape == (TK_padded,) and pool_recv_token.dtype == torch.int32, (
        f"pool_recv_token must be [{TK_padded}] int32, got {tuple(pool_recv_token.shape)} {pool_recv_token.dtype}")
    assert x_recv.shape[1] == H and x_recv.dtype == pool.dtype
    assert TK_padded * H < _INT32_MAX, (
        f"int32 offset overflow: TK_padded*H={TK_padded * H} >= 2**31; widen offsets to int64 in-kernel")
    return TK_padded, H


def compress_pool_to_recv(
    pool: torch.Tensor, pool_recv_token: torch.Tensor, out: torch.Tensor, BLOCK_H: int = 512
) -> None:
    """pool[TK_padded, H] -> out[T_recv, H], in place. Scatter-by-slot: every
    token's identical-broadcast slots write the same row, so the collision is
    benign. ``out`` is fully written (every recv-token has >=1 slot)."""
    TK_padded, H = _check(pool, out, pool_recv_token)
    grid = (TK_padded, triton.cdiv(H, BLOCK_H))
    _index_copy_rows[grid](out, pool, pool_recv_token, H, GATHER=False, BLOCK_H=BLOCK_H)


def reexpand_recv_to_pool(
    x_recv: torch.Tensor, pool_recv_token: torch.Tensor, out: torch.Tensor, BLOCK_H: int = 512
) -> None:
    """x_recv[T_recv, H] -> out[TK_padded, H], in place. Gather-by-slot; padding
    slots (-1) are zeroed (masked / cu_seqlens-excluded by every consumer)."""
    TK_padded, H = _check(out, x_recv, pool_recv_token)
    grid = (TK_padded, triton.cdiv(H, BLOCK_H))
    _index_copy_rows[grid](out, x_recv, pool_recv_token, H, GATHER=True, BLOCK_H=BLOCK_H)
