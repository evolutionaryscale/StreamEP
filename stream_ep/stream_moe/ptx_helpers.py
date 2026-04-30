"""PTX wrappers for the streaming-MoE pipeline.

Two groups:

1. **Cross-stream signaling** (kernel A scheduler) — system-scope acquire /
   release / fence pair so the linear-claim scheduler can spin on
   `tile_ready[tile_id]` written by DeepEP dispatch's Pass 2 on a different
   stream.

2. **Vectorized atomic-add** (kernel Y AtomicScatterStore epilogue) — packed
   bf16x2 / f16x2 and scalar f32 reductions into the local-rank output buffer
   `o[T_recv, H]`. Intra-GPU scope is enough; cross-stream visibility for the
   eventual combine sender is carried by `compute_done_per_token[r]`'s separate
   release-store.

Lifted out of `quack/utils.py` so the streaming code is self-contained in
evolutionaryscale and doesn't require a quack fork that adds these wrappers.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def st_release_sys_global(
    gmem_ptr: cute.Pointer, val: cutlass.Int64, *, loc=None, ip=None
) -> None:
    """Release-store an int64 value to a global pointer with .sys scope."""
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr_i64, cutlass.Int64(val).ir_value(loc=loc, ip=ip)],
        "st.release.sys.global.b64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def ld_acquire_sys_global(
    gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> cutlass.Int64:
    """Acquire-load an int64 value from a global pointer with .sys scope."""
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [gmem_ptr_i64],
            "ld.acquire.sys.global.b64 $0, [$1];",
            "=l,l",
            has_side_effects=True,
            is_align_stack=False,
        )
    )


@dsl_user_op
def threadfence_system(*, loc=None, ip=None) -> None:
    """System-scope memory fence (membar.sys)."""
    llvm.inline_asm(
        None, [], "membar.sys;", "", has_side_effects=True, is_align_stack=False
    )


@dsl_user_op
def red_add_bf16x2(
    gmem_ptr: cute.Pointer, packed: cutlass.Int32, *, loc=None, ip=None
) -> None:
    """Packed bf16x2 atomic-add into 4-byte-aligned global memory.

    `packed` carries two bf16 lanes in a single 32-bit register (lo lane in
    bits 0-15, hi lane in bits 16-31). Caller is responsible for the pack —
    typically via `cute.recast_tensor(rD_bf16, Int32)` on a contiguous-2 view
    of the register tensor.

    The pointer must be 4-byte aligned; the targeted bf16x2 pair is the
    `(2*j, 2*j+1)` adjacent elements at that address.

    Default `.gpu`-scope `.relaxed` semantics: kernel Y's atomic-scatter is
    intra-GPU, and the per-token completion signal (`compute_done_per_token`)
    carries the actual release/acquire to combine.

    PTX requires `.noftz` for `red.add.bf16x2` (denormals must NOT be flushed
    to zero — required even though SM90 hardware ignores the bit; ptxas rejects
    the instruction without it).
    """
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr_i64, cutlass.Int32(packed).ir_value(loc=loc, ip=ip)],
        "red.global.add.noftz.bf16x2 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def red_add_f16x2(
    gmem_ptr: cute.Pointer, packed: cutlass.Int32, *, loc=None, ip=None
) -> None:
    """Packed f16x2 atomic-add. Same packing/alignment contract as
    `red_add_bf16x2`. ``.noftz`` required by ptxas (same as bf16x2)."""
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr_i64, cutlass.Int32(packed).ir_value(loc=loc, ip=ip)],
        "red.global.add.noftz.f16x2 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def red_add_f32(
    gmem_ptr: cute.Pointer, val: cutlass.Float32, *, loc=None, ip=None
) -> None:
    """Scalar f32 atomic-add. Used when `o[r, :]` is fp32 (e.g. accumulator
    debug path) or for the trailing odd-H tail when H is not a multiple of 2."""
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr_i64, cutlass.Float32(val).ir_value(loc=loc, ip=ip)],
        "red.global.add.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
    )
