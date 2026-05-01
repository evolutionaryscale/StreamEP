"""PTX wrappers for the streaming-MoE pipeline.

Three groups:

1. **Cross-stream signaling** (kernel A scheduler) — system-scope acquire /
   release / fence pair so the linear-claim scheduler can spin on
   `tile_ready[tile_id]` written by DeepEP dispatch's Pass 2 on a different
   stream.

2. **bf16-pair packing** — `pack_bf16x2` reads two bf16 from SMEM and packs
   them into a single Int32 (bf16x2 layout). Sidesteps the
   `cute.recast_tensor`-on-swizzled-SMEM trap (recast does NOT preserve
   bf16x2 pair-adjacency under swizzle).

3. **Vectorized atomic-add** (kernel Y AtomicScatterStore epilogue) —
   `red_add_bf16x2_v4` issues a 16-byte / 8-bf16 reduction. Intra-GPU
   scope is enough; cross-stream visibility for the eventual combine sender
   is carried by `compute_done_per_token[r]`'s separate release-store.

Lifted out of `quack/utils.py` so the streaming code is self-contained in
evolutionaryscale and doesn't require a quack fork that adds these wrappers.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16
from cutlass._mlir.dialects import llvm, vector
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
def red_add_bf16x2_v4(
    gmem_ptr: cute.Pointer,
    p0: cutlass.Int32,
    p1: cutlass.Int32,
    p2: cutlass.Int32,
    p3: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """16-byte vector packed bf16x2 atomic-add (8 bf16 lanes per call).

    Issues ``red.global.add.noftz.v4.bf16x2 [ptr], {p0, p1, p2, p3};``.
    Each Int32 payload is one bf16x2 packed pair (lo bits 0-15, hi bits
    16-31). Pointer must be **16-byte aligned**; targets the contiguous
    8 bf16 elements (p0 → bf16[0..1], p1 → bf16[2..3], p2 → bf16[4..5],
    p3 → bf16[6..7]).

    Default `.gpu`-scope `.relaxed` semantics: kernel Y's atomic-scatter is
    intra-GPU; cross-stream visibility for the eventual combine sender is
    carried by `compute_done_per_token[r]`'s separate release-store.

    Verified on SM90 H100: ptxas accepts the encoding and the hardware
    executes it as 8 logically-atomic adds (functionally equivalent to four
    sequential `red.add.bf16x2`). Cuts atomic-op count 4× at the SMEM-staged
    scatter loop in kernel Y when `epi_tile_N >= 8`. PTX requires `.noftz`
    (denormals must NOT be flushed; ptxas rejects the instruction without it).
    """
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            gmem_ptr_i64,
            cutlass.Int32(p0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(p1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(p2).ir_value(loc=loc, ip=ip),
            cutlass.Int32(p3).ir_value(loc=loc, ip=ip),
        ],
        "red.global.add.noftz.v4.bf16x2 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def pack_bf16x2(lo: BFloat16, hi: BFloat16, *, loc=None, ip=None) -> cutlass.Int32:
    """Pack two bf16 values into a single Int32 (lo bits 0-15, hi bits 16-31).

    Used by kernel Y's atomic-scatter epilogue: bf16x2 atomic-add (`red.global.add.bf16x2`)
    requires the two bf16 lanes packed in one 32-bit register. Reading them
    individually from swizzled SMEM and packing via this helper is correct
    regardless of the SMEM swizzle (which can shuffle 4-byte adjacency
    relative to the original bf16 layout — `cute.recast_tensor(swizzled_smem,
    Int32)` does NOT preserve the bf16x2 pair-adjacency in general).
    """
    vec_bf16x2 = vector.from_elements(
        T.vector(2, T.bf16()),
        (lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)),
        loc=loc,
        ip=ip,
    )
    vec_i32x1 = vector.bitcast(T.vector(1, T.i32()), vec_bf16x2)
    return cutlass.Int32(
        vector.extract(
            vec_i32x1, dynamic_position=[], static_position=[0], loc=loc, ip=ip
        )
    )
