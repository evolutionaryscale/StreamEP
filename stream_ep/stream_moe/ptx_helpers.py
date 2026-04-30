"""PTX wrappers for cross-stream signaling used by streaming-MoE kernel A.

These three primitives implement the system-scope acquire/release/fence pair
that the linear-claim scheduler uses to spin on `tile_ready[tile_id]` (set by
DeepEP dispatch's Pass 2 on a different stream).

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
