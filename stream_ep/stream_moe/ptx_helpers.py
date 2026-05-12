"""PTX wrappers for the streaming-MoE pipeline.

Three groups:

1. **Cross-stream signaling** — acquire / release / fence wrappers used to
   pair producers and consumers on different streams. Two scopes:
   `_gpu_global` for intra-GPU stamps (`tile_ready`, `a_ready`,
   `y_done_per_token` and their bwd analogs — producer and consumer on the
   same device, different streams); `_sys_global` for cross-rank stamps
   (channel tails over NVLink / RDMA). `threadfence_system` carries
   cross-rank visibility on the producer side where needed.

2. **bf16-pair packing** — `pack_bf16x2` reads two bf16 from SMEM and packs
   them into a single Int32 (bf16x2 layout). Sidesteps the
   `cute.recast_tensor`-on-swizzled-SMEM trap (recast does NOT preserve
   bf16x2 pair-adjacency under swizzle).

3. **Vectorized predicated atomic-add** (kernel Y AtomicScatterStore
   epilogue) — `red_add_bf16x2_v4_pred` issues a 16-byte / 8-bf16
   reduction with PTX-level predication so out-of-range / padding lanes
   skip the atomic entirely (no HBM op, no need for a trash-row sink).
   Intra-GPU scope is enough; cross-stream visibility for the eventual
   combine sender is carried by `y_done_per_token[r]`'s separate
   release-store.

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
    """Release-store an int64 value to a global pointer with .sys scope.

    The ``~{memory}`` clobber tells the LLVM optimizer this asm reads/writes
    arbitrary memory, preventing it from sinking earlier writes to AFTER
    this release-store. Without it the release fence is correct in PTX but
    the source-level ordering can be silently broken by the optimizer.
    Zero runtime cost — purely a compiler-barrier annotation.
    """
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr_i64, cutlass.Int64(val).ir_value(loc=loc, ip=ip)],
        "st.release.sys.global.b64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def st_release_gpu_global(
    gmem_ptr: cute.Pointer, val: cutlass.Int64, *, loc=None, ip=None
) -> None:
    """Release-store an int64 value to a global pointer with .gpu scope.

    Use for intra-GPU producer→consumer stamps (different streams, same
    device): ``tile_ready`` / ``a_ready`` / ``y_done_per_token`` and their
    bwd analogs. Cheaper than ``_sys_global``: stays in L2, no NVLink
    coherence traversal. Cross-rank stamps must keep ``_sys_global``.

    Memory clobber semantics identical to ``st_release_sys_global``.
    """
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr_i64, cutlass.Int64(val).ir_value(loc=loc, ip=ip)],
        "st.release.gpu.global.b64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def ld_acquire_sys_global(
    gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> cutlass.Int64:
    """Acquire-load an int64 value from a global pointer with .sys scope.

    The ``~{memory}`` clobber prevents the LLVM optimizer from hoisting
    surrounding loads ABOVE this acquire-load. Without it, source-level
    ``val = pool[slot]`` after a spin-on-acquire can be reordered to
    fetch ``pool[slot]`` once before the spin even starts, defeating the
    synchronizes-with edge. Zero runtime cost.
    """
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [gmem_ptr_i64],
            "ld.acquire.sys.global.b64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
        )
    )


@dsl_user_op
def ld_acquire_gpu_global(
    gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> cutlass.Int64:
    """Acquire-load an int64 value from a global pointer with .gpu scope.

    Pair with ``st_release_gpu_global`` for intra-GPU producer→consumer
    stamps. Memory clobber semantics identical to ``ld_acquire_sys_global``.
    """
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [gmem_ptr_i64],
            "ld.acquire.gpu.global.b64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
        )
    )


@dsl_user_op
def ld_acquire_gpu_global_i32(
    gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> cutlass.Int32:
    """Acquire-load an int32 value from a global pointer with .gpu scope.

    Used by the tile scheduler's spin on
    `pool_arrival_count[tile_id] == pool_arrival_target[tile_id]` — the
    acquire pairs with dispatch's Pass 2 `red.release.gpu.global.add.s32`
    so pool writes are visible once the count hits target.
    """
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [gmem_ptr_i64],
            "ld.acquire.gpu.global.b32 $0, [$1];",
            "=r,l,~{memory}",
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
def red_add_bf16x2_v4_pred(
    gmem_ptr: cute.Pointer,
    p0: cutlass.Int32,
    p1: cutlass.Int32,
    p2: cutlass.Int32,
    p3: cutlass.Int32,
    pred: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """16-byte vector predicated bf16x2 atomic-add (8 bf16 lanes per call).

    Issues ``@%p red.global.add.noftz.v4.bf16x2 [ptr], {p0, p1, p2, p3};``
    where ``%p`` is set from `pred != 0`. When pred is false, the
    instruction is skipped entirely at issue time — no memory op, no
    atomic side-effect — so the address is allowed to be any valid 16-byte-
    aligned pointer (caller can clamp to a safe in-bounds address without
    needing a trash row).

    PTX-level predicated execution is **not** the same as a divergent C-level
    branch around `red`: the latter drops atomics non-deterministically when
    threads in a warp diverge around the instruction. Predication is per-
    instruction at issue time and stress-tested to drop zero atomics across
    32 iters × 1M divergent-predicate threads on H100.

    Each Int32 payload is one bf16x2 packed pair (lo bits 0-15, hi bits
    16-31). Pointer must be **16-byte aligned**; targets the contiguous
    8 bf16 elements (p0 → bf16[0..1], p1 → bf16[2..3], p2 → bf16[4..5],
    p3 → bf16[6..7]).

    Default `.gpu`-scope `.relaxed` semantics: kernel Y's atomic-scatter is
    intra-GPU; cross-stream visibility for the eventual combine sender is
    carried by `y_done_per_token[r]`'s separate release-store.

    PTX requires `.noftz` (denormals must NOT be flushed; ptxas rejects the
    instruction without it).
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
            cutlass.Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n\t"
            ".reg .pred %p1;\n\t"
            "setp.ne.b32 %p1, $5, 0;\n\t"
            "@%p1 red.global.add.noftz.v4.bf16x2 [$0], {$1, $2, $3, $4};\n\t"
            "}"
        ),
        "l,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def red_add_f32(
    gmem_ptr: cute.Pointer, val: cutlass.Float32, *, loc=None, ip=None
) -> None:
    """Single fp32 atomic-add via ``red.global.add.f32 [ptr], val;``.

    Used by kernel_y_bwd's epilogue to accumulate per-pid_n stripe partials
    into a flat per-slot ``dL_dweight[slot]`` fp32 buffer. ~256K total
    atomics per layer at production (TK_padded × num_pid_n_y_bwd ≈ 32K × 8),
    all hot in L2 — throughput-trivial.

    Default ``.gpu``-scope ``.relaxed`` semantics: the cross-stream
    visibility to combine_grads's sender is carried by the per-tile
    ``bwd_a_ready`` release-store's ``threadfence_system`` (in the
    ``TileReadyRelease`` epilogue end), not by this atomic itself.
    """
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr_i64, cutlass.Float32(val).ir_value(loc=loc, ip=ip)],
        "red.global.add.f32 [$0], $1;",
        "l,f",
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
