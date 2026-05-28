"""Streaming-MoE kernel A (CuTeDSL, SM90, pool layout).

Forward kernel A of the problem-tile streaming pipeline:
  * Persistent CTAs pull tiles via a count-vs-target spin
    (`pool_arrival_count[tile] == pool_arrival_target[tile]`).
  * For each claimed tile_id, the scheduler derives `expert_id` from a
    warp-cooperative ballot over `expert_pool_block_offset` and computes
    `pid_m = tile_id - expert_pool_block_offset[expert_id]`.
  * Standard varlen_m strided TMA load of `pool[tile_id * tile_M : ..., :]`
    (the row offset = `cu_seqlens_m[expert_id] + pid_m * tile_m` lands at
    the correct expert-major pool row by construction).
  * GEMM against W1[expert_id], SwiGLU register-resident epilogue, TMA-store
    the I-half post-activation to `postact_a[tile_id * tile_M : ..., :]`.

Kernel A's downstream consumer (kernel Y) runs on the SAME compute stream and
is FIFO-ordered after A: by the time Y issues its first instruction, A has
fully retired and its TMA stores are drained. No per-tile A→Y release/acquire
signal is needed.

Inherits the GEMM mainloop, SwiGLU epilogue, scheduler-warp + pipeline-state
machinery from `quack.gemm_act.GemmGatedSm90`. Streaming-specific behavior is
isolated to two overrides:
  (1) get_scheduler_class — return StreamingTileScheduler.
  (2) get_scheduler_arguments — build StreamingTileSchedulerArguments from
      pool-shape metadata.

The streaming scheduler uses the upstream 4-int sched payload
(pid_m, pid_n, batch_idx, is_valid) — no streaming-specific SMEM extension.
Kernel A's mainloop and postact path land at the right pool rows via
`cu_seqlens_m[batch_idx] + pid_m * tile_m` alone; tile_id is computed
locally inside the scheduler warp's queue-pull and used only for the
ready-spin and to derive expert_id/pid_m.
"""

from typing import Callable, NamedTuple, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from quack.activation import gate_fn_map
from quack.cache_utils import COMPILE_ONLY, jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    ParamsBase,
    get_device_capacity,
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.gemm_act import GemmGatedMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_tvm_ffi_utils import compile_gemm_kernel
from quack.rounding import RoundingMode
from quack.tile_scheduler import PersistenceMode
from quack.varlen_utils import VarlenArguments

from stream_ep.stream_moe.ptx_helpers import threadfence_system
from stream_ep.stream_moe.tile_scheduler import (
    StreamingTileScheduler,
    StreamingTileSchedulerArguments,
)


# ---------------------------------------------------------------------------
# Host-facing scheduler-options NamedTuple. Mirrors TileSchedulerOptions but
# carries the streaming-specific tensors/pointers that the scheduler needs.
# ---------------------------------------------------------------------------
@mlir_namedtuple
class StreamingTileSchedulerOptions(NamedTuple):
    max_active_clusters: Int32
    consumer_head: cute.Tensor  # [1] int32 — global linear claim counter
    # Per-tile ready spin source for kernel A's dispatch handoff. The
    # scheduler does `count[tile] == target[tile]` (count-vs-target). Dispatch's
    # metadata kernel fills `pool_arrival_target` with the per-tile firing
    # target; dispatch's Pass 2 release-adds into `pool_arrival_count`.
    pool_arrival_count: cute.Tensor   # [total_tiles] int32 — release-add destination
    pool_arrival_target: cute.Tensor  # [total_tiles] int32 — per-tile firing target
    expert_pool_block_offset: (
        cute.Tensor
    )  # [E_local + 1] int32 — pool-block prefix-sum. Source for the
    # warp-cooperative ballot lookup that retired per-claim `tile_id_to_expert`.
    total_tiles: Int32  # passed as scalar so get_grid_shape doesn't deref device tensor
    # Optional cross-stream launch-gate "started" flag (single int32 in
    # device memory). When supplied, the CTA that wins ``linear_idx == 0``
    # in ``_fetch_next_work_idx`` atomicAdd's this flag once. The host on
    # the consumer stream (typically communicate) issues
    # ``cuStreamBatchMemOp wait_value_geq`` against this flag before
    # launching combine_main / combine_grads_main, so combine's 80-CTA
    # sender grid can't grab SMs ahead of kernel_y / kernel_a_bwd's
    # 132-CTA grid. Pass ``None`` to disable (kernel_a / kernel_y_bwd
    # don't bump anything — only kernel_y / kernel_a_bwd own a flag).
    started_flag: Optional[cute.Tensor] = None  # [1] int32 device flag


# ---------------------------------------------------------------------------
# Streaming kernel A class.
# ---------------------------------------------------------------------------
class StreamingMoeA(GemmGatedMixin, GemmSm90):
    """Streaming-MoE kernel A: standard strided varlen_m GEMM + SwiGLU with
    queue-pull scheduler. Pool layout means kernel A uses the base GEMM
    mainloop's varlen_m path verbatim — no per-tile gather indirection.

    Kernel Y runs on the SAME compute stream and is FIFO-ordered after A
    fully retires — same-stream FIFO covers cross-stage visibility, no
    per-tile release/acquire is needed.
    """

    _epi_ops = GemmGatedMixin._epi_ops
    _epi_param_bases = (ParamsBase,)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mAuxOut: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops + _extra_param_fields by
    # ComposableEpiMixin.__init_subclass__.

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args: tuple,
        scheduler_args: StreamingTileSchedulerOptions,
        varlen_args: Optional[VarlenArguments],
        stream: cuda.CUstream,
        trace_ptr: Optional[Int64] = None,
    ):
        """Type-shim override so CuTeDSL accepts StreamingTileSchedulerOptions
        as the scheduler_args type (base annotation is TileSchedulerOptions).
        Body delegates to GemmSm90.__call__ unchanged.
        """
        from quack.gemm_sm90 import GemmSm90 as _GemmSm90Base

        _GemmSm90Base.__call__(
            self,
            mA,
            mB,
            mD,
            mC,
            epilogue_args,
            scheduler_args,
            varlen_args,
            stream,
            trace_ptr,
        )

    # -- scheduler hooks -----------------------------------------------------

    def get_scheduler_class(self, varlen_m: bool = False):
        return StreamingTileScheduler

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,  # pool: (TK_padded, H)
        mB: cute.Tensor,  # W1: (2I, H, E_local)
        mD: Optional[cute.Tensor],  # None (no D for streaming kernel A)
        scheduler_args: StreamingTileSchedulerOptions,
        varlen_args: VarlenArguments,
        epilogue_args,
    ):
        # mB shape is (n=2I, k=H, l=E_local); n-dim tile count = ceil(2I / tile_N).
        num_pid_n = cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1])
        E_local = cute.size(mB, mode=[2])
        return StreamingTileSchedulerArguments(
            problem_shape_ntile_mnl=(None, num_pid_n, E_local),
            consumer_head=scheduler_args.consumer_head,
            arrival_count=scheduler_args.pool_arrival_count,
            arrival_target=scheduler_args.pool_arrival_target,
            expert_pool_block_offset=scheduler_args.expert_pool_block_offset,
            total_tiles=scheduler_args.total_tiles,
            tile_shape_mn=self.cta_tile_shape_mnk[:2],
            cluster_shape_mnk=self.cluster_shape_mnk,
            scheduler_warp_id=self.ab_load_warp_id,
            persistence_mode=PersistenceMode.DYNAMIC,
            started_flag=scheduler_args.started_flag,
        )

    # postact destination: inherited GemmGatedSm90.epi_setup_postact uses
    # `varlen_manager.offset_batch_epi(mAuxOut, batch_idx)` (shift by
    # cu_seqlens_m[batch_idx]) + `local_tile((pid_m, pid_n))`. The combined
    # row offset is `cu_seqlens_m[batch_idx] + pid_m * tile_m =
    # expert_pool_block_offset[e] * tile_m + (tile_id - expert_pool_block_offset[e])
    # * tile_m = tile_id * tile_m`, which lands postact_a's per-tile slab — no
    # streaming-specific override needed.


# ---------------------------------------------------------------------------
# JIT compile factory.
# ---------------------------------------------------------------------------
@jit_cache
def _compile_streaming_moe_a(
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    postact_dtype: Type[cutlass.Numeric],
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    cluster_n: int,
    activation: str,
    device_capacity,
    *,
    store_preact: bool = False,
):
    assert device_capacity[0] == 9, "Streaming MoE kernel A is SM90-only for now"
    assert activation in gate_fn_map, f"Need a gated activation; got {activation}"

    H_sym = cute.sym_int()
    I2_sym = cute.sym_int()
    I_sym = cute.sym_int()
    E_sym = cute.sym_int()
    TK_padded_sym = cute.sym_int()
    Mflat_sym = cute.sym_int()  # total_tiles * tile_m, in postact's M dim
    total_tiles_sym = cute.sym_int()
    cu_seqlens_len_sym = cute.sym_int()  # E_local + 1 at runtime

    # A: pool (TK_padded, H), k-major (H is contiguous).
    mA = fake_tensor(a_dtype, (TK_padded_sym, H_sym), leading_dim=1, divisibility=8)
    # B: W1 (2I, H, E_local), k-major per expert (H contiguous), batch dim = E_local.
    mB = fake_tensor(b_dtype, (I2_sym, H_sym, E_sym), leading_dim=1, divisibility=8)
    # mD: optional pre-SwiGLU output. When `store_preact=True`, kernel A's
    # standard mD TMA-store path (inherited from GemmDefaultEpiMixin via
    # GemmGatedMixin) writes the [2I] accumulator (post-alpha/beta/RowVec/
    # ColVec, pre-act-fn) to gmem alongside the postact_a [I] write.
    # Bwd consumes preact via `kernel_a_bwd`'s SwiGLU-bwd in registers
    # (skipping the otherwise-required `pool @ W1.T` recompute GEMM); fwd
    # paths that don't need bwd activations leave it None.
    if store_preact:
        # Same pool layout as postact_a (Mflat = total_tiles * tile_m), but
        # full N=2I instead of half-N=I. n-major (2I contiguous), bf16.
        mD = fake_tensor(
            postact_dtype, (Mflat_sym, I2_sym), leading_dim=1, divisibility=8
        )
    else:
        mD = None
    mC = None
    # mAuxOut: flat (total_tiles * tile_m, I), n-major (I contiguous).
    mAuxOut = fake_tensor(
        postact_dtype, (Mflat_sym, I_sym), leading_dim=1, divisibility=8
    )

    # cu_seqlens_m drives the standard varlen_m m-offset for kernel A: each
    # entry is `expert_pool_block_offset[e] * tile_m`. Length E_local + 1.
    mCuSeqlensM = fake_tensor(
        cutlass.Int32, (cu_seqlens_len_sym,), leading_dim=0, divisibility=1
    )

    consumer_head = fake_tensor(cutlass.Int32, (cute.sym_int(),), divisibility=1)
    pool_arrival_count = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)
    pool_arrival_target = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)
    expert_pool_block_offset = fake_tensor(
        cutlass.Int32, (cu_seqlens_len_sym,), divisibility=1
    )

    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(0),  # set at runtime; 0 here keeps fake compile happy
        consumer_head=consumer_head,
        pool_arrival_count=pool_arrival_count,
        pool_arrival_target=pool_arrival_target,
        expert_pool_block_offset=expert_pool_block_offset,
        total_tiles=Int32(0),
    )

    epi_args = StreamingMoeA.EpilogueArguments(
        mAuxOut=mAuxOut,
        act_fn=gate_fn_map[activation],
        rounding_mode=RoundingMode.RN,
    )

    varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlensM, mCuSeqlensK=None, mAIdx=None)

    return compile_gemm_kernel(
        StreamingMoeA,
        a_dtype,
        (tile_m, tile_n),
        (cluster_m, cluster_n, 1),
        pingpong=False,
        persistent=True,
        gather_A=False,
        is_dynamic_persistent=False,
        device_capacity=device_capacity,
        mA=mA,
        mB=mB,
        mD=mD,
        mC=mC,
        epi_args=epi_args,
        scheduler_args=scheduler_args,
        varlen_args=varlen_args,
    )


# ---------------------------------------------------------------------------
# Test-only producer: walks pool_arrival_count slot-by-slot and writes the
# matching `pool_arrival_target[i]` value (single-producer simulation of
# dispatch's Pass 2 release-add chain — when count == target, kernel A's
# scheduler spin unblocks). Used by tests to validate kernel A's per-tile
# count-vs-target spin without DeepEP.
# ---------------------------------------------------------------------------
class _StreamingTileProducer:
    @cute.jit
    def __call__(
        self,
        pool_arrival_count: cute.Tensor,  # [total_tiles] int32
        pool_arrival_target: cute.Tensor,  # [total_tiles] int32
        total_tiles: cutlass.Int32,
        delay_clocks: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            pool_arrival_count, pool_arrival_target, total_tiles, delay_clocks
        ).launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        pool_arrival_count: cute.Tensor,
        pool_arrival_target: cute.Tensor,
        total_tiles: cutlass.Int32,
        delay_clocks: cutlass.Int32,
    ):
        from cutlass._mlir.dialects import nvvm
        from cutlass.cutlass_dsl import T
        from quack import utils

        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            for i in cutlass.range(total_tiles):
                start = cutlass.Int64(nvvm.read_ptx_sreg_clock64(T.i64()))
                end = start + cutlass.Int64(delay_clocks)
                while cutlass.Int64(nvvm.read_ptx_sreg_clock64(T.i64())) < end:
                    pass
                target = pool_arrival_target[i]
                count_ptr = utils.elem_pointer(pool_arrival_count, (i,))
                threadfence_system()
                # `red.release.gpu.global.add.s32 [ptr], target` — single PTX
                # mirroring the real fire_pool_blocks. Pre-set the target so
                # one shot brings count to it.
                from cutlass._mlir.dialects import llvm
                count_ptr_i64 = count_ptr.toint().ir_value()
                llvm.inline_asm(
                    None,
                    [count_ptr_i64, target.ir_value()],
                    "red.release.gpu.global.add.s32 [$0], $1;",
                    "l,r,~{memory}",
                    has_side_effects=True,
                    is_align_stack=False,
                )


@jit_cache
def _compile_streaming_tile_producer():
    total_tiles_sym = cute.sym_int()
    count = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)
    target = fake_tensor(cutlass.Int32, (total_tiles_sym,), divisibility=1)
    op = _StreamingTileProducer()
    return cute.compile(
        op,
        count,
        target,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def fire_tiles_with_delay(
    pool_arrival_count: torch.Tensor,
    pool_arrival_target: torch.Tensor,
    delay_us: int = 50,
) -> None:
    """Test helper: launches a single-thread producer kernel on the current
    CUDA stream that `red.release.gpu.global.add.s32`s
    ``pool_arrival_target[i]`` into ``pool_arrival_count[i]`` for each tile
    (one fire per tile, ``delay_us`` between fires). Mirrors dispatch's
    Pass 2 protocol exactly; tests can wait on the standard scheduler spin.
    """
    assert pool_arrival_count.dtype == torch.int32
    assert pool_arrival_target.dtype == torch.int32
    assert pool_arrival_count.is_cuda and pool_arrival_count.is_contiguous()
    assert pool_arrival_target.shape == pool_arrival_count.shape
    total_tiles = pool_arrival_count.shape[0]
    # H100 clock ~1.5 GHz → 1500 cycles/μs.
    delay_clocks = max(1, int(delay_us * 1500))
    compiled = _compile_streaming_tile_producer()
    compiled(
        pool_arrival_count,
        pool_arrival_target,
        cutlass.Int32(total_tiles),
        cutlass.Int32(delay_clocks),
    )


def streaming_moe_a(
    pool: torch.Tensor,  # (TK_padded, H) bf16 — k-major (pool data, expert-major)
    W1: torch.Tensor,  # (E_local, 2I, H) bf16 — k-major per expert
    postact_a: torch.Tensor,  # (total_tiles, tile_M, I) bf16
    expert_pool_block_offset: torch.Tensor,  # (E_local + 1,) int32 — pool-block prefix sum
    pool_arrival_count: torch.Tensor,  # (total_tiles,) int32 — dispatch Pass 2 release-add destination
    pool_arrival_target: torch.Tensor,  # (total_tiles,) int32 — per-tile firing target
    *,
    preact_a: torch.Tensor | None = None,
    tile_m: int = 128,
    tile_n: int = 256,
    cluster_m: int = 1,
    cluster_n: int = 1,
    activation: str = "swiglu",
    num_sms: int | None = None,
) -> None:
    """Launch streaming-MoE kernel A on the caller's current CUDA stream (pool layout).

    ``num_sms`` caps the persistent-grid CTA count to the given value. When
    ``None`` (default) the kernel fills the GPU via ``get_max_active_clusters``.

    Caller is responsible for:
      - allocating ``postact_a`` ``(total_tiles, tile_M, I)`` ON THE SAME STREAM
        this function is called from (so the kernel's TMA stores are naturally
        ordered with the allocation; otherwise stale memory may leak through).
      - ensuring ``pool_arrival_count`` / ``pool_arrival_target`` are
        populated by the producer (DeepEP's ``Buffer.dispatch`` Pass 2 or a
        test stub) on a stream that ``red.release.gpu.global.add.s32``s into
        ``pool_arrival_count[tile_id]`` until it equals
        ``pool_arrival_target[tile_id]``. Kernel A's per-tile count-vs-target
        spin handles cross-stream visibility for those tensors and the
        dispatch metadata they transitively depend on.

    The internal ``consumer_head`` counter is allocated on the calling stream
    so its zero-init is naturally ordered with the kernel.

    Kernel Y runs on the same compute stream and is FIFO-ordered after A
    fully retires — no per-tile A→Y signal is needed.

    Optional ``preact_a`` ``(total_tiles, tile_M, 2*I) bf16`` is the pre-SwiGLU
    accumulator destination for bwd. When passed, kernel A's standard mD
    TMA-store path (inherited from ``GemmGatedMixin → GemmDefaultEpiMixin``)
    writes the [2I] gate-up values to gmem alongside the postact_a [I] write.
    Saving preact lets ``kernel_a_bwd`` apply SwiGLU bwd in registers without
    a recompute GEMM (~370 µs/layer perf win at production); fwd-only paths
    leave ``preact_a=None`` to skip the extra TMA-store traffic. The two
    cases compile to separate kernels (different mD signature), keyed on
    ``store_preact`` in ``_compile_streaming_moe_a``'s jit_cache.
    """
    assert pool.is_cuda and W1.is_cuda and postact_a.is_cuda
    assert pool.dim() == 2 and pool.is_contiguous()
    assert W1.dim() == 3
    assert postact_a.dim() == 3
    total_tiles, postact_tile_m, I = postact_a.shape
    assert postact_tile_m == tile_m
    assert (
        pool_arrival_count.shape == (total_tiles,)
        and pool_arrival_count.dtype == torch.int32
    )
    assert (
        pool_arrival_target.shape == (total_tiles,)
        and pool_arrival_target.dtype == torch.int32
    )
    H = pool.shape[1]
    E_local = W1.shape[0]
    assert expert_pool_block_offset.shape == (E_local + 1,)
    assert (
        W1.shape[1] == 2 * I
    ), f"W1 dim 1 must be 2*I = {2 * I}; got W1.shape={tuple(W1.shape)}"
    assert (
        W1.shape[2] == H
    ), f"W1 dim 2 (H) must match pool dim 1; got W1.shape={tuple(W1.shape)}, H={H}"
    two_I = W1.shape[1]
    # tile_n MUST divide the output N dim. The grouped GEMM emits one CTA per
    # (pid_m, pid_n) and writes a full tile_n-wide slab via TMA; partial-tile
    # tail handling silently corrupts adjacent memory. kernel A's output N is
    # 2I (mD/preact) or I (mAuxOut/postact); both must be divisible.
    assert two_I % tile_n == 0, (
        f"tile_n ({tile_n}) must divide 2I ({two_I}); 2I % tile_n = {two_I % tile_n}"
    )
    assert I % tile_n == 0, (
        f"tile_n ({tile_n}) must divide I ({I}); I % tile_n = {I % tile_n}"
    )
    if preact_a is not None:
        assert preact_a.is_cuda and preact_a.dim() == 3
        assert preact_a.shape == (total_tiles, tile_m, two_I), (
            f"preact_a must be (total_tiles, tile_m, 2*I) = "
            f"{(total_tiles, tile_m, two_I)}; got {tuple(preact_a.shape)}"
        )
        assert preact_a.dtype == postact_a.dtype, (
            f"preact_a dtype must match postact_a's; got {preact_a.dtype} vs "
            f"{postact_a.dtype}"
        )
    # Caller passes W1 as (E_local, 2I, H) k-major contiguous (each expert's
    # slab has H contiguous). We need the kernel to see shape (2I, H, E_local)
    # with leading_dim=1 (H is contiguous along K). torch.permute(1, 2, 0)
    # gives this layout WITHOUT a copy.
    W1_p = W1.permute(1, 2, 0)
    assert (
        W1_p.stride(1) == 1
    ), "W1[:,e,:] must be H-contiguous (caller passes k-major weights)"
    assert W1_p.shape == (two_I, H, E_local)

    # Flatten postact_a's leading two dims to (total_tiles * tile_m, I).
    postact_flat = postact_a.view(total_tiles * tile_m, I)
    # Flatten preact_a similarly when present — same Mflat dim, but full N=2I.
    preact_flat = (
        preact_a.view(total_tiles * tile_m, two_I) if preact_a is not None else None
    )

    # Build cu_seqlens_m = expert_pool_block_offset * tile_m. The standard
    # varlen_m path inside the GEMM uses this as the per-batch m-row offset:
    #   m_offset(tile) = cu_seqlens_m[batch_idx] + pid_m * tile_m
    #                  = expert_pool_block_offset[expert_id] * tile_m + tile_in_e * tile_m
    #                  = tile_id * tile_m
    # which lands at the correct pool row (pool is contiguous in tile_id order
    # by construction).
    cu_seqlens_m = (expert_pool_block_offset.to(torch.int32) * tile_m).contiguous()

    device_capacity = get_device_capacity(pool.device)
    assert device_capacity[0] == 9, "Streaming MoE kernel A is SM90-only for now"

    a_dtype = torch2cute_dtype_map[pool.dtype]
    b_dtype = torch2cute_dtype_map[W1.dtype]
    postact_dtype = torch2cute_dtype_map[postact_a.dtype]

    compiled_fn = _compile_streaming_moe_a(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        postact_dtype=postact_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        activation=activation,
        device_capacity=device_capacity,
        store_preact=preact_a is not None,
    )

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_m * cluster_n)
    if num_sms is not None:
        max_active_clusters = min(
            max_active_clusters, num_sms // (cluster_m * cluster_n)
        )

    # Internal scheduler counter — allocate on the calling stream (which is the
    # one the kernel will run on) so the zero-init is naturally ordered with
    # kernel A's `atomicAdd(consumer_head, 1)`. Allocating on a different stream
    # would race with the kernel's first atomic-claim and could leak stale
    # values from a recycled allocator slot, causing CTAs to early-exit.
    consumer_head = torch.zeros(1, dtype=torch.int32, device=pool.device)

    epi_args = StreamingMoeA.EpilogueArguments(
        mAuxOut=postact_flat,
        act_fn=None,  # Constexpr; pass None at call time
        rounding_mode=None,  # Constexpr; pass None at call time
    )
    scheduler_args = StreamingTileSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        consumer_head=consumer_head,
        pool_arrival_count=pool_arrival_count,
        pool_arrival_target=pool_arrival_target,
        expert_pool_block_offset=expert_pool_block_offset,
        total_tiles=Int32(total_tiles),
    )
    varlen_args = VarlenArguments(
        mCuSeqlensM=cu_seqlens_m, mCuSeqlensK=None, mAIdx=None
    )

    compiled_fn(
        pool, W1_p, preact_flat, None, epi_args, scheduler_args, varlen_args, None
    )
