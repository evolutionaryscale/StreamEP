"""Streaming-MoE forward layer: dispatch → kernel A → kernel Y → combine on four streams.

Streams are caller-owned — created once via :func:`make_streams` and passed
in as a :class:`StreamHolder`. Per-call inputs vary (routing changes
layer-to-layer in production), and so do the per-call intermediate
allocations: ``postact_a`` is sized by ``total_tiles × tile_M × I`` (varies
with routing). The kernel-Y output ``handle.o`` is allocated by DeepEP inside
``Buffer.dispatch`` at ``[T_recv, hidden]`` zero-init; kernel Y atomic-scatters
into it via PTX-predicated atomics, and the combine sender consumes it on a
separate stream — its per-warp send loop spins on
``handle.compute_done_per_token[r] >= dispatch_seq`` (the per-token gate)
before pushing ``o[r]`` back to ``r``'s origin rank.

Within a layer the four streams overlap (intra-layer); across layers they
serialize via the layer-start ``streams.dispatch.wait_stream(caller)`` and
layer-end ``caller.wait_stream(...)`` back-edges. See design.md §"Cross-stream
sync chain per layer".

Host-side ordering. The setup work (``record_stream`` calls + ``wait_event``)
is split per-stream and interleaved with each stream's kernel launch — kernel
A's path runs first so its launch reaches the GPU before dispatch_main's
persistent CTAs have finished exiting, preserving the per-tile dispatch→A
streaming overlap. Each CUDA API call is ~5 µs of host latency at production
shape; bunching all 19 ``record_stream`` calls + 3 ``wait_event`` calls in
front of kernel A would cost ~100 µs on dispatch_main's tail, which would
collapse the overlap window (the entire dispatch_main is only ~190 µs).

Public surface
--------------
* :class:`StreamHolder` — dataclass holding the four caller-owned streams.
* :func:`make_streams` — construct a ``StreamHolder`` for a given device.
* :class:`StreamMoEFunc` — ``torch.autograd.Function`` running the layer
  forward and backward. Backward orchestrates dispatch_grads → kernel_y_bwd
  → kernel_a_bwd → combine_grads on the same four streams as forward, with
  dW1 / dW2 grouped GEMMs in the compute_a / compute_y tails.
* :func:`stream_moe_func` — thin wrapper around ``StreamMoEFunc.apply`` with
  the public keyword-arg API.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from quack.gemm import gemm
from stream_ep import Buffer as StreamEPBuffer

from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_a import (
    streaming_moe_a,
)
from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_a_bwd import (
    streaming_moe_a_bwd,
)
from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_y import (
    streaming_moe_y,
)
from evolutionaryscale.models.moe.streaming_moe.streaming_kernel_y_bwd import (
    streaming_moe_y_bwd,
)


@dataclass(frozen=True)
class StreamHolder:
    """The five caller-owned streams driving one streaming-MoE layer.

    Forward uses four (``dispatch`` / ``compute_a`` / ``compute_y`` /
    ``combine``); backward adds a fifth ``grads`` stream that hosts the
    dW1 / dW2 grouped GEMMs. The dW grads are purely LOCAL (each rank
    holds its E_local slice of the expert weight grads — never crosses
    ranks), so they can run on a side stream that doesn't block
    combine_grads. Without this, dW1 / dW2 sat in the tail of
    ``compute_a`` / ``compute_y`` respectively and contended for SMs
    with kernel_a_bwd / kernel_y_bwd, stretching the streaming-kernel
    critical path.
    """

    dispatch: torch.cuda.Stream
    compute_a: torch.cuda.Stream
    compute_y: torch.cuda.Stream
    combine: torch.cuda.Stream
    grads: torch.cuda.Stream


def make_streams(device: torch.device | int | None = None) -> StreamHolder:
    """Allocate the five streams the layer expects.

    Caller creates this once (per training process) and reuses it across all
    layers and iterations — streams are not per-call state.
    """
    return StreamHolder(
        dispatch=torch.cuda.Stream(device=device),
        compute_a=torch.cuda.Stream(device=device),
        compute_y=torch.cuda.Stream(device=device),
        combine=torch.cuda.Stream(device=device),
        grads=torch.cuda.Stream(device=device),
    )


class StreamMoEFunc(torch.autograd.Function):
    """Streaming-MoE layer as a differentiable autograd boundary.

    Forward: dispatch → kernel A → kernel Y → combine on the four caller-owned
    streams. Backward orchestrates dispatch_grads → kernel_y_bwd → kernel_a_bwd
    → combine_grads on the same four streams (per-stage stream role preserved
    by comm direction; chronological order swaps compute_a / compute_y because
    of the chain rule). dW1 / dW2 grouped GEMMs run as separate `quack.gemm`
    calls in the tail of the streams that own each output:
      - dW2 on streams.compute_y after kernel_y_bwd (consumes dL_do_pool).
      - dW1 on streams.compute_a after kernel_a_bwd (consumes dL_dswiglu_in).

    Cross-stage synchronization is per-tile / per-recv-token via system-scope
    release/acquire stamps fired by each kernel — no `cudaStreamWaitEvent`
    between bwd stages, no host syncs, and no cross-stream events at all.
    kernel_y_bwd's per-tile ``bwd_a_ready`` release-store
    (threadfence_system) fences ALL three of its outputs (dL_dswiglu_in,
    postact_a_for_dW2, dL_dweight) system-scope. combine_grads's sender's
    per-recv-token gate (``bwd_compute_done_per_token[r]``, fired by
    kernel_a_bwd which acquires bwd_a_ready) transitively makes
    ``dL_dweight`` visible to the sender — same chain that publishes
    ``dL_dx_per_r``. Per-recv-token streaming on combine_grads is
    preserved end-to-end (first packet ships when the first r's tile is
    done, not when all of compute_y has drained).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        streams: StreamHolder,
        buffer: StreamEPBuffer,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        is_token_in_rank: torch.Tensor,
        w1_local: torch.Tensor,
        w2_local: torch.Tensor,
        num_experts: int,
        dispatch_seq: int,
        tile_m: int,
        tile_n_a: int,
        tile_n_y: int,
        num_sms_a: int | None,
        num_sms_y: int | None,
    ) -> torch.Tensor:
        caller_stream = torch.cuda.current_stream()

        # Layer-start back-edge — see design.md §"Cross-stream sync chain per
        # layer". Without this, iter N's persistent kernel A / kernel Y CTAs
        # (and the combine sender's per-channel CTAs) starve iter N+1's
        # dispatch_main on shared SMs. We only need to gate ``streams.dispatch``
        # here; the other three consumer streams are gated by ``metadata_done``
        # below, which is recorded downstream of caller_stream →
        # streams.dispatch's serialization.
        streams.dispatch.wait_stream(caller_stream)

        with torch.cuda.stream(streams.dispatch):
            pool, handle, metadata_done = buffer.dispatch(
                x,
                topk_idx,
                topk_weights,
                is_token_in_rank,
                num_experts,
                tile_m=tile_m,
                dispatch_seq=dispatch_seq,
            )

        # ── Kernel A path (queued FIRST so its launch hits the GPU before
        # dispatch_main's persistent CTAs have all exited). dispatch_main runs
        # ~210 µs at production num_sms=80; its CTAs exit gradually as their
        # per-channel work drains AND the 80-CTA grid leaves ~52 SMs free for
        # kernel A's persistent grid to land on while dispatch is still
        # running. Bunching combine-side host setup before kernel A would
        # push its launch ~50 µs later and collapse the overlap window.
        streams.compute_a.wait_event(metadata_done)
        pool.record_stream(streams.compute_a)
        handle.tile_ready.record_stream(streams.compute_a)
        handle.tile_id_to_expert.record_stream(streams.compute_a)
        handle.expert_pool_block_offset.record_stream(streams.compute_a)
        handle.a_ready.record_stream(streams.compute_a)
        with torch.cuda.stream(streams.compute_a):
            postact_a = torch.empty(
                handle.total_tiles,
                tile_m,
                w2_local.shape[2],
                dtype=x.dtype,
                device=pool.device,
            )
            # preact_a holds the [2I] pre-SwiGLU gate-up accumulator from kernel
            # A's mD TMA-store path (opt-in via the ``preact_a`` kwarg below).
            # Kept alive across fwd→bwd via ctx.save_for_backward; bwd consumes
            # it for SwiGLU bwd in registers (kernel_a_bwd skips the recompute
            # GEMM) and for postact_a recompute in dL/dweight (kernel_y_bwd's
            # per-row dot product). postact_a stays as fwd-only scratch (kernel
            # Y reads it then it's freed at the layer-end exit chain) — the
            # cheap thing element-wise-recomputes from preact in bwd, the
            # expensive thing (preact, would otherwise need a GEMM to recover)
            # is the one we save.
            preact_a = torch.empty(
                handle.total_tiles,
                tile_m,
                2 * w2_local.shape[2],
                dtype=x.dtype,
                device=pool.device,
            )
            streaming_moe_a(
                pool,
                w1_local,
                postact_a,
                handle.tile_id_to_expert,
                handle.expert_pool_block_offset,
                handle.tile_ready,
                handle.a_ready,
                dispatch_seq=handle.dispatch_seq,
                compute_seq=handle.dispatch_seq,
                preact_a=preact_a,
                tile_m=tile_m,
                tile_n=tile_n_a,
                num_sms=num_sms_a,
            )

        # ── Kernel Y path. ``postact_a`` is allocated on streams.compute_a
        # where kernel A writes it, and read by kernel Y on streams.compute_y.
        # Per-tile cross-stream data visibility comes from kernel Y's in-kernel
        # ``a_ready`` spin; ``record_stream`` only governs the caching
        # allocator's recycle policy.
        postact_a.record_stream(streams.compute_y)
        streams.compute_y.wait_event(metadata_done)
        handle.tile_id_to_expert.record_stream(streams.compute_y)
        handle.expert_pool_block_offset.record_stream(streams.compute_y)
        handle.a_ready.record_stream(streams.compute_y)
        handle.pool_recv_token.record_stream(streams.compute_y)
        handle.pool_topk_weight.record_stream(streams.compute_y)
        handle.per_token_remaining.record_stream(streams.compute_y)
        handle.compute_done_per_token.record_stream(streams.compute_y)
        handle.o.record_stream(streams.compute_y)
        with torch.cuda.stream(streams.compute_y):
            streaming_moe_y(
                postact_a,
                w2_local,
                handle.o,
                handle.pool_recv_token,
                handle.pool_topk_weight,
                handle.per_token_remaining,
                handle.compute_done_per_token,
                handle.tile_id_to_expert,
                handle.expert_pool_block_offset,
                handle.a_ready,
                compute_seq=handle.dispatch_seq,
                combine_seq=dispatch_seq,
                tile_m=tile_m,
                tile_n=tile_n_y,
                num_sms=num_sms_y,
            )

        # ── Combine sender path. Per-(channel, dst_rank) sender warps spin on
        # ``handle.compute_done_per_token[r] >= dispatch_seq`` before pushing
        # ``o[r]`` to r's origin rank. The per-token gate lets early-completing
        # tokens (K_local=1 first-wave-expert tokens) ship while late-wave
        # tokens are still landing — combine ↔ kernel Y tail overlap. Streaming
        # granularity = NVL chunk size (default 4 tokens per chunk at 8-rank
        # intranode per ``Buffer.get_combine_config``).
        #
        # Cross-stream visibility on ``compute_done_per_token`` is carried by
        # the per-token ``.sys``-scope release/acquire pair the kernels
        # themselves issue; ``streams.combine``'s only cross-stream sync is the
        # ``metadata_done`` event (one-shot, before dispatch main).
        streams.combine.wait_event(metadata_done)
        handle.compute_done_per_token.record_stream(streams.combine)
        handle.o.record_stream(streams.combine)
        handle.rank_prefix_matrix.record_stream(streams.combine)
        handle.channel_prefix_matrix.record_stream(streams.combine)
        handle.send_head.record_stream(streams.combine)
        handle.pool_topk_weight.record_stream(streams.combine)
        handle.recv_token_to_slots.record_stream(streams.combine)
        with torch.cuda.stream(streams.combine):
            out, _ = buffer.combine(handle.o, handle, combine_seq=dispatch_seq)

        # Layer-end back-edges — see design.md §"Cross-stream sync chain per layer".
        caller_stream.wait_stream(streams.dispatch)
        caller_stream.wait_stream(streams.compute_a)
        caller_stream.wait_stream(streams.compute_y)
        caller_stream.wait_stream(streams.combine)

        # Save tensors bwd will consume. The contract is "save preact, drop
        # postact" — preact_a is the strictly more useful saved activation:
        # kernel_y_bwd reads it as mC for in-epilogue SwiGLU bwd (dswiglu's
        # postact byproduct feeds the dL/dweight ColVecReduce dot product),
        # and the orchestrator element-wise-recomputes postact_a from it
        # (silu(gate)*up) into a transient buffer just before dW2's grouped
        # GEMM. Saving postact_a instead would force kernel_y_bwd / kernel_a_bwd
        # to recover preact via a `pool @ W1[e].T` recompute GEMM (~370 µs/layer
        # at production) — save the expensive thing, recompute the cheap thing.
        # `pool` and `w1_local` / `w2_local` go through save_for_backward too
        # (autograd's standard activation-save plumbing); `handle`, `streams`,
        # `buffer`, and the int knobs ride on `ctx` attributes since they are
        # not Tensors. `dispatch_seq` lives on the handle already.
        ctx.save_for_backward(preact_a, pool, w1_local, w2_local)
        ctx.streams = streams
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.tile_m = tile_m
        ctx.tile_n_a = tile_n_a
        ctx.tile_n_y = tile_n_y
        ctx.num_sms_a = num_sms_a
        ctx.num_sms_y = num_sms_y
        return out

    @staticmethod
    def backward(ctx, dL_dy):  # type: ignore[override]
        """Four-stage backward on the same four streams as fwd.

        Stages by stream:
          dispatch_grads   on streams.dispatch
          kernel_y_bwd     on streams.compute_y  (+ dW2 grouped GEMM tail)
          kernel_a_bwd     on streams.compute_a  (+ dW1 grouped GEMM tail)
          combine_grads    on streams.combine

        Inter-stage waits are per-tile / per-recv-token system-scope
        release/acquire stamps embedded in the kernels (`bwd_y_ready` →
        kernel_y_bwd, `bwd_a_ready` → kernel_a_bwd, `bwd_compute_done_per_token`
        → combine_grads). No `cudaStreamWaitEvent` between bwd stages — the
        per-tile acquire-spins inside each kernel handle cross-stream
        visibility.

        kernel_y_bwd writes THREE outputs (mD = dL_dswiglu_in,
        mPostAct = postact_a_for_dW2, mColVecReduce = dL_dweight via
        per-pid_n red.global.add.f32). All three are fenced by the
        per-tile bwd_a_ready release-store; combine_grads's per-token
        gate (bwd_compute_done_per_token[r], fired by kernel_a_bwd which
        acquires bwd_a_ready) transitively publishes dL_dweight to
        combine's sender. No cross-stream events anywhere in the bwd
        path — purely device-side per-tile / per-token release/acquire.

        Per-stream zero-init setup ops are issued BEFORE the
        `wait_stream(caller_stream)` fan-out so they overlap with the upstream
        layer's bwd tail running on caller_stream. Only the
        `bwd_per_token_remaining.copy_(handle.k_local_count)` op is
        caller-dependent (handle.k_local_count was last-written on
        streams.dispatch in fwd → caller-visible after fwd's exit chain), so it
        runs after streams.compute_a's wait_stream(caller_stream).

        Returns gradients matching forward's 15 args:
            (None, None, dL_dx, None, dL_dtopk_weights, None,
             dW1_local, dW2_local, None, None, None, None, None, None, None)
        """
        preact_a, pool, w1_local, w2_local = ctx.saved_tensors
        streams: StreamHolder = ctx.streams
        buffer: StreamEPBuffer = ctx.buffer
        handle = ctx.handle
        tile_m: int = ctx.tile_m
        tile_n_a: int = ctx.tile_n_a
        tile_n_y: int = ctx.tile_n_y
        num_sms_a: int | None = ctx.num_sms_a
        num_sms_y: int | None = ctx.num_sms_y

        # Upstream may pass a non-contiguous grad (e.g. `out.sum().backward()`
        # produces a stride-(0,0) broadcast view); `dispatch_grads` asserts
        # contiguity, so normalise here.
        dL_dy = dL_dy.contiguous()

        caller_stream = torch.cuda.current_stream()
        device = pool.device
        dtype = pool.dtype

        E_local, two_I, H = w1_local.shape
        I = two_I // 2
        total_tiles = handle.total_tiles
        TK_padded = pool.shape[0]
        T_recv = handle.k_local_count.shape[0]

        # ── Per-stream setup, in parallel, before any wait_stream ──────────
        # Fresh-tensor zero-inits + the `bwd_a_ready` cross-stream signal
        # (zero-init on its consumer stream so the first kernel_a_bwd
        # acquire-load sees an unfired counter). Nothing here references
        # caller-resident state, so the allocator doesn't need caller_stream
        # ordering — these ops can overlap with the upstream layer's bwd tail.

        with torch.cuda.stream(streams.compute_a):
            dL_dx_per_r = torch.zeros(T_recv, H, dtype=dtype, device=device)
            bwd_per_token_remaining = torch.empty(
                T_recv, dtype=torch.int32, device=device
            )
            bwd_compute_done_per_token = torch.zeros(
                T_recv, dtype=torch.int64, device=device
            )

        # dW1_local / dW2_local live on the dedicated grads stream — see
        # below. They're zero-init here on grads (no in-stream-allocator
        # entanglement with compute_a/compute_y) so the grouped GEMM's
        # write reaches a known-zero buffer.
        with torch.cuda.stream(streams.grads):
            # quack.gemm with default `add_to_output=False` overwrites D, so
            # the dW1 / dW2 destinations don't need zero-init — saves ~225
            # MB / layer of memset bandwidth (~150 MB dW1 + ~75 MB dW2 at
            # production) without changing the GEMM result.
            dW1_local = torch.empty_like(w1_local)
            dW2_local = torch.empty_like(w2_local)

        with torch.cuda.stream(streams.compute_y):
            # kernel_y_bwd's mD output, consumed by kernel_a_bwd on
            # streams.compute_a and by dW1's grouped GEMM on streams.grads.
            # Bf16 (M, 2I); fp32-view applied internally by the kernel host
            # wrapper. `torch.empty` is safe: kernel_y_bwd's mPaddingMask
            # predicate (`pool_recv_token >= 0`) zeros (dgate, dup) BEFORE
            # the mD TMA-store, so padding rows land as clean zero — dW1's
            # grouped GEMM then contributes 0 at padding K-positions
            # regardless of dL_do_pool's padding-row content.
            dL_dswiglu_in = torch.empty(
                total_tiles, tile_m, two_I, dtype=dtype, device=device
            )
            # postact_a_for_dW2: kernel_y_bwd's mPostAct output (weighted
            # postact, in-kernel ``postact * pool_topk_weight``). Direct
            # input to dW2's grouped GEMM — eliminates the orchestrator-side
            # 6-elementwise-kernel torch recompute path that cost ~1170 µs
            # / iter on the production trace. Same predicate covers the
            # mPostAct store, so `torch.empty` is safe here too.
            postact_a_for_dW2 = torch.empty(
                total_tiles, tile_m, I, dtype=dtype, device=device
            )
            # dL_dweight: kernel_y_bwd's per-pid_n fp32 atomic-add target.
            # Zero-init mandatory (kernel atomic-adds, doesn't overwrite).
            # No more dL_dweight_per_stripe + post-hoc .sum() — collapsed
            # in-kernel via red.global.add.f32 across pid_n stripes.
            dL_dweight = torch.zeros(TK_padded, dtype=torch.float32, device=device)
            # bwd_a_ready is produced by kernel_y_bwd on compute_y and acquired
            # by kernel_a_bwd on compute_a. Allocate + zero-init on the
            # producer stream so the kernel's first release-store is naturally
            # ordered with the zero-init.
            bwd_a_ready = torch.zeros(total_tiles, dtype=torch.int64, device=device)

        # ── Single fan-out gate on caller_stream ───────────────────────────
        streams.dispatch.wait_stream(caller_stream)
        streams.compute_a.wait_stream(caller_stream)
        streams.compute_y.wait_stream(caller_stream)
        streams.combine.wait_stream(caller_stream)
        streams.grads.wait_stream(caller_stream)

        # ── Caller-dependent setup op ──────────────────────────────────────
        # handle.k_local_count was written on streams.dispatch in fwd; visible
        # on caller_stream after fwd's exit chain. The copy onto compute_a must
        # wait for caller_stream first.
        with torch.cuda.stream(streams.compute_a):
            bwd_per_token_remaining.copy_(handle.k_local_count, non_blocking=True)

        # ── Cross-stream record_stream for handle tensors used on streams
        # they were NOT recorded on during fwd. Held alive by ctx; ensures the
        # caching allocator considers all relevant streams when the tensor is
        # eventually freed.
        dL_dy.record_stream(streams.dispatch)
        handle.recv_token_to_slots.record_stream(streams.dispatch)
        handle.is_token_in_rank.record_stream(streams.dispatch)
        handle.base_pool.record_stream(streams.dispatch)
        handle.seen_per_substream.record_stream(streams.dispatch)
        handle.pool_arrival_target.record_stream(streams.dispatch)
        handle.pool_recv_token.record_stream(streams.compute_a)
        handle.pool_recv_token.record_stream(streams.compute_y)
        handle.k_local_count.record_stream(streams.compute_a)
        pool.record_stream(streams.compute_a)

        # ── Stage 1 — dispatch_grads on streams.dispatch ───────────────────
        # Ships dL/dy origin → expert ranks along fwd's routing, K-fans into
        # dL_do_pool[slot] via recv_token_to_slots[r, k]. Fires
        # bwd_y_ready[tile] when each pool block's writes drain.
        with torch.cuda.stream(streams.dispatch):
            dL_do_pool, bwd_y_ready = buffer.dispatch_grads(
                handle, dL_dy, dispatch_seq=handle.dispatch_seq
            )

        # dL_do_pool / bwd_y_ready were allocated on streams.dispatch by the
        # runtime; mark cross-stream readers so the allocator doesn't recycle
        # them prematurely.
        dL_do_pool.record_stream(streams.compute_y)
        bwd_y_ready.record_stream(streams.compute_y)

        # ── Stage 2 — kernel_y_bwd on streams.compute_y ────────────────────
        # Acquire-spins on bwd_y_ready[tile] internally; no cudaStreamWaitEvent
        # between dispatch_grads and kernel_y_bwd. SwiGLU bwd folded into the
        # epilogue. The kernel writes THREE outputs in one tile-streamed pass:
        #   - mD = dL_dswiglu_in  (bf16 (M, 2I) viewed fp32 (M, I))
        #   - mPostAct = postact_a_for_dW2  (bf16 (M, I) — weighted postact)
        #   - mColVecReduce = dL_dweight  (fp32 (M,) — per-pid_n atomic-add)
        # Per-tile bwd_a_ready release fences ALL three outputs system-scope,
        # so consumers on other streams (combine_grads on streams.combine
        # via the bwd_compute_done chain, dW2 grouped GEMM in-stream below)
        # see consistent values.
        with torch.cuda.stream(streams.compute_y):
            streaming_moe_y_bwd(
                dL_do_pool,
                w2_local,
                dL_dswiglu_in,
                postact_a_for_dW2,
                handle.pool_topk_weight,
                handle.pool_recv_token,
                preact_a,
                dL_dweight,
                handle.tile_id_to_expert,
                handle.expert_pool_block_offset,
                bwd_y_ready,
                bwd_a_ready,
                dispatch_seq=handle.dispatch_seq,
                tile_m=tile_m,
                tile_n=tile_n_a,
                num_sms=num_sms_a,
            )

        # bwd_a_ready / dL_dswiglu_in are written on compute_y; consumed on
        # compute_a (kernel_a_bwd) and grads (dW1 GEMM).
        bwd_a_ready.record_stream(streams.compute_a)
        dL_dswiglu_in.record_stream(streams.compute_a)
        dL_dswiglu_in.record_stream(streams.grads)
        postact_a_for_dW2.record_stream(streams.grads)
        dL_do_pool.record_stream(streams.grads)
        pool.record_stream(streams.grads)
        # dL_dweight is read by combine_grads on streams.combine. The
        # cross-stream visibility is carried by the per-tile bwd_a_ready
        # release-store's threadfence_system (kernel_y_bwd) → kernel_a_bwd's
        # acquire of bwd_a_ready → kernel_a_bwd's release of
        # bwd_compute_done_per_token → combine_grads's per-token gate
        # acquire. record_stream is just allocator bookkeeping (combine
        # might free dL_dweight before compute_y has retired its writes
        # without it).
        dL_dweight.record_stream(streams.combine)

        # ── Stage 3 — kernel_a_bwd on streams.compute_a ────────────────────
        # Acquire-spins on bwd_a_ready[tile] internally. Vanilla streaming GEMM
        # `dL/dpool = dL/dswiglu_in @ W1` + atomic-scatter into dL_dx_per_r;
        # per-row stripe-done bookkeeping fires bwd_compute_done_per_token[r]
        # on the last contributor.
        with torch.cuda.stream(streams.compute_a):
            streaming_moe_a_bwd(
                dL_dswiglu_in,
                w1_local,
                dL_dx_per_r,
                handle.pool_recv_token,
                bwd_per_token_remaining,
                bwd_compute_done_per_token,
                handle.tile_id_to_expert,
                handle.expert_pool_block_offset,
                bwd_a_ready,
                dispatch_seq=handle.dispatch_seq,
                tile_m=tile_m,
                tile_n=tile_n_y,
                num_sms=num_sms_y,
            )

        # dL_dx_per_r / bwd_compute_done_per_token are written on compute_a,
        # consumed by combine_grads on streams.combine.
        dL_dx_per_r.record_stream(streams.combine)
        bwd_compute_done_per_token.record_stream(streams.combine)

        # ── Stage 4 — combine_grads on streams.combine ─────────────────────
        # Sender per-warp loop spins on bwd_compute_done_per_token[r] >=
        # dispatch_seq before reading dL_dx_per_r[r] AND dL_dweight[slot]
        # (gathered via recv_token_to_slots[r, k]). Cross-stream visibility
        # of BOTH is carried by the device-side fence chain:
        #   kernel_y_bwd's per-tile bwd_a_ready release-store
        #     (threadfence_system fences mPostAct + dL_dweight + dL_dswiglu_in)
        #   → kernel_a_bwd acquire bwd_a_ready
        #   → kernel_a_bwd release-store bwd_compute_done_per_token[r]
        #     (per-token, fenced)
        #   → combine_grads sender acquire bwd_compute_done_per_token[r]
        # No explicit cross-stream event — per-recv-token streaming preserved
        # (combine_grads's first packet ships when the FIRST r's tile is done,
        # not when all of compute_y has drained).
        with torch.cuda.stream(streams.combine):
            dL_dx, dL_dtopk_weights = buffer.combine_grads(
                dL_dx_per_r,
                handle,
                dL_dweight,
                bwd_compute_done_per_token,
                dispatch_seq=handle.dispatch_seq,
            )

        # ── Stage 5 — dW1 / dW2 grouped GEMMs on streams.grads ─────────────
        # Both grads are LOCAL (each rank holds its E_local slice; never
        # crosses ranks via combine_grads), so they can run on a side stream
        # that doesn't block combine_grads. They consume kernel_y_bwd's
        # outputs (postact_a_for_dW2 / dL_dswiglu_in) plus pool / dL_do_pool
        # — no dependency on kernel_a_bwd. A single
        # ``wait_stream(streams.compute_y)`` captures the kernel_y_bwd
        # completion edge (compute_y has nothing else queued at this point).
        # On a 132-SM H100 dW1 + dW2 each want a full grid, so they queue
        # serially on grads — but during their run, combine_grads's sender
        # (which uses few SMs for IPC ring + cross-rank packets) overlaps
        # naturally; previously dW1 / dW2 sat in compute_a / compute_y's
        # tails and competed with kernel_a_bwd / kernel_y_bwd for SMs,
        # stretching both.
        streams.grads.wait_stream(streams.compute_y)
        with torch.cuda.stream(streams.grads):
            # dW2[e] = postact_a_for_dW2[slot_range_e].T @ dL_do_pool[slot_range_e]
            #   → (E_local, H, I) — same shape as w2_local, varlen-K grouped GEMM.
            # postact_a_for_dW2 is the weighted postact (postact *
            # pool_topk_weight) already materialised in bf16 by kernel_y_bwd's
            # mPostAct path — no orchestrator-side recompute. Chain rule on
            #   o[t] = Σ_k topk_weights[t, k] · y_for_(t, k)
            # gives dW2[e] = Σ_{slot in e} pool_topk_weight[slot] *
            #               dL_do_pool[slot] outer postact_a[slot] — the
            # weight factor rides ON postact via the in-kernel multiply.
            #
            # quack.gemm with cu_seqlens_k expects A: (M, total_K) m-major
            # and B: (N, total_K) n-major. For dW2: M=H, N=I, L=E_local,
            # total_K=TK_padded.
            #   A = dL_do_pool.t()             (H, TK_padded), m-major
            #   B = postact_a_for_dW2_flat.t() (I, TK_padded), n-major
            #   D = dW2_local                  (E_local, H, I)
            postact_a_for_dW2_flat = postact_a_for_dW2.view(TK_padded, I)
            cu_seqlens_k = (
                handle.expert_pool_block_offset.to(torch.int32) * tile_m
            ).contiguous()
            # `lens_k` = per-expert REAL recv count; decoupled from
            # cu_seqlens_k's tile-padded storage offsets. Quack passes lens_k
            # as the OOB-fill bound on the K-axis (offset_ragged_tensor's
            # `length` arg), so TMA reads beyond `expert_frequency[e]` in
            # batch e's K-tile come back as hardware zeros — no need for
            # dL_do_pool[padding] to be zero, allocator-garbage NaN bits at
            # those rows are simply not read.
            lens_k_dW = handle.expert_frequency.to(torch.int32)
            gemm(
                dL_do_pool.t(),
                postact_a_for_dW2_flat.t(),
                dW2_local,
                None,
                None,
                tile_M=tile_m,
                tile_N=tile_n_a,
                cluster_M=1,
                cluster_N=1,
                cu_seqlens_k=cu_seqlens_k,
                lens_k=lens_k_dW,
            )
            # dW1[e] = (dL_dswiglu_in[slot_range_e]).T @ pool[slot_range_e]
            # quack.gemm cu_seqlens_k: A m-major, B n-major.
            #   A = dL_dswiglu_in_flat.t() (2I, TK_padded), m-major
            #   B = pool.t()               (H,  TK_padded), n-major
            #   D = dW1_local              (E_local, 2I, H)
            dL_dswiglu_in_flat = dL_dswiglu_in.view(TK_padded, two_I)
            gemm(
                dL_dswiglu_in_flat.t(),
                pool.t(),
                dW1_local,
                None,
                None,
                tile_M=tile_m,
                tile_N=tile_n_a,
                cluster_M=1,
                cluster_N=1,
                cu_seqlens_k=cu_seqlens_k,
                lens_k=lens_k_dW,
            )

        # ── Exit chain back to caller_stream ───────────────────────────────
        caller_stream.wait_stream(streams.dispatch)
        caller_stream.wait_stream(streams.compute_y)
        caller_stream.wait_stream(streams.compute_a)
        caller_stream.wait_stream(streams.combine)
        caller_stream.wait_stream(streams.grads)

        # Outputs of combine_grads were allocated on streams.combine; record
        # their use on caller_stream so they're not freed before the upstream
        # backward consumes them.
        dL_dx.record_stream(caller_stream)
        dL_dtopk_weights.record_stream(caller_stream)
        dW1_local.record_stream(caller_stream)
        dW2_local.record_stream(caller_stream)

        # Gradient tuple matches forward's 15 args, in order:
        #   (streams, buffer, x, topk_idx, topk_weights, is_token_in_rank,
        #    w1_local, w2_local, num_experts, dispatch_seq, tile_m, tile_n_a,
        #    tile_n_y, num_sms_a, num_sms_y)
        return (
            None,  # streams
            None,  # buffer
            dL_dx,  # x
            None,  # topk_idx
            dL_dtopk_weights,  # topk_weights
            None,  # is_token_in_rank
            dW1_local,  # w1_local
            dW2_local,  # w2_local
            None,  # num_experts
            None,  # dispatch_seq
            None,  # tile_m
            None,  # tile_n_a
            None,  # tile_n_y
            None,  # num_sms_a
            None,  # num_sms_y
        )


def stream_moe_func(
    buffer: StreamEPBuffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    *,
    streams: StreamHolder,
    num_experts: int,
    dispatch_seq: int,
    tile_m: int = 128,
    tile_n_a: int = 256,
    tile_n_y: int = 128,
    num_sms_a: int | None = None,
    num_sms_y: int | None = None,
) -> torch.Tensor:
    """One MoE forward layer: dispatch + kernel A + kernel Y + combine.

    Returns the cross-rank-reduced output of shape ``[num_tokens, hidden]``
    produced by the combine receiver — the standard MoE forward output for
    this rank's source tokens.
    """
    return StreamMoEFunc.apply(
        streams,
        buffer,
        x,
        topk_idx,
        topk_weights,
        is_token_in_rank,
        w1_local,
        w2_local,
        num_experts,
        dispatch_seq,
        tile_m,
        tile_n_a,
        tile_n_y,
        num_sms_a,
        num_sms_y,
    )
