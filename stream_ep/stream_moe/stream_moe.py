"""Streaming-MoE layer: dispatch → kernel A → kernel Y → combine on two streams.

Stream graph:

  * ``communicate`` — fwd ``dispatch`` + fwd ``combine``; bwd ``dispatch_grads``
    + bwd ``combine_grads``. Same-stream FIFO orders combine after dispatch
    and combine_grads after dispatch_grads — no cross-stream serialization
    is needed between the comm halves.
  * ``compute`` — fwd kernel_a + kernel_y; bwd kernel_y_bwd + kernel_a_bwd
    + dW1 + dW2. Same-stream FIFO covers all intra-compute handoffs
    (A→Y, Y_bwd→A_bwd, and A_bwd→dW since dW1/dW2 saturate 132 SMs and
    serialize after A_bwd anyway). dW does not need its own stream because
    it cannot overlap with A_bwd; what it CAN overlap with — combine_grads
    on the communicate stream — happens naturally as soon as A_bwd retires.

Real cross-stream overlap windows in this layout:
  * fwd dispatch ↔ kernel_a (communicate ↔ compute): dispatch's 80 persistent
    CTAs drain on the copy engines as kernel_a's 132-CTA grid lands. Per-tile
    handshake is dispatch's ``pool_arrival_count`` release-add chain, which
    kernel_a's scheduler spins on. Communicate → compute hop is
    ``compute.wait_event(metadata_done)`` (event recorded BETWEEN dispatch's
    metadata kernel and dispatch_main, so kernel_a launches concurrent with
    dispatch_main). A second GPU-front-end gate
    (``buffer.wait_dispatch_main_started``) holds kernel_a's launch until
    dispatch_main's block 0 has actually entered execution; without it,
    under CDMC>1, kernel_a's 132-CTA grid can grab SMs before dispatch_main
    is co-resident and starve it.
  * fwd kernel_y ↔ combine (compute ↔ communicate): combine's sender warps
    spin on ``y_done_per_token[r] >= dispatch_seq`` and grab SMs as kernel_y
    tiles retire. The orchestrator bumps ``kernel_y_issued`` before
    launching kernel_y, then queues ``buffer.wait_kernel_y_started`` on
    ``communicate`` before combine — a GPU-front-end wait that holds until
    kernel_y's first CTA is co-resident on an SM (the scheduler's
    linear-claim CTA bumps the flag). Combine's 80 sender CTAs cannot
    grab SMs ahead of kernel_y's 132.
  * bwd dispatch_grads ↔ kernel_y_bwd: mirror of fwd dispatch ↔ A.
    ``buffer.dispatch_grads`` returns a ``grads_started_event`` recorded
    BEFORE its main kernel launches; compute waits on it so kernel_y_bwd
    launches concurrent with dispatch_grads_main, spinning on
    ``bwd_dispatch_arrival_count`` per-tile. The mirror of the
    ``wait_dispatch_main_started`` gate
    (``buffer.wait_dispatch_grads_started``) holds kernel_y_bwd's launch
    until dispatch_grads_main's block 0 is co-resident.
  * bwd kernel_a_bwd ↔ combine_grads: mirror of fwd Y ↔ combine. The
    orchestrator bumps ``kernel_a_bwd_issued`` before launching A_bwd, then
    queues ``buffer.wait_kernel_a_bwd_started`` on ``communicate`` before
    combine_grads so A_bwd wins the SM race. Per-recv-token streaming is
    driven by ``bwd_a_done_per_token[r]``.
  * bwd dW ↔ combine_grads: dW launches when A_bwd retires (same-stream
    FIFO on compute). combine_grads is still running its NVLink-bound
    sender phase on communicate; dW1/dW2's GEMMs fill the SMs that
    combine_grads's CTAs are leaving idle on per-token gates.

Within a layer the two streams overlap; across layers they serialize via the
layer-start ``stream.wait_stream(caller)`` and layer-end
``caller.wait_stream(stream)`` back-edges.

Public surface
--------------
* :class:`StreamHolder` — dataclass holding the two caller-owned streams.
* :func:`make_streams` — construct a ``StreamHolder`` for a given device.
* :class:`TileConfig` — frozen all-optional overrides struct for every GEMM
  tuning knob (``None`` field ⇒ auto-pick).
* :func:`default_tile_config` — bench-tuned resolved ``TileConfig`` picked from
  ``(I, H)`` with a power-of-2 divisibility fallback.
* :class:`StreamMoEFunc` — ``torch.autograd.Function`` running the layer
  forward and backward.
* :func:`stream_moe_func` — thin wrapper around ``StreamMoEFunc.apply`` with
  the public keyword-arg API (incl. optional ``tile_config``).

Compile interaction
-------------------
This module is eager-only. Callers that want ``torch.compile`` around the
outer model must apply ``@torch.compiler.disable`` at the consumer boundary
(e.g. evoscale's ``StreamMoEWrapper.forward``). The streaming-MoE surface
launches across two caller-owned streams and runs cross-rank IPC barriers
inside the metadata kernel; dynamo has no documented way to enumerate
user-managed streams (pytorch/pytorch#92804), and per-rank recompile skew
under CDMC>1 would deadlock the ``barrier_block`` inside the metadata
kernel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace

import torch
from quack.gemm import gemm
from stream_ep import Buffer as StreamEPBuffer

from stream_ep.stream_moe.kernel_a import (
    streaming_moe_a,
)
from stream_ep.stream_moe.kernel_a_bwd import (
    streaming_moe_a_bwd,
)
from stream_ep.stream_moe.kernel_y import (
    streaming_moe_y,
)
from stream_ep.stream_moe.kernel_y_bwd import (
    streaming_moe_y_bwd,
)
from stream_ep.stream_moe.recv_pool_compress import (
    compress_pool_to_recv,
    reexpand_recv_to_pool,
)


# ---------------------------------------------------------------------------
# Public tile-config surface.
#
# ``TileConfig`` is an all-optional overrides struct: every field defaults to
# ``None`` meaning "auto-pick this one". ``stream_moe_func`` resolves a config
# per call by overlaying the caller's non-``None`` fields onto
# ``default_tile_config(I, H)`` (the bench-tuned baseline) — so setting one
# knob never freezes the others at a stale hardcoded default; the rest are
# still picked for the actual shape. ``default_tile_config`` enforces the
# per-shape divisibility constraints on the 4 output ``tile_n`` values
# (largest power-of-2 ≤ 256 fallback when a default doesn't divide), and
# ``TileConfig.validate(I, H)`` re-checks them on the resolved config so a bad
# caller override fails with a clear message instead of inside a kernel.
# ---------------------------------------------------------------------------


def _largest_pow2_divisor(x: int, cap: int) -> int:
    """Largest power-of-two ≤ ``cap`` that divides ``x``."""
    t = cap
    while t > 1 and x % t != 0:
        t //= 2
    return t


@dataclass(frozen=True)
class TileConfig:
    """Every tuning knob the matched-pair GEMMs accept — as overrides.

    Public, frozen, all-optional. Every field defaults to ``None`` meaning
    "auto-pick this one". Construct with only the knobs you want to pin
    (``TileConfig(tile_n_a=128, num_sms_a=120)``) and pass to
    :func:`stream_moe_func` as ``tile_config=...``; the unset (``None``) fields
    are filled from :func:`default_tile_config` for the actual weight shape, so
    pinning one knob never freezes the others at a stale hardcoded default. The
    resolution (overlay + :meth:`validate`) happens in
    :func:`stream_moe_func`; :func:`default_tile_config` returns the fully
    resolved baseline (where ``None`` retains its kernel-level meaning — defer
    to the quack / backward default).

    ``tile_m`` is STRUCTURAL — the dispatch pool's BLOCK_M padding + per-tile
    signaling grain, shared by dispatch and all four compute kernels — not a
    per-kernel knob; changing it re-tiles the pool. :meth:`validate` rejects
    ``tile_m`` 192/320 (the quack ``atom_layout_n=2`` sizes the streaming
    backward GEMMs can't codegen): 320 outright, 192 unless every ``tile_n`` <=
    128. In practice ``tile_m`` 128 is the operating point — 256 keeps
    ``atom_layout_n=1`` but exceeds the dispatch-pool memory budget at
    production shapes. See markdowns/reserved_memory_stream_segregation.md.

    Constraints NOT checked by :meth:`validate` (quack ``GemmSm90`` raises on
    them directly, see ``quack/gemm_sm90.py``):
      * ``tile_m`` must be one of ``{64, 128, 192, 256, 320}`` (and only
        ``{64, 128, 192}`` for a dW GEMM that sets ``pingpong``).
      * each ``tile_n`` must additionally satisfy quack's CTA-N rule
        (``% 16`` and ≤256, or ``% 32`` and ≤512, tighter when tile_m is
        192/320). The bench-tuned defaults + power-of-2 fallback all comply.
    """

    tile_m: int | None = None
    # fwd kernel A output N = 2I; must divide both 2I and I.
    tile_n_a: int | None = None
    # fwd kernel Y output N = H; must divide H.
    tile_n_y: int | None = None
    # bwd kernel Y output N = I; must divide I.
    tile_n_y_bwd: int | None = None
    # bwd kernel A output N = H; must divide H.
    tile_n_a_bwd: int | None = None
    # dW GEMM tiles; ``None`` falls back to (tile_m, tile_n_a) inside the
    # backward; dW2 also accepts None tile_n meaning "use kernel A default".
    tile_m_dW1: int | None = None
    tile_n_dW1: int | None = None
    tile_m_dW2: int | None = None
    tile_n_dW2: int | None = None
    # dW grouped-GEMM cluster / pingpong / swizzle (quack epilogue tuning).
    cluster_m_dW1: int | None = None
    cluster_n_dW1: int | None = None
    cluster_m_dW2: int | None = None
    cluster_n_dW2: int | None = None
    pingpong_dW1: bool | None = None
    pingpong_dW2: bool | None = None
    swizzle_dW1: int | None = None
    swizzle_dW2: int | None = None
    # Persistent-kernel SM counts. ``None`` defers to quack's default for
    # ``streaming_moe_{a,y}{,_bwd}``.
    num_sms_a: int | None = None
    num_sms_y: int | None = None
    num_sms_a_bwd: int | None = None
    num_sms_y_bwd: int | None = None

    def validate(self, I: int, H: int) -> None:
        """Raise ``ValueError`` if a tile knob violates a StreamEP constraint
        that quack can't catch, for weight shapes ``(I, H)``. Two classes:

        * Output ``tile_n`` divisibility — one CTA per ``(tile_m, tile_n)``
          output tile; a non-dividing ``tile_n`` makes the last CTA write out of
          bounds (the streaming kernels A / Y bypass quack's epilogue
          predication, so quack can't catch it). Covers the four output
          ``tile_n`` knobs.
        * ``tile_m`` backward codegen — ``tile_m`` 192 / 320 make quack split
          the N dim across two MMA atoms (``atom_layout_n=2``), which the
          streaming backward GEMMs cannot CuTeDSL-codegen. 320 forces it always;
          192 only when a ``tile_n`` > 128. So ``tile_m=192`` is rejected unless
          every ``tile_n`` <= 128, and ``tile_m=320`` is rejected outright.

        :func:`stream_moe_func` calls this on the *resolved* config, so a bad
        caller override fails here rather than as a deep backward SIGABRT.
        ``None`` fields are skipped (they mean "auto-pick" and are resolved
        before a real launch). ``tile_m`` has no divisibility constraint — it
        rides the dynamic recv-token (M) dim (the scheduler pads partial tiles).
        Other quack CTA-shape rules (the ``tile_m`` allowlist, the
        ``tile_n`` %16/%32 bounds) are quack's to enforce; see the class
        docstring.
        """
        errs = []
        # tile_m 192/320 are the quack GemmSm90 CTA sizes that split the N
        # dimension across two MMA atoms (atom_layout_n=2). The streaming
        # BACKWARD GEMMs (kernel_y_bwd / kernel_a_bwd, plus the dW grouped GEMMs
        # that inherit tile_m / tile_n_a) cannot CuTeDSL-codegen that split
        # (verified: SIGABRT deep in backward). 320 forces it unconditionally;
        # 192 forces it only when a tile_n > 128. So tile_m=192 is allowed iff
        # every tile_n <= 128 (atom_layout_n=1); 320 is not. tile_m in
        # {64,128,256} keep atom_layout_n=1 and are unaffected. See
        # markdowns/reserved_memory_stream_segregation.md.
        if self.tile_m == 320:
            errs.append(
                "tile_m=320 forces quack atom_layout_n=2 unconditionally; the "
                "streaming backward GEMMs (kernel_y_bwd / kernel_a_bwd) cannot "
                "codegen the N-split. Use tile_m in {64, 128, 256}."
            )
        elif self.tile_m == 192:
            over = [
                f"{name}={v}"
                for name, v in (
                    ("tile_n_a", self.tile_n_a),
                    ("tile_n_y", self.tile_n_y),
                    ("tile_n_y_bwd", self.tile_n_y_bwd),
                    ("tile_n_a_bwd", self.tile_n_a_bwd),
                    ("tile_n_dW1", self.tile_n_dW1),
                    ("tile_n_dW2", self.tile_n_dW2),
                )
                if v is not None and v > 128
            ]
            if over:
                errs.append(
                    "tile_m=192 requires every tile_n <= 128 (else quack "
                    "atom_layout_n=2, which the streaming backward GEMMs cannot "
                    "codegen); tile_n > 128: " + ", ".join(over)
                )
        if self.tile_n_a is not None and (
            (2 * I) % self.tile_n_a != 0 or I % self.tile_n_a != 0
        ):
            errs.append(
                f"tile_n_a={self.tile_n_a} must divide both 2*I={2 * I} and "
                f"I={I} (fwd kernel A out N=2I; kernel_a also asserts "
                f"I % tile_n_a == 0)"
            )
        if self.tile_n_y is not None and H % self.tile_n_y != 0:
            errs.append(
                f"tile_n_y={self.tile_n_y} must divide H={H} (fwd kernel Y out N=H)"
            )
        if self.tile_n_y_bwd is not None and I % self.tile_n_y_bwd != 0:
            errs.append(
                f"tile_n_y_bwd={self.tile_n_y_bwd} must divide I={I} "
                f"(bwd kernel Y out N=I)"
            )
        if self.tile_n_a_bwd is not None and H % self.tile_n_a_bwd != 0:
            errs.append(
                f"tile_n_a_bwd={self.tile_n_a_bwd} must divide H={H} "
                f"(bwd kernel A out N=H)"
            )
        if errs:
            raise ValueError(
                f"TileConfig invalid for (I={I}, H={H}): " + "; ".join(errs)
            )


def default_tile_config(I: int, H: int) -> TileConfig:
    """Pick a tile config from the weight shapes.

    Prefers the bench-tuned defaults (tile_n_a=192, tile_n_y=256,
    tile_n_y_bwd=192, tile_n_a_bwd=256) where they satisfy the kernel's
    divisibility constraint; otherwise substitutes the largest power-of-2
    ≤ 256 that does. The constraints come from the grouped-GEMM tile
    scheduler — one CTA per (tile_m, tile_n) output tile; if ``N`` isn't an
    integer number of ``tile_n`` chunks the last CTA writes out of bounds.
    """
    two_I = 2 * I

    def pick(default: int, constraint: int) -> int:
        if constraint % default == 0:
            return default
        return _largest_pow2_divisor(constraint, cap=256)

    tile_n_a = pick(192, two_I)
    # kernel_a also asserts I % tile_n_a == 0 (kernel_a.py:485).
    if I % tile_n_a != 0:
        tile_n_a = _largest_pow2_divisor(I, cap=256)

    return TileConfig(
        tile_m=128,
        tile_n_a=tile_n_a,
        tile_n_y=pick(256, H),
        # y_bwd forces ab_stage=4 (deep W2/A mainloop prefetch — its scoreboard
        # stall is the long-K mainloop load; see StreamingMoeYBwd._compute_stages).
        # tile_n_y_bwd must stay small enough to leave SMEM for that 4th AB
        # stage; 192 fits at the supported shapes (H=3072, I=768). A larger
        # tile_n_y_bwd raises a descriptive ValueError from _compute_stages
        # rather than silently overflowing SMEM at launch.
        tile_n_y_bwd=pick(192, I),
        tile_n_a_bwd=pick(256, H),
        tile_m_dW1=None,
        tile_n_dW1=256,
        tile_m_dW2=None,
        tile_n_dW2=None,
        cluster_m_dW1=2,
        cluster_n_dW1=2,
        cluster_m_dW2=1,
        cluster_n_dW2=1,
        pingpong_dW1=False,
        pingpong_dW2=False,
        swizzle_dW1=8,
        swizzle_dW2=8,
        num_sms_a=None,
        num_sms_y=None,
        num_sms_a_bwd=None,
        num_sms_y_bwd=None,
    )


def _resolve_tile_config(
    tile_config: TileConfig | None, I: int, H: int
) -> TileConfig:
    """Resolve a caller override against the shape-picked baseline.

    Overlays ``tile_config``'s non-``None`` fields onto
    ``default_tile_config(I, H)`` and validates the result. ``None`` (the
    default for every ``TileConfig`` field) means "auto-pick", so a field the
    caller didn't set takes the baseline's shape-picked value rather than a
    frozen hardcoded default. ``tile_config=None`` returns the bare baseline.

    Caller ``None`` always resolves to the baseline value — a caller cannot
    request a field's kernel-level ``None`` fallback (e.g. ``tile_n_dW1`` →
    "use tile_n_a") through the public surface; that internal default is not
    something callers need, and the dW GEMMs predicate partial tiles anyway.
    """
    cfg = default_tile_config(I, H)
    if tile_config is not None:
        overrides = {
            f.name: getattr(tile_config, f.name)
            for f in fields(tile_config)
            if getattr(tile_config, f.name) is not None
        }
        if overrides:
            cfg = replace(cfg, **overrides)
    cfg.validate(I, H)
    return cfg


@dataclass(frozen=True)
class StreamHolder:
    """The two caller-owned streams driving one streaming-MoE layer.

    * ``communicate`` — dispatch + combine (fwd) and dispatch_grads +
      combine_grads (bwd). Single comm stream, FIFO-ordered.
    * ``compute`` — kernel_a + kernel_y (fwd) and
      kernel_y_bwd + kernel_a_bwd + dW1 + dW2 (bwd). Single compute stream,
      FIFO-ordered. dW lives on this stream because it SM-contends with
      kernel_a_bwd at the full 132-CTA grid anyway — same-stream FIFO is
      identical to a separate stream gated on compute.
    """

    communicate: torch.cuda.Stream
    compute: torch.cuda.Stream


def make_streams(
    device: torch.device | int | None = None,
    prioritize_communicate: bool = False,
) -> StreamHolder:
    """Allocate the two streams the layer expects.

    Caller creates this once (per training process) and reuses it across all
    layers and iterations — streams are not per-call state.

    When ``prioritize_communicate`` is True, the ``communicate`` stream is
    created at the highest available priority (Megatron-style pattern for
    comm/compute overlap under CUDA_DEVICE_MAX_CONNECTIONS>1). Lower
    priority number = higher priority in PyTorch's Stream API.
    """
    if prioritize_communicate:
        hi = -99
        lo = 0
        return StreamHolder(
            communicate=torch.cuda.Stream(device=device, priority=hi),
            compute=torch.cuda.Stream(device=device, priority=lo),
        )
    return StreamHolder(
        communicate=torch.cuda.Stream(device=device),
        compute=torch.cuda.Stream(device=device),
    )


# NVTX call counters — used to label fwd/bwd nsys ranges with a per-rank
# monotonic id. Set STREAM_EP_NVTX=1 in the env to enable. Off by default
# so we don't pay the nvtx push/pop overhead in production runs.
_NVTX_FWD_COUNT = 0
_NVTX_BWD_COUNT = 0
_NVTX_ENABLED = bool(os.environ.get("STREAM_EP_NVTX"))


def _nvtx_push(name: str) -> None:
    if _NVTX_ENABLED:
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    if _NVTX_ENABLED:
        torch.cuda.nvtx.range_pop()


class StreamMoEFunc(torch.autograd.Function):
    """Streaming-MoE layer as a differentiable autograd boundary.

    Forward: dispatch → kernel A → kernel Y → combine on the
    ``communicate`` / ``compute`` streams. A GPU-front-end
    ``wait_kernel_y_started`` queued on ``communicate`` before combine
    holds combine's launch until kernel Y's first CTA is co-resident on
    an SM, so kernel Y's 132 CTAs win the SM race against combine's
    80-CTA sender grid.

    Backward: dispatch_grads → kernel_y_bwd → kernel_a_bwd → combine_grads
    on the same ``communicate`` / ``compute`` streams; dW1 / dW2 FIFO-after
    kernel_a_bwd on the same compute stream.
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
        cfg: TileConfig,
        activation_checkpoint_level: int = 0,
    ) -> torch.Tensor:
        # activation_checkpoint_level: 0 = save preact_a + pool (no ckpt);
        # 1 = save pool, recompute preact_a in bwd; 2 = save compressed
        # recv-token x_recv, re-expand pool + recompute preact_a in bwd
        # (markdowns/act_ckpt_level2_plan.md).
        if activation_checkpoint_level not in (0, 1, 2):
            raise ValueError(f"activation_checkpoint_level must be 0/1/2, got {activation_checkpoint_level}")
        global _NVTX_FWD_COUNT
        _NVTX_FWD_COUNT += 1
        _fid = _NVTX_FWD_COUNT
        _nvtx_push(f"moe_fwd_{_fid}")

        # Forward reads only these knobs directly; the rest of ``cfg`` rides on
        # ctx for backward (see the ctx store below).
        tile_m = cfg.tile_m
        tile_n_a = cfg.tile_n_a
        tile_n_y = cfg.tile_n_y
        num_sms_a = cfg.num_sms_a
        num_sms_y = cfg.num_sms_y

        caller_stream = torch.cuda.current_stream()

        # Layer-start back-edges. Prevent iter N's persistent kernel A / Y
        # CTAs (and combine sender CTAs) from starving iter N+1's dispatch
        # on shared SMs. We gate both streams the layer will launch work on.
        streams.communicate.wait_stream(caller_stream)
        streams.compute.wait_stream(caller_stream)

        _nvtx_push(f"fwd_dispatch_{_fid}")
        with torch.cuda.stream(streams.communicate):
            pool, handle, metadata_done = buffer.dispatch(
                x,
                topk_idx,
                topk_weights,
                is_token_in_rank,
                num_experts,
                tile_m=tile_m,
            )
        _nvtx_pop()  # fwd_dispatch
        # Buffer-owned monotonic counter — ``handle.dispatch_seq`` is the
        # source of truth for the layer's seq. Threaded through kernel Y's
        # ``combine_seq`` (and ``buffer.combine`` defaults to it too).
        dispatch_seq = handle.dispatch_seq

        # Cross-stream visibility from communicate → compute. ``metadata_done``
        # is recorded by dispatch *between* the metadata kernel and
        # dispatch_main, so waiting on it lets kernel A launch BEFORE
        # dispatch_main retires — kernel A's scheduler then spins on
        # ``pool_arrival_count`` (which dispatch_main's Pass 2 release-adds
        # fill per-tile) for the actual dispatch ↔ A streaming overlap.
        # The Z_pre + N + Z_post slab regions in ``PostPollBundle`` are all
        # zeroed BEFORE this event records (the CDMC>1 fix in
        # ``allocate_post_poll_bundle``), so every handle tensor is visible
        # the moment ``metadata_done`` fires.
        streams.compute.wait_event(metadata_done)

        # GPU-front-end gate: hold kernel_a's launch on ``streams.compute``
        # until dispatch_main_kernel's block 0 has actually entered execution
        # (atomicAdd from block 0 thread 0 → cuStreamWaitValue ge). Without
        # this, under CDMC>1, kernel_a's 132-CTA grid can grab SMs before
        # dispatch_main is co-resident, leaving dispatch_main queued behind
        # kernel_a's CTAs forever. Wait sits at the GPU front-end and does
        # NOT consume an SM.
        buffer.wait_dispatch_main_started(streams.compute)

        # ── Kernel A on `streams.compute` ─────────────────────────────────
        # dispatch_main has retired (wait_stream above), but its per-tile
        # release-add into `pool_arrival_count` runs *before* dispatch_main
        # finishes — kernel A's scheduler spin on
        # `pool_arrival_count[tile] == pool_arrival_target[tile]` re-claims
        # those, terminating immediately on every tile.
        # Allocate kernel A/Y outputs on the CALLER stream (default pool), not the
        # compute stream, so the default pool can reuse this memory across layers
        # (and absorb the model's freed activations) instead of carving a separate
        # compute-stream pool. The kernels below still run on streams.compute.
        # NO record_stream: free-side safety comes from the layer-end
        # caller.wait_stream(compute) back-edge (these are locals freed at return,
        # after that gate, so any reuse is ordered after compute). record_stream
        # would instead defer reclamation behind the lagging compute stream and
        # bloat the pool (measured: +9 GB of active_pending_free).
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
        # it for SwiGLU bwd in registers and for postact_a recompute. The
        # cheap thing element-wise-recomputes from preact in bwd, the
        # expensive thing (preact, would otherwise need a GEMM to recover)
        # is the one we save.
        #
        # At ``activation_checkpoint_level >= 1`` we DON'T save preact_a: the fwd
        # ``[2I]`` TMA-store is skipped (preact_a=None below) and bwd
        # recomputes it from the saved ``pool`` @ ``w1_local`` (one
        # kernel-A-sized GEMM that overlaps with dispatch_grads' comm tail).
        # Trades ~2I·2 B/slot of saved activation per layer for that GEMM.
        preact_a = (
            None
            if activation_checkpoint_level >= 1
            else torch.empty(
                handle.total_tiles,
                tile_m,
                2 * w2_local.shape[2],
                dtype=x.dtype,
                device=pool.device,
            )
        )
        # Order compute after caller before its FIRST use of postact_a/preact_a
        # above (alloc→first-use safety: a recycled caller-pool block's previous
        # caller-stream kernel must finish before compute touches it). Cheap here:
        # caller has no GPU work outstanding at this point (dispatch runs on
        # communicate), so it doesn't serialize the dispatch ↔ kernel-A overlap.
        streams.compute.wait_stream(caller_stream)

        _nvtx_push(f"fwd_compute_{_fid}")
        with torch.cuda.stream(streams.compute):
            # Empty-rank guard: when no token is routed to any of this rank's
            # local experts, ``total_tiles == 0`` ⟹ ``T_recv == 0``. Skipping
            # kernel A / kernel Y is required because their CuTeDSL launches
            # reject grid_x=0 with cudaErrorInvalidConfiguration. The bump /
            # wait pair around kernel_y is skipped together — host counter
            # and device started_flag stay in sync at their prior values.
            # ``dispatch_main`` always launched (sender side has source-token
            # work regardless), so the wait_dispatch_main_started flag is
            # already bumped from above; no skip there. ``combine`` below
            # still runs — its receiver side gathers contributions for our
            # source tokens, and the sender side has T_recv=0 work.
            if handle.total_tiles > 0:
                streaming_moe_a(
                    pool,
                    w1_local,
                    postact_a,
                    handle.expert_pool_block_offset,
                    handle.pool_arrival_count,
                    handle.pool_arrival_target,
                    preact_a=preact_a,
                    tile_m=tile_m,
                    tile_n=tile_n_a,
                    cluster_n=2,
                    num_sms=num_sms_a,
                )

                # ── Kernel Y on the same compute stream ──────────────────────
                # Bump the kernel_y issued counter BEFORE launching so a
                # subsequent ``wait_kernel_y_started`` on the communicate stream
                # has the right comparison target. Inside the kernel, the CTA
                # that wins ``linear_idx == 0`` in StreamingTileScheduler
                # atomicAdd's ``kernel_y_started_flag`` — a stronger signal than
                # a torch.cuda.Event between A and Y, which only marks "Y's
                # launch packet is queued" rather than "Y is on an SM".
                buffer.bump_kernel_y_issued()
                streaming_moe_y(
                    postact_a,
                    w2_local,
                    handle.o,
                    handle.pool_recv_token,
                    handle.pool_topk_weight,
                    handle.k_local_remaining,
                    handle.y_done_per_token,
                    handle.expert_pool_block_offset,
                    handle.pool_arrival_count,
                    handle.pool_arrival_target,
                    combine_seq=dispatch_seq,
                    started_flag=buffer.kernel_y_started_flag_view(),
                    tile_m=tile_m,
                    tile_n=tile_n_y,
                    cluster_n=2,
                    num_sms=num_sms_y,
                )
        _nvtx_pop()  # fwd_compute

        # ── Combine on `streams.communicate` (same stream as dispatch) ────
        # FIFO-ordered after dispatch_main. Waits on the
        # ``kernel_y_started_flag`` cross-stream launch gate so combine's
        # 80-CTA sender grid doesn't grab SMs before kernel_y's 132 CTAs
        # have at least one block co-resident. Combine's per-warp sender
        # then spins on ``y_done_per_token[r] >= dispatch_seq`` for
        # per-recv-token streaming.
        _nvtx_push(f"fwd_combine_{_fid}")
        if handle.total_tiles > 0:
            buffer.wait_kernel_y_started(streams.communicate)
        with torch.cuda.stream(streams.communicate):
            # combine_seq defaults to ``handle.dispatch_seq`` per Buffer.combine.
            out, _ = buffer.combine(handle.o, handle)
        _nvtx_pop()  # fwd_combine

        # ── Level 2: compress pool -> x_recv on the CALLER stream, overlapping
        # combine. caller.wait_stream(compute) syncs the caller past kernel Y
        # (hence past kernel A's gated pool reads), so pool is fully materialized
        # and readable here; x_recv is then allocated AND written on the caller
        # stream -> a pure default-pool tensor (no record_stream, no cross-stream
        # bracketing). pool is freed right after (its last reader, this
        # compaction, is done) with no extra wait — combine reads `o`, never
        # pool. The saved x_recv (T_recv rows) replaces the expanded, 128-padded
        # pool (TK_padded rows); bwd re-expands it. markdowns/act_ckpt_level2_plan.md sec 4.
        x_recv = None
        if activation_checkpoint_level == 2:
            caller_stream.wait_stream(streams.compute)
            # Allocate x_recv on the CALLER stream (current here) -> default pool,
            # then fill in place. A pure default-pool tensor: no record_stream.
            x_recv = torch.empty(
                handle.recv_token_to_slots.shape[0], pool.shape[1],
                dtype=pool.dtype, device=pool.device,
            )
            compress_pool_to_recv(pool, handle.pool_recv_token, x_recv)
            # Free pool: drop ALL THREE refs. handle.pool=None + the local are
            # not enough — handle._dispatch_out (the C++ struct) holds the other
            # ref (exactly like `o`/release_o), so without release_pool() pool
            # stays pinned through bwd and L2 saves NOTHING. dispatch_grads reads
            # handle.TK_padded now (not pool.shape); bwd re-expands from x_recv.
            handle.pool = None
            if handle._dispatch_out is not None:
                handle._dispatch_out.release_pool()
            pool = None

        # Layer-end back-edges. caller_stream waits on both streams so the
        # layer-as-barrier invariant holds across layers.
        caller_stream.wait_stream(streams.communicate)
        caller_stream.wait_stream(streams.compute)

        # Release o from the saved state, AFTER the back-edges. o is forward-only
        # — kernel Y's atomic-scatter target, consumed by combine above, never
        # read in bwd — but ctx.save_for_backward holds the whole handle, so
        # without this it would be pinned through backward (one ~T_recv×hidden
        # buffer per layer). o is allocated in the DEFAULT pool WITHOUT
        # record_stream (csrc allocate_post_poll_bundle, like `pool`), so its
        # free-side safety is exactly these back-edges: releasing it here orders
        # any caller-FIFO reuse of its storage after kernel Y (compute) and
        # combine (communicate), both of which caller_stream just joined. Doing
        # this after the back-edges (vs mid-forward) preserves the bwd-peak
        # benefit — the storage is reclaimable before the bwd peak either way —
        # while letting o drop record_stream. The handle's _dispatch_out (the C++
        # StreamingDispatchOutputs, saved for bwd internode routing) holds the
        # OTHER reference, so handle.o=None alone wouldn't free it — release_o()
        # drops the struct's ref too. All other _dispatch_out members are
        # bwd-needed, so only o is released.
        handle.o = None
        if handle._dispatch_out is not None:
            handle._dispatch_out.release_o()

        # Save tensors bwd will consume. The contract is "save preact, drop
        # postact" — preact_a is the strictly more useful saved activation:
        # kernel_y_bwd reads it as mC for in-epilogue SwiGLU bwd, and the
        # orchestrator element-wise-recomputes postact_a from it (silu(gate)
        # * up) into a transient buffer just before dW2's grouped GEMM.
        #
        # At level >= 1, preact_a was never materialized (None); bwd recomputes
        # it from the saved pool @ w1_local. Save only the GEMM inputs so
        # save_for_backward stays a homogeneous tensor tuple.
        ctx.activation_checkpoint_level = activation_checkpoint_level
        if activation_checkpoint_level == 2:
            ctx.save_for_backward(x_recv, w1_local, w2_local)
        elif activation_checkpoint_level == 1:
            ctx.save_for_backward(pool, w1_local, w2_local)
        else:
            ctx.save_for_backward(preact_a, pool, w1_local, w2_local)
        ctx.streams = streams
        ctx.buffer = buffer
        ctx.handle = handle
        # The full tuning config rides on ctx for backward — a small frozen
        # dataclass of ints/bools, no tensors, so (unlike streams/buffer/handle)
        # it needs no eager release. The num_sms_*_bwd "fall back to the fwd
        # count when None" derivation happens in backward, off ``cfg``.
        ctx.cfg = cfg
        _nvtx_pop()  # moe_fwd
        return out

    @staticmethod
    def backward(ctx, dL_dy):  # type: ignore[override]
        """Backward on the same two streams as fwd.

        Stages by stream:
          ``streams.communicate``: dispatch_grads → combine_grads (FIFO).
          ``streams.compute``: kernel_y_bwd → kernel_a_bwd → dW2 → dW1 (FIFO).

        Cross-stream sync — mirrors fwd's three real overlap windows:

          ``streams.compute.wait_event(grads_started)`` after dispatch_grads
              launches — analogous to fwd's
              ``streams.compute.wait_event(metadata_done)``. The event is
              recorded by ``buffer.dispatch_grads`` between the channel-
              control barrier and ``dispatch_grads_main`` so kernel_y_bwd
              can launch CONCURRENT with dispatch_grads_main; its scheduler
              spins on ``bwd_dispatch_arrival_count`` per-tile.
          ``buffer.wait_dispatch_grads_started`` — GPU-front-end gate on
              ``streams.compute`` that holds kernel_y_bwd's launch until
              dispatch_grads_main's block 0 has entered execution. Same
              rationale as the fwd ``wait_dispatch_main_started`` gate.
          ``buffer.wait_kernel_a_bwd_started`` on ``communicate`` before
              combine_grads — analogous to fwd's
              ``wait_kernel_y_started``. The orchestrator bumps
              ``kernel_a_bwd_issued`` before launching A_bwd on
              ``compute``; the GPU-front-end wait holds combine_grads
              until A_bwd's first CTA is co-resident on an SM, so A_bwd's
              132 CTAs win the SM race against combine_grads's 80. Per-
              recv-token streaming inside combine_grads is driven by
              ``bwd_a_done_per_token[r] >= dispatch_seq`` as kernel_a_bwd
              tiles retire.

        dW1 / dW2 live on the SAME compute stream, FIFO after kernel_a_bwd.
        They cannot overlap with A_bwd (both saturate the 132-CTA grid),
        but they DO overlap with combine_grads on the communicate stream —
        combine_grads's 80 sender CTAs spend much of their duration gated
        on per-token signals / NVLink completion, leaving SMs idle for
        dW's GEMMs. Same-stream FIFO after A_bwd handles dW's inputs from
        kernel_y_bwd (dL_dswiglu_in, postact_a_for_dW2) directly; dW2's
        read of dL_do_pool (written by dispatch_grads on communicate) is
        covered transitively because kernel_y_bwd's per-tile
        ``bwd_dispatch_arrival_count`` acquire-chain already published
        dispatch_grads's writes to the compute stream.

        Within ``streams.compute`` the kernel_y_bwd → kernel_a_bwd handoff
        is implicit same-stream FIFO — no per-tile release-add is needed.

        Returns gradients matching forward's args, in order.
        """
        global _NVTX_BWD_COUNT
        _NVTX_BWD_COUNT += 1
        _bid = _NVTX_BWD_COUNT
        _nvtx_push(f"moe_bwd_{_bid}")

        # At level >= 1 the fwd saved only the recompute inputs; preact_a is
        # reconstructed below from pool @ w1_local before kernel_y_bwd reads it.
        # Level 2 saved the compressed x_recv (not pool) — pool is re-expanded
        # from it after the dispatch_grads gate (sec 6). Level 0 saved preact_a.
        if ctx.activation_checkpoint_level == 2:
            x_recv, w1_local, w2_local = ctx.saved_tensors
            pool = None      # re-expanded below (only when total_tiles > 0)
            preact_a = None
        elif ctx.activation_checkpoint_level == 1:
            pool, w1_local, w2_local = ctx.saved_tensors
            preact_a = None  # recomputed after the dispatch_grads gate
        else:
            preact_a, pool, w1_local, w2_local = ctx.saved_tensors
        streams: StreamHolder = ctx.streams
        buffer: StreamEPBuffer = ctx.buffer
        handle = ctx.handle
        # Drop ctx attributes immediately. autograd's grad_fn owns ctx and is
        # held by the output tensor's graph until that output is dropped —
        # in a multi-layer model that means every layer's ctx (and every
        # tensor it references) stays pinned through the iteration. The
        # saved-tensors machinery handles its own release after backward
        # returns; the side-channel attributes do not, so explicitly clear
        # them so handle's tensors (pool_arrival_count et al.) can be freed
        # at this layer's bwd return rather than at the end of the iter.
        del ctx.streams, ctx.buffer, ctx.handle
        cfg: TileConfig = ctx.cfg
        tile_m: int = cfg.tile_m
        tile_n_a: int = cfg.tile_n_a
        tile_n_y_bwd: int = cfg.tile_n_y_bwd
        tile_n_a_bwd: int = cfg.tile_n_a_bwd
        tile_m_dW1: int = cfg.tile_m_dW1 if cfg.tile_m_dW1 is not None else tile_m
        tile_n_dW1: int = cfg.tile_n_dW1 if cfg.tile_n_dW1 is not None else tile_n_a
        tile_m_dW2: int = cfg.tile_m_dW2 if cfg.tile_m_dW2 is not None else tile_m
        tile_n_dW2: int = cfg.tile_n_dW2 if cfg.tile_n_dW2 is not None else tile_n_a
        cluster_m_dW1: int = cfg.cluster_m_dW1
        cluster_n_dW1: int = cfg.cluster_n_dW1
        cluster_m_dW2: int = cfg.cluster_m_dW2
        cluster_n_dW2: int = cfg.cluster_n_dW2
        pingpong_dW1: bool = cfg.pingpong_dW1
        pingpong_dW2: bool = cfg.pingpong_dW2
        swizzle_dW1: int = cfg.swizzle_dW1
        swizzle_dW2: int = cfg.swizzle_dW2
        # bwd kernels take the *_bwd SM counts (falling back to the fwd count
        # when unset); the fwd-only num_sms_a / num_sms_y aren't used here.
        num_sms_a_bwd: int | None = (
            cfg.num_sms_a_bwd if cfg.num_sms_a_bwd is not None else cfg.num_sms_a
        )
        num_sms_y_bwd: int | None = (
            cfg.num_sms_y_bwd if cfg.num_sms_y_bwd is not None else cfg.num_sms_y
        )

        # Upstream may pass a non-contiguous grad (e.g. `out.sum().backward()`
        # produces a stride-(0,0) broadcast view); `dispatch_grads` asserts
        # contiguity, so normalise here.
        dL_dy = dL_dy.contiguous()

        caller_stream = torch.cuda.current_stream()
        # device/dtype from dL_dy (always present, same activation dtype as pool)
        # so this works at level 2 where pool is None until re-expanded.
        device = dL_dy.device
        dtype = dL_dy.dtype

        E_local, two_I, H = w1_local.shape
        I = two_I // 2
        total_tiles = handle.total_tiles
        TK_padded = handle.TK_padded  # was pool.shape[0]; pool may be None at level 2
        T_recv = handle.k_local_total.shape[0]

        # Allocate the backward scratch on the CALLER stream (default pool), not
        # streams.compute, so it doesn't carve a segregated compute-stream pool.
        # Memsets for the zero-init targets run on streams.compute after the
        # fan-out gate below. Free-side safety is the layer-end back-edges (see
        # the note below the allocations), so no record_stream here.
        dL_dx_per_r = torch.empty(T_recv, H, dtype=dtype, device=device)
        bwd_k_local_remaining = torch.empty(T_recv, dtype=torch.int32, device=device)
        bwd_a_done_per_token = torch.empty(T_recv, dtype=torch.int64, device=device)
        dL_dswiglu_in = torch.empty(
            total_tiles, tile_m, two_I, dtype=dtype, device=device
        )
        postact_a_for_dW2 = torch.empty(
            total_tiles, tile_m, I, dtype=dtype, device=device
        )
        # dL_dweight: per-pid_n fp32 atomic-add target; MUST zero-init.
        dL_dweight = torch.empty(TK_padded, dtype=torch.float32, device=device)
        # quack `gemm` with default `add_to_output=False` overwrites D, so neither
        # dW destination needs zero-init.
        dW1_local = torch.empty_like(w1_local)
        dW2_local = torch.empty_like(w2_local)
        # NO record_stream on the scratch above. It's caller-allocated and freed
        # as locals at bwd return — AFTER the layer-end
        # caller.wait_stream(compute/communicate) back-edges — so any later reuse
        # (next layer's scratch or the model's activations) is ordered after both
        # consumer streams finish with it. record_stream would instead defer
        # reclamation behind the lagging side streams, piling up pending-free
        # blocks the default pool can't reuse (measured: +9 GB).

        # ── Single fan-out gate on caller_stream ───────────────────────────
        # MUST precede the FIRST side-stream use of the caller-allocated scratch
        # above: ``torch.empty`` may hand back caller-pool blocks whose previous
        # caller-stream kernel is still in flight, so compute/communicate must
        # join caller's completion state before touching them (alloc→first-use).
        # (Also required before the FWD-written handle reads below.)
        streams.communicate.wait_stream(caller_stream)
        streams.compute.wait_stream(caller_stream)

        # Zero-init targets — AFTER the fan-out gate so the memsets are ordered
        # after caller's prior use of any recycled block.
        with torch.cuda.stream(streams.compute):
            dL_dx_per_r.zero_()
            bwd_a_done_per_token.zero_()
            dL_dweight.zero_()

        # Reads of FWD-written handle tensors (expert_pool_block_offset,
        # expert_frequency, k_local_total — all on the fwd post-poll bundle,
        # written on FWD's communicate stream) MUST happen AFTER the fan-out
        # gate above so compute has joined caller_stream's FWD-completion
        # state. Doing these `.to()` / arithmetic before the gate was a real
        # bug at internode: the .to() reads on compute would race against
        # FWD's communicate writes and pull stale / partially-initialized
        # values, producing wrong cu_seqlens_k → wrong K-tile bounds in dW2
        # → reads of garbage padding rows of pool → NaN dW2 grads on
        # specific experts.
        with torch.cuda.stream(streams.compute):
            cu_seqlens_k = (
                handle.expert_pool_block_offset.to(torch.int32) * tile_m
            ).contiguous()
            lens_k_dW = handle.expert_frequency.to(torch.int32)
            bwd_k_local_remaining.copy_(handle.k_local_total, non_blocking=True)

        # ── Stage 1 — dispatch_grads on streams.communicate ────────────────
        # Ships dL/dy origin → expert ranks along fwd's routing, K-fans into
        # dL_do_pool[slot]. Pass 2's `red.release.gpu.add.s32` builds
        # bwd_dispatch_arrival_count; kernel_y_bwd's scheduler spins on
        # count == pool_arrival_target. The returned `grads_started_event`
        # is recorded between the channel-control barrier and
        # dispatch_grads_main's launch — analogous to fwd's metadata_done.
        with torch.cuda.stream(streams.communicate):
            dL_do_pool, bwd_dispatch_arrival_count, grads_started = (
                buffer.dispatch_grads(
                    handle, dL_dy, dispatch_seq=handle.dispatch_seq
                )
            )

        # Cross-stream visibility from communicate → compute. Waiting on
        # `grads_started` (recorded BEFORE dispatch_grads_main launches) lets
        # kernel_y_bwd launch concurrent with dispatch_grads_main; the
        # per-tile bwd_dispatch_arrival_count release-adds inside
        # dispatch_grads drive the streaming overlap (analogous to fwd's
        # pool_arrival_count chain).
        streams.compute.wait_event(grads_started)

        # GPU-front-end gate: queue a cuStreamWaitValue so kernel_y_bwd's
        # launch cannot grab SMs before dispatch_grads_main's block 0 has
        # entered execution. Mirror of the fwd dispatch ↔ kernel_a gate
        # above; same rationale.
        buffer.wait_dispatch_grads_started(streams.compute)

        # ── Level 2: re-expand x_recv -> pool on the compute stream, overlapping
        # dispatch_grads's comm tail (the same window Stage 1.5 uses). FIFO-before
        # Stage 1.5 (preact recompute) and Stage 5 (dW1) — pool's only bwd
        # consumers. Transient: a bwd local on the compute pool, freed at bwd
        # return; one layer live at a time, so it never touches the fwd-end peak.
        # markdowns/act_ckpt_level2_plan.md sec 6.
        if ctx.activation_checkpoint_level == 2 and total_tiles > 0:
            # Allocate the transient pool on the CALLER stream -> default pool
            # (so it reuses freed x_recv space as bwd drains them, instead of
            # stranding in the compute pool); fill it in place on the compute
            # stream. Same caller-alloc / compute-write pattern as the bwd scratch
            # (alloc->first-use via the fan-out gate above; free-side via the
            # bwd back-edges). NO record_stream.
            pool = torch.empty(TK_padded, H, dtype=dtype, device=device)
            with torch.cuda.stream(streams.compute):
                reexpand_recv_to_pool(x_recv, handle.pool_recv_token, pool)

        # ── Stage 1.5 — recompute preact_a (checkpoint level >= 1 only) ─────
        # Reconstruct the [2I] pre-SwiGLU accumulator that fwd kernel A would
        # have TMA-stored, by re-running the same kernel-A GEMM on the saved
        # pool @ w1_local. Same kernel + same inputs ⟹ preact_a is bit-
        # identical to the value we'd have saved, so kernel_y_bwd's SwiGLU bwd
        # is unchanged. The recompute reads ONLY saved tensors (pool, w1) —
        # never dispatch_grads' output — so it overlaps with dispatch_grads_
        # main's NVLink/RDMA-bound tail on the communicate stream. It sits
        # AFTER the wait_dispatch_grads_started gate so dispatch_grads_main is
        # already co-resident and the recompute's 132-CTA grid can't starve
        # it. Immediate-claim: pass pool_arrival_target as BOTH count and
        # target so the scheduler's count-vs-target spin terminates on the
        # first read (every tile already "arrived") — the recompute must NOT
        # couple to dispatch_grads' per-tile fill of bwd_dispatch_arrival_count.
        # postact (silu(gate)*up) is a throwaway here; kernel_y_bwd recomputes
        # its own pool_topk_weight-scaled postact_a_for_dW2 from preact_a.
        if ctx.activation_checkpoint_level >= 1 and total_tiles > 0:
            with torch.cuda.stream(streams.compute):
                preact_a = torch.empty(
                    total_tiles, tile_m, two_I, dtype=dtype, device=device
                )
                # Recompute preact = pool @ w1 with a plain grouped-M GEMM (pool
                # is fully materialized from the fwd save, so a streaming kernel
                # would be pure overhead). cu_seqlens_m = the per-expert padded
                # token offsets (identical to dW's cu_seqlens_k); the per-expert
                # weight w1[e] is selected by the M-group index. Padding rows are
                # computed (each M-row is independent) and masked downstream by
                # kernel_y_bwd's mPaddingMask, exactly as the fwd kernel-A path.
                # Not bit-identical to fwd's preact (different tiling) but same
                # math within the bf16 recompute-noise floor checkpointing
                # tolerates. MUST stay after the wait_event(grads_started) +
                # wait_dispatch_grads_started gates above so dispatch_grads_main
                # is co-resident first and this GEMM only fills leftover SMs.
                gemm(
                    pool,                              # A (TK_padded, H) k-major
                    w1_local,                          # B (E_local, 2I, H)=(l,n,k)
                    preact_a.view(TK_padded, two_I),   # D (TK_padded, 2I) n-major
                    None,                              # C
                    None,                              # tile_count_semaphore
                    tile_M=tile_m,
                    tile_N=tile_n_a,
                    cluster_M=1,
                    cluster_N=1,
                    cu_seqlens_m=cu_seqlens_k,         # per-expert padded token offsets
                )

        # ── Stage 2 — kernel_y_bwd on streams.compute ──────────────────────
        # Scheduler count-vs-target spin on bwd_dispatch_arrival_count[tile]
        # == pool_arrival_target[tile] terminates immediately because
        # dispatch_grads has fully retired (wait_stream above).
        # Writes dL_dswiglu_in, postact_a_for_dW2, dL_dweight (atomic).
        # Empty-rank guard: mirror of fwd — skip kernel_y_bwd / kernel_a_bwd
        # (and their bump/wait pair) when this rank produced no tiles in
        # fwd. combine_grads still runs; dW1/dW2 GEMMs are skipped and the
        # dW destinations are zero'd below.
        if total_tiles > 0:
            with torch.cuda.stream(streams.compute):
                streaming_moe_y_bwd(
                    dL_do_pool,
                    w2_local,
                    dL_dswiglu_in,
                    postact_a_for_dW2,
                    handle.pool_topk_weight,
                    handle.pool_recv_token,
                    preact_a,
                    dL_dweight,
                    handle.expert_pool_block_offset,
                    bwd_dispatch_arrival_count,
                    handle.pool_arrival_target,
                    tile_m=tile_m,
                    tile_n=tile_n_y_bwd,
                    num_sms=num_sms_y_bwd,
                )

                # ── Stage 3 — kernel_a_bwd on the SAME compute stream ──────
                # FIFO-ordered after kernel_y_bwd retires — no cross-stream
                # release-add is needed. Scheduler reuses
                # (bwd_dispatch_arrival_count, pool_arrival_target) — at-target
                # by the time A_bwd runs because Y_bwd already spun on it.
                # Bump the kernel_a_bwd issued counter BEFORE launching so the
                # communicate-stream wait below has the right target. The CTA
                # that wins ``linear_idx == 0`` in StreamingTileScheduler
                # atomicAdd's ``kernel_a_bwd_started_flag``; the
                # ``wait_kernel_a_bwd_started`` GPU-front-end gate on
                # ``communicate`` reads this flag, so combine_grads cannot
                # grab SMs ahead of A_bwd's first co-resident CTA.
                buffer.bump_kernel_a_bwd_issued()
                streaming_moe_a_bwd(
                    dL_dswiglu_in,
                    w1_local,
                    dL_dx_per_r,
                    handle.pool_recv_token,
                    bwd_k_local_remaining,
                    bwd_a_done_per_token,
                    handle.expert_pool_block_offset,
                    bwd_dispatch_arrival_count,
                    handle.pool_arrival_target,
                    dispatch_seq=handle.dispatch_seq,
                    started_flag=buffer.kernel_a_bwd_started_flag_view(),
                    tile_m=tile_m,
                    tile_n=tile_n_a_bwd,
                    num_sms=num_sms_a_bwd,
                )

        # Combine_grads launches when A_bwd has at least one CTA co-resident.
        # Per-recv-token streaming overlap (combine_grads sender ↔ A_bwd
        # tail) is driven by `bwd_a_done_per_token[r] >= dispatch_seq`,
        # fired by A_bwd's last contributor per recv-token.
        if total_tiles > 0:
            buffer.wait_kernel_a_bwd_started(streams.communicate)

        # ── Stage 4 — combine_grads on streams.communicate ─────────────────
        # Sender per-warp loop spins on bwd_a_done_per_token[r] >=
        # dispatch_seq before reading dL_dx_per_r[r] AND dL_dweight[slot]
        # (gathered via recv_token_to_slots[r, k]).
        with torch.cuda.stream(streams.communicate):
            dL_dx, dL_dtopk_weights = buffer.combine_grads(
                dL_dx_per_r,
                handle,
                dL_dweight,
                bwd_a_done_per_token,
                dispatch_seq=handle.dispatch_seq,
            )

        # ── Stage 5 — dW1 / dW2 grouped GEMMs on streams.compute ──────────
        # Both dWs are purely local (no cross-rank). They depend on
        # kernel_y_bwd's outputs (dL_dswiglu_in / postact_a_for_dW2) and
        # dW2 reads dL_do_pool (written by dispatch_grads on communicate).
        # Running on the SAME compute stream after kernel_a_bwd:
        #   * same-stream FIFO covers Y_bwd → dW inputs and the
        #     dispatch_grads → dW2 path (transitively via kernel_y_bwd's
        #     per-tile acquire-chain that already published dispatch_grads's
        #     writes on compute);
        #   * dW cannot overlap with kernel_a_bwd (both saturate 132 SMs),
        #     so FIFO ordering is identical in cost to a separate stream
        #     gated on compute — a dedicated ``dw`` stream would buy nothing;
        #   * dW DOES overlap with combine_grads on the communicate stream
        #     (which is gated on the kernel_a_bwd_started flag and runs
        #     comm-bound), so the useful dW ↔ combine_grads overlap is
        #     preserved.
        # Empty-rank guard: skip the dW1/dW2 grouped GEMMs and zero the
        # destinations. With ``total_tiles == 0`` the per-expert K-lens are
        # all zero — the GEMMs would have zero work, but quack's grouped-GEMM
        # path isn't guaranteed to handle all-zero ``cu_seqlens_k`` / ``lens_k``
        # gracefully. dW1_local / dW2_local were allocated as
        # ``torch.empty_like`` so they hold uninitialized garbage; the genuine
        # gradient is zero, so explicit ``.zero_()`` is correct.
        if total_tiles > 0:
            with torch.cuda.stream(streams.compute):
                # dW2[e] = postact_a_for_dW2[slot_range_e].T @ dL_do_pool[slot_range_e]
                postact_a_for_dW2_flat = postact_a_for_dW2.view(TK_padded, I)
                gemm(
                    dL_do_pool.t(),
                    postact_a_for_dW2_flat.t(),
                    dW2_local,
                    None,
                    None,
                    tile_M=tile_m_dW2,
                    tile_N=tile_n_dW2,
                    cluster_M=cluster_m_dW2,
                    cluster_N=cluster_n_dW2,
                    pingpong=pingpong_dW2,
                    max_swizzle_size=swizzle_dW2,
                    cu_seqlens_k=cu_seqlens_k,
                    lens_k=lens_k_dW,
                )
                # dW1[e] = (dL_dswiglu_in[slot_range_e]).T @ pool[slot_range_e]
                dL_dswiglu_in_flat = dL_dswiglu_in.view(TK_padded, two_I)
                gemm(
                    dL_dswiglu_in_flat.t(),
                    pool.t(),
                    dW1_local,
                    None,
                    None,
                    tile_M=tile_m_dW1,
                    tile_N=tile_n_dW1,
                    cluster_M=cluster_m_dW1,
                    cluster_N=cluster_n_dW1,
                    pingpong=pingpong_dW1,
                    max_swizzle_size=swizzle_dW1,
                    cu_seqlens_k=cu_seqlens_k,
                    lens_k=lens_k_dW,
                )
        else:
            with torch.cuda.stream(streams.compute):
                dW1_local.zero_()
                dW2_local.zero_()

        # ── Exit chain back to caller_stream ───────────────────────────────
        caller_stream.wait_stream(streams.communicate)
        caller_stream.wait_stream(streams.compute)

        # Outputs of combine_grads were allocated on streams.communicate;
        # dW1 / dW2 on streams.compute. Record them on caller_stream so
        # the caching allocator can recycle correctly once the upstream
        # backward consumes them.
        dL_dx.record_stream(caller_stream)
        dL_dtopk_weights.record_stream(caller_stream)
        dW1_local.record_stream(caller_stream)
        dW2_local.record_stream(caller_stream)

        _nvtx_pop()  # moe_bwd

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
            None,  # cfg (TileConfig)
            None,  # activation_checkpoint_level
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
    tile_config: TileConfig | None = None,
    activation_checkpoint_level: int = 0,
) -> torch.Tensor:
    """One MoE forward layer: dispatch + kernel A + kernel Y + combine.

    Tile / cluster / swizzle / persistent-kernel-SM tuning comes from
    ``tile_config``, an all-optional :class:`TileConfig` of overrides. Every
    unset (``None``) field — which is all of them when ``tile_config`` is
    ``None`` (the default) — is auto-picked from ``w1_local``'s shape via
    :func:`default_tile_config` (bench-tuned defaults with a power-of-2
    fallback per kernel's tile_n divisibility constraint). To pin specific
    knobs, pass e.g. ``tile_config=TileConfig(tile_n_a=128, num_sms_a=120)``;
    the rest stay shape-picked, and the resolved config is validated against
    ``w1_local``'s shape before launch (:meth:`TileConfig.validate`).

    Returns the cross-rank-reduced output of shape ``[num_tokens, hidden]``
    produced by the combine receiver — the standard MoE forward output for
    this rank's source tokens.

    ``activation_checkpoint_level`` (default 0):
      * 0 — save both ``preact_a`` and ``pool`` (no recompute).
      * 1 — DON'T save the ``[2I]`` pre-SwiGLU accumulator ``preact_a``
        (saving ~2I·dtype_size bytes/recv-slot/layer); backward recomputes it
        from the saved ``pool @ w1_local`` (one kernel-A-sized GEMM that
        overlaps with ``dispatch_grads``' comm tail). Trades that activation
        memory for bwd compute — use when a layer's saved-activation footprint
        OOMs (e.g. the 82B shape on 80 GB H100 vs 141 GB H200).
      * 2 — additionally save only the *unexpanded* recv tokens
        ``x_recv [T_recv, H]`` instead of the expanded, 128-padded ``pool``,
        re-expanding ``pool`` in backward. NOT YET IMPLEMENTED (raises);
        see ``markdowns/act_ckpt_level2_plan.md``.

    Compile interaction: this entry point is eager-only. Callers that want
    ``torch.compile`` around the outer model must apply
    ``@torch.compiler.disable`` at the consumer boundary; see this module's
    docstring for the underlying constraint.
    """
    # w1_local shape is [E_local, 2*I, H]; derive I and H to pick / validate tiles.
    _, two_I, H = w1_local.shape
    I = two_I // 2
    # Overlay the caller's non-None overrides onto the shape-picked baseline,
    # then validate; unset knobs are auto-picked for this shape.
    cfg = _resolve_tile_config(tile_config, I, H)

    # Register the compute stream so the C++ Buffer's per-dispatch slabs
    # (``torch::empty``'d on the communicate stream) are record_stream'd onto
    # ``compute`` too. Without this, the caching allocator only tracks the
    # communicate stream and can recycle a slab while kernel_y / kernel_a on
    # compute are still reading or writing. Cheap; idempotent stores an int
    # handle, so calling per-layer is fine.
    buffer.runtime.set_compute_stream_handle(streams.compute.cuda_stream)
    # Register the caller (default) stream so the C++ Buffer allocates ``pool`` /
    # ``dL_do_pool`` in the default caching-allocator pool instead of a segregated
    # communicate-stream pool. ``stream_moe_func`` runs on the caller stream (the
    # model's default stream), which is also where StreamMoEFunc captures
    # ``caller_stream`` and where the fwd/bwd scratch is allocated — so all
    # StreamMoE memory shares one pool. Free-side safety is the layer-end
    # caller.wait_stream(compute/communicate) back-edges (NOT record_stream).
    buffer.runtime.set_default_stream_handle(torch.cuda.current_stream().cuda_stream)
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
        cfg,
        activation_checkpoint_level,
    )
