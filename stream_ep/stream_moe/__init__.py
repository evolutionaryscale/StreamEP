"""Streaming-MoE pipeline (pool-layout, SM90).

Public surface:
  - ``stream_moe.stream_moe_func`` — one MoE forward layer (dispatch →
    kernel A → kernel Y → combine, on four caller-owned streams). Wraps
    ``StreamMoEFunc.apply`` with the keyword-arg public API.
  - ``stream_moe.StreamMoEFunc`` — ``torch.autograd.Function`` running the
    layer forward and backward (dispatch_grads → kernel_y_bwd → kernel_a_bwd
    → combine_grads on the same four streams as forward, plus dW1 / dW2
    grouped GEMMs on a dedicated ``grads`` stream).
  - ``stream_moe.StreamHolder`` / ``stream_moe.make_streams`` —
    dataclass holding the four caller-owned streams + helper to allocate them.
  - ``kernel_a.streaming_moe_a`` /
    ``kernel_y.streaming_moe_y`` — host wrappers for kernels A and Y.
  - ``tile_scheduler.StreamingTileScheduler`` /
    ``StreamingTileSchedulerArguments`` — linear-claim + per-tile-ready-spin
    scheduler. One protocol for every per-tile handoff: count-vs-target on
    ``arrival_count[tile] == arrival_target[tile]``, with each producer
    firing ``red.release.gpu.global.add.s32`` per contribution. Kernel A
    and kernel Y both consume dispatch's ``pool_arrival_count`` /
    ``pool_arrival_target`` (Y on the same compute stream after A, so its
    spin no-ops). Kernel_y_bwd and kernel_a_bwd consume
    ``bwd_dispatch_arrival_count`` / ``pool_arrival_target``.
  - ``kernel_a.fire_tiles_with_delay`` — test-only producer for the
    cross-stream count-vs-target signals.

The base GEMM machinery (``GemmGatedSm90``, ``TileScheduler``,
``PersistenceMode``, etc.) is imported from upstream-compatible quack modules.
"""
