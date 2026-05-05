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
    scheduler shared between A (acquire on ``tile_ready``) and Y (acquire on
    ``a_ready``).
  - ``kernel_a.fire_tiles_with_delay`` /
    ``kernel_y.fire_a_ready_with_delay`` — test-only producers for
    the cross-stream ready signals.

The base GEMM machinery (``GemmGatedSm90``, ``TileScheduler``,
``PersistenceMode``, etc.) is imported from upstream-compatible quack modules.
"""
