"""Streaming-MoE pipeline (pool-layout, SM90).

Public surface:
  - ``streaming_moe.stream_moe_func`` — one MoE forward layer (dispatch →
    kernel A → kernel Y → combine, on four caller-owned streams). Wraps
    ``StreamMoEFunc.apply`` with the keyword-arg public API.
  - ``streaming_moe.StreamMoEFunc`` — ``torch.autograd.Function`` running the
    layer forward (``backward`` returns all-``None``; the layer is a no-grad
    boundary).
  - ``streaming_moe.StreamHolder`` / ``streaming_moe.make_streams`` —
    dataclass holding the four caller-owned streams + helper to allocate them.
  - ``streaming_kernel_a.streaming_moe_a`` /
    ``streaming_kernel_y.streaming_moe_y`` — host wrappers for kernels A and Y.
  - ``streaming_tile_scheduler.StreamingTileScheduler`` /
    ``StreamingTileSchedulerArguments`` — linear-claim + per-tile-ready-spin
    scheduler shared between A (acquire on ``tile_ready``) and Y (acquire on
    ``a_ready``).
  - ``streaming_kernel_a.fire_tiles_with_delay`` /
    ``streaming_kernel_y.fire_a_ready_with_delay`` — test-only producers for
    the cross-stream ready signals.

The base GEMM machinery (``GemmGatedSm90``, ``TileScheduler``,
``PersistenceMode``, etc.) is imported from upstream-compatible quack modules.
"""
