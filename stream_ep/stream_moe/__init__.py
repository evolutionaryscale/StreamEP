"""Streaming-MoE pipeline (pool-layout, SM90).

Public surface:
  - ``streaming_moe.streaming_moe_layer`` — one MoE forward layer
    (dispatch → kernel A → kernel Y → combine, on four caller-owned streams).
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
