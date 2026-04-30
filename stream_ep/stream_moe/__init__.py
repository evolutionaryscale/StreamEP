"""Streaming-MoE pipeline (pool-layout, SM90).

Public surface:
  - ``streaming_kernel_a.streaming_moe_a`` — host wrapper for kernel A.
  - ``streaming_kernel_a.fire_tiles_with_delay`` — test-only producer for the
    cross-stream tile_ready signal.
  - ``streaming_tile_scheduler.StreamingTileScheduler`` /
    ``StreamingTileSchedulerArguments`` / ``StreamingWorkTileInfo`` — the
    linear-claim + per-tile-ready-spin scheduler that DeepEP's
    ``Buffer.dispatch`` Pass 2 feeds via ``tile_ready[tile_id]``.

The streaming additions used to live in the ``quack`` fork; they're hosted here
so the rest of the streaming-MoE pipeline (DeepEP wrapper, profile/bench/smoke
harnesses, future kernel Y) lives in one repo. The base GEMM machinery
(``GemmGatedSm90``, ``TileScheduler``, ``PersistenceMode``, etc.) is imported
from upstream-compatible quack modules.
"""
