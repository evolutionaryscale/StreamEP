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
    consumes dispatch's ``pool_arrival_count`` / ``pool_arrival_target``;
    kernel Y consumes A's ``a_ready_count`` against a uniform
    ``a_ready_target`` filled with kernel A's ``num_pid_n``.
  - ``kernel_a.fire_tiles_with_delay`` — test-only producer for the
    cross-stream count-vs-target signals (the kernel Y test stub was
    retired with the protocol unification).

The base GEMM machinery (``GemmGatedSm90``, ``TileScheduler``,
``PersistenceMode``, etc.) is imported from upstream-compatible quack modules.
"""
