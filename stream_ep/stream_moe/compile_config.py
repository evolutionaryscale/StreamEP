"""StreamEP-owned compile-only gate.

quack `main` removed the global `quack.cache_utils.COMPILE_ONLY` flag (its cache
moved to an async CPU-worker model in `quack.cache`). StreamEP only ever
piggybacked on that flag for its compile-check tests, so it owns the gate here.

When ``COMPILE_ONLY`` is True, the streaming-MoE kernel entry points still run
the CuTeDSL compile (exercising the compile path — the point of the check) but
`return` before any host-side launch setup. Read it as a *module attribute*
(``compile_config.COMPILE_ONLY``), never ``from ... import COMPILE_ONLY``, so a
test toggling it is seen live by the kernels.
"""

COMPILE_ONLY: bool = False
