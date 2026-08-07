"""StreamEP-owned compile-only gate.

When ``COMPILE_ONLY`` is True, the streaming-MoE kernel entry points still run
the CuTeDSL compile (exercising the compile path — the point of the check) but
`return` before any host-side launch setup. Read it as a *module attribute*
(``compile_config.COMPILE_ONLY``), never ``from ... import COMPILE_ONLY``, so a
test toggling it is seen live by the kernels.
"""

COMPILE_ONLY: bool = False
