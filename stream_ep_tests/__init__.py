"""StreamEP's tests, grouped by what they need to run. All are pytest.

- ``kernels/`` — single process, 1 GPU: the streaming-MoE kernels against
  torch references.
- ``dist/`` — every rank runs the same session under torchrun; passes at any
  world size (one node runs the intranode paths, several the internode ones).
- ``internode/`` — needs two or more nodes.

``python -m stream_ep_tests.run {kernels,dist,internode}`` runs a suite, one
pytest session per file (``internode`` runs ``dist/`` too, once per node under
``srun``); the pixi tasks ``test-kernels`` / ``test-dist`` / ``test-internode``
wrap it. See AGENTS.md for the rules distributed tests follow.
"""
