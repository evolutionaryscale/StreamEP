"""StreamEP's tests, grouped by what they need to run.

- ``kernels/`` — pytest, single process, 1 GPU: the streaming-MoE kernels
  against eager references.
- ``dist/`` — torchrun scripts that pass at any world size: one node runs the
  intranode paths, several nodes the internode ones.
- ``internode/`` — torchrun scripts that need two or more nodes.

``python -m stream_ep_tests.run {kernels,dist,internode}`` discovers and runs a
suite (``internode`` runs ``dist/`` too, once per node under ``srun``); the
pixi tasks ``test-kernels`` / ``test-dist`` / ``test-internode`` wrap it. One
script: ``torchrun --standalone --nproc-per-node=8 -m stream_ep_tests.dist.test_dispatch``.
``scripts/check_tests.py`` (run by pre-commit) keeps files in the directory
that matches what they need.
"""
