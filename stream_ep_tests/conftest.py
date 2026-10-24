"""Shared fixtures. The distributed suites run as one pytest session per test
file on every rank (``torchrun ... -m pytest <file>``), so every rank collects
and runs the same tests in the same order.

Each test phase ends with a cross-rank verdict: if any rank failed, every rank
fails that phase, so ranks never diverge into different collectives. Tests
must still keep their collectives on every rank and put checks after them; a
rank that raises while its peers are inside a collective hangs them until the
process-group timeout (NCCL) or the kernel's device timeout (StreamEP comm
kernels) ends the run.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from stream_ep import Buffer
from stream_ep.stream_moe.stream_moe import make_streams
from stream_ep_tests.utils import DTYPE, H


def _share_verdict(outcome) -> None:
    if not dist.is_initialized():
        return
    error = outcome.excinfo
    if error is not None and isinstance(error[1], pytest.skip.Exception):
        error = None  # skips are decided identically on every rank
    local = None if error is None else f"rank {dist.get_rank()}: {error[0].__name__}: {error[1]}"
    verdicts = [None] * dist.get_world_size()
    dist.all_gather_object(verdicts, local)
    failed = [v for v in verdicts if v is not None]
    if failed:
        # Keep this rank's own traceback when it failed, so the report points at the failing line.
        outcome.force_exception(AssertionError("\n".join(failed)).with_traceback(None if error is None else error[2]))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    _share_verdict((yield))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _share_verdict((yield))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    _share_verdict((yield))


@pytest.fixture(scope="session")
def group():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank), timeout=timedelta(minutes=5))
    yield dist.group.WORLD
    dist.barrier()
    dist.destroy_process_group()


@pytest.fixture(scope="session")
def buffer(group):
    """One Buffer per session, sized for the production hidden size; tests with
    smaller hidden sizes fit in it."""
    return Buffer.sized_for(group, H * DTYPE.itemsize)


@pytest.fixture(scope="session")
def streams(group):
    """The layer's compute and communicate streams, shared by every test as in a model."""
    return make_streams()


@pytest.fixture(scope="session")
def internode_only(group):
    """Skip unless the group spans more than one node (8 GPUs each)."""
    if group.size() <= 8:
        pytest.skip("needs two or more nodes")
