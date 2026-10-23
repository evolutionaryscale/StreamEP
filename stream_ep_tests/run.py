"""Run a StreamEP test suite.

    python -m stream_ep_tests.run kernels      # pytest, single process, 1 GPU
    python -m stream_ep_tests.run dist         # every dist/ script on this node
    python -m stream_ep_tests.run internode    # dist/ + internode/ scripts across the Slurm step's nodes
    python -m stream_ep_tests.run dist --list  # print what would run

Suites are discovered from the package directories, so a test runs as soon as
it lands in the right one: kernels/ (single process), dist/ (any world size),
internode/ (needs two or more nodes). For internode, start one runner per node
(`srun --ntasks-per-node=1 ...`); each launches torchrun with its node rank and
a c10d rendezvous on the step's first node. Every script runs even if an
earlier one fails; the exit code is non-zero if any failed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
TORCHRUN_SUITES = {"dist": ("dist",), "internode": ("dist", "internode")}


def discover(dirs: tuple[str, ...]) -> list[str]:
    return [f"{PACKAGE.name}.{d}.{p.stem}" for d in dirs for p in sorted((PACKAGE / d).glob("test_*.py"))]


def torchrun_launch(nproc: int, port: int, rdzv_id: str) -> list[str]:
    nnodes = int(os.environ.get("SLURM_NNODES", "1"))
    if nnodes == 1:
        return ["--standalone", f"--nproc-per-node={nproc}"]
    nodelist = os.environ.get("SLURM_STEP_NODELIST") or os.environ["SLURM_JOB_NODELIST"]
    master = subprocess.run(["scontrol", "show", "hostnames", nodelist], check=True, capture_output=True, text=True).stdout.split()[0]
    return [
        f"--nnodes={nnodes}",
        f"--node-rank={os.environ['SLURM_NODEID']}",
        f"--nproc-per-node={nproc}",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint={master}:{port}",
        f"--rdzv-id={rdzv_id}",
    ]


def run_torchrun_suite(suite: str, nproc: int, base_port: int, timeout: int) -> int:
    if suite == "internode" and int(os.environ.get("SLURM_NNODES", "1")) < 2:
        raise SystemExit("the internode suite needs a Slurm step spanning two or more nodes")
    step = f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.environ.get('SLURM_STEP_ID', '0')}"
    failed = []
    for i, module in enumerate(discover(TORCHRUN_SUITES[suite])):
        cmd = [sys.executable, "-m", "torch.distributed.run", *torchrun_launch(nproc, base_port + i, f"{step}-{i}"), "-m", module]
        print(f"=== {module}", flush=True)
        try:
            rc = subprocess.run(cmd, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        print(f"=== {module}: {'ok' if rc == 0 else f'FAILED ({rc})'}", flush=True)
        if rc != 0:
            failed.append(module)
    print(f"=== {suite}: {len(failed)} failed" + (f": {', '.join(failed)}" if failed else ""), flush=True)
    return 1 if failed else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite", choices=["kernels", *TORCHRUN_SUITES])
    p.add_argument("--list", action="store_true", help="print the tests that would run and exit")
    p.add_argument("--nproc", type=int, default=8, help="processes per node for torchrun suites")
    p.add_argument("--port", type=int, default=29500, help="rendezvous port of the first script; each later one uses the next")
    p.add_argument("--timeout", type=int, default=1800, help="per-script timeout in seconds")
    args = p.parse_args()

    if args.suite == "kernels":
        target = str(PACKAGE / "kernels")
        cmd = [sys.executable, "-m", "pytest", target] + (["--collect-only", "-q"] if args.list else [])
        raise SystemExit(subprocess.run(cmd).returncode)
    if args.list:
        print("\n".join(discover(TORCHRUN_SUITES[args.suite])))
        return
    raise SystemExit(run_torchrun_suite(args.suite, args.nproc, args.port, args.timeout))


if __name__ == "__main__":
    main()
