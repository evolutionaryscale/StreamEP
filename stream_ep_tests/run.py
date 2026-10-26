"""Run a StreamEP test suite, one pytest session per test file.

    python -m stream_ep_tests.run kernels      # single process, 1 GPU
    python -m stream_ep_tests.run dist         # dist/ under torchrun on this node
    python -m stream_ep_tests.run internode    # dist/ + internode/ across the Slurm step's nodes
    python -m stream_ep_tests.run dist --list  # print the files that would run
    python -m stream_ep_tests.run dist -- -k uniform -x   # extra pytest arguments

Files are discovered from the package directories, so a test runs as soon as it
lands in the right one. Each file gets its own session (fresh processes, its
own process group and Buffer), so a crash or hang costs that file only; every
file runs even if an earlier one fails, and the exit code is non-zero if any
did. For internode, start one runner per node (``srun --ntasks-per-node=1 ...``);
each launches torchrun with its node rank and a c10d rendezvous on the step's
first node. Only local rank 0's output is shown; failures on other ranks reach
it through the cross-rank verdict in ``conftest.py``.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SUITES = {"kernels": ("kernels",), "dist": ("dist",), "internode": ("dist", "internode")}


def discover(suite: str) -> list[Path]:
    return [p for d in SUITES[suite] for p in sorted((PACKAGE / d).glob("test_*.py"))]


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


def command(suite: str, path: Path, index: int, args) -> list[str]:
    pytest = ["-m", "pytest", "-p", "no:cacheprovider", "-rfE", str(path.relative_to(PACKAGE.parent)), *args.pytest_args]
    if suite == "kernels":
        return [sys.executable, *pytest]
    step = f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.environ.get('SLURM_STEP_ID', '0')}-{index}"
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *torchrun_launch(args.nproc, args.port + index, step),
        "--local-ranks-filter",
        "0",
        *pytest,
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite", choices=SUITES)
    p.add_argument("--list", action="store_true", help="print the test files that would run and exit")
    p.add_argument("--nproc", type=int, default=8, help="processes per node for the torchrun suites")
    p.add_argument("--port", type=int, default=29500, help="rendezvous port of the first file; each later one uses the next")
    p.add_argument("--timeout", type=int, default=1800, help="per-file timeout in seconds")
    p.add_argument("pytest_args", nargs="*", help="extra pytest arguments (after --)")
    args = p.parse_args()

    if args.suite == "internode" and int(os.environ.get("SLURM_NNODES", "1")) < 2:
        raise SystemExit("the internode suite needs a Slurm step spanning two or more nodes")
    files = discover(args.suite)
    if args.list:
        print("\n".join(str(f.relative_to(PACKAGE.parent)) for f in files))
        return

    failed = []
    for i, path in enumerate(files):
        name = str(path.relative_to(PACKAGE.parent))
        print(f"=== {name}", flush=True)
        try:
            rc = subprocess.run(command(args.suite, path, i, args), cwd=PACKAGE.parent, timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        print(f"=== {name}: {'ok' if rc == 0 else f'FAILED ({rc})'}", flush=True)
        if rc != 0:
            failed.append(name)
    print(f"=== {args.suite}: {len(failed)} of {len(files)} files failed" + (f": {', '.join(failed)}" if failed else ""), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
