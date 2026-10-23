"""Check the test layout stream_ep_tests.run relies on (no GPU needed).

- every test_*.py lives in stream_ep_tests/{kernels,dist,internode}/;
- dist/ and internode/ scripts are torchrun entry points: a __main__ guard
  and cleanup_dist(), so the runner can trust their exit code;
- internode/ scripts guard world_size > 8, and dist/ scripts don't, because
  dist/ runs at every world size;
- kernels/ tests stay single-process (no init_process_group);
- imports are package-qualified (stream_ep_tests....): the runner launches
  modules from the repo root, so bare `from utils` / `from test_x` fail.

    python scripts/check_tests.py
"""

import re
import subprocess
import sys
from pathlib import Path

SUITES = ("kernels", "dist", "internode")
INTERNODE_GUARD = re.compile(r"world_size\s*>\s*8")
BARE_IMPORT = re.compile(r"^\s*(from|import)\s+(utils|test_\w+)\b", re.M)


def problems() -> list[str]:
    tracked = subprocess.run(["git", "ls-files", "*test_*.py"], capture_output=True, text=True, check=True).stdout.split()
    found = []
    for path in map(Path, tracked):
        if not (len(path.parts) == 3 and path.parts[0] == "stream_ep_tests" and path.parts[1] in SUITES):
            found.append(f"{path}: tests belong in stream_ep_tests/{{{','.join(SUITES)}}}/")
            continue
        suite, src = path.parts[1], path.read_text()
        if BARE_IMPORT.search(src):
            found.append(f"{path}: import test helpers as stream_ep_tests.<...>, not by bare module name")
        if suite == "kernels":
            if "init_process_group" in src:
                found.append(f"{path}: kernels/ tests are single-process")
            continue
        if 'if __name__ == "__main__":' not in src or "cleanup_dist()" not in src:
            found.append(f"{path}: torchrun scripts need a __main__ guard and cleanup_dist()")
        guarded = any(INTERNODE_GUARD.search(line) for line in src.splitlines() if line.lstrip().startswith("assert"))
        if suite == "internode" and not guarded:
            found.append(f"{path}: internode/ scripts must assert world_size > 8")
        if suite == "dist" and guarded:
            found.append(f"{path}: dist/ scripts run at every world size; move internode-only tests to internode/")
    return found


if __name__ == "__main__":
    found = problems()
    print("\n".join(found) or "test layout ok")
    sys.exit(1 if found else 0)
