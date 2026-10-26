"""Check the test layout stream_ep_tests.run relies on (no GPU needed).

- every test_*.py lives in stream_ep_tests/{kernels,dist,internode}/;
- dist/ and internode/ tests run under torchrun on every rank and take the
  process group from the ``group`` fixture: they never initialize it
  themselves or define a ``__main__`` entry point;
- internode/ modules skip below two nodes (``pytestmark =
  pytest.mark.usefixtures("internode_only")``), and dist/ modules don't,
  because dist/ runs at every world size;
- kernels/ tests stay single-process (no process group);
- imports are package-qualified (stream_ep_tests....): tests run from the
  repo root, so bare `from utils` / `from test_x` fail.

    python scripts/check_tests.py
"""

import re
import subprocess
import sys
from pathlib import Path

SUITES = ("kernels", "dist", "internode")
INTERNODE_MARK = 'pytestmark = pytest.mark.usefixtures("internode_only")'
BARE_IMPORT = re.compile(r"^\s*(from|import)\s+(utils|test_\w+|_common)\b", re.M)


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
        if "init_process_group" in src:
            found.append(
                f"{path}: take the process group from the `group` fixture"
                if suite != "kernels"
                else f"{path}: kernels/ tests are single-process"
            )
        if 'if __name__ == "__main__":' in src:
            found.append(f"{path}: tests run through pytest; drop the __main__ entry point")
        if suite == "internode" and INTERNODE_MARK not in src:
            found.append(f"{path}: internode/ modules must declare `{INTERNODE_MARK}`")
        if suite == "dist" and INTERNODE_MARK in src:
            found.append(f"{path}: dist/ tests run at every world size; move internode-only tests to internode/")
    return found


if __name__ == "__main__":
    found = problems()
    print("\n".join(found) or "test layout ok")
    sys.exit(1 if found else 0)
