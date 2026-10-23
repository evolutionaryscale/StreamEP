"""Build the release wheel into dist/.

conda-forge's gcc specs append ``-rpath <env>/lib`` to every non-static link,
which would point the wheel's extension at the build machine's env. After
``pip wheel`` this unpacks the wheel, keeps only the extension's
$ORIGIN-relative rpath entries (set in setup.py), and repacks it; ``wheel
pack`` regenerates RECORD so the hashes stay valid.

    python scripts/build_wheel.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIST = REPO / "dist"


def run(*cmd: str | Path) -> str:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout


def main() -> None:
    for stale in (DIST, REPO / "build"):
        shutil.rmtree(stale, ignore_errors=True)
    DIST.mkdir()
    with tempfile.TemporaryDirectory() as tmp:
        raw, unpacked = Path(tmp, "raw"), Path(tmp, "unpacked")
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "--wheel-dir", raw, REPO], check=True)
        (wheel,) = raw.glob("*.whl")
        run(sys.executable, "-m", "wheel", "unpack", "--dest", unpacked, wheel)
        (root,) = unpacked.iterdir()
        for so in root.glob("*.so"):
            rpath = run("patchelf", "--print-rpath", so).strip().split(":")
            run("patchelf", "--set-rpath", ":".join(p for p in rpath if p.startswith("$ORIGIN")), so)
        run(sys.executable, "-m", "wheel", "pack", "--dest-dir", DIST, root)
    print(f"built {', '.join(w.name for w in DIST.glob('*.whl'))}")


if __name__ == "__main__":
    main()
