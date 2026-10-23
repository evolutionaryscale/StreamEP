"""Gate a built wheel before it is published as a release asset.

Checks that dist/ holds exactly one wheel for pyproject.toml's version and that
its compiled extension is loadable from evoscale's env rather than the build
machine's: $ORIGIN-relative rpath only, SM90 SASS only, and no glibc symbol
newer than evoscale's libc floor.

    python scripts/check_wheel.py [dist_dir]
"""

import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAX_GLIBC = (2, 28)
MAX_GLIBCXX = (3, 4, 33)  # libstdc++ of gcc 14, evoscale's compiler
CUDA_ARCHS = {"sm_90"}
EXTENSION = "stream_ep_cpp.cpython-312-x86_64-linux-gnu.so"


def tool(*names: str) -> str:
    for name in names:
        if path := shutil.which(name):
            return path
    raise SystemExit(f"none of {names} on PATH")


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def check_metadata(wheel: zipfile.ZipFile, version: str, pinned: list[str]) -> None:
    metadata = wheel.read(f"stream_ep-{version}.dist-info/METADATA").decode()
    assert f"\nVersion: {version}\n" in metadata, "METADATA version does not match pyproject.toml"
    for requirement in pinned:
        assert f"\nRequires-Dist: {requirement}\n" in metadata, f"METADATA is missing Requires-Dist: {requirement}"


def check_extension(so: Path) -> None:
    readelf = tool("readelf", "x86_64-conda-linux-gnu-readelf")
    dynamic = run(readelf, "-d", str(so))
    rpaths = [p for m in re.finditer(r"\((?:RPATH|RUNPATH)\).*\[(.*)\]", dynamic) for p in m.group(1).split(":")]
    absolute = [p for p in rpaths if not p.startswith("$ORIGIN")]
    assert rpaths and not absolute, f"rpath must be $ORIGIN-relative only, got {rpaths}"

    versions = run(readelf, "-V", str(so))
    for name, ceiling in (("GLIBC", MAX_GLIBC), ("GLIBCXX", MAX_GLIBCXX)):
        needed = max(tuple(map(int, v.split("."))) for v in re.findall(rf"\b{name}_([0-9.]+)", versions))
        assert needed <= ceiling, f"needs {name}_{'.'.join(map(str, needed))}, above {ceiling}"

    archs = set(re.findall(r"\.(sm_\w+)\.", run(tool("cuobjdump"), "--list-elf", str(so))))
    assert archs == CUDA_ARCHS, f"SASS archs {sorted(archs)}, expected {sorted(CUDA_ARCHS)}"


def main() -> None:
    dist = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "dist"
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    version = project["version"]

    wheels = sorted(dist.glob("*.whl"))
    expected = f"stream_ep-{version}-cp312-cp312-linux_x86_64.whl"
    assert [w.name for w in wheels] == [expected], f"expected only {expected} in {dist}, found {[w.name for w in wheels]}"

    with zipfile.ZipFile(wheels[0]) as wheel, tempfile.TemporaryDirectory() as tmp:
        check_metadata(wheel, version, project["dependencies"])
        check_extension(Path(wheel.extract(EXTENSION, tmp)))
    print(f"ok: {wheels[0].name}")


if __name__ == "__main__":
    main()
