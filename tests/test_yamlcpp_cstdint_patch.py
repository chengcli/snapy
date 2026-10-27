#!/usr/bin/env python3
"""The yaml-cpp patch gives src/emitterutils.cpp its <cstdint>.

yaml-cpp 0.8.0 declares EncodeUTF16SurrogatePair as std::pair<uint16_t, uint16_t> and
uses uint32_t beneath it, but src/emitterutils.cpp includes only <algorithm>, <iomanip>
and <sstream>. Up to libstdc++ 14 those pulled <cstdint> in transitively; libstdc++ 15
no longer does, so the translation unit stops compiling:

  src/emitterutils.cpp:221:11: error: 'uint16_t' was not declared in this scope
  note: 'uint16_t' is defined in header '<cstdint>'

GCC 15 is the default on ubuntu-latest, so the fetched yaml-cpp has to be patched.
cmake/yamlpp.cmake applies patches/02.yaml-cpp.patch, which must therefore add the
include.

Two checks:

  patch    the unified diff is parsed, and the hunks for src/emitterutils.cpp must add
           an `#include <cstdint>` line. This is the check that fails before the fix.
  compile  only where the toolchain reproduces the condition. The probe compiles the
           header set of emitterutils.cpp with its uint16_t declaration and no
           <cstdint>. If that builds, the standard library still supplies the include
           transitively, the failure cannot occur here, and the arm says so and stops.
           If it does not build, the same unit must build once <cstdint> is added.

  python test_yamlcpp_cstdint_patch.py [--cxx g++]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCH = ROOT / "patches" / "02.yaml-cpp.patch"
TARGET = "src/emitterutils.cpp"
INCLUDE = "#include <cstdint>"

# The header set and declaration of yaml-cpp 0.8.0 src/emitterutils.cpp, which is what
# libstdc++ 15 stopped carrying <cstdint> into.
PROBE = """#include <algorithm>
%s
#include <iomanip>
#include <sstream>
std::pair<uint16_t, uint16_t> EncodeUTF16SurrogatePair(int codePoint) {
  const uint32_t leadOffset = 0xD800 - (0x10000 >> 10);
  return std::pair<uint16_t, uint16_t>(static_cast<uint16_t>(leadOffset),
                                       static_cast<uint16_t>(codePoint));
}
int main() { return 0; }
"""


def added_lines(patch_text, target):
    """Lines the unified diff adds to `target`, parsed per file rather than grepped."""
    added, in_target = [], False
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            in_target = line.endswith(f"b/{target}")
        elif line.startswith("+++ "):
            in_target = line.split("\t")[0].endswith(f"/{target}")
        elif in_target and line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return added


def compiles(cxx, source):
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "probe.cpp"
        src.write_text(source)
        r = subprocess.run([cxx, "-std=c++11", "-c", str(src), "-o", str(Path(d) / "probe.o")],
                           capture_output=True, text=True)
        return r.returncode == 0, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    args = ap.parse_args()

    failures = []

    if not PATCH.exists():
        print(f"FAIL {PATCH.name} is missing")
        return 1
    added = added_lines(PATCH.read_text(), TARGET)
    if INCLUDE not in [a.strip() for a in added]:
        failures.append(
            f"{PATCH.name} does not add '{INCLUDE}' to {TARGET}; "
            f"it adds {added or 'nothing'} there. yaml-cpp 0.8.0 will not compile "
            f"with libstdc++ 15."
        )
        print(f"FAIL {failures[-1]}")
    else:
        print(f"pass patch: {PATCH.name} adds '{INCLUDE}' to {TARGET}")

    cxx = shutil.which(args.cxx)
    if cxx is None:
        print(f"skip compile: no {args.cxx} on PATH")
    else:
        version = subprocess.run([cxx, "--version"], capture_output=True, text=True
                                 ).stdout.splitlines()[0]
        without, err = compiles(cxx, PROBE % "")
        if without:
            print(f"skip compile: {version} still supplies <cstdint> transitively, "
                  f"so the failure cannot be reproduced with this toolchain")
        else:
            first = next((l for l in err.splitlines() if "uint16_t" in l), err.splitlines()[0])
            print(f"pass compile: {version} rejects the unpatched unit -- {first.strip()}")
            with_include, err2 = compiles(cxx, PROBE % INCLUDE)
            if not with_include:
                failures.append(f"the unit still fails with '{INCLUDE}' added: {err2.strip()}")
                print(f"FAIL {failures[-1]}")
            else:
                print(f"pass compile: the same unit builds once '{INCLUDE}' is added")

    if failures:
        print(f"yaml-cpp <cstdint> patch: {len(failures)} problem(s)")
        return 1
    print("yaml-cpp <cstdint> patch: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
