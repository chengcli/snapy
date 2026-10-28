#!/usr/bin/env python3
"""The yaml-cpp patch gives src/emitterutils.cpp its <cstdint>.

yaml-cpp 0.8.0 declares EncodeUTF16SurrogatePair as std::pair<uint16_t, uint16_t> and
uses uint32_t beneath it, but src/emitterutils.cpp includes only <algorithm>, <iomanip>
and <sstream> before its own headers. Whether uint16_t is nonetheless declared depends
on what those headers drag in, and that changed: with libstdc++ 15 the translation unit
stops compiling, while libstdc++ 13 still builds it.

  src/emitterutils.cpp:221:11: error: 'uint16_t' was not declared in this scope
  note: 'uint16_t' is defined in header '<cstdint>'

GCC 15 is the default on ubuntu-latest, so the fetched yaml-cpp has to be patched.
cmake/yamlpp.cmake applies patches/02.yaml-cpp.patch, which must therefore add the
include.

Two checks:

  patch    the unified diff is parsed, and the hunks for src/emitterutils.cpp must add
           an `#include <cstdint>` line. This arm is compiler independent and is the one
           that fails before the fix.
  compile  the real src/emitterutils.cpp, when the yaml-cpp source can be found, built
           exactly as the build would leave it: the file with every `#include <cstdint>`
           stripped, plus the include back only if the patch above actually adds it.
           That unit must compile. It is the real file with its real include graph, so
           a toolchain whose headers already supply uint16_t is never failed for it.
           The unstripped-minus-include variant is compiled too and only reported, to
           say whether this toolchain reproduces the defect at all.

The compile arm needs the yaml-cpp source. It is taken from --yaml-cpp-src, else
$YAML_CPP_SRC, else <build dir>/_deps/yaml-cpp-src; without any of those it is skipped.

  python test_yamlcpp_cstdint_patch.py [--cxx g++] [--build-dir DIR] [--yaml-cpp-src DIR]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCH = ROOT / "patches" / "02.yaml-cpp.patch"
TARGET = "src/emitterutils.cpp"
INCLUDE = "#include <cstdint>"


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


def find_source(args):
    for candidate in (args.yaml_cpp_src, os.environ.get("YAML_CPP_SRC")):
        if candidate:
            return Path(candidate)
    if args.build_dir:
        guess = Path(args.build_dir) / "_deps" / "yaml-cpp-src"
        if guess.is_dir():
            return guess
    return None


def strip_include(text):
    return "".join(l for l in text.splitlines(keepends=True)
                   if l.strip() != INCLUDE)


def compile_variant(cxx, source_dir, text, workdir, name):
    """Compile `text` as if it were source_dir/src/emitterutils.cpp."""
    unit = workdir / f"{name}.cpp"
    unit.write_text(text)
    r = subprocess.run(
        [cxx, "-std=c++11", f"-I{source_dir / 'include'}", f"-I{source_dir / 'src'}",
         "-c", str(unit), "-o", str(workdir / f"{name}.o")],
        capture_output=True, text=True)
    return r.returncode == 0, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    ap.add_argument("--build-dir")
    ap.add_argument("--yaml-cpp-src")
    args = ap.parse_args()

    failures = []

    if not PATCH.exists():
        print(f"FAIL {PATCH.name} is missing")
        return 1
    added = [a.strip() for a in added_lines(PATCH.read_text(), TARGET)]
    patch_adds_include = INCLUDE in added
    if not patch_adds_include:
        failures.append(
            f"{PATCH.name} does not add '{INCLUDE}' to {TARGET}; "
            f"it adds {added or 'nothing'} there. yaml-cpp 0.8.0 will not compile "
            f"with libstdc++ 15.")
        print(f"FAIL {failures[-1]}")
    else:
        print(f"pass patch: {PATCH.name} adds '{INCLUDE}' to {TARGET}")

    cxx = shutil.which(args.cxx)
    source = find_source(args)
    if cxx is None:
        print(f"skip compile: no {args.cxx} on PATH")
    elif source is None or not (source / TARGET).is_file():
        print("skip compile: yaml-cpp source not found; pass --yaml-cpp-src, set "
              "$YAML_CPP_SRC, or give --build-dir of a build that fetched it")
    else:
        version = subprocess.run([cxx, "--version"], capture_output=True, text=True
                                 ).stdout.splitlines()[0]
        pristine = strip_include((source / TARGET).read_text())
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            # what the build will actually see, given the patch as it stands
            as_built = (INCLUDE + "\n" + pristine) if patch_adds_include else pristine
            ok, err = compile_variant(cxx, source, as_built, work, "as_built")
            if ok:
                print(f"pass compile: {version} builds the real {TARGET} as the patch "
                      f"leaves it")
            else:
                first = next((l for l in err.splitlines() if "uint16_t" in l),
                             err.splitlines()[0] if err else "")
                failures.append(f"{version} cannot build the real {TARGET} as the patch "
                                f"leaves it -- {first.strip()}")
                print(f"FAIL {failures[-1]}")
            # informational only: does this toolchain reproduce the defect?
            bare, _ = compile_variant(cxx, source, pristine, work, "bare")
            if bare:
                print(f"note compile: {version} builds {TARGET} even without "
                      f"'{INCLUDE}', so the defect does not show on this toolchain")
            else:
                print(f"note compile: {version} rejects {TARGET} without "
                      f"'{INCLUDE}', so the defect reproduces on this toolchain")

    if failures:
        print(f"yaml-cpp <cstdint> patch: {len(failures)} problem(s)")
        return 1
    print("yaml-cpp <cstdint> patch: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
