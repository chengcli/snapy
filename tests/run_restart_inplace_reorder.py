#!/usr/bin/env python3
"""An in-place resume must never rewrite a frame the base leg already completed. The output
file name carries the block's POSITION (out<n>), so restoring the file counter by schedule
KEY hands a reordered block another position's counter and writes over finished frames.
Reuses the harness of run_restart_new_output.py."""
import argparse
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_restart_new_output import (BASE_TLIM, FRAME_DT, RESTART_DT, RESUME_TLIM,
                                    restart_schedule, run, write_case)

SKIP_CODE = 125
SLOW_DT = 20.0  # out2's cadence: fewer frames than out1, so a swapped counter goes BACKWARDS


def frames(case_dir: Path) -> dict:
  """{file name: (md5, frame time)} for every combined netcdf frame in the directory."""
  out = {}
  for f in sorted(case_dir.glob("*.nc")):
    if ".block" in f.name or re.search(r"\.out\d+\.\d+\.nc$", f.name) is None:
      continue
    with netCDF4.Dataset(f, "r") as d:
      t = float(np.asarray(d.variables["time"][:]).ravel()[0])
    out[f.name] = (hashlib.md5(f.read_bytes(), usedforsecurity=False).hexdigest(), t)
  return out


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--build-dir", required=True)
  parser.add_argument("--build-type", required=True)
  args = parser.parse_args()

  build_dir = Path(args.build_dir).resolve()
  tests_dir = build_dir / "tests"
  repo_root = Path(__file__).resolve().parent.parent
  exe = build_dir / "bin" / f"straka.{args.build_type}"
  if not exe.exists():
    raise FileNotFoundError(f"missing executable {exe}")
  base_yaml = repo_root / "examples" / "straka.yaml"

  torchrun = shutil.which("torchrun")
  if torchrun is None:
    raise FileNotFoundError("torchrun not found in PATH")
  env = os.environ.copy()
  env["BACKEND"] = "gloo"
  env["PYTHONPATH"] = ":".join([str(repo_root / "python"), str(repo_root)]
                              + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

  restart = {"type": "restart", "dt": RESTART_DT}
  fast = {"type": "netcdf", "variables": ["prim"], "dt": FRAME_DT}
  slow = {"type": "netcdf", "variables": ["uov"], "dt": SLOW_DT}
  launch = [torchrun, "--no-python", "--nproc-per-node=1", str(exe)]

  case_dir = tests_dir / "restart_inplace_reorder"
  case_yaml = write_case(base_yaml, case_dir, BASE_TLIM, [restart, fast, slow])
  run(launch + [str(case_yaml)], case_dir, env)
  restart_file = sorted(case_dir.glob("*.restart"))[-1]
  resume_t, _, saved_numbers = restart_schedule(restart_file)
  if saved_numbers[1] == saved_numbers[2]:
    raise AssertionError(
        f"fixture defused: the two streams saved the same file number {saved_numbers}, so a "
        f"swapped counter would be invisible")
  before = frames(case_dir)
  finished = {n: v for n, v in before.items() if v[1] <= resume_t + 1e-9}
  if len(finished) < 4:
    raise AssertionError(f"too few completed frames to judge: {before}")

  # resume IN PLACE -- same directory, same basename -- with the two netcdf blocks SWAPPED
  write_case(base_yaml, case_dir, RESUME_TLIM, [restart, slow, fast], wipe=False)
  run(launch + [str(case_yaml), "--restart", str(restart_file.resolve())], case_dir, env)

  after = frames(case_dir)
  damaged = sorted(n for n, v in finished.items() if n in after and after[n][0] != v[0])
  if damaged:
    detail = ", ".join(f"{n}: t {finished[n][1]} -> {after[n][1]}" for n in damaged)
    raise AssertionError(
        f"the in-place resume rewrote {len(damaged)} completed frame(s) ({detail}); the "
        f"restart restored a file counter from another position's slot "
        f"(saved file_number {saved_numbers}, resume at {resume_t})")
  print(f"ok: resume at {resume_t}; saved file_number {saved_numbers}; "
        f"{len(finished)} completed frames intact, {len(after) - len(before)} new")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
