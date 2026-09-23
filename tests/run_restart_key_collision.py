#!/usr/bin/env python3
"""Editing one output block's interval to equal another block's must not renumber that
other block: before the fix the edited block claimed the twin's saved slot, and the twin,
denied its own position, restarted at file number zero and overwrote its own frames. The
resume runs IN PLACE, in the base run's directory, which is where that destroys data."""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_restart_new_output import (FRAME_DT, RESTART_DT, RESUME_TLIM, BASE_TLIM,
                                    restart_schedule, run, write_case)

import netCDF4
import numpy as np

SLOW_DT = 11.0  # a second netcdf cadence, distinct from FRAME_DT until the resume edits it


def frames(case_dir: Path) -> dict[tuple[str, int], float]:
  """{(stream, file number): time} for every combined netcdf frame in the directory."""
  out = {}
  for f in case_dir.glob("*.nc"):
    m = re.search(r"\.(out\d+)\.(\d+)\.nc$", f.name)
    if m is None:
      continue
    with netCDF4.Dataset(f, "r") as d:
      out[(m.group(1), int(m.group(2)))] = float(np.asarray(d.variables["time"][:]).ravel()[0])
  return out


def written_after(table, stream: str, resume_t: float) -> list[int]:
  return sorted(n for (s, n), t in table.items() if s == stream and t > resume_t + 1e-6)


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
  slow = {"type": "netcdf", "variables": ["prim"], "dt": SLOW_DT}
  edited = dict(fast, dt=SLOW_DT)  # its schedule key becomes the slow block's
  if fast["dt"] == slow["dt"]:
    raise AssertionError("the two netcdf blocks already collide in leg 1: test defused")
  launch = [torchrun, "--no-python", "--nproc-per-node=1", str(exe)]

  # leg 1: three output blocks, the two netcdf ones on DIFFERENT cadences
  case_dir = tests_dir / "restart_key_collision"
  case_yaml = write_case(base_yaml, case_dir, BASE_TLIM, [restart, fast, slow])
  run(launch + [str(case_yaml)], case_dir, env)

  restart_file = sorted(case_dir.glob("*.restart"))[-1]
  resume_t, _, saved = restart_schedule(restart_file)
  if len(saved) != 3:
    raise AssertionError(f"restart stores {len(saved)} output slots, expected 3")
  # a base leg too short makes "the numbering continued" vacuously true
  if saved[1] < 2 or saved[2] < 2:
    raise AssertionError(f"base leg too short to judge continuation: file_number {saved}")
  before = frames(case_dir)

  # leg 2: SAME directory, same basename, with the fast block's interval edited to the slow one's
  staging = tests_dir / "restart_key_collision_cfg"
  resumed = write_case(base_yaml, staging, RESUME_TLIM, [restart, edited, slow])
  shutil.copy2(resumed, case_yaml)  # basename comes from the card's stem, so reuse the name
  run(launch + [str(case_yaml), "--restart", str(restart_file.resolve())], case_dir, env)
  after = frames(case_dir)

  # 1. no frame the base leg completed may be rewritten. Number saved[fid] is excluded: it
  #    is the base leg's final write, whose number the next scheduled write legitimately reuses.
  for (stream, number), t0 in sorted(before.items()):
    fid = int(stream[3:])
    if number >= saved[fid]:
      continue
    if (stream, number) not in after:
      raise AssertionError(f"{stream} frame {number} (t={t0}) disappeared across the resume")
    t1 = after[(stream, number)]
    if abs(t1 - t0) > 1e-9:
      raise AssertionError(
          f"the resume overwrote {stream} frame {number}: time was {t0}, now {t1}; that stream "
          f"restarted its numbering instead of continuing from {saved[fid]}")

  # 2. and each stream must really have written again, at or above its saved number
  for stream in ("out1", "out2"):
    post = written_after(after, stream, resume_t)
    if len(post) < 2 or post[0] < saved[int(stream[3:])]:
      raise AssertionError(f"{stream} resumed at {post}, saved file_number {saved}")

  print(f"ok: resume at {resume_t:.3f}; saved file_number {saved}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
