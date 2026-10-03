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
from run_restart_output_schedule import (
    BASE_TLIM, FRAME_DT, RESTART_DT, RESUME_TLIM,
    restart_schedule, restart_tensors, run, write_case,
)

import netCDF4
import numpy as np
import torch
import yaml

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


def checkpoint_variant(source: Path, target: Path, key_mode: str) -> Path:
  tensors = restart_tensors(source)
  nsaved = tensors["file_number"].numel()
  if key_mode == "legacy":
    tensors["output_key"] = tensors["output_key"][:nsaved]
  elif key_mode == "keyless":
    tensors.pop("output_key")
  else:
    raise ValueError(f"unknown key mode {key_mode}")

  class TensorModule(torch.nn.Module):
    def __init__(self):
      super().__init__()
      for name, tensor in tensors.items():
        self.register_buffer(name, tensor)

  torch.jit.script(TensorModule()).save(str(target))
  return target


def run_precision_reorder(
    base_yaml: Path, tests_dir: Path, launch, env, old_exe: Path | None = None,
) -> None:
  restart = {"type": "restart", "dt": 6.e-7}
  short = {"type": "netcdf", "variables": ["prim"], "dt": 4.e-7}
  long = {"type": "netcdf", "variables": ["prim"], "dt": 4.9e-7}

  def scaled_case(case_dir: Path, tlim: float, outputs) -> Path:
    card = write_case(base_yaml, case_dir, tlim, outputs)
    config = yaml.safe_load(card.read_text())
    for key in ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max"):
      config["geometry"]["bounds"][key] = float(config["geometry"]["bounds"][key]) * 1.e-7
    card.write_text(yaml.safe_dump(config, sort_keys=False))
    return card

  base_dir = tests_dir / "restart_key_precision_base"
  card = scaled_case(base_dir, 9.e-7, [restart, short, long])
  run(launch + [str(card)], base_dir, env)
  restart_file = sorted(p for p in base_dir.glob("*.restart") if ".final." not in p.name)[-1]
  resume_t, saved_next, _ = restart_schedule(restart_file)
  if not (saved_next[1] != saved_next[2] and saved_next[1] > resume_t and
          saved_next[2] > resume_t):
    raise AssertionError(
        f"precision fixture has indistinguishable schedules at {resume_t}: {saved_next}")

  tensors = restart_tensors(restart_file)
  nsaved = tensors["file_number"].numel()
  if "output_key_v2" in tensors:
    raise AssertionError("precise keys must not use a separate legacy-visible buffer")
  if tensors["output_key"].numel() != 2 * nsaved:
    raise AssertionError(
        f"new output_key has {tensors['output_key'].numel()} entries, expected "
        f"{nsaved} legacy keys followed by {nsaved} precise keys")
  legacy_file = checkpoint_variant(
      restart_file, base_dir / "legacy.restart", "legacy")
  keyless_file = checkpoint_variant(
      restart_file, base_dir / "keyless.restart", "keyless")

  def resume(name: str, checkpoint: Path):
    resumed_dir = tests_dir / name
    resumed = scaled_case(resumed_dir, 1.35e-6, [restart, long, short])
    run(launch + [str(resumed), "--restart", str(checkpoint.resolve())],
        resumed_dir, env)
    return restart_schedule(sorted(resumed_dir.glob("*.restart"))[-1])[:2]

  precise_t, precise_next = resume("restart_key_precision_resumed", restart_file)
  legacy_t, legacy_next = resume("restart_key_legacy_resumed", legacy_file)
  keyless_t, keyless_next = resume("restart_key_keyless_resumed", keyless_file)

  def advance(saved: float, dt: float, final_time: float) -> float:
    while saved <= final_time:
      saved += dt
    return saved

  identity = [precise_next[0], advance(saved_next[2], long["dt"], precise_t),
              advance(saved_next[1], short["dt"], precise_t)]
  positional = [legacy_next[0], advance(saved_next[1], long["dt"], legacy_t),
                advance(saved_next[2], short["dt"], legacy_t)]
  for label, actual, expected in (
      ("precise", precise_next, identity),
      ("legacy", legacy_next, positional),
      ("keyless", keyless_next, positional),
  ):
    for slot in (1, 2):
      if abs(actual[slot] - expected[slot]) > 1.e-15:
        raise AssertionError(
            f"{label} cadence reorder restored slot {slot} to {actual[slot]}, "
            f"expected {expected[slot]} from saved schedules {saved_next}; "
            f"resume {resume_t}")

  if old_exe is not None:
    run_old_binary_roundtrip(
        restart_file, tests_dir, scaled_case, launch, env, old_exe,
        restart, long, short, advance)


def run_old_binary_roundtrip(
    restart_file: Path, tests_dir: Path, scaled_case, launch, env,
    old_exe: Path, restart, long, short, advance,
) -> None:
  old_dir = tests_dir / "restart_key_old_reader_resumed"
  old_card = scaled_case(old_dir, 1.15e-6, [restart, long, short])
  old_launch = launch[:-1] + [str(old_exe)]
  run(old_launch + [str(old_card), "--restart", str(restart_file.resolve())],
      old_dir, env)
  old_file = sorted(old_dir.glob("*.restart"))[-1]
  old_t, old_next, _ = restart_schedule(old_file)
  old_tensors = restart_tensors(old_file)
  old_nsaved = old_tensors["file_number"].numel()
  if old_tensors["output_key"].numel() != old_nsaved:
    raise AssertionError(
        "old writer did not replace the precise suffix with legacy-only keys")

  new_dir = tests_dir / "restart_key_after_old_writer"
  new_card = scaled_case(new_dir, 1.6e-6, [restart, long, short])
  run(launch + [str(new_card), "--restart", str(old_file.resolve())],
      new_dir, env)
  new_t, new_next, _ = restart_schedule(
      sorted(new_dir.glob("*.restart"))[-1])
  expected = [new_next[0], advance(old_next[1], long["dt"], new_t),
              advance(old_next[2], short["dt"], new_t)]
  for slot in (1, 2):
    if abs(new_next[slot] - expected[slot]) > 1.e-15:
      raise AssertionError(
          f"new reader rebound old-writer slot {slot} to {new_next[slot]}, "
          f"expected positional legacy schedule {expected[slot]} from "
          f"{old_next} at {old_t}")
  if "output_key_v2" in old_tensors:
    raise AssertionError("old writer carried a stale precise-key buffer forward")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--build-dir", required=True)
  parser.add_argument("--build-type", required=True)
  parser.add_argument("--precision-only", action="store_true")
  parser.add_argument("--old-exe", type=Path)
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
  launch = [torchrun, "--master-port=" + env.get("MASTER_PORT", "29500"),
            "--no-python", "--nproc-per-node=1", str(exe)]
  if args.old_exe is not None:
    args.old_exe = args.old_exe.resolve()
    if not args.old_exe.is_file():
      raise FileNotFoundError(f"missing old executable {args.old_exe}")
  run_precision_reorder(base_yaml, tests_dir, launch, env, args.old_exe)
  if args.precision_only:
    return 0

  # leg 1: three output blocks, the two netcdf ones on DIFFERENT cadences
  case_dir = tests_dir / "restart_key_collision"
  case_yaml = write_case(base_yaml, case_dir, BASE_TLIM, [restart, fast, slow])
  run(launch + [str(case_yaml)], case_dir, env)

  restart_file = sorted(case_dir.glob("*.restart"))[-1]
  resume_t, saved_next, saved = restart_schedule(restart_file)
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

  # Each colliding stream must keep its own saved schedule. Also reject a newly numbered
  # out2 frame at the resume instant: that means it was treated as a new output block.
  final_t, final_next, _ = restart_schedule(sorted(case_dir.glob("*.restart"))[-1])
  cadence_errors = []
  for stream in ("out1", "out2"):
    fid = int(stream[3:])
    expected_next = saved_next[fid]
    while expected_next <= final_t:
      expected_next += edited["dt"]
    if abs(final_next[fid] - expected_next) > 1e-9:
      cadence_errors.append(
          f"{stream} next_time {final_next[fid]} (expected {expected_next} from its saved cadence)")
  out2_resumed = sorted(t for (stream, number), t in after.items()
                        if stream == "out2" and number >= saved[2])
  if out2_resumed and out2_resumed[0] <= resume_t + 1e-6:
    cadence_errors.append(
        f"out2 wrote an extra frame at resume time {out2_resumed[0]} (resume {resume_t})")
  if cadence_errors:
    raise AssertionError("streams did not keep their own restart cadence:\n  "
                         + "\n  ".join(cadence_errors))

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
