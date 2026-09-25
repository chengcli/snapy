#!/usr/bin/env python3
"""An output block added to the card AFTER a restart was written must land on the same dt
grid as the blocks the restart restores. Before the fix its next_time was the resume
instant, so its frames trailed the restored streams by one cycle for the rest of the run."""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import netCDF4
import numpy as np
import torch

SKIP_CODE = 125

try:
  import yaml
except Exception as exc:  # pragma: no cover - dependency guard
  print(f"Skipping test_restart_new_output: yaml import failed: {exc}")
  sys.exit(SKIP_CODE)

FRAME_DT = 7.0     # netcdf cadence
RESTART_DT = 23.0  # not a multiple of FRAME_DT, so the resume instant is OFF the frame grid
BASE_TLIM = 40.0
RESUME_TLIM = 75.0


def run(cmd, cwd: Path, env) -> None:
  print(f"+ (cd {cwd} && {' '.join(cmd)})")
  subprocess.run(cmd, cwd=cwd, env=env, check=True)


def write_case(base_yaml: Path, case_dir: Path, tlim: float, outputs, wipe=True) -> Path:
  if wipe and case_dir.exists():
    shutil.rmtree(case_dir)
  case_dir.mkdir(parents=True, exist_ok=True)
  config = yaml.safe_load(base_yaml.read_text())
  dist = config.setdefault("distribute", {})
  dist["backend"] = "gloo"
  dist["blocks_per_process"] = int(dist.get("nb2", 1)) * int(dist.get("nb3", 1))  # one process
  integration = config.setdefault("integration", {})
  integration["tlim"] = tlim
  integration["nlim"] = -1
  integration["ncycle_out"] = 0
  config["output_dir"] = "."
  config["outputs"] = outputs
  target = case_dir / base_yaml.name
  target.write_text(yaml.safe_dump(config, sort_keys=False))
  return target


def stream_times(case_dir: Path) -> dict[str, list[float]]:
  times = defaultdict(list)
  for f in case_dir.glob("*.nc"):
    m = re.search(r"\.(out\d+)\.\d+\.nc$", f.name)
    if m is None:
      continue
    with netCDF4.Dataset(f, "r") as d:
      times[m.group(1)].append(float(np.asarray(d.variables["time"][:]).ravel()[0]))
  return {k: sorted(v) for k, v in times.items()}


def restart_schedule(path: Path):
  """(last_time, next_time[], file_number[]) from block 0 of a restart, bundle or plain."""
  with path.open("rb") as f:
    if f.readline().strip() == b"SNAPY_RESTART_BUNDLE_V1":
      n = int(f.readline())
      sizes = [int(f.readline().rstrip(b"\n").split(b"\t")[1]) for _ in range(n)]
      if f.readline() != b"\n":
        raise ValueError(f"{path}: bundle header missing terminator")
      payload = f.read(sizes[0])
    else:
      f.seek(0)
      payload = f.read()
  with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
    tmp.write(payload)
  try:
    m = torch.jit.load(tmp.name, map_location="cpu")
  finally:
    os.unlink(tmp.name)
  d = dict(m.named_buffers())
  d.update(dict(m.named_parameters()))
  return (float(d["last_time"].item()), d["next_time"].tolist(), d["file_number"].tolist())


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
  py_paths = [str(repo_root / "python"), str(repo_root)]
  if env.get("PYTHONPATH"):
    py_paths.append(env["PYTHONPATH"])
  env["PYTHONPATH"] = ":".join(py_paths)

  restart = {"type": "restart", "dt": RESTART_DT}
  prim = {"type": "netcdf", "variables": ["prim"], "dt": FRAME_DT}
  launch = [torchrun, "--no-python", "--nproc-per-node=1", str(exe)]

  base_dir = tests_dir / "restart_new_output_base"
  base_yaml_run = write_case(base_yaml, base_dir, BASE_TLIM, [restart, prim])
  run(launch + [str(base_yaml_run)], base_dir, env)
  restart_file = sorted(base_dir.glob("*.restart"))[-1]
  resume_t, _, _ = restart_schedule(restart_file)
  if resume_t % FRAME_DT == 0.0:
    raise AssertionError(f"resume time {resume_t} sits on the frame grid; the test cannot discriminate")

  # resume with a SECOND prim stream the restart file knows nothing about
  resumed_dir = tests_dir / "restart_new_output_resumed"
  resumed_yaml = write_case(base_yaml, resumed_dir, RESUME_TLIM, [restart, prim, dict(prim)])
  run(launch + [str(resumed_yaml), "--restart", str(restart_file.resolve())], resumed_dir, env)

  times = stream_times(resumed_dir)
  if set(times) != {"out1", "out2"}:
    raise AssertionError(f"expected streams out1/out2, found {sorted(times)}")
  after = {s: [t for t in ts if abs(t - resume_t) > 1e-3] for s, ts in times.items()}
  if len(after["out1"]) < 4:
    raise AssertionError(f"too few frames after resume to judge: {times['out1']}")
  if after["out2"] != after["out1"]:
    raise AssertionError(
        f"new stream is not co-temporal with the restored one after resume:\n"
        f"  out1 {after['out1']}\n  out2 {after['out2']}")

  _, next_time, file_number = restart_schedule(sorted(resumed_dir.glob("*.restart"))[-1])
  if len(next_time) != 3:
    raise AssertionError(f"resumed restart stores {len(next_time)} outputs, expected 3")
  if next_time[2] != next_time[1]:
    raise AssertionError(f"new stream's next_time {next_time[2]} != restored stream's {next_time[1]}")
  if min(next_time[2] % FRAME_DT, FRAME_DT - next_time[2] % FRAME_DT) > 1e-9:
    raise AssertionError(f"new stream's next_time {next_time[2]} is off its {FRAME_DT} grid")
  print(f"ok: resume at {resume_t:.3f}; frames {after['out1']}; next_time {next_time}; file_number {file_number}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
