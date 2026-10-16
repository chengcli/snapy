#!/usr/bin/env python3
"""Axis and unit attributes of the netcdf/pnetcdf outputs.

The file layout is {time, x1, x3, x2} with x2 the X axis and x3 the Y axis, so each
face coordinate must carry the long_name of its centre. The horizontal axes of the
cubed sphere are the equiangle angles (radian), not lengths, and pressure is in Pa
("pa" reads as pico-are in UDUNITS).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import netCDF4

SKIP_CODE = 125

try:
  import yaml
except Exception as exc:  # pragma: no cover - dependency guard
  print(f"Skipping test_netcdf_axis_attrs: yaml import failed: {exc}")
  sys.exit(SKIP_CODE)


def write_case(base_yaml: Path, case_dir: Path, out_type: str, edit) -> tuple[Path, int]:
  if case_dir.exists():
    shutil.rmtree(case_dir)
  case_dir.mkdir(parents=True)
  config = yaml.safe_load(base_yaml.read_text())
  dist = config.setdefault("distribute", {})
  dist["backend"] = "gloo"
  integration = config.setdefault("integration", {})
  integration["nlim"] = 1
  integration["ncycle_out"] = 0
  config["output_dir"] = "."
  config["outputs"] = [{"type": out_type, "variables": ["prim"], "dt": 1.e30}]
  nproc = edit(config, out_type)
  target = case_dir / base_yaml.name
  target.write_text(yaml.safe_dump(config, sort_keys=False))
  return target, nproc


def slab(config, out_type) -> int:
  config["geometry"]["cells"].update({"nx1": 16, "nx2": 16, "nx3": 8})
  config["distribute"].update({"nb2": 1, "nb3": 1, "blocks_per_process": 1})
  return 1


def cubed_sphere(config, out_type) -> int:
  """The six faces; pnetcdf output takes one block per process, so six ranks."""
  config["geometry"]["cells"].update({"nx2": 8, "nx3": 8})
  nproc = 6 if out_type == "pnetcdf" else 1
  config["distribute"]["blocks_per_process"] = 6 // nproc
  return nproc


def run_case(exe: Path, case_yaml: Path, nproc: int, env) -> bool:
  """Run one case; False if this build has no pnetcdf output."""
  cmd = [shutil.which("torchrun"), "--no-python", f"--nproc-per-node={nproc}", str(exe),
         str(case_yaml)]
  print(f"+ (cd {case_yaml.parent} && {' '.join(cmd)})")
  proc = subprocess.run(cmd, cwd=case_yaml.parent, env=env, capture_output=True, text=True)
  print(proc.stdout[-2000:], proc.stderr[-2000:])
  if proc.returncode != 0 and "requires PNETCDF=ON" in proc.stdout + proc.stderr:
    return False
  proc.check_returncode()
  return True


def check_files(case_dir: Path, units: str, has_press: bool) -> int:
  files = sorted(case_dir.glob("*.out*.nc"))
  if not files:
    raise AssertionError(f"no netcdf output in {case_dir}")
  for f in files:
    with netCDF4.Dataset(f, "r") as d:
      v = d.variables
      for face, axis in (("x2f", "X"), ("x3f", "Y")):
        if not v[face].long_name.startswith(axis):
          raise AssertionError(f"{f.name}: {face}.long_name = {v[face].long_name!r}, "
                               f"expected the {axis} axis like its centre")
      for name in ("x2", "x2f", "x3", "x3f"):
        if v[name].units != units:
          raise AssertionError(f"{f.name}: {name}.units = {v[name].units!r}, expected {units!r}")
      if has_press and v["press"].units != "Pa":
        raise AssertionError(f"{f.name}: press.units = {v['press'].units!r}, expected 'Pa'")
  return len(files)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--build-dir", required=True)
  parser.add_argument("--build-type", required=True)
  args = parser.parse_args()

  build_dir = Path(args.build_dir).resolve()
  repo_root = Path(__file__).resolve().parent.parent
  if shutil.which("torchrun") is None:
    raise FileNotFoundError("torchrun not found in PATH")
  env = os.environ.copy()
  env["BACKEND"] = "gloo"

  cases = [("straka", "straka.yaml", slab, "m", True),
           ("shallow_splash", "shallow_splash_mesh6.yaml", cubed_sphere, "radian", False)]
  for problem, deck, edit, units, has_press in cases:
    exe = build_dir / "bin" / f"{problem}.{args.build_type}"
    if not exe.exists():
      raise FileNotFoundError(f"missing executable {exe}")
    for out_type in ("netcdf", "pnetcdf"):
      case_dir = build_dir / "tests" / f"netcdf_axis_attrs_{problem}_{out_type}"
      case_yaml, nproc = write_case(repo_root / "examples" / deck, case_dir, out_type, edit)
      if not run_case(exe, case_yaml, nproc, env):
        print(f"skip {problem} {out_type}: built without PNETCDF")
        continue
      n = check_files(case_dir, units, has_press)
      print(f"ok: {problem} {out_type}: {n} file(s), horizontal units {units!r}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
