#!/usr/bin/env python3
"""A resumed run does not write its source restart and t0 frames again.

make_outputs advances an output's schedule after writing it, so a restart that
saved the schedule as it was while writing would fire again at the resume instant,
and so would every output listed after it on the same cadence.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SKIP_CODE = 125

try:
  import yaml
except Exception as exc:  # pragma: no cover - dependency guard
  print(f"Skipping test_restart_resume_schedule: yaml import failed: {exc}")
  sys.exit(SKIP_CODE)

OUT_DT = 3.0
TLIM = 7.0


def write_case(base_yaml: Path, case_dir: Path) -> Path:
  if case_dir.exists():
    shutil.rmtree(case_dir)
  case_dir.mkdir(parents=True)
  config = yaml.safe_load(base_yaml.read_text())
  config["geometry"]["cells"].update({"nx1": 16, "nx2": 32})
  dist = config.setdefault("distribute", {})
  dist.update({"backend": "gloo", "nb2": 1, "nb3": 1, "blocks_per_process": 1})
  integration = config.setdefault("integration", {})
  integration.update({"tlim": TLIM, "nlim": -1, "ncycle_out": 0})
  config["output_dir"] = "."
  config["outputs"] = [{"type": "restart", "dt": OUT_DT},
                       {"type": "netcdf", "variables": ["prim"], "dt": OUT_DT}]
  target = case_dir / base_yaml.name
  target.write_text(yaml.safe_dump(config, sort_keys=False))
  return target


def numbered(case_dir: Path) -> dict[str, list[int]]:
  """File numbers written per kind (restart, netcdf)."""
  found = {"restart": [], "netcdf": []}
  for f in case_dir.iterdir():
    m = re.search(r"\.(\d{5})\.restart$", f.name)
    if m:
      found["restart"].append(int(m.group(1)))
    m = re.search(r"\.out\d+\.(\d{5})\.nc$", f.name)
    if m:
      found["netcdf"].append(int(m.group(1)))
  return {k: sorted(v) for k, v in found.items()}


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--build-dir", required=True)
  parser.add_argument("--build-type", required=True)
  args = parser.parse_args()

  build_dir = Path(args.build_dir).resolve()
  repo_root = Path(__file__).resolve().parent.parent
  exe = build_dir / "bin" / f"straka.{args.build_type}"
  if not exe.exists():
    raise FileNotFoundError(f"missing executable {exe}")
  torchrun = shutil.which("torchrun")
  if torchrun is None:
    raise FileNotFoundError("torchrun not found in PATH")
  env = os.environ.copy()
  env["BACKEND"] = "gloo"
  launch = [torchrun, "--no-python", "--nproc-per-node=1", str(exe)]

  base_dir = build_dir / "tests" / "restart_resume_schedule_base"
  base_yaml = write_case(repo_root / "examples" / "straka.yaml", base_dir)
  subprocess.run(launch + [str(base_yaml)], cwd=base_dir, env=env, check=True)
  base = numbered(base_dir)
  if any(base[k][:3] != [0, 1, 2] for k in base):
    raise AssertionError(f"base run wrote {base}, expected #0-#2 of both")
  source = next(base_dir.glob("*.00001.restart"))

  resumed_dir = build_dir / "tests" / "restart_resume_schedule_resumed"
  resumed_yaml = write_case(repo_root / "examples" / "straka.yaml", resumed_dir)
  subprocess.run(launch + [str(resumed_yaml), "--restart", str(source)],
                 cwd=resumed_dir, env=env, check=True)
  resumed = numbered(resumed_dir)
  if any(not resumed[k] or resumed[k][0] != 2 for k in resumed):
    raise AssertionError(
        f"resumed from restart #1, the run wrote {resumed}; expected both to start at #2 "
        f"(a #1 is the source restart or its t0 frame written again)")
  print(f"ok: base {base}, resumed from #1 wrote {resumed}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
