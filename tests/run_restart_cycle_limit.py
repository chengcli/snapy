#!/usr/bin/env python3
import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

SKIP_CODE = 125

try:
  import yaml
except Exception as exc:  # pragma: no cover - dependency guard
  print(f"Skipping test_restart_cycle_limit: yaml import failed: {exc}")
  sys.exit(SKIP_CODE)


def run(cmd, cwd: Path, log_path: Path) -> None:
  print(f"+ (cd {cwd} && {' '.join(cmd)})")
  with log_path.open("w") as log:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=log, stderr=subprocess.STDOUT)


def make_case_yaml(target: Path) -> None:
  source = Path(__file__).resolve().parent.parent / "examples" / "straka.yaml"
  with source.open("r") as f:
    config = yaml.safe_load(f)

  config["distribute"]["backend"] = "gloo"

  integration = config["integration"]
  integration["tlim"] = 1.0e9
  integration["nlim"] = 120
  integration["ncycle_out"] = 1

  config["outputs"] = [
      {"type": "restart", "dt": 30.0},
  ]

  with target.open("w") as f:
    yaml.safe_dump(config, f, sort_keys=False)


def parse_cycle_line(line: str) -> dict[str, float]:
  fields = {}
  for token in line.split():
    if "=" not in token:
      continue
    key, value = token.split("=", 1)
    fields[key] = value
  return {
      "cycle": int(fields["cycle"]),
      "time": float(fields["time"]),
      "dt": float(fields["dt"]),
      "mass0": float(fields["mass0"]),
      "energy": float(fields["energy"]),
  }


def parse_cycle_lines(log_path: Path) -> list[dict[str, float]]:
  lines = []
  for raw in log_path.read_text().splitlines():
    if raw.startswith("cycle="):
      lines.append(parse_cycle_line(raw))
  if not lines:
    raise ValueError(f"No cycle lines found in {log_path}")
  return lines


def parse_termination(log_path: Path) -> tuple[float, int]:
  lines = log_path.read_text().splitlines()
  for line in reversed(lines):
    if line.startswith("time=") and " cycle=" in line:
      parts = dict(token.split("=", 1) for token in line.split())
      return float(parts["time"]), int(parts["cycle"])
  raise ValueError(f"No termination summary found in {log_path}")


def assert_close(actual: float, expected: float, label: str,
                 atol: float = 1.0e-12) -> None:
  if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
    raise AssertionError(
        f"{label} mismatch: expected {expected:.17e}, got {actual:.17e}")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--build-dir", required=True)
  parser.add_argument("--build-type", required=True)
  args = parser.parse_args()

  build_dir = Path(args.build_dir).resolve()
  exe = build_dir / "bin" / f"straka.{args.build_type}"
  if not exe.exists():
    raise FileNotFoundError(f"missing executable {exe}")

  torchrun = shutil.which("torchrun")
  if torchrun is None:
    print("Skipping test_restart_cycle_limit: torchrun not found")
    return SKIP_CODE

  tests_dir = build_dir / "tests"
  base_dir = tests_dir / "restart_cycle_limit_base"
  restart_dir = tests_dir / "restart_cycle_limit_restart"
  for path in (base_dir, restart_dir):
    if path.exists():
      shutil.rmtree(path)
    path.mkdir(parents=True)

  make_case_yaml(base_dir / "straka.yaml")
  base_log = tests_dir / "restart_cycle_limit_base.log"
  run(
      [
          torchrun,
          "--no-python",
          "--nproc-per-node=2",
          str(exe),
          "straka.yaml",
      ],
      cwd=base_dir,
      log_path=base_log,
  )

  restart_file = base_dir / "straka.00001.restart"
  if not restart_file.exists():
    raise FileNotFoundError(f"missing restart file {restart_file}")

  shutil.copy2(base_dir / "straka.yaml", restart_dir / "straka.yaml")
  restart_log = tests_dir / "restart_cycle_limit_restart.log"
  run(
      [
          torchrun,
          "--no-python",
          "--nproc-per-node=2",
          str(exe),
          "straka.yaml",
          "--restart",
          str(restart_file.resolve()),
      ],
      cwd=restart_dir,
      log_path=restart_log,
  )

  base_cycles = parse_cycle_lines(base_log)
  restart_cycles = parse_cycle_lines(restart_log)

  first_restart = restart_cycles[0]
  matched = next(
      (entry for entry in base_cycles
       if math.isclose(entry["time"], first_restart["time"], rel_tol=0.0,
                       abs_tol=1.0e-12)),
      None,
  )
  if matched is None:
    raise AssertionError("Could not align restarted run with uninterrupted run")

  if first_restart["cycle"] != matched["cycle"]:
    raise AssertionError(
        f"restart resumed at cycle {first_restart['cycle']}, expected "
        f"{matched['cycle']}")

  for key in ("time", "dt", "mass0", "energy"):
    assert_close(first_restart[key], matched[key],
                 f"first resumed cycle {key}")

  base_term = parse_termination(base_log)
  restart_term = parse_termination(restart_log)
  assert_close(restart_term[0], base_term[0], "termination time", atol=1.0e-9)
  if base_term[1] != 120:
    raise AssertionError(
        f"base termination cycle mismatch: expected 120, got {base_term[1]}")
  if restart_term[1] != base_term[1]:
    raise AssertionError(
        f"termination cycle mismatch: expected {base_term[1]}, got {restart_term[1]}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
