#!/usr/bin/env python3
"""Check that pd_regrid moves a restart onto a taller x1 domain correctly.

The source is a synthetic isothermal hydrostatic column, which has an exact
answer everywhere: rho and P fall as exp(-z/H) with a known H, so both the
interpolation inside the old domain and the extension above it can be checked
against the analytic profile rather than against themselves.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

SKIP_CODE = 125

try:
  import numpy as np
  import torch
  import yaml
except Exception as exc:  # pragma: no cover - dependency guard
  print(f"Skipping test_restart_regrid: import failed: {exc}")
  sys.exit(SKIP_CODE)

API_DIR = Path(__file__).resolve().parents[1] / "python" / "api"
sys.path.insert(0, str(API_DIR))

try:
  import pd_regrid
except Exception as exc:  # pragma: no cover - dependency guard
  print(f"Skipping test_restart_regrid: pd_regrid import failed: {exc}")
  sys.exit(SKIP_CODE)

NGHOST = 3
NVAR = 7  # rho, 3 velocities, energy, 2 mass fractions
H = 50.0e3  # scale height, m
RHO0 = 1.0e-2
CV_T = 2.0e5  # specific internal energy, J/kg
NX2 = 8


def config(x1max: float, nx1: int) -> dict:
  return {
      # Slot 5 is a vapour, slot 6 a condensate by Snapy's parenthesised name.
      "species": [{"name": "dry"}, {"name": "H2O"}, {"name": "H2O(l)"}],
      "boundary-condition": {"external": {"x1-inner": "reflecting",
                                          "x1-outer": "reflecting"}},
      "geometry": {
          "bounds": {"x1min": 0.0, "x1max": x1max,
                     "x2min": 0.0, "x2max": 1.0e5},
          "cells": {"nx1": nx1, "nx2": NX2, "nx3": 1, "nghost": NGHOST},
      }
  }


def analytic(z: np.ndarray) -> np.ndarray:
  return RHO0 * np.exp(-z / H)


def build_restart(grid, path: Path) -> None:
  """An isothermal column, uniform in x2, with ghosts mirrored about the walls."""
  z = grid.x1v()
  rho = analytic(z)
  u = np.zeros((NVAR, grid.nc3, grid.nc2, grid.nc1), dtype=np.float64)
  u[0] = rho
  u[1] = rho * 3.0            # a uniform x1 velocity
  u[4] = rho * CV_T
  u[5] = rho * 0.2            # mass fractions, constant with height
  u[6] = rho * 0.05
  w = np.zeros_like(u)
  w[0] = rho
  w[1] = 3.0
  w[4] = rho * CV_T * 0.4     # a pressure that falls like density
  w[5] = 0.2
  w[6] = 0.05
  tensors = {
      "hydro_u": torch.from_numpy(u),
      "hydro_w": torch.from_numpy(w),
      "last_time": torch.tensor([1234.5], dtype=torch.float64),
      "last_cycle": torch.tensor([678], dtype=torch.int64),
      "file_number": torch.tensor([2, 3], dtype=torch.int64),
      "next_time": torch.tensor([10.0, 20.0], dtype=torch.float64),
  }
  pd_regrid.write_part(tensors, str(path))


def check(name: str, ok: bool, detail: str = "") -> None:
  print(f"  {'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
  if not ok:
    raise SystemExit(1)


def main() -> int:
  old_cfg, new_cfg = config(600.0e3, 142), config(760.0e3, 180)
  old, new = pd_regrid.Grid(old_cfg), pd_regrid.Grid(new_cfg)

  with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "old.yaml").write_text(yaml.safe_dump(old_cfg))
    (tmp / "new.yaml").write_text(yaml.safe_dump(new_cfg))
    build_restart(old, tmp / "in.restart")

    cmd = [sys.executable, str(API_DIR / "pd_regrid.py"),
           "--old-config", str(tmp / "old.yaml"),
           "--new-config", str(tmp / "new.yaml"),
           "--restart", str(tmp / "in.restart"),
           "--output", str(tmp / "out.restart")]
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    out = pd_regrid.read_part(str(tmp / "out.restart"))

    check("shape", tuple(out["hydro_u"].shape) == (NVAR, new.nc3, new.nc2, new.nc1),
          str(tuple(out["hydro_u"].shape)))
    check("clock carried over",
          out["last_time"].item() == 1234.5 and out["last_cycle"].item() == 678)

    u = out["hydro_u"].numpy()
    w = out["hydro_w"].numpy()
    z = new.x1v()[NGHOST:NGHOST + new.nx1]
    sl = (0, NGHOST, slice(NGHOST, NGHOST + new.nx1))

    rho = u[0][sl]
    want = analytic(z)
    err = np.abs(rho / want - 1.0)
    # The old domain ends at 600 km; above it the tool extrapolates, and an
    # isothermal column is exactly what it assumes, so one tolerance covers both.
    check("density matches the analytic column", err.max() < 2.0e-3,
          f"max relative error {err.max():.2e} at z = {z[err.argmax()] / 1e3:.1f} km")

    check("extension reaches the new lid", z[-1] > 755.0e3, f"{z[-1] / 1e3:.1f} km")
    check("velocity preserved", np.abs(u[1][sl] / rho - 3.0).max() < 1e-9)
    check("specific energy preserved",
          np.abs(u[4][sl] / rho - CV_T).max() / CV_T < 1e-9)
    check("vapour mass fraction preserved",
          np.abs(u[5][sl] / rho - 0.2).max() < 1e-12)
    # The condensate is carried inside the old domain and dropped above it.
    zsrc = pd_regrid.Grid(old_cfg).x1v()[NGHOST:NGHOST + old.nx1 - 2][-1]
    inside, above = z <= zsrc, z > zsrc
    check("condensate kept inside the old domain",
          np.abs(u[6][sl][inside] / rho[inside] - 0.05).max() < 1e-12)
    check("condensate not carried into the extension",
          np.abs(u[6][sl][above]).max() == 0.0,
          f"{int(above.sum())} extension levels")
    check("primitive pressure falls like density",
          np.abs(w[4][sl] / (rho * CV_T * 0.4) - 1.0).max() < 2.0e-3)
    # P/rho is proportional to T/mu, so a constant one means the extension
    # above the old lid is isothermal rather than slowly drifting.
    t_like = w[4][sl] / rho
    check("extension stays isothermal",
          np.abs(t_like / t_like[0] - 1.0).max() < 1.0e-9,
          f"max drift {np.abs(t_like / t_like[0] - 1.0).max():.2e}")

    # A widening domain needs the periodic x2 resample, and refuses without it.
    wide_cfg = config(760.0e3, 180)
    wide_cfg["geometry"]["cells"]["nx2"] = NX2 + 4
    (tmp / "wide.yaml").write_text(yaml.safe_dump(wide_cfg))
    cmd[cmd.index("--new-config") + 1] = str(tmp / "wide.yaml")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    check("nx2 change refused without --x2-mode stretch", proc.returncode != 0)
    proc = subprocess.run(cmd + ["--x2-mode", "stretch"], capture_output=True, text=True)
    check("nx2 change accepted with --x2-mode stretch", proc.returncode == 0, proc.stderr[-200:])
    wide = pd_regrid.read_part(str(tmp / "out.restart"))
    check("stretched shape",
          tuple(wide["hydro_u"].shape) == (NVAR, 1, NX2 + 4 + 2 * NGHOST, new.nc1),
          str(tuple(wide["hydro_u"].shape)))

    # A reflecting wall stores the mirror of the interior, with the
    # wall-normal velocity negated. Snapy does not refill these on restart, so
    # the file itself has to be right.
    ug = out["hydro_u"].numpy()
    for name, gh, act in (
        ("bottom", ug[:, 0, NGHOST, NGHOST - 1::-1], ug[:, 0, NGHOST, NGHOST:2 * NGHOST]),
        ("top", ug[:, 0, NGHOST, -NGHOST:], ug[:, 0, NGHOST, -NGHOST - 1:-2 * NGHOST - 1:-1]),
    ):
      check(f"{name} x1 ghosts mirror the interior",
            np.abs(gh[0] - act[0]).max() < 1e-12,
            f"density {gh[0]} vs {act[0]}")
      check(f"{name} x1 ghosts negate the wall-normal velocity",
            np.abs(gh[1] + act[1]).max() < 1e-12,
            f"momentum {gh[1]} vs {act[1]}")

  print("test_restart_regrid passed")
  return 0


if __name__ == "__main__":
  sys.exit(main())
