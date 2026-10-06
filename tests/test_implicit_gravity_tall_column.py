#!/usr/bin/env python3
"""A tall isothermal column at rest must stay at rest under the implicit solver at a large time step.

The column is 45 cells of dz = 29.9 km (11.3 scale heights; g = 23.1 m/s^2, T = 786 K,
gamma = 1.403461, Rd = 3515 J/kg/K, p_bot = 100 bar), 8 periodic columns in x2,
reflecting x1 walls, rk3 + weno5 + lmars, implicit-scheme 9 (vic-full), no
perturbation. It is put into the scheme's own discrete hydrostatic balance with
snapy.balance_column and run for NSTEP fixed steps at two rungs:
  dt = 997 s  (acoustic Courant 65.6, what the implicit solver exists for)
  dt = 100 s  (acoustic Courant 6.6, control)
At each rung the run must stay finite with max |w| < W_TOL.

W_TOL = 1e-7 m/s: a column that keeps its balance stays near 5e-9 m/s on the first
step (the balance residual, rtol 1e-10) and decays to ~1e-12 m/s, at both rungs and
over 10 days. An unstable one grows by ~2.7x per step from ~1e-6 m/s at step 10 and
is non-finite or above 1e10 m/s by step 40.

The implicit matrix linearises the cell-centred gravity work (g * rho w in the
energy row), while the energy equation also receives face-mass-flux gravity work
terms added after the implicit solve. Those terms are not part of the implicit
operator, so the large-step rung is unstable.

  python test_implicit_gravity_tall_column.py [--device cpu] [--nstep 40]
"""
import argparse
import math
import os
import sys
import tempfile

import torch
import yaml

GRAV, GAMMA, RD, T0, PS = 23.1, 1.403461, 3515.0, 786.0, 1.0e7
NZ, DZ, NX2 = 45, 29946.8085106, 8
RUNGS = (("large step", 997.0), ("control", 100.0))
W_TOL = 1.0e-7  # m/s


def config():
    rgas = 8.31446261815324
    return {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": NZ * DZ, "x2min": 0.0, "x2max": 31415926.53589793,
                                "x3min": 0.0, "x3max": 3926990.8169872416},
                     "cells": {"nx1": NZ, "nx2": NX2, "nx3": 1, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": GAMMA, "weight": rgas / RD,
                                  "density-floor": 3.4682e-18, "pressure-floor": 3.4682e-18,
                                  "temperature-floor": 1.e-6, "limiter": True},
            "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                            "horizontal": {"type": "weno5", "scale": True, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.5, "implicit-scheme": 9, "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -GRAV}},
    }


def run(dt, nstep, device):
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(), f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))

    ng = 3
    z = (torch.arange(NZ, dtype=torch.float64) + 0.5) * DZ
    p = PS * torch.exp(-GRAV * z / (RD * T0))
    col = torch.zeros(kIPR + 1, 1, 1, NZ, dtype=torch.float64)
    col[kIDN, 0, 0] = p / (RD * T0)
    col[kIPR, 0, 0] = p
    wb, err, _ = snapy.balance_column(col, torch.full((NZ,), DZ, dtype=torch.float64), GRAV)
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + NZ] = wb[c, 0, 0].to(w)
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ:] = w[c][..., ng + NZ - 1:ng + NZ]
    block_vars, _ = block.initialize({"hydro_w": w})

    interior = (Ellipsis, slice(ng, ng + NX2), slice(ng, ng + NZ))
    nstage = len(block.module("intg").stages)
    wmax, history = 0.0, []
    for n in range(1, nstep + 1):
        for stage in range(nstage):
            block.forward(block_vars, dt, stage)
        u = block_vars["hydro_u"][interior]
        if not torch.isfinite(u).all():
            return False, n, wmax, history, err
        wn = (u[kIV1] / u[kIDN]).abs().max().item()
        wmax = max(wmax, wn)
        if n in (1, 10, 20, 30, nstep):
            history.append((n, wn))
    return True, nstep, wmax, history, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nstep", type=int, default=40)
    args = ap.parse_args()
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 0
    torch.set_default_dtype(torch.float64)

    cs = math.sqrt(GAMMA * RD * T0)
    failures = []
    for name, dt in RUNGS:
        finite, n, wmax, history, err = run(dt, args.nstep, args.device)
        hist = "  ".join("step %d %.3e" % h for h in history)
        print("%-10s dt=%6.1f s  acoustic Courant %5.1f  balance err %.1e  %s" % (name, dt, cs * dt / DZ, err, hist))
        if not finite:
            failures.append("%s (dt %g s): state non-finite at step %d (max|w| before %.3e m/s)" % (name, dt, n, wmax))
        elif not wmax < W_TOL:
            failures.append("%s (dt %g s): max|w| = %.3e m/s over %d steps (tol %.0e)" % (name, dt, wmax, n, W_TOL))
        else:
            print("%-10s PASS: max|w| = %.3e m/s over %d steps (tol %.0e)" % (name, wmax, n, W_TOL))
    for f in failures:
        print("FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
