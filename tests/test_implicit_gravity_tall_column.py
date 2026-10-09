#!/usr/bin/env python3
"""Time-stepped Cartesian and spherical x1 face-work columns (#288).

The discretely balanced 11.3H column must remain at rest through the tested
vertical acoustic Courants up to 250. Results include the first failed rung;
passing a finite ladder does not establish a universal stability threshold.
Cartesian retains eight periodic copies; spherical x1 uses one angular cell
to test a strictly radial column. Angular-mode stability is a separate gate.
"""
import argparse
import math
import json
import os
import sys
import tempfile

import torch
import yaml

GRAV, GAMMA, RD, T0, PS = 23.1, 1.403461, 3515.0, 786.0, 1.0e7
NZ, DZ, NX2 = 45, 29946.8085106, 8
COURANTS = (6.6, 65.6, 100.0, 197.0, 250.0)
RADIUS = 7.e7
W_TOL = 1.0e-7  # m/s


def config(geometry="cartesian"):
    rgas = 8.31446261815324
    cfg = {
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
        "forcing": {"const-gravity": {"grav1": -GRAV, "gravity-work": "face",
                                          "gravity-work-fixer": False}},
    }

    if geometry == "spherical-polar":
        cfg["geometry"]["type"] = geometry
        cfg["geometry"]["cells"]["nx2"] = 1
        cfg["geometry"]["bounds"] = {
            "x1min": RADIUS, "x1max": RADIUS + NZ * DZ,
            "x2min": math.pi / 2 - 0.2, "x2max": math.pi / 2 + 0.2,
            "x3min": 0.0, "x3max": 0.05}
    return cfg


def run(dt, nstep, device, geometry="cartesian"):

    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(geometry), f)
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

    nx2 = config(geometry)["geometry"]["cells"]["nx2"]
    jl = ng if nx2 > 1 else 0
    interior = (Ellipsis, slice(jl, jl + nx2), slice(ng, ng + NZ))
    nstage = len(block.module("intg").stages)
    wmax, history = 0.0, []
    for n in range(1, nstep + 1):
        for stage in range(nstage):
            block.forward(block_vars, dt, stage)
        redo = block.check_redo(block_vars)
        if redo:
            return False, n, wmax, history, err
        u = block_vars["hydro_u"][interior]
        if not torch.isfinite(u).all():
            return False, n, wmax, history, err
        wn = (u[kIV1] / u[kIDN]).abs().max().item()
        wmax = max(wmax, wn)
        if wmax >= W_TOL:
            return False, n, wmax, history, err
        if n in (1, 10, 20, 30, nstep):
            history.append((n, wn))
    return True, nstep, wmax, history, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nstep", type=int, default=40)
    ap.add_argument("--geometry", choices=("cartesian", "spherical-polar", "both"), default="both")
    ap.add_argument("--courants", type=float, nargs="+", default=COURANTS)
    args = ap.parse_args()
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 0
    torch.set_default_dtype(torch.float64)

    cs = math.sqrt(GAMMA * RD * T0)
    failures = []
    geometries = ("cartesian", "spherical-polar") if args.geometry == "both" else (args.geometry,)
    for geometry in geometries:
        first_failure = None
        for courant in args.courants:
            dt = courant * DZ / cs
            finite, n, wmax, history, err = run(dt, args.nstep, args.device, geometry)
            passed = finite and wmax < W_TOL
            print(json.dumps({"geometry": geometry, "courant": courant, "dt": dt,
                              "steps": n, "passed": passed, "max_w": wmax,
                              "balance_error": err, "history": history}), flush=True)
            if not passed:
                if first_failure is None:
                    first_failure = courant
                failures.append((geometry, courant, n, wmax))
        print(json.dumps({"geometry": geometry, "first_failing_courant": first_failure,
                          "tested_courants": args.courants}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
