#!/usr/bin/env python3
"""kappa_iso leaves a resting adiabatic column at rest.

A dry ideal-gas column at rest with T = Ts - g z / cp (constant potential temperature)
in hydrostatic balance is neutrally stable, so a conduction that stands in for turbulent
mixing (flux proportional to d(theta)/dn) moves no heat through it. Two arms step the same
column (64 cells, reflecting x1 walls) N times at one shared dt, with kappa_iso = K and
with kappa_iso = 0; the second is the control, i.e. the scheme's own hydrostatic
imbalance. The oracle is max |T(K) - T(0)| over the active cells, which must stay below
TOL. Conduction on T (flux -kappa_iso * (rho cv)_face * dT/dn) fails it: dT/dz = -g/cp
is not zero, so the walls drift by about -/+ K g / (cp dz) * t.

  python test_kappa_adiabatic_rest.py [--device cuda] [--steps 10] [--kappa 75]
"""
import argparse
import os
import sys
import tempfile

import torch
import yaml

G, TS, P0 = 9.8, 300.0, 1.0e5
NX1, ZTOP = 64, 6.4e3
TOL = 1.0e-9  # K; the control's own drift is ~1e-13 K at N = 10


def config(kappa):
    return {
        "reference-state": {"Tref": TS, "Pref": P0},
        "species": [{"name": "dry", "composition": {"O": 0.42, "N": 1.56, "Ar": 0.01}, "cv_R": 2.5}],
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": ZTOP, "x2min": 0.0, "x2max": 1.0,
                                "x3min": 0.0, "x3max": 1.0},
                     "cells": {"nx1": NX1, "nx2": 1, "nx3": 1, "nghost": 3}},
        "dynamics": {"equation-of-state": {"type": "ideal-gas", "limiter": False},
                     "reconstruct": {"vertical": {"type": "weno5", "scale": False, "shock": False},
                                     "horizontal": {"type": "weno5", "scale": False, "shock": False}},
                     "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting"}},
        "integration": {"type": "rk3", "cfl": 0.9},
        "forcing": {"const-gravity": {"grav1": -G},
                    "diffusion": {"nu_iso": 0.0, "kappa_iso": float(kappa)}},
    }


def make_block(kappa, device):
    from snapy import MeshBlock, MeshBlockOptions

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(kappa), f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))
    return block


def gas_constants():
    import kintera

    Rd = 8.31446261815324 / kintera.species_weights()[0]
    return Rd, 3.5 * Rd  # cv_R = 2.5


def column(block):
    """The adiabat at rest: T = Ts - g z / cp, p = P0 (T / Ts)^(cp / Rd)."""
    from snapy import kIDN, kIPR

    Rd, cp = gas_constants()
    buf = dict(block.named_buffers())
    w = buf["hydro.D"].clone().zero_()
    z = buf["coord.x1v"].to(w).view(1, 1, -1).expand_as(w[0])
    T = TS - G * z / cp
    w[kIPR] = P0 * (T / TS) ** (cp / Rd)
    w[kIDN] = w[kIPR] / (Rd * T)
    return w


def temperature(block_vars):
    from snapy import kIDN, kIPR

    Rd, _ = gas_constants()
    w = block_vars["hydro_w"]
    return (w[kIPR] / (w[kIDN] * Rd)).flatten()


def run_arm(kappa, nsteps, dt, device):
    block = make_block(kappa, device)
    block_vars, _ = block.initialize({"hydro_w": column(block)})
    for _ in range(nsteps):
        for stage in range(len(block.module("intg").stages)):
            block.forward(block_vars, dt, stage)
    return temperature(block_vars)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--kappa", type=float, default=75.0)
    args = ap.parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("SKIP: cuda requested but not available")
        return 125
    torch.set_default_dtype(torch.float64)

    ref = make_block(0.0, args.device)  # builds the species table first
    x1v = dict(ref.named_buffers())["coord.x1v"].flatten()
    active = (x1v > 0.0) & (x1v < ZTOP)
    ref_vars, _ = ref.initialize({"hydro_w": column(ref)})
    dt = float(ref.max_time_step(ref_vars))

    T0 = temperature(ref_vars)[active]
    Tc = run_arm(0.0, args.steps, dt, args.device)[active]
    Tk = run_arm(args.kappa, args.steps, dt, args.device)[active]
    d = Tk - Tc
    print("N=%d dt=%.4f s  control max|T-T0| %.3e K  kappa=%g: max|T(K)-T(0)| %.3e K "
          "(bottom %+.3e, top %+.3e)" % (args.steps, dt, (Tc - T0).abs().max().item(), args.kappa,
                                          d.abs().max().item(), d[0].item(), d[-1].item()))
    if d.abs().max().item() > TOL:
        print("FAIL (%s): kappa_iso moves heat in a resting constant-theta column" % args.device)
        return 1
    print("PASS (%s): kappa_iso leaves a resting adiabatic column at rest" % args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
