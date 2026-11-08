#!/usr/bin/env python3
"""A uniform passive-tracer ratio r = s / rho_dry stays uniform on nx2 = 4 identical columns.
Dry isothermal column on a stretched x1 grid (snapy.balance_column) with a pressure bump, no wind,
eos limiter off, covariance on (default), one rk3 step at x1 Courant 15 (implicit-scheme 1).
With nx2 > 1 the covariance rows switch the scalar positivity limiter on with the eos limiter off;
where the unlimited mass flux drains a cell's dry air past zero (outflow Courant 2.08 here) it
scaled the tracer flux by that cell's theta (0.4808) and left the mass flux alone.
Positivity: one filled cell (5, the over-drained one) in an empty column, and one empty cell (6,
above it) in a uniform one; the implicit refill from an empty neighbour carries no tracer back, so
draining cell 5 as far as its dry air drove it negative (-4.455e-05 and -3.702e-05).
python test_scalar_ratio_2d.py [--device cuda] [--scheme 0]"""
import math, os, sys, tempfile, torch, yaml, snapy
from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR

DEV = "cuda" if "cuda" in sys.argv else "cpu"
SCHEME = 0 if "0" in sys.argv else 1
NX1, ZTOP, T0, RG, G, R0 = 8, 3.2e4, 320., 3900., 10., 1e-3


def deck(nx2, scheme, nx3=1):
    weno = {"type": "weno5", "scale": True, "shock": True}
    periodic = {f"x{d}-{s}": "periodic" for d in (2, 3) for s in ("inner", "outer")}
    return {
        "reference-state": {"Tref": 0., "Pref": 1.e5},
        "species": [{"name": "dry", "composition": {"H": 1.6667, "He": 0.16667}, "cv_R": 2.5},
                    {"name": "vapor", "composition": {"H": 2, "O": 1}, "cv_R": 2.5, "u0_R": 0.},
                    {"name": "cloud", "composition": {"H": 2, "O": 1}, "cv_R": 9.0, "u0_R": -3430.}],
        "reactions": [{"equation": "vapor <=> cloud", "type": "nucleation",
                       "rate-constant": {"formula": "h2o_ideal"}}],
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0., "x1max": ZTOP, "x2min": 0., "x2max": 4.e6,
                                "x3min": 0., "x3max": 4.e6 if nx3 > 1 else 1.e3},
                     "cells": {"nx1": NX1, "nx2": nx2, "nx3": nx3, "nghost": 3,
                               "x1rat": 4. ** (1. / (NX1 - 1))}},
        "distribute": {"layout": "slab", "nb2": 1, "nb3": 1, "verbose": False},
        "dynamics": {"equation-of-state": {"type": "ideal-moist", "density-floor": 1.e-10,
                                           "pressure-floor": 1.e-10, "limiter": False},
                     "reconstruct": {"vertical": weno, "horizontal": weno},
                     "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            **periodic}},
        "integration": {"type": "rk3", "cfl": 0.9, "implicit-scheme": scheme, "nlim": -1,
                        "tlim": 1.e9, "ncycle_out": 0},
        "forcing": {"const-gravity": {"grav1": -G}},
        "scalar": {"nvar": 1, "names": ["tracer"], "reconstruct": {"type": "weno5", "shock": False}},
    }


def column(p, like):  # a number, or one value per interior x1 cell; ghosts take the edge values
    p = torch.as_tensor(p, dtype=like.dtype, device=like.device).reshape(-1)
    return torch.cat([p[:1].repeat(3), p, p[-1:].repeat(3)]) if p.numel() > 1 else p.expand(like.size(-1))


def setup(nx2, scheme, q0=0., r=R0, nx3=1):  # the block, its state at tracer ratio r and vapour fraction q0, dt
    with tempfile.TemporaryDirectory() as wd:
        p = os.path.join(wd, "deck.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(deck(nx2, scheme, nx3), f)
        block = MeshBlock(MeshBlockOptions.from_yaml(p))
    block.to(torch.device(DEV))
    f = block.module("coord").buffer("x1f").double().cpu()[3:3 + NX1 + 1]
    h, hs = f[1:] - f[:-1], RG * T0 / G
    pbar = 1.e5 * hs * (torch.exp(-f[:-1] / hs) - torch.exp(-f[1:] / hs)) / h  # exact cell means
    col = torch.zeros((5, 1, 1, NX1))
    col[0, 0, 0], col[4, 0, 0] = pbar / (RG * T0), pbar
    wb, _, _ = snapy.balance_column(col, h, G, wall_clamp=True, rtol=1.e-13, max_iter=400,
                                    stretched=True)
    wb = wb.to(DEV)
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    for k, src in ((kIDN, 0), (kIPR, 4)):
        w[k][..., 3:3 + NX1] = wb[src]
        w[k][..., :3] = wb[src][..., :1]
        w[k][..., 3 + NX1:] = wb[src][..., -1:]
    w[kICY] = column(q0, w[kICY])
    z = block.module("coord").buffer("x1v").double()
    w[kIPR] *= (1. + torch.exp(-((z - 0.5 * ZTOP) / 2.e3) ** 2 / 2)).view(1, 1, -1)
    v, _ = block.initialize({"hydro_w": w, "scalar_r": torch.zeros_like(w[:1]) + column(r, w)})
    return block, v, 15. * float(h.min()) / math.sqrt(1.4 * RG * T0)


def run(nx2, nx3=1):  # max|r/r0 - 1| over one rk3 step, and min/max of the stage-0 x1 F_s / (r0 F_rho)
    block, v, dt = setup(nx2, SCHEME, nx3=nx3)
    B, J = dict(block.named_buffers()), slice(3, -3) if nx2 > 1 else slice(None)
    K = slice(3, -3) if nx3 > 1 else slice(None)
    dev, face = 0., None
    for stage in range(3):
        block.forward(v, dt, stage)
        if stage == 0:
            Fr, Fs = B["hydro.F1"][kIDN][K, J, 3:3 + NX1 + 1], B["scalar.F1"][0, K, J, 3:3 + NX1 + 1]
            face = Fs[Fr != 0] / (R0 * Fr[Fr != 0])
        r = v["scalar_s"][0, K, J, 3:-3] / v["hydro_u"][kIDN][K, J, 3:-3] / R0
        dev = max(dev, float((r - 1).abs().max()))
    return dev, float(face.min()), float(face.max())


def one(a, k, hole):  # one cell k at a in an empty column, or one empty cell k in a column at a
    p = torch.full((NX1,), a if hole else 0., dtype=torch.float64)
    p[k] = 0. if hole else a
    return p


def lowest(species, p):  # min of the tracer (or vapour) density over one rk3 step at nx2 = 4
    block, v, dt = setup(4, SCHEME, **({"q0": p} if species else {"r": p}))
    lo = float("inf")
    for stage in range(3):
        block.forward(v, dt, stage)
        x = v["hydro_u"][kICY] if species else v["scalar_s"][0]
        lo = min(lo, float(x[:, 3:-3, 3:-3].min()))
    return lo


def positivity(species, a):  # True if a one-cell or one-hole column goes below round-off
    bad = False
    for name, p in (("one-cell 5", one(a, 5, False)), ("one-hole 6", one(a, 6, True))):
        lo = lowest(species, p)
        print(f"{DEV} scheme {SCHEME} nx2=4 {name}: min {'rho_v' if species else 's'} = {lo:.3e}")
        bad |= lo < -1e-12 * a
    return bad


if __name__ == "__main__":
    if DEV == "cuda" and not torch.cuda.is_available(): sys.exit(125)  # no visible GPU: skip
    bad = positivity(False, R0)  # 9032cbef: -4.455e-05, -3.702e-05; main 68f4e066: 0, 9.08e-06
    for nx2, nx3 in ((4, 1), (4, 4), (1, 1)):  # nx3 = 4 adds the x3 covariance rows (main: 0.242)
        dev, fmin, fmax = run(nx2, nx3)
        print(f"{DEV} scheme {SCHEME} nx2={nx2} nx3={nx3}: max|r/r0-1| = {dev:.3e}, "
              f"stage-0 x1 F_s/(r0 F_rho) in [{fmin:.4f}, {fmax:.4f}]")
        bad |= dev > 1e-12 or abs(fmin - 1) > 1e-12 or abs(fmax - 1) > 1e-12
    sys.exit(1 if bad else 0)  # main 68f4e066: nx2=4 1.07e+00, faces down to 0.4808; nx2=1 2.2e-16
