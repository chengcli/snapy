#!/usr/bin/env python3
"""A uniform tracer ratio r = s / rho_dry stays uniform with the eos limiter ON.
Moist column, nx1 = 32 stretched (x1rat 4^(1/31)), nx2 = 4, a pressure bump in one column, one rk3
step at vertical acoustic Courant 30 and 100 (implicit-scheme 1; --scheme 9), accepted by the
redo rule. In stage 1 the bump drains cells 19 and 22 past their dry air (outflow 1.14 and 1.19 of
the content at C30), and the limiter scaled the tracer flux on all their outgoing faces by their
theta (0.8799, 0.8389) while the mass flux, never limited, moved in full. Positivity: one filled
cell 22 in an empty column and one empty cell 22 in a uniform one.
python test_tracer_ratio_limiter.py [--device cuda] [--scheme 9]"""
import math, os, sys, tempfile, torch, yaml, snapy
from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR

DEV = "cuda" if "cuda" in sys.argv else "cpu"
SCHEME = 9 if "9" in sys.argv else 1
NX1, NX2, ZTOP, T0, RG, G, R0, Q0 = 32, 4, 3.2e4, 320., 3900., 10., 1e-3, 5e-3


def deck():
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
                     "bounds": {"x1min": 0., "x1max": ZTOP, "x2min": 0., "x2max": 1.6e6,
                                "x3min": 0., "x3max": 1.e3},
                     "cells": {"nx1": NX1, "nx2": NX2, "nx3": 1, "nghost": 3,
                               "x1rat": 4. ** (1. / (NX1 - 1))}},
        "distribute": {"layout": "slab", "nb2": 1, "nb3": 1, "verbose": False},
        "dynamics": {"equation-of-state": {"type": "ideal-moist", "density-floor": 1.e-10,
                                           "pressure-floor": 1.e-10, "limiter": True},
                     "reconstruct": {"vertical": weno, "horizontal": weno},
                     "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            **periodic}},
        "integration": {"type": "rk3", "cfl": 0.9, "implicit-scheme": SCHEME, "nlim": -1,
                        "tlim": 1.e9, "ncycle_out": 0},
        "forcing": {"const-gravity": {"grav1": -G}},
        "scalar": {"nvar": 1, "names": ["tracer"], "reconstruct": {"type": "weno5", "shock": False}},
    }


def step(courant, amp, p):  # one rk3 step from ratio R0 * p (tracer) and Q0 * p (vapour)
    with tempfile.TemporaryDirectory() as wd:
        f = os.path.join(wd, "deck.yaml")
        with open(f, "w") as fh:
            yaml.safe_dump(deck(), fh)
        block = MeshBlock(MeshBlockOptions.from_yaml(f))
    block.to(torch.device(DEV))
    x1f = block.module("coord").buffer("x1f").double().cpu()[3:3 + NX1 + 1]
    h, hs = x1f[1:] - x1f[:-1], RG * T0 / G
    pbar = 1.e5 * hs * (torch.exp(-x1f[:-1] / hs) - torch.exp(-x1f[1:] / hs)) / h  # exact cell means
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
    p = torch.cat([p[:1].repeat(3), p, p[-1:].repeat(3)]).to(w)
    w[kICY] = Q0 * p
    z = block.module("coord").buffer("x1v").double()
    bump = torch.zeros(w.shape[2], dtype=w.dtype, device=w.device)
    bump[3] = amp  # the first interior column only
    w[kIPR] *= 1. + bump.view(1, -1, 1) * torch.exp(-((z - 0.5 * ZTOP) / 2.e3) ** 2 / 2).view(1, 1, -1)
    v, _ = block.initialize({"hydro_w": w, "scalar_r": torch.zeros_like(w[:1]) + R0 * p})
    dt = courant * float(h.min()) / math.sqrt(1.4 * RG * T0)
    dev, lo, I = 0., float("inf"), (slice(None), slice(3, -3), slice(3, -3))
    for stage in range(3):
        block.forward(v, dt, stage)
        s, rho = v["scalar_s"][0][I], v["hydro_u"][kIDN][I]
        if p.min() > 0:
            dev = max(dev, float((s / rho / R0 - 1).abs().max()))
        lo = min(lo, float(s.min()))
    return dev, lo, block.check_redo(v)


if __name__ == "__main__":
    if DEV == "cuda" and not torch.cuda.is_available(): sys.exit(125)  # no visible GPU: skip
    bad = False
    for courant, amp in ((30., 0.55), (100., 0.18)):  # main 68f4e066 (= 49aed098): 4.401e-02, 7.055e-03
        dev, _, redo = step(courant, amp, torch.ones(NX1, dtype=torch.float64))
        print(f"{DEV} scheme {SCHEME} C={courant:g}: max|r/r0-1| = {dev:.3e}, redo = {redo}")
        bad |= dev > 1e-12 or redo != 0
    for name, hole in (("one-cell 22", False), ("one-hole 22", True)):
        p = torch.full((NX1,), 1. if hole else 0., dtype=torch.float64)
        p[22] = 0. if hole else 1.
        _, lo, _ = step(30., 0.55, p)
        print(f"{DEV} scheme {SCHEME} C=30 {name}: min s = {lo:.3e}")
        bad |= lo < -1e-12 * R0
    sys.exit(1 if bad else 0)
