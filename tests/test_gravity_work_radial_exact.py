#!/usr/bin/env python3
"""Corrected-PE gravity work (switch SNAP_GRAVITY_WORK_RADIAL_EXACT; derivation
docs/derivations/curved-gravity-work-weight.md, option F).

With gravity-work: face the switch adds g1 sigma^2 s[drho] to each cell's x1 gravity work,
sigma^2 = <(x1 - x1v)^2> over the cell (h^2/12 on a Cartesian grid), s = the slope at x1v of
the quadratic through the cell and its two x1 neighbours (one-sided at the block's x1 ends),
so that E + P is conserved, where
  P = sum V [rho phi(x1v) - g1 sigma^2 s[rho]],  phi = -g1 x1,
is the exact potential energy to O(dx^4). E + PE_d (PE_d = sum V rho phi(x1v)) is not
conserved with the switch on; its drift is printed. Isentropic column (p0 100, rho0 1, R 1,
g 1, depth 100) between closed (reflecting) x1 walls, seeded u1 = 0.05 c_s sin(pi z / L)
(2-D: times cos(2 pi x2 / L)), weno5, lmars, rk3. Checked:
  1. switch unset and 0 give the same states bit for bit; on differs from off;
  2. on: |d(E + P)| / |E + P| per step <= 1e-14 on a spherical-polar column (x1 in [300, 400])
     and a Cartesian column, explicit and VIC (implicit-scheme 9), and on a 2-D Cartesian
     box with nx3 = 1, explicit and VIC;
  3. on, one explicit plm stage (plm has no curvature flux): the energy change on - off is
     g1 sigma^2 s[drho] cell by cell (option F's eq. 7), to round-off of the energy;
  4. rest: a discretely balanced Cartesian column (snapy.balance_column), unseeded, keeps
     max|u1|/c_s with the switch on no worse than off, explicit and VIC;
  5. the cycle diagnostics (print_cycle_info) log the potential energy the booked work conserves:
     ie= + pe= equals this test's E + P with the switch on and E + PE_d with it off, to the printed digits
     (spherical, Cartesian and 2-D Cartesian, after a few explicit steps).
The switch is read once per process, so each arm runs in a child process.

  python test_gravity_work_radial_exact.py [--device cpu]
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

import torch
import yaml

GAMMA, P0, RHO0, G, DEPTH, NZ, NG = 1.4, 100., 1., 1., 100., 32, 3
NSTEP, EP_TOL, REST_STEPS = 20, 1.e-14, 50
DIAG_STEPS, DIAG_TOL, DIAG_CASES = 5, 1.e-11, ("sph", "cart", "cart2d")
ARMS = {"unset": None, "zero": "0", "on": "1"}
CASES = {  # name: (geometry, implicit scheme, nx2)
    "sph": ("spherical-polar", 0, 1), "sph_vic": ("spherical-polar", 9, 1),
    "cart": ("cartesian", 0, 1), "cart_vic": ("cartesian", 9, 1),
    "cart2d": ("cartesian", 0, 16), "cart2d_vic": ("cartesian", 9, 16)}


def config(geometry, scheme, nx2, recon, ncycle_out=0):
    if geometry == "spherical-polar":
        x1min, x2 = 300., (0.5 * math.pi - 0.05, 0.5 * math.pi + 0.05)
        bounds = {"x1min": x1min, "x1max": x1min + DEPTH, "x2min": x2[0], "x2max": x2[1],
                  "x3min": 0.0, "x3max": 0.1}
        x2bc = "reflecting"
    else:
        bounds = {"x1min": 0.0, "x1max": DEPTH, "x2min": 0.0, "x2max": DEPTH, "x3min": 0.0, "x3max": 1.0}
        x2bc = "periodic"
    return {
        "geometry": {"type": geometry, "bounds": bounds,
                     "cells": {"nx1": NZ, "nx2": nx2, "nx3": 1, "nghost": NG}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": GAMMA, "weight": 8.31446,
                                  "density-floor": 1.e-12, "pressure-floor": 1.e-12,
                                  "temperature-floor": 1.e-12, "limiter": True},
            "reconstruct": {"vertical": {"type": recon, "scale": False, "shock": False},
                            "horizontal": {"type": recon, "scale": False, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": x2bc, "x2-outer": x2bc,
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": scheme, "nlim": -1, "tlim": 1.e9,
                        "ncycle_out": ncycle_out},
        "forcing": {"const-gravity": {"grav1": -G, "gravity-work": "face"}},
    }


def interior(shape):
    return tuple(slice(NG, -NG) if n > 1 else slice(None) for n in shape)


def build(case, seed=0.05, recon="weno5", balanced=False, device="cpu", ncycle_out=0):
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1
    geometry, scheme, nx2 = CASES[case]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(geometry, scheme, nx2, recon, ncycle_out), f)
        tmp = f.name
    try:
        b = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    b.to(torch.device(device), torch.float64)
    x1f, x1v, x2v = (b.buffer("coord." + k).cpu() for k in ("x1f", "x1v", "x2v"))
    w = b.buffer("hydro.D").clone().zero_().cpu()
    K, ex = P0 / RHO0 ** GAMMA, (GAMMA - 1.) / GAMMA
    z = x1v - float(x1f[NG])
    p = (P0 ** ex - ex * G * z / K ** (1. / GAMMA)).clamp(min=1.e-3) ** (1. / ex)
    w[kIDN], w[kIPR] = (p / K) ** (1. / GAMMA), p
    if balanced:  # Cartesian columns only
        I = slice(NG, NG + NZ)
        col = w[:kIPR + 1, ..., I].clone()
        dx1f = (x1f[1:] - x1f[:-1])[I].contiguous()
        wb, _, _ = snapy.balance_column(col.contiguous(), dx1f, G, True, 3e-14, 400)
        w[:kIPR + 1, ..., I] = wb
        for c in (kIDN, kIPR):
            w[c][..., :NG] = w[c][..., NG:NG + 1]
            w[c][..., NG + NZ:] = w[c][..., NG + NZ - 1:NG + NZ]
    u1 = seed * math.sqrt(GAMMA * P0 / RHO0) * torch.sin(math.pi * z / DEPTH)
    u1 = torch.where((z > 0) & (z < DEPTH), u1, torch.zeros_like(u1))
    if nx2 > 1:
        u1 = u1 * torch.cos(2. * math.pi * x2v / DEPTH)[:, None]
    w[kIV1] = u1.expand_as(w[kIV1])
    v, _ = b.initialize({"hydro_w": w.to(device)})
    return b, v


def slope(q, x):
    """s[q] of gravity_work_radial.hpp along the last axis (x = interior x1v)."""
    s = torch.zeros_like(q)
    hm, hp = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    s[..., 1:-1] = (-hp / (hm * (hm + hp)) * q[..., :-2] + (hp - hm) / (hm * hp) * q[..., 1:-1]
                    + hm / (hp * (hm + hp)) * q[..., 2:])
    a, c = x[1] - x[0], x[2] - x[1]
    s[..., 0] = -(2 * a + c) / (a * (a + c)) * q[..., 0] + (a + c) / (a * c) * q[..., 1] - a / ((a + c) * c) * q[..., 2]
    a, c = x[-2] - x[-3], x[-1] - x[-2]
    s[..., -1] = c / (a * (a + c)) * q[..., -3] - (a + c) / (a * c) * q[..., -2] + (a + 2 * c) / ((a + c) * c) * q[..., -1]
    return s


def variance(x1f, spherical):
    """sigma^2 about x1v, written about the cell midpoint (no cancellation at large x1 / dx)."""
    h, rb = x1f[1:] - x1f[:-1], 0.5 * (x1f[1:] + x1f[:-1])
    if not spherical:
        return h * h / 12.
    vol = rb * rb * h + h ** 3 / 12.
    return (rb * rb * h ** 3 / 12. + h ** 5 / 80.) / vol - (rb * h ** 3 / (6. * vol)) ** 2


class Column:
    def __init__(self, b, case):
        sph = CASES[case][0] == "spherical-polar"
        u = b.buffer("hydro.D")
        self.sl = (slice(None),) + interior(u.shape[1:])
        f1, f2, f3 = (b.buffer("coord." + k).cpu() for k in ("x1f", "x2f", "x3f"))
        f2, f3 = (f[NG:-NG] if f.numel() > 2 else f for f in (f2, f3))
        x1f = f1[NG:NG + NZ + 1]
        if sph:  # (r+^3 - r-^3)/3 (cos th- - cos th+) dphi
            rad, lat = (x1f[1:] ** 3 - x1f[:-1] ** 3) / 3., f2[:-1].cos() - f2[1:].cos()
        else:
            rad, lat = x1f[1:] - x1f[:-1], f2[1:] - f2[:-1]
        self.vol = (f3[1:] - f3[:-1])[:, None, None] * lat[None, :, None] * rad[None, None, :]
        self.x = b.buffer("coord.x1v").cpu()[NG:NG + NZ]
        self.var = variance(x1f, sph)

    def energies(self, v):
        """(E + PE_d, E + P) of the interior"""
        from snapy import kIDN, kIPR
        u = v["hydro_u"][self.sl].cpu()
        E = (u[kIPR] * self.vol).sum()
        ped = (u[kIDN] * G * self.x * self.vol).sum()
        return float(E + ped), float(E + ped + G * (self.var * slope(u[kIDN], self.x) * self.vol).sum())


def run(case, steps, seed=0.05, balanced=False, device="cpu"):
    from snapy import kIDN, kIPR, kIV1
    b, v = build(case, seed=seed, balanced=balanced, device=device)
    col = Column(b, case)
    dz = DEPTH / NZ
    dt = (0.3 if CASES[case][1] == 0 else 1.2) * dz / math.sqrt(GAMMA * P0 / RHO0)
    epd0, ep0 = col.energies(v)
    dEP = dEPd = umax = 0.
    for _ in range(steps):
        b.inc_cycle()
        for st in range(len(b.module("intg").stages)):
            b.forward(v, dt, st)
        assert b.check_redo(v) == 0
        epd, ep = col.energies(v)
        dEP, dEPd = max(dEP, abs(ep - ep0) / abs(ep0)), max(dEPd, abs(epd - epd0) / abs(epd0))
        epd0, ep0 = epd, ep
        w = b.module("hydro.eos").compute("U->W", [v["hydro_u"]])[col.sl].cpu()
        umax = max(umax, float((w[kIV1].abs() / torch.sqrt(GAMMA * w[kIPR] / w[kIDN])).max()))
    return {"dEP": dEP, "dEPd": dEPd, "umax": umax}, v["hydro_u"].cpu().clone()


def stage(case, device="cpu"):
    """one explicit plm stage: interior u before and after"""
    b, v = build(case, recon="plm", device=device)
    col = Column(b, case)
    u0 = v["hydro_u"][col.sl].cpu().clone()
    b.forward(v, 0.3 * DEPTH / NZ / math.sqrt(GAMMA * P0 / RHO0), 0)
    return {"u0": u0, "u1": v["hydro_u"][col.sl].cpu().clone(), "var": col.var, "x": col.x}


def diag(case, out, device="cpu"):
    """a few explicit steps, then the cycle diagnostics on stdout; this test's (E + PE_d, E + P) to out"""
    b, v = build(case, device=device, ncycle_out=1)
    col = Column(b, case)
    dt = 0.3 * DEPTH / NZ / math.sqrt(GAMMA * P0 / RHO0)
    for _ in range(DIAG_STEPS):
        b.inc_cycle()
        for st in range(len(b.module("intg").stages)):
            b.forward(v, dt, st)
    b.print_cycle_info(v, 0., dt)
    json.dump(col.energies(v), open(os.path.join(out, "diag_%s.json" % case), "w"))


def child(out, device):
    res, saved = {}, {}
    for case in CASES:
        res[case], saved[case] = run(case, NSTEP, device=device)
    for case in ("cart", "cart_vic"):
        res["rest_" + case] = run(case, REST_STEPS, seed=0., balanced=True, device=device)[0]["umax"]
    for case in ("sph", "cart"):
        saved["stage_" + case] = stage(case, device)
    json.dump(res, open(os.path.join(out, "res.json"), "w"))
    torch.save(saved, os.path.join(out, "u.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--child", default=None)
    ap.add_argument("--diag", default=None)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    if a.diag:
        diag(a.diag, a.child, a.device)
        return
    if a.child:
        child(a.child, a.device)
        return
    from snapy import kIDN, kIPR
    failures, res, u = [], {}, {}
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
        for arm, value in ARMS.items():
            env = dict(os.environ)
            env.pop("SNAP_GRAVITY_WORK_RADIAL_EXACT", None)
            if value is not None:
                env["SNAP_GRAVITY_WORK_RADIAL_EXACT"] = value
            out = os.path.join(tmp, arm)
            os.makedirs(out)
            subprocess.run([sys.executable, os.path.abspath(__file__), "--device", a.device,
                            "--child", out], env=env, check=True)
            res[arm] = json.load(open(os.path.join(out, "res.json")))
            u[arm] = torch.load(os.path.join(out, "u.pt"))
            if arm == "zero":
                continue
            for case in DIAG_CASES:  # one process each: the log is flushed at exit
                log = subprocess.run([sys.executable, os.path.abspath(__file__), "--device", a.device,
                                      "--diag", case, "--child", out], env=env, check=True,
                                     capture_output=True, text=True).stdout
                ie, pe = (float(re.search(r" %s=(\S+)" % k, log).group(1)) for k in ("ie", "pe"))
                epd, ep = json.load(open(os.path.join(out, "diag_%s.json" % case)))
                want, name = (ep, "E + P") if arm == "on" else (epd, "E + PE_d")
                err = abs(ie + pe - want) / abs(want)
                print(f"{case:10s} switch {arm:5s}: logged ie + pe vs {name}: rel diff {err:.2e} "
                      f"(E + P vs E + PE_d: {abs(ep - epd) / abs(ep):.2e})", flush=True)
                if not err <= DIAG_TOL:
                    failures.append(f"{case}: switch {arm}, logged ie + pe is not {name} (rel diff {err:.2e})")
    for case in CASES:
        off, on = res["unset"][case], res["on"][case]
        print(f"{case:10s} max per-step |d(E+P)|/|E+P|: off {off['dEP']:.2e} on {on['dEP']:.2e}   "
              f"|d(E+PE_d)|/|E+PE_d|: off {off['dEPd']:.2e} on {on['dEPd']:.2e}", flush=True)
        if not torch.equal(u["unset"][case], u["zero"][case]):
            failures.append(f"{case}: switch unset and 0 differ")
        if torch.equal(u["unset"][case], u["on"][case]):
            failures.append(f"{case}: the switch changed nothing")
        if not on["dEP"] <= EP_TOL:
            failures.append(f"{case}: on, per-step E+P change {on['dEP']:.2e} > {EP_TOL}")
    for case in ("cart", "cart_vic"):
        r_off, r_on = res["unset"]["rest_" + case], res["on"]["rest_" + case]
        print(f"{case:10s} rest, {REST_STEPS} steps, max|u1|/c_s: off {r_off:.3e} on {r_on:.3e}", flush=True)
        if not r_on <= 1.05 * r_off + 1.e-14:
            failures.append(f"{case}: rest max|u1|/c_s on {r_on:.3e} worse than off {r_off:.3e}")
    for case in ("sph", "cart"):
        off, on = u["unset"]["stage_" + case], u["on"]["stage_" + case]
        if not torch.equal(on["u1"][kIDN], off["u1"][kIDN]):
            failures.append(f"{case}: the switch changed the stage's density")
        dE = on["u1"][kIPR] - off["u1"][kIPR]
        pred = -G * on["var"] * slope(on["u1"][kIDN] - on["u0"][kIDN], on["x"])
        err = float((dE - pred).abs().max())
        scale = float(on["u1"][kIPR].abs().max())
        print(f"{case:10s} one plm stage: max|dE(on - off) - g1 sigma^2 s[drho]| = {err:.2e}, "
              f"max|g1 sigma^2 s[drho]| = {float(pred.abs().max()):.2e}, max|E| = {scale:.2e}", flush=True)
        if not (err <= 1.e-13 * scale and float(pred.abs().max()) >= 1.e3 * err):
            failures.append(f"{case}: one-stage energy change is not g1 sigma^2 s[drho] (err {err:.2e})")
    for failure in failures:
        print("FAIL", failure)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
