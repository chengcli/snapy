#!/usr/bin/env python3
"""Global axial angular momentum (AAM) on the cubed sphere with the coriolis angular-momentum-fixer (#314).

Six gnomonic-equiangle panels in one process (Mesh API): a dry isothermal atmosphere on a shell,
constant gravity, closed x1 walls, rotation Omega about z (coriolis omega1). The AAM
  L = sum_cells [ sum_i aam_coef[i] m^i + Omega rho aam_inertia ]
is computed here from the GnomonicEquiangle buffers (m^i: the conserved covariant momentum rows
raised with cosine_cell_kj), independently of the fixer's own per-stage sum.
  (a) coefficients: aam_coef[0] (radial) is zero; r cos(lat) from the latitude buffer squares to
      aam_inertia / V; a solid-body rotation u_east = w r cos(lat) has L_rel = w sum rho aam_inertia;
  (b) a zonal jet with a wavenumber-2 perturbation over NSTEP rk3 steps:
        fixer off   |dL| / |L| > 100 TOL   (planted control: the defect the fixer removes is real)
        fixer on    |dL| / |L| <= TOL, and the fixer's total (buffers hydro.aamfix_total +
                    aamfix_pending, printed as fixaam= in the cycle diagnostics) is the control's
                    final dL undone, within a factor of two;
  (c) a discretely balanced rest state with the fixer on stays at rest (max |v| / c_s <= REST_TOL)
      and keeps L to TOL;
  (e) an open top (x1-outer: outflow): the discrete balance of (b) does not hold against the
      outflow ghost, so mass falls in through the top face (about 1.5% of L in 20 steps). The AAM
      it carries is physical and must stay. B, the stage-weighted sum of
      dt A_f [r_f (n x b_i)_z F^i + Omega r_f^2 cos^2(lat) F_mass] over the physical x1 faces, is
      computed here from the face flux buffer hydro.F1 after each stage:
        fixer on    |dL - B| / |L| <= TOL, with |B| / |L| > 1e6 TOL (the transport is real)
        fixer off   |dL - B| / |L| > 100 TOL;
  (d) the key is refused on a Cartesian grid and with omega2 != 0, with clear messages, and is
      reachable from Python.
Each arm builds its own Mesh and frees it before the next: the local exchange of six panels in
one process keeps state per process, and a second live Mesh can wait on the first forever.

  python test_angular_momentum_fixer.py [--device cuda] [--nstep 20]
"""
import argparse
import gc
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

TOL = 1.0e-12
REST_TOL = 1.0e-6
OMEGA = 7.0e-5
W_SB = 5.0e-6  # solid-body rate of the test wind, u_east ~ 30 m/s
U_PERT = 10.0  # wavenumber-2 perturbation amplitude (m/s)


def build(cfg, device):
    from snapy import Mesh, MeshOptions

    nx = cfg["geometry"]["cells"]["nx2"]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        options = MeshOptions.from_yaml(tmp)
        options.blocks_per_process(6)
        options.set_local_horizontal_cells(nx, nx)
        mesh = Mesh(options)
    finally:
        os.unlink(tmp)
    blocks = list(mesh.blocks)
    for block in blocks:
        block.to(torch.device(device))
    return mesh, blocks


def interior(block):
    return tuple(block.part((0, 0, 0), False)[1:])


def aam(blocks, mesh_vars, omega):
    """Global AAM from the coordinate buffers, double precision, interior cells."""
    from snapy import kIDN, kIV1, kIV2, kIV3

    total = 0.0
    for ib, block in enumerate(blocks):
        coord = block.module("coord")
        u = mesh_vars[ib]["hydro_u"].double()
        c = coord.buffer("cosine_cell_kj").double()
        s2 = 1.0 - c * c
        m2, m3 = u[kIV2], u[kIV3]  # covariant rows -> contravariant
        v2 = (m2 - c * m3) / s2
        v3 = (m3 - c * m2) / s2
        coef = coord.buffer("aam_coef").double()
        inertia = coord.buffer("aam_inertia").double()
        l = coef[0] * u[kIV1] + coef[1] * v2 + coef[2] * v3 + omega * u[kIDN] * inertia
        total += l[interior(block)].sum().item()
    return total


def inertia_sum(blocks, mesh_vars):
    from snapy import kIDN

    total = 0.0
    for ib, block in enumerate(blocks):
        inertia = block.module("coord").buffer("aam_inertia").double()
        u = mesh_vars[ib]["hydro_u"].double()
        total += (u[kIDN] * inertia)[interior(block)].sum().item()
    return total


def initial_state(cfg, blocks, wind):
    """Isothermal rest column in the scheme's discrete balance, plus an east wind u_east(lon, lat)."""
    import snapy
    from snapy import kIDN, kIPR, kIV2, kIV3

    g = -cfg["forcing"]["const-gravity"]["grav1"]
    p0, rho0 = 1.0e5, 0.1
    ng, nx1 = cfg["geometry"]["cells"]["nghost"], cfg["geometry"]["cells"]["nx1"]
    mesh_vars = []
    for block in blocks:
        coord = block.module("coord")
        r = coord.buffer("x1v")[ng:ng + nx1].cpu()
        dr = coord.buffer("dx1f")[ng:ng + nx1].cpu()
        p = p0 * torch.exp(-(r - r[0]) * rho0 * g / p0)
        col = torch.zeros(kIPR + 1, 1, 1, nx1, dtype=torch.float64)
        col[kIDN, 0, 0] = p * rho0 / p0
        col[kIPR, 0, 0] = p
        wb, _, _ = snapy.balance_column(col, dr.contiguous(), g)
        w = dict(block.named_buffers())["hydro.D"].clone().zero_()
        for c in (kIDN, kIPR):
            w[c][..., ng:ng + nx1] = wb[c, 0, 0].to(w)
            w[c][..., :ng] = w[c][..., ng:ng + 1]
            w[c][..., ng + nx1:] = w[c][..., ng + nx1 - 1:ng + nx1]
        if wind is not None:
            lon = coord.buffer("longitude").to(w)
            lat = coord.buffer("latitude").to(w)
            rc = coord.buffer("x1v").to(w) * torch.cos(lat)  # r cos(lat), [nc3, nc2, nc1]
            east = coord.buffer("aam_east").to(w) / rc  # unit east vector, contravariant
            u_east = wind(lon, lat, rc)
            w[kIV2] = u_east * east[1]
            w[kIV3] = u_east * east[2]
        mesh_vars.append({"hydro_w": w})
    return mesh_vars


def boundary_gain(cfg, blocks, dt):
    """AAM the last stage's x1 face flux carries in through the physical x1 faces (inner in, outer
    out), from hydro.F1 and the coordinate buffers."""
    from snapy import kIDN, kIV1, kIV2, kIV3

    ng, nx1 = cfg["geometry"]["cells"]["nghost"], cfg["geometry"]["cells"]["nx1"]
    bc = cfg["boundary-condition"]["external"]
    faces = [(f, sign) for f, sign, key in ((ng, 1.0, "x1-inner"), (ng + nx1, -1.0, "x1-outer"))
             if bc[key] != "periodic"]
    total = 0.0
    for block in blocks:
        coord = block.module("coord")
        F = dict(block.named_buffers())["hydro.F1"].double()
        c = coord.buffer("cosine_cell_kj").double()[..., 0]
        coef = coord.buffer("aam_coef").double()
        inertia = coord.buffer("aam_inertia").double()
        vol = coord.cell_volume().double()
        area = coord.face_area1().double()
        x1v, x1f = coord.buffer("x1v").double(), coord.buffer("x1f").double()
        k, j, _ = interior(block)
        for face, sign in faces:
            cell = face if sign > 0 else face - 1
            m2, m3 = F[kIV2][..., face], F[kIV3][..., face]
            v2 = (m2 - c * m3) / (1.0 - c * c)
            v3 = (m3 - c * m2) / (1.0 - c * c)
            rc, rf, v = x1v[cell], x1f[face], vol[..., cell]
            ang = coef[..., cell] / (rc * v)
            cos2 = inertia[..., cell] / (rc * rc * v)
            fm = F[kIDN][..., face]
            g = area[..., face] * (rf * (ang[0] * F[kIV1][..., face] + ang[1] * v2 + ang[2] * v3)
                                   + OMEGA * rf * rf * cos2 * fm)
            total += sign * dt * g[k, j].sum().item()
    return total


def run_open(cfg, mesh, blocks, mesh_vars, nstep):
    """Run nstep steps; return L0, the final dL and B, the stage-weighted boundary gain."""
    mesh_vars, _ = mesh.initialize(mesh_vars)
    stages = blocks[0].module("intg").stages
    weight = []
    for s in range(len(stages)):
        cw = stages[s].wght2()
        for t in range(s + 1, len(stages)):
            cw *= stages[t].wght1()
        weight.append(cw)
    l0 = aam(blocks, mesh_vars, OMEGA)
    gain = 0.0
    for cycle in range(1, nstep + 1):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        for stage in range(len(stages)):
            mesh.forward(mesh_vars, dt, stage)
            gain += weight[stage] * boundary_gain(cfg, blocks, dt)
        assert mesh.check_redo(mesh_vars) == 0, "step rejected at cycle %d" % cycle
    return l0, aam(blocks, mesh_vars, OMEGA) - l0, gain


def run(cfg, mesh, blocks, mesh_vars, nstep):
    mesh_vars, _ = mesh.initialize(mesh_vars)
    intg = blocks[0].module("intg")
    l0 = aam(blocks, mesh_vars, OMEGA)
    drift = 0.0
    for cycle in range(1, nstep + 1):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
        assert mesh.check_redo(mesh_vars) == 0, "step rejected at cycle %d" % cycle
        d = abs(aam(blocks, mesh_vars, OMEGA) - l0) / abs(l0)
        drift = max(drift, d) if math.isfinite(d) else math.inf
    buf = dict(blocks[0].named_buffers())
    fix = (buf["hydro.aamfix_total"] + buf["hydro.aamfix_pending"]).item()
    final = (aam(blocks, mesh_vars, OMEGA) - l0) / l0  # signed
    return mesh_vars, l0, drift, fix, final


def vmax_over_cs(blocks, mesh_vars):
    from snapy import kIDN, kIPR

    vmax, cs = 0.0, 0.0
    for ib, block in enumerate(blocks):
        sl = interior(block)
        u = mesh_vars[ib]["hydro_u"]
        v = u[1:4][(slice(None),) + sl] / u[kIDN][sl]
        vmax = max(vmax, v.abs().max().item())
        cs = max(cs, math.sqrt(1.4 * (u[kIPR][sl] / u[kIDN][sl]).max().item()))
    return vmax / cs


def options_from(cfg):
    from snapy import MeshBlockOptions

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        return MeshBlockOptions.from_yaml(tmp)
    finally:
        os.unlink(tmp)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--nstep", type=int, default=20)
    ap.add_argument("--yaml", default=str(Path(__file__).resolve().parent / "test_angular_momentum_fixer.yaml"))
    args = ap.parse_args(argv)
    if args.device == "cuda" and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 125
    torch.set_default_dtype(torch.float64)

    with open(args.yaml) as f:
        base = yaml.safe_load(f)
    failures = []

    def config(fixer):
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["forcing"]["coriolis"]["angular-momentum-fixer"] = fixer
        return cfg

    def solid_body(lon, lat, rc):
        return W_SB * rc

    def jet(lon, lat, rc):
        return W_SB * rc + U_PERT * torch.cos(lat) ** 2 * torch.sin(2.0 * lon)

    # (a) the coefficients on the initial solid-body state
    cfg = config(False)
    mesh, blocks = build(cfg, args.device)
    coef_err, rc_err = 0.0, 0.0
    for block in blocks:
        coord = block.module("coord")
        coef = coord.buffer("aam_coef").double()
        coef_err = max(coef_err, (coef[0].abs().max() / coef.abs().max()).item())
        rc = coord.buffer("x1v").double() * torch.cos(coord.buffer("latitude").double())
        rc2 = coord.buffer("aam_inertia").double() / coord.cell_volume().double()
        rc_err = max(rc_err, ((rc * rc - rc2).abs().max() / rc2.max()).item())
    mesh_vars, _ = mesh.initialize(initial_state(cfg, blocks, solid_body))
    l_rel = aam(blocks, mesh_vars, 0.0)
    l_sb = W_SB * inertia_sum(blocks, mesh_vars)
    sb_err = abs(l_rel / l_sb - 1.0)
    print("(a) |aam_coef[0]| / max |aam_coef| = %.3e, r cos(lat) vs aam_inertia: %.3e, "
          "solid body L_rel / (w I) - 1 = %.3e" % (coef_err, rc_err, sb_err))
    if not coef_err < TOL:
        failures.append("the radial momentum carries AAM: %g" % coef_err)
    if not rc_err < TOL:
        failures.append("latitude buffer and aam_inertia disagree on r cos(lat): %g" % rc_err)
    if not sb_err < TOL:
        failures.append("solid-body L_rel != w sum rho aam_inertia: %g" % sb_err)
    del mesh, blocks, mesh_vars
    gc.collect()

    # (b) the jet, fixer off and on
    finals = {}
    for fixer in (False, True):
        cfg = config(fixer)
        mesh, blocks = build(cfg, args.device)
        _, l0, drift, fix, final = run(cfg, mesh, blocks, initial_state(cfg, blocks, jet), args.nstep)
        del mesh, blocks
        gc.collect()
        finals[fixer] = final
        print("(b) fixer %-5s max |dL| / |L| over %d steps = %.3e (final dL / L = %+.3e)   fixaam = %.3e (%+.3e of L)"
              % (fixer, args.nstep, drift, final, fix, fix / l0))
        if not math.isfinite(drift):
            failures.append("fixer %s: L is not finite" % fixer)
        elif fixer and not drift <= TOL:
            failures.append("fixer on: L drifts %.3e > %.0e" % (drift, TOL))
        elif not fixer and not drift > 100 * TOL:
            failures.append("fixer off: drift %.3e <= %.0e, the fixer would not be load-bearing" % (drift, 100 * TOL))
        if fixer:
            # the fixer's total is the control's accumulated defect undone, to the O(drift^2)
            # feedback of the corrected trajectory on the defect
            off = finals[False]
            if not (abs(off) > 0.0 and 0.5 <= -fix / l0 / off <= 2.0):
                failures.append("fixer on: fixaam / L = %+.3e is not the control's dL / L = %+.3e undone" % (fix / l0, off))

    # (c) rest with the fixer on
    cfg = config(True)
    mesh, blocks = build(cfg, args.device)
    mesh_vars, l0, drift, fix, _ = run(cfg, mesh, blocks, initial_state(cfg, blocks, None), args.nstep)
    v = vmax_over_cs(blocks, mesh_vars)
    del mesh, blocks, mesh_vars
    gc.collect()
    print("(c) rest + fixer: max |v| / c_s after %d steps = %.3e, max |dL| / |L| = %.3e, fixaam / L = %.3e"
          % (args.nstep, v, drift, fix / l0))
    if not v < REST_TOL:
        failures.append("rest state moved with the fixer on: max |v| / c_s = %g" % v)
    if not drift <= TOL:
        failures.append("rest + fixer: L drifts %.3e > %.0e" % (drift, TOL))

    # (e) an open top: the AAM crossing it is physical and stays out of the defect
    for fixer in (False, True):
        cfg = config(fixer)
        cfg["boundary-condition"]["external"]["x1-outer"] = "outflow"
        mesh, blocks = build(cfg, args.device)
        state = initial_state(cfg, blocks, jet)
        l0, dl, gain = run_open(cfg, mesh, blocks, state, args.nstep)
        del mesh, blocks, state
        gc.collect()
        miss = abs(dl - gain) / abs(l0)
        print("(e) open top, fixer %-5s dL / L = %+.6e, B / L = %+.6e, |dL - B| / |L| = %.3e"
              % (fixer, dl / l0, gain / l0, miss))
        if not (math.isfinite(dl) and math.isfinite(gain)):
            failures.append("open top, fixer %s: L or B is not finite" % fixer)
        elif not abs(gain / l0) > 1.0e6 * TOL:
            failures.append("open top: boundary transport %.3e of L is too small to test" % (gain / l0))
        elif fixer and not miss <= TOL:
            failures.append("open top + fixer: dL - B = %.3e of L, the fixer removes physical "
                            "boundary transport" % miss)
        elif not fixer and not miss > 100 * TOL:
            failures.append("open top, fixer off: |dL - B| = %.3e of L, no defect to fix" % miss)

    # (d) refusals and the Python option
    from snapy import MeshBlock

    rgas = 8.31446261815324
    cart = {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": 1.0e4, "x2min": 0.0, "x2max": 1.0e4,
                                "x3min": 0.0, "x3max": 1.0e4},
                     "cells": {"nx1": 4, "nx2": 4, "nx3": 4, "nghost": 3}},
        "dynamics": {"equation-of-state": {"type": "ideal-gas", "gammad": 1.4, "weight": rgas / 287.0},
                     "reconstruct": {"vertical": {"type": "weno5"}, "horizontal": {"type": "weno5"}},
                     "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.4, "nlim": -1, "tlim": 1.0e9},
        "forcing": {"coriolis": {"type": "xyz", "omega1": OMEGA, "angular-momentum-fixer": True}},
    }
    try:
        MeshBlock(options_from(cart))
        failures.append("the fixer accepted a Cartesian grid")
    except RuntimeError as e:
        ok = "gnomonic-equiangle" in str(e)
        print("(d) cartesian + fixer: refused (%s)" % ("expected message" if ok else "unexpected message"))
        if not ok:
            failures.append("cartesian + fixer: unexpected error: %s" % str(e).splitlines()[0])
    cfg = config(True)
    cfg["forcing"]["coriolis"]["omega2"] = 1.0e-5
    try:
        options_from(cfg)
        failures.append("the fixer accepted omega2 != 0")
    except RuntimeError as e:
        ok = "omega2 = omega3 = 0" in str(e)
        print("(d) omega2 != 0 + fixer: refused (%s)" % ("expected message" if ok else "unexpected message"))
        if not ok:
            failures.append("omega2 + fixer: unexpected error: %s" % str(e).splitlines()[0])
    for fixer in (False, True):
        c = options_from(config(fixer)).hydro().coriolis()
        if c.angular_momentum_fixer() is not fixer:
            failures.append("python options do not read angular-momentum-fixer: %s" % fixer)
    print("(d) python options: angular_momentum_fixer reads back")

    for msg in failures:
        print("FAIL:", msg)
    if failures:
        return 1
    print("### angular momentum fixer test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
