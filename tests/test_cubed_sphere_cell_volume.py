#!/usr/bin/env python3
"""Exact cubed-sphere cell volumes and radial face areas (#289).

Six panels in one process (Mesh API) on a shell ri <= r <= ro:
  (a) the cell volumes of the six panels sum to 4 pi (ro^3 - ri^3) / 3;
  (b) the discrete divergence of F = r rhat, (A1_{i+1} r_{i+1} - A1_i r_i) / V (the x2/x3 faces
      carry no flux of a radial field), is 3 in every cell. Both hold to round-off only when
      cell_volume() is the exact radial integral (rp^3 - rm^3)/3 times the same solid angle as
      face_area1(), and that solid angle is the exact one of the gnomonic cell. The trapezoid
      0.5 (A1_i + A1_{i+1}) dr misses (b) by O((dr/r)^2);
  (c) a discretely balanced isothermal rest state stays at rest over a short run (reported).

  python test_cubed_sphere_cell_volume.py [--device cuda] [--yaml PATH]
"""
import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROUNDOFF = 1e-12
REST_TOL = 1e-6  # max |v| / c_s after the short rest run


def build(yaml_file: str, device: str):
    from snapy import Mesh, MeshOptions

    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)
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
    return cfg, mesh, blocks


def geometry(cfg, blocks):
    ng, nx1 = cfg["geometry"]["cells"]["nghost"], cfg["geometry"]["cells"]["nx1"]
    total, err = 0.0, 0.0
    for block in blocks:
        coord = block.module("coord")
        k, j, _ = block.part((0, 0, 0), False)[1:]
        i = slice(ng, ng + nx1)
        r = coord.buffer("x1f")[ng:ng + nx1 + 1]
        vol = coord.cell_volume()[k, j, i]
        area = coord.face_area1()[k, j, ng:ng + nx1 + 1]
        total += vol.sum().item()
        div = (area[..., 1:] * r[1:] - area[..., :-1] * r[:-1]) / vol
        err = max(err, (div - 3.0).abs().max().item())
    return total, err


def rest(cfg, mesh, blocks):
    from snapy import kIDN, kIPR
    import snapy

    g = -cfg["forcing"]["const-gravity"]["grav1"]
    p0, rho0 = 1.0e5, 0.1  # isothermal, scale height p0 / (rho0 g)
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
        mesh_vars.append({"hydro_w": w})
    mesh_vars, _ = mesh.initialize(mesh_vars)
    intg = blocks[0].module("intg")
    for cycle in range(1, cfg["integration"]["nlim"] + 1):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
        assert mesh.check_redo(mesh_vars) == 0, "step rejected at cycle %d" % cycle
    vmax, cs = 0.0, math.sqrt(1.4 * p0 / rho0)
    for ib, block in enumerate(blocks):
        sl = block.part((0, 0, 0), False)[1:]
        u = mesh_vars[ib]["hydro_u"]  # end-of-step state (hydro_w is a stage input)
        v = u[1:4][(slice(None),) + tuple(sl)] / u[kIDN][sl]
        vmax = max(vmax, v.abs().max().item())
    return vmax / cs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument(
        "--yaml",
        default=str(Path(__file__).resolve().parent / "test_cubed_sphere_cell_volume.yaml"),
    )
    args = ap.parse_args(argv)
    if args.device == "cuda" and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 125

    cfg, mesh, blocks = build(args.yaml, args.device)
    b = cfg["geometry"]["bounds"]
    ri, ro = float(b["x1min"]), float(b["x1max"])
    exact = 4.0 * math.pi * (ro**3 - ri**3) / 3.0
    total, div_err = geometry(cfg, blocks)
    vol_err = abs(total / exact - 1.0)
    vmax = rest(cfg, mesh, blocks)
    print("(a) sum V / (4 pi (ro^3 - ri^3) / 3) - 1 = %.3e" % (total / exact - 1.0))
    print("(b) max |div(r rhat) - 3| = %.3e" % div_err)
    print("(c) rest: max |v| / c_s after %d cycles = %.3e" % (cfg["integration"]["nlim"], vmax))

    failures = []
    if not vol_err < ROUNDOFF:
        failures.append("six-panel volume is not 4 pi (ro^3 - ri^3) / 3: %g" % vol_err)
    if not div_err < ROUNDOFF:
        failures.append("div(r rhat) != 3: %g -- cell_volume() and face_area1() disagree" % div_err)
    if not vmax < REST_TOL:
        failures.append("rest state moved: max |v| / c_s = %g" % vmax)
    for msg in failures:
        print("FAIL:", msg)
    if failures:
        sys.exit(1)
    print("### cubed-sphere cell volume test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
