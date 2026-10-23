#!/usr/bin/env python3
"""An outflow x1 boundary under an implicit scheme at large time steps.

Six cubed-sphere panels in one process (Mesh API), the shell of test_cubed_sphere_cell_volume.yaml
with x1-outer: outflow, a discretely balanced isothermal rest column whose top ghosts continue the
hydrostatic profile, and the adaptive time step of an implicit scheme (vertical acoustic Courant
number about 9). Constant gravity adds dt rho g to every cell of du, ghost cells included, so after
the RK update the ghost cells that the exchange has not yet refilled hold a state whose kinetic
energy exceeds its total energy. The outflow boundary must not read them as the gas state at the
boundary: it once checked the sound speed over the whole block and stopped the run with
"outflow: invalid EOS sound speed".

For implicit schemes 1 (vic-partial) and 9 (vic-full), NSTEP steps must run with no redo, stay at
rest (max |v| / c_s <= REST_TOL) and keep the vertical Courant number above COURANT_MIN, so the
step stays in the regime that failed.

  python test_outflow_stale_ghosts.py [--device cuda] [--nstep 5]
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

REST_TOL = 1.0e-3
COURANT_MIN = 2.0
P0, RHO0 = 1.0e5, 0.1  # isothermal, scale height P0 / (RHO0 g)


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


def rest_state(cfg, blocks):
    import snapy
    from snapy import kIDN, kIPR

    g = -cfg["forcing"]["const-gravity"]["grav1"]
    ng, nx1 = cfg["geometry"]["cells"]["nghost"], cfg["geometry"]["cells"]["nx1"]
    mesh_vars = []
    for block in blocks:
        coord = block.module("coord")
        x1v = coord.buffer("x1v").cpu()
        r = x1v[ng:ng + nx1]
        dr = coord.buffer("dx1f")[ng:ng + nx1].cpu()
        p = P0 * torch.exp(-(r - r[0]) * RHO0 * g / P0)
        col = torch.zeros(kIPR + 1, 1, 1, nx1, dtype=torch.float64)
        col[kIDN, 0, 0] = p * RHO0 / P0
        col[kIPR, 0, 0] = p
        wb, _, _ = snapy.balance_column(col, dr.contiguous(), g)
        w = dict(block.named_buffers())["hydro.D"].clone().zero_()
        for c in (kIDN, kIPR):
            w[c][..., ng:ng + nx1] = wb[c, 0, 0].to(w)
            w[c][..., :ng] = w[c][..., ng:ng + 1]
            # the outflow background above the top continues the hydrostatic profile
            for k in range(ng):
                i = ng + nx1 + k
                decay = math.exp(-(x1v[i] - x1v[i - 1]).item() * RHO0 * g / P0)
                w[c][..., i] = w[c][..., i - 1] * decay
        mesh_vars.append({"hydro_w": w})
    return mesh_vars


def run(cfg, device, nstep):
    from snapy import kIDN, kIPR

    mesh, blocks = build(cfg, device)
    mesh_vars, _ = mesh.initialize(rest_state(cfg, blocks))
    intg = blocks[0].module("intg")
    ng, nx1 = cfg["geometry"]["cells"]["nghost"], cfg["geometry"]["cells"]["nx1"]
    b = cfg["geometry"]["bounds"]
    dr = (float(b["x1max"]) - float(b["x1min"])) / nx1
    cs = math.sqrt(1.4 * P0 / RHO0)
    courant = math.inf
    for cycle in range(1, nstep + 1):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        courant = min(courant, cs * dt / dr)
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
        if mesh.check_redo(mesh_vars) != 0:
            raise RuntimeError("step rejected at cycle %d" % cycle)
    vmax = 0.0
    for ib, block in enumerate(blocks):
        sl = tuple(block.part((0, 0, 0), False)[1:])
        u = mesh_vars[ib]["hydro_u"]
        v = u[1:4][(slice(None),) + sl] / u[kIDN][sl]
        vmax = max(vmax, v.abs().max().item())
    # free this Mesh before the next one is built in the same process
    del mesh, blocks, mesh_vars
    gc.collect()
    return courant, vmax / cs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--nstep", type=int, default=5)
    ap.add_argument(
        "--yaml",
        default=str(Path(__file__).resolve().parent / "test_cubed_sphere_cell_volume.yaml"),
    )
    args = ap.parse_args(argv)
    if args.device == "cuda" and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 125
    torch.set_default_dtype(torch.float64)

    with open(args.yaml) as f:
        base = yaml.safe_load(f)
    failures = []
    for scheme in (1, 9):
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["boundary-condition"]["external"]["x1-outer"] = "outflow"
        cfg["integration"]["implicit-scheme"] = scheme
        try:
            courant, v = run(cfg, args.device, args.nstep)
        except RuntimeError as e:
            print("implicit-scheme %d: FAILED: %s" % (scheme, str(e).splitlines()[0]))
            failures.append("implicit-scheme %d: %s" % (scheme, str(e).splitlines()[0]))
            continue
        print("implicit-scheme %d: %d steps, min vertical Courant %.1f, max |v| / c_s = %.3e"
              % (scheme, args.nstep, courant, v))
        if not courant > COURANT_MIN:
            failures.append("implicit-scheme %d: vertical Courant %.2f <= %.1f, the step no longer "
                            "reaches the regime under test" % (scheme, courant, COURANT_MIN))
        if not v <= REST_TOL:
            failures.append("implicit-scheme %d: rest state moved, max |v| / c_s = %g" % (scheme, v))

    for msg in failures:
        print("FAIL:", msg)
    if failures:
        return 1
    print("### outflow stale-ghost test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
