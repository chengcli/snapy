#!/usr/bin/env python3
"""Two live Mesh objects in one process, and a failing block under the mesh worker pool.

Six cubed-sphere panels per Mesh (blocks_per_process 6). The same-process exchange keeps a
registry of the blocks of a process; before the local_group key it was shared by every Mesh of
the process, so a second Mesh built while the first was alive waited forever in initialize.
  (a) mesh A is initialized, then mesh B is built and initialized with A still alive; the two
      give the same ghost zones from the same input, and both still exchange afterwards, B also
      after A is freed;
  (b) a block whose initialize fails (wrong hydro_w shape) raises from Mesh.initialize instead
      of leaving the other five workers waiting at the exchange barrier, and the mesh works
      again afterwards.

  python test_two_meshes_one_process.py [--yaml PATH]
"""
import argparse
import gc
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml


def build(cfg):
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
    return mesh, list(mesh.blocks)


def rest_state(cfg, blocks):
    """Isothermal column in the scheme's discrete balance; ghost zones left at zero."""
    import snapy
    from snapy import kIDN, kIPR

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
        # a per-panel marker, so the ghost zones show which panel they came from
        w[kIDN] *= 1.0 + 0.01 * len(mesh_vars)
        w[kIPR] *= 1.0 + 0.01 * len(mesh_vars)
        mesh_vars.append({"hydro_w": w})
    return mesh_vars


def same_state(va, vb):
    return all(torch.equal(a["hydro_w"], b["hydro_w"]) and torch.equal(a["hydro_u"], b["hydro_u"])
               for a, b in zip(va, vb))


def two_meshes(cfg):
    mesh_a, blocks_a = build(cfg)
    vars_a, _ = mesh_a.initialize(rest_state(cfg, blocks_a))
    mesh_b, blocks_b = build(cfg)
    vars_b, _ = mesh_b.initialize(rest_state(cfg, blocks_b))  # A still alive
    failures = []
    if not same_state(vars_a, vars_b):
        failures.append("mesh B initialized next to a live mesh A differs from A")
    mesh_a.exchange_ghost_zones(vars_a)
    mesh_b.exchange_ghost_zones(vars_b)
    if not same_state(vars_a, vars_b):
        failures.append("the two live meshes exchange differently")
    del blocks_a, vars_a, mesh_a
    gc.collect()
    vars_c = [{"hydro_w": v["hydro_w"].clone(), "hydro_u": v["hydro_u"].clone()} for v in vars_b]
    mesh_b.exchange_ghost_zones(vars_b)
    if not same_state(vars_b, vars_c):
        failures.append("mesh B exchanges differently after mesh A was freed")
    return failures


def failing_block(cfg):
    mesh, blocks = build(cfg)
    bad = rest_state(cfg, blocks)
    bad[3]["hydro_w"] = bad[3]["hydro_w"][..., :-1].contiguous()  # wrong nc1: initialize_local throws
    try:
        mesh.initialize(bad)
    except RuntimeError as e:
        msg = str(e)
    else:
        return ["Mesh.initialize with a bad block did not raise"]
    failures = []
    if "hydro_w has incorrect shape" not in msg:
        failures.append("the block's own error was not the one raised: %s" % msg.splitlines()[0])
    mesh_vars, _ = mesh.initialize(rest_state(cfg, blocks))  # the pool and the barrier recovered
    mesh.exchange_ghost_zones(mesh_vars)
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--yaml",
        default=str(Path(__file__).resolve().parent / "test_cubed_sphere_cell_volume.yaml"),
    )
    args = ap.parse_args(argv)
    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)

    failures = two_meshes(cfg)
    print("(a) two live meshes: %s" % ("ok" if not failures else "FAIL"))
    more = failing_block(cfg)
    print("(b) failing block under the worker pool: %s" % ("ok" if not more else "FAIL"))
    failures += more
    for msg in failures:
        print("FAIL:", msg)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
