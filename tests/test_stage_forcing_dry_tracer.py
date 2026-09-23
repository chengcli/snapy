#!/usr/bin/env python3
"""Tracers are per DRY air, so dry air a forcing creates carries the cell's own r with it.
The native forcings record that increment (hydro_forward.cpp) and the scalar update follows it;
a USER STAGE forcing is applied later, in MeshBlockImpl::advance_local, and was not bracketed the
same way -- so it changed dry air while every tracer stayed put, and a uniform tracer stopped
being uniform.

Limit: a UNIFORM tracer with one scalar checks that tracer and dry air stay consistent; it cannot
see a wrong donor cell, an r read one cell off, or a mis-broadcast across the scalar dimension.

  python test_stage_forcing_dry_tracer.py [--device cuda]
"""
import argparse
import sys

import yaml
import tempfile
from pathlib import Path
from typing import Dict

import torch

R0 = 1.0e-3
TOL = 1.0e-12


class _DryForcing(torch.nn.Module):
    """Adds a thousandth of the dry density everywhere, and nothing else."""

    def forward(self, variables: Dict[str, torch.Tensor], dt: float,
                stage: int) -> Dict[str, torch.Tensor]:
        hydro_u = variables["hydro_u"]
        hydro_du = torch.zeros_like(hydro_u)
        hydro_du[0] = 1.0e-3 * hydro_u[0]
        return {"hydro_du": hydro_du}


def main() -> int:
    from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    yaml_path = Path(__file__).resolve().parent / "test_tracer_dry_convention.yaml"
    block = MeshBlock(MeshBlockOptions.from_yaml(str(yaml_path)))
    block.to(torch.device(args.device))

    with tempfile.TemporaryDirectory(prefix="snapy-dry-forcing-") as directory:
        path = Path(directory) / "dry.pt"
        torch.jit.script(_DryForcing().eval()).save(str(path))
        block.set_user_stage_forcings([str(path)])

        coord = block.module("coord")
        eos = block.module("hydro.eos")
        shape = (eos.nvar(), coord.buffer("x3v").shape[0], coord.buffer("x2v").shape[0],
                 coord.buffer("x1v").shape[0])
        w = torch.zeros(shape, dtype=torch.float64, device=args.device)
        w[kIDN] = 1.0
        w[kIPR] = 1.06e6   # ~300 K in this fixture; 1e5 gives 28 K and 130 solver warnings
        w[kICY] = 1.0e-3
        r = torch.full((1,) + tuple(w.shape[1:]), R0, dtype=w.dtype, device=w.device)
        v, _ = block.initialize({"hydro_w": w, "scalar_r": r})

        with open(yaml_path) as f:
            ng = int(yaml.safe_load(f)["geometry"]["cells"]["nghost"])
        nx3 = int(coord.buffer("x3v").shape[0])
        k = slice(ng, -ng) if nx3 > 2 * ng else slice(None)
        interior = (slice(None), k, slice(ng, -ng), slice(ng, -ng))
        dt = block.max_time_step(v)
        dry0 = float(v["hydro_u"][kIDN][interior[1:]].sum())
        for stage in range(len(block.intg.stages)):
            block.forward(v, dt, stage)
        dry1 = float(v["hydro_u"][kIDN][interior[1:]].sum())

        ratio = v["scalar_s"][interior] / v["hydro_u"][kIDN][interior[1:]]
        # How far the TRACER drifted: 9.995e-04 on the unbracketed code, which is one
        # SSPRK3-weighted step of this forcing, not the 3e-3 the raw stage increments sum to.
        dev = float((ratio / R0 - 1.0).abs().max())
        # How much DRY MASS the forcing added, which is a different quantity. Reflecting x1
        # walls, periodic x2/x3, floors that cannot fire and a species-only saturation
        # adjustment leave the interior sum untouched by everything else, and the SSPRK3
        # weights put this forcing's one-step signature at f + f**2/2 = 1.0005e-3.
        grew = dry1 / dry0 - 1.0
        print(f"max |r/r0 - 1| = {dev:.3e}  (tol {TOL:.0e})")
        print(f"interior dry mass change = {grew:.4e}  (expect 1.0005e-03)")
        if not 5.0e-4 < grew < 5.0e-3:
            print("FAIL the stage forcing did not fire; the tracer check above is vacuous")
            return 1
        if dev > TOL:
            print("FAIL a stage forcing changed dry air without moving the tracers")
            return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
