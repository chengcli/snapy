#!/usr/bin/env python3
"""A/B test of the tracer flux positivity limiter (dynamics: positivity).

Setup: 2-D slab, periodic in x2, a smooth uniform wind (Mach ~0.05) carrying a
sharp vapor top-hat, reconstructed with the unlimited cp5 polynomial. The
Gibbs undershoot just downwind of the hat edge makes the reconstructed face
mixing ratio negative there, i.e. a spurious backward species flux that drains
cells which hold NO vapor at all:

  - base arm (positivity: false): vapor goes negative within a few cycles;
  - limited arm (positivity: true): a face draining an empty donor gets
    theta_donor = 0, so min(vapor) stays >= 0 while the hydro flow (uniform
    rho, p, u) is untouched.

One hat edge sits just upwind of the periodic wrap so the limited faces
straddle it, exercising the theta ghost fill through the periodic boundary
path: both images of the wrap face must carry the same donor factor, or the
total drifts. Both arms must conserve the vapor total to round-off (flux-form
transport conserves regardless of sign).

A passive-scalar top-hat rides along to verify the scalar-module wiring runs
and conserves (scalar face values are clamped >= 0 by its reconstruction, so
the scalar itself cannot go negative here).

Run with PYTHONPATH pointing at the snapy install under test, from a neutral
cwd:

  python test_flux_positivity.py [--device cuda] [--yaml PATH]
"""
import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml


def run_arm(yaml_file: str, positivity: bool, device: str):
    from snapy import MeshBlock, MeshBlockOptions, kICY

    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    config["dynamics"]["positivity"] = positivity
    nghost = config["geometry"]["cells"]["nghost"]
    nx2 = config["geometry"]["cells"]["nx2"]
    x2min = config["geometry"]["bounds"]["x2min"]
    x2max = config["geometry"]["bounds"]["x2max"]
    nlim = config["integration"]["nlim"]

    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, dir=os.getcwd()
    ) as f:
        yaml.safe_dump(config, f)
        tmp = f.name

    try:
        op = MeshBlockOptions.from_yaml(tmp)
        block = MeshBlock(op)
        block.to(torch.device(device))

        bufs = dict(block.named_buffers())
        w = bufs["hydro.D"].clone().zero_()  # (nvar, nc3, nc2, nc1)
        assert w.size(0) == kICY + 1, f"expected 1 vapor, got nvar={w.size(0)}"
        nc2 = w.size(2)

        dx2 = (x2max - x2min) / nx2
        j = torch.arange(nc2, dtype=torch.float64, device=w.device)
        frac = ((j - nghost + 0.5) * dx2 - x2min) / (x2max - x2min)

        # uniform smooth carrier flow
        w[0] = 1.0  # IDN: total density
        w[2] = 20.0  # IVY: x2 wind
        w[4] = 1.0e5  # IPR
        # vapor top-hat with its downwind edge just short of the periodic wrap
        hat = (frac % 1.0 >= 0.55) & (frac % 1.0 < 0.98)
        w[kICY] = 0.2 * hat.to(torch.float64).view(1, -1, 1)

        # passive-scalar top-hat rides along
        r = torch.zeros((1,) + tuple(w.shape[1:]), dtype=w.dtype, device=w.device)
        r[0] = hat.to(torch.float64).view(1, -1, 1)

        block_vars, _t = block.initialize({"hydro_w": w, "scalar_r": r})

        interior = (
            slice(None),
            slice(None),
            slice(nghost, -nghost),
            slice(nghost, -nghost),
        )
        v0 = block_vars["hydro_u"][interior][kICY].sum().item()
        s0 = block_vars["scalar_s"][interior].sum().item()

        min_seen = 0.0
        for _cycle in range(nlim):
            dt = block.max_time_step(block_vars)
            for stage in range(len(block.intg.stages)):
                block.forward(block_vars, dt, stage)
                m = block_vars["hydro_u"][interior][kICY].min().item()
                if not math.isnan(m):
                    min_seen = min(min_seen, m)

        v1 = block_vars["hydro_u"][interior][kICY].sum().item()
        s1 = block_vars["scalar_s"][interior].sum().item()
        hits = dict(block.named_buffers())["hydro.positivity_hits"].item()
        return {
            "min": min_seen,
            "vapor_drift": abs(v1 - v0) / abs(v0),
            "scalar_drift": abs(s1 - s0) / abs(s0),
            "hits": hits,
        }
    finally:
        os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--yaml",
        default=str(Path(__file__).resolve().parent / "test_flux_positivity.yaml"),
    )
    args = parser.parse_args()

    base = run_arm(args.yaml, positivity=False, device=args.device)
    lim = run_arm(args.yaml, positivity=True, device=args.device)

    for name, arm in (("base", base), ("limited", lim)):
        print(
            f"{name:7s}: min={arm['min']:+.6e} vapor_drift={arm['vapor_drift']:.3e} "
            f"scalar_drift={arm['scalar_drift']:.3e} hits={arm['hits']}"
        )

    failures = []
    if not base["min"] < -1e-12:
        failures.append(
            "base arm did not go negative (min=%g): the test no longer bites"
            % base["min"]
        )
    if not lim["min"] >= -1e-15:
        failures.append("limited arm went negative: min=%g" % lim["min"])
    if not lim["hits"] > 0:
        failures.append("limiter never fired (hits=0) despite base going negative")
    for name, arm in (("base", base), ("limited", lim)):
        if not arm["vapor_drift"] < 1e-12:
            failures.append(
                "%s arm vapor total drifted: %g%s"
                % (
                    name,
                    arm["vapor_drift"],
                    (
                        " -- theta not single-valued at the periodic wrap?"
                        if name == "limited"
                        else ""
                    ),
                )
            )
        if not arm["scalar_drift"] < 1e-12:
            failures.append("%s arm scalar total drifted: %g" % (name, arm["scalar_drift"]))

    if failures:
        for msg in failures:
            print("FAIL:", msg)
        sys.exit(1)
    print("### flux positivity test passed. ###")


if __name__ == "__main__":
    main()
