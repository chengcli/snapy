#!/usr/bin/env python3
"""kintera's species tables are process-global and can be REWRITTEN after a MeshBlock is built
(kintera's photochem kinetics-base reader clears and refills them; so do kintera.set_species_*).
A block's moist EOS snapshots its molar masses and heat capacities at construction; nothing it
does per step may read the global tables afterwards, or its temperature, energies and fluxes
change under it.

Two identical blocks (cloud + sedimentation on) are built first. Then:

  compute  W->T, UT->I and W->E of block 1 agree before and after the global tables are rewritten.
  step     one step of block 2 taken AFTER the rewrite is bitwise identical to the same step of
           block 1 taken BEFORE it.

The bite check asserts the rewrite really changed the global dry molar mass and heat capacity.
The tables are restored before exit.

  python test_eos_species_registry.py [--device cuda]
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
TOL = 1.0e-14


def make_block(config, device):
    from snapy import MeshBlock, MeshBlockOptions

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))
    return block


def card():
    with open(HERE / "test_tracer_dry_convention.yaml") as f:
        c = yaml.safe_load(f)
    c.pop("scalar", None)
    c["integration"]["implicit-scheme"] = 0
    c["dynamics"]["equation-of-state"]["limiter"] = True
    c["sedimentation"] = {"radius": {"cloud": 1.0e-5}, "density": {"cloud": 1000.0},
                          "const-vsed": {"cloud": -10.0}}
    return c


def state(block, config):
    from snapy import kICY, kIDN, kIPR

    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    ng = config["geometry"]["cells"]["nghost"]
    dz = float(config["geometry"]["bounds"]["x1max"]) / config["geometry"]["cells"]["nx1"]
    z = (torch.arange(w.size(-1), dtype=torch.float64, device=w.device) - ng + 0.5) * dz
    T, R = 320.0, 3900.0
    p = 1.0e5 * torch.exp(-z * 10.0 / (R * T))
    w[kIPR] = p.view(1, 1, -1)
    w[kIDN] = (p / (R * T)).view(1, 1, -1)
    w[kICY] = 0.02         # vapour
    w[kICY + 1] = 1.0e-3   # cloud, so sedimentation carries mass and energy
    return w


def one_step(block, v):
    dt = block.max_time_step(v)
    for stage in range(len(block.intg.stages)):
        block.forward(v, dt, stage)
    return dt


def main():
    import kintera

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    failures = []
    c = card()

    b1, b2 = make_block(c, args.device), make_block(c, args.device)
    v1, _ = b1.initialize({"hydro_w": state(b1, c)})
    v2, _ = b2.initialize({"hydro_w": state(b2, c)})

    eos = b1.module("hydro.eos")
    w0, u0 = v1["hydro_w"].clone(), v1["hydro_u"].clone()
    t_before = eos.compute("W->T", [w0])
    before = {"W->T": t_before, "UT->I": eos.compute("UT->I", [u0, t_before]), "W->E": eos.compute("W->E", [w0])}
    dt1 = one_step(b1, v1)

    weights = list(kintera.species_weights())
    cref = list(kintera.species_cref_R())
    try:
        kintera.set_species_weights([x * 12.0 for x in weights])   # e.g. N2 instead of H2/He
        kintera.set_species_cref_R([x * 1.2 for x in cref])
        bite = abs(float(kintera.species_weights()[0]) / weights[0] - 1.0)
        print(f"global dry molar mass changed by {bite:.3e} (relative)")
        if not bite > 1.0e-2:
            failures.append("rewriting kintera's tables did not take effect; test cannot bite")

        after = {"W->T": eos.compute("W->T", [w0])}
        after["UT->I"] = eos.compute("UT->I", [u0, t_before])
        after["W->E"] = eos.compute("W->E", [w0])
        for name in ("W->T", "UT->I", "W->E"):
            rel = float(((after[name] - before[name]).abs() / before[name].abs().clamp_min(1e-300)).max())
            print(f"compute : {name:5s} max rel change after the rewrite = {rel:.3e}")
            if not rel < TOL:
                failures.append(f"compute: {name} changed by {rel:.3e} after kintera's tables were rewritten")

        dt2 = one_step(b2, v2)
        diff = float((v2["hydro_u"] - v1["hydro_u"]).abs().max())
        print(f"step    : dt equal={dt1 == dt2}  max|u(after rewrite) - u(before)| = {diff:.3e}")
        if dt1 != dt2 or diff != 0.0:
            failures.append(f"step: a step after the rewrite differs (dt equal={dt1 == dt2}, diff {diff:.3e})")
    finally:
        kintera.set_species_weights(weights)
        kintera.set_species_cref_R(cref)

    if failures:
        for msg in failures:
            print("FAIL:", msg)
        sys.exit(1)
    print("### eos species-registry test passed. ###")


if __name__ == "__main__":
    main()
