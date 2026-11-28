"""Compare the flat-bottom case at two resolutions on matched simulation times.

Run with --balanced for the 30-day twin. The disturbed default runs ten days.
The field is pressure at the lowest cell centre, in Pa. The clipped shell's
lower boundary has variable pressure. A C96 field is restricted to C48 by
physical cell volume. Each daily result is printed as JSON.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import jw06


def snapshot(mesh, states):
    import snapy

    pressures, weights = [], []
    mass = 0.0
    vmax = 0.0
    for b, s in zip(mesh.blocks, states):
        w = jw06.final_primitive(b, s)[:, 3:-3, 3:-3, 3:-3]
        if not w.isfinite().all() or w[snapy.kIDN].min() <= 0 or w[snapy.kIPR].min() <= 0:
            raise ValueError("invalid accepted state")
        vol = b.module("coord").cell_volume()[3:-3, 3:-3, 3:-3]
        mass += (vol * w[snapy.kIDN]).sum().item()
        vmax = max(vmax, w[snapy.kIV1].abs().max().item())
        pressures.append(w[snapy.kIPR, :, :, 0].cpu())
        weights.append(vol[:, :, 0].cpu())
    return torch.stack(pressures), torch.stack(weights), mass, vmax


def run_variant(args, n, mode, directory):
    # One mesh lives on the accelerator at a time. Keep only daily diagnostics.
    opts = SimpleNamespace(**vars(args), resolution=n, mode=mode, subdivide=1)
    mesh, states = jw06.make(opts)
    records = []
    t = 0.0
    cycle = 0
    for target in args.times:
        while t < target:
            dt = min(mesh.max_time_step(states), target - t)
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError(f"invalid timestep {dt}")
            mesh.set_cycle(cycle + 1)
            while True:
                for stage in range(3):
                    mesh.forward(states, dt, stage)
                status = mesh.check_redo(states)
                if not status:
                    break
                if status < 0:
                    raise RuntimeError(f"unrecoverable {mode} C{n} step at t={t}")
                dt *= 0.5
            cycle += 1
            t += dt
        p, weights, mass, vmax = snapshot(mesh, states)
        if not records:
            mass0, p0 = mass, p
        drift = ((weights * (p - p0).square()).sum() / weights.sum()).sqrt().item()
        print(
            json.dumps(
                dict(
                    mode=mode,
                    resolution=n,
                    time=t,
                    relative_mass=mass / mass0 - 1,
                    pressure_drift_rms=drift,
                    max_radial_speed=vmax,
                )
            ),
            flush=True,
        )
        records.append(p)
        if directory:
            torch.save(
                dict(pressure=p, weight=weights, mass=mass, max_radial_speed=vmax),
                directory / f"{mode}-C{n}-{target:g}s.pt",
            )
    return records, weights


def compare(args):
    coarse, weight = run_variant(args, args.coarse, "fp64", args.output)
    fine, wf = run_variant(args, args.coarse * 2, "fp64", args.output)
    narrow, _ = run_variant(args, args.coarse, "fp32", args.output)

    def restrict(x):
        return x.reshape(6, args.coarse, 2, args.coarse, 2).sum(dim=(2, 4))

    def rms(x):
        return ((weight * x.square()).sum() / weight.sum()).sqrt().item()

    failures = []
    for t, c, f, s in zip(args.times, coarse, fine, narrow):
        resolution = rms(c - restrict(f * wf) / restrict(wf))
        precision = rms(s - c)
        print(
            json.dumps(
                dict(time=t, pressure_precision_rms=precision, pressure_resolution_rms=resolution)
            ),
            flush=True,
        )
        if 0 < t <= 9 * 86400 and precision > resolution:
            failures.append(t)
    if failures:
        raise RuntimeError(f"precision error exceeded resolution error at {failures} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--execution", choices=("cuda-graph", "eager"), default="eager")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--coarse", type=int, default=48)
    parser.add_argument("--levels", type=int, default=26)
    parser.add_argument("--seconds", type=float, help="shorten the run for a smoke test")
    parser.add_argument("--output", type=Path, help="directory for daily pressure fields")
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    args.times = (
        [0.0, args.seconds]
        if args.seconds
        else [day * 86400 for day in range(31 if args.balanced else 11)]
    )
    compare(args)
