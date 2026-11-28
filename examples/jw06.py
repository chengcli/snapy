"""Run the flat-bottom JW06 benchmark, e.g. python examples/jw06.py --device cuda.

Initial fields follow equations 1–11 of Jablonowski and Williamson (2006),
doi:10.1256/qj.06.12. The reflecting spherical bottom clips the atmosphere
at the highest analytic surface. Surface pressure therefore varies with latitude.
"""

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import torch
import yaml

A, OMEGA, G, RD = 6371229.0, 7.29212e-5, 9.80616, 287.0
T0, LAPSE, DTEMP, ET, U0 = 288.0, 0.005, 4.8e5, 0.2, 35.0
EXPONENT = RD * LAPSE / G


def latitude_terms(lat):
    a = -2 * lat.sin() ** 6 * (lat.cos() ** 2 + 1 / 3) + 10 / 63
    b = (8 / 5 * lat.cos() ** 3 * (lat.sin() ** 2 + 2 / 3) - math.pi / 4) * A * OMEGA
    return a, b


def geopotential(eta, a, b):
    jet = U0 * ((eta - 0.252) * math.pi / 2).cos() ** 1.5
    x = (eta / ET).clamp_max(1.0)
    integral = x.log() + 137 / 60 - 5 * x + 5 * x**2 - 10 / 3 * x**3 + 5 / 4 * x**4 - x**5 / 5
    return (
        T0 * G / LAPSE * (1 - eta**EXPONENT) - RD * DTEMP * ET**5 * integral + jet * (a * jet + b)
    )


def bottom_height():
    # Both latitude coefficients attain their maximum at the equator.
    one = torch.tensor(1.0, dtype=torch.float64)
    return math.ceil(geopotential(one, *latitude_terms(one * 0)).item() / G)


def initial_fields(lon, lat, height, disturbed):
    """Pressure, density and eastward wind at broadcastable FP64 points."""
    a, b = latitude_terms(lat)
    lo, hi = torch.full_like(height + lat, 1.0e-8), torch.ones_like(height + lat)
    for _ in range(55):
        mid = (lo + hi) / 2
        above = geopotential(mid, a, b) > G * height
        lo, hi = torch.where(above, mid, lo), torch.where(above, hi, mid)
    eta = (lo + hi) / 2
    angle = (eta - 0.252) * math.pi / 2
    jet = U0 * angle.cos() ** 1.5
    temp = (
        T0 * eta**EXPONENT
        + DTEMP * (ET - eta).clamp_min(0) ** 5
        + 0.75 * math.pi * eta * U0 / RD * angle.sin() * angle.cos().sqrt() * (2 * a * jet + b)
    )
    wind = jet * (2 * lat).sin() ** 2
    if disturbed:
        center = 2 * math.pi / 9
        dot = (
            math.sin(center) * lat.sin() + math.cos(center) * lat.cos() * (lon - math.pi / 9).cos()
        )
        wind = wind + (-100 * dot.clamp(-1, 1).acos().square()).exp()
    pressure = 1.0e5 * eta
    return pressure, pressure / (RD * temp), wind


def make(args):
    import snapy

    device = torch.device(args.device)
    os.environ.update(DEVICE=device.type, DEVICE_ID="0")
    cfg = yaml.safe_load(Path(__file__).with_suffix(".yaml").read_text())
    n, m = args.resolution, args.subdivide
    if m < 1 or n % m or n // m < 6 or args.levels < 6:
        raise ValueError("each block needs at least six cells per axis, and equal subdivisions")
    bottom = bottom_height()
    cfg["geometry"]["bounds"].update(x1min=A + bottom)
    cfg["geometry"]["cells"].update(nx1=args.levels, nx2=n // m, nx3=n // m)
    cfg["distribute"].update(nb2=m, nb3=m, blocks_per_process=6 * m * m)
    split = args.mode != "fp64"
    cfg["dynamics"].update(
        precision="float32" if args.mode == "fp32" else "float64",
        anomaly=split,
        **{"perturbation-flux": split},
    )
    cfg["integration"].update(
        execution=args.execution, **{"execution-strict": args.execution == "cuda-graph"}
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as deck:
        yaml.safe_dump(cfg, deck)
        deck.flush()
        opts = snapy.MeshOptions.from_yaml(deck.name)
        opts.blocks_per_process(6 * m * m)
        opts.set_local_horizontal_cells(n // m, n // m)
        mesh = snapy.Mesh(opts)
    states = []
    for rank, block in enumerate(mesh.blocks):
        block.to(device)
        coord = block.module("coord")
        beta, alpha = torch.meshgrid(
            coord.buffer("x3v").double(), coord.buffer("x2v").double(), indexing="ij"
        )
        face = block.get_layout().loc_of(rank)[2]
        lon, lat = snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face), alpha, beta)
        height = (coord.buffer("x1v").double() - A).clamp(bottom, 30000)
        p, rho, wind = initial_fields(lon[..., None], lat[..., None], height, not args.balanced)
        # The basis is height-independent. Transform the eastward unit vector once.
        east = torch.zeros((3, *alpha.shape), dtype=alpha.dtype, device=device)
        east[2].fill_(1)
        snapy.coord.cs_sph_to_contra_(east, alpha, beta, face)
        w = torch.zeros_like(dict(block.named_buffers())["hydro.D"], dtype=torch.float64)
        w[snapy.kIPR], w[snapy.kIDN] = p, rho
        w[snapy.kIV1 : snapy.kIV3 + 1] = east[..., None] * wind
        states.append({"hydro_w": w})
    states, _ = mesh.initialize(states)
    return mesh, states


def final_primitive(block, state):
    # hydro_w contains the input to the last stage. Convert the accepted state.
    key = "U'->W" if block.anomaly() else "U->W"
    output = torch.empty_like(state["hydro_w"])
    return block.module("hydro.eos").compute(key, [state["hydro_u"].clone(), output])


def run(args):
    import snapy

    if (
        args.steps < 1
        or args.warmup < 0
        or args.days < 0
        or (args.dt is not None and (not math.isfinite(args.dt) or args.dt <= 0))
    ):
        raise ValueError("expected positive steps and dt, and nonnegative days and warmup")
    if args.days and args.warmup:
        raise ValueError("a science integration starts from the analytic state, with zero warmup")
    mesh, states = make(args)
    dt = mesh.max_time_step(states) if args.dt is None else args.dt
    cycle = 0

    def advance(step):
        nonlocal cycle
        cycle += 1
        mesh.set_cycle(cycle)
        for stage in range(3):
            mesh.forward(states, step, stage)
        if mesh.check_redo(states):
            raise RuntimeError(f"mesh rejected the RK3 timestep of {step} seconds")

    for _ in range(args.warmup):
        advance(dt)
    if args.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    elapsed = 0.0
    count = 0
    while elapsed < args.days * 86400 if args.days else count < args.steps:
        step = min(mesh.max_time_step(states), args.days * 86400 - elapsed, dt) if args.days else dt
        if not math.isfinite(step) or step <= 0:
            raise RuntimeError(f"invalid timestep {step}")
        advance(step)
        elapsed += step
        count += 1
    if args.device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    report = mesh.execution_report()
    if args.execution == "cuda-graph" and (not report["replays"] or report["eager"]):
        raise RuntimeError(f"strict CUDA graph execution was not used: {report}")
    minima = []
    for block, state in zip(mesh.blocks, states):
        w = final_primitive(block, state)[:, 3:-3, 3:-3, 3:-3]
        if not w.isfinite().all() or not state["hydro_u"].isfinite().all():
            raise ValueError("nonfinite state")
        minima.append([w[snapy.kIPR].min().item(), w[snapy.kIDN].min().item()])
    if min(map(min, minima)) <= 0:
        raise ValueError("nonpositive pressure or density")
    print(
        json.dumps(
            dict(
                mode=args.mode,
                device=args.device,
                execution=args.execution,
                graph_replays=report["replays"],
                resolution=args.resolution,
                levels=args.levels,
                steps=count,
                dt=dt,
                model_seconds=elapsed,
                wall_seconds=seconds,
                min_pressure=min(x[0] for x in minima),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--levels", type=int, default=26)
    parser.add_argument("--subdivide", type=int, default=1)
    parser.add_argument("--mode", choices=("fp64", "split64", "fp32"), default="fp64")
    parser.add_argument("--execution", choices=("eager", "cuda-graph"), default="eager")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--days", type=float, default=0.0)
    parser.add_argument("--dt", type=float)
    run(parser.parse_args())
