"""Verify the height inversion, both hemisphere jets, and the pressure split."""

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import jw06


def analytic():
    lat = torch.tensor([0.0, math.pi / 4, -math.pi / 4, 2 * math.pi / 9], dtype=torch.float64)
    lon = torch.full_like(lat, math.pi / 9)
    eta = torch.tensor([0.9, 0.252, 0.252, 0.08], dtype=lat.dtype, requires_grad=True)
    phi = jw06.geopotential(eta, *jw06.latitude_terms(lat))
    (dphi,) = torch.autograd.grad(phi.sum(), eta)
    p, rho, u = jw06.initial_fields(lon, lat, phi.detach() / jw06.G, False)
    torch.testing.assert_close(p, eta * 1.0e5, rtol=2.0e-14, atol=0)
    torch.testing.assert_close(rho, -1.0e5 / dphi, rtol=2.0e-13, atol=0)
    torch.testing.assert_close(u[1:3], torch.full_like(u[1:3], 35.0))
    _, _, disturbed = jw06.initial_fields(lon, lat, phi.detach() / jw06.G, True)
    torch.testing.assert_close((disturbed - u)[-1], torch.ones_like(u[-1]))


def evolve(device, subdivide, mode, execution):
    args = SimpleNamespace(
        device=device,
        mode=mode,
        subdivide=subdivide,
        resolution=16,
        levels=8,
        balanced=False,
        execution=execution,
    )
    mesh, states = jw06.make(args)
    for _ in range(3):
        for stage in range(3):
            mesh.forward(states, 1.0, stage)
        assert mesh.check_redo(states) == 0
    if execution == "cuda-graph":
        report = mesh.execution_report()
        assert report["captures"] > 0 and report["eager"] == 0, report
    dtype = torch.float32 if mode == "fp32" else torch.float64
    assert all(s["hydro_u"].dtype == dtype for s in states)
    panels = torch.empty((6, 5, args.resolution, args.resolution, args.levels), dtype=torch.float64)
    width = args.resolution // subdivide
    for rank, (block, state) in enumerate(zip(mesh.blocks, states)):
        x, y, face = block.get_layout().loc_of(rank)
        w = jw06.final_primitive(block, state)[:, 3:-3, 3:-3, 3:-3]
        panels[face, :, y * width : (y + 1) * width, x * width : (x + 1) * width] = w.cpu()
    return panels


def compare(device, subdivide):
    plain = evolve(device, subdivide, "fp64", "eager")
    for mode, rtol, atol in (("split64", 1.0e-11, 1.0e-10), ("fp32", 2.0e-5, 1.0e-4)):
        result = evolve(device, subdivide, mode, "eager")
        torch.testing.assert_close(result, plain, rtol=rtol, atol=atol)
        if subdivide > 1:
            single = evolve(device, 1, mode, "eager")
            torch.testing.assert_close(single, result, rtol=rtol, atol=atol)
        if device == "cuda" and mode == "fp32":
            captured = evolve(device, subdivide, mode, "cuda-graph")
            torch.testing.assert_close(captured, result, rtol=0, atol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--subdivide", type=int, default=1)
    args = parser.parse_args()
    analytic()
    compare(args.device, args.subdivide)
    print("JW06 analytic and evolution checks passed")
