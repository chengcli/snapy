#!/usr/bin/env python3
"""Dry-density sources must preserve passive-scalar bounds at every RK order."""
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml


class _DrySource(torch.nn.Module):
    def __init__(self, increment: float, at_stage: int):
        super().__init__()
        self.increment = increment
        self.at_stage = at_stage

    def forward(self, variables: Dict[str, torch.Tensor], dt: float,
                stage: int) -> Dict[str, torch.Tensor]:
        del dt
        du = torch.zeros_like(variables["hydro_u"])
        if self.at_stage < 0 or stage == self.at_stage:
            du[0].select(-1, du.size(-1) // 2).fill_(self.increment)
        return {"hydro_du": du}


class _UniformDrySource(torch.nn.Module):
    """Adds one thousandth of the dry density everywhere."""

    def forward(self, variables: Dict[str, torch.Tensor], dt: float,
                stage: int) -> Dict[str, torch.Tensor]:
        del dt, stage
        du = torch.zeros_like(variables["hydro_u"])
        du[0] = 1.0e-3 * variables["hydro_u"][0]
        return {"hydro_du": du}


class _ScalarSource(torch.nn.Module):
    def __init__(self, increment: float, at_stage: int):
        super().__init__()
        self.increment = increment
        self.at_stage = at_stage

    def forward(self, variables: Dict[str, torch.Tensor], dt: float,
                stage: int) -> Dict[str, torch.Tensor]:
        del dt
        ds = torch.zeros_like(variables["scalar_s"])
        if self.at_stage < 0 or stage == self.at_stage:
            ds.fill_(self.increment)
        return {"scalar_ds": ds}


def _run(kind: str, dry: float, scalar: float, velocity: float,
         device: torch.device, at_stage: int = -1,
         stop_stage: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

    source = Path(__file__).resolve().parent / "test_scalar_source_bound.yaml"
    with source.open() as stream:
        card = yaml.safe_load(stream)
    card["integration"]["type"] = kind

    with tempfile.TemporaryDirectory(prefix="snapy-source-bound-") as directory:
        directory = Path(directory)
        card_path = directory / "case.yaml"
        with card_path.open("w") as stream:
            yaml.safe_dump(card, stream)
        forcing_paths: List[str] = []
        if dry != 0.0:
            path = directory / "dry.pt"
            torch.jit.script(_DrySource(dry, at_stage).eval()).save(str(path))
            forcing_paths.append(str(path))
        if scalar != 0.0:
            path = directory / "scalar.pt"
            torch.jit.script(_ScalarSource(scalar, at_stage).eval()).save(str(path))
            forcing_paths.append(str(path))

        block = MeshBlock(MeshBlockOptions.from_yaml(str(card_path)))
        block.to(device)
        if forcing_paths:
            block.set_user_stage_forcings(forcing_paths)

        buffers = dict(block.named_buffers())
        w = torch.zeros_like(buffers["hydro.D"])
        w[kIDN].fill_(1.0)
        w[kIPR].fill_(1.0e5)
        w[kIV1].fill_(velocity)
        r = torch.full((1,) + tuple(w.shape[1:]), 0.9,
                       dtype=w.dtype, device=device)
        ng = 2
        r[..., ng + 7] = 1.0
        variables, _ = block.initialize({"hydro_w": w, "scalar_r": r})
        dt = 6.0e-4
        assert dt <= block.max_time_step(variables)
        last = stop_stage if stop_stage >= 0 else len(block.intg.stages) - 1
        for stage in range(last + 1):
            block.forward(variables, dt, stage)
        interior = (..., slice(ng, -ng))
        return (variables["scalar_s"][interior].cpu(),
                variables["hydro_u"][kIDN][interior].cpu(),
                variables["scalar_r"][interior].cpu())


def _uniform_moist_regression(device: torch.device) -> str:
    """Keep the original moist-fixture and total-dry-mass coverage."""
    from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR

    yaml_path = Path(__file__).resolve().parent / "test_tracer_dry_convention.yaml"
    block = MeshBlock(MeshBlockOptions.from_yaml(str(yaml_path)))
    block.to(device)

    with tempfile.TemporaryDirectory(prefix="snapy-dry-forcing-") as directory:
        path = Path(directory) / "dry.pt"
        torch.jit.script(_UniformDrySource().eval()).save(str(path))
        block.set_user_stage_forcings([str(path)])

        coord = block.module("coord")
        eos = block.module("hydro.eos")
        shape = (eos.nvar(), coord.buffer("x3v").shape[0],
                 coord.buffer("x2v").shape[0], coord.buffer("x1v").shape[0])
        w = torch.zeros(shape, dtype=torch.float64, device=device)
        w[kIDN] = 1.0
        w[kIPR] = 1.06e6
        w[kICY] = 1.0e-3
        r = torch.full((1,) + tuple(w.shape[1:]), 1.0e-3,
                       dtype=w.dtype, device=device)
        variables, _ = block.initialize({"hydro_w": w, "scalar_r": r})

        with yaml_path.open() as stream:
            ng = int(yaml.safe_load(stream)["geometry"]["cells"]["nghost"])
        nx3 = int(coord.buffer("x3v").shape[0])
        k = slice(ng, -ng) if nx3 > 2 * ng else slice(None)
        interior = (slice(None), k, slice(ng, -ng), slice(ng, -ng))
        dt = block.max_time_step(variables)
        dry0 = float(variables["hydro_u"][kIDN][interior[1:]].sum())
        for stage in range(len(block.intg.stages)):
            block.forward(variables, dt, stage)
        dry1 = float(variables["hydro_u"][kIDN][interior[1:]].sum())

        ratio = (variables["scalar_s"][interior] /
                 variables["hydro_u"][kIDN][interior[1:]])
        deviation = float((ratio / 1.0e-3 - 1.0).abs().max())
        growth = dry1 / dry0 - 1.0
        print("uniform moist max ratio deviation", deviation,
              "dry mass growth", growth)
        if not 5.0e-4 < growth < 5.0e-3:
            return "uniform moist dry source did not change dry mass"
        if deviation > 1.0e-12:
            return f"uniform moist tracer drifted ({deviation})"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    device = torch.device(args.device)

    weights = {
        "rk1": (1.0,),
        "rk2": (1.0, 0.5),
        "rk3": (1.0, 0.25, 2.0 / 3.0),
        "rk3s4": (0.5, 0.5, 1.0 / 6.0, 0.5),
    }
    failures = []
    for kind, wght2 in weights.items():
        for dry in (-0.5, 0.5):
            _, _, ratio = _run(kind, dry, 0.0, 1000.0, device)
            lo, hi = float(ratio.min()), float(ratio.max())
            print(kind, "user dry", dry, "range", lo, hi)
            if lo < -1.0e-12 or hi > 1.0 + 1.0e-12:
                failures.append(f"{kind} dry {dry}: ratio range [{lo}, {hi}]")

        for stage, weight in enumerate(wght2):
            _, base_rho, _ = _run(kind, 0.0, 0.0, 0.0, device, stage, stage)
            dry_results = {}
            for dry in (-0.2, 0.2):
                dry_s, dry_rho, _ = _run(
                    kind, dry, 0.0, 0.0, device, stage, stage)
                delta = dry_rho - base_rho
                expected = torch.zeros_like(delta)
                expected[..., expected.size(-1) // 2] = weight * dry
                dry_error = float((delta - expected).abs().max())
                print(kind, "stage", stage, "dry", dry,
                      "density increment error", dry_error)
                if dry_error > 1.0e-12:
                    failures.append(
                        f"{kind} stage {stage}: dry {dry} did not apply "
                        f"with RK weight ({dry_error})")
                dry_results[dry] = dry_s

            both_s, _, _ = _run(kind, -0.2, 0.03, 0.0, device, stage, stage)
            error = float(
                ((both_s - dry_results[-0.2]) - weight * 0.03).abs().max())
            print(kind, "stage", stage, "explicit scalar_ds additive error", error)
            if error > 1.0e-12:
                failures.append(
                    f"{kind} stage {stage}: explicit scalar_ds was coupled "
                    f"to dry removal ({error})")

    old_failure = _uniform_moist_regression(device)
    if old_failure:
        failures.append(old_failure)

    if failures:
        print("\n".join("FAIL " + failure for failure in failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
