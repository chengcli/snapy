#!/usr/bin/env python3
"""A user stage forcing must not take a species below what transport leaves.

A user stage forcing is added to the stage tendency after the flux-positivity
limiter (theta) and the vertical implicit correction, and before the stage
average and the conserved limiter (MeshBlockImpl::advance_local). Nothing
bounds its species increment by what the stage's transport left in the cell.

The box: a 3x3x3 column triple with a one-cell dry slot at the middle level
(vapor density 2.026e-6 between 4.7e-3 and 1.9e-5), 1.096e-6 below and
3.29e-9 above it, 20 m/s along x2, dt 405 s. The forcing is a boundary-layer
style vertical diffusion of q (backward Euler over the full stage dt, zero
flux at the box walls) applied as a vapor-density increment computed from the
stage-input state, as a physics stage forcing does.

Required, for both wind directions, at stage 0 (weights 0, 1, 1):
  * the slot vapor after the stage is >= 0;
  * the cell below the slot is exactly what transport plus the forcing give,
    i.e. no species repair borrowed from it (a repair that marks on the
    deficit instead of re-mixing still borrows, so it does not pass this);
and over the whole step: no limiter mark at the first dt.
"""
import argparse
import tempfile
from pathlib import Path
from typing import Dict

import torch

NG = 3
DT = 405.0
DZ = 343.0
KDIFF = 3.5e-3      # m2/s, a boundary-layer diffusivity tapered at ~570 hPa
T0 = 275.0          # K
P_MID = 5.8e4       # Pa at the middle level
RD = 287.0
GRAV = 9.80616

# vapor density [kg/m3], bottom to top, of the slot column and of the slot's
# x2 neighbours at the middle level (the other levels of the neighbour
# columns copy the slot column's)
SLOT = (1.096e-6, 2.026e-6, 3.29e-9)
LEFT_MID = 4.7e-3
RIGHT_MID = 1.9e-5


class _VaporDiffusion(torch.nn.Module):
    """Backward-Euler diffusion of q along x1 over the stage dt, as a vapor
    density increment from the stage-input state; zero flux at the walls."""

    def __init__(self, coef: float, dz: float, ng: int):
        super().__init__()
        self.coef = coef
        self.dz = dz
        self.ng = ng

    def forward(self, variables: Dict[str, torch.Tensor], dt: float,
                stage: int) -> Dict[str, torch.Tensor]:
        u = variables["hydro_u"]
        du = torch.zeros_like(u)
        ng = self.ng
        n = u.size(-1) - 2 * ng
        rho = u[0] + u[5:].sum(0)
        q = (u[5] / rho).narrow(-1, ng, n)
        r = rho.narrow(-1, ng, n)
        a = self.coef * dt / (self.dz * self.dz)
        # tridiagonal (I - a L) x = q, Thomas from the bottom
        cp = torch.zeros_like(q)
        dp = torch.zeros_like(q)
        for k in range(n):
            lo = a if k > 0 else 0.
            hi = a if k < n - 1 else 0.
            b = 1. + lo + hi
            if k == 0:
                cp[..., k] = -hi / b
                dp[..., k] = q[..., k] / b
            else:
                den = b + lo * cp[..., k - 1]
                cp[..., k] = -hi / den
                dp[..., k] = (q[..., k] + lo * dp[..., k - 1]) / den
        x = torch.zeros_like(q)
        x[..., n - 1] = dp[..., n - 1]
        for k in range(n - 2, -1, -1):
            x[..., k] = dp[..., k] - cp[..., k] * x[..., k + 1]
        du[5].narrow(-1, ng, n).copy_(r * (x - q))
        return {"hydro_du": du}


def _block(card: Path, forcing: str, device: torch.device, v2: float):
    from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR, kIV2

    block = MeshBlock(MeshBlockOptions.from_yaml(str(card)))
    block.to(device)
    if forcing:
        block.set_user_stage_forcings([forcing])
    coord = block.module("coord")
    x1v = coord.buffer("x1v")
    w = torch.zeros_like(dict(block.named_buffers())["hydro.D"])
    zmid = x1v[NG + 1]
    p = P_MID * torch.exp(-GRAV * (x1v - zmid) / (RD * T0))
    rho = p / (RD * T0)
    w[kIDN] = rho
    w[kIPR] = p
    w[kIV2] = v2
    rhov = torch.zeros_like(w[kICY])
    nc2 = rhov.size(1)
    for k, value in enumerate(SLOT):
        rhov[..., NG + k] = value
    mid = NG + 1
    j = NG + 1
    rhov[:, :, mid] = SLOT[1]
    rhov[:, j - 1, mid] = LEFT_MID
    rhov[:, j + 1, mid] = RIGHT_MID
    # periodic ghosts in x2 copy the interior pattern
    for g in range(NG):
        rhov[:, g, :] = rhov[:, nc2 - 2 * NG + g, :]
        rhov[:, nc2 - NG + g, :] = rhov[:, NG + g, :]
    w[kICY] = rhov / rho
    variables, _ = block.initialize({"hydro_w": w})
    return block, variables


def _case(card: Path, forcing_path: str, forcing: torch.nn.Module,
          device: torch.device, v2: float) -> list:
    from snapy import kICY

    j, below, mid, above = NG + 1, NG, NG + 1, NG + 2
    cell = (kICY, NG + 1, j)

    # what transport, theta and the implicit correction leave (no forcing)
    bare, bv = _block(card, "", device, v2)
    bare.forward(bv, DT, 0)
    rem = bv["hydro_u"][cell].clone()

    # the forcing's increment from the same stage-input state
    forced, fv = _block(card, forcing_path, device, v2)
    sink = forcing({"hydro_u": fv["hydro_u"].clone(),
                    "hydro_w": fv["hydro_w"].clone()}, DT, 0)["hydro_du"][cell]
    forced.forward(fv, DT, 0)
    post = fv["hydro_u"][cell]

    pre = rem + sink  # the stage-0 vapor before the conserved limiter
    tag = f"v2 {v2:+.0f} m/s"
    print(f"{tag}: remainder after transport/theta/VIC "
          f"[below, slot, above] = {rem[below]:.6e} {rem[mid]:.6e} "
          f"{rem[above]:.6e}")
    print(f"{tag}: forcing increment = {sink[below]:.6e} {sink[mid]:.6e} "
          f"{sink[above]:.6e}")
    print(f"{tag}: slot vapor before the repair = {pre[mid]:.6e}, "
          f"after the stage = {post[mid]:.6e}")
    print(f"{tag}: below the slot, transport+forcing = {pre[below]:.17e}, "
          f"after the stage = {post[below]:.17e}")

    failures = []
    tol = 1.e-14 * abs(float(pre[below])) + 1.e-30
    if float(post[mid]) < 0.:
        failures.append(f"{tag}: slot vapor {float(post[mid]):.3e} < 0")
    if abs(float(post[below] - pre[below])) > tol:
        failures.append(
            f"{tag}: a species repair moved the cell below the slot by "
            f"{float(post[below] - pre[below]):.3e} (forcing "
            f"{float(sink[mid]):.3e} vs remainder {float(rem[mid]):.3e})")
    if abs(float(post[above] - pre[above])) > 1.e-14 * abs(float(pre[above])):
        failures.append(f"{tag}: control cell above the slot changed")

    # the whole step at the first dt, then halvings until no limiter mark
    step, sv = _block(card, forcing_path, device, v2)
    dt, halvings, accepted = DT, 0, False
    for _ in range(7):
        for stage in range(len(step.intg.stages)):
            step.forward(sv, dt, stage)
            slot = float(sv["hydro_u"][cell][mid])
            if slot < 0.:
                failures.append(f"{tag}: dt {dt}: stage {stage} slot {slot:.3e}")
        if step.check_redo(sv) == 0:
            accepted = True
            break
        halvings += 1
        dt *= 0.5
    print(f"{tag}: halvings before the step was accepted: {halvings}"
          f"{'' if accepted else ' (still marked at dt ' + str(dt * 2) + ' s)'}")
    if halvings:
        failures.append(f"{tag}: limiter mark at dt {DT} s "
                        f"({halvings} halvings{'' if accepted else ', never accepted'})")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)

    card = Path(__file__).resolve().parent / "test_forcing_species_clip.yaml"
    forcing = _VaporDiffusion(KDIFF, DZ, NG).eval()
    failures = []
    with tempfile.TemporaryDirectory(prefix="snapy-species-clip-") as directory:
        path = str(Path(directory) / "vapor_diffusion.pt")
        torch.jit.script(forcing).save(path)
        for v2 in (20.0, -20.0):
            failures += _case(card, path, forcing, device, v2)
    for f in failures:
        print("FAIL:", f)
    print("RESULT", "FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
