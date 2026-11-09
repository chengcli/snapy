#!/usr/bin/env python3
"""A sideways over-drain must not take a tracer or a species negative where the dry air is positive.
The relative rule lets a cell drain past its content only as far as its poorer neighbour repays.
The column is uniform in x1 and its ratios alternate 1, 0.1 along x2 (x3 in the 3-D case), while the
wind blows along that direction at dt = dx: the dry air leaves each cell twice and comes back from
upwind, but at the upwind cell's ratio. With phi taken from the x1 neighbours alone the rich cells
were relaxed fully and went negative after one stage (tracer -7.9e-04, vapour -8.0e-03 at rho_dry
0.981); main never limits past the content. EOS limiter on and off (the covariance path).
python test_ratio_drain_sideways.py [--device cuda]"""
import os, sys, tempfile, torch, yaml
from pathlib import Path
from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR, kIV1

DEV = "cuda" if "cuda" in sys.argv else "cpu"
NG = 2


def stage(nx3, limiter):  # min tracer and vapour over the cells with rho_dry > 0, after one stage
    card = yaml.safe_load(open(Path(__file__).resolve().parent / "test_flux_positivity_carry.yaml"))
    card["geometry"]["bounds"].update(x2max=4., x3max=4. if nx3 > 1 else 1.)
    card["geometry"]["cells"].update(nx2=4, nx3=nx3)
    card["boundary-condition"]["external"].update(
        {f"x{d}-{s}": "periodic" for d in (2, 3) for s in ("inner", "outer")})
    card["dynamics"]["equation-of-state"]["limiter"] = limiter
    card["scalar"] = {"nvar": 1, "names": ["tracer"], "reconstruct": {"type": "dc"}}
    with tempfile.TemporaryDirectory() as wd:
        p = os.path.join(wd, "card.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(card, f)
        block = MeshBlock(MeshBlockOptions.from_yaml(p))
    block.to(torch.device(DEV))
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    w[kIDN], w[kIPR] = 1., 1.e5
    w[kIV1 + (2 if nx3 > 1 else 1)] = 2.
    alt = torch.tensor([0.1, 1., 0.1, 1., 0.1, 1., 0.1, 1.], dtype=w.dtype, device=w.device)
    prof = alt.view(-1, 1, 1) if nx3 > 1 else alt.view(1, -1, 1)  # period 2, ghosts included
    w[kICY] = 0.01 * prof
    w[kICY + 1] = 0.
    v, _ = block.initialize({"hydro_w": w, "scalar_r": (1e-3 * prof).expand(w.shape[1:]).unsqueeze(0).clone()})
    block.forward(v, 1., 0)
    I = (slice(NG, -NG) if nx3 > 1 else slice(None), slice(NG, -NG), slice(NG, -NG))
    ok = v["hydro_u"][kIDN][I] > 0
    return float(v["scalar_s"][0][I][ok].min()), float(v["hydro_u"][kICY][I][ok].min())


if __name__ == "__main__":
    if DEV == "cuda" and not torch.cuda.is_available(): sys.exit(125)  # no visible GPU: skip
    bad = False
    for nx3 in (1, 4):
        for limiter in (False, True):
            s, y = stage(nx3, limiter)
            print(f"{DEV} {'x3' if nx3 > 1 else 'x2'} drain, eos limiter {'on ' if limiter else 'off'}: "
                  f"min tracer {s:+.3e}, min vapour {y:+.3e} (where rho_dry > 0)")
            bad |= s < 0. or y < 0.
    sys.exit(1 if bad else 0)
