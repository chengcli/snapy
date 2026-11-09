#!/usr/bin/env python3
"""A cell that settling drains keeps the plain species theta with the eos limiter on AND off.
The relative rule relaxes a species' theta toward the dry air's where the dry air is over-drained,
because the implicit refill repays dry air; nothing repays settling mass. Eight cells along x1, nx2 = 4
(the covariance rows switch the step 4.C limiter on with the eos limiter off), dt = dx, the column
rises at 2 m/s (the dry air leaves each cell twice) while the cloud settles at 1 m/s and falls off
0.9 per cell upward. Every cell above the bottom one is drained by settling: it must not export
more cloud than it holds, and its update must stay non-negative.
python test_settling_theta_both_paths.py [--device cuda]"""
import os, sys, tempfile, torch, yaml
from pathlib import Path
from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR, kIV1

DEV = "cuda" if "cuda" in sys.argv else "cpu"
NX1, NG = 8, 2


def run(limiter):  # worst cloud export past content over the settling-drained cells, min update
    card = yaml.safe_load(open(Path(__file__).resolve().parent / "test_flux_positivity_carry.yaml"))
    card["geometry"]["bounds"].update(x1max=float(NX1), x2max=4.)
    card["geometry"]["cells"].update(nx1=NX1, nx2=4)
    card["boundary-condition"]["external"].update({"x2-inner": "periodic", "x2-outer": "periodic"})
    card["dynamics"]["equation-of-state"]["limiter"] = limiter
    card["sedimentation"] = {"radius": {}, "density": {}, "const-vsed": {"cloud": -1.}}
    with tempfile.TemporaryDirectory() as wd:
        p = os.path.join(wd, "card.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(card, f)
        block = MeshBlock(MeshBlockOptions.from_yaml(p))
    block.to(torch.device(DEV))
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    w[kIDN], w[kIPR], w[kIV1], w[kICY] = 1., 1.e5, 2., 0.01
    for c in range(NX1):
        w[kICY + 1][..., NG + c] = 0.02 * 0.9 ** c
    v, _ = block.initialize({"hydro_w": w})
    u = v["hydro_u"].clone()
    du = block.module("hydro").forward(1., v["hydro_u"], v)  # dt = dx = dx2 = 1
    B = dict(block.named_buffers())
    F1, F2 = B["hydro.F1"][kICY + 1], B["hydro.F2"][kICY + 1]
    I = slice(NG + 1, NG + NX1)  # every cell above the bottom one: settling leaves its lower face
    J = slice(NG, -NG)
    out = (F1[..., NG + 2:NG + NX1 + 1].clamp_min(0.) + (-F1[..., I]).clamp_min(0.))[:, J]
    out = out + (F2[:, NG + 1:-NG + 1 or None, I].clamp_min(0.) + (-F2[:, J, I]).clamp_min(0.))
    held = u[kICY + 1][:, J, I]
    return (float((out - held).max()), float((u + du)[kICY + 1][:, J, I].min()),
            float(F1[:, J, NG:NG + NX1 + 1].sum()))


if __name__ == "__main__":
    if DEV == "cuda" and not torch.cuda.is_available(): sys.exit(125)  # no visible GPU: skip
    bad = False
    for limiter in (False, True):  # 629a38501: limiter off exports 1.6e-03 past content (cell 1)
        excess, low, flux = run(limiter)
        print(f"{DEV} eos limiter {'on ' if limiter else 'off'}: max(cloud export - content) = "
              f"{excess:+.3e}, min cloud update = {low:+.3e}, cloud x1 flux sum = {flux:.17g}")
        bad |= excess > 1e-12 * 0.02 or low < 0.
    sys.exit(1 if bad else 0)
