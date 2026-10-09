#!/usr/bin/env python3
"""#289 face-flux covariance (switch SNAP_FLUX_COVARIANCE): conservation across cubed-sphere seams, and the
per-direction gating of the p* geometric source.

  1. conservation: six cubed-sphere panels in one process, closed x1 walls, moist (vapor/cloud), the term on,
     a non-rest state (a sheared solid-body wind and a vapor field varying in r and in the horizontal). Every
     correction is added to the face flux, so total mass, vapor mass and E+PE = sum (E + rho g x1) dV must
     hold to round-off over NSTEP steps: a seam face whose two panels disagree on the correction would leak.
  2. gating: a discretely balanced spherical-polar shell at rest with the x3 flux disabled (nx3 > 1) and the
     term on stays at rest: the x2 faces carry p* in their flux, so the x2 geometric source must carry p*
     too, even though the x3 faces are not corrected.

The switch is read once per process, so each arm runs in its own process (as in test_flux_covariance_rows).

  python test_flux_covariance_seams.py [--device cpu]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_flux_covariance_rows as rows  # noqa: E402  (deck, builders, stepping)

NSTEP = 10
DRIFT_TOL = 1e-12     # relative drift of each total over NSTEP steps
REST_TOL = 1e-9       # max |v| / c_s after rows.NREST steps


def totals(blocks, mv):
    from snapy import kIDN, kIPR
    icy = kIPR + 1
    out = torch.zeros(3, dtype=torch.float64)
    for ib, b in enumerate(blocks):
        sl = b.part((0, 0, 0), False)[1:]
        vol = b.module("coord").cell_volume()[sl]
        u = mv[ib]["hydro_u"]
        k, j, i = sl
        x1v = b.buffer("coord.x1v")[i]
        mass = u[kIDN][sl] + u[icy:][(slice(None),) + tuple(sl)].sum(0)
        out += torch.stack([(mass * vol).sum(),
                            (u[icy][sl] * vol).sum(),
                            ((u[kIPR][sl] + rows.G * x1v * mass) * vol).sum()]).cpu()
    return out


def arm_conservation(device):
    import snapy
    from snapy import kIPR
    cfg = rows.card("gnomonic-equiangle", moist=True)
    mesh, blocks = rows.blocks_of(cfg, device)
    nx1 = cfg["geometry"]["cells"]["nx1"]
    r0 = float(cfg["geometry"]["bounds"]["x1min"])
    depth = float(cfg["geometry"]["bounds"]["x1max"]) - r0
    mv = []
    for ib, b in enumerate(blocks):
        w = rows.balanced_w(b, nx1)
        beta, alpha = torch.meshgrid(b.buffer("coord.x3v").cpu(), b.buffer("coord.x2v").cpu(),
                                     indexing="ij")
        face_id = int(b.get_layout().loc_of(ib)[2])
        lon, lat = snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face_id), alpha, beta)
        r = b.buffer("coord.x1v").cpu()[None, None, :]
        zeta = (r - r0) / depth
        # vapor varying in r and in the horizontal, far below saturation; cloud stays 0
        q = 0.01 * (1.0 + 0.5 * zeta) * (1.0 + 0.2 * torch.sin(lon) * torch.cos(lat)).unsqueeze(-1)
        w[kIPR + 1] = q.to(w)
        # sheared solid-body wind: u_n varies along x1, so the covariance rows are non-zero
        for kk in range(w.shape[-1]):
            vel = torch.zeros((3,) + tuple(alpha.shape), dtype=torch.float64)
            vel[2] = 30.0 * torch.cos(lat) * (1.0 + float(zeta[0, 0, min(max(kk - rows.NG, 0), nx1 - 1)]))
            snapy.coord.cs_sph_to_contra_(vel, alpha, beta, face_id)
            for c in range(3):
                w[1 + c][..., kk] = vel[c].to(w)
        mv.append({"hydro_w": w})
    mv, _ = mesh.initialize(mv)
    t0 = totals(blocks, mv)
    rows.advance(mesh, blocks, mv, NSTEP)
    t1 = totals(blocks, mv)
    drift = ((t1 - t0) / t0).abs().tolist()
    vmax = max(float(w[1:4].abs().max()) for w in rows.end_prims(blocks, mv))
    return {"mass": drift[0], "vapor": drift[1], "e_plus_pe": drift[2], "vmax": vmax}


def arm_gating(device):
    cfg = rows.card("spherical-polar")
    cfg["dynamics"]["disable-flux-x3"] = True
    mesh, blocks = rows.blocks_of(cfg, device)
    nx1 = cfg["geometry"]["cells"]["nx1"]
    v, _ = blocks[0].initialize({"hydro_w": rows.balanced_w(blocks[0], nx1)})
    mv = [v]
    rows.advance(None, blocks, mv, rows.NREST)
    cs = math.sqrt(1.4 * rows.P0 / rows.RHO0)
    return max(float(w[1:4].abs().max()) for w in rows.end_prims(blocks, mv)) / cs


def run(task, switch, device, tmpdir):
    env = dict(os.environ)
    env.pop("SNAP_FLUX_COVARIANCE", None)
    if switch is not None:
        env["SNAP_FLUX_COVARIANCE"] = switch
    out = os.path.join(tmpdir, f"{task}_{switch}.json")
    subprocess.run([sys.executable, __file__, "--child", task, "--out", out, "--device", device],
                   env=env, check=True)
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--child")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.child:
        res = arm_conservation(a.device) if a.child == "conservation" else arm_gating(a.device)
        json.dump(res, open(a.out, "w"))
        return 0
    failures = []
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
        for sw in (None, "1"):
            c = run("conservation", sw, a.device, tmp)
            print(f"six panels, closed walls, {NSTEP} steps, switch {sw or 'unset'}: relative drift "
                  f"mass {c['mass']:.3e}  vapor {c['vapor']:.3e}  E+PE {c['e_plus_pe']:.3e}  "
                  f"(max|v| {c['vmax']:.2f})", flush=True)
            if sw == "1":
                for k in ("mass", "vapor", "e_plus_pe"):
                    if not c[k] < DRIFT_TOL:
                        failures.append(f"{k} drift {c[k]:.3e} >= {DRIFT_TOL} with the term on")
        off = run("gating", None, a.device, tmp)
        on = run("gating", "1", a.device, tmp)
        print(f"spherical-polar rest, x3 flux disabled (nx3 > 1), {rows.NREST} steps: max|v|/c_s "
              f"off {off:.3e}  on {on:.3e}", flush=True)
        if not on < REST_TOL:
            failures.append(f"rest with x3 disabled: on {on:.3e} >= {REST_TOL}")
    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("### flux covariance seams test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
