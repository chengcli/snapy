#!/usr/bin/env python
"""Score the #250 T3 runs against the published reference values.

  python analyze_t3.py <rundir> [--md out.md] [--json out.json]

Reference values (the papers' own numbers, not our high-resolution runs):
  Straka et al. (1993), Int. J. Numer. Methods Fluids 17, 1-22, REFC 25 m
  reference: theta'_min -9.77 K, u 36.46 / -15.19 m/s, w 12.93 / -15.95 m/s
  (Table V), front location 15537.44 m (Table IV, REFC25).
  Bryan & Fritsch (2002), Mon. Wea. Rev. 130, 2917-2928, Fig. 3 (moist, 100 m,
  t = 1000 s): theta_e' 4.09521 / -0.305695 K, w 15.713 / -9.92698 m/s.
  BF02 publishes no bubble-top height or theta_rho' extrema; those are
  reported without a reference.
"""
import argparse
import glob
import json
import os

import netCDF4
import numpy as np

REF = {
    "straka": {
        "theta_p_min": -9.77,
        "u_max": 36.46,
        "u_min": -15.19,
        "w_max": 12.93,
        "w_min": -15.95,
        "front_m": 15537.44,
    },
    "bryan": {
        "theta_e_p_max": 4.09521,
        "theta_e_p_min": -0.305695,
        "w_max": 15.713,
        "w_min": -9.92698,
    },
}
RD, CPD, P0 = 287.0, 1004.5, 1.0e5  # bryan.cpp: kRd, kGamma/(kGamma-1) kRd


def frames(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.out*.nc"))):
        ds = netCDF4.Dataset(f)
        v = {k: np.array(ds[k][:]) for k in ds.variables}
        ds.close()
        # (time, x1, x3, x2) -> (x1, x2)
        for k in list(v):
            if v[k].ndim == 4:
                v[k] = v[k][0, :, 0, :]
        v["time"] = float(v["time"][0])
        out.append(v)
    return out


def theta_rho(v):
    return v["press"] / (v["rho"] * RD) * (P0 / v["press"]) ** (RD / CPD)


def front_location(thp_row, x):
    """largest x at which the lowest-level theta' crosses -1 K"""
    idx = np.where(thp_row <= -1.0)[0]
    if idx.size == 0:
        return np.nan
    k = idx[-1]
    if k + 1 >= x.size:
        return x[k]
    a, b = thp_row[k], thp_row[k + 1]
    return x[k] + (x[k + 1] - x[k]) * (-1.0 - a) / (b - a)


def top_height(field, z, level):
    """highest z at which any column still reaches `level`"""
    above = np.where((field >= level).any(axis=1))[0]
    if above.size == 0:
        return np.nan
    k = above[-1]
    if k + 1 >= z.size:
        return z[k]
    col = np.argmax(field[k])
    a, b = field[k, col], field[k + 1, col]
    return z[k] + (z[k + 1] - z[k]) * (level - a) / (b - a)


def score(d, info):
    fr = frames(d)
    v0, v1 = fr[0], fr[-1]
    case, rest = info["case"], info["rest"]
    m = {"t_end": v1["time"]}
    w, u = v1["vel1"], v1["vel2"]
    if case == "straka":
        thp = v1["theta"] - 300.0
        if rest:
            m["max|w|"] = np.abs(w).max()
            m["max|u|"] = np.abs(u).max()
            m["max|dtheta|"] = np.abs(v1["theta"] - v0["theta"]).max()
        else:
            m["theta_p_min"] = thp.min()
            m["theta_p_max"] = thp.max()
            m["u_max"], m["u_min"] = u.max(), u.min()
            m["w_max"], m["w_min"] = w.max(), w.min()
            m["front_m"] = front_location(thp[0], v1["x2"])
    else:
        # base state = the far-field (x = 0 edge) column at t = 0
        the0 = v0["theta_e"][:, :1]
        thr0 = theta_rho(v0)[:, :1]
        if rest:
            m["max|w|"] = np.abs(w).max()
            m["max|u|"] = np.abs(u).max()
            m["max|dtheta_e|"] = np.abs(v1["theta_e"] - v0["theta_e"]).max()
            m["max|dtheta_rho|"] = np.abs(theta_rho(v1) - theta_rho(v0)).max()
            m["theta_e_base_sfc"] = v0["theta_e"][0, 0]
            m["theta_e_base_span"] = np.ptp(v0["theta_e"][:, 0])
        else:
            tep = v1["theta_e"] - the0
            trp = theta_rho(v1) - thr0
            m["theta_e_p_max"], m["theta_e_p_min"] = tep.max(), tep.min()
            m["theta_rho_p_max"], m["theta_rho_p_min"] = trp.max(), trp.min()
            m["w_max"], m["w_min"] = w.max(), w.min()
            m["top_m(theta_e'=1K)"] = top_height(tep, v1["x1"], 1.0)
    return {k: float(x) for k, x in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    rows = []
    for tj in sorted(glob.glob(os.path.join(args.rundir, "*", "timing.json"))):
        info = json.load(open(tj))
        d = os.path.dirname(tj)
        if info["rc"] != 0:
            rows.append({**info, "metrics": None})
            continue
        rows.append({**info, "metrics": score(d, info)})
    if args.json:
        json.dump({"ref": REF, "rows": rows}, open(args.json, "w"), indent=1)
    for r in rows:
        print(r["name"], f"{r['wall_s']:.0f}s", r["metrics"])


if __name__ == "__main__":
    main()
