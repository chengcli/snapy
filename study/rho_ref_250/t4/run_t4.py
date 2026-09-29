#!/usr/bin/env python
"""Run the #250 T4 robustness matrix with t4_driver.release.

Groups (--only picks a comma list):
  anomaly   large anomalies: -40 K cold pool on a dry adiabat, a -30 K cold
            pool inside a strong inversion (+20 K/km), -20 K at a sharp
            tropopause; GPU
  positivity  tall isothermal columns (40 H, 200 H) with a cold anomaly,
            2000 steps, wb-rop-guard off and on; GPU
  fault     one corrupted cell (p = 0, p < 0, bottom p = 0, rho/p overflow),
            reference only, guard off and on; CPU and GPU
  stretch   rest columns (dT = 0), geometric x1 ratio q = 1, 1.02, 1.05, 1.1,
            plus a cold pool on q = 1.05; GPU
  seam      block seams and bit-reproducibility: one column, 1/2/4 ranks along
            x1 and x2 (Gloo, CPU), local blocks, and 1 vs 2 ranks on 2 GPUs
  cost      the anomaly case at 256 x 1024 on one GPU, with a timed loop of
            reference calls

Each run gets a directory under --out with its YAML, run.log and the
driver's summary/ref/state files.

  python run_t4.py --bin <build>/bin --out <rundir> [--only seam] [--dry]
"""
import argparse
import copy
import json
import os
import queue
import subprocess
import threading
import time

import yaml

FORMS = ["smooth5", "isentrope", "none", "local_polytrope"]
GRAV = 9.81
CP = 1004.5  # gammad 1.4, weight 28.9703e-3 -> Rd = 287.0
RD = 287.0
H250 = RD * 250. / GRAV  # isothermal scale height at 250 K, 7314 m

BASE = {
    "geometry": {
        "type": "cartesian",
        "bounds": {"x1min": 0., "x1max": 6.4e3, "x2min": 0., "x2max": 12.8e3,
                   "x3min": -0.5, "x3max": 0.5},
        "cells": {"nx1": 64, "nx2": 128, "nx3": 1, "nghost": 3},
    },
    "distribute": {"layout": "slab", "nb2": 1, "nb3": 1,
                   "blocks_per_process": 1},
    "dynamics": {
        "equation-of-state": {"type": "ideal-gas", "gammad": 1.4,
                              "weight": 28.9703e-3,
                              "density-floor": 1.e-300,
                              "pressure-floor": 1.e-300, "limiter": False},
        "reconstruct": {
            "vertical": {"type": "weno5", "scale": False, "shock": False},
            "horizontal": {"type": "weno5", "scale": False, "shock": False}},
        "riemann-solver": {"type": "lmars"},
        "wb-density-ref": "smooth5",
        "wb-rop-guard": False,
    },
    "boundary-condition": {"external": {
        "x1-inner": "reflecting", "x1-outer": "reflecting",
        "x2-inner": "periodic", "x2-outer": "periodic",
        "x3-inner": "periodic", "x3-outer": "periodic"}},
    "integration": {"type": "rk3", "cfl": 0.8, "implicit-scheme": 0,
                    "nlim": 200, "tlim": 1.e9, "ncycle_out": 100000},
    "forcing": {"const-gravity": {"grav1": -GRAV}},
    "problem": {"profile": "polytrope", "T0": 300., "p0": 1.e5,
                "lapse": GRAV / CP, "dT": 0., "xc": 6.4e3, "zc": 2.e3,
                "xr": 2.e3, "zr": 1.2e3},
}


def cfg(**kw):
    """BASE with dotted-key overrides, e.g. cfg(**{'problem.dT': -40})."""
    c = copy.deepcopy(BASE)
    for k, v in kw.items():
        node = c
        keys = k.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = v
    return c


def column(height, nx1, nx2=128, width=12.8e3):
    return {"geometry.bounds.x1max": height, "geometry.cells.nx1": nx1,
            "geometry.cells.nx2": nx2, "geometry.bounds.x2max": width,
            "problem.xc": width / 2}


ANOMALY = {  # name -> overrides; each is run for every form
    "coldpool40_adiabat": dict(**{"problem.dT": -40.}),
    "coldpool30_inversion": dict(**column(5.e3, 64), **{
        "problem.profile": "inversion", "problem.T0": 250.,
        "problem.lapse": -0.02, "problem.dT": -30., "problem.zc": 1.5e3}),
    "warm10_inversion": dict(**column(5.e3, 64), **{
        "problem.profile": "inversion", "problem.T0": 250.,
        "problem.lapse": -0.02, "problem.dT": 10., "problem.zc": 1.5e3}),
    "cold20_tropopause": dict(**column(30.e3, 96, 128, 40.e3), **{
        "problem.profile": "tropopause", "problem.T0": 300.,
        "problem.lapse": GRAV / CP, "problem.ztrop": 12.e3,
        "problem.dT": -20., "problem.zc": 12.e3, "problem.xr": 5.e3,
        "problem.zr": 3.e3}),
}


def jobs_anomaly():
    for name, ov in ANOMALY.items():
        for form in FORMS:
            c = cfg(**ov, **{"dynamics.wb-density-ref": form,
                             "integration.nlim": 2000})
            yield f"anomaly_{name}_{form}", c, "gpu", 1


def jobs_positivity():
    for nh in [40, 200]:
        for form in FORMS:
            for guard in [False, True]:
                c = cfg(**column(nh * H250, 128), **{
                    "problem.profile": "isothermal", "problem.T0": 250.,
                    "problem.dT": -10., "problem.zc": 2. * H250,
                    "problem.zr": H250, "problem.xr": 3.e3,
                    "dynamics.wb-density-ref": form,
                    "dynamics.wb-rop-guard": guard,
                    "integration.nlim": 2000})
                g = "on" if guard else "off"
                yield f"positivity_iso{nh}H_{form}_guard{g}", c, "gpu", 1


def jobs_fault():
    for fault in ["p_zero", "p_negative", "bottom_p_zero", "rho_overflow"]:
        for form in FORMS:
            for guard in [False, True]:
                for dev in ["cpu", "gpu"]:
                    c = cfg(**{"dynamics.wb-density-ref": form,
                               "dynamics.wb-rop-guard": guard,
                               "problem.fault": fault,
                               "integration.nlim": 0})
                    g = "on" if guard else "off"
                    yield f"fault_{fault}_{form}_guard{g}_{dev}", c, dev, 1


def jobs_stretch():
    profiles = {
        "isothermal": {"problem.profile": "isothermal", "problem.T0": 250.,
                       **column(4 * H250, 64)},
        "polytrope": {},
    }
    for prof, ov in profiles.items():
        for q in [1.0, 1.02, 1.05, 1.1]:
            for form in FORMS:
                c = cfg(**ov, **{"problem.stretch": q,
                                 "dynamics.wb-density-ref": form,
                                 "integration.nlim": 1000})
                yield f"stretch_rest_{prof}_q{q}_{form}", c, "gpu", 1
    for form in FORMS:
        c = cfg(**{"problem.stretch": 1.05, "problem.dT": -15.,
                   "dynamics.wb-density-ref": form, "integration.nlim": 1000})
        yield f"stretch_cold15_polytrope_q1.05_{form}", c, "gpu", 1


# (tag, nb1, nb2, nprocs, blocks_per_process)
LAYOUTS = [
    ("r1", 1, 1, 1, 1),
    ("x2r2", 1, 2, 2, 1),
    ("x2r4", 1, 4, 4, 1),
    ("x1r2", 2, 1, 2, 1),
    ("x1r4", 4, 1, 4, 1),
    ("x1loc2", 2, 1, 1, 2),
    ("x2loc2", 1, 2, 1, 2),
]
SEAM_CASES = {
    "cold15_polytrope": {"problem.dT": -15.},
    "cold20_tropopause": ANOMALY["cold20_tropopause"],
}


def jobs_seam():
    for case, ov in SEAM_CASES.items():
        for form in FORMS:
            for tag, nb1, nb2, np_, bpp in LAYOUTS:
                c = cfg(**ov, **{"dynamics.wb-density-ref": form,
                                 "integration.nlim": 200,
                                 "distribute.layout": "cubed",
                                 "distribute.nb1": nb1, "distribute.nb2": nb2,
                                 "distribute.blocks_per_process": bpp})
                yield f"seam_{case}_{form}_{tag}_cpu", c, "cpu", np_
            for tag, nb1, nb2 in [("r1", 1, 1), ("x2r2", 1, 2),
                                  ("x1r2", 2, 1)]:
                c = cfg(**ov, **{"dynamics.wb-density-ref": form,
                                 "integration.nlim": 200,
                                 "distribute.layout": "cubed",
                                 "distribute.nb1": nb1, "distribute.nb2": nb2})
                np_ = nb1 * nb2
                yield f"seam_{case}_{form}_{tag}_gpu", c, \
                    "gpu2" if np_ == 2 else "gpu", np_


def jobs_cost():
    for form in FORMS:
        c = cfg(**{**column(6.4e3, 256, 1024, 25.6e3),
            "problem.dT": -15.,
            "dynamics.wb-density-ref": form, "integration.nlim": 500,
            "problem.ref_bench": 200})
        yield f"cost_cold15_256x1024_{form}", c, "gpu", 1


GROUPS = {"anomaly": jobs_anomaly, "positivity": jobs_positivity,
          "fault": jobs_fault, "stretch": jobs_stretch, "seam": jobs_seam,
          "cost": jobs_cost}


def execute(job, slot, args):
    name, c, dev, np_ = job
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "timing.json")) and not args.force:
        return "cached"
    with open(os.path.join(d, "input.yaml"), "w") as f:
        yaml.safe_dump(c, f, sort_keys=False)
    env = dict(os.environ)
    # Gloo carries CPU tensors only; two ranks on two GPUs need UCX
    env["BACKEND"] = "ucx" if dev == "gpu2" else "gloo"
    if dev == "cpu":
        env["DEVICE"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif dev == "gpu2":
        env["DEVICE"] = "cuda"
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
    else:
        env["DEVICE"] = "cuda"
        env["CUDA_VISIBLE_DEVICES"] = slot.split(":")[1]
    exe = os.path.join(args.bin, "t4_driver.release")
    cmd = [exe, "input.yaml"]
    if np_ > 1:
        cmd = ["torchrun", "--standalone", f"--nproc-per-node={np_}",
               "--no-python"] + cmd
    t0 = time.time()
    with open(os.path.join(d, "run.log"), "w") as log:
        try:
            rc = subprocess.call(cmd, cwd=d, env=env, stdout=log,
                                 stderr=subprocess.STDOUT,
                                 timeout=args.timeout)
        except subprocess.TimeoutExpired:
            rc = "timeout"
    wall = time.time() - t0
    with open(os.path.join(d, "timing.json"), "w") as f:
        json.dump({"name": name, "slot": slot, "device": dev, "nprocs": np_,
                   "rc": rc, "wall_s": wall}, f, indent=1)
    return f"rc={rc} {wall:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=",".join(GROUPS))
    ap.add_argument("--gpu-slots", default="cuda:0,cuda:1")
    ap.add_argument("--cpu-slots", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    jobs = [j for g in args.only.split(",") for j in GROUPS[g]()]
    if args.dry:
        for j in jobs:
            print(j[0], j[2], j[3])
        print(len(jobs), "jobs")
        return
    # gpu2 jobs hold both GPUs, so they run after the single-GPU queue
    qs = {k: queue.Queue() for k in ["gpu", "cpu", "gpu2"]}
    for j in jobs:
        qs[j[2]].put(j)
    lock = threading.Lock()

    def worker(slot, kinds):
        for kind in kinds:
            while True:
                try:
                    job = qs[kind].get_nowait()
                except queue.Empty:
                    break
                status = execute(job, slot, args)
                with lock:
                    print(f"{time.strftime('%H:%M:%S')} {slot:7s} {job[0]} "
                          f"{status}", flush=True)

    gpu_slots = [s for s in args.gpu_slots.split(",") if s]
    threads = [threading.Thread(target=worker, args=(s, ["gpu"]))
               for s in gpu_slots]
    threads += [threading.Thread(target=worker, args=(f"cpu{i}", ["cpu"]))
                for i in range(args.cpu_slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if gpu_slots:  # an empty --gpu-slots runs the CPU jobs only
        worker("gpu0+1", ["gpu2"])


if __name__ == "__main__":
    main()
