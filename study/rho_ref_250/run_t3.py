#!/usr/bin/env python
"""Run the #250 T3 matrix: {straka, bryan} x resolution x rho_ref form.

Each run gets its own directory under --out with the generated YAML, the
solver log and a timing.json. Runs are dispatched to a small pool of workers,
one per listed device slot (cuda:0, cuda:1, cpu, ...).

  python run_t3.py --bin <build>/bin --out <rundir> [--only straka] [--dry]
"""
import argparse
import itertools
import json
import os
import queue
import re
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FORMS = ["smooth5", "isentrope", "none"]

# (case, dx [m]) -> (nx1, nx2); Straka is the half domain 25.6 x 6.4 km,
# Bryan & Fritsch the full 20 x 10 km
GRIDS = {
    ("straka", 100): (64, 256),
    ("straka", 50): (128, 512),
    ("straka", 25): (256, 1024),
    ("bryan", 200): (50, 100),
    ("bryan", 100): (100, 200),
    ("bryan", 50): (200, 400),
}
FINEST = {("straka", 25), ("bryan", 50)}  # bubble only, no rest run

# The CUDA WENO5 stencil launches one thread block per line with one thread
# per cell, so a meshblock may hold at most 1024 cells (ghosts included) along
# any direction. Wider grids are split into two x2 blocks in one process.
MAX_LINE = 1024


def make_yaml(case, dx, form, rest):
    nx1, nx2 = GRIDS[(case, dx)]
    text = open(os.path.join(HERE, f"{case}.yaml")).read()
    text, n = re.subn(r"cells: \{nx1: \d+, nx2: \d+,",
                      f"cells: {{nx1: {nx1}, nx2: {nx2},", text)
    assert n == 1
    text, n = re.subn(r"wb-density-ref: \w+", f"wb-density-ref: {form}", text)
    assert n == 1
    if nx2 + 6 > MAX_LINE:
        text, n = re.subn(r"\n  nb2: 1\n", "\n  nb2: 2\n", text)
        assert n == 1
        text, n = re.subn(r"blocks_per_process: 1", "blocks_per_process: 2",
                          text)
        assert n == 1
    if rest:  # the unperturbed background: the initial-state residual
        text, n = re.subn(r"\n  dT: [-0-9.e]+", "\n  dT: 0.", text)
        assert n == 1
    return text


def run_name(case, dx, form, rest, device):
    tag = "rest" if rest else "bubble"
    dev = "cpu" if device == "cpu" else "gpu"
    return f"{case}_{tag}_dx{dx}_{form}_{dev}"


def execute(job, slot, args):
    case, dx, form, rest, device = job
    name = run_name(*job)
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "timing.json")) and not args.force:
        return name, "cached"
    with open(os.path.join(d, f"{name}.yaml"), "w") as f:
        f.write(make_yaml(case, dx, form, rest))
    env = dict(os.environ)
    if slot == "cpu":
        env["DEVICE"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["DEVICE"] = "cuda"
        env["CUDA_VISIBLE_DEVICES"] = slot.split(":")[1]
    t0 = time.time()
    with open(os.path.join(d, "run.log"), "w") as log:
        exe = "straka_t3" if case == "straka" else case  # see straka_t3.cpp
        rc = subprocess.call([os.path.join(args.bin, f"{exe}.release"),
                              f"{name}.yaml"], cwd=d, env=env, stdout=log,
                             stderr=subprocess.STDOUT)
    wall = time.time() - t0
    info = {"name": name, "case": case, "dx": dx, "form": form, "rest": rest,
            "device": slot, "rc": rc, "wall_s": wall}
    logtxt = open(os.path.join(d, "run.log")).read()
    m = re.findall(r"cycle=(\d+)", logtxt)
    info["cycles_logged"] = int(m[-1]) if m else None
    m = re.search(r"million cells-per-cycle = ([0-9.e+]+)", logtxt)
    info["mcells_per_cycle"] = float(m.group(1)) if m else None
    with open(os.path.join(d, "timing.json"), "w") as f:
        json.dump(info, f, indent=1)
    return name, f"rc={rc} {wall:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slots", default="cuda:0,cuda:1")
    ap.add_argument("--only", default="")
    ap.add_argument("--cpu-spot", action="store_true",
                    help="also run the coarsest grid of each case on the CPU")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    jobs = []
    for (case, dx), rest, form in itertools.product(GRIDS, [False, True],
                                                    FORMS):
        if args.only and case not in args.only.split(","):
            continue
        if rest and (case, dx) in FINEST:  # the rest residual is form-blind
            continue
        jobs.append((case, dx, form, rest, "gpu"))
    if args.cpu_spot:
        for case, dx in [("straka", 100), ("bryan", 200)]:
            for form in FORMS:
                jobs.append((case, dx, form, False, "cpu"))
    # cheapest first so a first table exists early
    cost = {k: v[0] * v[1] ** 2 for k, v in GRIDS.items()}
    jobs.sort(key=lambda j: (j[4] == "cpu", cost[(j[0], j[1])]))
    if args.dry:
        for j in jobs:
            print(run_name(*j))
        return

    queues = {"gpu": queue.Queue(), "cpu": queue.Queue()}
    for j in jobs:
        queues[j[4]].put(j)
    slots = args.slots.split(",")
    if queues["cpu"].qsize() and "cpu" not in slots:
        slots.append("cpu")
    lock = threading.Lock()

    def worker(slot):
        q = queues["cpu" if slot == "cpu" else "gpu"]
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            name, status = execute(job, slot, args)
            with lock:
                print(f"{time.strftime('%H:%M:%S')} {slot:7s} {name} {status}",
                      flush=True)

    threads = [threading.Thread(target=worker, args=(s,)) for s in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
