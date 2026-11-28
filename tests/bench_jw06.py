"""Compare nine paired full RK3 timings on one device. Run once per GPU host."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def sample(args, mode):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "examples/jw06.py"),
        "--device",
        args.device,
        "--mode",
        mode,
        "--resolution",
        str(args.resolution),
        "--levels",
        str(args.levels),
        "--warmup",
        "10",
        "--steps",
        "100",
        "--execution",
        args.execution,
        "--dt",
        str(args.dt),
    ]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
    (record,) = (json.loads(line) for line in result.stdout.splitlines() if line.startswith("{"))
    return record["wall_seconds"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--execution", choices=("eager", "cuda-graph"), default="cuda-graph")
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--levels", type=int, default=26)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--baseline", choices=("fp64", "split64"), default="fp64")
    args = parser.parse_args()
    pairs = []
    for i in range(9):
        modes = [args.baseline, "fp32"]
        pair = {mode: sample(args, mode) for mode in (modes if i % 2 == 0 else reversed(modes))}
        pairs.append(pair)
        print(json.dumps(pair), flush=True)
    wins = sum(p["fp32"] < p[args.baseline] for p in pairs)
    median = {mode: statistics.median(p[mode] for p in pairs) for mode in modes}
    print(json.dumps(dict(wins=wins, median_seconds=median)))
    if args.device == "cuda" and (wins < 7 or median["fp32"] >= median[args.baseline]):
        sys.exit("GPU speed gate failed")
