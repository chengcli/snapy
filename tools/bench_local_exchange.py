"""Compare two Release builds of the same 64-block mesh exchange fixture.

For a base revision predating the fixture, copy tests/test_mesh_executor.cpp
from the candidate into the base checkout before building test_mesh_executor.release.
Run this script with the two CMake build directories, oldest first.
"""

import argparse
import os
import re
import statistics
import subprocess
from pathlib import Path


def measure(build, device):
    build = build.resolve()
    env = os.environ.copy()
    # Prefer this build's snap library when the environment supplies extra
    # dependency directories on Linux.
    search_path = [str(build / "lib")]
    if env.get("LD_LIBRARY_PATH"):
        search_path.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(search_path)
    result = subprocess.run(
        [
            "./test_mesh_executor.release",
            "--gtest_also_run_disabled_tests",
            f"--gtest_filter=*DISABLED_local_halo_latency/{device}_Float",
        ],
        cwd=build / "tests",
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    match = re.search(r"^Mean ms/exchange: ([\d.]+)$", result.stdout, re.M)
    if result.returncode or match is None:
        raise RuntimeError(f"Missing {device} timing in {build}\n{result.stdout}")
    return float(match[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--pairs", type=int, default=9)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")

    base, candidate = [], []
    revisions = [(args.base, base), (args.candidate, candidate)]
    for i in range(args.pairs):
        for build, samples in revisions:
            samples.append(measure(build, args.device))
        print(f"Pair {i + 1}: base {base[-1]:.4f}, candidate {candidate[-1]:.4f} ms",
              flush=True)
        revisions.reverse()
    before, after = statistics.median(base), statistics.median(candidate)
    wins = sum(b > c for b, c in zip(base, candidate))
    print(f"Median {before:.4f} -> {after:.4f} ms, "
          f"{before/after:.3f}x speedup, {wins}/{args.pairs} pairs faster")
    return int(after >= before)


if __name__ == "__main__":
    raise SystemExit(main())
