#!/usr/bin/env python3
"""Two ranks of test_sedimentation_cubed_seam on Gloo, and on UCX if asked.

The limiter exchanges more than one variable, and a Gloo send takes one
tensor; Linux CI defaults to UCX and would not reach Gloo. Both ranks must
exit 0 on Gloo, and with --compare-ucx its result line (17 digits) must be
identical to the UCX run's.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SKIP_CODE = 125
TAG = "sedimentation cubed seam:"


def run(torchrun, exe, backend):
    env = dict(os.environ, BACKEND=backend)
    p = subprocess.run(
        [torchrun, "--no-python", "--nproc-per-node=2", str(exe)],
        cwd=exe.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    lines = [l[l.index(TAG):] for l in p.stdout.splitlines() if TAG in l]
    print("%s: rc=%d %s" % (backend, p.returncode, lines[0] if lines else "(no result line)"))
    if p.returncode != 0:
        print(p.stdout[-4000:])
    return p.returncode, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--compare-ucx", action="store_true")
    args = ap.parse_args()

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        print("Skipping: torchrun not found")
        return SKIP_CODE
    exe = Path(args.exe).resolve()

    rc, gloo = run(torchrun, exe, "gloo")
    if rc != 0 or len(gloo) != 1:
        print("FAIL: the Gloo run did not finish with one result line")
        return 1
    if args.compare_ucx:
        rc, ucx = run(torchrun, exe, "ucx")
        if rc != 0 or ucx != gloo:
            print("FAIL: the UCX run differs from the Gloo run")
            return 1
    print("### cubed-seam Gloo run passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
