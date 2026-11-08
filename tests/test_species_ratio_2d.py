#!/usr/bin/env python3
"""A uniform hydro vapour mass fraction q stays uniform on nx2 = 4 identical columns.
The deck of test_scalar_ratio_2d.py with q = 1e-4 (unsaturated): with nx2 > 1 the covariance rows
switch the species positivity limiter (step 4.C) on with the eos limiter off, and where the
unlimited mass flux drains a cell's dry air past zero it scaled the vapour flux alone. Positivity:
the one-cell and one-hole vapour columns of test_scalar_ratio_2d.py.
python test_species_ratio_2d.py [--device cuda] [--scheme 0]"""
import sys, torch
from snapy import kICY, kIDN
import test_scalar_ratio_2d as S

Q0 = 1e-4


def run(nx2, nx3=1):  # max|q/q0 - 1| over one rk3 step
    block, v, dt = S.setup(nx2, S.SCHEME, Q0, nx3=nx3)
    J, dev = slice(3, -3) if nx2 > 1 else slice(None), 0.
    K = slice(3, -3) if nx3 > 1 else slice(None)
    for stage in range(3):
        block.forward(v, dt, stage)
        u = v["hydro_u"][:, K, J, 3:-3]
        q = u[kICY] / (u[kIDN] + u[kICY:].sum(0))
        dev = max(dev, float((q / Q0 - 1).abs().max()))
    return dev


if __name__ == "__main__":
    if S.DEV == "cuda" and not torch.cuda.is_available(): sys.exit(125)  # no visible GPU: skip
    bad = S.positivity(True, Q0)  # 45fe81e1: -4.359e-06, -3.702e-06; main 68f4e066: 0, 9.00e-07
    for nx2, nx3 in ((4, 1), (4, 4), (1, 1)):
        dev = run(nx2, nx3)
        print(f"{S.DEV} scheme {S.SCHEME} nx2={nx2} nx3={nx3}: max|q/q0-1| = {dev:.3e}")
        bad |= dev > 1e-12
    sys.exit(1 if bad else 0)
