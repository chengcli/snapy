#!/usr/bin/env python3
"""One unrepairable column among many must still raise from eos.compute("U->W").

The CPU failure counter is shared by the threads of a parallel loop. With a
non-atomic counter a thread that found no failure can store back a stale zero
over another thread's count, so a failure is reported as success.

  python test_fix_vapor_counts_every_column.py
"""
import os
import re
import sys
import tempfile

import torch

NX1, NX2, NX3 = 8, 128, 128
NPOS = 32  # broken-column positions, one per compute call


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "test_fix_vapor_reports_failure.yaml")) as f:
        text = f.read()
    text, n = re.subn(r"cells: \{[^}]*\}",
                      "cells: {nx1: %d, nx2: %d, nx3: %d, nghost: 3}" % (NX1, NX2, NX3), text)
    assert n == 1, "cells line not found"

    from snapy import MeshBlock, MeshBlockOptions, kICY

    # importing snapy sets one thread; the race needs several
    torch.set_num_threads(max(os.cpu_count() or 1, 4))
    if torch.get_num_threads() < 2:
        print("FAIL: need at least 2 threads, have %d" % torch.get_num_threads())
        return 1

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(f.name))
    finally:
        os.remove(f.name)
    eos = next(m for _, m in block.named_modules() if type(m).__name__ == "IdealMoist")

    w = dict(block.named_buffers())["hydro.D"].clone().zero_()  # (nvar, nc3, nc2, nc1)
    w[0] = 1.0
    w[4] = 1.0e5
    w[kICY] = 1.0e-3
    u = eos.compute("W->U", [w])
    eos.compute("U->W", [u.clone()])  # a benign state must pass

    ng = (u.size(3) - NX1) // 2
    missed = []
    for p in range(NPOS):
        c = p * (NX2 * NX3 - 1) // (NPOS - 1)  # spread over every thread's chunk
        k, j = ng + c // NX2, ng + c % NX2
        u_bad = u.clone()
        u_bad[kICY, k, j, :] = -1.0e-6
        try:
            eos.compute("U->W", [u_bad])
            missed.append((k, j))
        except RuntimeError as e:
            assert "Failed to fix vapor" in str(e), str(e)

    print("threads=%d columns=%d broken positions=%d missed=%d"
          % (torch.get_num_threads(), NX2 * NX3, NPOS, len(missed)))
    if missed:
        print("FAIL: unrepairable column NOT reported at", missed[:8])
        return 1
    print("PASS: every broken column reported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
