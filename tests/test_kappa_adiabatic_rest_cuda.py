#!/usr/bin/env python3
"""test_kappa_adiabatic_rest on CUDA; exits 125 (skipped) without a GPU."""
import sys

import test_kappa_adiabatic_rest

if __name__ == "__main__":
    sys.exit(test_kappa_adiabatic_rest.main(["--device", "cuda"]))
