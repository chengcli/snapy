#!/usr/bin/env python3
"""test_check_redo_saturation on CUDA; exits 125 (skipped) without a GPU."""
import sys

import test_check_redo_saturation

if __name__ == "__main__":
    sys.exit(test_check_redo_saturation.main(["--device", "cuda"]))
