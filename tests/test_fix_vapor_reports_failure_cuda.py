#!/usr/bin/env python3
"""test_fix_vapor_reports_failure on CUDA; exits 125 (skipped) without a GPU."""
import sys

import test_fix_vapor_reports_failure

if __name__ == "__main__":
    sys.exit(test_fix_vapor_reports_failure.main(["--device", "cuda"]))
