#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from netCDF4 import Dataset
import numpy as np


SKIP_CODE = 125
FIELDS = ("rho", "press", "temp", "theta", "vel1", "vel2", "vel3")


def skip(msg: str) -> int:
    print(f"Skipping test_straka_nccl_decomp: {msg}")
    return SKIP_CODE


def run(cmd, cwd: Path, env=None) -> None:
    print(f"+ (cd {cwd} && {' '.join(cmd)})")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def prepare_case(case_dir: Path, yaml_src: Path) -> None:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    target = case_dir / "straka.yaml"
    try:
        target.symlink_to(yaml_src)
    except OSError:
        shutil.copy2(yaml_src, target)


def compare_pair(path_a: Path, path_b: Path, label: str) -> None:
    with Dataset(path_a, "r") as data_a, Dataset(path_b, "r") as data_b:
        for field in FIELDS:
            arr_a = np.asarray(data_a[field][:])
            arr_b = np.asarray(data_b[field][:])
            if arr_a.shape != arr_b.shape:
                raise ValueError(
                    f"{label}: field {field} shape mismatch {arr_a.shape} != {arr_b.shape}"
                )

            diff = np.abs(arr_a - arr_b)
            max_abs = float(diff.max(initial=0.0))
            rms = float(np.sqrt(np.mean(diff * diff)))
            print(f"{label}: {field} max_abs={max_abs} rms={rms}")
            if max_abs != 0.0:
                raise ValueError(
                    f"{label}: field {field} differs, max_abs={max_abs}, rms={rms}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--build-type", required=True)
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        return skip(f"torch import failed: {exc}")

    if not torch.cuda.is_available():
        return skip("CUDA runtime is unavailable")

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        return skip(f"need at least 2 GPUs, found {gpu_count}")

    build_dir = Path(args.build_dir).resolve()
    tests_dir = build_dir / "tests"
    bin_dir = build_dir / "bin"
    exe = bin_dir / f"straka.{args.build_type}"
    if not exe.exists():
        return skip(f"missing executable {exe}")

    pd_run = shutil.which("pd-run")
    if pd_run is None:
        return skip("pd-run not found in PATH")

    cases = (
        ("single", 1, "0", Path(os.abspath(bin_dir / "straka_gpu_single.yaml"))),
        ("mesh2", 1, "0", Path(os.abspath(bin_dir / "straka_gpu_mesh2.yaml"))),
        ("proc2", 2, "0,1", Path(os.abspath(bin_dir / "straka_gpu_proc2.yaml"))),
    )

    outputs = {}
    for name, ranks, visible_devices, yaml_src in cases:
        if not yaml_src.exists():
            return skip(f"missing input file {yaml_src}")

        case_dir = tests_dir / f"straka_nccl_{name}"
        prepare_case(case_dir, yaml_src)

        env = os.environ.copy()
        env["BACKEND"] = "nccl"
        env["CUDA_VISIBLE_DEVICES"] = visible_devices

        run([pd_run, str(ranks), str(exe), "straka.yaml"], cwd=case_dir, env=env)

        output = case_dir / "straka.out0.00001.nc"
        if not output.exists():
            raise FileNotFoundError(f"missing output file {output}")
        outputs[name] = output

    compare_pair(outputs["single"], outputs["mesh2"], "single vs mesh2")
    compare_pair(outputs["single"], outputs["proc2"], "single vs proc2")
    compare_pair(outputs["mesh2"], outputs["proc2"], "mesh2 vs proc2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
