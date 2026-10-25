"""Run full six-panel compressible onset brackets at specified radius ratios.

Profiles come from the independent spherical eigenproblem. The evolving state
is neither projected nor filtered. Existing completed runs with matching
parameters are reused unless --overwrite is given; incomplete runs fail loudly.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from spherical_compressible_reference import export_profile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kill_owned_process_tree(pid):
    # torchrun gives its workers separate sessions, so killing only the
    # launcher's process group would leave GPU workers alive after a timeout.
    descendants = set()

    def visit(parent):
        for children in Path(f"/proc/{parent}/task").glob("*/children"):
            try:
                found = map(int, children.read_text().split())
                for child in found:
                    if child not in descendants:
                        descendants.add(child)
                        visit(child)
            except (FileNotFoundError, ProcessLookupError):
                pass

    visit(pid)
    for owned in [*descendants, pid]:
        try:
            os.kill(owned, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_with_progress(command, environment, log, csv_path, stall_timeout):
    """Kill the entire owned process group if a failed collective stops progress."""
    with subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    ) as process:
        previous_size = -1
        last_progress = time.monotonic()
        while True:
            try:
                code = process.wait(timeout=10)
                if code:
                    raise subprocess.CalledProcessError(code, command)
                return
            except subprocess.TimeoutExpired:
                size = csv_path.stat().st_size if csv_path.exists() else 0
                if size != previous_size:
                    previous_size, last_progress = size, time.monotonic()
                if time.monotonic() - last_progress > stall_timeout:
                    kill_owned_process_tree(process.pid)
                    process.wait()
                    raise RuntimeError(f"No CSV progress for {stall_timeout:g}s: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etas", default=".5,.6,.7,.8,.9,.95,.98,.99")
    parser.add_argument("--nr", type=int, default=8)
    parser.add_argument("--angular-per-degree", type=float, default=2)
    parser.add_argument("--angular-scale", type=float, default=1.0)
    parser.add_argument("--min-angular", type=int, default=16)
    parser.add_argument("--end", type=float, default=0.2)
    parser.add_argument("--seed", type=float, default=1.0e-6)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--diffusion-step-scale", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--factors", default=".94,1,1.06")
    parser.add_argument("--orientation", choices=("z", "diagonal"), default="z")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument(
        "--processes",
        type=int,
        choices=(1, 2),
        default=1,
        help="Use one process per GPU for a distributed six-panel run",
    )
    parser.add_argument("--tag", default="scan")
    parser.add_argument("--output-dir", type=Path, default=HERE / "spherical_samples")
    parser.add_argument("--exe", type=Path, default=ROOT / "build/bin/spherical_convection.release")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stall-timeout", type=float, default=300.0)
    args = parser.parse_args()
    reference = json.loads((HERE / "spherical_reference_results.json").read_text())
    devices = [f"cuda:{int(v)}" for v in args.devices.split(",")]
    if args.processes > 1 and len(devices) != args.processes:
        raise ValueError("A distributed run needs one selected device per process")
    worker_devices = ["cuda:local"] if args.processes > 1 else devices
    runs = []
    for eta in map(float, args.etas.split(",")):
        case = next(c for c in reference["cases"] if abs(c["eta"] - eta) < 1.0e-12)
        ell = case["ell"]
        n = int(
            8
            * np.ceil(max(args.min_angular, args.angular_per_degree * np.sqrt(ell * (ell + 1))) / 8)
        )
        scaled_n = args.angular_scale * n
        if abs(scaled_n - round(scaled_n)) > 1.0e-10 or round(scaled_n) % 2:
            raise ValueError("angular-scale must produce an even integral panel resolution")
        n = int(round(scaled_n))
        for factor in map(float, args.factors.split(",")):
            name = (
                f"eta{round(eta * 100):03d}-n{n}-r{args.nr}-{args.tag}-f{round(factor * 1000):04d}"
            )
            runs.append(
                dict(
                    name=name,
                    eta=eta,
                    ell=ell,
                    N=n,
                    nr=args.nr,
                    Ra=factor * case["Ra"],
                    end=args.end,
                    seed=args.seed,
                    cfl=args.cfl,
                    diffusion_step_scale=args.diffusion_step_scale,
                    dtype=args.dtype,
                    orientation=args.orientation,
                )
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    manifest = dict(
        runs=runs,
        devices=devices,
        processes=args.processes,
        reference=reference,
        executable=str(args.exe.resolve()),
        executable_sha256=digest(args.exe),
        library_sha256=digest(ROOT / "build/lib/libsnap_release.so"),
    )
    (args.output_dir / f"manifest-{args.tag}-{stamp}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used",
            "--format=csv,noheader",
        ],
        text=True,
    )
    (args.output_dir / f"hardware-{stamp}.txt").write_text(hardware)

    def worker(index, device):
        for run in runs[index :: len(worker_devices)]:
            path = args.output_dir / (run["name"] + ".csv")
            meta_path = Path(str(path) + ".json")
            if path.exists() and not args.overwrite:
                if not meta_path.exists():
                    raise RuntimeError(f"Incomplete run {path}; choose --overwrite to rerun")
                meta = json.loads(meta_path.read_text())
                for key in (
                    "Ra",
                    "eta",
                    "ell",
                    "N",
                    "nr",
                    "seed",
                    "cfl",
                    "diffusion_step_scale",
                    "orientation",
                    "dtype",
                ):
                    value = meta.get(key, "float64" if key == "dtype" else None)
                    matches = (
                        value == run[key]
                        if isinstance(run[key], str)
                        else value is not None
                        and np.isclose(value, run[key], rtol=1.0e-13, atol=0.0)
                    )
                    if not matches:
                        raise RuntimeError(f"Existing run parameter mismatch: {path}: {key}")
                if meta["end_diffusive_time"] < run["end"] - 1.0e-12:
                    raise RuntimeError(f"Existing run is too short: {path}")
                print("REUSE", run["name"], flush=True)
                continue
            profile = path.with_suffix(".profile")
            meta_path.unlink(missing_ok=True)
            export_profile(profile, run["nr"], run["eta"], run["ell"], run["Ra"])
            command = [
                str(args.exe.resolve()),
                str(ROOT / "examples/spherical_convection.yaml"),
                str(run["Ra"]),
                str(run["eta"]),
                str(run["ell"]),
                str(run["N"]),
                str(run["nr"]),
                str(run["end"]),
                str(run["seed"]),
                str(run["cfl"]),
                "1",
                ".5",
                str(path.resolve()),
                device,
                str(profile.resolve()),
                run["orientation"],
                str(run["diffusion_step_scale"]),
                run["dtype"],
            ]
            print("START", run["name"], device, flush=True)
            environment = os.environ.copy()
            if args.processes > 1:
                environment.update(
                    BACKEND="ucx",
                    DEVICE="cuda",
                    CUDA_VISIBLE_DEVICES=args.devices,
                    UCX_TLS="self,tcp,cuda_copy,cuda_ipc",
                )
                command = [
                    "torchrun",
                    "--standalone",
                    "--no-python",
                    f"--nproc-per-node={args.processes}",
                    *command,
                ]
            provenance = dict(
                command=command,
                executable_sha256=digest(args.exe),
                library_sha256=digest(ROOT / "build/lib/libsnap_release.so"),
                profile_sha256=digest(profile),
                process_count=args.processes,
                visible_devices=args.devices if args.processes > 1 else None,
            )
            with path.with_suffix(".log").open("w") as log:
                run_with_progress(command, environment, log, path, args.stall_timeout)
            meta = json.loads(meta_path.read_text())
            assert meta["device"] == ("cuda:0" if args.processes > 1 else device)
            assert meta.get("processes", 1) == args.processes
            assert meta["end_diffusive_time"] >= run["end"] - 1.0e-12
            assert meta["retries"] == 0
            meta.update(provenance)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            print("DONE", run["name"], f"{meta['seconds']:.1f}s", flush=True)

    with ThreadPoolExecutor(max_workers=len(worker_devices)) as pool:
        jobs = [pool.submit(worker, i, device) for i, device in enumerate(worker_devices)]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
