"""Run reproducible full snapy planar convection scans on two CUDA devices.

Build planar_convection.release first, then run this script from the repo root.
Existing outputs are overwritten only with --overwrite. Logs and per-run JSON
metadata remain alongside the CSV samples. A failed executable fails the batch.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=ROOT / "build/bin/planar_convection.release")
    parser.add_argument("--output-dir", type=Path, default=HERE / "planar_samples")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    reference = json.loads((HERE / "planar_reference_results.json").read_text())
    case = next(c for c in reference["cases"] if c["name"] == "stratified")
    k = case["levels"][-1]["k"]
    runs = []
    for n in (16, 32, 64):
        for label, ra in (("low", 680), ("middle", 695), ("high", 710)):
            runs.append(dict(name=f"n{n}-{label}", Ra=ra, nz=n, end=1.0, seed=1.0e-6, cfl=0.8))
    runs += [
        dict(name="rest", Ra=695, nz=32, end=0.3, seed=0.0, cfl=0.8),
        dict(name="small-seed", Ra=710, nz=32, end=1.0, seed=1.0e-7, cfl=0.8),
        dict(name="half-dt", Ra=710, nz=32, end=1.0, seed=1.0e-6, cfl=0.4),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    devices = [f"cuda:{int(d)}" for d in args.devices.split(",")]
    if not args.overwrite:
        for run in runs:
            assert not (args.output_dir / (run["name"] + ".csv")).exists(), run["name"]
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    )
    (args.output_dir / "hardware.txt").write_text(hardware)
    manifest = dict(reference=case, runs=runs, devices=devices, executable=str(args.exe.resolve()))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def worker(index, device):
        for run in runs[index :: len(devices)]:
            output = args.output_dir / (run["name"] + ".csv")
            command = [
                str(args.exe.resolve()),
                str(ROOT / "examples/planar_convection.yaml"),
                str(run["Ra"]),
                str(case["a"]),
                str(case["D"]),
                str(k),
                str(run["nz"]),
                str(run["end"]),
                str(run["seed"]),
                str(run["cfl"]),
                str(output.resolve()),
                device,
            ]
            print("START", run["name"], device, flush=True)
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
            meta = json.loads(Path(str(output) + ".json").read_text())
            assert meta["device"] == device and meta["end_diffusive_time"] >= run["end"] - 1.0e-12
            print("DONE", run["name"], f"{meta['seconds']:.2f} s", flush=True)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        jobs = [pool.submit(worker, i, device) for i, device in enumerate(devices)]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
