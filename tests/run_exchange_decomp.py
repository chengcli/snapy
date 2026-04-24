#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd: Path, env=None) -> None:
    print(f"+ (cd {cwd} && {' '.join(cmd)})")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def prepare_case(case_dir: Path, yaml_src: Path) -> None:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    target = case_dir / "test_exchange.yaml"
    try:
        target.symlink_to(yaml_src)
    except OSError:
        shutil.copy2(yaml_src, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--build-type", required=True)
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    tests_dir = build_dir / "tests"
    exe = tests_dir / f"test_exchange.{args.build_type}"
    if not exe.exists():
        raise FileNotFoundError(f"missing executable {exe}")

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise FileNotFoundError("torchrun not found in PATH")

    yaml_src = Path(os.path.abspath("test_exchange.yaml"))
    if not yaml_src.exists():
        raise FileNotFoundError(f"missing input file {yaml_src}")

    cases = (
        ("mesh6", 1, 6, True, False),
        ("proc6", 6, 1, False, True),
        ("proc2_mesh3", 2, 3, True, True),
    )

    for name, ranks, blocks_per_process, expect_local, expect_remote in cases:
        case_dir = tests_dir / f"test_exchange_{name}"
        prepare_case(case_dir, yaml_src)

        env = os.environ.copy()
        env["BACKEND"] = "gloo"
        env["BLOCKS_PER_PROCESS"] = str(blocks_per_process)
        env["EXPECT_LOCAL_NEIGHBOR"] = "1" if expect_local else "0"
        env["EXPECT_REMOTE_NEIGHBOR"] = "1" if expect_remote else "0"

        run(
            [
                torchrun,
                "--no-python",
                f"--nproc-per-node={ranks}",
                str(exe),
            ],
            cwd=case_dir,
            env=env,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
