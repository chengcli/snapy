#!/usr/bin/env python3
"""Inspect tensor fields saved in TorchScript restart payloads."""

import argparse
import os
import tempfile
from dataclasses import dataclass

import torch

RESTART_BUNDLE_MAGIC = "SNAPY_RESTART_BUNDLE_V1"


@dataclass(frozen=True)
class RestartBundleEntry:
    name: str
    size: int
    offset: int


def inspect_script_module(mod: torch.jit.ScriptModule, display_name: str) -> None:
    """Print information about buffers (tensors) stored in a ScriptModule."""
    print(f"\n=== {display_name} ===")

    # Buffers are where your tensors are, since you used register_buffer
    has_any = False
    for name, tensor in mod.named_buffers(recurse=True):
        has_any = True
        print(f"  buffer name     : {name}")
        print(f"    shape         : {tuple(tensor.shape)}")
        ## print first 10 tensor value if it is 1D
        if tensor.dim() == 1:
            print(f"    value         : {tensor[:10].tolist()}")
            if (tensor.numel() > 10):
                print("    ...")
        print(f"    dtype         : {tensor.dtype}")
        print(f"    device        : {tensor.device}")
        print(f"    requires_grad : {tensor.requires_grad}")
        print()

    # Just in case there are parameters too (unlikely with your save_tensors)
    for name, param in mod.named_parameters(recurse=True):
        if not has_any:
            has_any = True
        print(f"  parameter name  : {name}")
        print(f"    shape         : {tuple(param.shape)}")
        ## print first 10 tensor value if it is 1D
        if param.dim() == 1:
            print(f"    value         : {param[:10].tolist()}")
            if (param.numel() > 10):
                print("    ...")
        print(f"    dtype         : {param.dtype}")
        print(f"    device        : {param.device}")
        print(f"    requires_grad : {param.requires_grad}")
        print()

    if not has_any:
        print("  (no buffers or parameters found)")


def inspect_pt_file(path: str, display_name: str = None) -> None:
    """Load and inspect a single TorchScript tensor dump."""
    if display_name is None:
        display_name = path

    try:
        # Map everything to CPU just for inspection safety
        mod = torch.jit.load(path, map_location="cpu")
    except Exception as e:
        print(f"\n=== {display_name} ===")
        print(f"  ERROR: failed to load TorchScript file: {e}")
        return

    inspect_script_module(mod, display_name)


def is_restart_bundle(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.readline().decode("utf-8", errors="replace").rstrip("\n") == RESTART_BUNDLE_MAGIC
    except OSError:
        return False


def read_restart_bundle_index(path: str) -> list[RestartBundleEntry]:
    with open(path, "rb") as f:
        magic = f.readline().decode("utf-8", errors="strict").rstrip("\n")
        if magic != RESTART_BUNDLE_MAGIC:
            raise ValueError(f"{path}: not a restart bundle")

        count_line = f.readline().decode("utf-8", errors="strict").strip()
        if not count_line:
            raise ValueError(f"{path}: restart bundle missing entry count")
        entry_count = int(count_line)

        entries: list[RestartBundleEntry] = []
        for _ in range(entry_count):
            line = f.readline().decode("utf-8", errors="strict").rstrip("\n")
            name, size_str = line.split("\t", 1)
            entries.append(RestartBundleEntry(name=name, size=int(size_str), offset=0))

        terminator = f.readline().decode("utf-8", errors="strict").rstrip("\n")
        if terminator != "":
            raise ValueError(f"{path}: restart bundle header missing terminator")

        payload_offset = f.tell()
        running = 0
        out: list[RestartBundleEntry] = []
        for entry in entries:
            out.append(
                RestartBundleEntry(
                    name=entry.name,
                    size=entry.size,
                    offset=payload_offset + running,
                )
            )
            running += entry.size

        return out


def inspect_pt_from_bundle(path: str, entry: RestartBundleEntry) -> None:
    with open(path, "rb") as f:
        f.seek(entry.offset)
        payload = f.read(entry.size)

    if len(payload) != entry.size:
        print(f"\n=== {entry.name} ===")
        print("  ERROR: truncated restart bundle payload")
        return

    with tempfile.NamedTemporaryFile(suffix=".part") as tmp:
        tmp.write(payload)
        tmp.flush()
        inspect_pt_file(tmp.name, display_name=entry.name)


def inspect_path(path: str) -> None:
    """Dispatch based on whether `path` is a .part file or a restart bundle."""
    if is_restart_bundle(path):
        entries = [entry for entry in read_restart_bundle_index(path) if entry.name.endswith(".part")]
        if not entries:
            print(f"{path}: no .part files found in restart bundle")
            return

        for entry in entries:
            inspect_pt_from_bundle(path, entry)
    else:
        # Treat as a single .part TorchScript file
        inspect_pt_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect tensor fields (name, shape, dtype, etc.) "
                    "in TorchScript .part files.\n"
                    "Can also inspect all .part payloads inside a bundled .restart file."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Path(s) to .part file(s) or bundled .restart file(s).",
    )
    args = parser.parse_args()

    for p in args.paths:
        if not os.path.exists(p):
            print(f"{p}: does not exist, skipping")
            continue
        inspect_path(p)


if __name__ == "__main__":
    main()
