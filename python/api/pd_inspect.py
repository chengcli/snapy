#!/usr/bin/env python3
"""
Inspect tensor fields saved in TorchScript .part files.

Each .part file is assumed to be created by something like:

    class TensorModule(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            for name, tensor in tensors.items():
                self.register_buffer(name, tensor)

    scripted = torch.jit.script(TensorModule(tensor_map))
    scripted.save(filename)
"""

import argparse
import os
import tarfile
import tempfile
from typing import Optional

import torch


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
    """Load and inspect a single TorchScript .part file."""
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


def inspect_pt_from_tar(
    tar: tarfile.TarFile, member: tarfile.TarInfo) -> None:
    """
    Extract a .part member from a tar to a temporary file and inspect it.

    torch.jit.load generally expects a real file or a seekable file object,
    so using a NamedTemporaryFile is the safest option.
    """
    extracted = tar.extractfile(member)
    if extracted is None:
        print(f"\n=== {member.name} ===")
        print("  ERROR: could not extract file from tar")
        return

    with tempfile.NamedTemporaryFile(suffix=".part") as tmp:
        tmp.write(extracted.read())
        tmp.flush()
        inspect_pt_file(tmp.name, display_name=f"{member.name}")


def inspect_path(path: str) -> None:
    """Dispatch based on whether `path` is a .part file or a tar archive."""
    if tarfile.is_tarfile(path):
        # Treat as tar archive containing .part files
        with tarfile.open(path, "r:*") as tf:
            pt_members = [m for m in tf.getmembers() if m.name.endswith(".part")]
            if not pt_members:
                print(f"{path}: no .part files found in tar archive")
                return

            for m in pt_members:
                inspect_pt_from_tar(tf, m)
    else:
        # Treat as a single .part TorchScript file
        inspect_pt_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect tensor fields (name, shape, dtype, etc.) "
                    "in TorchScript .part files.\n"
                    "Can also inspect all .part files inside a tar/tar.gz archive."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Path(s) to .part file(s) or tar/tar.gz archive(s) containing .part files.",
    )
    args = parser.parse_args()

    for p in args.paths:
        if not os.path.exists(p):
            print(f"{p}: does not exist, skipping")
            continue
        inspect_path(p)


if __name__ == "__main__":
    main()
