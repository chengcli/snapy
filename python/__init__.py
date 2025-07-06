import torch
import pydisort
import pyharp
import kintera
import sysconfig
import ctypes
import os
import platform
from pathlib import Path

from .snapy import *

# link cuda library if on linux system
p = Path(__file__).parent / "lib" / "libsnap_cuda_release.so"
if platform.system() == "Linux" and p.exists():
    site_dir = sysconfig.get_paths()["purelib"]

    # combine NOW, GLOBAL, and NODELETE (fallback to 0x1000 if needed)
    NODELETE = getattr(os, "RTLD_NODELETE", 0x1000)
    mode = os.RTLD_NOW | os.RTLD_GLOBAL | NODELETE

    lib_path = f"{site_dir}/snapy/lib/libsnap_release.so"
    ctypes.CDLL(lib_path, mode=mode)

    lib_path = f"{site_dir}/snapy/lib/libsnap_cuda_release.so"
    ctypes.CDLL(lib_path, mode=mode)

__version__ = "0.1.0"
