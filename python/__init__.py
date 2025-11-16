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

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

NODELETE = getattr(os, "RTLD_NODELETE", 0x1000)
MODE = os.RTLD_NOW | os.RTLD_GLOBAL | NODELETE

def load_once(name):
    lib = Path(__file__).parent / "lib" / name
    if platform.system()=="Linux" and lib.exists():
        ctypes.CDLL(str(lib), mode=MODE)

load_once("libsnap_release.so")
load_once("libsnap_cuda_release.so")

__version__ = "0.8.6"
