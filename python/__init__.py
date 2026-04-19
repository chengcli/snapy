from importlib.metadata import PackageNotFoundError, version

import torch
import pydisort
import pyharp
import kintera

from .snapy import *

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

try:
    __version__ = version("snapy")
except PackageNotFoundError:
    __version__ = "0.0.0"
