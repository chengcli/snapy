import torch
import pydisort
import pyharp
import kintera

from .snapy import *

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

__version__ = "1.2.1"
