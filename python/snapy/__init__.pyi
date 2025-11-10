"""
Python bindings for SNAP (Scalable Nonhydrostatic Atmosphere Package)

This module provides Python bindings to the C++ SNAP library for
atmospheric dynamics simulations.
"""

from typing import Callable, Optional
import torch

# Type aliases
bcfunc_t = Optional[Callable[[torch.Tensor, int, "BoundaryFuncOptions"], None]]

# Enums
class index:
    """Index enumeration for variable types."""
    idn: int
    ivx: int
    ivy: int
    ivz: int
    ipr: int
    icy: int

class BoundaryFace:
    """Boundary face enumeration."""
    kUnknown: int
    kInnerX1: int
    kOuterX1: int
    kInnerX2: int
    kOuterX2: int
    kInnerX3: int
    kOuterX3: int

# Import all submodules
from .boundary import *
from .coordinate import *
from .eos import *
from .forcing import *
from .hydro import *
from .implicit import *
from .integrator import *
from .layout import *
from .mesh import *
from .output import *
from .reconstruction import *
from .riemann import *
