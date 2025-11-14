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
class Index:
    """
    Index enumeration for variable types.

    kIDN: Density index
    kIV1: Velocity in the X1 direction
    kIV2: Velocity in the X2 direction
    kIV3: Velocity in the X3 direction
    kIPR: Pressure index (or internal energy index for conserved variables)
    kICY: Tracer index
    """
    kIDN: int
    kIV1: int
    kIV2: int
    kIV3: int
    kIPR: int
    kICY: int

class BoundaryFace:
    """
    Boundary face enumeration.

    kUnknown: Unknown boundary face
    kInnerX1: Inner boundary in the X1 direction (bottom)
    kOuterX1: Outer boundary in the X1 direction (top)
    kInnerX2: Inner boundary in the X2 direction (south)
    kOuterX2: Outer boundary in the X2 direction (north)
    kInnerX3: Inner boundary in the X3 direction (west)
    kOuterX3: Outer boundary in the X3 direction (east)
    """
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
