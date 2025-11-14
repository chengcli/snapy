"""
Python bindings for SNAP (Scalable Nonhydrostatic Atmosphere Package)

This module provides Python bindings to the C++ SNAP library for
atmospheric dynamics simulations.
"""

from typing import Callable, Optional
import torch

# Type aliases
bcfunc_t = Optional[Callable[[torch.Tensor, int, "BoundaryFuncOptions"], None]]

kIDN: int   # Density index
kIV1: int   # Velocity index in X1 direction
kIV2: int   # Velocity index in X2 direction
kIV3: int   # Velocity index in X3 direction
kIPR: int   # Pressure (or internal energy) index
KICY: int   # Tracer index start

kInnerX1: int   # Inner boundary in the X1 direction (bottom)
kOuterX1: int   # Outer boundary in the X1 direction (top)
kInnerX2: int   # Inner boundary in the X2 direction (south)
kOuterX2: int   # Outer boundary in the X2 direction (north)
kInnerX3: int   # Inner boundary in the X3 direction (west)
kOuterX3: int   # Outer boundary in the X3 direction (east)

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
