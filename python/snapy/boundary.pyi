"""
Stub file for snapy.boundary module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch

# Type aliases
bcfunc_t = Optional[Callable[[torch.Tensor, int, "BoundaryFuncOptions"], None]]

# Boundary face constants
kUnknown: int
kInnerX1: int
kOuterX1: int
kInnerX2: int
kOuterX2: int
kInnerX3: int
kOuterX3: int

# Boundary Conditions
class BoundaryFuncOptions:
    """
    Boundary function configuration options.

    This class manages boundary condition function parameters.
    """

    def __init__(self) -> None:
        """Initialize BoundaryFuncOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def type(self) -> int:
        """Get the boundary condition type."""
        ...

    @overload
    def type(self, value: int) -> "BoundaryFuncOptions":
        """Set the boundary condition type."""
        ...

    @overload
    def nghost(self) -> int:
        """Get the number of ghost zones."""
        ...

    @overload
    def nghost(self, value: int) -> "BoundaryFuncOptions":
        """Set the number of ghost zones."""
        ...

class EmbeddedBoundaryGeometry:
    """Static Cartesian geometry. Positive level-set values denote fluid."""
    volume_fraction: torch.Tensor
    cell_centroid: torch.Tensor
    face_fraction: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    face_centroid: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    wall_area_vector: torch.Tensor
    wall_centroid: torch.Tensor
    flags: torch.Tensor
    @staticmethod
    def from_level_set(phi: torch.Tensor, x1f: torch.Tensor, x2f: torch.Tensor, x3f: torch.Tensor) -> "EmbeddedBoundaryGeometry": ...
    @staticmethod
    def from_metrics(volume_fraction: torch.Tensor, cell_centroid: torch.Tensor,
                     face_fraction: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     face_centroid: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     wall_area_vector: torch.Tensor, wall_centroid: torch.Tensor,
                     x1f: torch.Tensor, x2f: torch.Tensor, x3f: torch.Tensor) -> "EmbeddedBoundaryGeometry": ...
