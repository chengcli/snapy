"""
Stub file for snapy.coordinate module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch

# Coordinate System
class CoordinateOptions:
    """
    Coordinate system configuration options.

    This class manages grid and coordinate parameters.
    """

    def __init__(self) -> None:
        """Initialize CoordinateOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def x1min(self) -> float:
        """Get the minimum x1 coordinate."""
        ...

    @overload
    def x1min(self, value: float) -> "CoordinateOptions":
        """Set the minimum x1 coordinate."""
        ...

    @overload
    def x1max(self) -> float:
        """Get the maximum x1 coordinate."""
        ...

    @overload
    def x1max(self, value: float) -> "CoordinateOptions":
        """Set the maximum x1 coordinate."""
        ...

    @overload
    def x2min(self) -> float:
        """Get the minimum x2 coordinate."""
        ...

    @overload
    def x2min(self, value: float) -> "CoordinateOptions":
        """Set the minimum x2 coordinate."""
        ...

    @overload
    def x2max(self) -> float:
        """Get the maximum x2 coordinate."""
        ...

    @overload
    def x2max(self, value: float) -> "CoordinateOptions":
        """Set the maximum x2 coordinate."""
        ...

    @overload
    def x3min(self) -> float:
        """Get the minimum x3 coordinate."""
        ...

    @overload
    def x3min(self, value: float) -> "CoordinateOptions":
        """Set the minimum x3 coordinate."""
        ...

    @overload
    def x3max(self) -> float:
        """Get the maximum x3 coordinate."""
        ...

    @overload
    def x3max(self, value: float) -> "CoordinateOptions":
        """Set the maximum x3 coordinate."""
        ...

    @overload
    def nx1(self) -> int:
        """Get the number of grid cells in x1 direction."""
        ...

    @overload
    def nx1(self, value: int) -> "CoordinateOptions":
        """Set the number of grid cells in x1 direction."""
        ...

    @overload
    def nx2(self) -> int:
        """Get the number of grid cells in x2 direction."""
        ...

    @overload
    def nx2(self, value: int) -> "CoordinateOptions":
        """Set the number of grid cells in x2 direction."""
        ...

    @overload
    def nx3(self) -> int:
        """Get the number of grid cells in x3 direction."""
        ...

    @overload
    def nx3(self, value: int) -> "CoordinateOptions":
        """Set the number of grid cells in x3 direction."""
        ...

    @overload
    def nghost(self) -> int:
        """Get the number of ghost zones."""
        ...

    @overload
    def nghost(self, value: int) -> "CoordinateOptions":
        """Set the number of ghost zones."""
        ...

class Cartesian:
    """
    Cartesian coordinate system implementation.

    This module handles Cartesian grid operations.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: CoordinateOptions) -> None:
        """
        Construct a Cartesian module.

        Args:
            options: Coordinate configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: CoordinateOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

    def ifirst(self) -> int:
        """Get the first i index (inclusive)."""
        ...

    def ilast(self) -> int:
        """Get the last i index (exclusive)."""
        ...

    def jfirst(self) -> int:
        """Get the first j index (inclusive)."""
        ...

    def jlast(self) -> int:
        """Get the last j index (exclusive)."""
        ...

    def kfirst(self) -> int:
        """Get the first k index (inclusive)."""
        ...

    def klast(self) -> int:
        """Get the last k index (exclusive)."""
        ...

    def center_width1(self) -> torch.Tensor:
        """Get cell center widths in x1 direction."""
        ...

    def center_width2(self) -> torch.Tensor:
        """Get cell center widths in x2 direction."""
        ...

    def center_width3(self) -> torch.Tensor:
        """Get cell center widths in x3 direction."""
        ...

    def face_area1(self) -> torch.Tensor:
        """Get face areas perpendicular to x1."""
        ...

    def face_area2(self) -> torch.Tensor:
        """Get face areas perpendicular to x2."""
        ...

    def face_area3(self) -> torch.Tensor:
        """Get face areas perpendicular to x3."""
        ...

    def cell_volume(self) -> torch.Tensor:
        """Get cell volumes."""
        ...
