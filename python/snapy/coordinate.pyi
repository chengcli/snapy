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

class Coordinate:
    """
    Coordinate system base class.

    This class handles coordinate system operations.
    """

    def __init__(self, options: CoordinateOptions, hydro = None) -> None:
        """
        Initialize Coordinate system.

        Args:
            options: Coordinate configuration options
            hydro: Parent hydro module (optional)
        """
        ...

    def __repr__(self) -> str: ...

    def il(self) -> int:
        """Get the lower i index."""
        ...

    def iu(self) -> int:
        """Get the upper i index."""
        ...

    def jl(self) -> int:
        """Get the lower j index."""
        ...

    def ju(self) -> int:
        """Get the upper j index."""
        ...

    def kl(self) -> int:
        """Get the lower k index."""
        ...

    def ku(self) -> int:
        """Get the upper k index."""
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

class Cartesian(Coordinate):
    """
    Cartesian coordinate system implementation.

    This module handles Cartesian grid operations.
    """

    def __init__(self, options: CoordinateOptions, hydro = None) -> None:
        """
        Construct a Cartesian module.

        Args:
            options: Coordinate configuration options
            hydro: Parent hydro module (optional)
        """
        ...

    def __repr__(self) -> str: ...

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

class GnomonicEquiangle(Coordinate):
    """
    Gnomonic equiangle coordinate system implementation.

    This module handles gnomonic equiangle projection for cubed sphere grids.
    """

    def __init__(self, options: CoordinateOptions, hydro = None) -> None:
        """
        Construct a GnomonicEquiangle module.

        Args:
            options: Coordinate configuration options
            hydro: Parent hydro module (optional)
        """
        ...

    def __repr__(self) -> str: ...

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

# Coordinate submodule functions
class coord:
    """Coordinate utility functions."""

    @staticmethod
    def coord_vec_lower_(*args) -> None:
        """Lower vector indices in coordinate system."""
        ...

    @staticmethod
    def coord_vec_raise_(*args) -> None:
        """Raise vector indices in coordinate system."""
        ...

    @staticmethod
    def cs_cart_to_contra_(*args) -> None:
        """Convert Cartesian to contravariant coordinates on cubed sphere."""
        ...

    @staticmethod
    def cs_contra_to_cart_(*args) -> None:
        """Convert contravariant to Cartesian coordinates on cubed sphere."""
        ...

    @staticmethod
    def cs_ab_to_lonlat(*args) -> Tuple[float, float]:
        """
        Convert cubed sphere (a, b) coordinates to longitude/latitude.

        Returns:
            Tuple of (longitude, latitude)
        """
        ...

    @staticmethod
    def get_cs_face_name(face_id: int) -> str:
        """
        Get the name of a cubed sphere face.

        Args:
            face_id: Face index

        Returns:
            Face name string
        """
        ...
