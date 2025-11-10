"""
Stub file for snapy.implicit module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch
from .coordinate import CoordinateOptions


# Implicit Solver
class ImplicitOptions:
    """
    Implicit solver configuration options.

    This class manages implicit time integration parameters.
    """

    def __init__(self) -> None:
        """Initialize ImplicitOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def type(self) -> str:
        """Get the implicit solver type."""
        ...

    @overload
    def type(self, value: str) -> "ImplicitOptions":
        """Set the implicit solver type."""
        ...

    @overload
    def grav(self) -> float:
        """Get the gravity value."""
        ...

    @overload
    def grav(self, value: float) -> "ImplicitOptions":
        """Set the gravity value."""
        ...

    @overload
    def scheme(self) -> int:
        """Get the scheme type."""
        ...

    @overload
    def scheme(self, value: int) -> "ImplicitOptions":
        """Set the scheme type."""
        ...

    @overload
    def coord(self) -> CoordinateOptions:
        """Get the coordinate options."""
        ...

    @overload
    def coord(self, value: CoordinateOptions) -> "ImplicitOptions":
        """Set the coordinate options."""
        ...

class ImplicitHydro:
    """
    Implicit hydrodynamics solver.

    This module handles implicit time integration for hydrodynamics.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: ImplicitOptions) -> None:
        """
        Construct an ImplicitHydro module.

        Args:
            options: Implicit solver configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: ImplicitOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

class ImplicitCorrection:
    """
    Implicit correction solver.

    This module handles implicit corrections.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: ImplicitOptions) -> None:
        """
        Construct an ImplicitCorrection module.

        Args:
            options: Implicit solver configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: ImplicitOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...
