"""
Stub file for snapy.boundary module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch

# Type aliases
bcfunc_t = Optional[Callable[[torch.Tensor, int, "BoundaryFuncOptions"], None]]

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

class InternalBoundaryOptions:
    """
    Internal boundary configuration options.

    This class manages internal boundary parameters for solid boundaries.
    """

    def __init__(self) -> None:
        """Initialize InternalBoundaryOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def nghost(self) -> int:
        """Get the number of ghost zones."""
        ...

    @overload
    def nghost(self, value: int) -> "InternalBoundaryOptions":
        """Set the number of ghost zones."""
        ...

    @overload
    def max_iter(self) -> int:
        """Get the maximum number of iterations."""
        ...

    @overload
    def max_iter(self, value: int) -> "InternalBoundaryOptions":
        """Set the maximum number of iterations."""
        ...

    @overload
    def solid_density(self) -> float:
        """Get the solid density value."""
        ...

    @overload
    def solid_density(self, value: float) -> "InternalBoundaryOptions":
        """Set the solid density value."""
        ...

    @overload
    def solid_pressure(self) -> float:
        """Get the solid pressure value."""
        ...

    @overload
    def solid_pressure(self, value: float) -> "InternalBoundaryOptions":
        """Set the solid pressure value."""
        ...

class InternalBoundary:
    """
    Internal boundary implementation for solid boundaries.

    This module handles internal boundary conditions in the simulation.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: InternalBoundaryOptions) -> None:
        """
        Construct an InternalBoundary module.

        Args:
            options: Internal boundary configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: InternalBoundaryOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

    def mark_prim_solid_(self, *args) -> None:
        """Mark primitive variables in solid regions."""
        ...

    def fill_cons_solid_(self, *args) -> None:
        """Fill conserved variables in solid regions."""
        ...

    def rectify_solid(
        self,
        solid: torch.Tensor,
        bfuncs: List[bcfunc_t] = []
    ) -> Tuple[torch.Tensor, int]:
        """
        Rectify solid boundary.

        Args:
            solid: Solid boundary tensor
            bfuncs: List of boundary functions

        Returns:
            Tuple of (result tensor, total number of flips)
        """
        ...
