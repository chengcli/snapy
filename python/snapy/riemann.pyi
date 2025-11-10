"""
Stub file for snapy.riemann module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch

# Riemann Solver
class RiemannSolverOptions:
    """
    Riemann solver options.

    This class manages Riemann solver parameters.
    """

    def __init__(self) -> None:
        """Initialize RiemannSolverOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def type(self) -> str:
        """Get Riemann solver type."""
        ...

    @overload
    def type(self, value: str) -> "RiemannSolverOptions":
        """Set Riemann solver type."""
        ...

class UpwindSolver:
    """
    Upwind Riemann solver.

    This module implements a simple upwind solver.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: RiemannSolverOptions) -> None:
        """
        Construct an UpwindSolver module.

        Args:
            options: Riemann solver configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: RiemannSolverOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

class RoeSolver:
    """
    Roe approximate Riemann solver.

    This module implements the Roe solver.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: RiemannSolverOptions) -> None:
        """
        Construct a RoeSolver module.

        Args:
            options: Riemann solver configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: RiemannSolverOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

class LmarsSolver:
    """
    LMARS Riemann solver.

    This module implements the Low-Mach Approximate Riemann Solver.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: RiemannSolverOptions) -> None:
        """
        Construct an LmarsSolver module.

        Args:
            options: Riemann solver configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: RiemannSolverOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

class ShallowRoeSolver:
    """
    Shallow water Roe solver.

    This module implements the Roe solver for shallow water equations.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: RiemannSolverOptions) -> None:
        """
        Construct a ShallowRoeSolver module.

        Args:
            options: Riemann solver configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: RiemannSolverOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...
