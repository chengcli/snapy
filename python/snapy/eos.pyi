"""
Stub file for snapy.eos module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch
from .coordinate import CoordinateOptions


# Equation of State
class EquationOfStateOptions:
    """
    Equation of state configuration options.

    This class manages EOS parameters.
    """

    def __init__(self) -> None:
        """Initialize EquationOfStateOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def type(self) -> str:
        """Get the EOS type."""
        ...

    @overload
    def type(self, value: str) -> "EquationOfStateOptions":
        """Set the EOS type."""
        ...

    @overload
    def density_floor(self) -> float:
        """Get the density floor value."""
        ...

    @overload
    def density_floor(self, value: float) -> "EquationOfStateOptions":
        """Set the density floor value."""
        ...

    @overload
    def pressure_floor(self) -> float:
        """Get the pressure floor value."""
        ...

    @overload
    def pressure_floor(self, value: float) -> "EquationOfStateOptions":
        """Set the pressure floor value."""
        ...

    @overload
    def limiter(self) -> bool:
        """Get the limiter flag."""
        ...

    @overload
    def limiter(self, value: bool) -> "EquationOfStateOptions":
        """Set the limiter flag."""
        ...

    @overload
    def thermo(self):  # kintera.ThermoOptions
        """Get the thermodynamics options."""
        ...

    @overload
    def thermo(self, value) -> "EquationOfStateOptions":  # kintera.ThermoOptions
        """Set the thermodynamics options."""
        ...

    @overload
    def coord(self) -> CoordinateOptions:
        """Get the coordinate options."""
        ...

    @overload
    def coord(self, value: CoordinateOptions) -> "EquationOfStateOptions":
        """Set the coordinate options."""
        ...

class EquationOfState:
    """
    Equation of state implementation.

    This class handles thermodynamic state calculations.
    """

    def __repr__(self) -> str: ...

    def nvar(self) -> int:
        """Get the number of variables."""
        ...

    def compute(self, *args) -> torch.Tensor:
        """Compute thermodynamic properties."""
        ...

    def forward(self, *args) -> torch.Tensor:
        """Forward pass to compute EOS."""
        ...
