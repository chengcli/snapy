"""
Stub file for snapy.hydro module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch
from .boundary import InternalBoundaryOptions
from .coordinate import CoordinateOptions
from .eos import EquationOfStateOptions
from .forcing import ConstGravityOptions, CoriolisOptions
from .implicit import ImplicitOptions
from .reconstruction import ReconstructOptions
from .riemann import RiemannSolverOptions


# Type aliases
bcfunc_t = Optional[Callable[[torch.Tensor, int, "BoundaryFuncOptions"], None]]

# Hydro
class PrimitiveProjectorOptions:
    """
    Primitive variable projector options.

    This class manages primitive variable projection parameters.
    """

    def __init__(self) -> None:
        """Initialize PrimitiveProjectorOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def type(self) -> str:
        """Get the projector type."""
        ...

    @overload
    def type(self, value: str) -> "PrimitiveProjectorOptions":
        """Set the projector type."""
        ...

    @overload
    def margin(self) -> float:
        """Get the margin value."""
        ...

    @overload
    def margin(self, value: float) -> "PrimitiveProjectorOptions":
        """Set the margin value."""
        ...

    @overload
    def nghost(self) -> int:
        """Get the number of ghost zones."""
        ...

    @overload
    def nghost(self, value: int) -> "PrimitiveProjectorOptions":
        """Set the number of ghost zones."""
        ...

    @overload
    def grav(self) -> float:
        """Get the gravity value."""
        ...

    @overload
    def grav(self, value: float) -> "PrimitiveProjectorOptions":
        """Set the gravity value."""
        ...

    @overload
    def Rd(self) -> float:
        """Get the gas constant Rd."""
        ...

    @overload
    def Rd(self, value: float) -> "PrimitiveProjectorOptions":
        """Set the gas constant Rd."""
        ...

class HydroOptions:
    """
    Hydrodynamics configuration options.

    This class manages hydrodynamics parameters.
    """

    def __init__(self) -> None:
        """Initialize HydroOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @staticmethod
    def from_yaml(filename: str, dist: "DistributeInfo" = ...) -> "HydroOptions":
        """
        Load HydroOptions from a YAML file.

        Args:
            filename: Path to YAML file
            dist: Distribution info (optional)

        Returns:
            HydroOptions loaded from file
        """
        ...

    @overload
    def grav(self) -> ConstGravityOptions:
        """Get gravity options."""
        ...

    @overload
    def grav(self, value: ConstGravityOptions) -> "HydroOptions":
        """Set gravity options."""
        ...

    @overload
    def coriolis(self) -> CoriolisOptions:
        """Get Coriolis options."""
        ...

    @overload
    def coriolis(self, value: CoriolisOptions) -> "HydroOptions":
        """Set Coriolis options."""
        ...

    @overload
    def visc(self):  # DiffusionOptions
        """Get viscosity/diffusion options."""
        ...

    @overload
    def visc(self, value) -> "HydroOptions":  # DiffusionOptions
        """Set viscosity/diffusion options."""
        ...

    @overload
    def coord(self) -> CoordinateOptions:
        """Get coordinate options."""
        ...

    @overload
    def coord(self, value: CoordinateOptions) -> "HydroOptions":
        """Set coordinate options."""
        ...

    @overload
    def eos(self) -> EquationOfStateOptions:
        """Get equation of state options."""
        ...

    @overload
    def eos(self, value: EquationOfStateOptions) -> "HydroOptions":
        """Set equation of state options."""
        ...

    @overload
    def proj(self) -> PrimitiveProjectorOptions:
        """Get primitive projector options."""
        ...

    @overload
    def proj(self, value: PrimitiveProjectorOptions) -> "HydroOptions":
        """Set primitive projector options."""
        ...

    @overload
    def recon1(self) -> "ReconstructOptions":
        """Get reconstruction options for dimension 1."""
        ...

    @overload
    def recon1(self, value: "ReconstructOptions") -> "HydroOptions":
        """Set reconstruction options for dimension 1."""
        ...

    @overload
    def recon23(self) -> "ReconstructOptions":
        """Get reconstruction options for dimensions 2 and 3."""
        ...

    @overload
    def recon23(self, value: "ReconstructOptions") -> "HydroOptions":
        """Set reconstruction options for dimensions 2 and 3."""
        ...

    @overload
    def riemann(self) -> "RiemannSolverOptions":
        """Get Riemann solver options."""
        ...

    @overload
    def riemann(self, value: "RiemannSolverOptions") -> "HydroOptions":
        """Set Riemann solver options."""
        ...

    @overload
    def ib(self) -> InternalBoundaryOptions:
        """Get internal boundary options."""
        ...

    @overload
    def ib(self, value: InternalBoundaryOptions) -> "HydroOptions":
        """Set internal boundary options."""
        ...

    @overload
    def imp(self) -> "ImplicitOptions":
        """Get implicit solver options."""
        ...

    @overload
    def imp(self, value: "ImplicitOptions") -> "HydroOptions":
        """Set implicit solver options."""
        ...

class Hydro:
    """
    Hydrodynamics implementation.

    This module handles hydrodynamic calculations.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: HydroOptions) -> None:
        """
        Construct a Hydro module.

        Args:
            options: Hydrodynamics configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: HydroOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

    def max_time_step(self, *args) -> float:
        """Calculate maximum stable time step."""
        ...

    def reset_timer(self) -> None:
        """Reset performance timers."""
        ...

    def get_eos(self) -> EquationOfState:
        """Get the equation of state object."""
        ...

    def report_timer(self) -> str:
        """Get performance timer report."""
        ...
