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

    @staticmethod
    def from_yaml(filename: str, verbose: bool = False) -> "PrimitiveProjectorOptions":
        """
        Load PrimitiveProjectorOptions from a YAML file.

        Args:
            filename: Path to YAML file
            verbose: Enable verbose output

        Returns:
            PrimitiveProjectorOptions loaded from file
        """
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
    def from_yaml(filename: str, verbose: bool = False) -> "HydroOptions":
        """
        Load HydroOptions from a YAML file.

        Args:
            filename: Path to YAML file
            verbose: Enable verbose output (optional)

        Returns:
            HydroOptions loaded from file
        """
        ...

    @overload
    def verbose(self) -> bool:
        """Get verbose flag."""
        ...

    @overload
    def verbose(self, value: bool) -> "HydroOptions":
        """Set verbose flag."""
        ...

    @overload
    def disable_flux_x1(self) -> bool:
        """Get disable flux x1 flag."""
        ...

    @overload
    def disable_flux_x1(self, value: bool) -> "HydroOptions":
        """Set disable flux x1 flag."""
        ...

    @overload
    def disable_flux_x2(self) -> bool:
        """Get disable flux x2 flag."""
        ...

    @overload
    def disable_flux_x2(self, value: bool) -> "HydroOptions":
        """Set disable flux x2 flag."""
        ...

    @overload
    def disable_flux_x3(self) -> bool:
        """Get disable flux x3 flag."""
        ...

    @overload
    def disable_flux_x3(self, value: bool) -> "HydroOptions":
        """Set disable flux x3 flag."""
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
    def eos(self) -> "EquationOfStateOptions":
        """Get equation of state options."""
        ...

    @overload
    def eos(self, value: "EquationOfStateOptions") -> "HydroOptions":
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
    def icorr(self) -> "ImplicitOptions":
        """Get implicit correction options."""
        ...

    @overload
    def icorr(self, value: "ImplicitOptions") -> "HydroOptions":
        """Set implicit correction options."""
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
    def __init__(self, options: HydroOptions, block = None) -> None:
        """
        Construct a Hydro module.

        Args:
            options: Hydrodynamics configuration options
            block: Parent block module (optional)
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
