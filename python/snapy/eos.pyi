"""
Stub file for snapy.eos module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch


# Equation of State
class EquationOfStateOptions:
    """
    Equation of state configuration options.

    This class manages EOS parameters.
    """

    def __init__(self) -> None:
        """Initialize EquationOfStateOptions with default values."""
        ...

    @staticmethod
    def from_yaml(filename: str, verbose: bool = False) -> "EquationOfStateOptions":
        """
        Load EquationOfStateOptions from a YAML file.

        Args:
            filename: Path to YAML file
            verbose: Enable verbose output

        Returns:
            EquationOfStateOptions loaded from file
        """
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
    def temperature_floor(self) -> float:
        """Get the temperature floor value."""
        ...

    @overload
    def temperature_floor(self, value: float) -> "EquationOfStateOptions":
        """Set the temperature floor value."""
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
    def verbose(self) -> bool:
        """Get the verbose flag."""
        ...

    @overload
    def verbose(self, value: bool) -> "EquationOfStateOptions":
        """Set the verbose flag."""
        ...

    @overload
    def gammad(self) -> float:
        """Get the adiabatic index (gamma_d)."""
        ...

    @overload
    def gammad(self, value: float) -> "EquationOfStateOptions":
        """Set the adiabatic index (gamma_d)."""
        ...

    @overload
    def weight(self) -> float:
        """Get the molecular weight."""
        ...

    @overload
    def weight(self, value: float) -> "EquationOfStateOptions":
        """Set the molecular weight."""
        ...

    @overload
    def eos_file(self) -> str:
        """Get the EOS data file path."""
        ...

    @overload
    def eos_file(self, value: str) -> "EquationOfStateOptions":
        """Set the EOS data file path."""
        ...

    @overload
    def thermo(self):  # kintera.ThermoOptions
        """Get the thermodynamics options."""
        ...

    @overload
    def thermo(self, value) -> "EquationOfStateOptions":  # kintera.ThermoOptions
        """Set the thermodynamics options."""
        ...

class EquationOfState:
    """
    Equation of state base class.

    This class handles thermodynamic state calculations.
    """

    @overload
    def __init__(self) -> None:
        """Initialize EquationOfState with default values."""
        ...

    @overload
    def __init__(self, options: EquationOfStateOptions, phydro = None) -> None:
        """
        Initialize EquationOfState with options.

        Args:
            options: EOS configuration options
            phydro: Parent hydro module (optional)
        """
        ...

    def __repr__(self) -> str: ...

    def nvar(self) -> int:
        """Get the number of variables."""
        ...

    def species_weight(self, n: int = 0) -> float:
        """
        Get species molecular weight.

        Args:
            n: Species index (default 0)

        Returns:
            Molecular weight
        """
        ...

    def species_cv_ref(self, n: int = 0) -> float:
        """
        Get species reference specific heat at constant volume.

        Args:
            n: Species index (default 0)

        Returns:
            Reference specific heat cv
        """
        ...

    def compute(self, *args) -> torch.Tensor:
        """Compute thermodynamic properties."""
        ...

class IdealGas(EquationOfState):
    """
    Ideal gas equation of state.

    This module implements an ideal gas EOS.
    """

    def __init__(self, options: EquationOfStateOptions, phydro = None) -> None:
        """
        Initialize IdealGas EOS.

        Args:
            options: EOS configuration options
            phydro: Parent hydro module (optional)
        """
        ...

    def __repr__(self) -> str: ...

    options: EquationOfStateOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

    def nvar(self) -> int:
        """Get the number of variables."""
        ...

    def compute(self, *args) -> torch.Tensor:
        """Compute thermodynamic properties."""
        ...

class IdealMoist(EquationOfState):
    """
    Ideal moist gas equation of state.

    This module implements an ideal moist gas EOS.
    """

    def __init__(self, options: EquationOfStateOptions, phydro = None) -> None:
        """
        Initialize IdealMoist EOS.

        Args:
            options: EOS configuration options
            phydro: Parent hydro module (optional)
        """
        ...

    def __repr__(self) -> str: ...

    options: EquationOfStateOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

    def nvar(self) -> int:
        """Get the number of variables."""
        ...

    def compute(self, *args) -> torch.Tensor:
        """Compute thermodynamic properties."""
        ...

class MoistMixture(EquationOfState):
    """
    Moist mixture equation of state.

    This module implements a moist mixture EOS.
    """

    def __init__(self, options: EquationOfStateOptions, phydro = None) -> None:
        """
        Initialize MoistMixture EOS.

        Args:
            options: EOS configuration options
            phydro: Parent hydro module (optional)
        """
        ...

    def __repr__(self) -> str: ...

    options: EquationOfStateOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

    def nvar(self) -> int:
        """Get the number of variables."""
        ...

    def compute(self, *args) -> torch.Tensor:
        """Compute thermodynamic properties."""
        ...

class ShallowWater(EquationOfState):
    """
    Shallow water equation of state.

    This module implements a shallow water EOS.
    """

    def __init__(self, options: EquationOfStateOptions, phydro = None) -> None:
        """
        Initialize ShallowWater EOS.

        Args:
            options: EOS configuration options
            phydro: Parent hydro module (optional)
        """
        ...

    def __repr__(self) -> str: ...

    options: EquationOfStateOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

    def nvar(self) -> int:
        """Get the number of variables."""
        ...

    def compute(self, *args) -> torch.Tensor:
        """Compute thermodynamic properties."""
        ...
