"""
Stub file for snapy.forcing module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch

# Forcing
class ConstGravityOptions:
    """
    Constant gravity forcing options.

    This class manages constant gravity parameters.
    """

    def __init__(self) -> None:
        """Initialize ConstGravityOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def grav1(self) -> float:
        """Get gravity in x1 direction."""
        ...

    @overload
    def grav1(self, value: float) -> "ConstGravityOptions":
        """Set gravity in x1 direction."""
        ...

    @overload
    def grav2(self) -> float:
        """Get gravity in x2 direction."""
        ...

    @overload
    def grav2(self, value: float) -> "ConstGravityOptions":
        """Set gravity in x2 direction."""
        ...

    @overload
    def grav3(self) -> float:
        """Get gravity in x3 direction."""
        ...

    @overload
    def grav3(self, value: float) -> "ConstGravityOptions":
        """Set gravity in x3 direction."""
        ...

class CoriolisOptions:
    """
    Coriolis forcing options.

    This class manages Coriolis force parameters.
    """

    def __init__(self) -> None:
        """Initialize CoriolisOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def omega1(self) -> float:
        """Get rotation rate omega1."""
        ...

    @overload
    def omega1(self, value: float) -> "CoriolisOptions":
        """Set rotation rate omega1."""
        ...

    @overload
    def omega2(self) -> float:
        """Get rotation rate omega2."""
        ...

    @overload
    def omega2(self, value: float) -> "CoriolisOptions":
        """Set rotation rate omega2."""
        ...

    @overload
    def omega3(self) -> float:
        """Get rotation rate omega3."""
        ...

    @overload
    def omega3(self, value: float) -> "CoriolisOptions":
        """Set rotation rate omega3."""
        ...

    @overload
    def omegax(self) -> float:
        """Get rotation rate omegax."""
        ...

    @overload
    def omegax(self, value: float) -> "CoriolisOptions":
        """Set rotation rate omegax."""
        ...

    @overload
    def omegay(self) -> float:
        """Get rotation rate omegay."""
        ...

    @overload
    def omegay(self, value: float) -> "CoriolisOptions":
        """Set rotation rate omegay."""
        ...

    @overload
    def omegaz(self) -> float:
        """Get rotation rate omegaz."""
        ...

    @overload
    def omegaz(self, value: float) -> "CoriolisOptions":
        """Set rotation rate omegaz."""
        ...
