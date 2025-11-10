"""
Stub file for snapy.reconstruction module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch

# Reconstruction
class InterpOptions:
    """
    Interpolation options.

    This class manages interpolation parameters for reconstruction.
    """

    @overload
    def __init__(self) -> None:
        """Initialize InterpOptions with default values."""
        ...

    @overload
    def __init__(self, type: str) -> None:
        """
        Initialize InterpOptions with type.

        Args:
            type: Interpolation type
        """
        ...

    def __repr__(self) -> str: ...

    @overload
    def type(self) -> str:
        """Get interpolation type."""
        ...

    @overload
    def type(self, value: str) -> "InterpOptions":
        """Set interpolation type."""
        ...

    @overload
    def scale(self) -> bool:
        """Get scaling flag."""
        ...

    @overload
    def scale(self, value: bool) -> "InterpOptions":
        """Set scaling flag."""
        ...

class ReconstructOptions:
    """
    Reconstruction options.

    This class manages reconstruction parameters.
    """

    def __init__(self) -> None:
        """Initialize ReconstructOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def shock(self) -> bool:
        """Get shock detection flag."""
        ...

    @overload
    def shock(self, value: bool) -> "ReconstructOptions":
        """Set shock detection flag."""
        ...

    @overload
    def interp(self) -> InterpOptions:
        """Get interpolation options."""
        ...

    @overload
    def interp(self, value: InterpOptions) -> "ReconstructOptions":
        """Set interpolation options."""
        ...

class Reconstruct:
    """
    Spatial reconstruction implementation.

    This module handles high-order spatial reconstruction.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: ReconstructOptions) -> None:
        """
        Construct a Reconstruct module.

        Args:
            options: Reconstruction configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: ReconstructOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...
