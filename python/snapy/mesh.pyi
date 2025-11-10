"""
Stub file for snapy.mesh module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch
from .hydro import HydroOptions
from .integrator import IntegratorOptions
from .layout import DistributeInfo


# Type aliases
bcfunc_t = Optional[Callable[[torch.Tensor, int, "BoundaryFuncOptions"], None]]

# MeshBlock
class ScalarOptions:
    """Scalar transport options (placeholder)."""
    pass

class MeshBlockOptions:
    """
    Mesh block configuration options.

    This class manages mesh block parameters.
    """

    def __init__(self) -> None:
        """Initialize MeshBlockOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @staticmethod
    def from_yaml(filename: str, dist: DistributeInfo = ...) -> "MeshBlockOptions":
        """
        Load MeshBlockOptions from a YAML file.

        Args:
            filename: Path to YAML file
            dist: Distribution info (optional)

        Returns:
            MeshBlockOptions loaded from file
        """
        ...

    def set_bfunc(
        self,
        dx3: int,
        dx2: int,
        dx1: int,
        func: bcfunc_t
    ) -> None:
        """
        Set boundary function for a specific face.

        Args:
            dx3: Direction in x3 (-1, 0, or 1)
            dx2: Direction in x2 (-1, 0, or 1)
            dx1: Direction in x1 (-1, 0, or 1)
            func: Boundary function or None
        """
        ...

    @overload
    def dist(self) -> DistributeInfo:
        """Get distribution info."""
        ...

    @overload
    def dist(self, value: DistributeInfo) -> "MeshBlockOptions":
        """Set distribution info."""
        ...

    @overload
    def intg(self) -> IntegratorOptions:
        """Get integrator options."""
        ...

    @overload
    def intg(self, value: IntegratorOptions) -> "MeshBlockOptions":
        """Set integrator options."""
        ...

    @overload
    def hydro(self) -> HydroOptions:
        """Get hydro options."""
        ...

    @overload
    def hydro(self, value: HydroOptions) -> "MeshBlockOptions":
        """Set hydro options."""
        ...

    @overload
    def scalar(self) -> ScalarOptions:
        """Get scalar options."""
        ...

    @overload
    def scalar(self, value: ScalarOptions) -> "MeshBlockOptions":
        """Set scalar options."""
        ...

    @overload
    def bfuncs(self) -> List[bcfunc_t]:
        """Get boundary functions."""
        ...

    @overload
    def bfuncs(self, value: List[bcfunc_t]) -> "MeshBlockOptions":
        """Set boundary functions."""
        ...

class MeshBlock:
    """
    Mesh block implementation.

    This module represents a computational block in the domain.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: MeshBlockOptions) -> None:
        """
        Construct a MeshBlock module.

        Args:
            options: Mesh block configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: MeshBlockOptions

    def forward(
        self,
        dt: float,
        stage: int,
        vars: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward integration step.

        Args:
            dt: Time step size
            stage: Integration stage
            vars: Dictionary of variable tensors

        Returns:
            Updated variables dictionary
        """
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

    def part(
        self,
        offset: Tuple[int, int, int],
        exterior: bool = False,
        extend_x1: int = 0,
        extend_x2: int = 0,
        extend_x3: int = 0
    ) -> Tuple:
        """
        Get index slices for a mesh block part.

        Args:
            offset: Index offset tuple
            exterior: Whether to include exterior
            extend_x1: Extension in x1 direction
            extend_x2: Extension in x2 direction
            extend_x3: Extension in x3 direction

        Returns:
            Tuple of slice objects
        """
        ...

    def initialize(self, *args) -> None:
        """Initialize the mesh block."""
        ...

    def max_time_step(self, vars: Dict[str, torch.Tensor]) -> float:
        """
        Calculate maximum stable time step.

        Args:
            vars: Dictionary of variable tensors

        Returns:
            Maximum stable time step
        """
        ...
