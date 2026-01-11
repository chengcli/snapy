"""
Stub file for snapy.mesh module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch
from .hydro import HydroOptions
from .layout import LayoutOptions


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
    def from_yaml(filename: str, verbose: bool = False) -> "MeshBlockOptions":
        """
        Load MeshBlockOptions from a YAML file.

        Args:
            filename: Path to YAML file
            verbose: Enable verbose output

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
    def intg(self):  # IntegratorOptions
        """Get integrator options."""
        ...

    @overload
    def intg(self, value) -> "MeshBlockOptions":  # IntegratorOptions
        """Set integrator options."""
        ...

    @overload
    def coord(self):  # CoordinateOptions
        """Get coordinate options."""
        ...

    @overload
    def coord(self, value) -> "MeshBlockOptions":  # CoordinateOptions
        """Set coordinate options."""
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
    def ib(self):  # InternalBoundaryOptions
        """Get internal boundary options."""
        ...

    @overload
    def ib(self, value) -> "MeshBlockOptions":  # InternalBoundaryOptions
        """Set internal boundary options."""
        ...

    @overload
    def bfuncs(self) -> List[bcfunc_t]:
        """Get boundary functions."""
        ...

    @overload
    def bfuncs(self, value: List[bcfunc_t]) -> "MeshBlockOptions":
        """Set boundary functions."""
        ...

    @overload
    def layout(self) -> LayoutOptions:
        """Get layout options."""
        ...

    @overload
    def layout(self, value: LayoutOptions) -> "MeshBlockOptions":
        """Set layout options."""
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

    def inc_cycle(self) -> int:
        """
        Increment and return the cycle number.

        Returns:
            Previous cycle number
        """
        ...

    def set_user_output_func(self, func: Callable) -> None:
        """
        Set user output callback function.

        Args:
            func: User output function
        """
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

    def make_outputs(
        self,
        vars: Dict[str, torch.Tensor],
        time: float,
        final_write: bool = False
    ) -> None:
        """
        Generate output files.

        Args:
            vars: Dictionary of variable tensors
            time: Current simulation time
            final_write: Whether this is a final write
        """
        ...

    def part(
        self,
        offset: Tuple[int, int, int],
        exterior: bool = True,
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

    def initialize(self, vars) -> Tuple:
        """
        Initialize the mesh block.

        Args:
            vars: Variables dictionary

        Returns:
            Tuple of (vars, time)
        """
        ...

    def print_cycle_info(self, *args) -> None:
        """Print cycle information."""
        ...

    def finalize(self, *args) -> None:
        """Finalize the mesh block."""
        ...

    def device(self) -> torch.device:
        """
        Get the device of the mesh block.

        Returns:
            PyTorch device
        """
        ...

    def check_redo(self, *args) -> bool:
        """
        Check if step needs to be redone.

        Returns:
            True if redo is needed
        """
        ...
