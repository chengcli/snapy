"""
Stub file for snapy.layout module
"""

from typing import Tuple, overload

# Layout
class LayoutOptions:
    """
    Layout configuration options.

    This class manages domain decomposition parameters.
    """

    def __init__(self) -> None:
        """Initialize LayoutOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def type(self) -> str:
        """Get the layout type."""
        ...

    @overload
    def type(self, value: str) -> "LayoutOptions":
        """Set the layout type."""
        ...

    @overload
    def px(self) -> int:
        """Get the number of processes in x direction."""
        ...

    @overload
    def px(self, value: int) -> "LayoutOptions":
        """Set the number of processes in x direction."""
        ...

    @overload
    def py(self) -> int:
        """Get the number of processes in y direction."""
        ...

    @overload
    def py(self, value: int) -> "LayoutOptions":
        """Set the number of processes in y direction."""
        ...

    @overload
    def pz(self) -> int:
        """Get the number of processes in z direction."""
        ...

    @overload
    def pz(self, value: int) -> "LayoutOptions":
        """Set the number of processes in z direction."""
        ...

    @overload
    def periodic_x(self) -> bool:
        """Get periodic x flag."""
        ...

    @overload
    def periodic_x(self, value: bool) -> "LayoutOptions":
        """Set periodic x flag."""
        ...

    @overload
    def periodic_y(self) -> bool:
        """Get periodic y flag."""
        ...

    @overload
    def periodic_y(self, value: bool) -> "LayoutOptions":
        """Set periodic y flag."""
        ...

    @overload
    def periodic_z(self) -> bool:
        """Get periodic z flag."""
        ...

    @overload
    def periodic_z(self, value: bool) -> "LayoutOptions":
        """Set periodic z flag."""
        ...

    @overload
    def verbose(self) -> bool:
        """Get verbose flag."""
        ...

    @overload
    def verbose(self, value: bool) -> "LayoutOptions":
        """Set verbose flag."""
        ...

    @overload
    def backend(self) -> str:
        """Get backend name."""
        ...

    @overload
    def backend(self, value: str) -> "LayoutOptions":
        """Set backend name."""
        ...

    @overload
    def device(self) -> str:
        """Get execution device hint."""
        ...

    @overload
    def device(self, value: str) -> "LayoutOptions":
        """Set execution device hint."""
        ...

    @overload
    def master_addr(self) -> str:
        """Get master address."""
        ...

    @overload
    def master_addr(self, value: str) -> "LayoutOptions":
        """Set master address."""
        ...

    @overload
    def master_port(self) -> int:
        """Get master port."""
        ...

    @overload
    def master_port(self, value: int) -> "LayoutOptions":
        """Set master port."""
        ...

    @overload
    def root_rank(self) -> int:
        """Get root rank."""
        ...

    @overload
    def root_rank(self, value: int) -> "LayoutOptions":
        """Set root rank."""
        ...

    @overload
    def process_rank(self) -> int:
        """Get process rank."""
        ...

    @overload
    def process_rank(self, value: int) -> "LayoutOptions":
        """Set process rank."""
        ...

    @overload
    def process_world_size(self) -> int:
        """Get process world size."""
        ...

    @overload
    def process_world_size(self, value: int) -> "LayoutOptions":
        """Set process world size."""
        ...

    @overload
    def world_size(self) -> int:
        """Get world size."""
        ...

    @overload
    def world_size(self, value: int) -> "LayoutOptions":
        """Set world size."""
        ...

    @overload
    def rank(self) -> int:
        """Get rank."""
        ...

    @overload
    def rank(self, value: int) -> "LayoutOptions":
        """Set rank."""
        ...

    @overload
    def local_rank(self) -> int:
        """Get local rank."""
        ...

    @overload
    def local_rank(self, value: int) -> "LayoutOptions":
        """Set local rank."""
        ...

    @overload
    def blocks_per_process(self) -> int:
        """Get number of local blocks per process."""
        ...

    @overload
    def blocks_per_process(self, value: int) -> "LayoutOptions":
        """Set number of local blocks per process."""
        ...

    @overload
    def device_id(self) -> int:
        """Get explicit device id."""
        ...

    @overload
    def device_id(self, value: int) -> "LayoutOptions":
        """Set explicit device id."""
        ...


class SyncOptions:
    """Options controlling mesh ghost-zone synchronization."""

    DIM1: int
    DIM2: int
    DIM3: int

    def __init__(self) -> None: ...
    def dz_min(self) -> int: ...
    def dz_max(self) -> int: ...
    def dx_min(self) -> int: ...
    def dx_max(self) -> int: ...
    def dy_min(self) -> int: ...
    def dy_max(self) -> int: ...

    @overload
    def cross_panel_only(self) -> bool: ...
    @overload
    def cross_panel_only(self, value: bool) -> "SyncOptions": ...

    @overload
    def skip_corner(self) -> bool: ...
    @overload
    def skip_corner(self, value: bool) -> "SyncOptions": ...

    @overload
    def interpolate(self) -> bool: ...
    @overload
    def interpolate(self, value: bool) -> "SyncOptions": ...

    @overload
    def type(self) -> int: ...
    @overload
    def type(self, value: int) -> "SyncOptions": ...

    @overload
    def dim(self) -> int: ...
    @overload
    def dim(self, value: int) -> "SyncOptions": ...

    @overload
    def phyid(self) -> int: ...
    @overload
    def phyid(self, value: int) -> "SyncOptions": ...


class Layout:
    """
    Layout base class.

    This class manages domain decomposition.
    """

    def __init__(self, options: LayoutOptions) -> None:
        """
        Initialize Layout.

        Args:
            options: Layout configuration options
        """
        ...

    def __repr__(self) -> str: ...

    def get_procs(self) -> int:
        """Get total number of processes."""
        ...

    def rank_of(self, *args) -> int:
        """
        Get rank for given process coordinates.

        Returns:
            Process rank
        """
        ...

    def loc_of(self, rank: int) -> Tuple:
        """
        Get process coordinates for given rank.

        Args:
            rank: Process rank

        Returns:
            Tuple of process coordinates
        """
        ...

    def neighbor_rank(self, *args) -> int:
        """
        Get neighbor rank.

        Returns:
            Neighbor rank
        """
        ...

class SlabLayout(Layout):
    """
    2D slab domain layout.

    This class manages 2D domain decomposition.
    """

    def __init__(self, options: LayoutOptions) -> None:
        """
        Initialize SlabLayout.

        Args:
            options: Layout configuration options
        """
        ...

    def __repr__(self) -> str: ...

class CubedLayout(Layout):
    """
    3D cubed domain layout.

    This class manages 3D domain decomposition.
    """

    def __init__(self, options: LayoutOptions) -> None:
        """
        Initialize CubedLayout.

        Args:
            options: Layout configuration options
        """
        ...

    def __repr__(self) -> str: ...

class CubedSphereLayout(Layout):
    """
    Cubed sphere domain layout.

    This class manages cubed sphere domain decomposition.
    """

    @overload
    def __init__(self) -> None:
        """Initialize CubedSphereLayout with default values."""
        ...

    @overload
    def __init__(self, options: LayoutOptions) -> None:
        """
        Initialize CubedSphereLayout.

        Args:
            options: Layout configuration options
        """
        ...

    def __repr__(self) -> str: ...

# Distributed submodule functions
class distributed:
    """Distributed utility functions."""

    @staticmethod
    def get_rank() -> int:
        """Get the current process rank."""
        ...

    @staticmethod
    def get_local_rank() -> int:
        """Get the current local process rank."""
        ...

    @staticmethod
    def get_layout(*args):  # Returns Layout
        """Get the layout object."""
        ...

    @staticmethod
    def set_process_group(pg) -> None:
        """Register an externally initialized torch.distributed process group."""
        ...

    @staticmethod
    def is_process_group_initialized() -> bool:
        """Return whether an external process group has been registered."""
        ...
