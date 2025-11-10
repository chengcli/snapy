"""
Stub file for snapy.layout module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch

# Layout
class DistributeInfo:
    """
    Domain distribution information.

    This class manages domain decomposition parameters.
    """

    def __init__(self) -> None:
        """Initialize DistributeInfo with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def face(self) -> int:
        """Get the face index."""
        ...

    @overload
    def face(self, value: int) -> "DistributeInfo":
        """Set the face index."""
        ...

    @overload
    def level(self) -> int:
        """Get the refinement level."""
        ...

    @overload
    def level(self, value: int) -> "DistributeInfo":
        """Set the refinement level."""
        ...

    @overload
    def gid(self) -> int:
        """Get the global ID."""
        ...

    @overload
    def gid(self, value: int) -> "DistributeInfo":
        """Set the global ID."""
        ...

    @overload
    def lx1(self) -> int:
        """Get the local x1 index."""
        ...

    @overload
    def lx1(self, value: int) -> "DistributeInfo":
        """Set the local x1 index."""
        ...

    @overload
    def lx2(self) -> int:
        """Get the local x2 index."""
        ...

    @overload
    def lx2(self, value: int) -> "DistributeInfo":
        """Set the local x2 index."""
        ...

    @overload
    def lx3(self) -> int:
        """Get the local x3 index."""
        ...

    @overload
    def lx3(self, value: int) -> "DistributeInfo":
        """Set the local x3 index."""
        ...

    @overload
    def nb1(self) -> int:
        """Get the number of blocks in x1."""
        ...

    @overload
    def nb1(self, value: int) -> "DistributeInfo":
        """Set the number of blocks in x1."""
        ...

    @overload
    def nb2(self) -> int:
        """Get the number of blocks in x2."""
        ...

    @overload
    def nb2(self, value: int) -> "DistributeInfo":
        """Set the number of blocks in x2."""
        ...

    @overload
    def nb3(self) -> int:
        """Get the number of blocks in x3."""
        ...

    @overload
    def nb3(self, value: int) -> "DistributeInfo":
        """Set the number of blocks in x3."""
        ...

class SlabLayout:
    """
    2D slab domain layout.

    This class manages 2D domain decomposition.
    """

    def __init__(
        self,
        px: int,
        py: int,
        periodic_x: bool = False,
        periodic_y: bool = False
    ) -> None:
        """
        Initialize SlabLayout.

        Args:
            px: Number of processes in x direction
            py: Number of processes in y direction
            periodic_x: Whether x direction is periodic
            periodic_y: Whether y direction is periodic
        """
        ...

    def __repr__(self) -> str: ...

    def get_procs(self) -> int:
        """Get total number of processes."""
        ...

    def rank_of(self, rx: int, ry: int) -> int:
        """
        Get rank for given process coordinates.

        Args:
            rx: Process x coordinate
            ry: Process y coordinate

        Returns:
            Process rank
        """
        ...

    def loc_of(self, rank: int) -> Tuple[int, int]:
        """
        Get process coordinates for given rank.

        Args:
            rank: Process rank

        Returns:
            Tuple of (rx, ry) process coordinates
        """
        ...

    def neighbor_rank(
        self,
        rx: int,
        ry: int,
        dx: int,
        dy: int,
        dz: int = 0
    ) -> int:
        """
        Get neighbor rank.

        Args:
            rx: Current process x coordinate
            ry: Current process y coordinate
            dx: Offset in x direction
            dy: Offset in y direction
            dz: Offset in z direction (unused for slab)

        Returns:
            Neighbor rank
        """
        ...

class CubedLayout:
    """
    3D cubed domain layout.

    This class manages 3D domain decomposition.
    """

    def __init__(
        self,
        px: int,
        py: int,
        pz: int,
        periodic_x: bool = False,
        periodic_y: bool = False,
        periodic_z: bool = False
    ) -> None:
        """
        Initialize CubedLayout.

        Args:
            px: Number of processes in x direction
            py: Number of processes in y direction
            pz: Number of processes in z direction
            periodic_x: Whether x direction is periodic
            periodic_y: Whether y direction is periodic
            periodic_z: Whether z direction is periodic
        """
        ...

    def __repr__(self) -> str: ...

    def get_procs(self) -> int:
        """Get total number of processes."""
        ...

    def rank_of(self, rx: int, ry: int, rz: int) -> int:
        """
        Get rank for given process coordinates.

        Args:
            rx: Process x coordinate
            ry: Process y coordinate
            rz: Process z coordinate

        Returns:
            Process rank
        """
        ...

    def loc_of(self, rank: int) -> Tuple[int, int, int]:
        """
        Get process coordinates for given rank.

        Args:
            rank: Process rank

        Returns:
            Tuple of (rx, ry, rz) process coordinates
        """
        ...

    def neighbor_rank(
        self,
        rx: int,
        ry: int,
        rz: int,
        dx: int,
        dy: int,
        dz: int
    ) -> int:
        """
        Get neighbor rank.

        Args:
            rx: Current process x coordinate
            ry: Current process y coordinate
            rz: Current process z coordinate
            dx: Offset in x direction
            dy: Offset in y direction
            dz: Offset in z direction

        Returns:
            Neighbor rank
        """
        ...

class CubedSphereLayout:
    """
    Cubed sphere domain layout.

    This class manages cubed sphere domain decomposition.
    """

    def __init__(self, pxy: int) -> None:
        """
        Initialize CubedSphereLayout.

        Args:
            pxy: Number of processes per face dimension
        """
        ...

    def __repr__(self) -> str: ...

    def get_procs(self) -> int:
        """Get total number of processes."""
        ...

    def rank_of(self, face: int, rx: int, ry: int) -> int:
        """
        Get rank for given face and process coordinates.

        Args:
            face: Cube face index
            rx: Process x coordinate on face
            ry: Process y coordinate on face

        Returns:
            Process rank
        """
        ...

    def loc_of(self, rank: int) -> Tuple[int, int, int]:
        """
        Get face and process coordinates for given rank.

        Args:
            rank: Process rank

        Returns:
            Tuple of (face, rx, ry)
        """
        ...

    def neighbor_rank(
        self,
        face: int,
        rx: int,
        ry: int,
        dx: int,
        dy: int,
        dz: int = 0
    ) -> int:
        """
        Get neighbor rank on cubed sphere.

        Args:
            face: Current cube face
            rx: Current process x coordinate
            ry: Current process y coordinate
            dx: Offset in x direction
            dy: Offset in y direction
            dz: Offset in z direction (unused)

        Returns:
            Neighbor rank
        """
        ...
