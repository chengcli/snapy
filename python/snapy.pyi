"""
Python bindings for SNAP (Scalable Nonhydrostatic Atmosphere Package)

This module provides Python bindings to the C++ SNAP library for
atmospheric dynamics simulations.
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch

# Type aliases
bcfunc_t = Optional[Callable[[torch.Tensor, int, "BoundaryFuncOptions"], None]]

# Enums
class index:
    """Index enumeration for variable types."""
    idn: int
    ivx: int
    ivy: int
    ivz: int
    ipr: int
    icy: int

class BoundaryFace:
    """Boundary face enumeration."""
    kUnknown: int
    kInnerX1: int
    kOuterX1: int
    kInnerX2: int
    kOuterX2: int
    kInnerX3: int
    kOuterX3: int

# Boundary Conditions
class BoundaryFuncOptions:
    """
    Boundary function configuration options.
    
    This class manages boundary condition function parameters.
    """
    
    def __init__(self) -> None:
        """Initialize BoundaryFuncOptions with default values."""
        ...
    
    def __repr__(self) -> str: ...
    
    @overload
    def type(self) -> int:
        """Get the boundary condition type."""
        ...
    
    @overload
    def type(self, value: int) -> "BoundaryFuncOptions":
        """Set the boundary condition type."""
        ...
    
    @overload
    def nghost(self) -> int:
        """Get the number of ghost zones."""
        ...
    
    @overload
    def nghost(self, value: int) -> "BoundaryFuncOptions":
        """Set the number of ghost zones."""
        ...

class InternalBoundaryOptions:
    """
    Internal boundary configuration options.
    
    This class manages internal boundary parameters for solid boundaries.
    """
    
    def __init__(self) -> None:
        """Initialize InternalBoundaryOptions with default values."""
        ...
    
    def __repr__(self) -> str: ...
    
    @overload
    def nghost(self) -> int:
        """Get the number of ghost zones."""
        ...
    
    @overload
    def nghost(self, value: int) -> "InternalBoundaryOptions":
        """Set the number of ghost zones."""
        ...
    
    @overload
    def max_iter(self) -> int:
        """Get the maximum number of iterations."""
        ...
    
    @overload
    def max_iter(self, value: int) -> "InternalBoundaryOptions":
        """Set the maximum number of iterations."""
        ...
    
    @overload
    def solid_density(self) -> float:
        """Get the solid density value."""
        ...
    
    @overload
    def solid_density(self, value: float) -> "InternalBoundaryOptions":
        """Set the solid density value."""
        ...
    
    @overload
    def solid_pressure(self) -> float:
        """Get the solid pressure value."""
        ...
    
    @overload
    def solid_pressure(self, value: float) -> "InternalBoundaryOptions":
        """Set the solid pressure value."""
        ...

class InternalBoundary:
    """
    Internal boundary implementation for solid boundaries.
    
    This module handles internal boundary conditions in the simulation.
    """
    
    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...
    
    @overload
    def __init__(self, options: InternalBoundaryOptions) -> None:
        """
        Construct an InternalBoundary module.
        
        Args:
            options: Internal boundary configuration options
        """
        ...
    
    def __repr__(self) -> str: ...
    
    options: InternalBoundaryOptions
    
    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...
    
    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...
    
    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...
    
    def mark_prim_solid_(self, *args) -> None:
        """Mark primitive variables in solid regions."""
        ...
    
    def fill_cons_solid_(self, *args) -> None:
        """Fill conserved variables in solid regions."""
        ...
    
    def rectify_solid(
        self, 
        solid: torch.Tensor, 
        bfuncs: List[bcfunc_t] = []
    ) -> Tuple[torch.Tensor, int]:
        """
        Rectify solid boundary.
        
        Args:
            solid: Solid boundary tensor
            bfuncs: List of boundary functions
            
        Returns:
            Tuple of (result tensor, total number of flips)
        """
        ...

# Coordinate System
class CoordinateOptions:
    """
    Coordinate system configuration options.
    
    This class manages grid and coordinate parameters.
    """
    
    def __init__(self) -> None:
        """Initialize CoordinateOptions with default values."""
        ...
    
    def __repr__(self) -> str: ...
    
    @overload
    def x1min(self) -> float:
        """Get the minimum x1 coordinate."""
        ...
    
    @overload
    def x1min(self, value: float) -> "CoordinateOptions":
        """Set the minimum x1 coordinate."""
        ...
    
    @overload
    def x1max(self) -> float:
        """Get the maximum x1 coordinate."""
        ...
    
    @overload
    def x1max(self, value: float) -> "CoordinateOptions":
        """Set the maximum x1 coordinate."""
        ...
    
    @overload
    def x2min(self) -> float:
        """Get the minimum x2 coordinate."""
        ...
    
    @overload
    def x2min(self, value: float) -> "CoordinateOptions":
        """Set the minimum x2 coordinate."""
        ...
    
    @overload
    def x2max(self) -> float:
        """Get the maximum x2 coordinate."""
        ...
    
    @overload
    def x2max(self, value: float) -> "CoordinateOptions":
        """Set the maximum x2 coordinate."""
        ...
    
    @overload
    def x3min(self) -> float:
        """Get the minimum x3 coordinate."""
        ...
    
    @overload
    def x3min(self, value: float) -> "CoordinateOptions":
        """Set the minimum x3 coordinate."""
        ...
    
    @overload
    def x3max(self) -> float:
        """Get the maximum x3 coordinate."""
        ...
    
    @overload
    def x3max(self, value: float) -> "CoordinateOptions":
        """Set the maximum x3 coordinate."""
        ...
    
    @overload
    def nx1(self) -> int:
        """Get the number of grid cells in x1 direction."""
        ...
    
    @overload
    def nx1(self, value: int) -> "CoordinateOptions":
        """Set the number of grid cells in x1 direction."""
        ...
    
    @overload
    def nx2(self) -> int:
        """Get the number of grid cells in x2 direction."""
        ...
    
    @overload
    def nx2(self, value: int) -> "CoordinateOptions":
        """Set the number of grid cells in x2 direction."""
        ...
    
    @overload
    def nx3(self) -> int:
        """Get the number of grid cells in x3 direction."""
        ...
    
    @overload
    def nx3(self, value: int) -> "CoordinateOptions":
        """Set the number of grid cells in x3 direction."""
        ...
    
    @overload
    def nghost(self) -> int:
        """Get the number of ghost zones."""
        ...
    
    @overload
    def nghost(self, value: int) -> "CoordinateOptions":
        """Set the number of ghost zones."""
        ...

class Cartesian:
    """
    Cartesian coordinate system implementation.
    
    This module handles Cartesian grid operations.
    """
    
    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...
    
    @overload
    def __init__(self, options: CoordinateOptions) -> None:
        """
        Construct a Cartesian module.
        
        Args:
            options: Coordinate configuration options
        """
        ...
    
    def __repr__(self) -> str: ...
    
    options: CoordinateOptions
    
    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...
    
    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...
    
    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...
    
    def ifirst(self) -> int:
        """Get the first i index (inclusive)."""
        ...
    
    def ilast(self) -> int:
        """Get the last i index (exclusive)."""
        ...
    
    def jfirst(self) -> int:
        """Get the first j index (inclusive)."""
        ...
    
    def jlast(self) -> int:
        """Get the last j index (exclusive)."""
        ...
    
    def kfirst(self) -> int:
        """Get the first k index (inclusive)."""
        ...
    
    def klast(self) -> int:
        """Get the last k index (exclusive)."""
        ...
    
    def center_width1(self) -> torch.Tensor:
        """Get cell center widths in x1 direction."""
        ...
    
    def center_width2(self) -> torch.Tensor:
        """Get cell center widths in x2 direction."""
        ...
    
    def center_width3(self) -> torch.Tensor:
        """Get cell center widths in x3 direction."""
        ...
    
    def face_area1(self) -> torch.Tensor:
        """Get face areas perpendicular to x1."""
        ...
    
    def face_area2(self) -> torch.Tensor:
        """Get face areas perpendicular to x2."""
        ...
    
    def face_area3(self) -> torch.Tensor:
        """Get face areas perpendicular to x3."""
        ...
    
    def cell_volume(self) -> torch.Tensor:
        """Get cell volumes."""
        ...

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

# Implicit Solver
class ImplicitOptions:
    """
    Implicit solver configuration options.
    
    This class manages implicit time integration parameters.
    """
    
    def __init__(self) -> None:
        """Initialize ImplicitOptions with default values."""
        ...
    
    def __repr__(self) -> str: ...
    
    @overload
    def type(self) -> str:
        """Get the implicit solver type."""
        ...
    
    @overload
    def type(self, value: str) -> "ImplicitOptions":
        """Set the implicit solver type."""
        ...
    
    @overload
    def grav(self) -> float:
        """Get the gravity value."""
        ...
    
    @overload
    def grav(self, value: float) -> "ImplicitOptions":
        """Set the gravity value."""
        ...
    
    @overload
    def scheme(self) -> int:
        """Get the scheme type."""
        ...
    
    @overload
    def scheme(self, value: int) -> "ImplicitOptions":
        """Set the scheme type."""
        ...
    
    @overload
    def coord(self) -> CoordinateOptions:
        """Get the coordinate options."""
        ...
    
    @overload
    def coord(self, value: CoordinateOptions) -> "ImplicitOptions":
        """Set the coordinate options."""
        ...

class ImplicitHydro:
    """
    Implicit hydrodynamics solver.
    
    This module handles implicit time integration for hydrodynamics.
    """
    
    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...
    
    @overload
    def __init__(self, options: ImplicitOptions) -> None:
        """
        Construct an ImplicitHydro module.
        
        Args:
            options: Implicit solver configuration options
        """
        ...
    
    def __repr__(self) -> str: ...
    
    options: ImplicitOptions
    
    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...
    
    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...
    
    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

class ImplicitCorrection:
    """
    Implicit correction solver.
    
    This module handles implicit corrections.
    """
    
    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...
    
    @overload
    def __init__(self, options: ImplicitOptions) -> None:
        """
        Construct an ImplicitCorrection module.
        
        Args:
            options: Implicit solver configuration options
        """
        ...
    
    def __repr__(self) -> str: ...
    
    options: ImplicitOptions
    
    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...
    
    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...
    
    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

# Integrator
class IntegratorWeight:
    """
    Time integrator weight configuration.
    
    This class manages integrator weights for multi-stage methods.
    """
    
    def __init__(self) -> None:
        """Initialize IntegratorWeight with default values."""
        ...
    
    def __repr__(self) -> str: ...
    
    @overload
    def wght0(self) -> float:
        """Get weight 0."""
        ...
    
    @overload
    def wght0(self, value: float) -> "IntegratorWeight":
        """Set weight 0."""
        ...
    
    @overload
    def wght1(self) -> float:
        """Get weight 1."""
        ...
    
    @overload
    def wght1(self, value: float) -> "IntegratorWeight":
        """Set weight 1."""
        ...
    
    @overload
    def wght2(self) -> float:
        """Get weight 2."""
        ...
    
    @overload
    def wght2(self, value: float) -> "IntegratorWeight":
        """Set weight 2."""
        ...

class IntegratorOptions:
    """
    Time integrator configuration options.
    
    This class manages time integration parameters.
    """
    
    def __init__(self) -> None:
        """Initialize IntegratorOptions with default values."""
        ...
    
    def __repr__(self) -> str: ...
    
    @overload
    def type(self) -> str:
        """Get the integrator type."""
        ...
    
    @overload
    def type(self, value: str) -> "IntegratorOptions":
        """Set the integrator type."""
        ...
    
    @overload
    def cfl(self) -> float:
        """Get the CFL number."""
        ...
    
    @overload
    def cfl(self, value: float) -> "IntegratorOptions":
        """Set the CFL number."""
        ...

class Integrator:
    """
    Time integrator implementation.
    
    This module handles time integration.
    """
    
    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...
    
    @overload
    def __init__(self, options: IntegratorOptions) -> None:
        """
        Construct an Integrator module.
        
        Args:
            options: Time integrator configuration options
        """
        ...
    
    def __repr__(self) -> str: ...
    
    options: IntegratorOptions
    stages: int
    
    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...
    
    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...
    
    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...
    
    def stop(self, steps: int, current_time: float) -> bool:
        """
        Check if integration should stop.
        
        Args:
            steps: Number of steps taken
            current_time: Current simulation time
            
        Returns:
            True if should stop, False otherwise
        """
        ...

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

# Output
class OutputOptions:
    """
    Output configuration options.
    
    This class manages output parameters.
    """
    
    def __init__(self) -> None:
        """Initialize OutputOptions with default values."""
        ...
    
    def __repr__(self) -> str: ...
    
    @overload
    def fid(self) -> int:
        """Get file ID."""
        ...
    
    @overload
    def fid(self, value: int) -> "OutputOptions":
        """Set file ID."""
        ...
    
    @overload
    def dt(self) -> float:
        """Get output time interval."""
        ...
    
    @overload
    def dt(self, value: float) -> "OutputOptions":
        """Set output time interval."""
        ...
    
    @overload
    def output_slicex1(self) -> bool:
        """Get x1 slice output flag."""
        ...
    
    @overload
    def output_slicex1(self, value: bool) -> "OutputOptions":
        """Set x1 slice output flag."""
        ...
    
    @overload
    def output_slicex2(self) -> bool:
        """Get x2 slice output flag."""
        ...
    
    @overload
    def output_slicex2(self, value: bool) -> "OutputOptions":
        """Set x2 slice output flag."""
        ...
    
    @overload
    def output_slicex3(self) -> bool:
        """Get x3 slice output flag."""
        ...
    
    @overload
    def output_slicex3(self, value: bool) -> "OutputOptions":
        """Set x3 slice output flag."""
        ...
    
    @overload
    def output_sumx1(self) -> bool:
        """Get x1 sum output flag."""
        ...
    
    @overload
    def output_sumx1(self, value: bool) -> "OutputOptions":
        """Set x1 sum output flag."""
        ...
    
    @overload
    def output_sumx2(self) -> bool:
        """Get x2 sum output flag."""
        ...
    
    @overload
    def output_sumx2(self, value: bool) -> "OutputOptions":
        """Set x2 sum output flag."""
        ...
    
    @overload
    def output_sumx3(self) -> bool:
        """Get x3 sum output flag."""
        ...
    
    @overload
    def output_sumx3(self, value: bool) -> "OutputOptions":
        """Set x3 sum output flag."""
        ...
    
    @overload
    def include_ghost_zones(self) -> bool:
        """Get ghost zone inclusion flag."""
        ...
    
    @overload
    def include_ghost_zones(self, value: bool) -> "OutputOptions":
        """Set ghost zone inclusion flag."""
        ...
    
    @overload
    def cartesian_vector(self) -> bool:
        """Get Cartesian vector flag."""
        ...
    
    @overload
    def cartesian_vector(self, value: bool) -> "OutputOptions":
        """Set Cartesian vector flag."""
        ...
    
    @overload
    def x1_slice(self) -> float:
        """Get x1 slice position."""
        ...
    
    @overload
    def x1_slice(self, value: float) -> "OutputOptions":
        """Set x1 slice position."""
        ...
    
    @overload
    def x2_slice(self) -> float:
        """Get x2 slice position."""
        ...
    
    @overload
    def x2_slice(self, value: float) -> "OutputOptions":
        """Set x2 slice position."""
        ...
    
    @overload
    def x3_slice(self) -> float:
        """Get x3 slice position."""
        ...
    
    @overload
    def x3_slice(self, value: float) -> "OutputOptions":
        """Set x3 slice position."""
        ...
    
    @overload
    def variables(self) -> List[str]:
        """Get list of output variables."""
        ...
    
    @overload
    def variables(self, value: List[str]) -> "OutputOptions":
        """Set list of output variables."""
        ...
    
    @overload
    def file_type(self) -> str:
        """Get output file type."""
        ...
    
    @overload
    def file_type(self, value: str) -> "OutputOptions":
        """Set output file type."""
        ...
    
    @overload
    def data_format(self) -> str:
        """Get data format."""
        ...
    
    @overload
    def data_format(self, value: str) -> "OutputOptions":
        """Set data format."""
        ...

class OutputType:
    """
    Output type base class.
    
    This class manages output file generation.
    """
    
    @overload
    def __init__(self) -> None:
        """Initialize OutputType with default values."""
        ...
    
    @overload
    def __init__(self, options: OutputOptions) -> None:
        """
        Initialize OutputType with options.
        
        Args:
            options: Output configuration options
        """
        ...
    
    def __repr__(self) -> str: ...
    
    file_number: int
    next_time: float
    
    def increment_file_number(self) -> int:
        """
        Increment and return the file number.
        
        Returns:
            New file number
        """
        ...

class NetcdfOutput(OutputType):
    """
    NetCDF output implementation.
    
    This class handles NetCDF file output.
    """
    
    def __init__(self, options: OutputOptions) -> None:
        """
        Initialize NetcdfOutput.
        
        Args:
            options: Output configuration options
        """
        ...
    
    def __repr__(self) -> str: ...
    
    def write_output_file(
        self,
        block,  # MeshBlock object
        vars: Dict[str, torch.Tensor],
        time: float,
        wtflag: int = 0
    ) -> None:
        """
        Write output file.
        
        Args:
            block: MeshBlock object
            vars: Dictionary of variable tensors
            time: Current simulation time
            wtflag: Write flag (default 0)
        """
        ...

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

# Riemann Solver
class RiemannSolverOptions:
    """
    Riemann solver options.
    
    This class manages Riemann solver parameters.
    """
    
    def __init__(self) -> None:
        """Initialize RiemannSolverOptions with default values."""
        ...
    
    def __repr__(self) -> str: ...
    
    @overload
    def type(self) -> str:
        """Get Riemann solver type."""
        ...
    
    @overload
    def type(self, value: str) -> "RiemannSolverOptions":
        """Set Riemann solver type."""
        ...

class UpwindSolver:
    """
    Upwind Riemann solver.
    
    This module implements a simple upwind solver.
    """
    
    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...
    
    @overload
    def __init__(self, options: RiemannSolverOptions) -> None:
        """
        Construct an UpwindSolver module.
        
        Args:
            options: Riemann solver configuration options
        """
        ...
    
    def __repr__(self) -> str: ...
    
    options: RiemannSolverOptions
    
    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...
    
    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...
    
    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

class RoeSolver:
    """
    Roe approximate Riemann solver.
    
    This module implements the Roe solver.
    """
    
    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...
    
    @overload
    def __init__(self, options: RiemannSolverOptions) -> None:
        """
        Construct a RoeSolver module.
        
        Args:
            options: Riemann solver configuration options
        """
        ...
    
    def __repr__(self) -> str: ...
    
    options: RiemannSolverOptions
    
    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...
    
    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...
    
    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

class LmarsSolver:
    """
    LMARS Riemann solver.
    
    This module implements the Low-Mach Approximate Riemann Solver.
    """
    
    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...
    
    @overload
    def __init__(self, options: RiemannSolverOptions) -> None:
        """
        Construct an LmarsSolver module.
        
        Args:
            options: Riemann solver configuration options
        """
        ...
    
    def __repr__(self) -> str: ...
    
    options: RiemannSolverOptions
    
    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...
    
    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...
    
    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

class ShallowRoeSolver:
    """
    Shallow water Roe solver.
    
    This module implements the Roe solver for shallow water equations.
    """
    
    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...
    
    @overload
    def __init__(self, options: RiemannSolverOptions) -> None:
        """
        Construct a ShallowRoeSolver module.
        
        Args:
            options: Riemann solver configuration options
        """
        ...
    
    def __repr__(self) -> str: ...
    
    options: RiemannSolverOptions
    
    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...
    
    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...
    
    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...
