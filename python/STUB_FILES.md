# Stub Files for Type Hints and Documentation

This directory contains Python stub files (`.pyi`) that provide type hints and API documentation for the `snapy` package, which is implemented using pybind11 C++ bindings.

## What are Stub Files?

Stub files are special Python files with a `.pyi` extension that contain type signatures and documentation for modules without including the actual implementation. They are used by:

- **Type checkers** (mypy, pyright, etc.) to verify type correctness
- **IDEs** (VS Code, PyCharm, etc.) to provide autocomplete and documentation
- **Documentation generators** to extract API information

## Structure

- `snapy.pyi` - Main stub file containing all type signatures for the compiled pybind11 module
- `py.typed` - Marker file indicating this package supports type hints (PEP 561)

## Benefits of Separation

By separating the pybind11 implementation (`python/csrc/*.cpp`) from the API documentation (stub files), we achieve:

1. **Clear Documentation**: Type hints and docstrings are in pure Python syntax
2. **Better IDE Support**: IDEs can parse stub files to provide autocomplete
3. **Type Safety**: Static type checkers can verify correct usage
4. **Maintainability**: Documentation is separate from C++ binding code
5. **PEP 561 Compliance**: Package properly declares type information

## Usage

### For Users

The stub files are automatically included when you install the package. Your IDE will automatically use them for autocomplete and type checking:

```python
from snapy import MeshBlockOptions, MeshBlock, HydroOptions
import torch

# IDE will show autocomplete and parameter hints
hydro_opts = HydroOptions()
mesh_opts = MeshBlockOptions().hydro(hydro_opts)
block = MeshBlock(mesh_opts)

# Type checkers will verify correct types
vars = {"cons": torch.zeros(5, 10, 10, 10)}
dt = 0.01
stage = 1
result = block.forward(dt, stage, vars)  # result is Dict[str, torch.Tensor]
```

### For Type Checking

Run mypy to verify type correctness:

```bash
mypy your_script.py
```

### For Developers

When modifying the C++ bindings in `python/csrc/`, remember to update the corresponding type signatures in `snapy.pyi`:

1. Add new classes or functions to the stub file
2. Update parameter types and return types
3. Include docstrings with examples
4. Test with `mypy` to ensure validity

## Implementation Details

### C++ Bindings Location

The actual pybind11 implementation is in:
- `python/csrc/snapy.cpp` - Main module definition
- `python/csrc/pybc.cpp` - Boundary conditions
- `python/csrc/pycoord.cpp` - Coordinate systems
- `python/csrc/pyeos.cpp` - Equation of state
- `python/csrc/pyforcing.cpp` - Forcing terms (gravity, Coriolis)
- `python/csrc/pyhydro.cpp` - Hydrodynamics
- `python/csrc/pyimplicit.cpp` - Implicit solvers
- `python/csrc/pyintg.cpp` - Time integration
- `python/csrc/pylayout.cpp` - Domain decomposition layouts
- `python/csrc/pymesh.cpp` - Mesh block
- `python/csrc/pyoutput.cpp` - Output management
- `python/csrc/pyrecon.cpp` - Spatial reconstruction
- `python/csrc/pyriemann.cpp` - Riemann solvers
- `python/csrc/pyscalar.cpp` - Scalar transport (placeholder)

### Stub File Contents

The `snapy.pyi` stub file includes:

- All public classes with their methods
- Type signatures using Python's `typing` module
- Overloaded methods using `@overload` decorator
- Comprehensive docstrings with parameter descriptions
- Module-level enums and type aliases
- All exposed C++ classes:
  - Boundary conditions: `BoundaryFuncOptions`, `InternalBoundaryOptions`, `InternalBoundary`
  - Coordinates: `CoordinateOptions`, `Cartesian`
  - EOS: `EquationOfStateOptions`, `EquationOfState`
  - Forcing: `ConstGravityOptions`, `CoriolisOptions`
  - Hydro: `HydroOptions`, `PrimitiveProjectorOptions`, `Hydro`
  - Implicit: `ImplicitOptions`, `ImplicitHydro`, `ImplicitCorrection`
  - Integration: `IntegratorOptions`, `IntegratorWeight`, `Integrator`
  - Layout: `DistributeInfo`, `SlabLayout`, `CubedLayout`, `CubedSphereLayout`
  - Mesh: `MeshBlockOptions`, `MeshBlock`
  - Output: `OutputOptions`, `OutputType`, `NetcdfOutput`
  - Reconstruction: `InterpOptions`, `ReconstructOptions`, `Reconstruct`
  - Riemann: `RiemannSolverOptions`, `UpwindSolver`, `RoeSolver`, `LmarsSolver`, `ShallowRoeSolver`

### Package Configuration

The stub files should be included in the package via `pyproject.toml`:

```toml
[tool.setuptools.package-data]
"snapy" = ["*.pyi", "py.typed", ...]
```

## References

- [PEP 561 - Distributing and Packaging Type Information](https://www.python.org/dev/peps/pep-0561/)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [mypy - Optional Static Typing for Python](http://mypy-lang.org/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)

## Contributing

When adding new features to the C++ bindings:

1. Implement the binding in the appropriate `python/csrc/*.cpp` file
2. Add the corresponding type signature to `snapy.pyi`
3. Include docstrings with parameter descriptions
4. Run `mypy` to validate the stub file
5. Test that IDEs can properly autocomplete the new features

This separation ensures that the API remains well-documented and type-safe while maintaining the performance benefits of C++ implementation.

## Module Overview

### Boundary Conditions (`pybc.cpp`)
Defines boundary condition functions and internal boundary handling for solid boundaries.

### Coordinate Systems (`pycoord.cpp`)
Provides Cartesian coordinate system with grid geometry calculations.

### Equation of State (`pyeos.cpp`)
Thermodynamic equation of state for computing pressure, temperature, and other state variables.

### Forcing Terms (`pyforcing.cpp`)
Implements constant gravity and Coriolis forcing.

### Hydrodynamics (`pyhydro.cpp`)
Main hydrodynamics module with options for reconstruction, Riemann solvers, and primitive variable projection.

### Implicit Solvers (`pyimplicit.cpp`)
Implicit time integration schemes for stiff terms.

### Time Integration (`pyintg.cpp`)
Time stepping schemes with CFL condition handling.

### Domain Layout (`pylayout.cpp`)
Domain decomposition for parallel computing: slab (2D), cubed (3D), and cubed-sphere layouts.

### Mesh Block (`pymesh.cpp`)
Computational mesh block representing a subdomain with all physics modules.

### Output (`pyoutput.cpp`)
NetCDF output for simulation results with options for slices and full domain output.

### Reconstruction (`pyrecon.cpp`)
High-order spatial reconstruction schemes for flux calculations.

### Riemann Solvers (`pyriemann.cpp`)
Approximate Riemann solvers: upwind, Roe, LMARS, and shallow water Roe.

## Example Usage

### Basic Simulation Setup

```python
import snapy
import torch

# Set up coordinate system
coord_opts = snapy.CoordinateOptions()
coord_opts.x1min(-1.0).x1max(1.0).nx1(100)
coord_opts.x2min(-1.0).x2max(1.0).nx2(100)
coord_opts.x3min(-1.0).x3max(1.0).nx3(1)
coord_opts.nghost(2)

# Set up hydrodynamics
hydro_opts = snapy.HydroOptions()
hydro_opts.coord(coord_opts)

# Set up mesh block
mesh_opts = snapy.MeshBlockOptions()
mesh_opts.hydro(hydro_opts)

# Create mesh block
block = snapy.MeshBlock(mesh_opts)

# Initialize
block.initialize()

# Time integration
dt = block.max_time_step(vars)
vars = block.forward(dt, stage=0, vars=vars)
```

### Type Checking

```python
from typing import Dict
import torch
import snapy

def simulate(block: snapy.MeshBlock, vars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Type-checked simulation function."""
    dt = block.max_time_step(vars)
    return block.forward(dt, 0, vars)
```

Run type checking:
```bash
mypy simulation_script.py
```
