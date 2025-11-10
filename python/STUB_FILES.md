# Stub Files for Type Hints and Documentation

This directory contains Python stub files (`.pyi`) that provide type hints and API documentation for the `snapy` package, which is implemented using pybind11 C++ bindings.

## What are Stub Files?

Stub files are special Python files with a `.pyi` extension that contain type signatures and documentation for modules without including the actual implementation. They are used by:

- **Type checkers** (mypy, pyright, etc.) to verify type correctness
- **IDEs** (VS Code, PyCharm, etc.) to provide autocomplete and documentation
- **Documentation generators** to extract API information

## Structure

The stub files are organized in the `snapy/` subdirectory to match the compiled module structure:

- `snapy/__init__.pyi` - Main stub file with imports and common types (enums, type aliases)
- `snapy/boundary.pyi` - Boundary condition classes and functions
- `snapy/coordinate.pyi` - Coordinate system classes
- `snapy/eos.pyi` - Equation of state classes
- `snapy/forcing.pyi` - Forcing terms (gravity, Coriolis)
- `snapy/hydro.pyi` - Hydrodynamics classes
- `snapy/implicit.pyi` - Implicit solver classes
- `snapy/integrator.pyi` - Time integration classes
- `snapy/layout.pyi` - Domain decomposition layout classes
- `snapy/mesh.pyi` - Mesh block classes
- `snapy/output.pyi` - Output classes
- `snapy/reconstruction.pyi` - Reconstruction classes
- `snapy/riemann.pyi` - Riemann solver classes
- `py.typed` - Marker file indicating this package supports type hints (PEP 561)

This modular structure makes it easier to maintain and navigate the type definitions.

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

When modifying the C++ bindings in `python/csrc/`, remember to update the corresponding type signatures in the appropriate stub file:

1. Identify which module the change belongs to (e.g., boundary, hydro, mesh)
2. Update the corresponding `snapy/<module>.pyi` file
3. Add new classes or functions with type signatures
4. Update parameter types and return types
5. Include docstrings with examples
6. If adding cross-module dependencies, add necessary imports
7. Test with `mypy` to ensure validity

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

The stub files are split into logical modules:

- **`snapy/__init__.pyi`**: Main module with common enums (`index`, `BoundaryFace`) and type aliases (`bcfunc_t`), imports all submodules
- **`snapy/boundary.pyi`**: `BoundaryFuncOptions`, `InternalBoundaryOptions`, `InternalBoundary`
- **`snapy/coordinate.pyi`**: `CoordinateOptions`, `Cartesian`
- **`snapy/eos.pyi`**: `EquationOfStateOptions`, `EquationOfState`
- **`snapy/forcing.pyi`**: `ConstGravityOptions`, `CoriolisOptions`
- **`snapy/hydro.pyi`**: `HydroOptions`, `PrimitiveProjectorOptions`, `Hydro`
- **`snapy/implicit.pyi`**: `ImplicitOptions`, `ImplicitHydro`, `ImplicitCorrection`
- **`snapy/integrator.pyi`**: `IntegratorOptions`, `IntegratorWeight`, `Integrator`
- **`snapy/layout.pyi`**: `DistributeInfo`, `SlabLayout`, `CubedLayout`, `CubedSphereLayout`
- **`snapy/mesh.pyi`**: `MeshBlockOptions`, `MeshBlock`, `ScalarOptions`
- **`snapy/output.pyi`**: `OutputOptions`, `OutputType`, `NetcdfOutput`
- **`snapy/reconstruction.pyi`**: `InterpOptions`, `ReconstructOptions`, `Reconstruct`
- **`snapy/riemann.pyi`**: `RiemannSolverOptions`, `UpwindSolver`, `RoeSolver`, `LmarsSolver`, `ShallowRoeSolver`

Each module file contains type signatures using Python's `typing` module, overloaded methods using `@overload` decorator, and comprehensive docstrings with parameter descriptions.

### Package Configuration

The stub files are included in the package via `pyproject.toml`:

```toml
[tool.setuptools.package-data]
"snapy" = ["snap/**/*", "lib/*.so", "lib/*.dylib", "*.pyi", "py.typed"]
```

## References

- [PEP 561 - Distributing and Packaging Type Information](https://www.python.org/dev/peps/pep-0561/)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [mypy - Optional Static Typing for Python](http://mypy-lang.org/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)

## Contributing

When adding new features to the C++ bindings:

1. Implement the binding in the appropriate `python/csrc/*.cpp` file
2. Identify which logical module it belongs to (boundary, coordinate, eos, etc.)
3. Update the corresponding stub file in `snapy/<module>.pyi`
4. Add type signatures for new classes/methods
5. Include docstrings with parameter descriptions
6. Add cross-module imports if needed (e.g., `from .coordinate import CoordinateOptions`)
7. Run `mypy` to validate the stub file
8. Test that IDEs can properly autocomplete the new features

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
