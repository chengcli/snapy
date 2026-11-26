# Snapy

Snapy is a finite-volume atmospheric dynamics solver built on PyTorch tensors. Its C++ core and Python bindings support compressible flow with thermodynamics and chemistry from Kintera, as well as shallow-water models on Cartesian and cubed-sphere grids.

[![PyPI version](https://badge.fury.io/py/snapy.svg)](https://pypi.org/project/snapy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Getting started

The release workflow targets CPython 3.10–3.13 on Linux x86-64 and macOS arm64. Install the Python package in an active environment with the matching PyTorch version.

```bash
python -m pip install 'torch==2.10.0' snapy
```

Each example pairs a YAML input deck with a driver. The deck selects the grid, numerical schemes, time integrator and outputs. The driver supplies the initial state and advances it. From the repository root, load the Sod shock-tube deck and construct its block in Python.

```python
import snapy

options = snapy.MeshBlockOptions.from_yaml("examples/shock.yaml")
block = snapy.MeshBlock(options)
```

The [shock driver](examples/shock.cpp) shows how to set the left and right gas states, initialize the block and advance Runge–Kutta stages. For a multi-block example, the [shallow-water driver](examples/shallow_xy.cpp) initializes each block and advances the mesh as a whole. The [example decks](examples/) include single-block and distributed versions of the same problem.

## How V3 advances a simulation

| Component | Role |
| --- | --- |
| [Mesh and MeshBlock](src/mesh/) | A block owns its local grid, physics and state tensors. A mesh coordinates its blocks, their exchanges and operations that need a global reduction. |
| [StageExecutor](src/mesh/stage_executor.hpp) | Schedules the operations in each integration stage. CPU and MPS execute eagerly. On CUDA, the default execution mode captures eligible stage segments per block and runs other segments eagerly. |
| [StateStore](src/mesh/state_store.hpp) and [PrecisionPolicy](src/mesh/precision_policy.hpp) | The store keeps conserved state and Runge–Kutta backups in stable buffers. The policy selects storage and computation dtypes for state, fluxes, reference fields and the vertical implicit solve. |
| [FluxRegister](src/hydro/flux_register.hpp) | Tracks face fluxes, corrections and cell sources so conservation budgets can account for transport across boundaries and changes within cells. |
| [Layout and exchange](src/layout/) | Map blocks to processes and exchange data across slab boundaries and cubed-sphere panel seams. |
| [Output](src/output/) | Writes requested fields and diagnostics. Restart handling lives with the [mesh block](src/mesh/meshblock_restart.cpp). |

### Choosing a configuration

Start from a deck with the geometry and physics you want. For example, [shock.yaml](examples/shock.yaml) selects an ideal gas, WENO5 reconstruction, HLLC fluxes and RK3 integration. Its two output entries give primitive fields and restarts independent write intervals.

- Set `dynamics.precision` to choose the block precision and `integration.implicit-precision` to control the vertical solve. The precision policy uses FP32 for quantities that would otherwise require FP64 on MPS.
- Set `integration.execution` to `auto`, `eager` or `cuda-graph`. In Python, `block.execution_capability(vars)` reports capture eligibility for the configured block and its initialized variables.
- Set `dynamics.flux-covariance` (default `true`) for the x2/x3 face-flux correction and `dynamics.x1-mass-covariance` (default `false`) for the reconstructed x1 velocity correction. Use the same settings on every block. The x1 correction needs at least three x1 cells per block.

The [option tables](src/options/) document keys and defaults for geometry, layout, output, hydro and individual forcings. CMake generates their C++ readers during configuration. Physics modules such as the [equations of state](src/eos/) also read their own options.

## Build and run from source

Use CMake 3.22 or newer, a C++17 compiler and the NetCDF C library. On Ubuntu, install the native build and test dependencies with the following command.

```bash
sudo apt-get install libnetcdf-dev libucx-dev nco autoconf automake libtool pkg-config
```

On macOS, use `brew install netcdf nco`. From the repository root, install the Python dependencies before configuring CMake so it can import PyTorch and generate the option readers with PyYAML.

```bash
python -m pip install numpy pyyaml wheel setuptools-scm pytest netcdf4 \
  'torch==2.10.0' 'kintera==2.6.13' 'pyharp>=2.7.17' \
  'commux>=0.1.4; platform_system == "Linux"' \
  'pinc>=0.2.0; platform_system == "Linux"'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA=OFF -DNETCDF=ON -DBUILD_TESTS=ON
cmake --build build --parallel 3
python -m pip install --no-build-isolation --no-deps .
```

This CPU build uses the Kintera version pinned in CI for its numerical references. Use `-DCUDA=ON` for a CUDA build with a compatible toolkit. Release builds place example executables and input decks in `build/bin`. For a quick CPU run, copy the single-block shallow-water deck and limit it to two steps. The driver writes NetCDF output in the run directory.

```bash
mkdir -p build/runs/shallow
cd build/runs/shallow
sed 's/nlim: -1/nlim: 2/' ../../bin/shallow_xy_single.yaml > shallow.yaml
BACKEND=gloo DEVICE=cpu pd-run 1 ../../bin/shallow_xy.release shallow.yaml
cd ../../..
```

## Tests

```bash
ctest --test-dir build/tests --output-on-failure
```

Configure with `-DFULL_TESTS=ON` to include additional reference and decomposition cases. The [test guide](tests/README.md) describes the platform-specific suites and how to test against a matching Python installation.

## License and contact

Snapy is distributed under the [MIT License](LICENSE). Contact Cheng Li at chengcli@umich.edu.
