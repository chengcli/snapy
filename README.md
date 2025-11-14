# Snapy

**Compressible Finite Volume Solver for Atmospheric Dynamics, Chemistry and Thermodynamics**

Snapy is a high-performance computational framework for simulating atmospheric and planetary dynamics using PyTorch tensors and GPU acceleration.

[![PyPI version](https://badge.fury.io/py/snapy.svg)](https://badge.fury.io/py/snapy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **GPU-Accelerated**: Built on PyTorch for efficient GPU computation
- **Flexible Interfaces**: Both Python and C++ APIs available
- **Compressible Flow**: Finite volume solver for atmospheric dynamics
- **Multi-platform**: Support for Linux and macOS
- **NetCDF Output**: Standard output format for scientific data

## Installation

### Quick Install (Python Interface)

The easiest way to get started is to install via pip:

```bash
pip install snapy
```

This will install the Python interface with pre-built binaries for Python 3.9-3.13 on Linux (x86_64) and macOS (ARM64).

**Requirements:**
- Python 3.9 or higher
- PyTorch 2.7.x
- NumPy
- kintera >= 1.1.5

### Build from Source (Advanced)

Building from source is recommended only for advanced users who need to:
- Modify the C++ core
- Use custom PyTorch versions
- Access the C++ interface directly
- Develop new features

**Prerequisites:**
- CMake 3.20+
- C++17 compatible compiler
- PyTorch 2.7.x with C++ libraries
- NetCDF C library
- kintera >= 1.1.5

**Build steps:**

1. Clone the repository:
```bash
git clone https://github.com/chengcli/snapy.git
cd snapy
```

2. Install dependencies:
```bash
pip install numpy kintera torch==2.7.1
```

3. Install NetCDF:
   - **Linux (Ubuntu/Debian):**
     ```bash
     sudo apt-get install libnetcdf-dev
     ```
   - **macOS:**
     ```bash
     brew install netcdf
     ```

4. Configure and build:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNETCDF=ON
cmake --build build --parallel 3
```

5. Install the Python package:
```bash
pip install .
```

## Quick Start

### Python Interface

Here's a simple example simulating a Sod shock tube problem:

```python
import torch
from snapy import index, MeshBlockOptions, MeshBlock

# Set up device
device = torch.device("cuda:0")  # or "cpu"
torch.set_default_dtype(torch.float64)

# Load configuration and create simulation block
op = MeshBlockOptions.from_yaml("shock.yaml")
block = MeshBlock(op)
block.to(device)

# Set up initial conditions
coord = block.hydro.module("coord")
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), 
    indexing="ij"
)

nc3, nc2, nc1 = x3v.shape[0], x2v.shape[0], x1v.shape[0]
w = torch.zeros((5, nc3, nc2, nc1), device=device)

# Sod shock tube: high pressure/density on left, low on right
w[index.idn] = torch.where(x1v < 0.0, 1.0, 0.125)
w[index.ipr] = torch.where(x1v < 0.0, 1.0, 0.1)

# Initialize and run
block_vars = {"hydro_w": w}
block_vars = block.initialize(block_vars)

current_time = 0.0
block.make_outputs(block_vars, current_time)

while not block.intg.stop(block.inc_cycle(), current_time):
    dt = block.max_time_step(block_vars)
    block.print_cycle_info(current_time, dt)
    
    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage, block_vars)
    
    current_time += dt
    block.make_outputs(block_vars, current_time)
```

### C++ Interface

```cpp
#include <snap/snap.h>
#include <snap/mesh/meshblock.hpp>

int main() {
    auto op = snap::MeshBlockOptions::from_yaml("shock.yaml");
    auto block = snap::MeshBlock(op);
    
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    block->to(device);
    
    // Set up initial conditions...
    std::map<std::string, torch::Tensor> vars;
    block->initialize(vars);
    
    // Run simulation...
    double current_time = 0.0;
    while (!block->pintg->stop(block->cycle++, current_time)) {
        auto dt = block->max_time_step(vars);
        for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
            block->forward(dt, stage, vars);
        }
        current_time += dt;
    }
    
    return 0;
}
```

## Examples

The `examples/` directory contains several working examples:

**Python Examples:**
- `shock.py` - Sod shock tube with internal boundary
- `straka.py` - Straka cold bubble convection test
- `robert.py` - Robert warm bubble convection test

**C++ Examples:**
- `shock.cpp` - Sod shock tube (C++)
- `straka.cpp` - Straka cold bubble (C++)

Run a Python example:
```bash
cd examples
python shock.py
```

Run a C++ example (after building):
```bash
cd build/examples
./shock
```

See `examples/README` for detailed documentation on the code structure and available examples.

## Configuration

Simulations are configured using YAML files that specify:
- Grid dimensions and domain size
- Time integration settings (RK stages, CFL number)
- Boundary conditions
- Output settings (frequency, variables, format)
- Equation of state and thermodynamics

Example configuration files (`.yaml`) are provided alongside the examples.

## Documentation

- **API Documentation**: [https://snapy.readthedocs.io](https://snapy.readthedocs.io)
- **Examples README**: See `examples/README` for detailed code walkthroughs
- **Project Homepage**: [https://github.com/chengcli/snapy](https://github.com/chengcli/snapy)

## Development

### Testing

Run tests after building:
```bash
cd build/tests
ctest --output-on-failure
```

### CI/CD

The project uses GitHub Actions for:
- **Continuous Integration**: Automated testing on Linux and macOS
- **Continuous Deployment**: Building and publishing wheels to PyPI

See `.github/workflows/` for pipeline configurations.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Snapy in your research, please cite:

```bibtex
@software{snapy,
  author = {Li, Cheng},
  title = {Snapy: Compressible Finite Volume Solver for Atmospheric Dynamics},
  year = {2025},
  url = {https://github.com/chengcli/snapy}
}
```

## Contact

- **Author**: Cheng Li
- **Email**: chengcli@umich.edu
- **GitHub**: [https://github.com/chengcli/snapy](https://github.com/chengcli/snapy)

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework and tensor library
- [Kintera](https://github.com/chengcli/kintera) - Thermodynamics and chemistry
- [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) - Scientific data format
