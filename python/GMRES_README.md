# GMRES Solver for PyTorch

This implementation provides a parallel GMRES (Generalized Minimal Residual) iterative solver in PyTorch with distributed computing support.

## Overview

GMRES is a Krylov subspace method for solving large, sparse linear systems:

```
Ax = b
```

This implementation is based on the classic GMRES algorithm originally developed by Youcef Saad (1985) and later enhanced for parallel computing. The code follows the Fortran implementation from the SWMF library.

## Features

- **Pure PyTorch Implementation**: Leverages PyTorch tensors for GPU acceleration
- **Distributed Computing**: Supports parallel execution using `torch.distributed`
- **Multiple Backends**: Compatible with both `gloo` and `nccl` communication backends
- **Flexible Configuration**: Customizable Krylov subspace size, tolerance, and stopping criteria
- **Restart Capability**: Implements restarted GMRES for memory efficiency

## Installation

The GMRES solver is included in the `snapy` package:

```python
from snapy import gmres
```

## Usage

### Basic Example

```python
import torch
from snapy import gmres

# Define your linear system
n = 100
A = torch.eye(n) * 4.0
# ... setup your matrix A ...

b = torch.ones(n)

# Define matrix-vector product function
def matvec(x):
    return A @ x

# Solve the system
x, iterations, info, tolerance = gmres(
    matvec, b,
    n_krylov=30,
    tol=1e-8,
    max_iter=200,
    verbose=True
)

# Check solution
residual = torch.norm(A @ x - b)
print(f"Solution computed in {iterations} iterations")
print(f"Residual: {residual:.6e}")
```

### Parallel/Distributed Example

```python
import torch
import torch.distributed as dist
from snapy import gmres

# Initialize distributed process group
dist.init_process_group(backend='gloo')

# Define your matrix-vector product (can be distributed)
def matvec(x):
    # Your distributed matrix-vector multiplication
    return result

# Solve in parallel
x, iterations, info, tolerance = gmres(
    matvec, b,
    n_krylov=30,
    tol=1e-8,
    verbose=(dist.get_rank() == 0)
)

dist.destroy_process_group()
```

Run with multiple processes:
```bash
torchrun --nproc_per_node=4 your_script.py
```

## API Reference

### gmres()

```python
gmres(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    n_krylov: int = 30,
    tol: float = 1e-6,
    max_iter: int = 100,
    stop_type: Literal["rel", "abs", "max"] = "rel",
    use_initial_guess: bool = False,
    verbose: bool = False,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, int, int, float]
```

#### Parameters

- **matvec**: Function that computes the matrix-vector product `A @ x`
- **rhs**: Right-hand side vector `b`
- **x0**: Initial guess for the solution (optional)
- **n_krylov**: Size of Krylov subspace before restart (default: 30)
- **tol**: Convergence tolerance (default: 1e-6)
- **max_iter**: Maximum number of iterations (default: 100)
- **stop_type**: Stopping criterion type:
  - `"rel"`: Relative residual ||r|| ≤ tol * ||r0||
  - `"abs"`: Absolute residual ||r|| ≤ tol
  - `"max"`: Maximum residual max(|r|) ≤ tol
- **use_initial_guess**: Whether to use x0 as initial guess (default: False)
- **verbose**: Print convergence information (default: False)
- **process_group**: ProcessGroup for distributed operations (optional)

#### Returns

A tuple containing:
- **solution**: Computed solution vector
- **iterations**: Number of iterations performed
- **info_code**: Convergence status:
  - `0`: Converged to tolerance
  - `2`: No convergence but residual decreased
  - `-2`: No convergence and residual did not decrease
  - `3`: Initial guess satisfies stopping criterion
- **achieved_tolerance**: Final residual norm

## Examples

See `examples/gmres_example.py` for a complete working example.

## Algorithm Details

The GMRES algorithm:

1. Builds an orthonormal basis for the Krylov subspace using Arnoldi iteration
2. Uses Gram-Schmidt orthogonalization to maintain orthogonality
3. Applies Givens rotations to solve the least-squares problem
4. Restarts after building `n_krylov` vectors to limit memory usage

### Key Features

- **Modified Gram-Schmidt**: Provides better numerical stability
- **Givens Rotations**: Efficiently solves the upper Hessenberg system
- **Projected Residual**: Monitors convergence using the Krylov projection
- **Memory Efficient**: Restart capability limits memory to O(n * n_krylov)

## Performance Considerations

- **Krylov Size**: Larger `n_krylov` gives better convergence but uses more memory
- **Preconditioning**: For difficult systems, consider preconditioning the matrix
- **Projected vs Actual Residual**: The algorithm monitors projected residual, which may converge faster than the actual residual. For high accuracy, use tight tolerance or larger `n_krylov`.

## References

1. Y. Saad and M. H. Schultz (1986). "GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems." SIAM J. Sci. Stat. Comput. 7: 856-869.

2. H. A. van der Vorst and M. Botchev (1996). "Parallel Krylov solvers" 

3. G. Toth (2006). "SWMF ModLinearSolver" - Parallel implementation in Fortran

## License

This implementation follows the same license as the original SWMF library code.
