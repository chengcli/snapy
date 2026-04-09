# Parallel GMRES Solver

This directory contains a standalone implementation of the GMRES (Generalized Minimal Residual) iterative solver with MPI support for solving sparse linear systems.

## Overview

GMRES is a Krylov subspace method for solving non-symmetric linear systems of equations:

```
Ax = b
```

This implementation includes:
- Full GMRES(m) algorithm with restart capability
- MPI parallelization for distributed memory systems
- Flexible matrix-free interface via function pointers
- Configurable convergence criteria and iteration limits

## Files

- `gmres.h` - Public API header file
- `gmres.c` - GMRES solver implementation
- `gmres_test.c` - Test program with 1D Laplacian example
- `Makefile` - Build system for standalone compilation
- `README.md` - This file

## Building

### Prerequisites

- MPI implementation (OpenMPI, MPICH, etc.)
- C compiler with C99 support
- Make

### Compilation

To build the library and test program:

```bash
make
```

This creates:
- `libgmres.a` - Static library
- `gmres_test` - Test executable

## Usage

### Basic Example

```c
#include "gmres.h"

// Define your matrix-vector multiplication
void my_matvec(void *A, const double *x, double *y, int n, MPI_Comm comm) {
    // Compute y = A*x
    // ...
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    // Problem setup
    int n = 1000;  // Local vector size
    double *b = ...; // Right-hand side
    double *x = ...; // Solution vector (initial guess)
    void *A = ...;   // Your matrix data structure
    
    // Configure GMRES
    gmres_config_t config;
    gmres_config_init(&config, MPI_COMM_WORLD);
    config.max_iter = 1000;
    config.restart = 30;
    config.tol = 1e-6;
    config.verbose = 1;
    
    // Solve
    gmres_result_t result;
    int status = gmres_solve(A, my_matvec, b, x, n, &config, &result);
    
    if (result.converged) {
        printf("Converged in %d iterations\n", result.iterations);
    }
    
    MPI_Finalize();
    return status;
}
```

### Configuration Parameters

The `gmres_config_t` structure allows customization:

- `max_iter`: Maximum number of outer iterations (default: 1000)
- `restart`: Restart parameter m in GMRES(m) (default: 30)
- `tol`: Relative residual tolerance for convergence (default: 1e-6)
- `verbose`: Output verbosity (0=quiet, 1=basic, 2=detailed)
- `comm`: MPI communicator to use

### Matrix-Vector Multiplication

The solver uses a function pointer interface for matrix-vector products:

```c
typedef void (*matvec_fn)(void *A, const double *x, double *y, int n, MPI_Comm comm);
```

Your function should:
1. Take the matrix data structure `A` and input vector `x`
2. Compute the product `y = A*x`
3. Handle any necessary MPI communication for distributed matrices
4. Store the result in vector `y`

## Testing

Run the included tests:

```bash
make test
```

This runs the 1D Laplacian test problem with 1, 2, and 4 MPI processes.

To run a custom test:

```bash
mpirun -np 4 ./gmres_test 200  # 200-point grid with 4 processes
```

## Algorithm Details

The implementation follows the standard GMRES(m) algorithm:

1. **Initialization**: Compute initial residual r₀ = b - Ax₀
2. **Arnoldi Iteration**: Build orthonormal basis for Krylov subspace
   - Modified Gram-Schmidt orthogonalization
   - Construct upper Hessenberg matrix H
3. **QR Factorization**: Apply Givens rotations to H
4. **Least Squares**: Solve minimization problem for coefficients
5. **Update**: Compute new approximation x_new = x + Σ(yᵢVᵢ)
6. **Check Convergence**: If ||r||/||b|| < tol, stop; else restart

### Parallelization

- Vectors are distributed across MPI processes
- Each process stores n_local elements
- Global operations (dot products, norms) use MPI_Allreduce
- Matrix-vector products must handle boundary communication
- The Hessenberg matrix is replicated on all processes (small size)

## Performance Considerations

- **Restart parameter**: Smaller m reduces memory but may increase iterations
  - Typical range: 20-50
- **Preconditioning**: For difficult problems, implement a preconditioner in matvec
- **Memory usage**: O(m×n_local) per process for Krylov vectors
- **Communication**: One MPI_Allreduce per Arnoldi iteration

## References

1. Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems. SIAM Journal on scientific and statistical computing, 7(3), 856-869.

2. Barrett, R., et al. (1994). Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods. SIAM.

3. Kelley, C. T. (1995). Iterative methods for linear and nonlinear equations. SIAM.

## License

This implementation is part of the snapy project. See LICENSE file in the repository root.
