/**
 * @file gmres_test.c
 * @brief Test program for parallel GMRES solver
 * 
 * This program tests the GMRES solver with a simple distributed matrix
 * representing a 1D Laplacian operator.
 */

#include "gmres.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief Matrix structure for a 1D Laplacian
 */
typedef struct {
    int n_local;      // Local size
    int n_global;     // Global size
    int offset;       // Global offset for this rank
    double h;         // Grid spacing
} laplacian_matrix_t;

/**
 * @brief Matrix-vector multiplication for 1D Laplacian with Dirichlet BCs
 * 
 * Computes y = A*x where A is the discretization of -d²/dx²:
 * A_ij = 2/h² if i==j, -1/h² if |i-j|==1, 0 otherwise
 * This is a symmetric positive definite matrix.
 */
void laplacian_matvec(void *A_data, const double *x, double *y, int n, MPI_Comm comm) {
    laplacian_matrix_t *A = (laplacian_matrix_t *)A_data;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    double scale = 1.0 / (A->h * A->h);
    
    // Handle boundary exchanges for parallel case
    double left_ghost = 0.0, right_ghost = 0.0;
    
    if (rank > 0) {
        // Send leftmost element to left neighbor, receive from left
        MPI_Sendrecv(&x[0], 1, MPI_DOUBLE, rank - 1, 0,
                     &left_ghost, 1, MPI_DOUBLE, rank - 1, 1,
                     comm, MPI_STATUS_IGNORE);
    }
    
    if (rank < size - 1) {
        // Send rightmost element to right neighbor, receive from right
        MPI_Sendrecv(&x[n - 1], 1, MPI_DOUBLE, rank + 1, 1,
                     &right_ghost, 1, MPI_DOUBLE, rank + 1, 0,
                     comm, MPI_STATUS_IGNORE);
    }
    
    // Apply stencil: (2*u[i] - u[i-1] - u[i+1])/h² 
    // This gives the operator for -d²u/dx² = f
    for (int i = 0; i < n; i++) {
        y[i] = 2.0 * x[i];
        
        // Left neighbor
        if (i > 0) {
            y[i] -= x[i - 1];
        } else if (rank > 0) {
            y[i] -= left_ghost;
        }
        // Left boundary condition (Dirichlet u=0) already satisfied by not subtracting
        
        // Right neighbor
        if (i < n - 1) {
            y[i] -= x[i + 1];
        } else if (rank < size - 1) {
            y[i] -= right_ghost;
        }
        // Right boundary condition (Dirichlet u=0) already satisfied by not subtracting
        
        y[i] *= scale;
    }
}

/**
 * @brief Compute L2 error
 */
double compute_error(const double *x, const double *x_exact, int n, MPI_Comm comm) {
    double local_error = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = x[i] - x_exact[i];
        local_error += diff * diff;
    }
    
    double global_error;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, comm);
    return sqrt(global_error);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Problem size
    int n_global = 100;
    if (argc > 1) {
        n_global = atoi(argv[1]);
    }
    
    // Distribute grid points among processes
    int n_local = n_global / size;
    int remainder = n_global % size;
    
    if (rank < remainder) {
        n_local++;
    }
    
    int offset = rank * (n_global / size) + (rank < remainder ? rank : remainder);
    
    // Setup grid
    double L = 1.0;  // Domain length
    double h = L / (n_global + 1);
    
    // Create matrix structure
    laplacian_matrix_t A;
    A.n_local = n_local;
    A.n_global = n_global;
    A.offset = offset;
    A.h = h;
    
    // Setup right-hand side: -u''(x) = sin(pi*x), u(0)=u(1)=0
    // Discretized: (2*u[i] - u[i-1] - u[i+1])/h² = sin(pi*x_i)
    // Matrix-vector: y = (2*x[i] - x[i-1] - x[i+1])/h²
    // So we solve: A*u = sin(pi*x_i) (no h² scaling needed!)
    // Exact solution: u(x) = sin(pi*x) / pi^2
    double *b = (double *)malloc(n_local * sizeof(double));
    double *x = (double *)malloc(n_local * sizeof(double));
    double *x_exact = (double *)malloc(n_local * sizeof(double));
    
    const double pi = 3.14159265358979323846;
    
    for (int i = 0; i < n_local; i++) {
        int global_i = offset + i;
        double xi = (global_i + 1) * h;
        b[i] = sin(pi * xi);  // No h² scaling!
        x_exact[i] = sin(pi * xi) / (pi * pi);
        x[i] = 0.0;  // Zero initial guess
    }
    
    // Configure GMRES
    gmres_config_t config;
    gmres_config_init(&config, MPI_COMM_WORLD);
    config.max_iter = 100;
    config.restart = 30;
    config.tol = 1e-10;
    config.verbose = (rank == 0) ? 2 : 0;  // Detailed output
    
    // Solve system
    gmres_result_t result;
    int status = gmres_solve(&A, laplacian_matvec, b, x, n_local, &config, &result);
    
    // Compute error
    double error = compute_error(x, x_exact, n_local, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\n");
        printf("========================================\n");
        printf("GMRES Test Results\n");
        printf("========================================\n");
        printf("Problem size:      %d\n", n_global);
        printf("MPI processes:     %d\n", size);
        printf("Converged:         %s\n", result.converged ? "Yes" : "No");
        printf("Iterations:        %d\n", result.iterations);
        printf("Final residual:    %.6e\n", result.residual);
        printf("L2 error:          %.6e\n", error);
        printf("Status:            %s\n", status == 0 ? "Success" : "Failed");
        printf("========================================\n");
        
        // Test passed if error is small (accounting for discretization error)
        if (error < 1e-3 && result.converged) {
            printf("TEST PASSED\n");
        } else {
            printf("TEST FAILED\n");
        }
    }
    
    // Cleanup
    free(b);
    free(x);
    free(x_exact);
    
    MPI_Finalize();
    return (status == 0 && error < 1e-3) ? 0 : 1;
}
