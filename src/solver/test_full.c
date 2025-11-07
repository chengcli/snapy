#include "gmres.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int n;
    double h;
} laplacian_t;

void matvec(void *A_data, const double *x, double *y, int n, MPI_Comm comm) {
    laplacian_t *A = (laplacian_t *)A_data;
    double scale = 1.0 / (A->h * A->h);
    
    for (int i = 0; i < n; i++) {
        y[i] = 2.0 * x[i];
        if (i > 0) y[i] -= x[i - 1];
        if (i < n - 1) y[i] -= x[i + 1];
        y[i] *= scale;
    }
}

int main() {
    MPI_Init(NULL, NULL);
    
    int n = 5;
    double h = 1.0 / (n + 1);
    double pi = 3.14159265358979323846;
    
    laplacian_t A = {n, h};
    double *b = malloc(n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    double *r = malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        double xi = (i + 1) * h;
        b[i] = h * h * sin(pi * xi);
        x[i] = 0.0;
    }
    
    // Compute initial residual manually
    matvec(&A, x, r, n, MPI_COMM_WORLD);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - r[i];
    }
    
    double norm_b = gmres_norm(b, n, MPI_COMM_WORLD);
    double norm_r = gmres_norm(r, n, MPI_COMM_WORLD);
    
    printf("||b|| = %e\n", norm_b);
    printf("||r|| = %e\n", norm_r);
    printf("||r||/||b|| = %e\n", norm_r / norm_b);
    
    // Now solve
    gmres_config_t config;
    gmres_config_init(&config, MPI_COMM_WORLD);
    config.max_iter = 10;
    config.restart = 5;
    config.tol = 1e-10;
    config.verbose = 2;
    
    gmres_result_t result;
    gmres_solve(&A, matvec, b, x, n, &config, &result);
    
    printf("\nSolution x:\n");
    for (int i = 0; i < n; i++) {
        double xi = (i + 1) * h;
        double exact = sin(pi * xi) / (pi * pi);
        printf("x[%d] = %e (exact = %e, error = %e)\n", i, x[i], exact, x[i] - exact);
    }
    
    free(b);
    free(x);
    free(r);
    MPI_Finalize();
    return 0;
}
