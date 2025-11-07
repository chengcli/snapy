#include "gmres.h"
#include <stdio.h>
#include <stdlib.h>

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
    double *Ax = malloc(n * sizeof(double));
    
    // Setup problem: -u'' = -sin(pi*x), u(0)=u(1)=0
    // Solution: u = sin(pi*x)/pi^2
    for (int i = 0; i < n; i++) {
        double xi = (i + 1) * h;
        b[i] = -sin(pi * xi);
        x[i] = sin(pi * xi) / (pi * pi);
    }
    
    // Test matrix-vector product
    matvec(&A, x, Ax, n, MPI_COMM_WORLD);
    
    printf("Testing matrix-vector product:\n");
    printf("h = %f, h² = %f\n", h, h*h);
    printf("i   x[i]        b[i]        Ax[i]       error\n");
    for (int i = 0; i < n; i++) {
        printf("%d  %.6f  %.6f  %.6f  %.6e\n", 
               i, x[i], b[i], Ax[i], Ax[i] - b[i]);
    }
    
    free(b);
    free(x);
    free(Ax);
    MPI_Finalize();
    return 0;
}
