#include <stdio.h>
#include <math.h>
#include "gmres.h"
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);
    
    int n = 5;
    double h = 1.0 / (n + 1);
    double pi = 3.14159265358979323846;
    
    double *b = malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        double xi = (i + 1) * h;
        b[i] = h * h * sin(pi * xi);
    }
    
    double norm_b = gmres_norm(b, n, MPI_COMM_WORLD);
    
    printf("RHS vector b:\n");
    for (int i = 0; i < n; i++) {
        printf("b[%d] = %e\n", i, b[i]);
    }
    printf("||b|| = %e\n", norm_b);
    
    free(b);
    MPI_Finalize();
    return 0;
}
