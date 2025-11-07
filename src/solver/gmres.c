/**
 * @file gmres.c
 * @brief Implementation of parallel GMRES solver with MPI support
 */

#include "gmres.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

void gmres_config_init(gmres_config_t *config, MPI_Comm comm) {
    config->max_iter = 1000;
    config->restart = 30;
    config->tol = 1e-6;
    config->verbose = 0;
    config->comm = comm;
}

double gmres_dot(const double *x, const double *y, int n, MPI_Comm comm) {
    double local_sum = 0.0;
    double global_sum = 0.0;
    
    for (int i = 0; i < n; i++) {
        local_sum += x[i] * y[i];
    }
    
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_sum;
}

double gmres_norm(const double *x, int n, MPI_Comm comm) {
    return sqrt(gmres_dot(x, x, n, comm));
}

void gmres_apply_givens(double *dx, double *dy, double cs, double sn) {
    double temp = cs * (*dx) + sn * (*dy);
    *dy = -sn * (*dx) + cs * (*dy);
    *dx = temp;
}

void gmres_generate_givens(double *dx, double *dy, double *cs, double *sn) {
    double temp, r;
    
    if (*dy == 0.0) {
        *cs = 1.0;
        *sn = 0.0;
    } else if (fabs(*dy) > fabs(*dx)) {
        temp = (*dx) / (*dy);
        *sn = 1.0 / sqrt(1.0 + temp * temp);
        *cs = temp * (*sn);
    } else {
        temp = (*dy) / (*dx);
        *cs = 1.0 / sqrt(1.0 + temp * temp);
        *sn = temp * (*cs);
    }
    
    r = (*cs) * (*dx) + (*sn) * (*dy);
    *dx = r;
    *dy = 0.0;
}

/**
 * @brief Vector operations
 */
static void vec_copy(double *dest, const double *src, int n) {
    memcpy(dest, src, n * sizeof(double));
}

static void vec_scale(double *x, double alpha, int n) {
    for (int i = 0; i < n; i++) {
        x[i] *= alpha;
    }
}

static void vec_axpy(double *y, double alpha, const double *x, int n) {
    for (int i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

static void vec_zero(double *x, int n) {
    memset(x, 0, n * sizeof(double));
}

/**
 * @brief Back substitution for upper triangular system
 */
static void back_solve(const double *H, const double *s, double *y, int m) {
    for (int i = m - 1; i >= 0; i--) {
        y[i] = s[i];
        for (int j = i + 1; j < m; j++) {
            y[i] -= H[i * (m + 1) + j] * y[j];
        }
        y[i] /= H[i * (m + 1) + i];
    }
}

int gmres_solve(void *A, matvec_fn matvec, const double *b, double *x,
                int n, const gmres_config_t *config, gmres_result_t *result) {
    int rank;
    MPI_Comm_rank(config->comm, &rank);
    
    // Parameters
    const int max_iter = config->max_iter;
    const int m = config->restart;
    const double tol = config->tol;
    const int verbose = config->verbose;
    
    // Allocate workspace
    double **V = (double **)malloc((m + 1) * sizeof(double *));
    double *H = (double *)calloc((m + 1) * (m + 1), sizeof(double));
    double *s = (double *)malloc((m + 1) * sizeof(double));
    double *cs = (double *)malloc((m + 1) * sizeof(double));
    double *sn = (double *)malloc((m + 1) * sizeof(double));
    double *y = (double *)malloc(m * sizeof(double));
    double *w = (double *)malloc(n * sizeof(double));
    double *r = (double *)malloc(n * sizeof(double));
    
    if (!V || !H || !s || !cs || !sn || !y || !w || !r) {
        if (rank == 0 && verbose > 0) {
            fprintf(stderr, "GMRES: Memory allocation failed\n");
        }
        // Free allocated memory
        free(V); free(H); free(s); free(cs); free(sn); free(y); free(w); free(r);
        return -1;
    }
    
    for (int i = 0; i <= m; i++) {
        V[i] = (double *)malloc(n * sizeof(double));
        if (!V[i]) {
            if (rank == 0 && verbose > 0) {
                fprintf(stderr, "GMRES: Memory allocation failed for V[%d]\n", i);
            }
            for (int j = 0; j < i; j++) free(V[j]);
            free(V); free(H); free(s); free(cs); free(sn); free(y); free(w); free(r);
            return -1;
        }
    }
    
    double normb = gmres_norm(b, n, config->comm);
    if (normb == 0.0) normb = 1.0;
    
    int total_iter = 0;
    int converged = 0;
    double final_residual = 0.0;
    
    // Main GMRES loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Compute residual: r = b - A*x
        matvec(A, x, w, n, config->comm);
        for (int i = 0; i < n; i++) {
            r[i] = b[i] - w[i];
        }
        
        double beta = gmres_norm(r, n, config->comm);
        final_residual = beta / normb;
        
        if (rank == 0 && verbose > 1) {
            printf("GMRES restart %d: residual = %e\n", iter, final_residual);
        }
        
        if (final_residual < tol) {
            converged = 1;
            break;
        }
        
        // Initialize Krylov subspace: V[0] = r / beta
        vec_copy(V[0], r, n);
        vec_scale(V[0], 1.0 / beta, n);
        
        // Initialize RHS of least squares problem
        vec_zero(s, m + 1);
        s[0] = beta;
        
        // Arnoldi iteration
        int j;
        for (j = 0; j < m && total_iter < max_iter; j++, total_iter++) {
            // w = A * V[j]
            matvec(A, V[j], w, n, config->comm);
            
            // Modified Gram-Schmidt orthogonalization
            for (int i = 0; i <= j; i++) {
                H[i * (m + 1) + j] = gmres_dot(w, V[i], n, config->comm);
                vec_axpy(w, -H[i * (m + 1) + j], V[i], n);
            }
            
            H[(j + 1) * (m + 1) + j] = gmres_norm(w, n, config->comm);
            
            if (H[(j + 1) * (m + 1) + j] != 0.0) {
                vec_copy(V[j + 1], w, n);
                vec_scale(V[j + 1], 1.0 / H[(j + 1) * (m + 1) + j], n);
            }
            
            // Apply previous Givens rotations to new column
            for (int i = 0; i < j; i++) {
                gmres_apply_givens(&H[i * (m + 1) + j], 
                                  &H[(i + 1) * (m + 1) + j], 
                                  cs[i], sn[i]);
            }
            
            // Generate and apply new Givens rotation
            gmres_generate_givens(&H[j * (m + 1) + j], 
                                 &H[(j + 1) * (m + 1) + j], 
                                 &cs[j], &sn[j]);
            gmres_apply_givens(&s[j], &s[j + 1], cs[j], sn[j]);
            
            // Check for convergence
            final_residual = fabs(s[j + 1]) / normb;
            
            if (rank == 0 && verbose > 1) {
                printf("  iter %d: residual = %e\n", total_iter + 1, final_residual);
            }
            
            if (final_residual < tol) {
                converged = 1;
                j++; // Include this iteration in the update
                break;
            }
        }
        
        // Update solution: x = x + V * y
        back_solve(H, s, y, j);
        for (int i = 0; i < j; i++) {
            vec_axpy(x, y[i], V[i], n);
        }
        
        if (converged) {
            break;
        }
    }
    
    if (rank == 0 && verbose > 0) {
        if (converged) {
            printf("GMRES converged in %d iterations. Final residual: %e\n", 
                   total_iter, final_residual);
        } else {
            printf("GMRES did not converge in %d iterations. Final residual: %e\n",
                   total_iter, final_residual);
        }
    }
    
    // Store results
    if (result != NULL) {
        result->iterations = total_iter;
        result->residual = final_residual;
        result->converged = converged;
    }
    
    // Free memory
    for (int i = 0; i <= m; i++) {
        free(V[i]);
    }
    free(V);
    free(H);
    free(s);
    free(cs);
    free(sn);
    free(y);
    free(w);
    free(r);
    
    return converged ? 0 : 1;
}
