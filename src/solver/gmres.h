/**
 * @file gmres.h
 * @brief Parallel GMRES (Generalized Minimal Residual) solver with MPI support
 * 
 * This implementation provides a parallel GMRES iterative solver for solving
 * sparse linear systems Ax = b. The solver uses MPI for distributed memory
 * parallelization.
 */

#ifndef SRC_SOLVER_GMRES_H_
#define SRC_SOLVER_GMRES_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <mpi.h>

/**
 * @brief GMRES solver configuration structure
 */
typedef struct {
    int max_iter;           /**< Maximum number of GMRES iterations */
    int restart;            /**< Restart parameter (m in GMRES(m)) */
    double tol;             /**< Convergence tolerance */
    int verbose;            /**< Verbosity level (0=quiet, 1=basic, 2=detailed) */
    MPI_Comm comm;          /**< MPI communicator */
} gmres_config_t;

/**
 * @brief GMRES solver result structure
 */
typedef struct {
    int iterations;         /**< Number of iterations performed */
    double residual;        /**< Final residual norm */
    int converged;          /**< Convergence flag (1=converged, 0=not converged) */
} gmres_result_t;

/**
 * @brief Matrix-vector multiplication function pointer type
 * 
 * @param A User-defined matrix data structure
 * @param x Input vector
 * @param y Output vector (y = A*x)
 * @param n Local vector size
 * @param comm MPI communicator
 */
typedef void (*matvec_fn)(void *A, const double *x, double *y, int n, MPI_Comm comm);

/**
 * @brief Initialize GMRES configuration with default values
 * 
 * @param config Pointer to configuration structure
 * @param comm MPI communicator
 */
void gmres_config_init(gmres_config_t *config, MPI_Comm comm);

/**
 * @brief Parallel GMRES solver
 * 
 * Solves the linear system Ax = b using the GMRES(m) algorithm with MPI
 * parallelization. The matrix A is represented implicitly through a
 * matrix-vector multiplication function.
 * 
 * @param A User-defined matrix data structure
 * @param matvec Matrix-vector multiplication function
 * @param b Right-hand side vector (input)
 * @param x Solution vector (input: initial guess, output: solution)
 * @param n Local vector size (size on this MPI rank)
 * @param config GMRES configuration
 * @param result Pointer to result structure (can be NULL)
 * @return 0 on success, non-zero on error
 */
int gmres_solve(void *A, matvec_fn matvec, const double *b, double *x, 
                int n, const gmres_config_t *config, gmres_result_t *result);

/**
 * @brief Compute dot product of two distributed vectors
 * 
 * @param x First vector
 * @param y Second vector
 * @param n Local vector size
 * @param comm MPI communicator
 * @return Global dot product
 */
double gmres_dot(const double *x, const double *y, int n, MPI_Comm comm);

/**
 * @brief Compute L2 norm of a distributed vector
 * 
 * @param x Input vector
 * @param n Local vector size
 * @param comm MPI communicator
 * @return Global L2 norm
 */
double gmres_norm(const double *x, int n, MPI_Comm comm);

/**
 * @brief Apply Givens rotation
 * 
 * @param dx First element
 * @param dy Second element
 * @param cs Cosine component
 * @param sn Sine component
 */
void gmres_apply_givens(double *dx, double *dy, double cs, double sn);

/**
 * @brief Generate Givens rotation
 * 
 * @param dx First element (modified)
 * @param dy Second element (modified)
 * @param cs Cosine component (output)
 * @param sn Sine component (output)
 */
void gmres_generate_givens(double *dx, double *dy, double *cs, double *sn);

#ifdef __cplusplus
}
#endif

#endif  // SRC_SOLVER_GMRES_H_
