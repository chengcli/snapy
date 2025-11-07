"""
GMRES (Generalized Minimal Residual) iterative solver in PyTorch.

This implementation is based on the Fortran code from:
https://github.com/aaronjridley/GITM/blob/476f28d7cf72a39bfc356ce1dce70f8ed9612533/share/Library/src/ModLinearSolver.f90

Original authors:
- Youcef Saad (May 23, 1985)
- Henk A. van der Vorst and Mike Botchev (Oct. 1996)
- Gabor Toth (May 2002, Dec 2006) - F90 and parallelization

PyTorch implementation with distributed support (2025).

Supports parallel execution using torch.distributed with gloo or nccl backends.
"""

import torch
import torch.distributed as dist
from typing import Callable, Optional, Tuple, Literal


def dot_product_distributed(
    x: torch.Tensor, y: torch.Tensor, process_group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """
    Compute distributed dot product of two vectors.
    
    Args:
        x: First vector
        y: Second vector
        process_group: Process group for distributed operations. If None, uses default group.
        
    Returns:
        Dot product as a scalar tensor
    """
    local_dot = torch.dot(x.flatten(), y.flatten())
    
    if dist.is_initialized():
        # All-reduce to get global dot product
        dist.all_reduce(local_dot, op=dist.ReduceOp.SUM, group=process_group)
    
    return local_dot


def gmres(
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
) -> Tuple[torch.Tensor, int, int, float]:
    """
    Solve linear system Ax = b using GMRES iterative method.
    
    The GMRES algorithm builds an orthogonal basis for the Krylov subspace
    and minimizes the residual over this subspace.
    
    Args:
        matvec: Function that computes matrix-vector product A @ x
        rhs: Right-hand side vector b
        x0: Initial guess for solution. If None, uses zero vector.
        n_krylov: Size of Krylov subspace (restart parameter)
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        stop_type: Type of stopping criterion:
            - 'rel': relative residual ||r|| <= tol * ||r0||
            - 'abs': absolute residual ||r|| <= tol
            - 'max': maximum residual max(|r|) <= tol
        use_initial_guess: Whether to use x0 as initial guess (if False, starts from zero)
        verbose: Print convergence information
        process_group: Process group for distributed operations
        
    Returns:
        Tuple of (solution, iterations, info_code, achieved_tolerance):
            - solution: Computed solution vector
            - iterations: Number of iterations performed
            - info_code: 
                * 0: converged to tolerance
                * 2: no convergence within max_iter (residual decreased)
                * -2: no convergence within max_iter (residual did not decrease)
                * 3: initial guess satisfies stopping criterion
            - achieved_tolerance: Final residual (relative or absolute based on stop_type)
    """
    device = rhs.device
    dtype = rhs.dtype
    n = rhs.numel()
    
    # Machine epsilon
    epsmac = 1e-16 if dtype == torch.float64 else 1e-8
    
    # Initialize solution
    if x0 is None:
        sol = torch.zeros_like(rhs).flatten()
        use_initial_guess = False
    else:
        sol = x0.clone().flatten()
    
    rhs_flat = rhs.flatten()
    
    # Allocate Krylov subspace vectors
    krylov = torch.zeros((n, n_krylov + 2), device=device, dtype=dtype)
    
    # Hessenberg matrix and auxiliary vectors
    hh = torch.zeros((n_krylov + 1, n_krylov), device=device, dtype=dtype)
    c = torch.zeros(n_krylov, device=device, dtype=dtype)
    s = torch.zeros(n_krylov, device=device, dtype=dtype)
    rs = torch.zeros(n_krylov + 1, device=device, dtype=dtype)
    
    its = 0  # iteration counter
    ro0 = 0.0  # initial residual norm
    
    if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
        print(f"GMRES: tol={tol}, max_iter={max_iter}, n_krylov={n_krylov}")
    
    # Outer restart loop
    converged = False
    info = 0
    
    while not converged and its < max_iter:
        # Compute initial residual: r = b - A*x
        if use_initial_guess or its > 0:
            ax = matvec(sol)
            krylov[:, 0] = rhs_flat - ax
        else:
            # Save a matvec when starting from zero
            krylov[:, 0] = rhs_flat
        
        # Compute residual norm
        ro = torch.sqrt(dot_product_distributed(krylov[:, 0], krylov[:, 0], process_group))
        
        if ro == 0.0:
            if its == 0:
                info = 3
            else:
                info = 0
            if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"GMRES: Residual is zero. info={info}")
            return sol, its, info, ro.item()
        
        # Set stopping tolerance
        if its == 0:
            ro0 = ro
            if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"GMRES: Initial residual norm = {ro0:.6e}")
            
            if stop_type == "abs":
                tol1 = tol
                if ro <= tol1:
                    info = 3
                    if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"GMRES: Initial guess satisfies stopping criterion. info={info}")
                    return sol, its, info, ro.item()
            else:
                tol1 = tol * ro0
        
        # Normalize first Krylov vector
        coeff = 1.0 / ro
        krylov[:, 0] = coeff * krylov[:, 0]
        
        # Initialize RHS of Hessenberg system
        rs[0] = ro
        
        # Inner Krylov loop
        for i in range(n_krylov):
            its += 1
            i1 = i + 1
            
            # Apply matrix-vector product
            krylov[:, i1] = matvec(krylov[:, i]).flatten()
            
            # Modified Gram-Schmidt orthogonalization
            for j in range(i1):
                t = dot_product_distributed(krylov[:, j], krylov[:, i1], process_group)
                hh[j, i] = t
                krylov[:, i1] = krylov[:, i1] - t * krylov[:, j]
            
            # Normalize new Krylov vector
            t = torch.sqrt(dot_product_distributed(krylov[:, i1], krylov[:, i1], process_group))
            hh[i1, i] = t
            
            if t != 0.0:
                krylov[:, i1] = krylov[:, i1] / t
            
            # Apply previous Givens rotations to i-th column of H
            for k in range(1, i1):
                k1 = k - 1
                t = hh[k1, i]
                hh[k1, i] = c[k1] * t + s[k1] * hh[k, i]
                hh[k, i] = -s[k1] * t + c[k1] * hh[k, i]
            
            # Compute next Givens rotation
            gam = torch.sqrt(hh[i, i]**2 + hh[i1, i]**2)
            if gam == 0.0:
                gam = epsmac
            
            c[i] = hh[i, i] / gam
            s[i] = hh[i1, i] / gam
            rs[i1] = -s[i] * rs[i]
            rs[i] = c[i] * rs[i]
            
            # Update Hessenberg matrix
            hh[i, i] = c[i] * hh[i, i] + s[i] * hh[i1, i]
            
            # Check convergence
            ro = torch.abs(rs[i1])
            
            if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
                if stop_type == "rel":
                    print(f"  Iteration {its}: ||r||/||r0|| = {(ro/ro0).item():.6e}")
                else:
                    print(f"  Iteration {its}: ||r|| = {ro.item():.6e}")
            
            # Check stopping criterion
            if ro <= tol1 or i + 1 >= n_krylov:
                # Solve upper triangular system: H*y = rs
                for jj in range(i, -1, -1):
                    if rs[jj] != 0.0:
                        rs[jj] = rs[jj] / hh[jj, jj]
                        tmp = rs[jj]
                        for k in range(jj - 1, -1, -1):
                            rs[k] = rs[k] - tmp * hh[k, jj]
                
                # Form solution: x = x0 + V*y
                for j in range(i1):
                    t = rs[j]
                    sol.add_(t * krylov[:, j])
                
                # Check if converged
                if ro <= tol1:
                    converged = True
                break
            
            if its >= max_iter:
                # Solve upper triangular system before exiting
                for jj in range(i, -1, -1):
                    if rs[jj] != 0.0:
                        rs[jj] = rs[jj] / hh[jj, jj]
                        tmp = rs[jj]
                        for k in range(jj - 1, -1, -1):
                            rs[k] = rs[k] - tmp * hh[k, jj]
                
                # Form solution
                for j in range(i1):
                    t = rs[j]
                    sol.add_(t * krylov[:, j])
                break
    
    # Compute final info code
    if ro <= tol1:
        info = 0
    elif ro < ro0:
        info = 2
    else:
        info = -2
    
    # Compute achieved tolerance
    if stop_type == "rel":
        achieved_tol = (ro / ro0).item()
    else:
        achieved_tol = ro.item()
    
    if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
        print(f"GMRES: Converged with info={info}, iterations={its}, tol={achieved_tol:.6e}")
    
    return sol.reshape_as(rhs), its, info, achieved_tol
