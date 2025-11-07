"""
Example demonstrating parallel GMRES solver with PyTorch distributed.

This script shows how to use the GMRES solver with torch.distributed
for parallel execution across multiple processes.

Usage:
    # Single process (serial)
    python examples/gmres_example.py
    
    # Multiple processes (parallel with gloo backend)
    torchrun --nproc_per_node=2 examples/gmres_example.py --distributed
    
    # Multiple processes (parallel with nccl backend, GPU required)
    torchrun --nproc_per_node=2 examples/gmres_example.py --distributed --backend=nccl
"""

import torch
import torch.distributed as dist
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from gmres import gmres


def create_test_system(n=100):
    """Create a test linear system Ax = b."""
    # Create a symmetric positive definite tridiagonal matrix
    A = torch.eye(n, dtype=torch.float64) * 4.0
    for i in range(n - 1):
        A[i, i + 1] = -1.0
        A[i + 1, i] = -1.0
    
    # Right-hand side
    b = torch.ones(n, dtype=torch.float64)
    
    return A, b


def serial_example():
    """Run GMRES in serial mode."""
    print("=" * 60)
    print("Serial GMRES Example")
    print("=" * 60)
    
    # Create test system
    n = 100
    A, b = create_test_system(n)
    
    # Matrix-vector product
    def matvec(x):
        return A @ x
    
    # Solve with GMRES
    print(f"\nSolving {n}x{n} system...")
    x, iterations, info, tol = gmres(
        matvec, b,
        n_krylov=30,
        tol=1e-8,
        max_iter=200,
        verbose=True
    )
    
    # Check solution
    residual = torch.norm(A @ x - b)
    print(f"\nResults:")
    print(f"  Iterations: {iterations}")
    print(f"  Info code: {info}")
    print(f"  Projected tolerance: {tol:.6e}")
    print(f"  Actual residual: {residual:.6e}")
    print("\n" + "=" * 60)


def parallel_example(backend='gloo'):
    """Run GMRES in parallel mode with torch.distributed."""
    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 60)
        print(f"Parallel GMRES Example (backend={backend})")
        print(f"  World size: {world_size}")
        print("=" * 60)
    
    # Create test system (same on all processes for simplicity)
    n = 100
    A, b = create_test_system(n)
    
    # Matrix-vector product (could be distributed in real application)
    def matvec(x):
        return A @ x
    
    # Solve with GMRES (collective operation)
    if rank == 0:
        print(f"\nSolving {n}x{n} system in parallel...")
    
    x, iterations, info, tol = gmres(
        matvec, b,
        n_krylov=30,
        tol=1e-8,
        max_iter=200,
        verbose=(rank == 0)
    )
    
    # Check solution
    residual = torch.norm(A @ x - b)
    
    if rank == 0:
        print(f"\nResults:")
        print(f"  Iterations: {iterations}")
        print(f"  Info code: {info}")
        print(f"  Projected tolerance: {tol:.6e}")
        print(f"  Actual residual: {residual:.6e}")
        print("\n" + "=" * 60)
    
    # Cleanup
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='GMRES solver example')
    parser.add_argument('--distributed', action='store_true',
                        help='Run in distributed mode')
    parser.add_argument('--backend', type=str, default='gloo',
                        choices=['gloo', 'nccl'],
                        help='Distributed backend to use')
    args = parser.parse_args()
    
    if args.distributed:
        parallel_example(backend=args.backend)
    else:
        serial_example()


if __name__ == "__main__":
    main()
