#!/usr/bin/env python3
"""
2D block domain decomposition of a 2D Jacobi stencil with GPU halos.
Neighbors: up/down/left/right. Uses PyTorch Distributed (NCCL) P2P ops.

Launch with torchrun (single/multi-node). Example Slurm file below.
"""

import os
import math
import argparse
import torch
import torch.distributed as dist

def init_cuda_dist():
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    ngpu = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, ngpu)))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device

def best_factors_2d(world_size: int) -> tuple[int, int]:
    """
    Choose nblock3 × nblock2 close to square: nblock3 * nblock2 == world_size.
    Returns (nblock3, nblock2).
    """
    root = int(math.sqrt(world_size))
    for nblock3 in range(root, 0, -1):
        if world_size % py == 0:
            nblock2 = world_size // nblock3
            return (nblock3, nblock2)
    return (1, world_size)

def cart_coords_2d(rank: int, nblock2: int) -> tuple[int, int]:
    """
    Map rank -> (r3, r2) with row-major layout:
    """
    r3 = rank // nblock2
    r2 = rank % nblock2
    return r3, r2

def split_1d(n_global: int, n_parts: int, part_idx: int) -> tuple[int, int, int]:
    """
    Block partition with remainder: the first 'rem' parts get an extra cell.
    Returns (start, end, local_n).
    """
    base = n_global // n_parts
    rem = n_global % n_parts
    local = base + (1 if part_idx < rem else 0)
    start = part_idx * base + min(part_idx, rem)
    end = start + local
    return start, end, local

def neighbors_2d(rank: int, px: int, py: int):
    up    = rank - px if ry > 0 else -1
    down  = rank + px if ry < py - 1 else -1
    left  = rank - 1  if rx > 0 else -1
    right = rank + 1  if rx < px - 1 else -1
    return up, down, left, right

def get_buffer_id_2d(x3_offset: int, x2_offset: int):
    return (x2_offset + 1) + (x3_offset + 1) * 3

def get_neighbor_rank_2d(nblock3: int, nblock2: int,
                         x3_offset: int, x2_offset: int,
                         periodic_x3 = False: bool,
                         periodic_x2 = False: bool):
    rank = dist.get_rank()
    r3, r2 = cart_coords(rank, nblock2)
    rank_x2 = rank + x2_offset
    if periodic_x2:
        rank_x2 = (rank_x2 + nblock2) % nblock2
    elif:
        pass

    rank_x3 = rank_x2 + x3_offset * nblock2
    if periodic_x3:
        rank_x3 = (rank_x3 + nblock3) % nblock3
    return rank_x3

def init_buffers_2d(block_vars: dict[str, torch.Tensor])
    send_bufs = [{}] * 9
    recv_bufs = [{}] * 9
    for x2_offset in [-1, 0, 1]:
        for x3_offset in [-1, 0, 1]:
            if x2_offset != 0 or x3_offset != 0:
                offset = (x3_offset, x2_offset, 0)
                bid = get_buffer_id(*offset)
                part = block.part(offset)
                send_bufs[bid]["hydro_u"] = torch.empty_like(block_vars["hydro_u"][part])
                recv_bufs[bid]["hydro_u"] = torch.empty_like(block_vars["hydro_u"][part])
    return send_bufs, recv_bufs

def serialize_2d(block_bufs, block_vars: dict[str, torch.Tensor])
    ranks = [] * 9
    for x2_offset in [-1, 0, 1]:
        for x3_offset in [-1, 0, 1]:
            if x2_offset != 0 or x3_offset != 0:
                offset = (x3_offset, x2_offset, 0)
                bid = get_buffer_id(*offset)
                part = block.part(offset)
                block_bufs[bid]["hydro_u"].copy_(block_vars["hydro_u"][part])
                rank[bid] = get_neighbor_rank_2d(

def deserialize(block_vars, block_vars: dict[str, torch.Tensor])
    for x2_offset in [-1, 0, 1]:
        for x3_offset in [-1, 0, 1]:
            if x2_offset != 0 or x3_offset != 0:
                offset = (x3_offset, x2_offset, 0)
                bid = get_buffer_id(*offset)
                part = block.part(offset)
                block_vars[bid]["hydro_u"][part].copy_(block_vars["hydro_u"][part])

def exchange_halos(block_vars: dict[str, torch.Tensor],
                   send_bufs, recv_bufs):
    ops = []

    serialize(send_bufs, block_vars):

    ops.append(dist.P2POp(dist.isend, send_bufs, left))
    ops.append(dist.P2POp(dist.irecv, , left))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    # Place received column halos into non-contiguous ghost columns
    if left != -1 or right != -1:
        write_column_ghosts_to_u()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=4096, help="global rows (y)")
    p.add_argument("--M", type=int, default=4096, help="global cols (x)")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--check_every", type=int, default=10)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--px", type=int, default=0, help="processes in x (cols); 0=auto")
    p.add_argument("--py", type=int, default=0, help="processes in y (rows); 0=auto")
    args = p.parse_args()

    rank, world_size, device = init_dist()

    # Global problem
    N, M = args.N, args.M

    # Local sizes (with remainders distributed)
    y0, y1, ny = split_1d(N, py, ry)  # rows (y)
    x0, x1, nx = split_1d(M, px, rx)  # cols (x)

    # Local array with 1-cell ghost around: shape = (ny+2, nx+2)
    # Indexing: [0,-1] are ghost rows/cols; interior [1:ny+1, 1:nx+1]
    u  = torch.zeros((ny + 2, nx + 2), dtype=torch.float32, device=device)
    un = torch.zeros_like(u)

    # Boundary conditions (Dirichlet):
    #   global top/bottom = 0.0 (y edges)
    #   global left/right = 1.0 (x edges)
    # Impose on our portion where we touch the global boundary.
    if ry == 0:         # touches top
        u[1, :] = 0.0
    if ry == py - 1:    # touches bottom
        u[-2, :] = 0.0
    if rx == 0:         # touches left
        u[:, 1] = 1.0
    if rx == px - 1:    # touches right
        u[:, -2] = 1.0
    un.copy_(u)

    # Handy views
    # Rows (contiguous):
    top_send    = u[1, 1:-1]   # to 'up'
    bottom_send = u[-2, 1:-1]  # to 'down'
    top_ghost   = u[0, 1:-1]   # from 'up'
    bottom_ghost= u[-1, 1:-1]  # from 'down'

    # Cols are NOT contiguous; stage via temporary contiguous buffers.
    left_send_buf   = u[1:-1, 1].contiguous()
    right_send_buf  = u[1:-1, -2].contiguous()
    left_recv_buf   = torch.empty_like(left_send_buf)
    right_recv_buf  = torch.empty_like(right_send_buf)

        if it % check_every == 0 or it == args.iters:
            r_local = (un[1:-1, 1:-1] - u[1:-1, 1:-1]).pow(2).sum()
            dist.all_reduce(r_local, op=dist.ReduceOp.SUM)
            res = torch.sqrt(r_local / (N * M)).item()
            if rank == 0 and it % args.print_every == 0:
                print(f"[iter {it:5d}] residual = {res:.3e}")
            if res < tol:
                if rank == 0:
                    print(f"Converged at iter {it} with residual {res:.3e}")
                break

        u, un = un, u

    # Simple checksum to verify correctness across ranks
    chk = u[1:-1, 1:-1].sum()
    dist.all_reduce(chk, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"Global checksum: {chk.item():.6e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    # os.environ.setdefault("NCCL_SOCKET_IFNAME", "ib0")  # set if your cluster needs it
    torch.backends.cudnn.benchmark = True
    main()
