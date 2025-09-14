import torch
import torch.distributed as dist
from snapy import (
        CoordinateOptions,
        HydroOptions,
        MeshBlock,
        MeshBlockOptions
        )
from snapy.exchange import (
        init_dist,
        init_buffers_2d,
        serialize_2d,
        slab_exchange
        )

def bc_null(*args):
    pass

class Args:
    device = 'cpu'
    px1 = 1
    px2 = 2
    px3 = 2
    layout = 'slab'

args = Args()
layout, ranks, device, info = init_dist(args,
                                        periodic_x1=False,
                                        periodic_x2=False,
                                        periodic_x3=False)
my_rank = ranks[0]

if my_rank == 0:
    print("Ranks:", ranks)
    print("layout = ", layout)

op_coord = CoordinateOptions().nx1(1).nx2(4).nx3(6).nghost(3)
#print("nghost = ", op_coord.nghost())
op_hydro = HydroOptions().coord(op_coord)
#print("nghost = ", op_hydro.coord().nghost())

op = MeshBlockOptions().hydro(op_hydro).bfuncs([bc_null]*6)
block = MeshBlock(op)

coord = block.hydro.module("coord")
nc1 = coord.buffer("x1v").shape[0]
nc2 = coord.buffer("x2v").shape[0]
nc3 = coord.buffer("x3v").shape[0]

if my_rank == 0:
    print("nc1, nc2, nc3 = ", nc1, nc2, nc3)

block_vars = {}
block_vars["hydro_u"] = torch.zeros((2, nc3, nc2, nc1), device=device)

for k in range(3, nc3 - 3):
    for j in range(3, nc2 - 3):
        block_vars["hydro_u"][0,k,j,:] = ((k - 3) * nc2 + (j - 3) + 1) * (my_rank + 1)
        block_vars["hydro_u"][1,k,j,:] = (100. + (k - 3) * nc2 + (j - 3) + 1) * (my_rank + 1)


send_bufs, recv_bufs = init_buffers_2d(layout, my_rank, block, block_vars)

if my_rank == 0:
    for i, s in enumerate(send_bufs):
        if s is not None:
            print(f"send_buf[{i}] shape = ", s[0].shape)

    for i, s in enumerate(recv_bufs):
        if s is not None:
            print(f"recv_buf[{i}] shape = ", s[0].shape)

if my_rank == 0:
    print("before exchange")
    print(block_vars["hydro_u"][:,:,:,0])

slab_exchange(block, block_vars, ranks, send_bufs, recv_bufs)

if my_rank == 0:
    print("after exchange")
    print(block_vars["hydro_u"][:,:,:,0])
    for b in block.options.bfuncs(): print(b)

dist.destroy_process_group()
