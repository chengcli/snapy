import torch
import math
import time
from snapy import MeshBlockOptions, MeshBlock

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("plume.yaml")

# initialize block
block = MeshBlock(op)

# get handles to modules
coord = block.module("hydro.coord")

# setup a meshgrid for simulation
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

# dimensions
nc3 = coord.buffer("x3v").shape[0]
nc2 = coord.buffer("x2v").shape[0]
nc1 = coord.buffer("x1v").shape[0]
nvar = 4

w = torch.zeros((nvar, nc3, nc2, nc1))

#

block_vars = {}
block_vars["hydro_w"] = w
block_vars, current_time = block.initialize(block_vars)

# integration
start_time = time.time()
block.make_outputs(block_vars, current_time)

while not block.intg.stop(block.inc_cycle(), current_time):
    dt = block.max_time_step(block_vars)
    #block.print_cycle_info(block_vars, current_time, dt)
    print('time = ', current_time, ', dt = ', dt)

    for stage in range(len(block.intg.stages)):
        block.forward(block_vars, dt, stage)

    err = block.check_redo(block_vars)
    if err > 0:
        continue  # redo current step
    if err < 0:
        break  # terminate

    current_time += dt
    block.make_outputs(block_vars, current_time)

block.finalize(block_vars, current_time)
