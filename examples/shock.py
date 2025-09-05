import torch
from snapy import (
        index,
        MeshBlockOptions,
        MeshBlock,
        OutputOptions,
        NetcdfOutput
        )

torch.set_default_dtype(torch.float64)

# device
device = torch.device("cuda:0")

# set hydrodynamic model
op = MeshBlockOptions.from_yaml("shock.yaml");

# initialize block
block = MeshBlock(op)
block.to(device)

# get handles to modules
coord = block.hydro.module("coord")

# set initial condition
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

# dimensions
nc3 = coord.buffer("x3v").shape[0]
nc2 = coord.buffer("x2v").shape[0]
nc1 = coord.buffer("x1v").shape[0]
nvar = 5

w = torch.zeros((nvar, nc3, nc2, nc1), device=device)

w[index.idn] = torch.where(x1v < 0.0, 1.0, 0.125)
w[index.ipr] = torch.where(x1v < 0.0, 1.0, 0.1)
w[index.ivx] = w[index.ivy] = w[index.ivz] = 0.0

block_vars = {}
block_vars["hydro_w"] = w
block_vars = block.initialize(block_vars)

out = NetcdfOutput(OutputOptions().file_basename("sod").variable("prim"))

count = 0;
current_time = 0.
while not block.intg.stop(count, current_time):
    dt = block.max_time_step(block_vars)

    if count % 10 == 0:
        print(f"count = {count}, dt = {dt}, time = {current_time}")
        out.increment_file_number()
        out.write_output_file(block, block_vars, current_time)
        out.combine_blocks()

    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage, block_vars)

    count += 1
    current_time += dt
