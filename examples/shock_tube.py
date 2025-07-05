import torch
from snapy import (
        index,
        MeshBlockOptions,
        MeshBlock,
        OutputOptions,
        NetcdfOutput
        )

# set domain size
nx1 = 512
nx2 = 512
nx3 = 1
nghost = 3

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("shock.yaml");

# initialize block
block = MeshBlock(op)
# block.to(torch.float32)
block.to(torch.device("cuda:0"))

# get handles to modules
coord = block.hydro.module("coord")

# set initial condition
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

w = block.buffer("hydro.eos.W")

w[index.idn] = torch.where(x1v < 0.0, 1.0, 0.125)
w[index.ipr] = torch.where(x1v < 0.0, 1.0, 0.1)
w[index.ivx] = w[index.ivy] = w[index.ivz] = 0.0

block.initialize(w)

out = NetcdfOutput(OutputOptions().file_basename("sod").variable("prim"))
current_time = 0.0

out.write_output_file(block, current_time)
out.combine_blocks()

count = 0;
current_time = 0.
while not block.intg.stop(count, current_time):
    dt = block.max_time_step()
    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage)

    current_time += dt
    count += 1
    if count % 10 == 0:
        print("count = ", count, " dt = ", dt, " time = ", current_time)
        out.increment_file_number()
        out.write_output_file(block, current_time)
        out.combine_blocks()
