import torch
import math
import time
from snapy import (
    index,
    MeshBlockOptions,
    MeshBlock,
    OutputOptions,
    NetcdfOutput
)

torch.set_default_dtype(torch.float64)

dT = 0.5
p0 = 1.0e5
Ts = 303.15
xc = 500.0
yc = 0.0
zc = 260.0
s = 100.0
a = 50.0
grav = 9.8
Rd = 287.0
gamma = 1.4
uniform_bubble = False

# device
device = torch.device("cuda:0")

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("robert.yaml");
block = MeshBlock(op)
block.to(device)

# get handles to modules
coord = block.hydro.module("coord")
thermo = block.hydro.module("eos.thermo")

# thermodynamics
cp = gamma / (gamma - 1.0) * Rd

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

temp = Ts - grav * x1v / cp
w[index.ipr] = p0 * torch.pow(temp / Ts, cp / Rd)

r = torch.sqrt((x3v - yc) ** 2 + (x2v - xc) ** 2 + (x1v - zc) ** 2)
temp += torch.where(r <= a, dT * torch.pow(w[index.ipr] / p0, Rd / cp), 0.0)
if not uniform_bubble:
    temp += torch.where(
        r > a,
        dT * torch.exp(-(((r - a) / s) ** 2)) * torch.pow(w[index.ipr] / p0, Rd / cp),
        0.0,
    )
w[index.idn] = w[index.ipr] / (Rd * temp)

block_vars = {}
block_vars["hydro_w"] = w
block_vars = block.initialize(block_vars)

# make output
# out1 = AsciiOutput(OutputOptions().file_basename("robert").fid(1).variable("hst"))
out2 = NetcdfOutput(OutputOptions().file_basename("robert").fid(2).variable("prim"))
out3 = NetcdfOutput(OutputOptions().file_basename("robert").fid(3).variable("uov"))

# integration
count = 0
start_time = time.time()
interior = block.part((0, 0, 0))
current_time = 0.
while not block.intg.stop(count, current_time):
    dt = block.max_time_step(block_vars)

    if count % 1000 == 0:
        print(f"count = {count}, dt = {dt}, time = {current_time}")
        u = block_vars["hydro_u"]
        print("mass = ", u[interior][index.idn].sum())

        ivol = thermo.compute("DY->V", (w[index.idn], w[index.icy:]))
        temp = thermo.compute("PV->T", (w[index.ipr], ivol))

        block.set_uov("temp", temp)
        block.set_uov("theta", temp * (p0 / w[index.ipr]).pow(Rd / cp))

        for out in [out2, out3]:
            out.increment_file_number()
            out.write_output_file(block, block_vars, current_time)
            out.combine_blocks()

    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage, block_vars)

    count += 1
    current_time += dt

print("elapsed time = ", time.time() - start_time)
