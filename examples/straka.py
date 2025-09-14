import torch
import math
import time
import kintera
import snapy
from snapy import (
        index,
        MeshBlockOptions,
        MeshBlock,
        OutputOptions,
        NetcdfOutput
        )

from torch.profiler import profile, record_function, ProfilerActivity

torch.set_default_dtype(torch.float64)

# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

p0 = 1.0e5
Ts = 300.0
xc = 0.0
xr = 4.0e3
zc = 3.0e3
zr = 2.0e3
dT = -15.0
grav = 9.8
Rd = 287.0
gamma = 1.4
K = 75.0

# device
#device = torch.device("cuda:0")
device = torch.device("cpu")

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("straka.yaml");

# initialize block
block = MeshBlock(op)
block.to(device)

# get handles to modules
coord = block.hydro.module("coord")
thermo = block.hydro.module("eos.thermo")
eos = block.hydro.module("eos")

# thermodynamics
Rd = kintera.constants.Rgas / kintera.species_weights()[0];
cv = kintera.species_cref_R()[0] * Rd;
cp = cv + Rd;

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

L = torch.sqrt(((x2v - xc) / xr) ** 2 + ((x1v - zc) / zr) ** 2)
temp = Ts - grav * x1v / cp

w[index.ipr] = p0 * torch.pow(temp / Ts, cp / Rd)
temp += torch.where(L <= 1, dT * (torch.cos(L * math.pi) + 1.0) / 2.0, 0)
w[index.idn] = w[index.ipr] / (Rd * temp)

block_vars = {}
block_vars["hydro_w"] = w
block_vars = block.initialize(block_vars)

# make output
out2 = NetcdfOutput(OutputOptions().file_basename("straka").fid(2).variable("prim"))
out3 = NetcdfOutput(OutputOptions().file_basename("straka").fid(3).variable("uov"))

block.set_uov("temp", temp)
block.set_uov("theta", temp * (p0 / w[index.ipr]).pow(Rd / cp))

activities = [ProfilerActivity.CPU]

# integration
count = 0;
start_time = time.time()
interior = block.part((0, 0, 0))
current_time = 0.0

# with profile(activities=activities, record_shapes=True) as prof:
while not block.intg.stop(count, current_time):
    dt = block.max_time_step(block_vars)

    if count % 100 == 0:
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
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
