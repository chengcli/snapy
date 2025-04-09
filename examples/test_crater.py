#! /usr/bin/env python3
import h5py
import torch
from canoe import *

start_logging("crater")

nghost = 3

data = h5py.File("fvmcell.0170.h5", "r")
rho = data["density_gas"][:].reshape(256, 256, 256)
eng = data["E_gas"][:].reshape(256, 256, 256)
vel = data["v"][:].reshape(256, 256, 256, 3)
solid = data["ifCondensed"][:].reshape(256, 256, 256)


# set hydrodynamic options
op_coord = (
    CoordinateOptions()
    .nx1(256 - 2 * nghost)
    .nx2(256 - 2 * nghost)
    .nx3(256 - 2 * nghost)
)
op_coord.x1min(-0.0011).x1max(0.0011).x2min(-0.0007).x2max(0.0015).x3min(-0.0005).x3max(
    0.0017
)

op_recon = ReconstructOptions().interp(InterpOptions("plm")).shock(True)
op_thermo = ThermodynamicsOptions().gammad_ref(1.4).Rd(287.0)
op_eos = EquationOfStateOptions().thermo(op_thermo).type("ideal_gas").limiter(True)
op_riemann = RiemannSolverOptions().type("hllc")
op_intg = IntegratorOptions().type("rk3").cfl(0.1)
op_ib = (
    InternalBoundaryOptions().solid_density(2.7e3).solid_pressure(1.0e10).max_iter(20)
)

op_hydro = HydroOptions().eos(op_eos).coord(op_coord).riemann(op_riemann)
op_hydro.recon1(op_recon).recon23(op_recon).ib(op_ib)


op_block = MeshBlockOptions().nghost(nghost).intg(op_intg).hydro(op_hydro)
op_block.bflags([BoundaryFlag.kReflect] * 6)

# initialize block
block = MeshBlock(op_block)

# block.to(torch.float)
block.to(torch.device("cuda:0"))
interior = block.part((0, 0, 0))


# get handles to modules
coord = block.hydro.module("coord")
eos = block.hydro.module("eos")
thermo = eos.module("thermo")
ib = block.hydro.module("ib")

block.hydro_u[index.idn] = torch.from_numpy(rho)
print(block.hydro_u[index.idn].min(), block.hydro_u[index.idn].max())
block.hydro_u[index.ivx] = torch.from_numpy(rho * vel[:, :, :, 0])
print(block.hydro_u[index.ivx].min(), block.hydro_u[index.ivx].max())
block.hydro_u[index.ivy] = torch.from_numpy(rho * vel[:, :, :, 1])
print(block.hydro_u[index.ivy].min(), block.hydro_u[index.ivy].max())
block.hydro_u[index.ivz] = torch.from_numpy(rho * vel[:, :, :, 2])
print(block.hydro_u[index.ivz].min(), block.hydro_u[index.ivz].max())
block.hydro_u[index.ipr] = torch.from_numpy(eng)
print(block.hydro_u[index.ipr].min(), block.hydro_u[index.ipr].max())

out = block.hydro_u.clone()
w = eos.forward(block.hydro_u, out)
print(w[index.idn].min(), w[index.idn].max())
print(w[index.ivx].min(), w[index.ivx].max())
print(w[index.ivy].min(), w[index.ivy].max())
print(w[index.ivz].min(), w[index.ivz].max())
print(w[index.ipr].min(), w[index.ipr].max())
block.set_primitives(w)

solid = torch.from_numpy(solid).to(torch.device("cuda:0"))
solid, flips = ib.rectify_solid(solid, block.bfuncs)
print("flips = ", flips)
is_solid = solid > 0
w = ib.mark_solid(w, is_solid)

# make output
out2 = NetcdfOutput(OutputOptions().file_basename("crater").fid(2).variable("prim"))

for out in [out2]:
    out.write_output_file(block, 0.0)
    out.combine_blocks()

current_time = 0.0
for n in range(10):
    dt = block.max_root_time_step(0, is_solid)
    print("dt = ", dt)
    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage, solid=is_solid)
    print(block.hydro_u[index.idn].min(), block.hydro_u[index.idn].max())
    print(block.hydro_u[index.ipr].min(), block.hydro_u[index.ipr].max())

    current_time += dt
    if (n + 1) % 1 == 0:
        print("time = ", current_time)
        print("mass = ", block.hydro_u[interior][index.idn].sum())
        for out in [out2]:
            out.increment_file_number()
            out.write_output_file(block, current_time)
            out.combine_blocks()

# print("elapsed time = ", time.time() - start_time)
