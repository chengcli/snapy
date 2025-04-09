import torch
from canoe import *

start_logging("swxy")

# set domain size
nx1 = 512
nx2 = 256
nx3 = 1
nghost = 3

# set hydrodynamic options
op_coord = CoordinateOptions().nx1(nx1).nx2(nx2)
op_coord.x1min(-10.0).x1max(10.0).x2min(-5.0).x2max(5.0)

op_recon = ReconstructOptions().interp(InterpOptions("weno5")).shock(True)
op_eos = EquationOfStateOptions().type("shallow_water")
op_riemann = RiemannSolverOptions().type("shallow_roe")

op_hydro = HydroOptions().eos(op_eos).coord(op_coord).riemann(op_riemann)
op_intg = IntegratorOptions().type("rk3").cfl(0.9)
op_hydro.recon1(op_recon).recon23(op_recon)

op_block = MeshBlockOptions().nghost(nghost).intg(op_intg).hydro(op_hydro)
op_block.bflags([BoundaryFlag.kPeriodic] * 4)

# initialize block
block = MeshBlock(op_block)
block.to(torch.device("cuda:0"))
# block.to(torch.float32)

# get handles to modules
coord = block.hydro.module("coord")
eos = block.hydro.module("eos")

# set initial condition
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)
phi = 10.0
uphi = 10.0
dphi = 2.0

w = torch.zeros_like(block.hydro_u)
w[index.idn] = phi
w[index.ivy] = 0.0
w[index.idn][torch.logical_and(x1v > 0.0, x1v < 5.0)] += dphi
w[index.ivx] = torch.where(x2v > 0.0, -uphi / w[0], uphi / w[0])
block.set_primitives(w)

out = NetcdfOutput(OutputOptions().file_basename("swxy").variable("prim"))
current_time = 0.0

out.write_output_file(block, current_time)
out.combine_blocks()

# integration
for n in range(1000):
    dt = block.max_root_time_step()
    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage)

    current_time += dt
    if (n + 1) % 10 == 0:
        print("time = ", current_time)
        out.increment_file_number()
        out.write_output_file(block, current_time)
        out.combine_blocks()
# print("w = ", block.var("hydro_w")[0,0,0,:])
# print("flux1 = ", block.var("hydro_flux1")[0,0,0,:])
# print("div = ", block.var("hydro_div")[0,0,0,:])
