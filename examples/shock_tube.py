import torch
from canoe import *

start_logging("sod")

# set domain size
nx1 = 512
nx2 = 512
nx3 = 1
nghost = 3

# set hydrodynamic options
op_coord = CoordinateOptions().nx1(nx1).nx2(nx2)
op_coord.x1min(-0.5).x1max(0.5)

op_recon = ReconstructOptions().interp(InterpOptions("plm")).shock(True)
op_thermo = ThermodynamicsOptions().gammad_ref(1.4)
op_eos = EquationOfStateOptions().thermo(op_thermo).type("ideal_gas")
op_riemann = RiemannSolverOptions().type("hllc")
op_intg = IntegratorOptions().type("rk3").cfl(0.9)

op_hydro = HydroOptions().eos(op_eos).coord(op_coord).riemann(op_riemann)
op_hydro.recon1(op_recon).recon23(op_recon)

op_block = MeshBlockOptions().nghost(nghost).intg(op_intg).hydro(op_hydro)
op_block.bflags(
    [
        BoundaryFlag.kReflect,
        BoundaryFlag.kOutflow,
        BoundaryFlag.kOutflow,
        BoundaryFlag.kOutflow,
    ]
)

# initialize block
block = MeshBlock(op_block)
# block.to(torch.float32)
block.to(torch.device("cuda:0"))

# get handles to modules
coord = block.hydro.module("coord")
eos = block.hydro.module("eos")

# set initial condition
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

w = torch.zeros_like(block.hydro_u)
w[index.idn] = torch.where(x1v < 0.0, 1.0, 0.125)
w[index.ipr] = torch.where(x1v < 0.0, 1.0, 0.1)
w[index.ivx] = w[index.ivy] = w[index.ivz] = 0.0

block.set_primitives(w)

out = NetcdfOutput(OutputOptions().file_basename("sod").variable("prim"))
current_time = 0.0

out.write_output_file(block, current_time)
out.combine_blocks()

for n in range(100):
    dt = block.max_root_time_step()
    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage)

    current_time += dt
    if (n + 1) % 10 == 0:
        print("time = ", current_time)
        out.increment_file_number()
        out.write_output_file(block, current_time)
        out.combine_blocks()
# print("w = ", block.var("hydro_w")[1,0,0,:])
# print("flux1 = ", block.var("hydro_flux1")[0,0,0,:])
# print("div = ", block.var("hydro_div")[0,0,0,:])
