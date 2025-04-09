import torch
from canoe import *

start_logging("explosion")

# set domain size
nx1 = 80
# nx2 = 80
nx2 = 160
# nx3 = 1
nx3 = 160
nghost = 3

Rd = 287.0

# set hydrodynamic options
op_coord = CoordinateOptions().nx1(nx1).nx2(nx2).nx3(nx3)
op_coord.x1min(0.0).x1max(1.0).x2min(-0.5).x2max(0.5).x3min(-0.5).x3max(0.5)

op_recon = ReconstructOptions().interp(InterpOptions("plm")).shock(True)
op_thermo = ThermodynamicsOptions().gammad_ref(1.4).Rd(Rd)
op_eos = EquationOfStateOptions().thermo(op_thermo).type("ideal_gas")
op_riemann = RiemannSolverOptions().type("hllc")
op_intg = IntegratorOptions().type("rk3").cfl(0.3)

op_hydro = HydroOptions().eos(op_eos).coord(op_coord).riemann(op_riemann)
op_hydro.recon1(op_recon).recon23(op_recon)

op_block = MeshBlockOptions().nghost(nghost).intg(op_intg).hydro(op_hydro)
op_block.bflags(
    [
        BoundaryFlag.kReflect,
        BoundaryFlag.kOutflow,
        BoundaryFlag.kOutflow,
        BoundaryFlag.kOutflow,
        BoundaryFlag.kOutflow,
        BoundaryFlag.kOutflow,
    ]
)

# initialize block
block = MeshBlock(op_block)
block.to(torch.device("cuda:0"))
# block.to(torch.float32)

# get handles to modules
coord = block.hydro.module("coord")
eos = block.hydro.module("eos")
thermo = eos.module("thermo")

# set initial condition
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

# set initial condition
p0 = 0.001
Ts = 10.0
a = 0.1
dT = 1000.0
dP = 1.0

w = torch.zeros_like(block.hydro_u)
temp = torch.zeros_like(w[0])
temp.fill_(Ts)

w[index.ipr].fill_(p0)

for n in range(5):
    z = 0.04 * torch.rand_like(temp) - 0.02
    x = 0.04 * torch.rand_like(temp) - 0.02 if block.nc2() > 1 else 0.0
    y = 0.04 * torch.rand_like(temp) - 0.02 if block.nc3() > 1 else 0.0
    r = torch.sqrt((x1v - z) ** 2 + (x2v - x) ** 2 + (x3v - y) ** 2)
    temp[r < a] = dT
    w[index.ipr][r < a] = dP

w[index.idn] = w[index.ipr] / (Rd * temp)
w[index.ivx] = w[index.ivy] = w[index.ivz] = 0.0

block.set_primitives(w)

# make output
out2 = NetcdfOutput(OutputOptions().file_basename("explosion").fid(2).variable("prim"))
out3 = NetcdfOutput(OutputOptions().file_basename("explosion").fid(3).variable("uov"))
current_time = 0.0

block.set_uov("temp", thermo.get_temp(w))
block.set_uov("theta", thermo.get_theta(w, p0))
for out in [out2, out3]:
    out.write_output_file(block, current_time)
    out.combine_blocks()

# integration
interior = block.part((0, 0, 0))
for n in range(1000):
    dt = block.max_root_time_step()
    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage)

    current_time += dt
    if (n + 1) % 10 == 0:
        print("dt = ", dt)
        print("time = ", current_time)
        print("mass = ", block.hydro_u[interior][index.idn].sum())
        block.set_uov("temp", thermo.get_temp(block.var("hydro/w")))
        block.set_uov("theta", thermo.get_theta(block.var("hydro/w"), p0))
        for out in [out2, out3]:
            out.increment_file_number()
            out.write_output_file(block, current_time)
            out.combine_blocks()
