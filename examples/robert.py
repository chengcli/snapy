import torch
import math
import time
from snapy import *

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

nx1 = 300
nx2 = 200
nx3 = 1
nghost = 3

# set hydrodynamic options
op_coord = CoordinateOptions().nx1(nx1).nx2(nx2)
op_coord.x1min(0.0).x1max(1.5e3).x2min(0.0).x2max(1.0e3)

op_recon = ReconstructOptions().interp(InterpOptions("weno5").scale(True)).shock(False)
op_thermo = ThermodynamicsOptions().gammad_ref(gamma).Rd(Rd)
op_eos = EquationOfStateOptions().thermo(op_thermo).type("ideal_gas")
op_riemann = RiemannSolverOptions().type("lmars")
op_grav = ConstGravityOptions().grav1(-grav)
op_proj = PrimitiveProjectorOptions().type("temperature")
op_vic = VerticalImplicitOptions().scheme(0)
op_intg = IntegratorOptions().type("rk3").cfl(0.9)

op_hydro = HydroOptions().eos(op_eos).coord(op_coord).riemann(op_riemann)
op_hydro.recon1(op_recon).recon23(op_recon).grav(op_grav).proj(op_proj).vic(op_vic)

op_block = MeshBlockOptions().nghost(nghost).intg(op_intg).hydro(op_hydro)
op_block.bflags([BoundaryFlag.kReflect] * 4)

# initialize block
block = MeshBlock(op_block)
# block.to(torch.float32)
# block.to(torch.device("cuda:0"))

# get handles to modules
coord = block.hydro.module("coord")
eos = block.hydro.module("eos")
thermo = eos.module("thermo")

# thermodynamics
cp = gamma / (gamma - 1.0) * Rd

# set initial condition
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

w = torch.zeros_like(block.hydro_u)

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

block.set_primitives(w)

# make output
# out1 = AsciiOutput(OutputOptions().file_basename("robert").fid(1).variable("hst"))
out2 = NetcdfOutput(OutputOptions().file_basename("robert").fid(2).variable("prim"))
out3 = NetcdfOutput(OutputOptions().file_basename("robert").fid(3).variable("uov"))
current_time = 0.0

block.set_uov("temp", thermo.get_temp(w))
block.set_uov("theta", thermo.get_theta(w, p0))
for out in [out2, out3]:
    out.write_output_file(block, current_time)
    out.combine_blocks()

# integration
n = 0
start_time = time.time()
interior = block.part((0, 0, 0))
while True:
    dt = block.max_root_time_step()
    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage)
    current_time += dt
    if (n + 1) % 1000 == 0:
        print("time = ", current_time)
        print("mass = ", block.hydro_u[interior][0].sum())
        block.set_uov("temp", thermo.get_temp(block.var("hydro/w")))
        block.set_uov("theta", thermo.get_theta(block.var("hydro/w"), p0))
        for out in [out2, out3]:
            out.increment_file_number()
            out.write_output_file(block, current_time)
            out.combine_blocks()

    n += 1
    if current_time > 1080:
        break
print("elapsed time = ", time.time() - start_time)
