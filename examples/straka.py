import torch
import math
import time
from snapy import *
from torch.profiler import profile, record_function, ProfilerActivity

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

nx1 = 64
nx2 = 256
# nx1 = 128
# nx2 = 512
nghost = 3

# set hydrodynamic options
op_coord = CoordinateOptions().nx1(nx1).nx2(nx2)
op_coord.x1min(0.0).x1max(6.4e3).x2min(0.0).x2max(25.6e3)

op_recon = ReconstructOptions().interp(InterpOptions("weno5")).shock(False)
op_thermo = ThermodynamicsOptions().gammad_ref(gamma).Rd(Rd)
op_eos = EquationOfStateOptions().thermo(op_thermo).type("ideal_gas")
op_riemann = RiemannSolverOptions().type("hllc")
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

L = torch.sqrt(((x2v - xc) / xr) ** 2 + ((x1v - zc) / zr) ** 2)
temp = Ts - grav * x1v / cp

w[index.ipr] = p0 * torch.pow(temp / Ts, cp / Rd)
temp += torch.where(L <= 1, dT * (torch.cos(L * math.pi) + 1.0) / 2.0, 0)
w[index.idn] = w[index.ipr] / (Rd * temp)

block.set_primitives(w)

# make output
out2 = NetcdfOutput(OutputOptions().file_basename("straka").fid(2).variable("prim"))
out3 = NetcdfOutput(OutputOptions().file_basename("straka").fid(3).variable("uov"))
current_time = 0.0

block.set_uov("temp", thermo.get_temp(w))
block.set_uov("theta", thermo.get_theta(w, p0))
for out in [out2, out3]:
    out.write_output_file(block, current_time)
    out.combine_blocks()

activities = [ProfilerActivity.CPU]

# integration
n = 0
start_time = time.time()
interior = block.part((0, 0, 0))
# with profile(activities=activities, record_shapes=True) as prof:
while True:
    dt = block.max_root_time_step()
    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage)
    current_time += dt
    if (n + 1) % 100 == 0:
        print("time = ", current_time)
        print("mass = ", block.hydro_u[interior][index.idn].sum())
        block.set_uov("temp", thermo.get_temp(block.var("hydro/w")))
        block.set_uov("theta", thermo.get_theta(block.var("hydro/w"), p0))
        for out in [out2, out3]:
            out.increment_file_number()
            out.write_output_file(block, current_time)
            out.combine_blocks()

    n += 1
    if current_time > 900:
        break

print("elapsed time = ", time.time() - start_time)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
