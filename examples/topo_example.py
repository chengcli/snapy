#! /usr/bin/env python
import torch
import math
import time
import numpy as np
from canoe import *


def gaussian_func(x, y, x0, y0, sigma):
    return torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


if __name__ == "__main__":
    start_logging("topo_example")

    # set hydrodynamic options
    op_coord = CoordinateOptions().nx1(90).nx2(90).nx3(90)
    op_coord.x1min(0.0).x1max(5.0e3).x2min(0.0).x2max(5.0e3).x3min(0.0).x3max(5.0e3)
    op_recon = ReconstructOptions().interp(InterpOptions("weno5")).shock(True)
    op_thermo = ThermodynamicsOptions().gammad_ref(1.4).Rd(287.0)
    op_eos = EquationOfStateOptions().thermo(op_thermo).type("ideal_gas")
    op_riemann = RiemannSolverOptions().type("hllc")
    op_grav = ConstGravityOptions().grav1(-9.8)
    op_proj = PrimitiveProjectorOptions().type("temperature")
    op_vic = VerticalImplicitOptions().scheme(0)
    op_intg = IntegratorOptions().type("rk3s4").cfl(0.9)
    op_ib = (
        InternalBoundaryOptions().solid_density(2.0).solid_pressure(2.0e5).max_iter(10)
    )
    op_hydro = HydroOptions().eos(op_eos).coord(op_coord).riemann(op_riemann)
    op_hydro.recon1(op_recon).recon23(op_recon).grav(op_grav).proj(op_proj).vic(
        op_vic
    ).ib(op_ib)
    op_block = MeshBlockOptions().nghost(3).intg(op_intg).hydro(op_hydro)
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

    # thermodynamics
    cp = thermo.get_cp()[0]

    # coordinates
    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )

    # add bottom boundary
    topo = torch.zeros_like(x3v)
    for n in range(10):
        x0 = np.random.uniform(0.5e3, 4.5e3)
        y0 = np.random.uniform(0.5e3, 4.5e3)
        sigma = np.random.uniform(100, 500)
        height = np.random.uniform(500, 2000)
        topo += height * gaussian_func(x2v, x3v, x0, y0, sigma)
    print("topo = ", topo.min(), topo.max())
    solid = torch.where(x1v < topo, 1, 0)
    solid, flips = ib.rectify_solid(solid, block.bfuncs)
    print("total number of flips = ", flips)
    # print("solid = ", solid)
    # print("solid shape = ", solid.shape)
    solid = torch.where(solid > 0, True, False)

    # set initial conditions
    p0 = 1.0e5
    Ts = 300.0

    xc = 0
    yc = 2.5e3
    zc = 2.0e3

    xr = 2.0e3
    yr = 2.0e3
    zr = 1.0e3

    dT = -15.0
    K = 75.0

    w = torch.zeros_like(block.hydro_u)

    L = torch.sqrt(
        ((x3v - yc) / yr) ** 2 + ((x2v - xc) / xr) ** 2 + ((x1v - zc) / zr) ** 2
    )
    temp = Ts + op_grav.grav1() * x1v / cp

    w[index.ipr] = p0 * torch.pow(temp / Ts, cp / op_thermo.Rd())
    temp += torch.where(L <= 1, dT * (torch.cos(L * math.pi) + 1.0) / 2.0, 0)
    w[index.idn] = w[index.ipr] / (op_thermo.Rd() * temp)

    w = ib.mark_solid(w, solid)
    block.set_primitives(w)

    # make output
    out2 = NetcdfOutput(OutputOptions().file_basename("topo").fid(2).variable("prim"))
    out3 = NetcdfOutput(OutputOptions().file_basename("topo").fid(3).variable("uov"))
    current_time = 0.0

    block.set_uov("temp", thermo.get_temp(w))
    block.set_uov("theta", thermo.get_theta(w, p0))
    for out in [out2, out3]:
        out.write_output_file(block, current_time)
        out.combine_blocks()

    # integration
    nlim = 10000
    tlim = 900.0
    start_time = time.time()

    for n in range(nlim):
        dt = block.max_root_time_step(0, solid)
        for stage in range(len(block.intg.stages)):
            block.forward(dt, stage, solid=solid)

        current_time += dt
        if (n + 1) % 500 == 0:
            print("time = ", current_time)
            print("mass = ", block.hydro_u[interior][index.idn].sum())
            block.set_uov("temp", thermo.get_temp(block.var("hydro/w")))
            block.set_uov("theta", thermo.get_theta(block.var("hydro/w"), p0))
            for out in [out2, out3]:
                out.increment_file_number()
                out.write_output_file(block, current_time)
                out.combine_blocks()

        if current_time > tlim:
            break

    print("elapsed time = ", time.time() - start_time)
