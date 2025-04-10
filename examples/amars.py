import torch
import pyharp
import time
import numpy as np
from snapy import *

from amars_rt import calc_amars_rt, config_amars_rt


# set hydrodynamic options
def setup_hydro():
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
    return op_block


def set_initial_conditions(block):
    # get handles to modules
    coord = block.hydro.module("coord")

    # set initial condition
    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )

    # thermodynamics
    cp = gamma / (gamma - 1.0) * Rd

    w = torch.zeros_like(block.hydro_u)

    temp = Ts - grav * x1v / cp

    w[index.ipr] = pbot * torch.pow(temp / Ts, cp / Rd)
    w[index.idn] = w[index.ipr] / (Rd * temp)

    block.set_primitives(w)

    return w


if __name__ == "__main__":
    ncol, nlyr, nstr = 40, 80, 4
    pbot, ptop = 0.5e5, 100.0

    gamma = 1.4
    Rd = 287.0
    Ts = 300.0
    grav = 3.0

    nx1 = 80
    nx2 = 40
    nghost = 3

    temp = 200.0 * torch.ones((ncol, nlyr), dtype=torch.float64)
    pres = torch.logspace(np.log10(pbot), np.log10(ptop), nlyr, dtype=torch.float64)
    pres = pres.unsqueeze_(0).expand(ncol, -1).contiguous()

    rad, xfrac, atm, bc, dz = config_amars_rt(pres, temp, nstr=4)
    netflux, downward_flux, upward_flux = calc_amars_rt(
        rad, xfrac, pres, temp, atm, dz, bc
    )

    print("net flux = ", netflux)
    print("downward flux = ", downward_flux)
    print("upward flux = ", upward_flux)

    op_block = setup_hydro()

    # initialize block
    block = MeshBlock(op_block)
    # block.to(torch.float32)
    # block.to(torch.device("cuda:0"))

    # get handles to modules
    coord = block.hydro.module("coord")
    eos = block.hydro.module("eos")
    thermo = eos.module("thermo")

    # set initial condition
    w = set_initial_conditions(block)

    # make output
    out2 = NetcdfOutput(OutputOptions().file_basename("amars").fid(2).variable("prim"))
    out3 = NetcdfOutput(OutputOptions().file_basename("amars").fid(3).variable("uov"))
    current_time = 0.0

    block.set_uov("temp", thermo.get_temp(w))
    block.set_uov("theta", thermo.get_theta(w, pbot))
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

            # add radiative forcing
            w = block.var("hydro/w")[interior]
            temp = thermo.get_temp(w)
            pres = w[index.ipr]
            # netflux, downward_flux, upward_flux = calc_amars_rt(
            #        pres[0], temp[0], nstr=4)
            netflux, downward_flux, upward_flux = calc_amars_rt(
                rad, xfrac, pres[0], temp[0], atm, dz, bc
            )
            block.hydro_u[interior][index.ipr] -= (
                (netflux[:, 1:] - netflux[:, :-1]) / 1.0e5 * dt
            )

        current_time += dt
        if (n + 1) % 10 == 0:
            print("time = ", current_time)
            print("mass = ", block.hydro_u[interior][index.idn].sum())

            block.set_uov("temp", thermo.get_temp(block.var("hydro/w")))
            block.set_uov("theta", thermo.get_theta(block.var("hydro/w"), pbot))
            for out in [out2, out3]:
                out.increment_file_number()
                out.write_output_file(block, current_time)
                out.combine_blocks()

        n += 1
        if current_time > 900:
            break

    print("elapsed time = ", time.time() - start_time)
