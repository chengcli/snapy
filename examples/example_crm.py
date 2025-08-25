import torch
import math
import time
import numpy as np
import yaml
import torch
import snapy
import kintera
from snapy import (
        index,
        MeshBlockOptions,
        MeshBlock,
        OutputOptions,
        NetcdfOutput
        )
from kintera import (
        ThermoX,
        KineticsOptions,
        Kinetics,
        evolve_implicit
        )

torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    infile = "jupiter3d.yaml"
    config = yaml.safe_load(open(infile, "r"))

    Ps = float(config["problem"]["Ps"])
    Ts = float(config["problem"]["Ts"])
    Tmin = float(config["problem"]["Tmin"])
    grav = - float(config["forcing"]["const-gravity"]["grav1"])

    # device
    device = torch.device("cuda:0")

    # set hydrodynamic options
    op = MeshBlockOptions.from_yaml(infile);
    block = MeshBlock(op)
    block.to(device)

    # get handles to modules
    coord = block.hydro.module("coord")
    thermo_y = block.hydro.module("eos.thermo")
    eos = block.hydro.get_eos()

    # dimensions
    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]
    ny = len(thermo_y.options.species()) - 1
    nvar = eos.nvar()
    print("ny = ", ny)
    print("nvar = ", nvar)

    # set initial conditions
    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    temp = Ts * torch.ones((nc3, nc2), device=device)
    pres = Ps * torch.ones((nc3, nc2), device=device)
    xfrac = torch.zeros((nc3, nc2, 1 + ny), device=device)

    # read in compositions
    for i in range(1, 1 + ny):
        name = 'x' + thermo_y.options.species()[i]
        if name in config["problem"]:
            xfrac[:, :, i] = float(config["problem"][name])

    # dry air mole fraction
    xfrac[:, :, 0] = 1.0 - xfrac[:, :, 1:].sum(dim=2)

    # adiabatic extrapolate half a grid
    ifirst = coord.ifirst()
    dx1f = coord.buffer("dx1f")
    dz = dx1f[ifirst].item()
    thermo_x.extrapolate_ad(temp, pres, xfrac, grav, dz / 2.0)

    nvapor = len(thermo_x.options.vapor_ids())
    ncloud = len(thermo_x.options.cloud_ids())
    i = ifirst
    w = torch.zeros((nvar, nc3, nc2, nc1), device=device)
    while i < coord.ilast():
        conc = thermo_x.compute("TPX->V", [temp, pres, xfrac])

        w[index.ipr, :, :, i] = pres
        w[index.idn, :, :, i] = thermo_x.compute("V->D", [conc])
        w[index.icy:, :, :, i] = thermo_x.compute("X->Y", [xfrac])

        if (temp < Tmin).any().item():
            raise ValueError("Temperature below minimum")
        dz = dx1f[i].item()
        thermo_x.extrapolate_ad(temp, pres, xfrac, grav, dz)
        i += 1

    # isothermal extrapolation
    while i < coord.ilast():
        mu = (thermo_x.buffer("mu") * xfrac).sum(-1);
        dz = dx1f[i].item()
        pres *= torch.exp(-grav * mu * dz / (kintera.constants.Rgas * temp))
        conc = thermo_x.compute("TPX->V", [temp, pres, xfrac])
        w[index.ipr, :, :, i] = pres
        w[index.idn, :, :, i] = thermo_x.compute("V->D", [conc])
        w[index.icy:, :, :, i] = thermo_x.compute("X->Y", [xfrac])
        i += 1

    # add noise
    w[index.ivx] += 0.01 * torch.rand_like(w[index.ivx])
    w[index.ivy] += 0.01 * torch.rand_like(w[index.ivy])

    # initialize
    block_vars = {}
    block_vars["hydro_w"] = w
    block_vars = block.initialize(block_vars)

    out2 = NetcdfOutput(OutputOptions().file_basename("jupiter3d").fid(2).variable("prim"))
    out3 = NetcdfOutput(OutputOptions().file_basename("jupiter3d").fid(3).variable("uov"))
    out4 = NetcdfOutput(OutputOptions().file_basename("jupiter3d").fid(4).variable("diag"))

    # kinetics model
    op_kinet = KineticsOptions.from_yaml(infile)
    kinet = Kinetics(op_kinet)
    kinet.to(device)

    # integration
    count = 0
    start_time = time.time()
    interior = block.part((0, 0, 0))
    current_time = 0.
    while not block.intg.stop(count, current_time):
        dt = block.max_time_step(block_vars)

        # make output
        if count % 100 == 0:
            print(f"count = {count}, dt = {dt}, time = {current_time}")
            #u = block_vars["hydro_u"]
            #print("mass = ", u[interior][index.idn].sum())

            qtol = block_vars["hydro_w"][index.icy:, :, :, :].sum(dim=0)
            block.set_uov("qtol", qtol)

            for out in [out2, out3, out4]:
                out.increment_file_number()
                out.write_output_file(block, block_vars, current_time)
                out.combine_blocks()

        # evolve dynamics
        for stage in range(len(block.intg.stages)):
            block.forward(dt, stage, block_vars)

        # evolve kinetics
        hydro_u = block_vars["hydro_u"]
        hydro_w = block_vars["hydro_w"]

        temp = eos.compute("W->T", [hydro_w])
        pres = hydro_w[index.ipr]
        xfrac = thermo_y.compute("Y->X", [hydro_w[index.icy:, :, :, :]])
        conc = thermo_x.compute("TPX->V", [temp, pres, xfrac])
        cp_vol = thermo_x.compute("TV->cp", [temp, conc])

        conc_kinet = conc[:, :, :, 1:]
        rate, rc_ddC, rc_ddT = kinet.forward_nogil(temp, pres, conc_kinet)
        jac = kinet.jacobian(temp, conc_kinet, cp_vol, rate, rc_ddC, rc_ddT)
        del_conc = evolve_implicit(rate, kinet.buffer("stoich"), jac, dt)
        del_rho = del_conc / thermo_y.buffer("inv_mu")[1:].view(1, 1, 1, -1)
        hydro_u[index.icy:, :, :, :] += del_rho.permute(3, 0, 1, 2)

        count += 1
        current_time += dt
