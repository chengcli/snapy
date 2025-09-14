import torch
import math
import time
import numpy as np
import yaml
import torch
import snapy
import kintera
import argparse
from typing import List
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

os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_IB_DISABLE", "0")
torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = True

def setup_moist_adiabatic_profile(config, coord, eos, thermo_x,
                                  device=torch.device("cpu")):
    Tmin = float(config["problem"]["Tmin"])
    grav = - float(config["forcing"]["const-gravity"]["grav1"])
    Ps = float(config["problem"]["Ps"])
    Ts = float(config["problem"]["Ts"])

    # dimensions
    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]
    ny = len(thermo_x.options.species()) - 1
    nvar = eos.nvar()

    temp = Ts * torch.ones((nc3, nc2), device=device)
    pres = Ps * torch.ones((nc3, nc2), device=device)
    xfrac = torch.zeros((nc3, nc2, 1 + ny), device=device)

    # read in compositions
    for i in range(1, 1 + ny):
        name = 'x' + thermo_x.options.species()[i]
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

    return w;

def evolve_kinetics(block_vars, eos, thermo_x, thermo_y, kinet, dt):
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

def main(args, rank, device):
    config = yaml.safe_load(open(args.input, "r"))

    # set hydrodynamic options
    op = MeshBlockOptions.from_yaml(args.input)
    block = MeshBlock(op)
    block.to(device)

    # get handles to modules
    coord = block.hydro.module("coord")
    thermo_y = block.hydro.module("eos.thermo")
    eos = block.hydro.get_eos()
    #thermo_y.options.max_iter(100)

    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    block_vars = {}
    interior = block.part((0, 0, 0))

    if "init_cond" in config["problem"]:
        nc3 = coord.buffer("x3v").shape[0]
        nc2 = coord.buffer("x2v").shape[0]
        nc1 = coord.buffer("x1v").shape[0]
        nvar = eos.nvar()
        block_vars["hydro_w"] = torch.zeros((nvar, nc3, nc2, nc1),
                                            device=device)

        module = torch.jit.load(config["problem"]["init_cond"])
        data = {name: param for name, param in module.named_buffers()}
        block_vars["hydro_w"][interior] = data["hydro_w"].to(device)
    else:
        block_vars["hydro_w"] = setup_moist_adiabatic_profile(
                config, coord, eos, thermo_x, device=device)

    # initialize
    block_vars = block.initialize(block_vars)

    out2 = NetcdfOutput(OutputOptions().file_basename("jupiter3d").fid(2).variable("prim"))
    out3 = NetcdfOutput(OutputOptions().file_basename("jupiter3d").fid(3).variable("uov"))
    out4 = NetcdfOutput(OutputOptions().file_basename("jupiter3d").fid(4).variable("diag"))

    # kinetics model
    op_kinet = KineticsOptions.from_yaml(args.input)
    kinet = Kinetics(op_kinet)
    kinet.to(device)

    # integration
    count, current_time, start_time = 0, 0, time.time()
    while not block.intg.stop(count, current_time):
        exchange_halos(block_vars)
        dt = block.max_time_step(block_vars)

        # make output
        if rank == 0 and count % args.print_every == 0:
            print(f"count = {count}, dt = {dt}, time = {current_time}", flush=True)
            u = block_vars["hydro_u"]
            print("mass = ", u[interior][index.idn].sum(), flush=True)

            qtol = block_vars["hydro_w"][index.icy:, :, :, :].sum(dim=0)
            block.set_uov("qtol", qtol)

            for out in [out2, out3, out4]:
                out.increment_file_number()
                out.write_output_file(block, block_vars, current_time)
                out.combine_blocks()

        # evolve dynamics
        for stage in range(len(block.intg.stages)):
            block.forward(dt, stage, block_vars)

        evolve_kinetics(block_vars, eos, thermo_x, thermo_y, kinet, dt)

        count += 1
        current_time += dt

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("-i", "--input", type=str,
                    help="path to input file", required=True)
    p.add_argument("-r", "--restart", type=str,
                    help="path to restart file")
    p.add_argument("-d", "--device", type=str, default="cpu",
                    choices=["cpu", "cuda"],
                    help="run and output dir")
    p.add_argument("--px1", type=int, default=1,
                    help="number of processes in x1")
    p.add_argument("--px2", type=int, default=1,
                    help="number of processes in x2")
    p.add_argument("--px3", type=int, default=1,
                    help="number of processes in x3")
    p.add_argument("--layout", type=str, default="slab",
                    choices=["slab", "cubed", "cubed_sphere"],
                    help="domain decomposition layout")
    p.add_argument("--print_every", type=int, default=1,
                    help="print running message every XX cycle")
    args = p.parse_args()

    ranks, device = init_dist(args)

    #ry, rx = cart_coords(rank, px, py)
    #up, down, left, right = neighbors_2d(rank, px, py)
    #main(args, rank, device)

    dist.destroy_process_group()
