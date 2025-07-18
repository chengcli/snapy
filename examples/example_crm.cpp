// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/constants.h>

#include <kintera/kinetics/evolve_implicit.hpp>
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/thermo/relative_humidity.hpp>

// snap
#include <snap/mesh/mesh_formatter.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>

using namespace snap;

int main(int argc, char** argv) {
  // read parameters
  std::string exp_name = "example_jupiter";

  auto config = YAML::LoadFile(fmt::format("{}.yaml", exp_name));
  auto Ps = config["problem"]["Ps"].as<double>(1.e5);
  auto Ts = config["problem"]["Ts"].as<double>(300.);
  auto Tmin = config["problem"]["Tmin"].as<double>(200.);
  auto grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();

  // initialize the block
  auto block =
      MeshBlock(MeshBlockOptions::from_yaml(fmt::format("{}.yaml", exp_name)));
  std::cout << fmt::format("MeshBlock Options: {}", block->options)
            << std::endl;

  // useful modules
  auto phydro = block->phydro;
  auto pcoord = phydro->pcoord;
  auto peos = phydro->peos;
  auto m = block->named_modules()["hydro.eos.thermo"];
  auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(m);

  // dimensions and indices
  int nc3 = pcoord->x3v.size(0);
  int nc2 = pcoord->x2v.size(0);
  int nc1 = pcoord->x1v.size(0);
  int ny = thermo_y->options.species().size() - 1;

  // construct an adiabatic atmosphere
  kintera::ThermoX thermo_x(thermo_y->options);

  auto temp = Ts * torch::ones({nc3, nc2}, torch::kDouble);
  auto pres = Ps * torch::ones({nc3, nc2}, torch::kDouble);
  auto xfrac = torch::zeros({nc3, nc2, 1 + ny}, torch::kDouble);

  // read in compositions
  for (int i = 1; i <= ny; ++i) {
    auto name = thermo_y->options.species()[i];
    auto xmixr = config["problem"]["x" + name].as<double>(0.);
    xfrac.select(2, i) = xmixr;
  }
  // dry air mole fraction
  xfrac.select(2, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  TORCH_CHECK(
      xfrac.sum(-1).allclose(torch::ones({nc3, nc2}, torch::kDouble), 1.e-6),
      "xfrac does not sum to 1");

  // set up initial conditions
  auto w = peos->get_buffer("W");

  // adiabatic extrapolate half a grid to cell center
  int is = pcoord->is();
  int ie = pcoord->ie();
  auto dz = pcoord->dx1f[is].item<double>();
  thermo_x->extrapolate_ad(temp, pres, xfrac, grav, dz / 2.);

  int i = is;
  for (; i <= ie; ++i) {
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});

    w[IPR].select(2, i) = pres;
    w[IDN].select(2, i) = thermo_x->compute("V->D", {conc});
    auto result = thermo_x->compute("X->Y", {xfrac});
    w.narrow(0, ICY, ny).select(3, i) = thermo_x->compute("X->Y", {xfrac});

    if ((temp < Tmin).any().item<double>()) break;
    dz = pcoord->dx1f[i].item<double>();
    thermo_x->extrapolate_ad(temp, pres, xfrac, grav, dz);
  }

  // isothermal extrapolation
  for (; i <= ie; ++i) {
    auto mu = (thermo_x->mu * xfrac).sum(-1);
    dz = pcoord->dx1f[i].item<double>();
    pres *= exp(-grav * mu * dz / (kintera::constants::Rgas * temp));
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
    w[IPR].select(2, i) = pres;
    w[IDN].select(2, i) = thermo_x->compute("V->D", {conc});
    w.narrow(0, ICY, ny).select(3, i) = thermo_x->compute("X->Y", {xfrac});
  }

  // populate ghost zones
  snap::BoundaryFuncOptions op;
  op.nghost(pcoord->options.nghost());
  op.type(kPrimitive);
  for (int i = 0; i < block->options.bfuncs().size(); ++i) {
    block->options.bfuncs()[i](w, 3 - i / 2, op);
  }

  // add noise
  w[IVX] += 1. * torch::rand_like(w[IVZ]);
  w[IVY] += 1. * torch::rand_like(w[IVY]);

  // populate the initial condition
  block->initialize(w);

  // total precipitable mass fraction [kg/kg]
  auto qtol = w.narrow(0, ICY, ny).sum(0);

  // make initial output
  auto out2 = NetcdfOutput(
      OutputOptions().file_basename(exp_name).fid(2).variable("prim"));
  auto out3 = NetcdfOutput(
      OutputOptions().file_basename(exp_name).fid(3).variable("uov"));
  auto out4 = NetcdfOutput(
      OutputOptions().file_basename(exp_name).fid(4).variable("diag"));
  double current_time = 0.;

  block->user_out_var.insert("qtol", qtol);

  out2.write_output_file(block, current_time, OctTreeOptions(), 0);
  out2.combine_blocks();

  out3.write_output_file(block, current_time, OctTreeOptions(), 0);
  out3.combine_blocks();

  out4.write_output_file(block, current_time, OctTreeOptions(), 0);
  out4.combine_blocks();

  // create kinetics model
  auto op_kinet =
      kintera::KineticsOptions::from_yaml(fmt::format("{}.yaml", exp_name));
  auto kinet = kintera::Kinetics(op_kinet);
  std::cout << fmt::format("Kinetics Options:\n{}", kinet->options)
            << std::endl;

  // time loop
  int count = 0;
  auto u = peos->get_buffer("U");

  while (!block->pintg->stop(count++, current_time)) {
    auto dt = block->max_time_step();
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(dt, stage);
    }

    // evolve kinetics
    auto temp = peos->compute("W->T", {w});
    auto pres = w[IPR];
    auto xfrac = thermo_y->compute("Y->X", {w.narrow(0, ICY, ny)});
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
    auto cp_vol = thermo_x->compute("TV->cp", {temp, conc});

    auto conc_kinet = kinet->options.narrow_copy(conc, thermo_y->options);
    auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc_kinet);
    auto jac = kinet->jacobian(temp, conc_kinet, cp_vol, rate, rc_ddC, rc_ddT);
    auto del_conc = kintera::evolve_implicit(rate, kinet->stoich, jac, dt);
    std::vector<int64_t> vec(del_conc.dim(), 1);
    vec[del_conc.dim() - 1] = -1;
    auto del_rho =
        del_conc.detach() / thermo_y->inv_mu.narrow(0, 1, ny).view(vec);
    u.narrow(0, ICY, ny) += del_rho.permute({3, 0, 1, 2});

    current_time += dt;
    if ((count + 1) % 10 == 0) {
      printf("count = %d, dt = %.6f, time = %.6f\n", count, dt, current_time);

      block->report_timer(std::cout);

      block->user_out_var["qtol"] = w.narrow(0, ICY, ny).sum(0);

      ++out2.file_number;
      out2.write_output_file(block, current_time, OctTreeOptions(), 0);
      out2.combine_blocks();

      ++out3.file_number;
      out3.write_output_file(block, current_time, OctTreeOptions(), 0);
      out3.combine_blocks();

      ++out4.file_number;
      out4.write_output_file(block, current_time, OctTreeOptions(), 0);
      out4.combine_blocks();
    }
  }
}
