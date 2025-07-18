// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/mesh/mesh_formatter.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>

using namespace snap;

int main(int argc, char** argv) {
  // read parameters
  auto config = YAML::LoadFile("example_earth.yaml");
  auto Ps = config["problem"]["Ps"].as<double>(1.e5);
  auto Ts = config["problem"]["Ts"].as<double>(300.);
  auto xH2O = config["problem"]["xH2O"].as<double>(0.02);
  auto Tmin = config["problem"]["Tmin"].as<double>(200.);
  auto grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();

  // initialize the block
  auto block = MeshBlock(MeshBlockOptions::from_yaml("example_earth.yaml"));
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
  int iH2O = thermo_y->options.vapor_ids()[1];

  // construct an adiabatic atmosphere
  kintera::ThermoX thermo_x(thermo_y->options);

  auto temp = Ts * torch::ones({nc3, nc2}, torch::kDouble);
  auto pres = Ps * torch::ones({nc3, nc2}, torch::kDouble);
  auto xfrac = torch::zeros({nc3, nc2, 1 + ny}, torch::kDouble);
  xfrac.select(2, iH2O) = xH2O;    // water vapor
  xfrac.select(2, 0) = 1. - xH2O;  // dry air

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
  w[IVY] += 1. * torch::randn_like(w[IVY]);
  w[IVZ] += 1. * torch::randn_like(w[IVZ]);

  // compute output variable
  temp = peos->compute("W->T", {w});
  pres = w[IPR];
  xfrac = thermo_y->compute("Y->X", {w.narrow(0, ICY, ny)});

  auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
  auto entropy_vol = thermo_x->compute("TPV->S", {temp, pres, conc});
  auto cp_vol = thermo_x->compute("TV->cp", {temp, conc});

  auto mu = (thermo_x->mu * xfrac).sum(-1);
  auto entropy = entropy_vol / mu;
  auto theta = (entropy_vol / cp_vol).exp();

  // make initial output
  auto out2 = NetcdfOutput(
      OutputOptions().file_basename("earth").fid(2).variable("prim"));
  auto out3 = NetcdfOutput(
      OutputOptions().file_basename("earth").fid(3).variable("uov"));
  double current_time = 0.;

  std::cout << "temp shape = " << temp.sizes() << std::endl;
  std::cout << "entropy shape = " << entropy.sizes() << std::endl;
  std::cout << "theta shape = " << theta.sizes() << std::endl;

  block->user_out_var.insert("temp", temp);
  block->user_out_var.insert("entropy", entropy);
  block->user_out_var.insert("theta", theta);

  out2.write_output_file(block, current_time, OctTreeOptions(), 0);
  out2.combine_blocks();

  out3.write_output_file(block, current_time, OctTreeOptions(), 0);
  out3.combine_blocks();
}
