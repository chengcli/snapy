// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/constants.h>

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
  w[IVY] += 1. * torch::randn_like(w[IVY]);
  w[IVZ] += 1. * torch::randn_like(w[IVZ]);

  // compute output variable
  // 2D -> 3D variables
  temp = peos->compute("W->T", {w});
  pres = w[IPR];
  xfrac = thermo_y->compute("Y->X", {w.narrow(0, ICY, ny)});

  // mole concentration [mol/m^3]
  auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});

  // volumetric entropy [J/(m^3 K)]
  auto entropy_vol = thermo_x->compute("TPV->S", {temp, pres, conc});

  // volumetric heat capacity [J/(m^3 K)]
  auto cp_vol = thermo_x->compute("TV->cp", {temp, conc});

  // molar entropy [J/(mol K)]
  auto entropy_mole = entropy_vol / conc.sum(-1);

  // molar heat capacity [J/(mol K)]
  auto cp_mole = cp_vol / conc.sum(-1);

  // mean molecular weight [kg/mol]
  auto mu = (thermo_x->mu * xfrac).sum(-1);

  // specific entropy [J/(kg K)]
  auto entropy = entropy_mole / mu;

  // potential temperature [K]
  auto theta = (entropy_vol / cp_vol).exp();

  // total precipitable mass fraction [kg/kg]
  auto qtol = w.narrow(0, ICY, ny).sum(0);

  // make initial output
  auto out2 = NetcdfOutput(
      OutputOptions().file_basename(exp_name).fid(2).variable("prim"));
  auto out3 = NetcdfOutput(
      OutputOptions().file_basename(exp_name).fid(3).variable("uov"));
  double current_time = 0.;

  block->user_out_var.insert("temp", temp);
  block->user_out_var.insert("entropy", entropy);
  block->user_out_var.insert("theta", theta);
  block->user_out_var.insert("qtol", qtol);

  out2.write_output_file(block, current_time, OctTreeOptions(), 0);
  out2.combine_blocks();

  out3.write_output_file(block, current_time, OctTreeOptions(), 0);
  out3.combine_blocks();
}
