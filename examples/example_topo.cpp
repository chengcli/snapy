// C/C++
#include <random>

// yaml
#include <yaml-cpp/yaml.h>

// fmt
#include <fmt/format.h>

// torch
#include <torch/script.h>

// kintera
#include <kintera/constants.h>

#include <kintera/kinetics/evolve_implicit.hpp>
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/thermo/relative_humidity.hpp>

// snap
#include <snap/input/command_line.hpp>
#include <snap/mesh/mesh_formatter.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>

using namespace snap;

torch::Tensor gaussian_func(torch::Tensor x, torch::Tensor y, double x0,
                            double y0, double sigma) {
  return torch::exp(-((x - x0) * (x - x0) + (y - y0) * (y - y0)) /
                    (2 * sigma * sigma));
}

std::map<std::string, torch::Tensor> load_tensors(const std::string& filename) {
  std::map<std::string, torch::Tensor> tensor_map;

  // Load scripted module
  torch::jit::script::Module module = torch::jit::load(filename);

  // Get all named buffers
  for (const auto& p : module.named_buffers(/*recurse=*/false)) {
    tensor_map[p.name] = p.value;
  }

  // Optionally, also load parameters (if register_parameter was used)
  for (const auto& p : module.named_parameters(/*recurse=*/false)) {
    tensor_map[p.name] = p.value;
  }

  return tensor_map;
}

int main(int argc, char** argv) {
  // read parameters
  auto cli = CommandLine::ParseArguments(argc, argv);
  if (!cli) return 0;

  // input file
  auto infile = std::string(cli->input_filename);
  auto device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "Running on CUDA" << std::endl;
    device = torch::kCUDA;
  }

  // experiment name is before "."
  auto exp_name = infile.substr(0, infile.find('.'));

  auto config = YAML::LoadFile(infile);
  auto Ps = config["problem"]["Ps"].as<double>(1.e5);
  auto Ts = config["problem"]["Ts"].as<double>(300.);
  auto Tmin = config["problem"]["Tmin"].as<double>(200.);
  auto grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();

  // initialize the block
  auto block = MeshBlock(MeshBlockOptions::from_yaml(infile));
  std::cout << fmt::format("{}", block->options) << std::endl;
  block->to(device);

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
  int nvar = peos->nvar();

  /* add bottom topography
  auto result = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto x3v = result[0];
  auto x2v = result[1];
  auto x1v = result[2];

  std::random_device rd;   // Seed source
  std::mt19937 gen(rd());  // Mersenne Twister engine
  std::uniform_real_distribution<> dist(0.0, 1.0);

  auto topo = torch::zeros_like(x1v);
  std::cout << "topo shape = " << topo.sizes() << std::endl;

  for (int n = 0; n < 10; ++n) {
    auto x0 = dist(gen) * 10.e3;
    auto y0 = dist(gen) * 20.e3;
    auto sigma = 500. + dist(gen) * 1000.;
    auto height = 500. + dist(gen) * 500.;
    topo += height * gaussian_func(x2v, x3v, x0, y0, sigma);
  }*/
  // std::cout << "topo = " << topo.min() << ", " << topo.max() << std::endl;
  // auto solid = torch::where(x1v < topo, 1, 0);

  auto data =
      load_tensors("topo_20kmx10km_32.7719_32.9881_-106.5098_-106.3802.pt");
  auto solid = data["solid_80m"].to(device);

  int flips;
  solid = phydro->pib->rectify_solid(solid, flips, block->options.bfuncs());
  std::cout << "total number of flips = " << flips << std::endl;
  solid = torch::where(solid > 0, true, false);

  // construct an adiabatic atmosphere
  kintera::ThermoX thermo_x(thermo_y->options);
  thermo_x->to(device);

  auto temp =
      Ts *
      torch::ones({nc3, nc2},
                  torch::TensorOptions().dtype(torch::kDouble).device(device));

  auto pres =
      Ps *
      torch::ones({nc3, nc2},
                  torch::TensorOptions().dtype(torch::kDouble).device(device));

  auto xfrac =
      torch::zeros({nc3, nc2, 1 + ny},
                   torch::TensorOptions().dtype(torch::kDouble).device(device));

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  // read in compositions
  for (int i = 1; i <= ny; ++i) {
    auto name = thermo_y->options.species()[i];
    auto xmixr = config["problem"]["x" + name].as<double>(0.);
    xfrac.select(2, i) = xmixr;
  }
  // dry air mole fraction
  xfrac.select(2, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  // adiabatic extrapolate half a grid to cell center
  int is = pcoord->is();
  int ie = pcoord->ie();
  auto dz = pcoord->dx1f[is].item<double>();
  thermo_x->extrapolate_ad(temp, pres, xfrac, grav, dz / 2.);

  int i = is;
  int nvapor = thermo_x->options.vapor_ids().size();
  int ncloud = thermo_x->options.cloud_ids().size();
  for (; i <= ie; ++i) {
    // remove clouds
    auto cloud_frac = xfrac.narrow(-1, nvapor, ncloud).sum(-1, true);
    xfrac.narrow(-1, nvapor, ncloud) = 0.;
    xfrac /= (1. - cloud_frac);

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

  // add noise
  w[IVX] += 0.01 * torch::rand_like(w[IVX]);
  w[IVY] += 0.01 * torch::rand_like(w[IVY]);

  // initialize
  std::map<std::string, torch::Tensor> vars;
  vars["hydro_w"] = w;
  vars["solid"] = solid;
  block->initialize(vars);

  // user output variables
  // (1) total precipitable mass fraction [kg/kg]
  block->user_output_callback = [&](Variables const& vars) {
    auto w = vars.at("hydro_w");
    Variables out;
    out["qtol"] = w.narrow(0, ICY, ny).sum(0);
    return out;
  };

  // create kinetics model
  auto op_kinet = kintera::KineticsOptions::from_yaml(infile);
  auto kinet = kintera::Kinetics(op_kinet);
  kinet->to(device);
  std::cout << fmt::format("Kinetics Options:\n{}", kinet->options)
            << std::endl;

  // time loop
  double current_time = 0.;
  block->make_outputs(vars, current_time);

  while (!block->pintg->stop(block->cycle++, current_time)) {
    auto dt = block->max_time_step(vars);
    block->print_cycle_info(current_time, dt);

    // evolve dynamics
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(dt, stage, vars);
    }

    // evolve kinetics
    auto& hydro_u = vars["hydro_u"];
    auto& hydro_w = vars["hydro_w"];

    auto temp = peos->compute("W->T", {hydro_w});
    auto pres = hydro_w[IPR];
    auto xfrac = thermo_y->compute("Y->X", {hydro_w.narrow(0, ICY, ny)});
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
    auto cp_vol = thermo_x->compute("TV->cp", {temp, conc});

    // auto conc_kinet = kinet->options.narrow_copy(conc, thermo_y->options);
    auto conc_kinet = conc.slice(-1, 1, conc.size(-1));
    auto [rate, rc_ddC, rc_ddT] = kinet->forward(temp, pres, conc_kinet);
    auto jac = kinet->jacobian(temp, conc_kinet, cp_vol, rate, rc_ddC, rc_ddT);
    auto del_conc = kintera::evolve_implicit(rate, kinet->stoich, jac, dt);
    std::vector<int64_t> vec(del_conc.dim(), 1);
    vec[del_conc.dim() - 1] = -1;
    auto del_rho = del_conc / thermo_y->inv_mu.narrow(0, 1, ny).view(vec);
    hydro_u.narrow(0, ICY, ny) += del_rho.permute({3, 0, 1, 2});

    current_time += dt;
    block->make_outputs(vars, current_time);
  }

  CommandLine::Destroy();
}
