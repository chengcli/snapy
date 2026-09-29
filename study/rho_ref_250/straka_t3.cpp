// #250 T3 study driver: examples/straka.cpp plus K lap(theta).
// C/C++
#include <string>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/species.hpp>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

struct RunConfig {
  std::string input_file;
  std::string restart_file;
};

RunConfig ParseArguments(int argc, char** argv,
                         std::string const& default_input) {
  RunConfig cfg{default_input, ""};
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if ((arg == "-r" || arg == "--restart") && i + 1 < argc) {
      cfg.restart_file = argv[++i];
    } else {
      cfg.input_file = arg;
    }
  }
  return cfg;
}

void set_user_output_callback(MeshBlock block, double p0, double Rd,
                              double cp) {
  block->user_output_callback = [Rd, cp, p0](Variables const& vars) {
    auto w = vars.at("hydro_w");
    auto temp = w[IPR] / (w[IDN] * Rd);

    Variables out;
    out["temp"] = temp;
    out["theta"] = temp * (p0 / w[IPR]).pow(Rd / cp);
    return out;
  };
}

//! Straka et al. (1993) diffuse theta, K lap(theta), not temperature: snapy's
//! kappa_iso is Fourier conduction of T, which on the dry-adiabatic
//! background heats the column (measured: 2.5 K in 900 s at 100 m). So the
//! study keeps nu_iso for momentum, sets kappa_iso = 0, and applies K
//! lap(theta) here after every full step (operator split, explicit;
//! K dt / dx^2 < 0.01 on these grids). Zero normal gradient at every wall,
//! ghosts take the mirrored update. The energy change is taken at fixed
//! density: T_new = T (theta_new / theta)^(cp / cv).
void diffuse_theta(MeshBlock block, Variables& vars, double K, double dt,
                   double p0, double Rd, double cp) {
  if (K <= 0.) return;
  auto pcoord = block->pcoord;
  auto peos = block->phydro->peos;
  int ng = pcoord->options->nghost();
  int il = pcoord->il(), iu = pcoord->iu();
  int jl = pcoord->jl(), ju = pcoord->ju();
  int n1 = iu - il + 1, n2 = ju - jl + 1;
  double dx1 = pcoord->dx1f[il].item<double>();
  double dx2 = pcoord->dx2f[jl].item<double>();
  double cv = cp - Rd;

  auto& u = vars.at("hydro_u");
  auto& w = vars.at("hydro_w");
  auto rho = w[IDN], pres = w[IPR];
  auto temp = pres / (rho * Rd);
  auto theta = temp * (p0 / pres).pow(Rd / cp);

  auto th = theta.narrow(-1, il, n1).narrow(-2, jl, n2);
  // Neumann: edge replication along both axes
  auto p1 = torch::cat({th.narrow(-1, 0, 1), th, th.narrow(-1, n1 - 1, 1)}, -1);
  auto p2 = torch::cat({th.narrow(-2, 0, 1), th, th.narrow(-2, n2 - 1, 1)}, -2);
  auto lap = (p1.narrow(-1, 0, n1) - 2. * th + p1.narrow(-1, 2, n1)) /
                 (dx1 * dx1) +
             (p2.narrow(-2, 0, n2) - 2. * th + p2.narrow(-2, 2, n2)) /
                 (dx2 * dx2);
  auto dth_in = dt * K * lap;

  // mirror the interior update into the ghosts (reflecting walls)
  auto dth = torch::zeros_like(theta);
  dth.narrow(-1, il, n1).narrow(-2, jl, n2).copy_(dth_in);
  dth.narrow(-1, 0, ng).copy_(dth.narrow(-1, il, ng).flip(-1));
  dth.narrow(-1, iu + 1, ng).copy_(dth.narrow(-1, iu + 1 - ng, ng).flip(-1));
  dth.narrow(-2, 0, ng).copy_(dth.narrow(-2, jl, ng).flip(-2));
  dth.narrow(-2, ju + 1, ng).copy_(dth.narrow(-2, ju + 1 - ng, ng).flip(-2));

  auto temp_new = temp * ((theta + dth) / theta).pow(cp / cv);
  u[IPR] += rho * cv * (temp_new - temp);
  peos->forward(u, w);
}

void initialize_block(MeshBlock block, Variables& vars,
                      YAML::Node const& config, torch::Device const& device) {
  auto p0 = config["problem"]["p0"].as<double>();
  auto Ts = config["problem"]["Ts"].as<double>();
  auto xc = config["problem"]["xc"].as<double>();
  auto zc = config["problem"]["zc"].as<double>();
  auto xr = config["problem"]["xr"].as<double>();
  auto zr = config["problem"]["zr"].as<double>();
  auto dT = config["problem"]["dT"].as<double>();
  auto grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();

  auto pcoord = block->pcoord;
  auto peos = block->phydro->peos;
  auto x1min = config["geometry"]["bounds"]["x1min"].as<double>();
  auto x1max = config["geometry"]["bounds"]["x1max"].as<double>();

  auto Rd = kintera::constants::Rgas / peos->species_weight();
  auto cv = peos->species_cv_ref();
  auto cp = cv + Rd;

  auto grids = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto x1v = grids[2];
  auto x2v = grids[1];

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = peos->nvar();

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  auto L = torch::sqrt(((x2v - xc) / xr).square() + ((x1v - zc) / zr).square());
  auto temp = Ts - grav * x1v / cp;

  w[IPR] = p0 * torch::pow(temp / Ts, cp / Rd);
  temp += torch::where(L <= 1, dT * (torch::cos(L * M_PI) + 1.) / 2., 0);
  w[IDN] = w[IPR] / (Rd * temp);

  vars["hydro_w"] = w;

  if (block->pscalar->nvar() > 0) {
    auto scalar_r = torch::zeros(
        {block->pscalar->nvar(), nc3, nc2, nc1},
        torch::TensorOptions().dtype(torch::kFloat64).device(device));

    // Initialize a passive tracer with a monotone vertical gradient.
    scalar_r[0] = ((x1v - x1min) / (x1max - x1min)).clamp(0.0, 1.0);
    vars["scalar_r"] = scalar_r;
  }

  set_user_output_callback(block, p0, Rd, cp);
}

}  // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  auto args = ParseArguments(argc, argv, "straka.yaml");
  auto config = YAML::LoadFile(args.input_file);

  auto mesh = Mesh(MeshOptionsImpl::from_yaml(args.input_file));
  auto device = torch::Device(mesh->options->device_str());
  if (device.is_cuda()) {
    std::cout << "Running on CUDA" << std::endl;
  }
  mesh->to(device);

  MeshVariables vars(mesh->blocks.size());
  double p0 = config["problem"]["p0"].as<double>();
  double K_theta = config["problem"]["K"].as<double>(0.);
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto peos = mesh->blocks[i]->phydro->peos;
    auto Rd = kintera::constants::Rgas / peos->species_weight();
    auto cp = peos->species_cv_ref() + Rd;
    set_user_output_callback(mesh->blocks[i], p0, Rd, cp);
    if (args.restart_file.empty()) {
      initialize_block(mesh->blocks[i], vars[i], config, device);
    }
  }

  double current_time = args.restart_file.empty()
                            ? mesh->initialize(vars)
                            : mesh->initialize(vars, args.restart_file.c_str());
  mesh->make_outputs(vars, current_time);

  int cycle = mesh->blocks.front()->cycle;
  while (!mesh->blocks.front()->pintg->stop(cycle, current_time)) {
    ++cycle;
    mesh->set_cycle(cycle);

    auto dt = mesh->max_time_step(vars);
    mesh->print_cycle_info(vars, current_time, dt);

    for (int stage = 0; stage < mesh->blocks.front()->pintg->stages.size();
         ++stage) {
      mesh->forward(vars, dt, stage);
    }

    int redo = mesh->check_redo(vars);
    if (redo > 0) {
      cycle = mesh->blocks.front()->cycle;
      continue;
    }
    if (redo < 0) break;

    for (size_t i = 0; i < mesh->blocks.size(); ++i) {
      auto peos = mesh->blocks[i]->phydro->peos;
      auto Rd = kintera::constants::Rgas / peos->species_weight();
      auto cp = peos->species_cv_ref() + Rd;
      diffuse_theta(mesh->blocks[i], vars[i], K_theta, dt, p0, Rd, cp);
    }

    current_time += dt;
    mesh->make_outputs(vars, current_time);
  }

  mesh->finalize(vars, current_time);
}
