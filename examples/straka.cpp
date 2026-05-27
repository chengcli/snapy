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

    current_time += dt;
    mesh->make_outputs(vars, current_time);
  }

  mesh->finalize(vars, current_time);
}
