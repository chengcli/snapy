// C/C++
#include <string>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/coord/cubed_sphere_utils.hpp>
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

void initialize_block(MeshBlock block, Variables& vars,
                      YAML::Node const& config, torch::Device const& device) {
  auto phi = config["problem"]["phi"].as<double>();
  auto dphi = config["problem"]["dphi"].as<double>();
  auto radius = config["problem"]["radius"].as<double>();

  auto pcoord = block->pcoord;
  auto peos = block->phydro->peos;
  auto [rx, ry, face_id] =
      block->get_layout()->loc_of(block->get_layout()->options->rank());
  (void)rx;
  (void)ry;
  auto face = CS_FACE_NAMES[face_id];

  auto grid = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto alpha = grid[1];
  auto beta = grid[0];
  auto r_planet = grid[2];
  auto [lon, lat] = cs_ab_to_lonlat(face, alpha, beta);
  (void)lon;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = peos->nvar();

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  auto dist = (M_PI / 2. - lat) * r_planet;
  auto anomaly = torch::logical_and((dist < radius), (lat > M_PI / 4.));

  w[IDN] = torch::where(anomaly, phi + dphi, phi);
  w[IVX] = 0.;
  w[IVY] = 0.;

  vars["hydro_w"] = w;

  if (block->pscalar->nvar() > 0) {
    auto scalar_r = torch::zeros(
        {block->pscalar->nvar(), nc3, nc2, nc1},
        torch::TensorOptions().dtype(torch::kFloat64).device(device));
    scalar_r[0] = torch::where(anomaly, torch::ones_like(r_planet),
                               torch::zeros_like(r_planet));
    vars["scalar_r"] = scalar_r;
  }
}

}  // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  auto args = ParseArguments(argc, argv, "shallow_splash.yaml");
  auto config = YAML::LoadFile(args.input_file);

  auto mesh = Mesh(MeshOptionsImpl::from_yaml(args.input_file));
  auto device = torch::Device(mesh->options->device_str());
  if (device.is_cuda()) {
    std::cout << "Running on CUDA" << std::endl;
  }
  mesh->to(device);

  MeshVariables vars(mesh->blocks.size());
  if (args.restart_file.empty()) {
    for (size_t i = 0; i < mesh->blocks.size(); ++i) {
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
    if (redo > 0) continue;
    if (redo < 0) break;

    current_time += dt;
    mesh->make_outputs(vars, current_time);
  }

  mesh->finalize(vars, current_time);
}
