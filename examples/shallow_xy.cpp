// C/C++
#include <string>

// yaml
#include <yaml-cpp/yaml.h>

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

void initialize_block(MeshBlock block, Variables& vars,
                      YAML::Node const& config, torch::Device const& device) {
  auto phi = config["problem"]["phi"].as<double>();
  auto uphi = config["problem"]["uphi"].as<double>();
  auto dphi = config["problem"]["dphi"].as<double>();
  auto x1min = config["geometry"]["bounds"]["x1min"].as<double>();
  auto x1max = config["geometry"]["bounds"]["x1max"].as<double>();
  auto x2min = config["geometry"]["bounds"]["x2min"].as<double>();
  auto x2max = config["geometry"]["bounds"]["x2max"].as<double>();

  auto pcoord = block->pcoord;
  auto peos = block->phydro->peos;

  auto grid = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto x1v = grid[2];
  auto x2v = grid[1];

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = peos->nvar();

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  w[IDN] =
      torch::where(torch::logical_and(x1v > 0., x1v < 5.), phi + dphi, phi);
  w[IVX] = torch::where(x2v > 0., -uphi / w[IDN], uphi / w[IDN]);
  w[IVY] = 0.;

  vars["hydro_w"] = w;

  if (block->pscalar->nvar() > 0) {
    auto scalar_r = torch::zeros(
        {block->pscalar->nvar(), nc3, nc2, nc1},
        torch::TensorOptions().dtype(torch::kFloat64).device(device));

    auto grad_x = (x1v - x1min) / (x1max - x1min);
    auto grad_y = (x2v - x2min) / (x2max - x2min);
    scalar_r[0] = (0.5 * (grad_x + grad_y)).clamp(0.0, 1.0);
    vars["scalar_r"] = scalar_r;
  }
}

}  // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  auto args = ParseArguments(argc, argv, "shallow_xy.yaml");
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
