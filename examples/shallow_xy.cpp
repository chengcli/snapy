// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/mesh/meshblock.hpp>

using namespace snap;

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  auto config = YAML::LoadFile("shallow_xy.yaml");

  auto phi = config["problem"]["phi"].as<double>();
  auto uphi = config["problem"]["uphi"].as<double>();
  auto dphi = config["problem"]["dphi"].as<double>();

  auto block_op = MeshBlockOptionsImpl::from_yaml("shallow_xy.yaml");
  auto block = MeshBlock(block_op);
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available() && block_op->layout()->backend() == "nccl") {
    std::cout << "Running on CUDA" << std::endl;
    device = block->get_layout()->pg->getBoundDeviceId().value();
  }

  block->to(device);

  // initial conditions
  auto pcoord = block->pcoord;
  auto peos = block->phydro->peos;

  // coordinates
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto x1v = mesh[2];
  auto x2v = mesh[1];

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

  // initialize
  std::map<std::string, torch::Tensor> vars;
  vars["hydro_w"] = w;
  block->initialize(vars);

  double current_time = 0.;
  block->make_outputs(vars, current_time);

  while (!block->pintg->stop(block->cycle++, current_time)) {
    auto dt = block->max_time_step(vars);
    block->print_cycle_info(vars, current_time, dt);

    // main loop
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(vars, dt, stage);
    }

    int err = block->check_redo(vars);
    if (err > 0) continue;  // redo this step with smaller dt
    if (err < 0) break;     // terminate simulation

    // make outputs
    current_time += dt;
    block->make_outputs(vars, current_time);
  }

  block->finalize(vars, current_time);
}
