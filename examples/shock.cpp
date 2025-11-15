// torch
#include <torch/script.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh_formatter.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>
#include <snap/utils/signal_handler.hpp>

using namespace snap;

int main(int argc, char** argv) {
  auto op = MeshBlockOptions::from_yaml("shock.yaml");
  auto block = MeshBlock(op);

  auto device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "Running on CUDA" << std::endl;
    device = torch::kCUDA;
  }

  std::cout << fmt::format("MeshBlock Options:\n{}", block->options)
            << std::endl;

  block->to(device);

  // initial conditions
  auto pcoord = block->phydro->pcoord;
  auto peos = block->phydro->peos;

  // coordinates
  auto grids = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto x1v = grids[2];
  auto x2v = grids[1];
  auto x3v = grids[0];

  int nc1 = pcoord->options.nc1();
  int nc2 = pcoord->options.nc2();
  int nc3 = pcoord->options.nc3();
  int nvar = peos->nvar();

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  w[IDN] = torch::where(x1v < 0, 1.0, 0.125);
  w[IPR] = torch::where(x1v < 0, 1.0, 0.1);

  std::map<std::string, torch::Tensor> vars;

  // internal boundary
  auto r1 = torch::sqrt(x1v * x1v + x2v * x2v + x3v * x3v);
  auto solid = torch::where(r1 < 0.1, 1, 0).to(torch::kBool);

  vars["hydro_w"] = w;
  vars["solid"] = solid;

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

    // make outputs
    current_time += dt;
    block->make_outputs(vars, current_time);

    // check for signals
    auto sig = SignalHandler::GetInstance();
    if (sig->CheckSignalFlags() != 0) break;
  }

  block->finalize(vars, current_time);
}
