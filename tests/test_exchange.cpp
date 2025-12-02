// C/C++
#include <cstdio>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/mesh/meshblock.hpp>

using namespace snap;

int main(int argc, char **argv) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_exchange.yaml", true);
  auto block = MeshBlock(op);

  auto device = torch::kCPU;
  // if (torch::cuda::is_available()) {
  //  std::cout << "Running on CUDA" << std::endl;
  //   device = torch::kCUDA;
  // }

  auto pcoord = block->phydro->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nghost = pcoord->options->nghost();

  auto w = torch::zeros(
      {5, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));
  int r = block->options->layout()->rank();

  auto playout = block->named_modules()["layout"];
  if (r == 0) playout->pretty_print(std::cout);

  auto interior = block->part({0, 0, 0}, /*exterior=*/false);
  w.index(interior)[IDN] = r + 1.0;
  w.index(interior)[IPR] = r + 1.0;

  std::map<std::string, torch::Tensor> vars;
  vars["hydro_w"] = w;
  block->initialize(vars);

  if (r == 0) {
    std::cout << "hydro_u = " << vars["hydro_u"].squeeze() << std::endl;
  }

  return 0;
}
