// C/C++
#include <cstdio>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/layout/distribute_env.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

int main(int argc, char **argv) {
  auto config = YAML::LoadFile("test_exchange.yaml");

  auto block = MeshBlock(MeshBlockOptions::from_yaml("test_exchange.yaml"));
  auto device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "Running on CUDA" << std::endl;
    device = torch::kCUDA;
  }

  auto pcoord = block->phydro->pcoord;

  int nc1 = pcoord->options.nc1();
  int nc2 = pcoord->options.nc2();
  int nc3 = pcoord->options.nc3();
  int nghost = pcoord->options.nghost();

  auto w = torch::zeros(
      {5, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));
  int r = block->pdist->options.rank();
  auto interior = block->part({0, 0, 0}, /*exterior=*/false);
  w.index(interior)[IDN] = r + 1.0;
  w.index(interior)[IPR] = r + 1.0;

  std::map<std::string, torch::Tensor> vars;
  vars["hydro_w"] = w;
  block->initialize(vars);

  if (r == 0) {
    std::cout << "rank = " << r << std::endl;
    std::cout << "before hydro_u = " << vars["hydro_u"].squeeze() << std::endl;
  }

  block->exchange(vars);

  if (r == 0) {
    std::cout << "after hydro_u = " << vars["hydro_u"].squeeze() << std::endl;
  }

  return 0;
}
