// C/C++
#include <cstdio>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/mesh/meshblock.hpp>

using namespace snap;

int main(int argc, char **argv) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_exchange.yaml");
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

  if (r == 0) {
    block->named_modules()["layout"]->pretty_print(std::cout);
  }

  auto interior = block->part({0, 0, 0}, /*exterior=*/false);
  w.index(interior)[IDN] = r + 1.0;
  w.index(interior)[IPR] = r + 1.0;

  std::map<std::string, torch::Tensor> vars;
  vars["hydro_w"] = w;
  block->initialize(vars);

  block->layout()->pg->barrier()->wait();

  auto [rx, ry, face] = block->layout()->loc_of(r);

  for (int i = 0; i < block->options->layout()->world_size(); ++i) {
    if (i == r) {
      std::cout << fmt::format("rx = {}, ry = {}, face = {}, rank = {}", rx, ry,
                               face, r)
                << std::endl;
      std::cout << "hydro_u = \n"
                << vars["hydro_u"][IDN].squeeze().transpose(0, 1).flip(0)
                << std::endl;
    }
    block->layout()->pg->barrier()->wait();
  }

  return 0;
}
