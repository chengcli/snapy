// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/recon/reconstruct.hpp>

// tests
#include "device_testing.hpp"

enum {
  DIM1 = 3,
  DIM2 = 2,
  DIM3 = 1,
};

const char *recon_config = R"(
vertical: {type: weno5, scale: false, shock: false}
horizontal: {type: weno5, scale: false, shock: false}
)";

using namespace snap;
using namespace torch::indexing;

TEST_P(DeviceTest, test_dc) {
  int nhydro = 1;
  int nc3 = 1;
  int nc2 = 1;
  int nc1 = 10;
  auto w =
      torch::randn({nhydro, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  auto op = ReconstructOptionsImpl::create();
  op->interp()->type("dc");

  Reconstruct precon(op);
  precon->to(device, dtype);

  std::cout << "w = " << w << std::endl;
  auto result = precon->forward(w, DIM1);
  std::cout << "wl = " << result[0] << std::endl;
  std::cout << "wr = " << result[1] << std::endl;
}

TEST_P(DeviceTest, test_plm) {
  int nhydro = 1;
  int nc3 = 1;
  int nc2 = 1;
  int nc1 = 10;
  auto w =
      torch::randn({nhydro, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  auto op = ReconstructOptionsImpl::create();
  op->interp()->type("plm");

  Reconstruct precon(op);
  precon->to(device, dtype);

  op->report(std::cout);

  std::cout << "w = " << w << std::endl;
  auto result = precon->forward(w, DIM1);
  std::cout << "wl = " << result[0] << std::endl;
  std::cout << "wr = " << result[1] << std::endl;
}

TEST_P(DeviceTest, test_small) {
  int nhydro = 5;
  int nghost = 3;
  int nc3 = 1;
  int nc2 = 200 + 2 * nghost;
  int nc1 = 100 + 2 * nghost;

  auto w =
      torch::randn({nhydro, nc3, nc2, nc1}, torch::device(device).dtype(dtype));

  auto op =
      ReconstructOptionsImpl::from_yaml(YAML::Load(recon_config), "horizontal");
  op->report(std::cout);

  Reconstruct precon(op);
  precon->to(device, dtype);

  auto result = precon->forward(w, DIM1);
  std::cout << result[0].sizes() << std::endl;
  std::cout << result[1].sizes() << std::endl;
  // std::cout << "w = " << w.index({0, 0, 0, Slice()}) << std::endl;
  // std::cout << "wl = " << result[0].index({0, 0, 0, Slice()}) << std::endl;
  // std::cout << "wr = " << result[1].index({0, 0, 0, Slice()}) << std::endl;
}

TEST_P(DeviceTest, test_large) {
  int nhydro = 5;
  int nghost = 3;
  int nc3 = 500 + 2 * nghost;
  int nc2 = 200 + 2 * nghost;
  int nc1 = 100 + 2 * nghost;

  auto w =
      torch::randn({nhydro, nc3, nc2, nc1}, torch::device(device).dtype(dtype));

  auto op =
      ReconstructOptionsImpl::from_yaml(YAML::Load(recon_config), "horizontal");

  Reconstruct precon(op);
  precon->to(device, dtype);

  auto result = precon->forward(w, DIM1);
  std::cout << result[0].sizes() << std::endl;
  std::cout << result[1].sizes() << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
