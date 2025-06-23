// C/C++
#include <fstream>
#include <iostream>

// external
#include <gtest/gtest.h>

// base
#include <globals.h>

// fvm
#include <fvm/recon/reconstruct.hpp>

// tests
#include "device_testing.hpp"

enum {
  DIM1 = 3,
  DIM2 = 2,
  DIM3 = 1,
};

using namespace canoe;

TEST_P(DeviceTest, test_small) {
  int nhydro = 5;
  int nghost = 3;
  int nc3 = 1;
  int nc2 = 200 + 2 * nghost;
  int nc1 = 100 + 2 * nghost;

  auto w =
      torch::randn({nhydro, nc3, nc2, nc1}, torch::device(device).dtype(dtype));

  Reconstruct precon(
      ReconstructOptions().interp(InterpOptions().type("weno5")));
  precon->to(device, dtype);

  auto start = std::chrono::high_resolution_clock::now();

  auto result = precon->forward(w, DIM1);
  std::cout << result[0].sizes() << std::endl;
  std::cout << result[1].sizes() << std::endl;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::cout << "Time taken by test body: " << elapsed.count() << " seconds"
            << std::endl;
}

TEST_P(DeviceTest, test_large) {
  int nhydro = 5;
  int nghost = 3;
  int nc3 = 500 + 2 * nghost;
  int nc2 = 200 + 2 * nghost;
  int nc1 = 100 + 2 * nghost;

  auto w =
      torch::randn({nhydro, nc3, nc2, nc1}, torch::device(device).dtype(dtype));

  Reconstruct precon(
      ReconstructOptions().interp(InterpOptions().type("weno5")));
  precon->to(device, dtype);

  auto start = std::chrono::high_resolution_clock::now();

  auto result = precon->forward(w, DIM1);
  std::cout << result[0].sizes() << std::endl;
  std::cout << result[1].sizes() << std::endl;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::cout << "Time taken by test body: " << elapsed.count() << " seconds"
            << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  start_logging(argc, argv);

  return RUN_ALL_TESTS();
}
