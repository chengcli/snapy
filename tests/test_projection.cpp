// C/C++
#include <fstream>
#include <iostream>

// external
#include <gtest/gtest.h>

// base
#include <globals.h>

// fvm
#include <fvm/hydro/primitive_projector.hpp>

// tests
#include "device_testing.hpp"

using namespace canoe;

TEST_P(DeviceTest, test_hydrostatic_pressure) {
  auto w = torch::ones({5, 1, 1, 10}, torch::device(device).dtype(dtype));
  auto dz = torch::ones({10}, torch::device(device).dtype(dtype));
  auto y = calc_hydrostatic_pressure(w, 10., dz, 2, 7);
  std::cout << y << std::endl;
}

TEST_P(DeviceTest, test_nonhydrostatic_pressure) {
  auto w = torch::ones({5, 1, 1, 10}, torch::device(device).dtype(dtype));
  auto dz = torch::ones({10}, torch::device(device).dtype(dtype));
  auto psf = calc_hydrostatic_pressure(w, 10., dz, 2, 7);
  auto y = calc_nonhydrostatic_pressure(w[4], psf);
  std::cout << y << std::endl;
}

TEST_P(DeviceTest, test_primitive_projector) {
  auto w = torch::ones({5, 1, 1, 10}, torch::device(device).dtype(dtype));
  auto dz = torch::ones({10}, torch::device(device).dtype(dtype));
  auto op =
      PrimitiveProjectorOptions().type("temperature").nghost(2).grav(-10.);
  auto projector = PrimitiveProjector(op);

  projector->to(device, dtype);
  auto y = projector->forward(w, dz);
  std::cout << y << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  start_logging(argc, argv);

  return RUN_ALL_TESTS();
}
