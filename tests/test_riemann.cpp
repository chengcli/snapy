// C/C++
#include <iostream>

// external
#include <gtest/gtest.h>

// base
#include <globals.h>

// fvm
#include <fvm/coord/coordinate.hpp>
#include <fvm/eos/equation_of_state.hpp>
#include <fvm/recon/reconstruct.hpp>
#include <fvm/riemann/riemann_solver.hpp>

// tests
#include "device_testing.hpp"

enum {
  DIM1 = 3,
  DIM2 = 2,
  DIM3 = 1,
};

using namespace canoe;

TEST_P(DeviceTest, mps_case1) {
  int ncloud = 0;
  int nvapor = 0;
  int nghost = 3;
  int nx1 = 200;
  int nx2 = 200;
  int nx3 = 1;

  Thermodynamics thermo(
      ThermodynamicsOptions().gammad_ref(1.4).nvapor(nvapor).ncloud(ncloud));

  Cartesian pcoord(CoordinateOptions().nx1(nx1).nx2(nx2).nx3(nx3));

  IdealMoist peos(
      EquationOfStateOptions().thermo(thermo->options).coord(pcoord->options));

  Reconstruct precon(
      ReconstructOptions().interp(InterpOptions().type("weno5")));

  LmarsSolver prsolver(
      RiemannSolverOptions().eos(peos->options).coord(pcoord->options));

  auto w =
      torch::randn({peos->nhydro(), nx3, nx2 + 2 * nghost, nx1 + 2 * nghost},
                   torch::device(device).dtype(dtype))
          .abs();
  precon->to(device, dtype);
  prsolver->to(device, dtype);

  std::cout << "w.sizes(): " << w.sizes() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  auto wlr = precon->forward(w, DIM1);

  auto gammad = prsolver->peos->pthermo->get_gammad(wlr.mean(0));
  auto flux = prsolver->forward(wlr[0], wlr[1], DIM1, gammad);
  std::cout << "flux.sizes(): " << flux.sizes() << std::endl;

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
