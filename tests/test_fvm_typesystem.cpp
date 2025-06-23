// external
#include <gtest/gtest.h>

// base
#include <globals.h>

#include <input/parameter_input.hpp>

// fvm
#include <fvm/hydro/hydro.hpp>
#include <fvm/mesh/meshblock.hpp>
#include <fvm/scalar/scalar.hpp>

// tests
#include "device_testing.hpp"

using namespace canoe;

TEST_P(DeviceTest, test) {
  float gammad = 1.4;
  int nvapor = 1;
  int ncloud = 1;
  int nghost = 3;
  int nhydro = 5;

  Thermodynamics thermo(
      ThermodynamicsOptions().gammad_ref(gammad).nvapor(nvapor).ncloud(ncloud));

  Cartesian coord(CoordinateOptions().nc1(100));

  IdealMoist eos(EquationOfStateOptions()
                     .thermo(torch::nn::AnyModule(thermo))
                     .coord(torch::nn::AnyModule(coord)));

  Reconstruct recon(ReconstructOptions()
                        .napply1(nhydro)
                        .interp1(torch::nn::AnyModule(Weno5Interp()))
                        .interp2(torch::nn::AnyModule(Center5Interp())));

  LmarsSolver rsolver(RiemannSolverOptions()
                          .eos(torch::nn::AnyModule(eos))
                          .coord(torch::nn::AnyModule(coord)));

  Hydro hydro(HydroOptions()
                  .eos(torch::nn::AnyModule(eos))
                  .coord(torch::nn::AnyModule(coord))
                  .recon1(torch::nn::AnyModule(recon))
                  .recon2(torch::nn::AnyModule(recon))
                  .riemann(torch::nn::AnyModule(rsolver)));

  Scalar scalar(ScalarOptions()
                    .coord(torch::nn::AnyModule(coord))
                    .recon(torch::nn::AnyModule(recon)));

  MeshBlock block(MeshBlockOptions()
                      .nx1(rs.nx1())
                      .nx2(rs.nx2())
                      .nx3(rs.nx3())
                      .nghost(nghost)
                      .hydro(torch::nn::AnyModule(hydro))
                      .scalar(torch::nn::AnyModule(scalar)));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  start_logging(argc, argv);

  return RUN_ALL_TESTS();
}
