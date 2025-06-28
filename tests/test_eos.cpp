// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// snapy
#include <snap/snap.h>

#include <snap/eos/equation_of_state.hpp>

// tests
#include "device_testing.hpp"

const char *eos_config = R"(
type: moist-mixture
density-floor:  1.e-6
pressure-floor: 1.e-3
limiter: false
)";

const char *thermo_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5
)";

const char *coord_config = R"(
type: cartesian
bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
cells: {nx1: 4, nx2: 4, nx3: 1, nghost: 1}
)";

using namespace snap;

TEST_P(DeviceTest, moist_mixture) {
  auto op = EquationOfStateOptions::from_yaml(YAML::Load(eos_config));

  op.coord() = CoordinateOptions::from_yaml(YAML::Load(coord_config));
  op.thermo() = kintera::ThermoOptions::from_yaml(YAML::Load(thermo_config));

  auto peos = MoistMixture(op);
  peos->to(device, dtype);

  std::cout << "molecular weight = " << 1. / peos->pthermo->inv_mu << std::endl;

  auto const &cons = peos->get_buffer("U");

  EXPECT_EQ(cons.sizes(), std::vector<int64_t>({5, 1, 6, 6}));

  cons.uniform_(0., 1.);

  cons[Index::IDN].abs_();
  cons[Index::IPR].abs_();
  cons[Index::IPR] *= 10.;

  auto prim = peos->forward(cons);
  auto cons2 = peos->compute("W->U", {prim});

  EXPECT_TRUE(torch::allclose(cons, cons2, 1.E-6, 1.E-6));

  auto gamma = peos->compute("W->A", {prim});

  EXPECT_EQ(gamma.sizes(), std::vector<int64_t>({1, 6, 6}));
  EXPECT_TRUE(gamma.allclose(torch::ones_like(gamma) * 1.4, 1.E-6, 1.E-6));
}

/*TEST_P(DeviceTest, cons2prim_hydro_ideal_ncloud5) {
  int32_t NHYDRO = 14;
  int32_t ncloud = 5;
  int32_t nvapor = NHYDRO - 5 - ncloud;

  auto cons =
      torch::randn({NHYDRO, 1, 200, 200}, torch::device(device).dtype(dtype));
  auto gammad = torch::ones({1, 200, 200}, torch::device(device).dtype(dtype));
  auto rmu =
      torch::randn({nvapor + ncloud}, torch::device(device).dtype(dtype));
  auto rcv =
      torch::randn({nvapor + ncloud}, torch::device(device).dtype(dtype));

  gammad *= 1.4;
  rmu.normal_(0, 1);
  rcv.normal_(0, 1);

  auto start = std::chrono::high_resolution_clock::now();

  auto prim = eos_cons2prim_hydro_ideal(cons, gammad, rmu, rcv, ncloud);
  auto cons2 = eos_prim2cons_hydro_ideal(prim, gammad, rmu, rcv, ncloud);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by test body: " << elapsed.count() << " seconds"
            << std::endl;

  std::cout << (cons - cons2).min() << std::endl;
  std::cout << (cons - cons2).max() << std::endl;

  if (dtype == torch::kFloat32) {
    EXPECT_TRUE(torch::allclose(cons, cons2, 1.E-2, 1.E-2));
  } else {
    EXPECT_TRUE(torch::allclose(cons, cons2, 1.E-9, 1.E-9));
  }
}

TEST_P(DeviceTest, prim2cons_hydro_ideal_ncloud5) {
  int32_t NHYDRO = 14;
  int32_t ncloud = 5;
  int32_t nvapor = NHYDRO - 5 - ncloud;

  auto prim =
      torch::randn({NHYDRO, 1, 5, 5}, torch::device(device).dtype(dtype));
  auto gammad = torch::ones({1, 5, 5}, torch::device(device).dtype(dtype));
  auto rmu =
      torch::randn({nvapor + ncloud}, torch::device(device).dtype(dtype));
  auto rcv =
      torch::randn({nvapor + ncloud}, torch::device(device).dtype(dtype));

  gammad *= 1.4;
  rmu.normal_(0, 1);
  rcv.normal_(0, 1);

  auto cons = eos_prim2cons_hydro_ideal(prim, gammad, rmu, rcv, ncloud);
  auto prim2 = eos_cons2prim_hydro_ideal(cons, gammad, rmu, rcv, ncloud);

  std::cout << (prim - prim2).abs().max() << std::endl;

  if (dtype == torch::kFloat32) {
    EXPECT_TRUE(torch::allclose(prim, prim2, 1.E-4, 1.E-4));
  } else {
    EXPECT_TRUE(torch::allclose(prim, prim2, 1.E-12, 1.E-12));
  }
}*/

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
