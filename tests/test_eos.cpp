// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// snapy
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

TEST_P(DeviceTest, moist_mixture) {
  auto op_block = MeshBlockOptionsImpl::from_yaml("test_eos.yaml");
  auto block = MeshBlock(op_block);
  block->to(device, dtype);

  auto peos = block->phydro->peos;
  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = peos->nvar();

  auto cons =
      torch::empty({nvar, nc3, nc2, nc1}, torch::device(device).dtype(dtype));

  cons.uniform_(0., 1.);

  cons[IDN].abs_();
  cons[IDN] += 1.E-6;  // avoid division by zero

  cons[IPR].abs_().mul_(10.);
  cons[IPR] += 0.5 * cons.narrow(0, IVX, 3).pow(2).sum() / cons[IDN];

  std::cout << "cons min = " << cons.min() << std::endl;
  std::cout << "cons max = " << cons.max() << std::endl;

  auto prim = peos->forward(cons);
  auto cons2 = peos->compute("W->U", {prim});

  EXPECT_TRUE(torch::allclose(cons, cons2, 1.E-6, 1.E-6));

  auto gamma = peos->compute("W->A", {prim});

  EXPECT_TRUE(gamma.allclose(torch::ones_like(gamma) * 1.4, 1.E-6, 1.E-6));

  auto cs = peos->compute("WA->L", {prim, gamma});
  std::cout << "cs min = " << cs.min() << std::endl;
  std::cout << "cs max = " << cs.max() << std::endl;
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
