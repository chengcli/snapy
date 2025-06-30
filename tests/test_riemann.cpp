// C/C++
#include <regex>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// snap
#include <snap/eos/equation_of_state.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/riemann/riemann_solver.hpp>

// tests
#include "device_testing.hpp"

enum {
  DIM1 = 3,
  DIM2 = 2,
  DIM3 = 1,
};

const char *eos_config = R"(
type: moist-mixture
density-floor:  1.e-10
pressure-floor: 1.e-10
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
cells: {nx1: 200, nx2: 200, nx3: 200, nghost: 3}
)";

const char *recon_config = R"(
vertical: {type: weno5, scale: false, shock: false}
horizontal: {type: weno5, scale: false, shock: false}
)";

const char *riemann_config = R"(
type: lmars
max-iter: 5
tol: 1.e-6
)";

using namespace snap;

TEST_P(DeviceTest, test_lmars) {
  auto op_riemann = RiemannSolverOptions::from_yaml(YAML::Load(riemann_config));

  op_riemann.eos() = EquationOfStateOptions::from_yaml(YAML::Load(eos_config));
  op_riemann.eos().coord() =
      CoordinateOptions::from_yaml(YAML::Load(coord_config));
  op_riemann.eos().thermo() =
      kintera::ThermoOptions::from_yaml(YAML::Load(thermo_config));

  LmarsSolver prsolver(op_riemann);
  prsolver->to(device, dtype);

  auto op_recon =
      ReconstructOptions::from_yaml(YAML::Load(recon_config), "vertical");

  Reconstruct precon(op_recon);
  precon->to(device, dtype);

  auto peos = prsolver->peosl;

  auto w =
      torch::randn({peos->nvar(), peos->options.coord().nc3(),
                    peos->options.coord().nc2(), peos->options.coord().nc1()},
                   torch::device(device).dtype(dtype))
          .abs();

  std::cout << "w.sizes(): " << w.sizes() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  auto wlr = precon->forward(w, DIM1);

  auto flux = prsolver->forward(wlr[0], wlr[1], DIM1);
  std::cout << "flux.sizes(): " << flux.sizes() << std::endl;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by test body: " << elapsed.count() << " seconds"
            << std::endl;
}

TEST_P(DeviceTest, test_hllc) {
  auto config = std::regex_replace(riemann_config, std::regex("lmars"), "hllc");
  auto op_riemann = RiemannSolverOptions::from_yaml(YAML::Load(config));

  op_riemann.eos() = EquationOfStateOptions::from_yaml(YAML::Load(eos_config));
  op_riemann.eos().coord() =
      CoordinateOptions::from_yaml(YAML::Load(coord_config));
  op_riemann.eos().thermo() =
      kintera::ThermoOptions::from_yaml(YAML::Load(thermo_config));

  HLLCSolver prsolver(op_riemann);
  prsolver->to(device, dtype);

  auto op_recon =
      ReconstructOptions::from_yaml(YAML::Load(recon_config), "vertical");

  Reconstruct precon(op_recon);
  precon->to(device, dtype);

  auto peos = prsolver->peosl;

  auto w =
      torch::randn({peos->nvar(), peos->options.coord().nc3(),
                    peos->options.coord().nc2(), peos->options.coord().nc1()},
                   torch::device(device).dtype(dtype))
          .abs();

  std::cout << "w.sizes(): " << w.sizes() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  auto wlr = precon->forward(w, DIM1);

  auto flux = prsolver->forward(wlr[0], wlr[1], DIM1);
  std::cout << "flux.sizes(): " << flux.sizes() << std::endl;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by test body: " << elapsed.count() << " seconds"
            << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
