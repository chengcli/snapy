// C/C++
#include <unistd.h>

#include <fstream>

// external
#include <gtest/gtest.h>

// snap
#include <snap/mesh/meshblock.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

namespace {

const char* scalar_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

dynamics:
  equation-of-state:
    type: ideal-gas
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver:
    type: lmars

scalar:
  nvar: 2
  names: [tr0, tr1]
  reconstruct: {type: weno5, scale: false, shock: false}

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: 16, nx2: 8, nx3: 4, nghost: 3}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: reflecting
    x3-outer: reflecting
)";

std::string write_temp_config() {
  char fname[] = "/tmp/snapy_scalar_test.XXXXXX";
  int fd = mkstemp(fname);
  close(fd);
  std::ofstream out(fname);
  out << scalar_config;
  out.close();
  return fname;
}

}  // namespace

TEST_P(DeviceTest, initialize_and_transport_scalar) {
  auto filename = write_temp_config();
  auto opts = MeshBlockOptionsImpl::from_yaml(filename);
  auto block = MeshBlock(opts);
  block->to(device, dtype);

  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();

  Variables vars;
  vars["hydro_w"] =
      torch::zeros({5, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  vars["hydro_w"][IDN].fill_(2.0);
  vars["hydro_w"][IPR].fill_(3.0);
  vars["scalar_r"] =
      torch::zeros({2, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  vars["scalar_r"][0].fill_(0.25);
  vars["scalar_r"][1].fill_(0.5);

  block->initialize(vars);

  ASSERT_TRUE(vars.count("scalar_s"));
  ASSERT_EQ(vars.at("scalar_s").sizes(),
            std::vector<int64_t>({2, nc3, nc2, nc1}));
  EXPECT_TRUE(torch::allclose(
      vars.at("scalar_s"),
      vars.at("hydro_w")[IDN].unsqueeze(0) * vars.at("scalar_r")));

  auto ds = block->pscalar->forward(1.0e-3, vars.at("scalar_s"), vars);
  EXPECT_EQ(ds.sizes(), vars.at("scalar_s").sizes());
  EXPECT_EQ(ds.device(), vars.at("scalar_s").device());
  EXPECT_EQ(ds.scalar_type(), vars.at("scalar_s").scalar_type());
  EXPECT_TRUE(torch::isfinite(ds).all().item<bool>());

  std::remove(filename.c_str());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
