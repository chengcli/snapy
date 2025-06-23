// C/C++
#include <fstream>
#include <iostream>

// external
#include <gtest/gtest.h>

// base
#include <globals.h>

#include <input/parameter_input.hpp>

// fvm
#include <fvm/mesh/mesh.hpp>
#include <fvm/mesh/meshblock.hpp>
#include <fvm/output/output.hpp>

using namespace canoe;

class SetupMesh : public testing::Test {
 protected:
  ParameterInput pinput = nullptr;
  Mesh pmesh = nullptr;
  Output poutput = nullptr;

  char fname[80] = "/tmp/tempfile.XXXXXX";

  void CreateInputFile() {
    const char *mesh_config = R"(
<job>
problem_id  = fvm         # Problem ID

<output1>
file_type   = netcdf      # Netcdf data dump
variable    = prim        # variables to be output
dt          = 300.        # time increment between outputs

<time>
cfl_number  = 0.9         # The Courant, Friedrichs, & Lewy (CFL) Number
nlim        = 1           # cycle limit
tlim        = 900         # time limit

<mesh>
nx1         = 64          # Number of zones in X1-direction
x1min       = 0.          # minimum value of X1
x1max       = 6.4E3       # maximum value of X1
ix1_bc      = reflecting  # Inner-X1 boundary condition flag
ox1_bc      = reflecting  # Outer-X1 boundary condition flag

nx2         = 256         # Number of zones in X2-direction
x2min       = 0.          # minimum value of X2
x2max       = 25.6E3      # maximum value of X2
ix2_bc      = reflecting  # Inner-X2 boundary condition flag
ox2_bc      = reflecting  # Outer-X2 boundary condition flag

nx3         = 1           # Number of zones in X3-direction
x3min       = -0.5        # minimum value of X3
x3max       = 0.5         # maximum value of X3
ix3_bc      = periodic    # Inner-X3 boundary condition flag
ox3_bc      = periodic    # Outer-X3 boundary condition flag

<meshblock>
nghost      = 3

<hydro>
gamma       = 1.4
x1order     = weno5
x23order    = weno5
)";
    // write to file
    mkstemp(fname);
    std::ofstream outfile(fname);
    outfile << mesh_config;
  }

  virtual void SetUp() {
    CreateInputFile();

    // code here will execute just before the test ensues
    IOWrapper infile;
    infile.Open(fname, IOWrapper::FileMode::read);

    pinput = std::make_shared<ParameterInputImpl>();
    pinput->LoadFromFile(infile);
    infile.Close();

    // set up mesh
    pmesh = Mesh(MeshOptions(pinput));

    // set up output
    poutput = Output(pmesh, pinput);
  }

  virtual void TearDown() {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
    std::remove(fname);
  }
};

TEST_F(SetupMesh, test_mesh) {
  auto pmb = pmesh->blocks[0];
  auto hydro_u = pmb->hydro_u;

  EXPECT_EQ(hydro_u.size(0), 5);
  EXPECT_EQ(hydro_u.size(1), 1);
  EXPECT_EQ(hydro_u.size(2), 262);
  EXPECT_EQ(hydro_u.size(3), 70);

  for (int i = 1; i <= hydro_u.size(0); ++i) {
    hydro_u[i - 1] = i * torch::ones({pmb->nc3(), pmb->nc2(), pmb->nc1()});
  }

  // forward a step
  pmesh->forward(/*time=*/1., /*max_steps=*/1);
  poutput->MakeOutput(pmesh, pinput);
}

TEST_F(SetupMesh, test_mesh_cuda) {
  pmesh->to(torch::kCUDA);

  auto pmb = pmesh->blocks[0];
  auto hydro_u = pmb->hydro_u;

  EXPECT_EQ(hydro_u.size(0), 5);
  EXPECT_EQ(hydro_u.size(1), 1);
  EXPECT_EQ(hydro_u.size(2), 262);
  EXPECT_EQ(hydro_u.size(3), 70);

  for (int i = 1; i <= hydro_u.size(0); ++i) {
    hydro_u[i - 1] = i * torch::ones({pmb->nc3(), pmb->nc2(), pmb->nc1()});
  }

  // forward a step
  pmesh->forward(/*time=*/1., /*max_steps=*/1);
  poutput->MakeOutput(pmesh, pinput);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  start_logging(argc, argv);

  return RUN_ALL_TESTS();
}
