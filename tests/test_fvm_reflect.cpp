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
nx1         = 4           # Number of zones in X1-direction
x1min       = 0.          # minimum value of X1
x1max       = 1.          # maximum value of X1
ix1_bc      = reflecting  # Inner-X1 boundary condition flag
ox1_bc      = reflecting  # Outer-X1 boundary condition flag

nx2         = 8           # Number of zones in X2-direction
x2min       = 0.          # minimum value of X2
x2max       = 1.          # maximum value of X2
ix2_bc      = reflecting  # Inner-X2 boundary condition flag
ox2_bc      = reflecting  # Outer-X2 boundary condition flag

nx3         = 1          # Number of zones in X3-direction
x3min       = 0.          # minimum value of X3
x3max       = 1.          # maximum value of X3
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

TEST_F(SetupMesh, test_reflect_cpu) {
  auto pmb = pmesh->blocks[0];
  auto hydro_u = pmb->hydro_u;

  for (int n = 0; n < pmb->phydro->nvar(); ++n) {
    for (int k = pmb->ks(); k <= pmb->ke(); ++k)
      for (int j = pmb->js(); j <= pmb->je(); ++j)
        for (int i = pmb->is(); i <= pmb->ie(); ++i) {
          hydro_u.index_put_({n, k, j, i},
                             (n + 1) * (k + 1) * (j + 1) * (i + 1));
        }
  }

  std::cout << hydro_u << std::endl;
  pmb->set_ghost_zones();
  std::cout << hydro_u << std::endl;
}

TEST_F(SetupMesh, test_reflect_cuda) {
  pmesh->to(torch::kCUDA);

  auto pmb = pmesh->blocks[0];
  auto hydro_u = pmb->hydro_u;

  for (int n = 0; n < pmb->phydro->nvar(); ++n) {
    for (int k = pmb->ks(); k <= pmb->ke(); ++k)
      for (int j = pmb->js(); j <= pmb->je(); ++j)
        for (int i = pmb->is(); i <= pmb->ie(); ++i) {
          hydro_u.index_put_({n, k, j, i},
                             (n + 1) * (k + 1) * (j + 1) * (i + 1));
        }
  }

  std::cout << hydro_u << std::endl;
  pmb->set_ghost_zones();
  std::cout << hydro_u << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  start_logging(argc, argv);

  return RUN_ALL_TESTS();
}
