// gtest
#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <thread>

#include "cuda_test_gate.hpp"

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/layout/layout.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

// One process owns every block (6 * nb * nb). nx2 = nx3 = 8 is divisible by
// nb2 = 1, 2 and 4. The cross-panel interpolator reads a strip widened by
// cs_interp_margin. With that margin forced to 0, initialize indexes off the
// block at nb2 = 2 and at nb2 = 4. nb2 = 1 has no subdivided panel seam.
std::string write_card(int nb) {
  int blocks = 6 * nb * nb;
  std::string path =
      "test_cubed_sphere_exchange_" + std::to_string(nb) + ".yaml";
  std::ofstream o(path);
  o << "reference-state: {Tref: 0., Pref: 1.e5}\n"
       "species:\n"
       "  - name: dry\n"
       "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
       "    cv_R: 2.5\n"
       "dynamics:\n"
       "  equation-of-state: {type: ideal-gas}\n"
       "distribute:\n"
       "  layout: cubed-sphere\n"
       "  nb2: "
    << nb << "\n  nb3: " << nb << "\n  blocks_per_process: " << blocks
    << "\n  verbose: false\n"
       "geometry:\n"
       "  type: gnomonic-equiangle\n"
       "  cells: {nx1: 1, nx2: 8, nx3: 8, nghost: 2}\n"
       "  bounds: {x2min_pi: -0.25, x2max_pi: 0.25, x3min_pi: -0.25, "
       "x3max_pi: 0.25}\n"
       "boundary-condition:\n"
       "  external:\n"
       "    x1-inner: reflecting\n"
       "    x1-outer: reflecting\n"
       "    x2-inner: custom\n"
       "    x2-outer: custom\n"
       "    x3-inner: custom\n"
       "    x3-outer: custom\n"
       "integration: {type: rk3, cfl: 0.4, implicit-scheme: 0}\n";
  return path;
}

struct PanelFields {
  torch::Tensor x2f;      // (6, 9) interior faces, nghost stripped
  torch::Tensor dx2f;     // (6, 8) interior x2 cell widths
  torch::Tensor density;  // (6, 8, 8) interior hydro_u[IDN] after one stage
};

PanelFields run(int nb, torch::Device device = torch::kCPU) {
  auto block_opts = MeshBlockOptionsImpl::from_yaml(write_card(nb));
  // A card cannot name the device (the layout reads $DEVICE), so set it here:
  // the Mesh constructor picks the block worker pool's device from it.
  if (device.is_cuda()) {
    block_opts->layout()->device("cuda");
    block_opts->layout()->device_id(device.index());
  }
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(6 * nb * nb);
  auto mesh = Mesh(mesh_opts);
  MeshVariables vars(mesh->blocks.size());
  if (device.is_cuda()) {
    // The pool exists only for more than one block and reads the first
    // block's device_str(); each layout reads its own for the exchange.
    EXPECT_GT(mesh->blocks.size(), 1u);
    for (auto const &block : mesh->blocks) {
      EXPECT_EQ(block->options->device_str(), device.str());
    }
    // the pool itself is CUDA: one stream per block
    EXPECT_EQ(mesh->num_worker_streams(), mesh->blocks.size());
  } else {
    EXPECT_EQ(mesh->num_worker_streams(), 0u);
  }

  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto block = mesh->blocks[i];
    // The constructor still builds the block's tensors on the CPU.
    if (device.is_cuda()) block->to(device);
    auto coord = block->pcoord;
    int nvar = block->phydro->peos->nvar();
    auto w = torch::zeros(
        {nvar, coord->options->nc3(), coord->options->nc2(),
         coord->options->nc1()},
        torch::TensorOptions().dtype(torch::kFloat64).device(device));
    w[IDN] = 1. + 0.2 * coord->x2v.unsqueeze(0).unsqueeze(-1) +
             0.05 * coord->x3v.unsqueeze(1).unsqueeze(-1);
    w[IPR].fill_(1.e5);
    vars[i]["hydro_w"] = w;
  }
  mesh->initialize(vars);
  mesh->forward(vars, 1.e-4, 0);

  constexpr int kNx = 8;
  PanelFields out;
  out.x2f = torch::empty({6, kNx + 1}, torch::kFloat64);
  out.dx2f = torch::empty({6, kNx}, torch::kFloat64);
  out.density = torch::empty({6, kNx, kNx}, torch::kFloat64);

  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto block = mesh->blocks[i];
    auto coord = block->pcoord;
    int ng = coord->options->nghost();
    int nx2 = coord->options->nx2();
    int nx3 = coord->options->nx3();
    int ix2 = coord->options->ix2();
    int ix3 = coord->options->ix3();
    auto [rx, ry, face] =
        block->get_layout()->loc_of(block->options->layout()->rank());
    (void)rx;
    (void)ry;
    auto faces = coord->x2f.narrow(0, ng, nx2 + 1);
    out.x2f[face].narrow(0, ix2, nx2 + 1).copy_(faces.cpu());
    out.dx2f[face]
        .narrow(0, ix2, nx2)
        .copy_(coord->dx2f.narrow(0, ng, nx2).cpu());

    // hydro_u keeps a single x1 cell (nc1 == nx1); there is no x1 ghost pad.
    auto rho = vars[i]
                   .at("hydro_u")[IDN]
                   .slice(0, ng, ng + nx3)
                   .slice(1, ng, ng + nx2)
                   .select(2, 0);
    out.density[face]
        .slice(0, ix3, ix3 + nx3)
        .slice(1, ix2, ix2 + nx2)
        .copy_(rho.cpu());
  }
  return out;
}

// Six panels in one process, initialized at rest.
Mesh initialized_mesh(MeshVariables &vars) {
  auto block_opts = MeshBlockOptionsImpl::from_yaml(write_card(1));
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(6);
  auto mesh = Mesh(mesh_opts);
  vars.assign(mesh->blocks.size(), Variables());
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto coord = mesh->blocks[i]->pcoord;
    int nvar = mesh->blocks[i]->phydro->peos->nvar();
    auto w = torch::zeros({nvar, coord->options->nc3(), coord->options->nc2(),
                           coord->options->nc1()},
                          torch::kFloat64);
    w[IDN].fill_(1.);
    w[IPR].fill_(1.e5);
    vars[i]["hydro_w"] = w;
  }
  mesh->initialize(vars);
  return mesh;
}

}  // namespace

// Variable n of a remote exchange adds n * 65536 to the tag, so phyid 64 would
// reuse phyid 0's tag for the second variable.
TEST(CommTag, rejects_tags_that_collide_with_the_variable_offset) {
  std::tuple<int, int, int> offset{0, 1, 0};
  EXPECT_EQ(make_comm_tag(3, offset, 63), 63 * 1024 + 3 * 32 + 1);
  EXPECT_THROW(make_comm_tag(0, offset, 64), c10::Error);
  EXPECT_THROW(make_comm_tag(0, offset, -1), c10::Error);
}

// Decomposition invariance of the gnomonic faces, one metric (dx2f), and the
// conserved density after one RK stage.
TEST(CubedSphere, subdivided_panel_exchange_matches_one_block) {
  PanelFields ref;
  try {
    ref = run(1);
  } catch (std::exception const &e) {
    FAIL() << "nb2=1 " << e.what();
  }
  for (int nb : {2, 4}) {
    try {
      PanelFields got = run(nb);
      EXPECT_TRUE(torch::equal(got.x2f, ref.x2f)) << "x2f nb2=" << nb;
      EXPECT_TRUE(torch::equal(got.dx2f, ref.dx2f)) << "dx2f nb2=" << nb;
      EXPECT_TRUE(torch::equal(got.density, ref.density))
          << "hydro_u nb2=" << nb;
    } catch (std::exception const &e) {
      ADD_FAILURE() << "nb2=" << nb << " " << e.what();
    }
  }
}

// A block that exchanges different variables than its neighbors used to make
// the same-process copy read past the neighbor's buffers (segfault). Through
// the Mesh it is refused up front, naming the block and the variable.
TEST(CubedSphere, mesh_exchange_refuses_a_block_missing_a_variable) {
  MeshVariables vars;
  auto mesh = initialized_mesh(vars);
  vars[3].erase("hydro_w");
  try {
    mesh->exchange_ghost_zones(vars);
    FAIL() << "exchange with a block missing hydro_w did not throw";
  } catch (c10::Error const &e) {
    EXPECT_NE(std::string(e.what()).find(
                  "block 3 has no variable 'hydro_w' (block 0 has it)"),
              std::string::npos)
        << e.what();
  }
}

// The blocks exchanging on their own (as Mesh::forward runs them) meet the
// guard in the copy itself: every block gets the error, none crashes or waits.
TEST(CubedSphere, block_exchange_refuses_mismatched_variables) {
  MeshVariables vars;
  auto mesh = initialized_mesh(vars);
  vars[3].erase("hydro_w");
  SyncOptions opts;
  opts.interpolate(true).type(kConserved);
  std::vector<std::string> errors(mesh->blocks.size());
  std::vector<std::thread> threads;
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    threads.emplace_back([&, i]() {
      try {
        mesh->blocks[i]->exchange(vars[i], opts);
      } catch (c10::Error const &e) {
        errors[i] = e.what();
      }
    });
  }
  for (auto &t : threads) t.join();
  for (size_t i = 0; i < errors.size(); ++i) {
    EXPECT_NE(errors[i].find("every block must exchange the same variables"),
              std::string::npos)
        << "block " << i << ": " << errors[i];
  }
}

// Each Mesh has its own same-process exchange group; its barrier states and
// registrations go with its blocks, so building and freeing meshes one after
// another leaves the process-wide tables where they started.
TEST(CubedSphere, freed_mesh_leaves_no_local_exchange_state) {
  size_t before = LayoutImpl::local_exchange_entries();
  for (int i = 0; i < 3; ++i) {
    for (int nb : {1, 2}) {
      try {
        run(nb);
      } catch (std::exception const &e) {
        FAIL() << "nb2=" << nb << " " << e.what();
      }
      EXPECT_EQ(LayoutImpl::local_exchange_entries(), before)
          << "after mesh " << i << " nb2=" << nb;
    }
  }
}

TEST(CubedSphere, subdivided_panel_exchange_matches_one_block_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  PanelFields ref;
  try {
    ref = run(1, torch::Device(torch::kCUDA, 0));
  } catch (std::exception const &e) {
    FAIL() << "nb2=1 " << e.what();
  }
  for (int nb : {2, 4}) {
    try {
      PanelFields got = run(nb, torch::Device(torch::kCUDA, 0));
      EXPECT_TRUE(torch::equal(got.x2f, ref.x2f)) << "x2f nb2=" << nb;
      EXPECT_TRUE(torch::equal(got.dx2f, ref.dx2f)) << "dx2f nb2=" << nb;
      EXPECT_TRUE(torch::equal(got.density, ref.density))
          << "hydro_u nb2=" << nb;
    } catch (std::exception const &e) {
      ADD_FAILURE() << "nb2=" << nb << " " << e.what();
    }
  }
}
