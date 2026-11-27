// external
#include <gtest/gtest.h>

// C/C++
#include <chrono>
#include <thread>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>
#include <snap/mesh/stage_executor.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

// V3 lane B: a Mesh owns one StageExecutor for all its blocks and its forward
// runs each block's local stage through it; a bare MeshBlock (or a clone of a
// Mesh's block) keeps a one-block executor of its own. The executor changes
// who runs the stage, not what runs: two blocks stepped by Mesh::forward equal
// the pre-executor sequence (advance_local per block, then the exchange),
// bitwise.
namespace {

Mesh make_mesh(torch::Device device, torch::Dtype dtype) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_mesh_multi_block.yaml");
  block_opts->layout()->py(1);  // nb2 = 2, nb3 = 1: two blocks along x2
  // the eager executor path against the pre-executor sequence (on CUDA the
  // default would capture)
  block_opts->execution("eager");
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(2);
  auto mesh = Mesh(mesh_opts);
  mesh->to(device, dtype);
  return mesh;
}

// a pressure and velocity field that varies across the block seam
MeshVariables initial(Mesh mesh) {
  MeshVariables vars(mesh->blocks.size());
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto pcoord = mesh->blocks[i]->pcoord;
    auto opts = pcoord->x2v.options();
    auto x2 = pcoord->x2v.view({1, -1, 1});
    auto x3 = pcoord->x3v.view({-1, 1, 1});
    auto w = torch::zeros({5, pcoord->options->nc3(), pcoord->options->nc2(),
                           pcoord->options->nc1()},
                          opts);
    w[IDN].fill_(1.);
    w[IPR] = 1.e5 * (1. + 0.1 * torch::sin(6.283185307179586 * x2) +
                     0.05 * torch::cos(6.283185307179586 * x3));
    w[IVY] = 10. * torch::cos(6.283185307179586 * x2) + 0. * x3;
    vars[i]["hydro_w"] = w;
  }
  mesh->initialize(vars);
  return vars;
}

}  // namespace

TEST_P(DeviceTest, mesh_owns_one_executor_and_steps_through_it) {
  auto mesh = make_mesh(device, dtype);
  ASSERT_EQ(mesh->blocks.size(), 2u);
  auto ex = mesh->executor();
  ASSERT_TRUE(ex);
  EXPECT_EQ(ex->num_blocks(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(mesh->blocks[i]->executor(), ex) << "block " << i;
    EXPECT_EQ(ex->block_at(i), mesh->blocks[i].get()) << "block " << i;
  }

  // the reference: the same deck stepped as Mesh::forward did before lane B
  auto ref = make_mesh(device, dtype);
  auto vars = initial(mesh);
  auto rvars = initial(ref);
  std::vector<torch::Tensor> u_init;
  for (auto& v : vars) u_init.push_back(v.at("hydro_u").clone());

  int nstage = mesh->blocks.front()->pintg->stages.size();
  int nstep = 3;
  double dt = 0.5 * mesh->max_time_step(vars);
  for (int n = 0; n < nstep; ++n)
    for (int s = 0; s < nstage; ++s) {
      mesh->forward(vars, dt, s);
      for (size_t i = 0; i < ref->blocks.size(); ++i) {
        if (s == 0) ref->blocks[i]->step_dt = dt;
        ref->blocks[i]->advance_local(rvars[i], dt, s);
      }
      // same-process blocks meet in the exchange: one thread each
      std::vector<std::thread> jobs;
      for (size_t i = 0; i < ref->blocks.size(); ++i)
        jobs.emplace_back([&, i] {
          c10::OptionalDeviceGuard guard(device);
          ref->blocks[i]->exchange_ghost_zones(rvars[i]);
        });
      for (auto& j : jobs) j.join();
    }

  for (size_t i = 0; i < mesh->blocks.size(); ++i)
    for (auto name : {"hydro_u", "hydro_w"})
      EXPECT_TRUE(torch::equal(vars[i].at(name), rvars[i].at(name)))
          << "block " << i << " " << name << " differs from the pre-B sequence";
  for (size_t i = 0; i < mesh->blocks.size(); ++i)
    EXPECT_FALSE(torch::equal(vars[i].at("hydro_u"), u_init[i]))
        << "block " << i
        << ": the deck did not move, so equality gates nothing";

  // the Mesh's block-stages ran through its executor, eagerly
  auto report = mesh->execution_report();
  EXPECT_EQ(report.at("eager"), 2 * nstage * nstep);
  EXPECT_EQ(report.at("captures"), 0);
  EXPECT_EQ(report.at("graphs"), 0);
  EXPECT_EQ(mesh->blocks.front()->execution_report(), report);
  // the reference mesh's executor ran nothing
  EXPECT_EQ(ref->execution_report().at("eager"), 0);
}

TEST_P(DeviceTest, bare_block_and_clone_own_their_executor) {
  auto mesh = make_mesh(device, dtype);
  auto bare = MeshBlock(mesh->blocks.front()->options);
  ASSERT_TRUE(bare->executor());
  EXPECT_EQ(bare->executor()->num_blocks(), 1);
  EXPECT_EQ(bare->executor()->block_at(0), bare.get());
  EXPECT_NE(bare->executor(), mesh->executor());

  auto copy =
      std::dynamic_pointer_cast<MeshBlockImpl>(mesh->blocks.back()->clone());
  ASSERT_TRUE(copy);
  ASSERT_TRUE(copy->executor());
  EXPECT_NE(copy->executor(), mesh->executor())
      << "a clone of a Mesh's block must not run through the Mesh's executor";
  EXPECT_EQ(copy->executor()->block_at(0), copy.get());
}

TEST_P(DeviceTest,
       repeated_local_exchange_reads_current_buffers_after_failure) {
  auto opts = MeshOptionsImpl::from_yaml("test_mesh_multi_block.yaml");
  opts->blocks_per_process(4);
  if (device.is_cuda()) opts->block()->layout()->device("cuda").device_id(0);
  auto mesh = Mesh(opts);
  mesh->to(device, dtype);
  MeshVariables vars(4);
  auto sync = SyncOptions().type(kScalar);
  auto fill = [&](int round, bool mismatch) {
    for (int i = 0; i < 4; ++i) {
      auto block = mesh->blocks[i];
      auto coord = block->pcoord->options;
      int channels = round + 1 + (mismatch && i == 0);
      auto field =
          torch::full({channels, coord->nc3(), coord->nc2(), coord->nc1()}, -1.,
                      block->pcoord->x2v.options());
      field.index_put_(block->part({0, 0, 0}, PartOptions().exterior(false)),
                       10 * round + i);
      vars[i]["field"] = field;
    }
  };
  for (int round = 0; round < 3; ++round) {
    if (round == 2) {
      fill(round, true);
      EXPECT_THROW(mesh->exchange(vars, sync), c10::Error);
    }
    fill(round, false);
    mesh->exchange(vars, sync);
    for (int i = 0; i < 4; ++i) {
      auto block = mesh->blocks[i];
      auto layout = block->get_layout();
      auto loc = layout->loc_of(i);
      layout->for_each_direction(sync, [&](int dy, int dx, int dz) {
        auto offset = std::make_tuple(dy, dx, dz);
        int peer = layout->neighbor_rank(loc, offset);
        if (peer < 0) return;
        auto ghost = vars[i]["field"].index(
            block->part(offset, PartOptions().exterior(true)));
        EXPECT_TRUE(ghost.eq(10 * round + peer).all().item<bool>());
      });
    }
  }
  if (device.is_cuda()) {
    // A host field on one block must fail before issuing a host/device copy.
    vars[0]["field"] = vars[0]["field"].cpu();
    try {
      mesh->exchange(vars, sync);
      FAIL() << "expected a device mismatch";
    } catch (c10::Error const& error) {
      EXPECT_NE(std::string(error.what()).find("both halo buffers on cuda"),
                std::string::npos);
    }
  }
}

// Opt-in A/B benchmark. Run this same fixture against both revisions with
// --gtest_also_run_disabled_tests
// --gtest_filter='*DISABLED_local_halo_latency*'.
TEST_P(DeviceTest, DISABLED_local_halo_latency) {
  torch::set_num_threads(1);
  torch::NoGradGuard no_grad;
  int side = 8;
  auto opts = MeshOptionsImpl::from_yaml("test_mesh_multi_block.yaml");
  opts->block()->layout()->px(side).py(side);
  opts->blocks_per_process(side * side);
  auto co = opts->block()->coord();
  co->nx1(16)
      .global_nx1(16)
      .nx2(side * 32)
      .global_nx2(side * 32)
      .nx3(side * 32)
      .global_nx3(side * 32)
      .nghost(2);
  if (device.is_cuda()) opts->block()->layout()->device("cuda").device_id(0);
  auto mesh = Mesh(opts);
  MeshVariables vars(mesh->blocks.size());
  for (size_t i = 0; i < vars.size(); ++i) {
    auto coord = mesh->blocks[i]->pcoord->options;
    vars[i]["field"] =
        torch::full({5, coord->nc3(), coord->nc2(), coord->nc1()}, double(i),
                    torch::TensorOptions().device(device).dtype(dtype));
  }
  auto sync = SyncOptions().type(kScalar);
  auto run = [&](int count) {
    for (int i = 0; i < count; ++i) mesh->exchange(vars, sync);
    if (device.is_cuda()) torch::cuda::synchronize();
  };
  run(10);
  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  run(50);
  auto ms =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  std::cout << "Mean ms/exchange: " << ms / 50 << "\n";
}
