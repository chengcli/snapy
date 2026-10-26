#include <gtest/gtest.h>

#include <cmath>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/mesh/mesh.hpp>

#include "cuda_test_gate.hpp"
#if defined(USE_CUDA) && defined(USE_UCX)
#include <c10/cuda/CUDAGuard.h>
#endif

using namespace snap;

// Exercises the real six-panel solver, including the diffusion seam exchanges,
// CPU timestep/retry collectives and CUDA diagnostics. A single-process CTest
// invocation deliberately skips; the _ucx registration supplies two ranks.
TEST(CubedSphereDiffusionTwoGpu, full_mesh_conserves_mass_and_energy) {
#if defined(USE_CUDA) && defined(USE_UCX)
  if (get_world_size() != 2 || get_env("LOCAL_WORLD_SIZE", "0") != "2")
    GTEST_SKIP() << "Requires two local processes launched by torchrun";
  if (!snapy_cuda_test_enabled() || torch::cuda::device_count() < 2)
    GTEST_SKIP() << "Requires two visible CUDA devices";

  int rank = get_rank(), local_rank = get_local_rank();
  if (local_rank < 0 || local_rank > 1)
    GTEST_SKIP() << "Requires LOCAL_RANK 0 or 1";
  auto opts = MeshOptionsImpl::from_yaml("test_diffusion_cubed_sphere.yaml");
  if (opts->block()->layout()->backend() != "ucx")
    GTEST_SKIP() << "Requires BACKEND=ucx";

  torch::set_num_threads(1);
  torch::NoGradGuard no_grad;
  torch::Device device(torch::kCUDA, local_rank);
  c10::cuda::CUDAGuard guard(device);
  opts->blocks_per_process(3);
  opts->block()->layout()->device("cuda").device_id(local_rank);
  opts->block()
      ->coord()
      ->nx1(8)
      .global_nx1(8)
      .nx2(8)
      .global_nx2(8)
      .nx3(8)
      .global_nx3(8);
  auto mesh = Mesh(opts);
  ASSERT_EQ(mesh->blocks.size(), 3);

  MeshVariables vars(mesh->blocks.size());
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto block = mesh->blocks[i];
    auto c = block->pcoord;
    auto w = torch::zeros(
        {5, c->options->nc3(), c->options->nc2(), c->options->nc1()},
        torch::kFloat64);
    int panel = std::get<2>(
        block->get_layout()->loc_of(block->options->layout()->rank()));
    auto normal = torch::zeros_like(w.narrow(0, IVX, 3));
    normal[0].fill_(1.);
    cs_contra_to_cart_(normal, c->x2v.view({1, -1, 1}), c->x3v.view({-1, 1, 1}),
                       panel);
    // A smooth degree-two temperature perturbation drives a nontrivial flow.
    // Its sphere integral vanishes, independently of the panel partition.
    auto temperature = 300. + normal[0] * normal[1];
    double gas = block->phydro->peos->species_cv_ref() *
                 (block->phydro->peos->options->gammad() - 1.);
    w[IDN].fill_(1.);
    w[IPR].copy_(gas * temperature);
    vars[i]["hydro_w"] = w.to(device);
  }
  mesh->to(device);
  mesh->initialize(vars);
  auto comm = mesh->blocks.front()->get_layout()->comm;

  std::vector<torch::Tensor> initial_energy;
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto sl =
        mesh->blocks[i]->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    initial_energy.push_back(vars[i]["hydro_u"][IPR].index(sl).clone());
  }

  auto diagnostic = [&] {
    auto tensor_options =
        torch::TensorOptions().dtype(torch::kFloat64).device(device);
    auto sums = torch::zeros({2}, tensor_options);
    // Global maxima of density/energy changes, momentum, and a rank marker.
    auto maxima = torch::zeros({4}, tensor_options);
    maxima[3].fill_(rank + 1.);
    for (size_t i = 0; i < mesh->blocks.size(); ++i) {
      auto block = mesh->blocks[i];
      auto sl = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
      auto u = vars[i]["hydro_u"].index(
          block->part({0, 0, 0}, PartOptions().exterior(false)));
      auto volume = block->pcoord->cell_volume().index(sl);
      sums[0].add_((u[IDN] * volume).sum());
      sums[1].add_((u[IPR] * volume).sum());
      auto changes = torch::stack(
          {(u[IDN] - 1.).abs().max(),
           ((u[IPR] - initial_energy[i]) / initial_energy[i]).abs().max(),
           u.narrow(0, IVX, 3).abs().max()});
      maxima.narrow(0, 0, 3).copy_(
          torch::maximum(maxima.narrow(0, 0, 3), changes));
    }
    // Compare host and device transport using the same local diagnostics.
    // The analytic global integrals below also detect a missing SUM reduction.
    std::vector<torch::Tensor> host_sum{sums.cpu()}, host_max{maxima.cpu()};
    std::vector<torch::Tensor> gpu_sum{sums}, gpu_max{maxima};
    comm->allreduce(host_sum, c10d::ReduceOp::SUM);
    comm->allreduce(host_max, c10d::ReduceOp::MAX);
    comm->allreduce(gpu_sum, c10d::ReduceOp::SUM);
    comm->allreduce(gpu_max, c10d::ReduceOp::MAX);
    auto global_sum = gpu_sum[0].cpu(), global_max = gpu_max[0].cpu();
    EXPECT_TRUE(torch::allclose(global_sum, host_sum[0], 1.e-13, 1.e-12));
    EXPECT_TRUE(torch::equal(global_max, host_max[0]));
    EXPECT_DOUBLE_EQ(global_max[3].item<double>(), 2.);
    return std::make_pair(global_sum, global_max);
  };

  auto first = diagnostic().first;
  double ri = opts->block()->coord()->x1min();
  double ro = opts->block()->coord()->x1max();
  double volume = 4. * std::acos(-1.) / 3. * (ro * ro * ro - ri * ri * ri);
  double energy =
      volume * 300. * mesh->blocks[0]->phydro->peos->species_cv_ref();
  ASSERT_NEAR(first[0].item<double>() / volume, 1., 1.e-12);
  ASSERT_NEAR(first[1].item<double>() / energy, 1., 1.e-12);

  for (int step = 0; step < 3; ++step) {
    mesh->set_cycle(step + 1);
    double dt = mesh->max_time_step(vars);  // CPU global MIN
    ASSERT_TRUE(std::isfinite(dt) && dt > 0.);
    for (size_t stage = 0; stage < mesh->blocks[0]->pintg->stages.size();
         ++stage)
      mesh->forward(vars, dt, stage);
    ASSERT_EQ(mesh->check_redo(vars), 0);  // Global retry decision
    auto [last, maxima] = diagnostic();
    EXPECT_TRUE(torch::isfinite(last).all().item<bool>());
    EXPECT_TRUE(torch::isfinite(maxima).all().item<bool>());
    EXPECT_LT(((last - first) / first).abs().max().item<double>(), 1.e-10);
    EXPECT_LT(maxima[0].item<double>(), 1.e-3);
    EXPECT_LT(maxima[1].item<double>(), 1.e-3);
    EXPECT_GT(maxima[2].item<double>(), 1.e-10);  // Reject a no-op evolution
  }
#else
  GTEST_SKIP() << "Requires a build with CUDA and UCX support";
#endif
}
