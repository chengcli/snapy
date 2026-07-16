// C++/CUDA
#include <algorithm>
#include <array>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

// external
#include <gtest/gtest.h>

// torch
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/torch.h>

// snap
#include "../src/hydro/fused_recon_riemann_dispatch.hpp"
#include <snap/layout/cubed_sphere_layout.hpp>

using namespace snap;

namespace {

constexpr int kSides = 4;
constexpr int kNvar = 4;
constexpr int kEdgeLen = 8;
constexpr int kNc1 = 3;
constexpr int kStates = 2;
constexpr int kMetaStride = 5;
// hydro side_meta: [enabled, peer_process, peer_local_block, peer_side, rev]
constexpr int kHydroMetaStride = 5;

int env_int(char const *name, int fallback) {
  char const *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0')
    return fallback;
  return std::stoi(value);
}

torch::Device select_cuda_device() {
  int local_rank = env_int("LOCAL_RANK", 0);
  c10::cuda::set_device(local_rank);
  return torch::Device(torch::kCUDA, local_rank);
}

LayoutOptions make_layout_options() {
  auto opts = LayoutOptionsImpl::create();
  opts->type("cubed-sphere");
  opts->backend("ucx");
  opts->device("cuda");
  opts->px(1);
  opts->py(1);
  opts->pz(1);
  opts->blocks_per_process(1);
  opts->verbose(false);
  return opts;
}

torch::Tensor make_edge_meta(CubedSphereLayoutImpl const &layout,
                             torch::Device device) {
  auto iloc = layout.loc_of(layout.options->rank());
  int face = std::get<2>(iloc);
  std::array<std::tuple<int, int, int>, kSides> offsets = {
      std::tuple<int, int, int>{0, -1, 0},
      std::tuple<int, int, int>{0, +1, 0},
      std::tuple<int, int, int>{-1, 0, 0},
      std::tuple<int, int, int>{+1, 0, 0},
  };

  std::vector<int> meta(kSides * kMetaStride);
  for (int side = 0; side < kSides; ++side) {
    auto edge = CS_FACE_EDGES[face][side];
    EXPECT_EQ(layout.neighbor_rank(iloc, offsets[side]), edge.nface);
    meta[side * kMetaStride + 0] = edge.nface;
    meta[side * kMetaStride + 1] = edge.nside;
    meta[side * kMetaStride + 2] = edge.rev;
    meta[side * kMetaStride + 3] = (side % 2) == (edge.nside % 2);
    meta[side * kMetaStride + 4] = (side - 1.5) * (edge.nside - 1.5) < 0;
  }
  return torch::tensor(meta, torch::dtype(torch::kInt32)).to(device);
}

torch::Tensor make_hydro_side_meta(CubedSphereLayoutImpl const &layout,
                                   torch::Device device) {
  auto iloc = layout.loc_of(layout.options->rank());
  int face = std::get<2>(iloc);
  std::array<std::tuple<int, int, int>, kSides> offsets = {
      std::tuple<int, int, int>{0, -1, 0},
      std::tuple<int, int, int>{0, +1, 0},
      std::tuple<int, int, int>{-1, 0, 0},
      std::tuple<int, int, int>{+1, 0, 0},
  };

  std::vector<int> meta(kSides * kHydroMetaStride);
  for (int side = 0; side < kSides; ++side) {
    int nb = layout.neighbor_rank(iloc, offsets[side]);
    EXPECT_GE(nb, 0);
    auto nb_loc = layout.loc_of(nb);
    EXPECT_NE(std::get<2>(nb_loc), face);
    auto edge = CS_FACE_EDGES[face][side];
    EXPECT_EQ(nb, edge.nface);
    meta[side * kHydroMetaStride + 0] = 1;
    meta[side * kHydroMetaStride + 1] = layout.options->owner_process_rank(nb);
    meta[side * kHydroMetaStride + 2] = layout.options->local_block_index(nb);
    meta[side * kHydroMetaStride + 3] = edge.nside;
    meta[side * kHydroMetaStride + 4] = edge.rev;
  }
  return torch::tensor(meta, torch::dtype(torch::kInt32)).to(device);
}

bool fused_exchange_uses_right_state_start_for_side(int side) {
  return side == SIDE_L || side == SIDE_B;
}

bool fused_exchange_uses_right_interp_for_side(int side) {
  return !fused_exchange_uses_right_state_start_for_side(side);
}

bool cuda_6rank_available_or_skip() {
  if (!torch::cuda::is_available()) {
    return false;
  }
  int world_size = env_int("WORLD_SIZE", 1);
  if (world_size != 6) {
    return false;
  }
  int device_count = c10::cuda::device_count();
  if (device_count < world_size) {
    return false;
  }
  return true;
}

struct SmokeContext {
  torch::Device device;
  LayoutOptions options;
  std::shared_ptr<CubedSphereLayoutImpl> layout;
  torch::Tensor edge_meta;
};

SmokeContext make_smoke_context() {
  auto device = select_cuda_device();
  auto opts = make_layout_options();
  auto layout = std::make_shared<CubedSphereLayoutImpl>(opts);
  static std::shared_ptr<ProcessGroupContext> shared_comm =
      ProcessGroupContext::create(opts);
  layout->comm = shared_comm;
  EXPECT_TRUE(layout->has_process_group());
  EXPECT_NE(layout->comm, nullptr);
  EXPECT_TRUE(layout->comm->store.defined());
  auto edge_meta = make_edge_meta(*layout, device);
  return {device, opts, layout, edge_meta};
}

torch::Tensor make_exchange_buffer(torch::Device device) {
  std::vector<int64_t> sizes = {kSides, kStates, kNvar, kEdgeLen, kNc1};
  std::vector<int64_t> strides = {kStates * kNvar * kEdgeLen * kNc1,
                                  kNvar * kEdgeLen * kNc1, kEdgeLen * kNc1,
                                  kNc1, 1};
  return torch::empty_strided(
      sizes, strides,
      torch::TensorOptions().dtype(torch::kFloat64).device(device));
}

struct ExchangedBuffers {
  std::vector<torch::Tensor> received;
  torch::Tensor pointers;

  void **pointers_dev() const {
    return reinterpret_cast<void **>(pointers.data_ptr<int64_t>());
  }
};

ExchangedBuffers exchange_with_face_neighbors(SmokeContext const &ctx,
                                               torch::Tensor local, int tag) {
  int rank = ctx.options->process_rank();
  int world_size = ctx.options->process_world_size();
  auto meta = ctx.edge_meta.cpu();
  std::set<int> peers;
  for (int side = 0; side < kSides; ++side) {
    peers.insert(meta[side * kMetaStride].item<int>());
  }

  ExchangedBuffers result;
  result.received.resize(world_size);
  std::vector<int64_t> pointers(world_size, 0);
  pointers[rank] = reinterpret_cast<int64_t>(local.data_ptr());
  std::vector<CommWorkPtr> works;
  for (int peer : peers) {
    result.received[peer] = torch::empty_like(local);
    pointers[peer] =
        reinterpret_cast<int64_t>(result.received[peer].data_ptr());
    std::vector<torch::Tensor> send_tensors{local};
    std::vector<torch::Tensor> recv_tensors{result.received[peer]};
    works.push_back(ctx.layout->comm->send(send_tensors, peer, tag));
    works.push_back(ctx.layout->comm->recv(recv_tensors, peer, tag));
  }
  for (auto const &work : works) work->wait();
  result.pointers =
      torch::tensor(pointers, torch::dtype(torch::kInt64)).to(ctx.device);
  return result;
}

__device__ double device_payload(int rank, int side, int state, int edge, int i,
                                 int v) {
  return static_cast<double>(100000 * rank + 10000 * side + 1000 * state +
                             100 * edge + 10 * i + v);
}

__global__ void write_edge_payload_kernel(double *buffer, int rank) {
  int line = blockIdx.x;
  int i = line % kNc1;
  int edge = (line / kNc1) % kEdgeLen;
  int side = line / (kNc1 * kEdgeLen);
  int stride_var = kEdgeLen * kNc1;
  for (int state = 0; state < kStates; ++state) {
    int base =
        (((side * kStates + state) * kNvar) * kEdgeLen + edge) * kNc1 + i;
    for (int v = 0; v < kNvar; ++v) {
      buffer[base + v * stride_var] =
          device_payload(rank, side, state, edge, i, v);
    }
  }
}

__global__ void verify_remote_edge_kernel(void **buffer_ptrs, int const *meta,
                                          int *errors) {
  int line = blockIdx.x;
  int i = line % kNc1;
  int edge = (line / kNc1) % kEdgeLen;
  int side = line / (kNc1 * kEdgeLen);
  int peer_rank = meta[side * kMetaStride + 0];
  int peer_side = meta[side * kMetaStride + 1];
  int rev = meta[side * kMetaStride + 2];
  int peer_edge = rev ? (kEdgeLen - 1 - edge) : edge;
  int stride_var = kEdgeLen * kNc1;
  auto peer_buffer = static_cast<double const *>(buffer_ptrs[peer_rank]);
  for (int state = 0; state < kStates; ++state) {
    int remote_base =
        (((peer_side * kStates + state) * kNvar) * kEdgeLen + peer_edge) *
            kNc1 +
        i;
    for (int v = 0; v < kNvar; ++v) {
      double actual = peer_buffer[remote_base + v * stride_var];
      double expected =
          device_payload(peer_rank, peer_side, state, peer_edge, i, v);
      if (actual != expected)
        atomicAdd(errors, 1);
    }
  }
}

__global__ void verify_staged_remote_state_kernel(void **buffer_ptrs,
                                                  int const *meta, int rank,
                                                  int *errors) {
  int line = blockIdx.x;
  int i = line % kNc1;
  int edge = (line / kNc1) % kEdgeLen;
  int side = line / (kNc1 * kEdgeLen);
  int peer_rank = meta[side * kHydroMetaStride + 1];
  int peer_side = meta[side * kHydroMetaStride + 3];
  int rev = meta[side * kHydroMetaStride + 4];
  int peer_edge = rev ? (kEdgeLen - 1 - edge) : edge;
  bool lower_side = side == SIDE_L || side == SIDE_B;
  bool peer_lower_side = peer_side == SIDE_L || peer_side == SIDE_B;
  int remote_state = peer_lower_side ? IRT : ILT;
  int local_state = lower_side ? IRT : ILT;
  int stride_var = kEdgeLen * kNc1;
  int remote_base =
      (((peer_side * kStates + remote_state) * kNvar) * kEdgeLen + peer_edge) *
          kNc1 +
      i;
  int local_base =
      (((side * kStates + local_state) * kNvar) * kEdgeLen + edge) * kNc1 + i;
  auto peer_buffer = static_cast<double const *>(buffer_ptrs[peer_rank]);
  auto local_buffer = static_cast<double const *>(buffer_ptrs[rank]);
  for (int v = 0; v < kNvar; ++v) {
    double remote_actual = peer_buffer[remote_base + v * stride_var];
    double remote_expected =
        device_payload(peer_rank, peer_side, remote_state, peer_edge, i, v);
    double local_actual = local_buffer[local_base + v * stride_var];
    double local_expected = device_payload(rank, side, local_state, edge, i, v);
    if (remote_actual != remote_expected || local_actual != local_expected) {
      atomicAdd(errors, 1);
    }
  }
}

__global__ void verify_hydro_remote_constant_kernel(void **buffer_ptrs,
                                                    int const *meta,
                                                    int *errors) {
  int line = blockIdx.x;
  int i = line % kNc1;
  int edge = (line / kNc1) % kEdgeLen;
  int side = line / (kNc1 * kEdgeLen);
  int peer_rank = meta[side * kHydroMetaStride + 1];
  int peer_side = meta[side * kHydroMetaStride + 3];
  int rev = meta[side * kHydroMetaStride + 4];
  int peer_edge = rev ? (kEdgeLen - 1 - edge) : edge;
  int stride_var = kEdgeLen * kNc1;
  int remote_base =
      (((peer_side * kStates) * kNvar) * kEdgeLen + peer_edge) * kNc1 + i;
  auto peer_buffer = static_cast<double const *>(buffer_ptrs[peer_rank]);
  double density = peer_buffer[remote_base + IDN * stride_var];
  if (fabs(density - 500.) > 1.e-10)
    atomicAdd(errors, 1);
  for (int v = IVX; v <= IVZ; ++v) {
    double velocity = peer_buffer[remote_base + v * stride_var];
    if (fabs(velocity) > 1.e-10)
      atomicAdd(errors, 1);
  }
}

} // namespace

TEST(FusedCubedSphereUCX, ProcessGroupInitializes) {
  if (!cuda_6rank_available_or_skip())
    GTEST_SKIP();
  auto ctx = make_smoke_context();
  EXPECT_TRUE(ctx.layout->comm->initialized());
  EXPECT_TRUE(ctx.layout->comm->is_ucx());
  EXPECT_EQ(ctx.options->process_world_size(), 6);
  if (ctx.layout->comm)
    ctx.layout->comm->barrier();
}

TEST(FusedCubedSphereUCX, PreviousKernelWritesReachPeers) {
  if (!cuda_6rank_available_or_skip())
    GTEST_SKIP();
  auto ctx = make_smoke_context();
  auto exchange_buffer = make_exchange_buffer(ctx.device);

  auto stream = at::cuda::getCurrentCUDAStream(ctx.device.index());
  int blocks = kSides * kEdgeLen * kNc1;
  write_edge_payload_kernel<<<blocks, 1, 0, stream>>>(
      exchange_buffer.data_ptr<double>(), ctx.options->process_rank());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto exchanged = exchange_with_face_neighbors(ctx, exchange_buffer, 701);
  auto errors =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(ctx.device));
  verify_remote_edge_kernel<<<blocks, 1, 0, stream>>>(
      exchanged.pointers_dev(), ctx.edge_meta.data_ptr<int>(),
      errors.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  EXPECT_EQ(errors.cpu().item<int>(), 0);
  if (ctx.layout->comm)
    ctx.layout->comm->barrier();
}

TEST(FusedCubedSphereUCX, OrientationMetadataMatchesStaged) {
  if (!cuda_6rank_available_or_skip())
    GTEST_SKIP();
  auto ctx = make_smoke_context();

  auto meta = ctx.edge_meta.cpu();
  auto iloc = ctx.layout->loc_of(ctx.layout->options->rank());
  int face = std::get<2>(iloc);
  for (int side = 0; side < kSides; ++side) {
    auto edge = CS_FACE_EDGES[face][side];
    EXPECT_EQ(meta[side * kMetaStride + 0].item<int>(), edge.nface);
    EXPECT_EQ(meta[side * kMetaStride + 1].item<int>(), edge.nside);
    EXPECT_EQ(meta[side * kMetaStride + 2].item<int>(), edge.rev);
    EXPECT_EQ(meta[side * kMetaStride + 3].item<int>(),
              (side % 2) == (edge.nside % 2));
    EXPECT_EQ(meta[side * kMetaStride + 4].item<int>(),
              (side - 1.5) * (edge.nside - 1.5) < 0);
  }
  if (ctx.layout->comm)
    ctx.layout->comm->barrier();
}

TEST(FusedCubedSphereUCX, BoundaryStateConventionMatchesStaged) {
  if (!cuda_6rank_available_or_skip())
    GTEST_SKIP();
  // Staged hydro sends the local right state to lower-side neighbors and the
  // local left state to upper-side neighbors.
  EXPECT_TRUE(fused_exchange_uses_right_state_start_for_side(SIDE_L));
  EXPECT_FALSE(fused_exchange_uses_right_interp_for_side(SIDE_L));
  EXPECT_FALSE(fused_exchange_uses_right_state_start_for_side(SIDE_R));
  EXPECT_TRUE(fused_exchange_uses_right_interp_for_side(SIDE_R));
  EXPECT_TRUE(fused_exchange_uses_right_state_start_for_side(SIDE_B));
  EXPECT_FALSE(fused_exchange_uses_right_interp_for_side(SIDE_B));
  EXPECT_FALSE(fused_exchange_uses_right_state_start_for_side(SIDE_T));
  EXPECT_TRUE(fused_exchange_uses_right_interp_for_side(SIDE_T));
}

TEST(FusedCubedSphereUCX, RemoteEdgePayloadMatches) {
  if (!cuda_6rank_available_or_skip())
    GTEST_SKIP();
  auto ctx = make_smoke_context();
  auto exchange_buffer = make_exchange_buffer(ctx.device);

  auto stream = at::cuda::getCurrentCUDAStream(ctx.device.index());
  int blocks = kSides * kEdgeLen * kNc1;
  write_edge_payload_kernel<<<blocks, 1, 0, stream>>>(
      exchange_buffer.data_ptr<double>(), ctx.options->process_rank());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto exchanged = exchange_with_face_neighbors(ctx, exchange_buffer, 702);

  auto errors =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(ctx.device));
  verify_remote_edge_kernel<<<blocks, 1, 0, stream>>>(
      exchanged.pointers_dev(), ctx.edge_meta.data_ptr<int>(),
      errors.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  EXPECT_EQ(errors.cpu().item<int>(), 0);
  if (ctx.layout->comm)
    ctx.layout->comm->barrier();
}

TEST(FusedCubedSphereUCX, RemoteStateSelectionMatchesStaged) {
  if (!cuda_6rank_available_or_skip())
    GTEST_SKIP();
  auto ctx = make_smoke_context();
  auto exchange_buffer = make_exchange_buffer(ctx.device);

  auto stream = at::cuda::getCurrentCUDAStream(ctx.device.index());
  int blocks = kSides * kEdgeLen * kNc1;
  write_edge_payload_kernel<<<blocks, 1, 0, stream>>>(
      exchange_buffer.data_ptr<double>(), ctx.options->process_rank());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto exchanged = exchange_with_face_neighbors(ctx, exchange_buffer, 703);

  auto errors =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(ctx.device));
  auto side_meta = make_hydro_side_meta(*ctx.layout, ctx.device);
  verify_staged_remote_state_kernel<<<blocks, 1, 0, stream>>>(
      exchanged.pointers_dev(), side_meta.data_ptr<int>(),
      ctx.options->process_rank(), errors.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  EXPECT_EQ(errors.cpu().item<int>(), 0);
  if (ctx.layout->comm)
    ctx.layout->comm->barrier();
}

TEST(FusedCubedSphereUCX, ConstantStateExchangeHasZeroMassFlux) {
  if (!cuda_6rank_available_or_skip())
    GTEST_SKIP();
  auto ctx = make_smoke_context();
  auto exchange_buffer = make_exchange_buffer(ctx.device);

  auto opts = torch::dtype(torch::kFloat64).device(ctx.device);
  auto w = torch::zeros({kNvar, kEdgeLen, kEdgeLen, kNc1}, opts);
  w[IDN].fill_(500.);
  auto flux2 = torch::zeros_like(w);
  auto flux3 = torch::zeros_like(w);
  auto side_meta = make_hydro_side_meta(*ctx.layout, ctx.device);
  auto x2v = torch::linspace(-0.7, 0.7, kEdgeLen, opts);
  auto x3v = torch::linspace(-0.7, 0.7, kEdgeLen, opts);
  auto x2f = torch::linspace(-0.8, 0.8, kEdgeLen + 1, opts);
  auto x3f = torch::linspace(-0.8, 0.8, kEdgeLen + 1, opts);

  auto iloc = ctx.layout->loc_of(ctx.layout->options->rank());
  int face = std::get<2>(iloc);
  // One panel per process (blocks_per_process == 1), so local_block == 0 and
  // the flat [side, state, var, edge, nc1] buffer is exactly this block's
  // slice.
  int local_block = 0;
  FusedCubedSpherePanelParams panel_params{side_meta, face, local_block, x2v,
                                           x2f,       x3v,  x3f};
  fused_cubed_sphere_pack_cuda(
      w, exchange_buffer,
      FusedCubedSpherePackParams{panel_params, FusedReconScheme::WENO5,
                                 FusedReconScheme::WENO5, false,
                                 FusedEos::ShallowWater, 0., 0., false});
  auto exchanged = exchange_with_face_neighbors(ctx, exchange_buffer, 704);
  fused_cubed_sphere_flux_cuda(
      w, flux2, flux3, exchange_buffer, exchanged.pointers_dev(),
      FusedCubedSphereFluxParams{
          panel_params,
          FusedPhysicsParams{FusedReconScheme::WENO5, FusedReconScheme::WENO5,
                             false, FusedRiemannSolver::ShallowRoe,
                             FusedEos::ShallowWater, 1.4, 0., 0., false,
                             torch::Tensor(), torch::Tensor(), torch::Tensor(),
                             0, 1}});
  AT_CUDA_CHECK(cudaStreamSynchronize(
      at::cuda::getCurrentCUDAStream(ctx.device.index())));

  auto expect_constant_state = [](torch::Tensor state, char const *label) {
    EXPECT_LT(std::abs(state.select(1, IDN).min().cpu().item<double>() - 500.),
              1.e-10)
        << label;
    EXPECT_LT(std::abs(state.select(1, IDN).max().cpu().item<double>() - 500.),
              1.e-10)
        << label;
    EXPECT_LT(state.select(1, IVX).abs().max().cpu().item<double>(), 1.e-10)
        << label;
    EXPECT_LT(state.select(1, IVY).abs().max().cpu().item<double>(), 1.e-10)
        << label;
    EXPECT_LT(state.select(1, IVZ).abs().max().cpu().item<double>(), 1.e-10)
        << label;
  };
  expect_constant_state(exchange_buffer.select(1, ILT), "left state");
  expect_constant_state(exchange_buffer.select(1, IRT), "right state");
  auto errors =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(ctx.device));
  auto stream = at::cuda::getCurrentCUDAStream(ctx.device.index());
  verify_hydro_remote_constant_kernel<<<kSides * kEdgeLen * kNc1, 1, 0,
                                        stream>>>(exchanged.pointers_dev(),
                                                  side_meta.data_ptr<int>(),
                                                  errors.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  EXPECT_EQ(errors.cpu().item<int>(), 0);
  EXPECT_LT(flux2[IDN].abs().max().cpu().item<double>(), 1.e-10);
  EXPECT_LT(flux3[IDN].abs().max().cpu().item<double>(), 1.e-10);
  if (ctx.layout->comm)
    ctx.layout->comm->barrier();
}
