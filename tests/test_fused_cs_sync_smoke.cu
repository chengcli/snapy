// C/C++
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
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAException.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
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
constexpr int kHydroMetaStride = 6;

int env_int(char const* name, int fallback) {
  char const* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  return std::stoi(value);
}

torch::Device select_cuda_device() {
  int local_rank = env_int("LOCAL_RANK", 0);
  c10::cuda::set_device(local_rank);
  return torch::Device(torch::kCUDA, local_rank);
}

void require_cuda_6rank_or_skip() {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA runtime is unavailable";
  }
  int world_size = env_int("WORLD_SIZE", 1);
  if (world_size != 6) {
    GTEST_SKIP() << "test_fused_cs_sync_smoke requires torchrun with 6 ranks";
  }
  int device_count = c10::cuda::device_count();
  if (device_count < world_size) {
    GTEST_SKIP() << "test_fused_cs_sync_smoke requires 6 CUDA devices";
  }
}

LayoutOptions make_layout_options() {
  auto opts = LayoutOptionsImpl::create();
  opts->type("cubed-sphere");
  opts->backend("gloo");
  opts->device("cuda");
  opts->px(1);
  opts->py(1);
  opts->pz(1);
  opts->blocks_per_process(1);
  opts->verbose(false);
  return opts;
}

torch::Tensor make_edge_meta(CubedSphereLayoutImpl const& layout,
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

torch::Tensor make_hydro_side_meta(CubedSphereLayoutImpl const& layout,
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
    meta[side * kHydroMetaStride + 1] = nb;
    meta[side * kHydroMetaStride + 2] = edge.nside;
    meta[side * kHydroMetaStride + 3] = edge.rev;
    meta[side * kHydroMetaStride + 4] = (side % 2) == (edge.nside % 2);
    meta[side * kHydroMetaStride + 5] =
        (side - 1.5) * (edge.nside - 1.5) < 0;
  }
  return torch::tensor(meta, torch::dtype(torch::kInt32)).to(device);
}

bool fused_exchange_uses_right_state_start_for_side(int side) {
  return side == SIDE_L || side == SIDE_B;
}

bool fused_exchange_uses_right_interp_for_side(int side) {
  return !fused_exchange_uses_right_state_start_for_side(side);
}

void initialize_symmetric_memory_group(LayoutImpl const& layout,
                                       std::string const& group_name) {
  static std::set<std::string> initialized;
  if (initialized.empty()) {
    c10d::symmetric_memory::set_backend("NVSHMEM");
    c10d::symmetric_memory::set_signal_pad_size(std::max<size_t>(
        1024, layout.options->process_world_size() * sizeof(uint32_t)));
  }

  auto set_group_info_once = [&](std::string const& name) {
    if (initialized.count(name)) return;
    c10d::symmetric_memory::set_group_info(
        name, layout.options->process_rank(),
        layout.options->process_world_size(), layout.comm->store);
    initialized.insert(name);
  };

  set_group_info_once("0");
  set_group_info_once(group_name);
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

torch::Tensor make_symmetric_buffer(torch::Device device,
                                    std::string const& group_name) {
  (void)group_name;
  std::vector<int64_t> sizes = {kSides, kStates, kNvar, kEdgeLen, kNc1};
  std::vector<int64_t> strides = {
      kStates * kNvar * kEdgeLen * kNc1, kNvar * kEdgeLen * kNc1,
      kEdgeLen * kNc1, kNc1, 1};
  return c10d::symmetric_memory::empty_strided_p2p(
      sizes, strides, torch::kFloat64, device, std::nullopt, std::nullopt);
}

void clear_signal_slots(
    c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> const& symm,
    std::shared_ptr<ProcessGroupContext> const& comm) {
  auto signal_pad = symm->get_signal_pad(
      symm->get_rank(), {2 * symm->get_world_size()}, torch::kInt32);
  signal_pad.zero_();
  (void)signal_pad.sum().item<int>();
  if (comm) comm->barrier();
}

__device__ double device_payload(int rank, int side, int edge, int i, int v) {
  return static_cast<double>(10000 * rank + 1000 * side + 100 * edge +
                             10 * i + v);
}

__global__ void write_edge_payload_kernel(double* buffer, int rank) {
  int line = blockIdx.x;
  int i = line % kNc1;
  int edge = (line / kNc1) % kEdgeLen;
  int side = line / (kNc1 * kEdgeLen);
  int stride_var = kEdgeLen * kNc1;
  int base = (((side * kStates) * kNvar) * kEdgeLen + edge) * kNc1 + i;
  for (int v = 0; v < kNvar; ++v) {
    buffer[base + v * stride_var] = device_payload(rank, side, edge, i, v);
  }
}

__global__ void sync_previous_kernel_writes(uint32_t** signal_pads, int rank,
                                            int world_size) {
  c10d::symmetric_memory::sync_remote_blocks<false, true>(
      signal_pads, rank, world_size);
}

__global__ void verify_remote_edge_kernel(void** buffer_ptrs, int const* meta,
                                          int* errors) {
  int line = blockIdx.x;
  int i = line % kNc1;
  int edge = (line / kNc1) % kEdgeLen;
  int side = line / (kNc1 * kEdgeLen);
  int peer_rank = meta[side * kMetaStride + 0];
  int peer_side = meta[side * kMetaStride + 1];
  int rev = meta[side * kMetaStride + 2];
  int peer_edge = rev ? (kEdgeLen - 1 - edge) : edge;
  int stride_var = kEdgeLen * kNc1;
  int remote_base =
      (((peer_side * kStates) * kNvar) * kEdgeLen + peer_edge) * kNc1 + i;
  auto peer_buffer = static_cast<double const*>(buffer_ptrs[peer_rank]);
  for (int v = 0; v < kNvar; ++v) {
    double actual = peer_buffer[remote_base + v * stride_var];
    double expected = device_payload(peer_rank, peer_side, peer_edge, i, v);
    if (actual != expected) atomicAdd(errors, 1);
  }
}

__global__ void verify_hydro_remote_constant_kernel(void** buffer_ptrs,
                                                    int const* meta,
                                                    int* errors) {
  int line = blockIdx.x;
  int i = line % kNc1;
  int edge = (line / kNc1) % kEdgeLen;
  int side = line / (kNc1 * kEdgeLen);
  int peer_rank = meta[side * kHydroMetaStride + 1];
  int peer_side = meta[side * kHydroMetaStride + 2];
  int rev = meta[side * kHydroMetaStride + 3];
  int peer_edge = rev ? (kEdgeLen - 1 - edge) : edge;
  int stride_var = kEdgeLen * kNc1;
  int remote_base =
      (((peer_side * kStates) * kNvar) * kEdgeLen + peer_edge) * kNc1 + i;
  auto peer_buffer = static_cast<double const*>(buffer_ptrs[peer_rank]);
  double density = peer_buffer[remote_base + IDN * stride_var];
  if (fabs(density - 500.) > 1.e-10) atomicAdd(errors, 1);
  for (int v = IVX; v <= IVZ; ++v) {
    double velocity = peer_buffer[remote_base + v * stride_var];
    if (fabs(velocity) > 1.e-10) atomicAdd(errors, 1);
  }
}

}  // namespace

TEST(FusedCubedSphereSymmetricMemory, RendezvousCompletes) {
  require_cuda_6rank_or_skip();
  auto ctx = make_smoke_context();
  std::string group_name = "snapy:test:fused-cs-rendezvous";
  initialize_symmetric_memory_group(*ctx.layout, group_name);

  auto symm_buffer = make_symmetric_buffer(ctx.device, group_name);
  auto symm = c10d::symmetric_memory::rendezvous(symm_buffer, group_name);
  EXPECT_EQ(symm->get_world_size(), 6);
  if (ctx.layout->comm) ctx.layout->comm->barrier();
}

TEST(FusedCubedSphereSymmetricMemory, PreviousKernelSyncCompletes) {
  require_cuda_6rank_or_skip();
  auto ctx = make_smoke_context();
  std::string group_name = "snapy:test:fused-cs-sync";
  initialize_symmetric_memory_group(*ctx.layout, group_name);

  auto symm_buffer = make_symmetric_buffer(ctx.device, group_name);
  auto symm = c10d::symmetric_memory::rendezvous(symm_buffer, group_name);
  clear_signal_slots(symm, ctx.layout->comm);

  auto stream = at::cuda::getCurrentCUDAStream(ctx.device.index());
  int blocks = kSides * kEdgeLen * kNc1;
  write_edge_payload_kernel<<<blocks, 1, 0, stream>>>(
      symm_buffer.data_ptr<double>(), symm->get_rank());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  sync_previous_kernel_writes<<<1, std::max(32, symm->get_world_size()), 0,
                                stream>>>(
      reinterpret_cast<uint32_t**>(symm->get_signal_pad_ptrs_dev()),
      symm->get_rank(), symm->get_world_size());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (ctx.layout->comm) ctx.layout->comm->barrier();
}

TEST(FusedCubedSphereSymmetricMemory, OrientationMetadataMatchesStaged) {
  require_cuda_6rank_or_skip();
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
  if (ctx.layout->comm) ctx.layout->comm->barrier();
}

TEST(FusedCubedSphereSymmetricMemory, BoundaryStateConventionMatchesStaged) {
  require_cuda_6rank_or_skip();
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

TEST(FusedCubedSphereSymmetricMemory, RemoteEdgePayloadMatches) {
  require_cuda_6rank_or_skip();
  auto ctx = make_smoke_context();
  std::string group_name = "snapy:test:fused-cs-remote-read";
  initialize_symmetric_memory_group(*ctx.layout, group_name);

  auto symm_buffer = make_symmetric_buffer(ctx.device, group_name);
  auto symm = c10d::symmetric_memory::rendezvous(symm_buffer, group_name);
  clear_signal_slots(symm, ctx.layout->comm);

  auto stream = at::cuda::getCurrentCUDAStream(ctx.device.index());
  int blocks = kSides * kEdgeLen * kNc1;
  write_edge_payload_kernel<<<blocks, 1, 0, stream>>>(
      symm_buffer.data_ptr<double>(), symm->get_rank());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  sync_previous_kernel_writes<<<1, std::max(32, symm->get_world_size()), 0,
                                stream>>>(
      reinterpret_cast<uint32_t**>(symm->get_signal_pad_ptrs_dev()),
      symm->get_rank(), symm->get_world_size());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto errors =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(ctx.device));
  verify_remote_edge_kernel<<<blocks, 1, 0, stream>>>(
      symm->get_buffer_ptrs_dev(), ctx.edge_meta.data_ptr<int>(),
      errors.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  EXPECT_EQ(errors.cpu().item<int>(), 0);
  if (ctx.layout->comm) ctx.layout->comm->barrier();
}

TEST(FusedCubedSphereSymmetricMemory, ConstantStateExchangeHasZeroMassFlux) {
  require_cuda_6rank_or_skip();
  auto ctx = make_smoke_context();
  std::string group_name = "snapy:test:fused-cs-constant-state";
  initialize_symmetric_memory_group(*ctx.layout, group_name);

  auto symm_buffer = make_symmetric_buffer(ctx.device, group_name);
  auto symm = c10d::symmetric_memory::rendezvous(symm_buffer, group_name);
  clear_signal_slots(symm, ctx.layout->comm);

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
  fused_cubed_sphere_exchange_cuda(
      w, flux2, flux3, symm_buffer, symm->get_buffer_ptrs_dev(),
      reinterpret_cast<uint32_t**>(symm->get_signal_pad_ptrs_dev()), face,
      symm->get_rank(), symm->get_world_size(), side_meta, x2v, x2f, x3v, x3f,
      FusedReconScheme::WENO5, FusedReconScheme::WENO5,
      FusedRiemannSolver::ShallowRoe, FusedEos::ShallowWater, 1.4, 0., 0.,
      false, torch::Tensor(), torch::Tensor(), torch::Tensor(), 1,
      FusedPrimitiveProjector::None);
  AT_CUDA_CHECK(cudaStreamSynchronize(
      at::cuda::getCurrentCUDAStream(ctx.device.index())));

  auto expect_constant_state = [](torch::Tensor state, char const* label) {
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
  expect_constant_state(symm_buffer.select(1, ILT), "left state");
  expect_constant_state(symm_buffer.select(1, IRT), "right state");
  auto errors =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(ctx.device));
  auto stream = at::cuda::getCurrentCUDAStream(ctx.device.index());
  verify_hydro_remote_constant_kernel<<<kSides * kEdgeLen * kNc1, 1, 0,
                                        stream>>>(
      symm->get_buffer_ptrs_dev(), side_meta.data_ptr<int>(),
      errors.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  EXPECT_EQ(errors.cpu().item<int>(), 0);
  EXPECT_LT(flux2[IDN].abs().max().cpu().item<double>(), 1.e-10);
  EXPECT_LT(flux3[IDN].abs().max().cpu().item<double>(), 1.e-10);
  if (ctx.layout->comm) ctx.layout->comm->barrier();
}
