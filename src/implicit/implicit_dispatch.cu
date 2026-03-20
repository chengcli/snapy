// eigen
#include <Eigen/Dense>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/ScalarType.h>

#include <array>
#include <cstdlib>
#include <iostream>

// snap
#include <snap/utils/cuda_utils.h>
#include <snap/utils/loops.cuh>

#include "implicit_profile.cuh"
#include "implicit_dispatch.hpp"
#include "vic_solve_full_impl.h"
#include "vic_solve_partial_impl.h"

namespace snap {

__device__ unsigned long long g_vic_profile_cycles[kVicProfileCount];
__device__ unsigned long long g_vic_profile_calls[kVicProfileCount];
__device__ int g_vic_profile_enabled;

namespace {

bool vic_profile_requested() {
  static bool enabled = std::getenv("SNAPY_PROFILE_VIC") != nullptr;
  return enabled;
}

void reset_vic_profile(cudaStream_t stream) {
  std::array<unsigned long long, kVicProfileCount> zeros{};
  int enabled = 1;
  AT_CUDA_CHECK(cudaMemcpyToSymbolAsync(g_vic_profile_cycles, zeros.data(),
                                        sizeof(unsigned long long) *
                                            kVicProfileCount,
                                        0, cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaMemcpyToSymbolAsync(g_vic_profile_calls, zeros.data(),
                                        sizeof(unsigned long long) *
                                            kVicProfileCount,
                                        0, cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaMemcpyToSymbolAsync(g_vic_profile_enabled, &enabled,
                                        sizeof(int), 0,
                                        cudaMemcpyHostToDevice, stream));
}

void disable_vic_profile(cudaStream_t stream) {
  int disabled = 0;
  AT_CUDA_CHECK(cudaMemcpyToSymbolAsync(g_vic_profile_enabled, &disabled,
                                        sizeof(int), 0,
                                        cudaMemcpyHostToDevice, stream));
}

void report_vic_profile(cudaStream_t stream) {
  std::array<unsigned long long, kVicProfileCount> cycles{};
  std::array<unsigned long long, kVicProfileCount> calls{};
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  AT_CUDA_CHECK(cudaMemcpyFromSymbol(cycles.data(), g_vic_profile_cycles,
                                     sizeof(unsigned long long) *
                                         kVicProfileCount));
  AT_CUDA_CHECK(cudaMemcpyFromSymbol(calls.data(), g_vic_profile_calls,
                                     sizeof(unsigned long long) *
                                         kVicProfileCount));

  char const* rank = std::getenv("LOCAL_RANK");
  std::cout << "[vic-profile rank " << (rank ? rank : "?") << "] "
            << "forward_sweep_cycles=" << cycles[kVicForwardSweep]
            << " forward_sweep_calls=" << calls[kVicForwardSweep]
            << " inverse_cycles=" << cycles[kVicInverse]
            << " inverse_calls=" << calls[kVicInverse]
            << " backward_cycles=" << cycles[kVicBackwardSubstitution]
            << " backward_calls=" << calls[kVicBackwardSubstitution]
            << std::endl;
}

}  // namespace

void vic_solve_partial_cuda(at::TensorIterator &iter, double dt, double grav, int dir) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_solve_partial_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    auto stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;
    bool first_block = true;
    bool last_block = true;
    bool periodic = false;

    using Matrix = Eigen::Matrix<scalar_t, 3, 3>;
    using Vector = Eigen::Matrix<scalar_t, 3, 1>;

    native::gpu_kernel<9>(
        iter, [=] GPU_LAMBDA(char* const data[9], unsigned int strides[9]) {
          auto du = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto gamma = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          auto area = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
          auto vol = reinterpret_cast<scalar_t*>(data[4] + strides[4]);
          auto a = reinterpret_cast<Matrix*>(data[5] + strides[5]);
          auto b = reinterpret_cast<Matrix*>(data[6] + strides[6]);
          auto c = reinterpret_cast<Matrix*>(data[7] + strides[7]);
          auto delta = reinterpret_cast<Vector*>(data[8] + strides[8]);

          vic_solve_partial_impl(du, w, gamma, area, vol, dt, grav, 0,
                                 nlayer - 1, dir, ny, stride1, stride2,
                                 first_block, last_block, periodic, a, b, c,
                                 delta);
        });
  });
}

void vic_solve_full_cuda(at::TensorIterator &iter, double dt, double grav, int dir) {
  at::cuda::CUDAGuard device_guard(iter.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool profiled_once = false;
  bool do_profile = vic_profile_requested() && !profiled_once;

  if (do_profile) {
    reset_vic_profile(stream);
  }

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_solve_full_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    auto stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;
    bool first_block = true;
    bool last_block = true;
    bool periodic = false;

    using Matrix = Eigen::Matrix<scalar_t, 5, 5>;
    using Vector = Eigen::Matrix<scalar_t, 5, 1>;

    native::gpu_kernel<9>(
        iter, [=] GPU_LAMBDA(char* const data[9], unsigned int strides[9]) {
          auto du = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto gamma = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          auto area = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
          auto vol = reinterpret_cast<scalar_t*>(data[4] + strides[4]);
          auto a = reinterpret_cast<Matrix*>(data[5] + strides[5]);
          auto b = reinterpret_cast<Matrix*>(data[6] + strides[6]);
          auto c = reinterpret_cast<Matrix*>(data[7] + strides[7]);
          auto delta = reinterpret_cast<Vector*>(data[8] + strides[8]);

          vic_solve_full_impl(du, w, gamma, area, vol, dt, grav, 0, nlayer - 1,
                              dir, ny, stride1, stride2, first_block,
                              last_block, periodic, a, b, c, delta);
        });
  });

  if (do_profile) {
    report_vic_profile(stream);
    disable_vic_profile(stream);
    profiled_once = true;
  }
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(vic_solve_partial, &snap::vic_solve_partial_cuda);
REGISTER_CUDA_DISPATCH(vic_solve_full, &snap::vic_solve_full_cuda);

}  // namespace at::native
