// eigen
#include <Eigen/Dense>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/ScalarType.h>

// snap
#include <snap/utils/cuda_utils.h>
#include <snap/utils/loops.cuh>

#include "implicit_dispatch.hpp"
#include "vic_solve_full_impl.h"
#include "vic_solve_partial_impl.h"

namespace snap {

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

    native::gpu_kernel<9>(
        iter, [=] GPU_LAMBDA(char* const data[9], unsigned int strides[9]) {
      auto du = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
      auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
      auto gamma = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
      auto area = reinterpret_cast<scalar_t *>(data[3] + strides[3]);
      auto vol = reinterpret_cast<scalar_t *>(data[4] + strides[4]);
      auto a = reinterpret_cast<Eigen::Matrix<scalar_t, 3, 3>*>(
          data[5] + strides[5]);
      auto b = reinterpret_cast<Eigen::Matrix<scalar_t, 3, 3>*>(
          data[6] + strides[6]);
      auto c = reinterpret_cast<Eigen::Matrix<scalar_t, 3, 3>*>(
          data[7] + strides[7]);
      auto delta = reinterpret_cast<Eigen::Matrix<scalar_t, 3, 1>*>(
          data[8] + strides[8]);

      vic_solve_partial_impl(du, w, gamma, area, vol, dt, grav, 0, nlayer - 1,
                             dir, ny, stride1, stride2,
                             first_block, last_block,
                             periodic, a, b, c, delta);
    });
  });
}

void vic_solve_full_cuda(at::TensorIterator &iter, double dt, double grav, int dir) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_solve_full_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    auto stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;
    bool first_block = true;
    bool last_block = true;
    bool periodic = false;

    native::gpu_kernel<9>(iter,
        [=] GPU_LAMBDA(char* const data[9], unsigned int strides[9]) {
      auto du = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
      auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
      auto gamma = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
      auto area = reinterpret_cast<scalar_t *>(data[3] + strides[3]);
      auto vol = reinterpret_cast<scalar_t *>(data[4] + strides[4]);
      auto a = reinterpret_cast<Eigen::Matrix<scalar_t, 5, 5>*>(
          data[5] + strides[5]);
      auto b = reinterpret_cast<Eigen::Matrix<scalar_t, 5, 5>*>(
          data[6] + strides[6]);
      auto c = reinterpret_cast<Eigen::Matrix<scalar_t, 5, 5>*>(
          data[7] + strides[7]);
      auto delta = reinterpret_cast<Eigen::Matrix<scalar_t, 5, 1>*>(
          data[8] + strides[8]);

      vic_solve_full_impl(du, w, gamma, area, vol, dt, grav, 0, nlayer - 1,
                          dir, ny, stride1, stride2, first_block, last_block,
                          periodic, a, b, c, delta);
    });
  });
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(vic_solve_partial, &snap::vic_solve_partial_cuda);
REGISTER_CUDA_DISPATCH(vic_solve_full, &snap::vic_solve_full_cuda);

}  // namespace at::native
