// eigen
#include <Eigen/Dense>

// C/C++
#include <array>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/utils/cuda_utils.h>

#include <snap/utils/loops.cuh>

#include "forward_sweep_impl.h"
#include "implicit_dispatch.hpp"
#include "vic_assemble_full_impl.h"
#include "vic_assemble_partial_impl.h"
#include "vic_redistribute_impl.h"

namespace snap {

// Phase 1 of the partial VIC solve: assemble the block-tridiagonal coefficients
// a, b, c. This is parallelized over EVERY cell (column x layer), not just per
// column, so the GPU is filled and the expensive Roe/eigendecomposition work no
// longer runs as a serial per-column loop. Results are bit-identical to the
// fused assembly that previously lived in vic_solve_partial_cuda.
void vic_assemble_partial_cuda(at::TensorIterator &iter, double dt, double grav,
                               int dir) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_assemble_partial_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    int nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    int stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    int stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;
    bool first_block = true;
    bool last_block = true;

    using Matrix = Eigen::Matrix<scalar_t, 3, 3>;

    int64_t ncol = iter.numel();
    auto offset_calc = ::make_offset_calculator<10>(iter);
    std::array<char *, 10> data;
    for (int k = 0; k < 10; ++k) data[k] = (char *)iter.data_ptr(k);

    int64_t total = ncol * (int64_t)nlayer;
    at::native::launch_legacy_kernel<128, 1>(total, [=] __device__(int idx) {
      int col = idx / nlayer;
      int i = idx % nlayer;
      auto offsets = offset_calc.get(col);
      auto w = reinterpret_cast<scalar_t *>(data[2] + offsets[2]);
      auto gamma = reinterpret_cast<scalar_t *>(data[3] + offsets[3]);
      auto area = reinterpret_cast<scalar_t *>(data[4] + offsets[4]);
      auto vol = reinterpret_cast<scalar_t *>(data[5] + offsets[5]);
      auto a = reinterpret_cast<Matrix *>(data[6] + offsets[6]);
      auto b = reinterpret_cast<Matrix *>(data[7] + offsets[7]);
      auto c = reinterpret_cast<Matrix *>(data[8] + offsets[8]);

      vic_assemble_partial_impl(a, b, c, w, gamma, area, vol, i, 0, nlayer - 1,
                                dt, grav, dir, ny, stride1, stride2,
                                first_block, last_block);
    });
  });
}

void vic_assemble_full_cuda(at::TensorIterator &iter, double dt, double grav,
                            int dir) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_assemble_full_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    auto stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;
    bool first_block = true;
    bool last_block = true;
    bool periodic = false;

    using Matrix = Eigen::Matrix<scalar_t, 5, 5>;

    int64_t ncol = iter.numel();
    auto offset_calc = ::make_offset_calculator<10>(iter);
    std::array<char *, 10> data;
    for (int k = 0; k < 10; ++k) data[k] = (char *)iter.data_ptr(k);

    int64_t total = ncol * (int64_t)nlayer;
    at::native::launch_legacy_kernel<128, 1>(total, [=] __device__(int idx) {
      int col = idx / nlayer;
      int i = idx % nlayer;
      auto offsets = offset_calc.get(col);
      auto w = reinterpret_cast<scalar_t *>(data[2] + offsets[2]);
      auto gamma = reinterpret_cast<scalar_t *>(data[3] + offsets[3]);
      auto area = reinterpret_cast<scalar_t *>(data[4] + offsets[4]);
      auto vol = reinterpret_cast<scalar_t *>(data[5] + offsets[5]);
      auto a = reinterpret_cast<Matrix *>(data[6] + offsets[6]);
      auto b = reinterpret_cast<Matrix *>(data[7] + offsets[7]);
      auto c = reinterpret_cast<Matrix *>(data[8] + offsets[8]);

      vic_assemble_full_impl(a, b, c, w, gamma, area, vol, i, 0, nlayer - 1, dt,
                             grav, dir, ny, stride1, stride2, first_block,
                             last_block, periodic);
    });
  });
}

template <int N>
void vic_solve_cuda(at::TensorIterator &iter, double dt, double grav, int dir) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_solve_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    auto stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;
    bool first_block = true;
    bool last_block = true;

    using Matrix = Eigen::Matrix<scalar_t, N, N>;
    using Vector = Eigen::Matrix<scalar_t, N, 1>;

    native::gpu_kernel<10>(
        iter, [=] GPU_LAMBDA(char *const data[10], unsigned int strides[10]) {
          auto du = reinterpret_cast<scalar_t *>(data[0] + strides[0]);
          auto a = reinterpret_cast<Matrix *>(data[6] + strides[6]);
          auto b = reinterpret_cast<Matrix *>(data[7] + strides[7]);
          auto c = reinterpret_cast<Matrix *>(data[8] + strides[8]);
          auto delta = reinterpret_cast<Vector *>(data[9] + strides[9]);

          ForwardSweep(a, b, c, delta, du, dt, 0, nlayer - 1, dir, ny, stride1,
                       stride2, first_block, last_block);
          vic_backward_substitute(a, delta, 0, nlayer - 1);
        });
  });
}

template void vic_solve_cuda<3>(at::TensorIterator &, double, double, int);
template void vic_solve_cuda<5>(at::TensorIterator &, double, double, int);

template <int N>
void vic_redistribute_cuda(at::TensorIterator &iter, double /*dt*/,
                           double /*grav*/, int dir) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_redistribute_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    int nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    int stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    int stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;

    using Vector = Eigen::Matrix<scalar_t, N, 1>;

    int64_t ncol = iter.numel();
    auto offset_calc = ::make_offset_calculator<10>(iter);
    std::array<char *, 10> data;
    for (int k = 0; k < 10; ++k) data[k] = (char *)iter.data_ptr(k);

    // Component B: the constituent column pass is serial per column (prefix sum
    // + sequential availability clamp); one thread per column, matching the
    // ForwardSweep pattern. Writes only mass_fix, so the cell-parallel
    // redistribution below stays race-free.
    at::native::launch_legacy_kernel<128, 1>(ncol, [=] __device__(int col) {
      auto offsets = offset_calc.get(col);
      auto du = reinterpret_cast<scalar_t *>(data[0] + offsets[0]);
      auto mass_fix = reinterpret_cast<scalar_t *>(data[1] + offsets[1]);
      auto w = reinterpret_cast<scalar_t *>(data[2] + offsets[2]);
      auto vol = reinterpret_cast<scalar_t *>(data[5] + offsets[5]);
      auto delta = reinterpret_cast<Vector *>(data[9] + offsets[9]);

      vic_constituent_column<scalar_t, N>(du, w, mass_fix, delta, vol, nlayer,
                                          dir, ny, stride1, stride2);
    });

    int64_t total = ncol * (int64_t)nlayer;
    at::native::launch_legacy_kernel<128, 1>(total, [=] __device__(int idx) {
      int col = idx / nlayer;
      int i = idx % nlayer;
      auto offsets = offset_calc.get(col);
      auto du = reinterpret_cast<scalar_t *>(data[0] + offsets[0]);
      auto mass_fix = reinterpret_cast<scalar_t *>(data[1] + offsets[1]);
      auto delta = reinterpret_cast<Vector *>(data[9] + offsets[9]);

      vic_redistribute_cell(du, mass_fix, delta, i, dir, ny, stride1, stride2);
    });
  });
}

template void vic_redistribute_cuda<3>(at::TensorIterator &, double, double,
                                       int);
template void vic_redistribute_cuda<5>(at::TensorIterator &, double, double,
                                       int);

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(vic_assemble_partial, &snap::vic_assemble_partial_cuda);
REGISTER_CUDA_DISPATCH(vic_assemble_full, &snap::vic_assemble_full_cuda);
REGISTER_CUDA_DISPATCH(vic_solve_partial, &snap::vic_solve_cuda<3>);
REGISTER_CUDA_DISPATCH(vic_solve_full, &snap::vic_solve_cuda<5>);
REGISTER_CUDA_DISPATCH(vic_redistribute_partial,
                       &snap::vic_redistribute_cuda<3>);
REGISTER_CUDA_DISPATCH(vic_redistribute_full, &snap::vic_redistribute_cuda<5>);

}  // namespace at::native
