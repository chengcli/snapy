// eigen
#include <Eigen/Dense>

// C/C++
#include <vector>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <torch/torch.h>

// snap
#include "implicit_dispatch.hpp"
#include "vic_solve_full_impl.h"
#include "vic_solve_partial_impl.h"

namespace snap {

// On CPU the partial solve stays fused (assembly + sweep in
// vic_solve_partial_cpu below, using thread-local scratch), so the assemble
// phase is a no-op. The GPU path splits these into two kernels; this keeps the
// CPU path bit-identical to its previous behavior while honoring the same
// assemble-then-solve calls.
void vic_assemble_partial_cpu(at::TensorIterator& /*iter*/, double /*dt*/,
                              double /*grav*/, int /*dir*/,
                              bool /*conservation*/) {}

// On CPU the solve stays fused (vic_solve_partial_cpu does the full backward
// substitution + MS-VIC redistribution), so the redistribute phase is a no-op.
// On GPU these are separate kernels (vic_solve_partial does the sweep + column
// reductions; vic_redistribute_partial does the cell-parallel map).
void vic_redistribute_partial_cpu(at::TensorIterator& /*iter*/, double /*dt*/,
                                  double /*grav*/, int /*dir*/,
                                  bool /*conservation*/) {}

void vic_solve_partial_cpu(at::TensorIterator& iter, double dt, double grav,
                           int dir, bool conservation) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_solve_partial_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    auto stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;
    bool first_block = true;
    bool last_block = true;
    bool periodic = false;

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::vector<Eigen::Matrix<scalar_t, 3, 3>> a(nlayer), b(nlayer),
              c(nlayer);
          std::vector<Eigen::Matrix<scalar_t, 3, 1>> delta(nlayer);

          for (int i = 0; i < n; i++) {
            auto du = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto w = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto gamma = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            auto area = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            auto vol = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);

            vic_solve_partial_impl(
                du, w, gamma, area, vol, dt, grav, 0, nlayer - 1, dir, ny,
                stride1, stride2, first_block, last_block, periodic, a.data(),
                b.data(), c.data(), delta.data(), conservation);
          }
        },
        grain_size);
  });
}

void vic_solve_full_cpu(at::TensorIterator& iter, double dt, double grav,
                        int dir, bool conservation) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_solve_full_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    auto stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;
    bool first_block = true;
    bool last_block = true;
    bool periodic = false;

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::vector<Eigen::Matrix<scalar_t, 5, 5>> a(nlayer), b(nlayer),
              c(nlayer);
          std::vector<Eigen::Matrix<scalar_t, 5, 1>> delta(nlayer);

          for (int i = 0; i < n; i++) {
            auto du = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto w = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto gamma = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            auto area = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            auto vol = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);

            vic_solve_full_impl(du, w, gamma, area, vol, dt, grav, 0,
                                nlayer - 1, dir, ny, stride1, stride2,
                                first_block, last_block, periodic, a.data(),
                                b.data(), c.data(), delta.data(), conservation);
          }
        },
        grain_size);
  });
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(vic_assemble_partial);
DEFINE_DISPATCH(vic_solve_partial);
DEFINE_DISPATCH(vic_redistribute_partial);
DEFINE_DISPATCH(vic_solve_full);

REGISTER_ALL_CPU_DISPATCH(vic_assemble_partial,
                          &snap::vic_assemble_partial_cpu);
REGISTER_ALL_CPU_DISPATCH(vic_solve_partial, &snap::vic_solve_partial_cpu);
REGISTER_ALL_CPU_DISPATCH(vic_redistribute_partial,
                          &snap::vic_redistribute_partial_cpu);
REGISTER_ALL_CPU_DISPATCH(vic_solve_full, &snap::vic_solve_full_cpu);

}  // namespace at::native
