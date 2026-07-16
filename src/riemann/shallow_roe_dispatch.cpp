// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>

// snap
#include <snap/snap.h>

#include "riemann_dispatch.hpp"
#include "shallow_roe_impl.h"

namespace snap {

void call_shallow_roe_cpu(at::TensorIterator& iter, int dim, int dir_yz) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_shallow_roe_cpu", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int64_t i = 0; i < n; ++i) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto wl = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto wr = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            shallow_roe_impl(out, wl, wr, dim, dir_yz, stride, stride);
          }
        },
        grain_size);
  });
}

void call_shallow_roe_mps(at::TensorIterator& iter, int dim, int dir_yz) {
  auto flx = iter.output(0);
  auto wl = iter.input(0);
  auto wr = iter.input(1);

  int ivx, ivy, ivz;
  if (dir_yz) {
    ivx = dim == 2 ? IVY : IVZ;
    ivy = dim == 2 ? IVZ : IVY;
    ivz = IVX;
  } else {
    ivx = dim == 3 ? IVX : IVY;
    ivy = dim == 3 ? IVY : IVX;
    ivz = IVZ;
  }

  auto sqrtdl = torch::sqrt(wl[IDN]);
  auto sqrtdr = torch::sqrt(wr[IDN]);
  auto isdlpdr = 1.0 / (sqrtdl + sqrtdr);
  auto ubar = (wl[ivx] * sqrtdl + wr[ivx] * sqrtdr) * isdlpdr;
  auto vbar = (wl[ivy] * sqrtdl + wr[ivy] * sqrtdr) * isdlpdr;
  auto cbar = torch::sqrt(0.5 * (wl[IDN] + wr[IDN]));
  auto del = wr - wl;
  auto hbar = torch::sqrt(wl[IDN] * wr[IDN]);

  auto a1 = 0.5 * (cbar * del[IDN] - hbar * del[ivx]) / cbar;
  auto a2 = hbar * del[ivy];
  auto a3 = 0.5 * (cbar * del[IDN] + hbar * del[ivx]) / cbar;
  auto wave0 = torch::zeros_like(del);
  auto wave1 = torch::zeros_like(del);
  auto wave2 = torch::zeros_like(del);
  wave0[IDN] = a1;
  wave0[ivx] = a1 * (ubar - cbar);
  wave0[ivy] = a1 * vbar;
  wave1[ivy] = a2;
  wave2[IDN] = a3;
  wave2[ivx] = a3 * (ubar + cbar);
  wave2[ivy] = a3 * vbar;

  auto speed0 = torch::abs(ubar - cbar);
  auto speed1 = torch::abs(ubar);
  auto speed2 = torch::abs(ubar + cbar);
  flx[IDN] = 0.5 * (wl[IDN] * wl[ivx] + wr[IDN] * wr[ivx]);
  flx[ivx] = 0.5 * (wl[IDN] * wl[ivx].square() + 0.5 * wl[IDN].square() +
                    wr[IDN] * wr[ivx].square() + 0.5 * wr[IDN].square());
  flx[ivy] = 0.5 * (wl[IDN] * wl[ivx] * wl[ivy] + wr[IDN] * wr[ivx] * wr[ivy]);
  flx[ivz] = 0.0;
  flx -= 0.5 * (speed0 * wave0 + speed1 * wave1 + speed2 * wave2);
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_shallow_roe);
REGISTER_ALL_CPU_DISPATCH(call_shallow_roe, &snap::call_shallow_roe_cpu);
REGISTER_MPS_DISPATCH(call_shallow_roe, &snap::call_shallow_roe_mps);

}  // namespace at::native
