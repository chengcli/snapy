// torch
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include "hydro_dispatch.hpp"
#include "hydro_ref_x1_impl.h"

namespace snap {

template <typename T>
__global__ void
hydro_ref_x1_cuda_kernel(T const *w, T const *dx1f, T const *anchor,
                         T const *gam, T *psf_lo, T *psf_hi, T *pref, T *dsf,
                         T *dref, int ncolumns, int nc1, int is, int iu, T grav,
                         bool uniform, bool phys_in, bool phys_out) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= ncolumns)
    return;
  hydro_ref_x1_impl(w, dx1f, anchor, gam, psf_lo, psf_hi, pref, dsf, dref,
                    column, ncolumns, nc1, is, iu, grav, uniform, phys_in,
                    phys_out);
}

void hydro_ref_x1_cuda(torch::Tensor const &w, torch::Tensor const &dx1f,
                       torch::Tensor const &anchor, torch::Tensor const &gam,
                       torch::Tensor const &psf_lo, torch::Tensor const &psf_hi,
                       torch::Tensor const &pref, torch::Tensor const &dsf,
                       torch::Tensor const &dref, int is, int iu, double grav,
                       bool uniform, bool phys_in, bool phys_out) {
  at::cuda::CUDAGuard device_guard(w.device());
  int ncolumns = w.size(1) * w.size(2);
  int nc1 = w.size(3);
  int threads = 128;
  int blocks = (ncolumns + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "hydro_ref_x1_cuda", [&] {
    auto anchor_ptr = anchor.defined() ? anchor.data_ptr<scalar_t>() : nullptr;
    hydro_ref_x1_cuda_kernel<<<blocks, threads, 0, stream>>>(
        w.data_ptr<scalar_t>(), dx1f.data_ptr<scalar_t>(), anchor_ptr,
        gam.data_ptr<scalar_t>(), psf_lo.data_ptr<scalar_t>(),
        psf_hi.data_ptr<scalar_t>(), pref.data_ptr<scalar_t>(),
        dsf.data_ptr<scalar_t>(), dref.data_ptr<scalar_t>(), ncolumns, nc1, is,
        iu, scalar_t(grav), uniform, phys_in, phys_out);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_hydro_ref_x1, &snap::hydro_ref_x1_cuda);

} // namespace at::native
