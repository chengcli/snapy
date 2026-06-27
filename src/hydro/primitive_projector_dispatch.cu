// torch
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include "primitive_projector_dispatch.hpp"
#include "primitive_projector_impl.h"

namespace snap {

template <typename T>
__global__ void call_primitive_projector_cuda(
    T const* w, T* wp, T* psf, T const* dx1f, int nvar, int nc3, int nc2,
    int nc1, int is, int ie, FusedPrimitiveProjector projector, T grav,
    T margin, T gas_constant) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= nc3 * nc2) return;
  primitive_projector_impl(w, wp, psf, dx1f, nvar, nc3, nc2, nc1, col, is, ie,
                           projector, grav, margin, gas_constant);
}

void primitive_projector_cuda(torch::Tensor w, torch::Tensor wp,
                              torch::Tensor psf, torch::Tensor dx1f, int is,
                              int ie, FusedPrimitiveProjector projector,
                              double grav, double margin,
                              double gas_constant) {
  at::cuda::CUDAGuard device_guard(w.device());
  int nc3 = w.size(1);
  int nc2 = w.size(2);
  int nc1 = w.size(3);
  int nvar = w.size(0);
  int cols = nc3 * nc2;
  int threads = 128;
  int blocks = (cols + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "primitive_projector_cuda", [&] {
    call_primitive_projector_cuda<scalar_t><<<blocks, threads, 0, stream>>>(
        w.data_ptr<scalar_t>(), wp.data_ptr<scalar_t>(),
        psf.data_ptr<scalar_t>(), dx1f.data_ptr<scalar_t>(), nvar, nc3, nc2,
        nc1, is, ie, projector, scalar_t(grav), scalar_t(margin),
        scalar_t(gas_constant));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_primitive_projector,
                       &snap::primitive_projector_cuda);

}  // namespace at::native
