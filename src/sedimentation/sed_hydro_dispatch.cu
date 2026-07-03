// torch
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// kintera
#include <kintera/constants.h>

// snap
#include "sed_hydro_dispatch.hpp"
#include "sed_hydro_impl.h"

namespace snap {

template <typename T>
__global__ void call_sedimentation_flux_cuda(
    T const* w, T* flux, T* vsed, T const* cosine_cell_kj, T const* radius,
    T const* density, T const* const_vsed, int64_t const* hydro_ids,
    T const* inv_mu_ratio_m1, T const* cv_ratio_m1, T const* u0,
    int nparticle, int ny, int nvapor, int nvar, int nc3, int nc2, int nc1,
    int ncells, int il, int iu, T grav, T gas_constant_dry, T cv_dry,
    T gas_diameter, T gas_epsilon_lj, T gas_mass, T upper_limit, T pi,
    T kboltz) {
  int flat = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat >= ncells) return;
  sedimentation_flux_impl(
      w, flux, vsed, cosine_cell_kj, radius, density, const_vsed, hydro_ids,
      inv_mu_ratio_m1, cv_ratio_m1, u0, nparticle, ny, nvapor, nvar, nc3, nc2,
      nc1, flat, il, iu, grav, gas_constant_dry, cv_dry, gas_diameter,
      gas_epsilon_lj, gas_mass, upper_limit, pi, kboltz);
}

void sedimentation_flux_cuda(
    torch::Tensor w, torch::Tensor flux, torch::Tensor vsed,
    torch::Tensor cosine_cell_kj, torch::Tensor radius, torch::Tensor density,
    torch::Tensor const_vsed, torch::Tensor hydro_ids,
    torch::Tensor inv_mu_ratio_m1, torch::Tensor cv_ratio_m1, torch::Tensor u0,
    int il, int iu, int ny, int nvapor, double grav, double gas_constant_dry,
    double cv_dry, double gas_diameter, double gas_epsilon_lj, double gas_mass,
    double upper_limit) {
  at::cuda::CUDAGuard device_guard(w.device());
  int nvar = w.size(0);
  int nc3 = w.size(1);
  int nc2 = w.size(2);
  int nc1 = w.size(3);
  int ncells = nc1 * nc2 * nc3;
  int nparticle = radius.size(0);
  int threads = 128;
  int blocks = (ncells + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "sedimentation_flux_cuda", [&] {
    call_sedimentation_flux_cuda<scalar_t><<<blocks, threads, 0, stream>>>(
        w.data_ptr<scalar_t>(), flux.data_ptr<scalar_t>(),
        vsed.data_ptr<scalar_t>(), cosine_cell_kj.data_ptr<scalar_t>(),
        radius.data_ptr<scalar_t>(), density.data_ptr<scalar_t>(),
        const_vsed.data_ptr<scalar_t>(), hydro_ids.data_ptr<int64_t>(),
        inv_mu_ratio_m1.data_ptr<scalar_t>(),
        cv_ratio_m1.data_ptr<scalar_t>(), u0.data_ptr<scalar_t>(), nparticle,
        ny, nvapor, nvar, nc3, nc2, nc1, ncells, il, iu, scalar_t(grav),
        scalar_t(gas_constant_dry), scalar_t(cv_dry), scalar_t(gas_diameter),
        scalar_t(gas_epsilon_lj), scalar_t(gas_mass), scalar_t(upper_limit),
        scalar_t(M_PI), scalar_t(kintera::constants::KBoltz));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_sedimentation_flux,
                       &snap::sedimentation_flux_cuda);

}  // namespace at::native
