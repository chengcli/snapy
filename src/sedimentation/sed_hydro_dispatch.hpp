#pragma once

// torch
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

namespace snap {

void sedimentation_flux_dispatch(
    torch::Tensor w, torch::Tensor flux, torch::Tensor vsed,
    torch::Tensor cosine_cell_kj, torch::Tensor radius, torch::Tensor density,
    torch::Tensor const_vsed, torch::Tensor hydro_ids,
    torch::Tensor inv_mu_ratio_m1, torch::Tensor cv_ratio_m1, torch::Tensor u0,
    int il, int iu, int ny, int nvapor, double grav, double gas_constant_dry,
    double cv_dry, double gas_diameter, double gas_epsilon_lj, double gas_mass,
    double upper_limit);

}  // namespace snap

namespace at::native {

using sedimentation_flux_fn = void (*)(
    torch::Tensor w, torch::Tensor flux, torch::Tensor vsed,
    torch::Tensor cosine_cell_kj, torch::Tensor radius, torch::Tensor density,
    torch::Tensor const_vsed, torch::Tensor hydro_ids,
    torch::Tensor inv_mu_ratio_m1, torch::Tensor cv_ratio_m1, torch::Tensor u0,
    int il, int iu, int ny, int nvapor, double grav, double gas_constant_dry,
    double cv_dry, double gas_diameter, double gas_epsilon_lj, double gas_mass,
    double upper_limit);

DECLARE_DISPATCH(sedimentation_flux_fn, call_sedimentation_flux);

}  // namespace at::native
