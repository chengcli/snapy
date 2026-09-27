#pragma once

// torch
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

namespace snap {

//! Moves per-cell sedimentation fluxes q (x1 last) onto x1 faces by donor
//! cell: face i takes cell i where vsed < 0 (settling) and cell i-1 where
//! vsed > 0 (rising). Faces at or below il and above iu are sealed.
inline torch::Tensor sedimentation_upwind(torch::Tensor q, torch::Tensor vsed,
                                          int il, int iu) {
  auto zero = torch::zeros_like(q);
  auto face = torch::where(vsed < 0., q, zero);
  int64_t n = face.size(-1);
  if (n > 1) {
    face.narrow(-1, 1, n - 1) +=
        torch::where(vsed > 0., q, zero).narrow(-1, 0, n - 1);
  }
  face.slice(-1, iu + 1, n).fill_(0.);
  face.slice(-1, 0, il + 1).fill_(0.);
  return face;
}

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
