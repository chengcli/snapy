#pragma once

// base
#include <configure.h>

// C/C++
#include <cstdint>

// snap
#include <snap/coord/coord_utils_impl.h>
#include <snap/snap.h>

namespace snap {

template <typename T>
inline DISPATCH_MACRO T sedimentation_feps(T const* w, int flat, int stride_var,
                                           int ny, int nvapor,
                                           T const* inv_mu_ratio_m1) {
  T feps = 1.;
  for (int n = 0; n < nvapor; ++n) {
    feps += w[(ICY + n) * stride_var + flat] * inv_mu_ratio_m1[n];
  }
  for (int n = nvapor; n < ny; ++n) {
    feps -= w[(ICY + n) * stride_var + flat];
  }
  return feps;
}

template <typename T>
inline DISPATCH_MACRO void sedimentation_flux_impl(
    T const* w, T* flux, T* vsed_out, T const* cosine_cell_kj, T const* radius,
    T const* density, T const* const_vsed, int64_t const* hydro_ids,
    T const* inv_mu_ratio_m1, T const* cv_ratio_m1, T const* u0, int nparticle,
    int ny, int nvapor, int nvar, int nc3, int nc2, int nc1, int flat, int il,
    int iu, T grav, T gas_constant_dry, T cv_dry, T gas_diameter,
    T gas_epsilon_lj, T gas_mass, T upper_limit, T pi, T kboltz) {
  int ncells = nc1 * nc2 * nc3;
  int i = flat % nc1;
  int j = (flat / nc1) % nc2;
  int k = flat / (nc1 * nc2);
  T sedimenting = (i <= il || i > iu) ? T(0) : T(1);

  T rho = w[IDN * ncells + flat];
  T pres = w[IPR * ncells + flat];
  T feps = sedimentation_feps(w, flat, ncells, ny, nvapor, inv_mu_ratio_m1);
  T temp = pres / (rho * gas_constant_dry * feps);

  T v1 = w[IVX * ncells + flat];
  T v2 = w[IVY * ncells + flat];
  T v3 = w[IVZ * ncells + flat];
  T cth = cosine_cell_kj[k * nc2 + j];
  coord_vec_lower_impl(&v2, &v3, cth);
  T ke = T(0.5) * (w[IVX * ncells + flat] * v1 + w[IVY * ncells + flat] * v2 +
                   w[IVZ * ncells + flat] * v3);

  T eta = (T(5) / T(16)) * sqrt(pi * kboltz) * sqrt(gas_mass) * sqrt(temp) *
          pow(kboltz / gas_epsilon_lj * temp, T(0.16)) /
          (pi * gas_diameter * gas_diameter * T(1.22));
  T lambda =
      (eta * sqrt(pi * kboltz * kboltz)) / (pres * sqrt(T(2) * gas_mass));

  for (int p = 0; p < nparticle; ++p) {
    int hydro_id = static_cast<int>(hydro_ids[p]);
    int species_id = hydro_id - ICY;
    // A non-zero const_vsed is the whole velocity (prescribed, athena
    // convention); otherwise Stokes. grav is the SIGNED x1 acceleration,
    // so dense particles settle toward -x1.
    T vsed;
    if (const_vsed[p] != T(0)) {
      vsed = const_vsed[p];
    } else {
      T r = radius[p];
      T kn = lambda / r;
      T beta = T(1) + kn * (T(1.256) + T(0.4) * exp(-T(1.1) / kn));
      vsed = beta / (T(9) * eta) * (T(2) * r * r * grav * (density[p] - rho));
    }
    if (vsed < -upper_limit) vsed = -upper_limit;
    if (vsed > upper_limit) vsed = upper_limit;
    vsed *= sedimenting;
    vsed_out[p * ncells + flat] = vsed;

    T y = w[hydro_id * ncells + flat];
    T rhos_vsed = rho * y * vsed;
    T species_energy = rho * y *
                       (u0[1 + species_id] +
                        (T(1) + cv_ratio_m1[species_id]) * cv_dry * temp + ke);

    flux[hydro_id * ncells + flat] += rhos_vsed;
    flux[IVX * ncells + flat] += v1 * rhos_vsed;
    flux[IVY * ncells + flat] += v2 * rhos_vsed;
    flux[IVZ * ncells + flat] += v3 * rhos_vsed;
    flux[IPR * ncells + flat] += vsed * species_energy;
  }
}

}  // namespace snap
