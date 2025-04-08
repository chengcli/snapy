#pragma once

// base
#include <configure.h>

namespace snap {

//! Eq.1 in Li2018
template <typename T>
inline DISPATCH_MACRO T cal_dlnT_dlnP(T const* q, T const* gammad,
                                      T const* cp_ratio_mole, T const* latent,
                                      int nvapor, int ncloud) {
  T x_gas = 1.;
  for (int n = nvapor; n < nvapor + ncloud; ++n) {
    x_gas -= q[n];
  }

  T f_sig = 1.;
  // vapor
  for (int n = 0; n < nvapor; ++n) {
    f_sig += q[n] * (cp_ratio_mole[n] - 1.);
  }
  T cphat_ov_r = (*gammad) / ((*gammad) - 1.) * f_sig / x_gas;

  // vapor
  T xd = x_gas;
  for (int n = 0; n < nvapor; ++n) xd -= q[n];

  T c1 = 0., c2 = 0., c3 = 0.;
  for (int n = 0; n < nvapor; ++n) {
    c1 += q[n] / xd * latent[n];
    c2 += q[n] / xd * latent[n] * latent[n];
    c3 += q[n] / xd;
  }

  return (1. + c1) / (cphat_ov_r + (c2 + c1 * c1) / (1. + c3));
}

}  // namespace snap
