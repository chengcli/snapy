#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#define PRIM(n) prim[(n) * stride]
#define CONS(n) cons[(n) * stride]
#define GAMMAD (*gammad)
#define FEPS (*feps)
#define FSIG (*fsig)

namespace snap {

template <typename T>
inline DISPATCH_MACRO T ideal_moist_feps(T const* y, int ny, int nvapor,
                                         T const* inv_mu_ratio_m1) {
  T out = 1.;
  for (int n = 0; n < nvapor; ++n) out += y[n] * inv_mu_ratio_m1[n];
  for (int n = nvapor; n < ny; ++n) out -= y[n];
  return out;
}

template <typename T>
inline DISPATCH_MACRO T ideal_moist_fsig(T const* y, int ny,
                                         T const* cv_ratio_m1) {
  T out = 1.;
  for (int n = 0; n < ny; ++n) out += y[n] * cv_ratio_m1[n];
  return out;
}

template <typename T>
inline DISPATCH_MACRO void ideal_moist_cons2prim(T* prim, T* cons, T* gammad,
                                                 T* feps, T* fsig, int nmass,
                                                 int stride) {
  // den -> mixr
  for (int n = 0; n < nmass; ++n) {
    PRIM(ICY + n) = CONS(ICY + n) / PRIM(IDN);
  }

  // mom -> vel
  PRIM(IVX) = CONS(IVX) / PRIM(IDN);
  PRIM(IVY) = CONS(IVY) / PRIM(IDN);
  PRIM(IVZ) = CONS(IVZ) / PRIM(IDN);

  // pcoord->vec_raise_inplace(prim);

  auto ke = 0.5 * (PRIM(IVX) * CONS(IVX) + PRIM(IVY) * CONS(IVY) +
                   PRIM(IVZ) * CONS(IVZ));

  // eng -> pr
  PRIM(IPR) = (GAMMAD - 1.) * (CONS(IPR) - ke) * FEPS / FSIG;
}

}  // namespace snap

#undef PRIM
#undef CONS
#undef GAMMAD
#undef FEPS
#undef FSIG
