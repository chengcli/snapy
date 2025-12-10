#pragma once

// base
#include <configure.h>

// snap
#include <snap/coord/coord_utils_impl.h>
#include <snap/snap.h>

#define PRIM(n) prim[(n) * stride]
#define CONS(n) cons[(n) * stride]

namespace snap {

template <typename T>
inline DISPATCH_MACRO void ideal_gas_cons2prim(T* prim, T* cons, T* cos_theta,
                                               double gammad, int stride) {
  // den -> den
  PRIM(IDN) = CONS(IDN);

  // mom -> vel
  PRIM(IVX) = CONS(IVX) / PRIM(IDN);
  PRIM(IVY) = CONS(IVY) / PRIM(IDN);
  PRIM(IVZ) = CONS(IVZ) / PRIM(IDN);

  // co to contra
  coord_vec_raise_impl(&PRIM(IVY), &PRIM(IVZ), *cos_theta);

  auto ke = 0.5 * (PRIM(IVX) * CONS(IVX) + PRIM(IVY) * CONS(IVY) +
                   PRIM(IVZ) * CONS(IVZ));
  auto ie = CONS(IPR) - ke;

  // eng -> pr
  PRIM(IPR) = (gammad - 1.) * ie;
}

}  // namespace snap

#undef PRIM
#undef CONS
