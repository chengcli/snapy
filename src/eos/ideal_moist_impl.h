#pragma once

// base
#include <configure.h>

// snap
#include <snap/index.h>

#define PRIM(n) prim[(n) * stride]
#define CONS(n) cons[(n) * stride]
#define GAMMAD (*gammad)
#define FEPS (*feps)
#define FSIG (*fsig)

namespace snap {

template <typename T>
inline DISPATCH_MACRO void ideal_moist_cons2prim(T* prim, T* cons, T* gammad,
                                                 T* feps, T* fsig, int nmass,
                                                 int stride) {
  constexpr int IDN = index::IDN;
  constexpr int IVX = index::IVX;
  constexpr int IVY = index::IVY;
  constexpr int IVZ = index::IVZ;
  constexpr int IPR = index::IPR;
  constexpr int ICY = index::ICY;

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
