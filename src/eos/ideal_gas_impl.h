#pragma once

// base
#include <configure.h>

// fvm
#include <fvm/index.h>

#define PRIM(n) prim[(n) * stride]
#define CONS(n) cons[(n) * stride]
#define GAMMAD (*gammad)

namespace snap {

template <typename T>
inline DISPATCH_MACRO void ideal_gas_cons2prim(T* prim, T* cons, T* gammad,
                                               int stride) {
  constexpr int IDN = index::IDN;
  constexpr int IVX = index::IVX;
  constexpr int IVY = index::IVY;
  constexpr int IVZ = index::IVZ;
  constexpr int IPR = index::IPR;

  // den -> den
  PRIM(IDN) = CONS(IDN);

  // mom -> vel
  PRIM(IVX) = CONS(IVX) / PRIM(IDN);
  PRIM(IVY) = CONS(IVY) / PRIM(IDN);
  PRIM(IVZ) = CONS(IVZ) / PRIM(IDN);

  auto ke = 0.5 * (PRIM(IVX) * CONS(IVX) + PRIM(IVY) * CONS(IVY) +
                   PRIM(IVZ) * CONS(IVZ));

  // eng -> pr
  PRIM(IPR) = (GAMMAD - 1.) * (CONS(IPR) - ke);
}

}  // namespace snap

#undef PRIM
#undef CONS
#undef GAMMAD
