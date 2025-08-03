#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#define PRIM(n) prim[(n) * stride]
#define CONS(n) cons[(n) * stride]

namespace snap {

template <typename T>
inline DISPATCH_MACRO void ideal_gas_cons2prim(T* prim, T* cons, double gammad,
                                               int stride) {
  constexpr int IDN = Index::IDN;
  constexpr int IVX = Index::IVX;
  constexpr int IVY = Index::IVY;
  constexpr int IVZ = Index::IVZ;
  constexpr int IPR = Index::IPR;

  // den -> den
  PRIM(IDN) = CONS(IDN);

  // mom -> vel
  PRIM(IVX) = CONS(IVX) / PRIM(IDN);
  PRIM(IVY) = CONS(IVY) / PRIM(IDN);
  PRIM(IVZ) = CONS(IVZ) / PRIM(IDN);

  auto ke = 0.5 * (PRIM(IVX) * CONS(IVX) + PRIM(IVY) * CONS(IVY) +
                   PRIM(IVZ) * CONS(IVZ));
  auto ie = CONS(IPR) - ke;

  // eng -> pr
  PRIM(IPR) = (gammad - 1.) * ie;
}

}  // namespace snap

#undef PRIM
#undef CONS
