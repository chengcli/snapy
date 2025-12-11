#pragma once

// base
#include <configure.h>

namespace snap {

#if NMASS > 0  // use legacy Athena++ indexing scheme

enum Index {
  // hydro variables
  IDN = 0,
  ICY = 1,
  IVX = 1 + NMASS,
  IVY = 2 + NMASS,
  IVZ = 3 + NMASS,
  IPR = 4 + NMASS,

  // reconstruction variables
  ILT = 0,  //!< left interface
  IRT = 1,  //!< right interface
};

#else  // use new indexing scheme

enum Index {
  // hydro variables
  IDN = 0,
  IVX = 1,
  IV1 = 1,
  IVY = 2,
  IV2 = 2,
  IVZ = 3,
  IV3 = 3,
  IPR = 4,
  ICY = 5,

  // reconstruction variables
  ILT = 0,  //!< left interface
  IRT = 1,  //!< right interface
};

#endif  // index scheme

enum {
  // variable type
  kPrimitive = 0,
  kConserved = 1,
  kScalar = 2,

  // temperature, pressure, mass fraction with LR states
  kTPMassLR = 5,
  kDPMassLR = 6,
};

enum { VEL1 = 0, VEL2 = 1, VEL3 = 2 };

}  // namespace snap
