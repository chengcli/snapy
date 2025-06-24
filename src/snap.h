#pragma once

namespace snap {

extern int my_rank, nranks;

enum Index {
  // hydro variables
  IDN = 0,
  IVX = 1,
  IVY = 2,
  IVZ = 3,
  IPR = 4,
  ICY = 5,

  // reconstruction variables
  ILT = 0,  //!< left interface
  IRT = 1,  //!< right interface
};

enum {
  // variable type
  kPrimitive = 0,
  kConserved = 1,
  kScalar = 2,

  // temperature, pressure, mass fraction with LR states
  kTPMassLR = 5,
  kDPMassLR = 6,
};

}  // namespace snap
