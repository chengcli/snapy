#pragma once

// base
#include <configure.h>

namespace snap {

extern int my_rank, nranks;

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
  IVY = 2,
  IVZ = 3,
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

#define CHECK_MODULE_LINKED(parent, module)                                   \
  TORCH_CHECK(options->module(), #module " module is not linked in " #parent, \
              "\n Use options->" #module "()",                                \
              " to link the module before creating " #parent)

}  // namespace snap
