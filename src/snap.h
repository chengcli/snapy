#pragma once

// C/C++
#include <memory>
#include <unordered_map>

#ifndef __CUDACC__
#include <torch/torch.h>
#endif

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

#ifndef __CUDACC__
//! dump of shared data to other modules
extern std::unordered_map<std::string, torch::Tensor> shared;

//! names of all species
extern std::vector<std::string> species_names;

//! molecular weights of all species [kg/mol]
extern std::vector<double> species_weights;
#endif

}  // namespace snap

// shared data
#define SET_SHARED(name) snap::shared[fmt::format("{}", name)]
#define GET_SHARED(name) snap::shared.at(fmt::format("{}", name))
#define HAS_SHARED(name) \
  (snap::shared.find(fmt::format("{}", name)) != snap::shared.end())
