#pragma once

// C/C++
#include <functional>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

// arg
#include <snap/add_arg.h>

namespace snap {
enum BoundaryFace {
  kUnknown = -1,
  kInnerX1 = 0,
  kOuterX1 = 1,
  kInnerX2 = 2,
  kOuterX2 = 3,
  kInnerX3 = 4,
  kOuterX3 = 5
};

enum class BoundaryFlag : int {
  kExchange = -1,
  kUser = 0,
  kReflect = 1,
  kOutflow = 2,
  kPeriodic = 3,
  kShearPeriodic = 4,
  kPolar = 5,
  kPolarWedge = 6,
};

struct BoundaryFuncOptions {
  ADD_ARG(int, type) = kConserved;
  ADD_ARG(int, nghost) = 1;
};

}  // namespace snap

#undef ADD_ARG
