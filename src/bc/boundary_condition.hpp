#pragma once

// C/C++
#include <functional>

// torch
#include <torch/torch.h>

// clang-format off
#include <configure.h>
#include <add_arg.h>
// clang-format on

// snap
#include <snap/index.h>

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

using bfunc_t = std::function<void(torch::Tensor, int, BoundaryFuncOptions)>;

void bc_null_op(torch::Tensor, int, BoundaryFuncOptions);

void reflect_inner(torch::Tensor, int dim, BoundaryFuncOptions);
void reflect_outer(torch::Tensor, int dim, BoundaryFuncOptions);

void outflow_inner(torch::Tensor, int dim, BoundaryFuncOptions);
void outflow_outer(torch::Tensor, int dim, BoundaryFuncOptions);

void periodic_inner(torch::Tensor, int dim, BoundaryFuncOptions);
void periodic_outer(torch::Tensor, int dim, BoundaryFuncOptions);

void exchange_inner(torch::Tensor, int dim, BoundaryFuncOptions);
void exchange_outer(torch::Tensor, int dim, BoundaryFuncOptions);

void solid_inner(torch::Tensor, int dim, BoundaryFuncOptions);
void solid_outer(torch::Tensor, int dim, BoundaryFuncOptions);

bfunc_t get_boundary_function(BoundaryFace face, BoundaryFlag flag);

}  // namespace snap
