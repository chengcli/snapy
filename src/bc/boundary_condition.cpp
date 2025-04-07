// C/C++
#include <stdexcept>

// fvm
#include "boundary_condition.hpp"

namespace snap {

void bc_null_op(torch::Tensor var, int dim, BoundaryFuncOptions op) {
  // do nothing
}

bfunc_t get_boundary_function(BoundaryFace face, BoundaryFlag flag) {
  if (flag == BoundaryFlag::kUser) {
    return bc_null_op;
  }

  switch (face) {
    case BoundaryFace::kInnerX1:
    case BoundaryFace::kInnerX2:
    case BoundaryFace::kInnerX3:
      switch (flag) {
        case BoundaryFlag::kReflect:
          return reflect_inner;
        case BoundaryFlag::kOutflow:
          return outflow_inner;
        case BoundaryFlag::kPeriodic:
          return periodic_inner;
        case BoundaryFlag::kExchange:
          return exchange_inner;
        default:
          throw std::runtime_error("get_boundary_function: invalid flag");
      }
    case BoundaryFace::kOuterX1:
    case BoundaryFace::kOuterX2:
    case BoundaryFace::kOuterX3:
      switch (flag) {
        case BoundaryFlag::kReflect:
          return reflect_outer;
        case BoundaryFlag::kOutflow:
          return outflow_outer;
        case BoundaryFlag::kPeriodic:
          return periodic_outer;
        case BoundaryFlag::kExchange:
          return exchange_outer;
        default:
          throw std::runtime_error("get_boundary_function: invalid flag");
      }
    default:
      throw std::runtime_error("get_boundary_function: invalid face");
  }
}

}  // namespace snap
