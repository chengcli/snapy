// fvm
#include "boundary_condition.hpp"

namespace canoe {

void exchange_inner(torch::Tensor var, int dim, BoundaryFuncOptions op) {}
void exchange_outer(torch::Tensor var, int dim, BoundaryFuncOptions op) {}

}  // namespace canoe
