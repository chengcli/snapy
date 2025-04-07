// fvm
#include "boundary_condition.hpp"

namespace canoe {

void solid_inner(torch::Tensor var, int dim, BoundaryFuncOptions op) {
  if (var.size(dim) == 1) return;
  int nghost = op.nghost();

  std::vector<int64_t> shape(var.dim(), -1);
  shape[dim] = nghost;

  var.narrow(dim, 0, nghost) = 1;
}

void solid_outer(torch::Tensor var, int dim, BoundaryFuncOptions op) {
  if (var.size(dim) == 1) return;
  int nc1 = var.size(dim);
  int nghost = op.nghost();

  std::vector<int64_t> shape(var.dim(), -1);
  shape[dim] = nghost;

  var.narrow(dim, nc1 - nghost, nghost) = 1;
}

}  // namespace canoe
