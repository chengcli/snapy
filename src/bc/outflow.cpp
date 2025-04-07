// fvm
#include "boundary_condition.hpp"

namespace canoe {

void outflow_inner(torch::Tensor var, int dim, BoundaryFuncOptions op) {
  if (var.size(dim) == 1) return;
  int nghost = op.nghost();

  std::vector<int64_t> shape(var.dim(), -1);
  shape[dim] = nghost;

  var.narrow(dim, 0, nghost) = var.narrow(dim, nghost, 1).expand(shape);
}

void outflow_outer(torch::Tensor var, int dim, BoundaryFuncOptions op) {
  if (var.size(dim) == 1) return;
  int nc1 = var.size(dim);
  int nghost = op.nghost();

  std::vector<int64_t> shape(var.dim(), -1);
  shape[dim] = nghost;

  var.narrow(dim, nc1 - nghost, nghost) =
      var.narrow(dim, nc1 - nghost - 1, 1).expand(shape);
}

}  // namespace canoe
