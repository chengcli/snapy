// fvm
#include "boundary_condition.hpp"

namespace snap {

void periodic_inner(torch::Tensor var, int dim, BoundaryFuncOptions op) {
  if (var.size(dim) == 1) return;
  int nc1 = var.size(dim);
  int nghost = op.nghost();

  var.narrow(dim, 0, nghost) = var.narrow(dim, nc1 - 2 * nghost, nghost);
}

void periodic_outer(torch::Tensor var, int dim, BoundaryFuncOptions op) {
  if (var.size(dim) == 1) return;
  int nc1 = var.size(dim);
  int nghost = op.nghost();

  var.narrow(dim, nc1 - nghost, nghost) = var.narrow(dim, nghost, nghost);
}

}  // namespace snap
