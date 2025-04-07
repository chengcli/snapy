// snap
#include "boundary_condition.hpp"

namespace snap {

void reflect_inner(torch::Tensor var, int dim, BoundaryFuncOptions op) {
  if (var.size(dim) == 1) return;
  int nghost = op.nghost();

  var.narrow(dim, 0, nghost) = var.narrow(dim, nghost, nghost).flip(dim);

  // normal velocities
  if (op.type() == kConserved || op.type() == kPrimitive) {
    var[4 - dim].narrow(dim - 1, 0, nghost) *= -1;
  }
}

void reflect_outer(torch::Tensor var, int dim, BoundaryFuncOptions op) {
  if (var.size(dim) == 1) return;
  int nc1 = var.size(dim);
  int nghost = op.nghost();

  var.narrow(dim, nc1 - nghost, nghost) =
      var.narrow(dim, nc1 - 2 * nghost, nghost).flip(dim);

  // normal velocities
  if (op.type() == kConserved || op.type() == kPrimitive) {
    var[4 - dim].narrow(dim - 1, nc1 - nghost, nghost) *= -1;
  }
}

}  // namespace snap
