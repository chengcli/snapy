// snap
#include "bc_func.hpp"

BC_FUNCTION(custom_inner, var, dim, op) {}
BC_FUNCTION(custom_outer, var, dim, op) {}

BC_FUNCTION(reflecting_inner, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nghost = op.nghost();

  var.narrow(dim, 0, nghost) = var.narrow(dim, nghost, nghost).flip(dim);

  // normal velocities
  if (op.type() == snap::kConserved || op.type() == snap::kPrimitive) {
    var[4 - dim].narrow(dim - 1, 0, nghost) *= -1;
  }
}

BC_FUNCTION(reflecting_outer, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nc1 = var.size(dim);
  int nghost = op.nghost();

  var.narrow(dim, nc1 - nghost, nghost) =
      var.narrow(dim, nc1 - 2 * nghost, nghost).flip(dim);

  // normal velocities
  if (op.type() == snap::kConserved || op.type() == snap::kPrimitive) {
    var[4 - dim].narrow(dim - 1, nc1 - nghost, nghost) *= -1;
  }
}

BC_FUNCTION(periodic_inner, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nc1 = var.size(dim);
  int nghost = op.nghost();

  var.narrow(dim, 0, nghost) = var.narrow(dim, nc1 - 2 * nghost, nghost);
}

BC_FUNCTION(periodic_outer, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nc1 = var.size(dim);
  int nghost = op.nghost();

  var.narrow(dim, nc1 - nghost, nghost) = var.narrow(dim, nghost, nghost);
}

BC_FUNCTION(outflow_inner, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nghost = op.nghost();

  std::vector<int64_t> shape(var.dim(), -1);
  shape[dim] = nghost;

  var.narrow(dim, 0, nghost) = var.narrow(dim, nghost, 1).expand(shape);
}

BC_FUNCTION(outflow_outer, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nc1 = var.size(dim);
  int nghost = op.nghost();

  std::vector<int64_t> shape(var.dim(), -1);
  shape[dim] = nghost;

  var.narrow(dim, nc1 - nghost, nghost) =
      var.narrow(dim, nc1 - nghost - 1, 1).expand(shape);
}

BC_FUNCTION(solid_inner, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nghost = op.nghost();

  std::vector<int64_t> shape(var.dim(), -1);
  shape[dim] = nghost;

  var.narrow(dim, 0, nghost) = 1;
}

BC_FUNCTION(solid_outer, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nc1 = var.size(dim);
  int nghost = op.nghost();

  std::vector<int64_t> shape(var.dim(), -1);
  shape[dim] = nghost;

  var.narrow(dim, nc1 - nghost, nghost) = 1;
}
