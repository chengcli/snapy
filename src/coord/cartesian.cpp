// snap
#include <configure.h>

#include "coordinate.hpp"

namespace snap {
void CartesianImpl::reset() {
  auto const& op = options;

  // dimension 1
  auto dx = (op.x1max() - op.x1min()) / op.nx1();
  dx1f = register_buffer("dx1f", torch::ones(op.nc1()) * dx);
  x1v = register_buffer(
      "x1v", 0.5 * (x1f.slice(0, 0, op.nc1()) + x1f.slice(0, 1, op.nc1() + 1)));

  // dimension 2
  dx = (op.x2max() - op.x2min()) / op.nx2();
  dx2f = register_buffer("dx2f", torch::ones(op.nc2()) * dx);
  x2v = register_buffer(
      "x2v", 0.5 * (x2f.slice(0, 0, op.nc2()) + x2f.slice(0, 1, op.nc2() + 1)));

  // dimension 3
  dx = (op.x3max() - op.x3min()) / op.nx3();
  dx3f = register_buffer("dx3f", torch::ones(op.nc3()) * dx);
  x3v = register_buffer(
      "x3v", 0.5 * (x3f.slice(0, 0, op.nc3()) + x3f.slice(0, 1, op.nc3() + 1)));

  // register buffers defined in the base class
  register_buffer("x1f", x1f);
  register_buffer("x2f", x2f);
  register_buffer("x3f", x3f);
}

void CartesianImpl::reset_coordinates(std::vector<MeshGenerator> meshgens) {
  CoordinateImpl::reset_coordinates(meshgens);

  // dimension 1
  dx1f.copy_(x1f.slice(0, 1, options.nc1() + 1) -
             x1f.slice(0, 0, options.nc1()));
  x1v.copy_(0.5 * (x1f.slice(0, 0, options.nc1()) +
                   x1f.slice(0, 1, options.nc1() + 1)));

  // dimension 2
  dx2f.copy_(x2f.slice(0, 1, options.nc2() + 1) -
             x2f.slice(0, 0, options.nc2()));
  x2v.copy_(0.5 * (x2f.slice(0, 0, options.nc2()) +
                   x2f.slice(0, 1, options.nc2() + 1)));

  // dimension 3
  dx3f.copy_(x3f.slice(0, 1, options.nc3() + 1) -
             x3f.slice(0, 0, options.nc3()));
  x3v.copy_(0.5 * (x3f.slice(0, 0, options.nc3()) +
                   x3f.slice(0, 1, options.nc3() + 1)));
}
}  // namespace snap
