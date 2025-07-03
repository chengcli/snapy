// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "internal_boundary.hpp"

namespace snap {

InternalBoundaryOptions InternalBoundaryOptions::from_yaml(
    const YAML::Node &root) {
  InternalBoundaryOptions op;

  if (!root["geometry"]) return op;
  if (!root["geometry"]["cells"]) return op;

  op.nghost() = root["geometry"]["cells"]["nghost"].as<int>(1);
  printf("* nghost = %d\n", op.nghost());

  if (!root["boundary-condition"]) return op;
  if (!root["boundary-condition"]["internal"]) return op;

  auto bc = root["boundary-condition"]["internal"];

  op.max_iter() = bc["max-iter"].as<int>(5);
  printf("* max-iter = %d\n", op.max_iter());

  op.solid_density() = bc["solid_density"].as<double>(1.e3);
  printf("* solid_density = %e\n", op.solid_density());

  op.solid_pressure() = bc["solid_pressure"].as<double>(1.9);
  printf("* solid_pressure = %e\n", op.solid_pressure());

  return op;
}

InternalBoundaryImpl::InternalBoundaryImpl(InternalBoundaryOptions options_)
    : options(options_) {
  reset();
}

void InternalBoundaryImpl::reset() {}

torch::Tensor InternalBoundaryImpl::mark_solid(torch::Tensor w,
                                               torch::Tensor solid) {
  if (!solid.defined()) return w;

  auto fill_solid = torch::zeros({w.size(0), 1, 1, 1}, w.options());

  fill_solid[Index::IDN] = options.solid_density();
  fill_solid[Index::IPR] = options.solid_pressure();

  return torch::where(solid, fill_solid, w);
}

torch::Tensor InternalBoundaryImpl::forward(torch::Tensor wlr, int dim,
                                            torch::Tensor solid) {
  if (!solid.defined()) return wlr;

  using Index::ILT;
  using Index::IRT;
  using Index::IVX;
  using Index::IVY;
  using Index::IVZ;

  auto solidl = solid;
  auto solidr = solid.roll(1, dim - 1);
  solidr.select(dim - 1, 0) = solidl.select(dim - 1, 0);

  for (size_t n = 0; n < wlr.size(1); ++n) {
    wlr[IRT][n] = torch::where(solidl, wlr[ILT][n], wlr[IRT][n]);
    wlr[ILT][n] = torch::where(solidr, wlr[IRT][n], wlr[ILT][n]);
  }

  if (dim == 3) {
    wlr[IRT][IVX] = torch::where(solidl, -wlr[ILT][IVX], wlr[IRT][IVX]);
    wlr[ILT][IVX] = torch::where(solidr, -wlr[IRT][IVX], wlr[ILT][IVX]);
  } else if (dim == 2) {
    wlr[IRT][IVY] = torch::where(solidl, -wlr[ILT][IVY], wlr[IRT][IVY]);
    wlr[ILT][IVY] = torch::where(solidr, -wlr[IRT][IVY], wlr[ILT][IVY]);
  } else if (dim == 1) {
    wlr[IRT][IVZ] = torch::where(solidl, -wlr[ILT][IVZ], wlr[IRT][IVZ]);
    wlr[ILT][IVZ] = torch::where(solidr, -wlr[IRT][IVZ], wlr[ILT][IVZ]);
  }

  return wlr;
}

}  // namespace snap
