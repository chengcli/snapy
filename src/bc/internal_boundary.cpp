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

  if (!root["boundary-condition"]) return op;
  if (!root["boundary-condition"]["internal"]) return op;

  auto bc = root["boundary-condition"]["internal"];

  op.max_iter() = bc["max-iter"].as<int>(5);
  op.solid_density() = bc["solid-density"].as<double>(1.e3);
  op.solid_pressure() = bc["solid-pressure"].as<double>(1.e9);

  return op;
}

InternalBoundaryImpl::InternalBoundaryImpl(InternalBoundaryOptions options_)
    : options(options_) {
  reset();
}

void InternalBoundaryImpl::reset() {}

void InternalBoundaryImpl::mark_prim_solid_(torch::Tensor w,
                                            torch::Tensor solid) {
  if (!solid.defined()) return;

  w[IDN].masked_fill_(solid, options.solid_density());
  w[IPR].masked_fill_(solid, options.solid_pressure());
  w.narrow(0, IVX, 3).masked_fill_(solid, 0.);

  int ny = w.size(0) - 5;
  if (ny > 0) {
    w.narrow(0, ICY, ny)
        .masked_fill_(solid.unsqueeze(0).expand_as(w.narrow(0, ICY, ny)), 0.);
  }
}

void InternalBoundaryImpl::fill_cons_solid_(torch::Tensor u,
                                            torch::Tensor solid,
                                            torch::Tensor fill) {
  if (!solid.defined()) return;

  u.set_(torch::where(solid.unsqueeze(0).expand_as(u), fill, u));
}

torch::Tensor InternalBoundaryImpl::forward(torch::Tensor wlr, int dim,
                                            torch::Tensor solid) {
  if (!solid.defined()) return wlr;

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
