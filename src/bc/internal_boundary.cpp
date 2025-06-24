// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "internal_boundary.hpp"

namespace snap {

InternalBoundaryOptions InternalBoundaryOptions::from_yaml(
    const YAML::Node &root) {
  InternalBoundaryOptions op;

  TORCH_CHECK(root["geometry"], "'geometry' section missing");
  TORCH_CHECK(root["geometry"]["cells"], "'geometry/cells' section missing");

  op.nghost() = root["geometry"]["cells"]["nghost"].as<int>(2);

  TORCH_CHECK(root["boundary"], "'boundary' section missing");
  TORCH_CHECK(root["boundary"]["internal"],
              "'boundary/internal' section missing");

  op.max_iter() = root["boundary"]["internal"]["max_iter"].as<int>(5);
  op.solid_density() =
      root["boundary"]["internal"]["solid_density"].as<double>(1.e3);
  op.solid_pressure() =
      root["boundary"]["internal"]["solid_pressure"].as<double>(1.9);

  return op;
}

InternalBoundaryImpl::InternalBoundaryImpl(InternalBoundaryOptions options_)
    : options(options_) {
  reset();
}

void InternalBoundaryImpl::reset() {}

torch::Tensor InternalBoundaryImpl::mark_solid(
    torch::Tensor w, torch::optional<torch::Tensor> solid) {
  if (!solid.has_value()) return w;

  auto fill_solid = torch::zeros({w.size(0), 1, 1, 1}, w.options());

  fill_solid[Index::IDN] = options.solid_density();
  fill_solid[Index::IPR] = options.solid_pressure();

  return torch::where(solid.value(), fill_solid, w);
}

torch::Tensor InternalBoundaryImpl::forward(
    torch::Tensor wlr, int dim, torch::optional<torch::Tensor> solid) {
  torch::NoGradGuard no_grad;

  if (!solid.has_value()) return wlr;

  using Index::ILT;
  using Index::IRT;
  using Index::IVX;
  using Index::IVY;
  using Index::IVZ;

  auto solidl = solid.value();
  auto solidr = solid.value().roll(1, dim - 1);
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
