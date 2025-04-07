// spdlog
#include <configure.h>
#include <spdlog/sinks/basic_file_sink.h>

// base
#include <globals.h>

// snap
#include <snap/index.h>

#include "bc_formatter.hpp"
#include "internal_boundary.hpp"

namespace snap {
InternalBoundaryImpl::InternalBoundaryImpl(InternalBoundaryOptions options_)
    : options(options_) {
  reset();
}

void InternalBoundaryImpl::reset() {
  LOG_INFO(logger, "{} resets with options: {}", name(), options);
}

torch::Tensor InternalBoundaryImpl::mark_solid(
    torch::Tensor w, torch::optional<torch::Tensor> solid) {
  if (!solid.has_value()) return w;

  auto fill_solid = torch::zeros({w.size(0), 1, 1, 1}, w.options());

  fill_solid[index::IDN] = options.solid_density();
  fill_solid[index::IPR] = options.solid_pressure();

  return torch::where(solid.value(), fill_solid, w);
}

torch::Tensor InternalBoundaryImpl::forward(
    torch::Tensor wlr, int dim, torch::optional<torch::Tensor> solid) {
  torch::NoGradGuard no_grad;

  if (!solid.has_value()) return wlr;

  using index::ILT;
  using index::IRT;
  using index::IVX;
  using index::IVY;
  using index::IVZ;

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
