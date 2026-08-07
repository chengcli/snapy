// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

TopSpongeLyrOptions TopSpongeLyrOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["top-sponge-lyr"]) return nullptr;

  auto node = forcing["top-sponge-lyr"];
  auto op = TopSpongeLyrOptionsImpl::create();

  op->tau() = node["tau"].as<double>(0.0);
  op->width() = node["width"].as<double>(0.0);

  TORCH_CHECK(op->tau() > 0.,
              "TopSpongeLyrOptions: tau must be greater than zero.");
  TORCH_CHECK(op->width() > 0.,
              "TopSpongeLyrOptions: width must be greater than zero.");

  return op;
}

TopSpongeLyrImpl::TopSpongeLyrImpl(TopSpongeLyrOptions const& options_,
                                   torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void TopSpongeLyrImpl::reset() {
  TORCH_CHECK(phydro, "[TopSpongeLyr] Parent Hydro is null");
}

torch::Tensor TopSpongeLyrImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  // Applies at the physical upper x1 boundary only: under an x1-decomposed
  // layout (nb1 > 1) a rank whose upper face is an internal block interface
  // must not force there.
  if (!phydro->pmb->options->is_physical_boundary(0, 0, 1)) return du;

  auto pcoord = phydro->pmb->pcoord;
  int il = pcoord->il();
  int iu = pcoord->iu();

  auto x1max = pcoord->x1f[iu + 1];
  auto eta = (options->width() - (x1max - pcoord->x1f.slice(0, 0, -1))) /
             options->width();
  eta.clamp_(0., 1.0);
  auto scale = torch::sin(M_PI / 2. * eta).pow(2).unsqueeze(0).unsqueeze(0);

  du[IVX] -= w[IDN] * w[IVX] / options->tau() * scale * dt;
  du[IVY] -= w[IDN] * w[IVY] / options->tau() * scale * dt;
  du[IVZ] -= w[IDN] * w[IVZ] / options->tau() * scale * dt;

  return du;
}

}  // namespace snap
