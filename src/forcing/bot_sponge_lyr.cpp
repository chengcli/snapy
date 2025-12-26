// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

BotSpongeLyrOptions BotSpongeLyrOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["bot-sponge-lyr"]) return nullptr;

  auto node = forcing["bot-sponge-lyr"];
  auto op = BotSpongeLyrOptionsImpl::create();

  op->tau() = node["tau"].as<double>(0.0);
  op->width() = node["width"].as<double>(0.0);

  return op;
}

BotSpongeLyrImpl::BotSpongeLyrImpl(BotSpongeLyrOptions const& options_,
                                   torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void BotSpongeLyrImpl::reset() {
  TORCH_CHECK(phydro, "[BotSpongeLyr] Parent Hydro is null");
}

torch::Tensor BotSpongeLyrImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  auto pcoord = phydro->pmb->pcoord;
  int il = pcoord->il();
  int iu = pcoord->iu();

  auto x1min = pcoord->x1f[il];
  auto eta = (options->width() - (pcoord->x1f.slice(0, 0, -1) - x1min)) /
             options->width();
  eta.clamp_(0., 1.0);
  auto scale = torch::sin(M_PI / 2. * eta).pow(2).unsqueeze(0).unsqueeze(0);

  du[IVX] -= w[IDN] * w[IVX] / options->tau() * scale * dt;
  du[IVY] -= w[IDN] * w[IVY] / options->tau() * scale * dt;
  du[IVZ] -= w[IDN] * w[IVZ] / options->tau() * scale * dt;

  return du;
}

}  // namespace snap
