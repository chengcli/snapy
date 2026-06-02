// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

RelaxBotVeloOptions RelaxBotVeloOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["relax-bot-velo"]) return nullptr;

  auto node = forcing["relax-bot-velo"];
  auto op = RelaxBotVeloOptionsImpl::create();

  op->tau() = node["tau"].as<double>(0.0);
  op->bvx() = node["bvx"].as<double>(0.0);
  op->bvy() = node["bvy"].as<double>(0.0);
  op->bvz() = node["bvz"].as<double>(0.0);

  TORCH_CHECK(op->tau() > 0.,
              "RelaxBotVeloOptions: tau must be greater than zero.");

  return op;
}

RelaxBotVeloImpl::RelaxBotVeloImpl(RelaxBotVeloOptions const& options_,
                                   torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void RelaxBotVeloImpl::reset() {
  TORCH_CHECK(phydro, "[RelaxBotVelo] Parent Hydro is null");
}

torch::Tensor RelaxBotVeloImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  auto bottom = phydro->pmb->part(
      {0, 0, -1}, PartOptions().exterior(false).depth(1).ndim(3));
  auto rho = w[IDN].index(bottom);
  auto scale = dt / options->tau() * rho;
  du[IVX].index(bottom) += scale * (options->bvx() - w[IVX].index(bottom));
  du[IVY].index(bottom) += scale * (options->bvy() - w[IVY].index(bottom));
  du[IVZ].index(bottom) += scale * (options->bvz() - w[IVZ].index(bottom));
  return du;
}

}  // namespace snap
