// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
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

  TORCH_CHECK(op->tau() > 0.,
              "BotSpongeLyrOptions: tau must be greater than zero.");
  TORCH_CHECK(op->width() > 0.,
              "BotSpongeLyrOptions: width must be greater than zero.");

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
  // Applies at the physical lower x1 boundary only: under an x1-decomposed
  // layout (nb1 > 1) a rank whose lower face is an internal block interface
  // must not force there.
  if (!phydro->pmb->options->is_physical_boundary(0, 0, -1)) return du;

  auto pcoord = phydro->pmb->pcoord;
  int il = pcoord->il();
  int iu = pcoord->iu();

  auto x1min = pcoord->x1f[il];
  auto eta = (options->width() - (pcoord->x1f.slice(0, 0, -1) - x1min)) /
             options->width();
  eta.clamp_(0., 1.0);
  auto scale = torch::sin(M_PI / 2. * eta).pow(2).unsqueeze(0).unsqueeze(0);

  // The force is built from CONTRAVARIANT primitive velocities, but `du` holds
  // COVARIANT momenta, so it must be lowered before it is added -- otherwise
  // the drag is not antiparallel to v and slowly ROTATES the horizontal wind
  // instead of braking it. Same conversion as the Coriolis force (#168).
  // `cosine_cell_kj` is zero on an orthogonal grid, so the lowering is the
  // identity there; operand order is preserved so that case stays
  // bit-identical.
  auto force = w[IDN] * w.narrow(0, IVX, 3) / options->tau() * scale * dt;
  coord_vec_lower_(force, pcoord->cosine_cell_kj);
  du.narrow(0, IVX, 3) -= force;

  return du;
}

}  // namespace snap
