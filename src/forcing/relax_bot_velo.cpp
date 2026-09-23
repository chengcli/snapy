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
  // Applies at the physical lower x1 boundary only: under an x1-decomposed
  // layout (nb1 > 1) a rank whose lower face is an internal block interface
  // must not force there.
  if (!phydro->pmb->options->is_physical_boundary(0, 0, -1)) return du;

  auto bottom = phydro->pmb->part(
      {0, 0, -1}, PartOptions().exterior(false).depth(1).ndim(3));
  auto rho = w[IDN].index(bottom);
  auto scale = dt / options->tau() * rho;

  // The force is built from CONTRAVARIANT primitive velocities, but `du` holds
  // COVARIANT momenta, so it must be lowered before it is added -- otherwise
  // the drag is not antiparallel to v and slowly ROTATES the horizontal wind
  // instead of braking it. Same conversion as the Coriolis force (#168).
  // `cosine_cell_kj` is zero on an orthogonal grid, so the lowering is the
  // identity there.
  auto force = torch::stack({scale * (options->bvx() - w[IVX].index(bottom)),
                             scale * (options->bvy() - w[IVY].index(bottom)),
                             scale * (options->bvz() - w[IVZ].index(bottom))});
  // expand_as is a view, so this costs nothing; it gives cth the slab's own
  // shape, which `bottom` can then index like any other field.
  auto cth =
      phydro->pmb->pcoord->cosine_cell_kj.expand_as(w[IDN]).index(bottom);
  coord_vec_lower_(force, cth);
  du[IVX].index(bottom) += force[VEL1];
  du[IVY].index(bottom) += force[VEL2];
  du[IVZ].index(bottom) += force[VEL3];
  return du;
}

}  // namespace snap
