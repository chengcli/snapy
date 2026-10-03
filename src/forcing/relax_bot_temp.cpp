// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

RelaxBotTempOptions RelaxBotTempOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["relax-bot-temp"]) return nullptr;

  auto node = forcing["relax-bot-temp"];
  check_keys(node, "forcing/relax-bot-temp", {"tau", "btemp", "at-face"});
  auto op = RelaxBotTempOptionsImpl::create();

  op->tau() = node["tau"].as<double>(0.0);

  TORCH_CHECK(node["btemp"],
              "RelaxBotTempOptions: btemp is required (no default).");
  op->btemp() = node["btemp"].as<double>();
  if (node["at-face"]) {
    auto const face = node["at-face"];
    TORCH_CHECK(face.IsScalar(),
                "RelaxBotTempOptions: at-face must be true or false.");
    auto const text = face.Scalar();
    TORCH_CHECK(text == "true" || text == "false",
                "RelaxBotTempOptions: at-face must be true or false, got '",
                text, "'.");
    op->at_face() = text == "true";
  }

  TORCH_CHECK(op->tau() > 0.,
              "RelaxBotTempOptions: tau must be greater than zero.");
  TORCH_CHECK(op->btemp() > 0.,
              "RelaxBotTempOptions: btemp must be greater than zero.");

  return op;
}

RelaxBotTempImpl::RelaxBotTempImpl(RelaxBotTempOptions const& options_,
                                   torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void RelaxBotTempImpl::reset() {
  TORCH_CHECK(phydro, "[RelaxBotTemp] Parent Hydro is null");
}

torch::Tensor RelaxBotTempImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  // Applies at the physical lower x1 boundary only: under an x1-decomposed
  // layout (nb1 > 1) a rank whose lower face is an internal block interface
  // must not force there.
  if (!phydro->pmb->options->is_physical_boundary(0, 0, -1)) return du;

  auto bottom = phydro->pmb->part(
      {0, 0, -1}, PartOptions().exterior(false).depth(1).ndim(3));
  auto rho = w[IDN].index(bottom);
  auto temp_bot = temp.index(bottom);
  auto cv = phydro->peos->specific_heat_cv(w, temp).index(bottom);

  // A bottom temperature is usually prescribed at a pressure level, and in a
  // finite-volume grid that level is the domain's lower FACE. Relaxing the
  // first interior cell CENTRE leaves the face itself off the prescribed value
  // and over-forces a stratified column. With `at-face: true`, extrapolate
  // from the first two cell centres using their actual x1 coordinates. Only
  // cell 0 is nudged, so divide the gain by d(T_face)/d(T0).
  // A relaxation is kept rather than a Dirichlet ghost condition: the wall is
  // rigid and no-flux, and pinning T there would imply a conductive flux the
  // equations do not carry.
  auto target = temp_bot;
  torch::Tensor gain;
  if (options->at_face()) {
    auto bottom2 = phydro->pmb->part(
        {0, 0, -1}, PartOptions().exterior(false).depth(2).ndim(3));
    auto t2 = temp.index(bottom2);
    // PartOptions::depth is capped at nghost even when exterior(false)
    // selects INTERIOR cells, so nghost = 1 would silently hand back a
    // width-1 slice. Shape query only -- no device sync.
    TORCH_CHECK(t2.size(-1) >= 2,
                "[RelaxBotTemp] at-face needs two interior cells at the "
                "lower boundary; got ",
                t2.size(-1), ". Set nghost >= 2.");
    auto T0 = t2.narrow(-1, 0, 1);
    auto T1 = t2.narrow(-1, 1, 1);
    int il = phydro->pmb->pcoord->il();
    auto x1v = phydro->pmb->pcoord->x1v;
    auto x1f = phydro->pmb->pcoord->x1f;
    auto a = (x1v[il] - x1f[il]) / (x1v[il + 1] - x1v[il]);
    target = (1. + a) * T0 - a * T1;
    gain = 1.0 / (1. + a);
  }
  auto heating = dt / options->tau() * rho * cv * (options->btemp() - target);
  if (gain.defined()) heating *= gain;
  du[IPR].index(bottom) += heating;
  return du;
}

}  // namespace snap
