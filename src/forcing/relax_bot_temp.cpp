// C/C++
#include <atomic>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

RelaxBotTempOptions RelaxBotTempOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["relax-bot-temp"]) return nullptr;

  auto node = forcing["relax-bot-temp"];
  auto op = RelaxBotTempOptionsImpl::create();

  op->tau() = node["tau"].as<double>(0.0);

  TORCH_CHECK(node["btemp"],
              "RelaxBotTempOptions: btemp is required (no default).");
  op->btemp() = node["btemp"].as<double>();
  op->at_face() = node["at-face"].as<bool>(false);

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
  // first interior cell CENTRE, half a cell above it, leaves the face itself
  // off the prescribed value and over-forces a stratified column, whose
  // centre deficit exceeds its face deficit. With `at-face: true`, relax the
  // extrapolated face temperature
  //   T_face = 1.5*T0 - 0.5*T1
  // instead of T0. Only cell 0 is nudged and d(T_face)/d(T0) = 1.5, so the
  // gain is divided by 1.5, keeping the damping coefficient on T0 unchanged.
  // A relaxation is kept rather than a Dirichlet ghost condition: the wall is
  // rigid and no-flux, and pinning T there would imply a conductive flux the
  // equations do not carry.
  auto target = temp_bot;
  double gain = 1.0;
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
    target = 1.5 * T0 - 0.5 * T1;
    gain = 1.0 / 1.5;
    static std::atomic<bool> checked{false};
    if (!checked.exchange(true)) {
      // Assert the index convention once (it costs a device sync): offset 0
      // of the depth-2 slice must be the cell adjacent to the lower face,
      // i.e. the deeper and so not-colder one. `>=`, not `>`: an isothermal
      // column is legal and gives t0 == t1 exactly, where the extrapolation
      // reduces to T0; only a strictly colder deeper cell indicates a flipped
      // convention that would extrapolate the wrong way.
      double t0 = T0.mean().item<double>(), t1 = T1.mean().item<double>();
      TORCH_CHECK(t0 >= t1,
                  "[RelaxBotTemp] index convention violated: offset 0 of the "
                  "lower-boundary slice (T=",
                  t0,
                  ") should be deeper, and so no colder, than offset 1 (T=", t1,
                  "). The face extrapolation would run the wrong way.");
    }
  }
  du[IPR].index(bottom) +=
      gain * dt / options->tau() * rho * cv * (options->btemp() - target);
  return du;
}

}  // namespace snap
