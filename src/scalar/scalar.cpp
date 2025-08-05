// base
#include <configure.h>

// snap
#include <snap/registry.hpp>

#include "scalar.hpp"
#include "scalar_formatter.hpp"

namespace snap {
ScalarImpl::ScalarImpl(const ScalarOptions& options_) : options(options_) {
  reset();
}

void ScalarImpl::reset() {
  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());

  // set up reconstruction model
  precon = register_module("recon", Reconstruct(options.recon()));

  // set up riemann-solver model
  priemann = register_module_op(this, "riemann", options.riemann());

  // set up thermodynamics model
  pthermo = register_module("thermo", kintera::ThermoX(options.thermo()));

  // set up kinetics model
  pkinetics =
      register_module("kinetics", kintera::Kinetics(options.kinetics()));

  // populate buffers
  int nc1 = options.coord().nc1();
  int nc2 = options.coord().nc2();
  int nc3 = options.coord().nc3();

  _X = register_buffer("X",
                       torch::empty({nvar(), nc3, nc2, nc1}, torch::kFloat64));

  _V = register_buffer("V",
                       torch::empty({nvar(), nc3, nc2, nc1}, torch::kFloat64));
}

torch::Tensor ScalarImpl::forward(double dt, torch::Tensor u,
                                  Variables const& other) {
  // TODO
  return u;
}

}  // namespace snap
