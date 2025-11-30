// snap
#include "scalar.hpp"

namespace snap {
ScalarImpl::ScalarImpl(const ScalarOptions& options_) : options(options_) {
  reset();
}

void ScalarImpl::reset() {
  pcoord = CoordinateImpl::create(options->coord(), this);
  precon = ReconstructImpl::create(options->recon(), this);
  priemann = RiemannSolverImpl::create(options->riemann(), this);
  pthermo = kintera::ThermoXImpl::create(options->thermo(), this);
  pkinetics = kintera::KineticsImpl::create(options->kinetics(), this);

  // populate buffers
  int nc1 = options->coord()->nc1();
  int nc2 = options->coord()->nc2();
  int nc3 = options->coord()->nc3();

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

std::shared_ptr<ScalarImpl> ScalarImpl::create(ScalarOptions const& opts,
                                               torch::nn::Module* p,
                                               std::string const& name) {
  return p->register_module(name, Scalar(opts));
}

}  // namespace snap
