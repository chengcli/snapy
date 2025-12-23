// snap
#include "scalar.hpp"

namespace snap {
ScalarImpl::ScalarImpl(const ScalarOptions& options_) : options(options_) {
  reset();
}

void ScalarImpl::reset() {
  if (nvar() > 0) {
    precon = ReconstructImpl::create(options->recon(), this);
    priemann = RiemannSolverImpl::create(options->riemann(), this);
    pthermo = kintera::ThermoXImpl::create(options->thermo(), this);
    pkinetics = kintera::KineticsImpl::create(options->kinetics(), this);
  }
}

torch::Tensor ScalarImpl::forward(double dt, torch::Tensor u,
                                  Variables const& other) {
  // TODO
  return u;
}

std::shared_ptr<ScalarImpl> ScalarImpl::create(ScalarOptions const& opts,
                                               torch::nn::Module* p,
                                               std::string const& name) {
  TORCH_CHECK(p, "[Scalar] Parent module is null");
  TORCH_CHECK(opts, "[Scalar] Options pointer is null");

  return p->register_module(name, Scalar(opts));
}

}  // namespace snap
