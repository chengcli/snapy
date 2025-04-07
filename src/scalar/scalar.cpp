// spdlog
#include <configure.h>
#include <spdlog/sinks/basic_file_sink.h>

// base
#include <globals.h>

// fvm
#include <fvm/registry.hpp>

#include "scalar.hpp"
#include "scalar_formatter.hpp"

namespace canoe {
ScalarOptions::ScalarOptions(ParameterInput pin) {
  nscalar(pin->GetOrAddInteger("scalar", "nscalar", 0));

  coord(CoordinateOptions(pin));
  riemann(RiemannSolverOptions(pin));

  recon(ReconstructOptions(pin, "scalar", "xorder"));
}

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

  LOG_INFO(logger, "{} resets with options: {}", name(), options);
}

torch::Tensor ScalarImpl::forward(torch::Tensor u, double dt) {
  // TODO
  return u;
}

}  // namespace canoe
