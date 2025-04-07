// fvm
#include "riemann_solver.hpp"

#include <fvm/registry.hpp>

namespace snap {
RiemannSolverOptions::RiemannSolverOptions(ParameterInput pin) {
  eos(EquationOfStateOptions(pin));
  coord(CoordinateOptions(pin));
}

RiemannSolverImpl::RiemannSolverImpl(const RiemannSolverOptions& options_)
    : options(options_) {}

torch::Tensor RiemannSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                         int dim, torch::Tensor vel) {
  auto ui = (vel > 0).to(torch::kInt);
  return vel * (ui * wl + (1 - ui) * wr);
}

torch::Tensor UpwindSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                        int dim, torch::Tensor vel) {
  auto ui = (vel > 0).to(torch::kInt);
  return vel * (ui * wl + (1 - ui) * wr);
}

}  // namespace snap
