// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/registry.hpp>

#include "riemann_solver.hpp"

namespace snap {

RiemannSolverOptions RiemannSolverOptions::from_yaml(YAML::Node const& node) {
  RiemannSolverOptions op;

  op.type() = node["type"].as<std::string>("roe");
  op.dir() = node["dir"].as<std::string>("omni");
  return op;
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
