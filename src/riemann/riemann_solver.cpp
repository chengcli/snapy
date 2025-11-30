// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "riemann_solver.hpp"

namespace snap {

RiemannSolverOptions RiemannSolverOptionsImpl::from_yaml(
    YAML::Node const& node) {
  auto op = RiemannSolverOptionsImpl::create();
  op->type() = node["type"].as<std::string>("roe");
  op->dir() = node["dir"].as<std::string>("omni");
  return op;
}

torch::Tensor RiemannSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                         int dim, torch::Tensor vel) {
  auto ui = (vel > 0).to(torch::kInt);
  return vel * (ui * wl + (1 - ui) * wr);
}

RiemannSolver RiemannSolverImpl::create(RiemannSolverOptions const& opts,
                                        torch::nn::Module* p,
                                        std::string const& name) {
  if (opts->type() == "roe") {
    return p->register_module(name, RoeSolver(opts));
  } else if (opts->type() == "lmars") {
    return p->register_module(name, LmarsSolver(opts));
  } else if (opts->type() == "hllc") {
    return p->register_module(name, HLLCSolver(opts));
  } else if (opts->type() == "upwind") {
    return p->register_module(name, UpwindSolver(opts));
  } else if (opts->type() == "shallow-roe") {
    return p->register_module(name, ShallowRoeSolver(opts));
  } else if (opts->type() == "plume-roe") {
    return p->register_module(name, PlumeRoeSolver(opts));
  } else {
    TORCH_CHECK(false, "RiemannSolver: unknown type " + opts->type());
  }
}

}  // namespace snap
