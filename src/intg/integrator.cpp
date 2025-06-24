// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "integrator.hpp"

namespace snap {
void call_average3_cpu(at::TensorIterator& iter, double w1, double w2,
                       double w3);
__attribute__((weak)) void call_average3_cuda(at::TensorIterator& iter,
                                              double w1, double w2, double w3) {
}

IntegratorOptions IntegratorOptions::from_yaml(std::string const& filename) {
  IntegratorOptions op;

  auto config = YAML::LoadFile(filename);
  if (config["integrator"]) {
    auto intg = config["integrator"];
    op.type() = intg["type"].as<std::string>("rk3");
    op.cfl() = intg["dt"].as<double>(0.9);
  }

  return op;
}

IntegratorImpl::IntegratorImpl(IntegratorOptions const& options_)
    : options(options_) {
  if (options.type() == "rk1" || options.type() == "euler") {
    stages.resize(1);
    stages[0].wght0(0.0);
    stages[0].wght1(1.0);
    stages[0].wght2(1.0);
  } else if (options.type() == "rk2") {
    stages.resize(2);
    stages[0].wght0(0.0);
    stages[0].wght1(1.0);
    stages[0].wght2(1.0);

    stages[1].wght0(0.5);
    stages[1].wght1(0.5);
    stages[1].wght2(0.5);
  } else if (options.type() == "rk3") {
    stages.resize(3);
    stages[0].wght0(0.0);
    stages[0].wght1(1.0);
    stages[0].wght2(1.0);

    stages[1].wght0(3. / 4.);
    stages[1].wght1(1. / 4.);
    stages[1].wght2(1. / 4.);

    stages[2].wght0(1. / 3.);
    stages[2].wght1(2. / 3.);
    stages[2].wght2(2. / 3.);
  } else if (options.type() == "rk3s4") {
    stages.resize(4);
    stages[0].wght0(0.5);
    stages[0].wght1(0.5);
    stages[0].wght2(0.5);

    stages[1].wght0(0.0);
    stages[1].wght1(1.0);
    stages[1].wght2(0.5);

    stages[2].wght0(2. / 3.);
    stages[2].wght1(1. / 3.);
    stages[2].wght2(1. / 6.);

    stages[3].wght0(0.);
    stages[3].wght1(1.);
    stages[3].wght2(1. / 2.);
  } else {
    throw std::runtime_error("Integrator not implemented");
  }

  reset();
}

void IntegratorImpl::reset() {}

torch::Tensor IntegratorImpl::forward(int s, torch::Tensor u0, torch::Tensor u1,
                                      torch::Tensor u2) {
  if (s < 0 || s >= stages.size()) {
    throw std::runtime_error("Invalid stage");
  }

  auto out = torch::empty_like(u0);
  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_input(u0)
                  .add_input(u1)
                  .add_input(u2)
                  .build();

  if (u0.is_cpu()) {
    call_average3_cpu(iter, stages[s].wght0(), stages[s].wght1(),
                      stages[s].wght2());
  } else if (u0.is_cuda()) {
    call_average3_cuda(iter, stages[s].wght0(), stages[s].wght1(),
                       stages[s].wght2());
  } else {
    return stages[s].wght0() * u0 + stages[s].wght1() * u1 +
           stages[s].wght2() * u2;
  }

  return out;
}
}  // namespace snap
