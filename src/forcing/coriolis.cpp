// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "forcing.hpp"

namespace snap {

CoriolisOptions CoriolisOptionsImpl::from_yaml(YAML::Node const& forcing) {
  if (!forcing["coriolis"]) return nullptr;

  auto node = forcing["coriolis"];
  auto op = CoriolisOptionsImpl::create();

  op->omega1() = node["omega1"].as<double>(0.);
  op->omega2() = node["omega2"].as<double>(0.);
  op->omega3() = node["omega3"].as<double>(0.);
  op->omegax() = node["omegax"].as<double>(0.);
  op->omegay() = node["omegay"].as<double>(0.);
  op->omegaz() = node["omegaz"].as<double>(0.);

  return op;
}

torch::Tensor Coriolis123Impl::forward(torch::Tensor du, torch::Tensor w,
                                       torch::Tensor temp, double dt) {
  if (options->omega1() != 0.0 || options->omega2() != 0.0 ||
      options->omega3() != 0.0) {
    auto m1 = w[IDN] * w[IVX];
    auto m2 = w[IDN] * w[IVY];
    auto m3 = w[IDN] * w[IVZ];
    du[IVX] += 2. * dt * (options->omega3() * m2 - options->omega2() * m3);
    du[IVY] += 2. * dt * (options->omega1() * m3 - options->omega3() * m1);

    if (w.size(1) > 1) {  // 3d
      du[IVZ] += 2. * dt * (options->omega2() * m1 - options->omega1() * m2);
    }
  }

  return du;
}

void CoriolisXYZImpl::reset() {
  TORCH_CHECK(options->coord() != nullptr,
              "CoordinateOptions not linked in CoriolisXYZ");
  pcoord = CoordinateImpl::create(options->coord(), this);
}

torch::Tensor CoriolisXYZImpl::forward(torch::Tensor du, torch::Tensor w,
                                       torch::Tensor temp, double dt) {
  if (options->omegax() != 0.0 || options->omegay() != 0.0 ||
      options->omegaz() != 0.0) {
    auto [omega1, omega2, omega3] = pcoord->vec_from_cartesian(
        {options->omegax(), options->omegay(), options->omegaz()});

    auto m1 = w[IDN] * w[IVX];
    auto m2 = w[IDN] * w[IVY];
    auto m3 = w[IDN] * w[IVZ];

    du[IVX] += 2. * dt * (omega3 * m2 - omega2 * m3);
    du[IVY] += 2. * dt * (omega1 * m3 - omega3 * m1);
    du[IVZ] += 2. * dt * (omega2 * m1 - omega1 * m2);
  }

  return du;
}
}  // namespace snap
