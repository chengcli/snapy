
// snap
#include <snap/constants.h>

#include "sedimentation.hpp"

#define sqr(x) ((x) * (x))

namespace snap {

void SedVelImpl::reset() {
  if (options.radius().size() != options.density().size()) {
    throw std::runtime_error(
        "Sedimentation: radius and density must have the same size");
  }

  radius = register_parameter(
      "radius",
      torch::clamp(torch::tensor(options.radius()), options.min_radius()));

  density = register_parameter(
      "density", torch::clamp(torch::tensor(options.density()), 0.));
}

torch::Tensor SedVelImpl::forward(torch::Tensor dens, torch::Tensor pres,
                                  torch::Tensor temp) {
  const auto d = options.a_diameter();
  const auto epsilon_LJ = options.a_epsilon_LJ();
  const auto m = options.a_mass();

  // cope with float precision
  auto eta = (5.0 / 16.0) * std::sqrt(M_PI * constants::kBoltz) * std::sqrt(m) *
             torch::sqrt(temp) *
             torch::pow(constants::kBoltz / epsilon_LJ * temp, 0.16) /
             (M_PI * d * d * 1.22);

  // Calculate mean free path, lambda
  auto lambda = (eta * std::sqrt(M_PI * sqr(constants::kBoltz))) /
                (pres * std::sqrt(2.0 * m));

  // Calculate Knudsen number, Kn
  auto Kn = lambda / radius.view({-1, 1, 1, 1});

  // Calculate Cunningham slip factor, beta
  auto beta = 1.0 + Kn * (1.256 + 0.4 * torch::exp(-1.1 / Kn));

  // Calculate vsed
  auto vel = beta / (9.0 * eta) *
             (2.0 * sqr(radius.view({-1, 1, 1, 1})) * options.grav() *
              (density.view({-1, 1, 1, 1}) - dens));

  // Set velocity to zero for particles with radius less than min_radius
  for (int i = 0; i < options.radius().size(); ++i) {
    if (options.radius()[i] < options.min_radius()) {
      vel[i] = torch::zeros_like(vel[i]);
    }

    if (options.const_vsed().size() > i) {
      vel[i] += options.const_vsed()[i];
    }
  }

  return torch::clamp(vel, -options.upper_limit(), options.upper_limit());
}

}  // namespace snap
