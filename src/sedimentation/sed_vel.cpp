
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

  radius = register_buffer("radius",
                           torch::tensor(options.radius(), torch::kFloat64));

  density = register_buffer("density",
                            torch::tensor(options.density(), torch::kFloat64));

  const_vsed = register_buffer(
      "const_vsed", torch::tensor(options.const_vsed(), torch::kFloat64));
}

torch::Tensor SedVelImpl::forward(torch::Tensor dens, torch::Tensor pres,
                                  torch::Tensor temp) {
  const auto d = options.a_diameter();
  const auto epsilon_LJ = options.a_epsilon_LJ();
  const auto m = options.a_mass();

  std::vector<int64_t> vec(temp.dim() + 1, 1);
  vec[0] = -1;

  // cope with float precision
  auto eta = (5.0 / 16.0) * std::sqrt(M_PI * constants::kBoltz) * std::sqrt(m) *
             torch::sqrt(temp) *
             torch::pow(constants::kBoltz / epsilon_LJ * temp, 0.16) /
             (M_PI * d * d * 1.22);

  // Calculate mean free path, lambda
  auto lambda = (eta * std::sqrt(M_PI * sqr(constants::kBoltz))) /
                (pres * std::sqrt(2.0 * m));

  // Calculate Knudsen number, Kn
  auto Kn = lambda / radius.view(vec);

  // Calculate Cunningham slip factor, beta
  auto beta = 1.0 + Kn * (1.256 + 0.4 * torch::exp(-1.1 / Kn));

  // Calculate vsed
  auto vel = beta / (9.0 * eta) *
             (2.0 * sqr(radius.view(vec)) * options.grav() *
              (density.view(vec) - dens));

  // add a constant sedimentation velocity
  vel += const_vsed.view(vec);

  return vel.clamp(-options.upper_limit(), options.upper_limit());
}

}  // namespace snap
