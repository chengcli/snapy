// kintera
#include <kintera/constants.h>

// snap
#include <snap/forcing/forcing.hpp>
#include <snap/hydro/hydro.hpp>

#include "sedimentation.hpp"

#define sqr(x) ((x) * (x))

namespace snap {
SedVelImpl::SedVelImpl(SedVelOptions const& options_, torch::nn::Module* p)
    : options(options_) {
  psed = dynamic_cast<SedHydroImpl const*>(p);
  reset();
}

void SedVelImpl::reset() {
  TORCH_CHECK(psed, "[SedVel] Parent SedHydro is null");
  TORCH_CHECK(options->radius().size() == options->density().size(),
              "[SedVel] radius and density must have the same size");

  radius = register_buffer("radius",
                           torch::tensor(options->radius(), torch::kFloat64));

  density = register_buffer("density",
                            torch::tensor(options->density(), torch::kFloat64));

  const_vsed = register_buffer(
      "const_vsed", torch::tensor(options->const_vsed(), torch::kFloat64));
}

torch::Tensor SedVelImpl::forward(torch::Tensor dens, torch::Tensor pres,
                                  torch::Tensor temp) const {
  using namespace kintera::constants;

  const auto d = options->a_diameter();
  const auto epsilon_LJ = options->a_epsilon_LJ();
  const auto m = options->a_mass();

  std::vector<int64_t> vec(temp.dim() + 1, 1);
  vec[0] = -1;

  // Dynamic viscosity from Chapman-Enskog theory
  auto reduced_temp = KBoltz * temp / epsilon_LJ;

  auto eta = (5.0 / 16.0) * std::sqrt(m * KBoltz / M_PI) * torch::sqrt(temp) *
             torch::pow(reduced_temp, 0.16) / (d * d * 1.22);

  // Effective molecular mean free path
  auto lambda = eta / pres * torch::sqrt(M_PI * KBoltz * temp / (2.0 * m));

  // Calculate Knudsen number, Kn
  auto Kn = lambda / radius.view(vec);

  // Calculate Cunningham slip factor, beta
  auto beta = 1.0 + Kn * (1.256 + 0.4 * torch::exp(-1.1 / Kn));

  // Calculate vsed
  auto grav = psed->phydro->options->grav()->grav1();
  auto stokes =
      beta / (9.0 * eta) *
      (2.0 * sqr(radius.view(vec)) * grav * (density.view(vec) - dens));

  // a non-zero const_vsed is the whole velocity (prescribed, athena convention)
  auto cvsed = const_vsed.view(vec);
  auto vel = torch::where(cvsed != 0.0, cvsed, stokes);

  return vel.clamp(-options->upper_limit(), options->upper_limit());
}

std::shared_ptr<SedVelImpl> SedVelImpl::create(SedVelOptions const& opts,
                                               torch::nn::Module* p,
                                               std::string const& name) {
  TORCH_CHECK(opts != nullptr, "[SedVel] Options pointer is nullptr");
  TORCH_CHECK(p != nullptr, "[SedVel] Parent module is nullptr");
  return p->register_module(name, SedVel(opts, p));
}

}  // namespace snap
