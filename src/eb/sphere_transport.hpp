#pragma once

#include <torch/types.h>

namespace snap {
class MeshBlockImpl;
class HydroImpl;

// State storage uses covariant momentum. Global Cartesian components here
// are X,Y,Z in the same order as the embedded boundary's physical centroids.
torch::Tensor sphere_momentum_to_cart(const MeshBlockImpl& block,
                                      const torch::Tensor& momentum,
                                      int direction = 0);
torch::Tensor sphere_cart_to_momentum(const MeshBlockImpl& block,
                                      const torch::Tensor& xyz,
                                      int direction = 0);
torch::Tensor sphere_velocity_to_cart(const MeshBlockImpl& block,
                                      const torch::Tensor& velocity,
                                      int direction = 0);
torch::Tensor sphere_cart_to_velocity(const MeshBlockImpl& block,
                                      const torch::Tensor& xyz,
                                      int direction = 0);
void sphere_riemann(HydroImpl& hydro, const torch::Tensor& lr, int dim,
                    torch::Tensor flux, torch::Tensor pressure,
                    torch::Tensor central);
torch::Tensor sphere_radial_wall_traction(HydroImpl& hydro,
                                          const torch::Tensor& w);
torch::Tensor sphere_flux_divergence(HydroImpl& hydro, const torch::Tensor& w,
                                     std::array<torch::Tensor, 3> fluxes);
}  // namespace snap
