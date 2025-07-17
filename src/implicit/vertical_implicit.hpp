#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/recon/reconstruct.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

struct ImplicitOptions {
  static ImplicitOptions from_yaml(const YAML::Node& root);
  ImplicitOptions() = default;
  void report(std::ostream& os) const {
    os << "* type = " << type() << "\n"
       << "* nghost = " << nghost() << "\n"
       << "* grav = " << grav() << "\n"
       << "* scheme = " << scheme() << "\n";
  }

  ADD_ARG(std::string, type) = "vic";
  ADD_ARG(int, nghost) = 1;
  ADD_ARG(double, grav) = 0.;
  ADD_ARG(int, scheme) = 0;

  //! submodules options
  ADD_ARG(ReconstructOptions, recon);
  ADD_ARG(CoordinateOptions, coord);
};

class VerticalImplicitImpl : public torch::nn::Cloneable<VerticalImplicitImpl> {
 public:
  //! options with which this `VerticalImplicit` was constructed
  ImplicitOptions options;

  //! submodules
  Reconstruct precon = nullptr;
  Coordinate pcoord = nullptr;

  //! Constructor to initialize the layer
  VerticalImplicitImpl() = default;
  explicit VerticalImplicitImpl(ImplicitOptions options);
  void reset() override;

  torch::Tensor diffusion_matrix(torch::Tensor w, torch::Tensor gm1);

  //! Forward function
  torch::Tensor forward(torch::Tensor w, torch::Tensor du, torch::Tensor gm1,
                        double dt);
};
TORCH_MODULE(VerticalImplicit);

//! Roe average scheme
/*
 * Flux in the interface between i-th and i+1-th cells:
 * A(i+1/2) = [sqrt(rho(i))*A(i) + sqrt(rho(i+1))*A(i+1)]/(sqrt(rho(i)) +
 * sqrt(rho(i+1)))
 */
torch::Tensor roe_average(torch::Tensor wlr, torch::Tensor gm1);

//! flux derivative
/*
 * Input variables are density, velocity field and energy.
 * The primitives of cell (n,i)
 */
torch::Tensor flux_jacobian(torch::Tensor w, torch::Tensor gm1);

std::pair<torch::Tensor, torch::Tensor> eigen_vectors(torch::Tensor prim,
                                                      torch::Tensor gm1,
                                                      torch::Tensor cs);
}  // namespace snap

#undef ADD_ARG
