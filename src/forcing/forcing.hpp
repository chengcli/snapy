#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/input/parameter_input.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {
struct ConstGravityOptions {
  ConstGravityOptions() = default;
  explicit ConstGravityOptions(ParameterInput pin);

  ADD_ARG(double, grav1) = 0.;
  ADD_ARG(double, grav2) = 0.;
  ADD_ARG(double, grav3) = 0.;
};

struct CoriolisOptions {
  CoriolisOptions() = default;
  explicit CoriolisOptions(ParameterInput pin);

  ADD_ARG(double, omega1) = 0.;
  ADD_ARG(double, omega2) = 0.;
  ADD_ARG(double, omega3) = 0.;

  ADD_ARG(double, omegax) = 0.;
  ADD_ARG(double, omegay) = 0.;
  ADD_ARG(double, omegaz) = 0.;

  ADD_ARG(CoordinateOptions, coord);
};

struct DiffusionOptions {
  DiffusionOptions() = default;
  explicit DiffusionOptions(ParameterInput pin);

  ADD_ARG(double, K) = 0.;
  ADD_ARG(std::string, type) = "theta";
};

class ConstGravityImpl : public torch::nn::Cloneable<ConstGravityImpl> {
 public:
  //! options with which this `ConstGravity` was constructed
  ConstGravityOptions options;

  // Constructor to initialize the layers
  ConstGravityImpl() = default;
  explicit ConstGravityImpl(ConstGravityOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, double dt);
};
TORCH_MODULE(ConstGravity);

class Coriolis123Impl : public torch::nn::Cloneable<Coriolis123Impl> {
 public:
  //! options with which this `Coriolis123` was constructed
  CoriolisOptions options;

  // Constructor to initialize the layers
  Coriolis123Impl() = default;
  explicit Coriolis123Impl(CoriolisOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, double dt);
};
TORCH_MODULE(Coriolis123);

class CoriolisXYZImpl : public torch::nn::Cloneable<CoriolisXYZImpl> {
 public:
  //! options with which this `CoriolisXYZ` was constructed
  CoriolisOptions options;

  //! submodules
  Coordinate pcoord;

  // Constructor to initialize the layers
  CoriolisXYZImpl() = default;
  explicit CoriolisXYZImpl(CoriolisOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, double dt);
};
TORCH_MODULE(CoriolisXYZ);

class DiffusionImpl : public torch::nn::Cloneable<DiffusionImpl> {
 public:
  //! options with which this `Diffusion` was constructed
  DiffusionOptions options;

  // Constructor to initialize the layers
  DiffusionImpl() = default;
  explicit DiffusionImpl(DiffusionOptions const& options_) : options(options_) {
    reset();
  }
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, double dt);
};
}  // namespace snap

#undef ADD_ARG
