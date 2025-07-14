#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/sedimentation/sedimentation.hpp>

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {

struct ConstGravityOptions {
  static ConstGravityOptions from_yaml(YAML::Node const& node);
  ConstGravityOptions() = default;

  ADD_ARG(double, grav1) = 0.;
  ADD_ARG(double, grav2) = 0.;
  ADD_ARG(double, grav3) = 0.;
};

struct CoriolisOptions {
  static CoriolisOptions from_yaml(YAML::Node const& node);
  CoriolisOptions() = default;

  ADD_ARG(double, omega1) = 0.;
  ADD_ARG(double, omega2) = 0.;
  ADD_ARG(double, omega3) = 0.;

  ADD_ARG(double, omegax) = 0.;
  ADD_ARG(double, omegay) = 0.;
  ADD_ARG(double, omegaz) = 0.;

  ADD_ARG(CoordinateOptions, coord);
};

struct DiffusionOptions {
  static DiffusionOptions from_yaml(YAML::Node const& node);
  DiffusionOptions() = default;

  ADD_ARG(double, K) = 0.;
  ADD_ARG(std::string, type) = "theta";
};

struct FricHeatOptions {
  static FricHeatOptions from_yaml(YAML::Node const& root);
  FricHeatOptions() = default;

  ADD_ARG(double, grav) = 0.;
  ADD_ARG(SedVelOptions, sedvel);
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
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
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
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
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

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
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

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(Diffusion);

class FricHeatImpl : public torch::nn::Cloneable<FricHeatImpl> {
 public:
  //! submodules
  SedVel psedvel = nullptr;

  //! options with which this `FricHeat` was constructed
  FricHeatOptions options;

  // Constructor to initialize the layers
  FricHeatImpl() = default;
  explicit FricHeatImpl(FricHeatOptions const& options_) : options(options_) {
    reset();
  }
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(FricHeat);

}  // namespace snap

#undef ADD_ARG
