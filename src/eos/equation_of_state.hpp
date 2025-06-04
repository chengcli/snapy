#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// base
#include <configure.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/input/parameter_input.hpp>
#include <snap/registry.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {
struct EquationOfStateOptions {
  EquationOfStateOptions() = default;

  explicit EquationOfStateOptions(ParameterInput pin);
  ADD_ARG(std::string, type) = "ideal_gas";
  ADD_ARG(double, density_floor) = 1.e-6;
  ADD_ARG(double, pressure_floor) = 1.e-3;
  ADD_ARG(bool, limiter) = false;

  //! submodules options
  ADD_ARG(kintera::ThermoOptions, thermo);
  ADD_ARG(CoordinateOptions, coord);
};

class EquationOfStateImpl {
 public:
  //! options with which this `EquationOfState` was constructed
  EquationOfStateOptions options;

  //! submodules
  Coordinate pcoord = nullptr;
  kintera::ThermoY pthermo = nullptr;

  virtual int nhydro() const {
    return 5 + pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size();
  }

  virtual void cons2prim(torch::Tensor prim, torch::Tensor cons) const {}
  virtual void prim2cons(torch::Tensor cons, torch::Tensor prim) const {}

  virtual torch::Tensor sound_speed(torch::Tensor prim) const {
    return torch::zeros_like(prim[0]);
  }

  torch::Tensor forward(torch::Tensor hydro_u,
                        torch::optional<torch::Tensor> out = torch::nullopt) {
    torch::NoGradGuard no_grad;

    if (out.has_value()) {
      cons2prim(out.value(), hydro_u);
      return out.value();
    } else {
      auto prim = torch::empty_like(hydro_u);
      cons2prim(prim, hydro_u);
      return prim;
    }
  }

 protected:
  //! Disable constructor
  EquationOfStateImpl() = default;
  explicit EquationOfStateImpl(EquationOfStateOptions const& options_)
      : options(options_) {}

  //! Implement this function.
  virtual void _apply_conserved_limiter_inplace(torch::Tensor& cons) const;

  //! Implement this function.
  virtual void _apply_primitive_limiter_inplace(torch::Tensor& prim) const;

 private:
  std::string name_() const { return "snap::EquationOfStateImpl"; }
};

using EquationOfState = std::shared_ptr<EquationOfStateImpl>;

class IdealGasImpl : public torch::nn::Cloneable<IdealGasImpl>,
                     public EquationOfStateImpl {
 public:
  // Constructor to initialize the layers
  IdealGasImpl() = default;
  explicit IdealGasImpl(EquationOfStateOptions const& options_)
      : EquationOfStateImpl(options_) {
    reset();
  }
  void reset() override;
  using EquationOfStateImpl::forward;

  void cons2prim(torch::Tensor prim, torch::Tensor cons) const override;
  void cons2prim_fallback(torch::Tensor prim, torch::Tensor cons,
                          torch::Tensor gammad) const;
  void prim2cons(torch::Tensor cons, torch::Tensor prim) const override;
  torch::Tensor sound_speed(torch::Tensor prim) const override;
};
TORCH_MODULE(IdealGas);

class IdealMoistImpl : public torch::nn::Cloneable<IdealMoistImpl>,
                       public EquationOfStateImpl {
 public:
  // Constructor to initialize the layers
  IdealMoistImpl() = default;
  explicit IdealMoistImpl(EquationOfStateOptions const& options_)
      : EquationOfStateImpl(options_) {
    reset();
  }
  void reset() override;
  using EquationOfStateImpl::forward;

  void cons2prim(torch::Tensor prim, torch::Tensor cons) const override;
  void cons2prim_fallback(torch::Tensor prim, torch::Tensor cons,
                          torch::Tensor gammad, torch::Tensor feps,
                          torch::Tensor fsig) const;
  void prim2cons(torch::Tensor cons, torch::Tensor prim) const override;
  torch::Tensor sound_speed(torch::Tensor prim) const override;
};
TORCH_MODULE(IdealMoist);

class ShallowWaterImpl : public torch::nn::Cloneable<ShallowWaterImpl>,
                         public EquationOfStateImpl {
 public:
  // Constructor to initialize the layers
  ShallowWaterImpl() = default;
  ShallowWaterImpl(EquationOfStateOptions const& options_)
      : EquationOfStateImpl(options_) {
    reset();
  }
  void reset() override;
  using EquationOfStateImpl::forward;

  int nhydro() const override { return 4; }
  void cons2prim(torch::Tensor prim, torch::Tensor cons) const override;
  void prim2cons(torch::Tensor cons, torch::Tensor prim) const override;
  torch::Tensor sound_speed(torch::Tensor prim) const override;
};
TORCH_MODULE(ShallowWater);

void apply_primitive_limiter_inplace(torch::Tensor prim);
}  // namespace snap

#undef ADD_ARG
