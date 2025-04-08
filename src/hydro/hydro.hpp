#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/add_arg.h>

#include <snap/bc/internal_boundary.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/eos/equation_of_state.hpp>
#include <snap/forcing/forcing.hpp>
#include <snap/implicit/vertical_implicit.hpp>
#include <snap/input/parameter_input.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/riemann/riemann_solver.hpp>

#include "primitive_projector.hpp"

namespace snap {

struct HydroOptions {
  HydroOptions() = default;
  explicit HydroOptions(ParameterInput pin);

  //! number of ghost cells (populated from MeshBlock)
  ADD_ARG(int, nghost) = 1;

  //! forcing options
  ADD_ARG(ConstGravityOptions, grav);
  ADD_ARG(CoriolisOptions, coriolis);
  ADD_ARG(DiffusionOptions, visc);

  //! submodule options
  ADD_ARG(EquationOfStateOptions, eos);
  ADD_ARG(CoordinateOptions, coord);
  ADD_ARG(RiemannSolverOptions, riemann);
  ADD_ARG(PrimitiveProjectorOptions, proj);

  ADD_ARG(ReconstructOptions, recon1);
  ADD_ARG(ReconstructOptions, recon23);

  ADD_ARG(InternalBoundaryOptions, ib);
  ADD_ARG(VerticalImplicitOptions, vic);
};

class HydroImpl : public torch::nn::Cloneable<HydroImpl> {
 public:
  //! options with which this `Hydro` was constructed
  HydroOptions options;

  //! concrete submodules
  EquationOfState peos = nullptr;
  Coordinate pcoord = nullptr;
  RiemannSolver priemann = nullptr;
  PrimitiveProjector pproj = nullptr;

  Reconstruct precon1 = nullptr;
  Reconstruct precon23 = nullptr;
  Reconstruct precon_dc = nullptr;

  InternalBoundary pib = nullptr;
  VerticalImplicit pvic = nullptr;

  //! forcings
  std::vector<torch::nn::AnyModule> forcings;

  //! Constructor to initialize the layers
  HydroImpl() = default;
  explicit HydroImpl(const HydroOptions& options_);
  void reset() override;

  int nvar() const { return peos->nhydro(); }
  virtual double max_time_step(
      torch::Tensor hydro_w,
      torch::optional<torch::Tensor> solid = torch::nullopt) const;

  //! Advance the conserved variables by one time step.
  torch::Tensor forward(torch::Tensor hydro_u, double dt,
                        torch::optional<torch::Tensor> solid = torch::nullopt);

  void fix_negative_dp_inplace(torch::Tensor wlr, torch::Tensor wdc) const;
};

/// A `ModuleHolder` subclass for `HydroImpl`.
/// See the documentation for `HydroImpl` class to learn what methods it
/// provides, and examples of how to use `Hydro` with
/// `torch::nn::HydroOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(Hydro);

void check_recon(torch::Tensor wlr, int nghost, int extend_x1, int extend_x2,
                 int extend_x3);
void check_eos(torch::Tensor w, int nghost);
}  // namespace snap
