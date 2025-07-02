#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/bc/internal_boundary.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/eos/equation_of_state.hpp>
#include <snap/forcing/forcing.hpp>
#include <snap/implicit/vertical_implicit.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/riemann/riemann_solver.hpp>

#include "primitive_projector.hpp"

// arg
#include <snap/add_arg.h>

namespace snap {

struct HydroOptions {
  static HydroOptions from_yaml(std::string const& filename);
  HydroOptions() = default;

  //! Thermodynamics options
  ADD_ARG(kintera::ThermoOptions, thermo);

  //! forcing options
  ADD_ARG(ConstGravityOptions, grav);
  ADD_ARG(CoriolisOptions, coriolis);
  ADD_ARG(DiffusionOptions, visc);

  //! submodule options
  ADD_ARG(CoordinateOptions, coord);
  ADD_ARG(EquationOfStateOptions, eos);
  ADD_ARG(PrimitiveProjectorOptions, proj);

  ADD_ARG(ReconstructOptions, recon1);
  ADD_ARG(ReconstructOptions, recon23);
  ADD_ARG(RiemannSolverOptions, riemann);

  ADD_ARG(InternalBoundaryOptions, ib);
  ADD_ARG(ImplicitOptions, vic);
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

  InternalBoundary pib = nullptr;
  VerticalImplicit pvic = nullptr;

  std::map<std::string, double> timer;

  //! forcings
  std::vector<torch::nn::AnyModule> forcings;

  //! Constructor to initialize the layers
  HydroImpl() = default;
  explicit HydroImpl(const HydroOptions& options_);
  void reset() override;

  virtual double max_time_step(torch::Tensor hydro_w,
                               torch::Tensor solid = torch::Tensor()) const;

  //! Advance the conserved variables by one time step.
  torch::Tensor forward(torch::Tensor hydro_u, double dt,
                        torch::Tensor solid = torch::Tensor());

  void fix_negative_dp_inplace(torch::Tensor wlr, torch::Tensor wdc) const;

  void reset_timer() {
    for (auto& t : timer) {
      t.second = 0.0;
    }
  }

  void report_timer(std::ostream& stream) {
    for (const auto& t : timer) {
      stream << "hydro[" << t.first << "] = " << t.second << " miliseconds"
             << std::endl;
    }
    reset_timer();
  }

 private:
  torch::Tensor _flux1, _flux2, _flux3, _div, _vic;
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

#undef ADD_ARG
