#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/eos/equation_of_state.hpp>
#include <snap/forcing/forcing.hpp>
#include <snap/implicit/implicit_hydro.hpp>
#include <snap/layout/layout.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/riemann/riemann_solver.hpp>
#include <snap/sedimentation/sedimentation.hpp>

#include "primitive_projector.hpp"

// arg
#include <snap/add_arg.h>

namespace snap {

class MeshBlockImpl;

struct HydroOptionsImpl {
  static std::shared_ptr<HydroOptionsImpl> create() {
    return std::make_shared<HydroOptionsImpl>();
  }
  static std::shared_ptr<HydroOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);

  HydroOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- hydro options --\n";
    os << "* verbose = " << verbose() << "\n"
       << "* disable_flux_x1 = " << disable_flux_x1() << "\n"
       << "* disable_flux_x2 = " << disable_flux_x2() << "\n"
       << "* disable_flux_x3 = " << disable_flux_x3() << "\n";
  }

  //! verbose
  ADD_ARG(bool, verbose) = false;

  ADD_ARG(bool, disable_flux_x1) = false;
  ADD_ARG(bool, disable_flux_x2) = false;
  ADD_ARG(bool, disable_flux_x3) = false;

  //! forcing options
  ADD_ARG(ConstGravityOptions, grav) = nullptr;
  ADD_ARG(CoriolisOptions, coriolis) = nullptr;
  ADD_ARG(DiffusionOptions, visc) = nullptr;
  ADD_ARG(FricHeatOptions, fricHeat) = nullptr;
  ADD_ARG(BodyHeatOptions, bodyHeat) = nullptr;
  ADD_ARG(BotHeatOptions, botHeat) = nullptr;
  ADD_ARG(TopCoolOptions, topCool) = nullptr;
  ADD_ARG(RelaxBotCompOptions, relaxBotComp) = nullptr;
  ADD_ARG(RelaxBotTempOptions, relaxBotTemp) = nullptr;
  ADD_ARG(RelaxBotVeloOptions, relaxBotVelo) = nullptr;
  ADD_ARG(TopSpongeLyrOptions, topSpongeLyr) = nullptr;
  ADD_ARG(BotSpongeLyrOptions, botSpongeLyr) = nullptr;
  ADD_ARG(PlumeForcingOptions, plumeForcing) = nullptr;

  //! submodule options
  ADD_ARG(EquationOfStateOptions, eos) = nullptr;
  ADD_ARG(PrimitiveProjectorOptions, proj) = nullptr;

  ADD_ARG(ReconstructOptions, recon1) = nullptr;
  ADD_ARG(ReconstructOptions, recon23) = nullptr;
  ADD_ARG(RiemannSolverOptions, riemann) = nullptr;

  ADD_ARG(ImplicitOptions, icorr) = nullptr;
  ADD_ARG(SedHydroOptions, sed) = nullptr;
};

using HydroOptions = std::shared_ptr<HydroOptionsImpl>;
using Variables = std::map<std::string, torch::Tensor>;

class HydroImpl : public torch::nn::Cloneable<HydroImpl> {
 public:
  //! \brief Create and register a `Hydro` module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the `Hydro` module
   * \param[in] p     parent module for registering the created module
   * \param[in] name  name for registering the created module
   * \return          created `Hydro` module
   */
  static std::shared_ptr<HydroImpl> create(HydroOptions const& opts,
                                           torch::nn::Module* p,
                                           std::string const& name = "hydro");

  //! options with which this `Hydro` was constructed
  HydroOptions options;

  //! non-owning reference to parent
  MeshBlockImpl const* pmb = nullptr;

  //! owning submodules
  EquationOfState peos = nullptr;
  RiemannSolver priemann = nullptr;
  PrimitiveProjector pproj = nullptr;

  Reconstruct precon1 = nullptr;
  Reconstruct precon23 = nullptr;

  ImplicitHydro picorr = nullptr;

  SedHydro psed = nullptr;

  //! forcings
  std::vector<torch::nn::AnyModule> forcings;

  //! Constructor to initialize the layers
  HydroImpl() = default;
  explicit HydroImpl(const HydroOptions& options_,
                     torch::nn::Module* p = nullptr);
  void reset() override;

  torch::Tensor max_time_step(torch::Tensor hydro_w,
                              torch::Tensor solid = torch::Tensor()) const;

  //! Advance the conserved variables by one time step.
  torch::Tensor forward(double dt, torch::Tensor hydro_u,
                        Variables const& other);

 private:
  //! Register all forcing modules
  std::vector<std::string> _register_forcings_module();

  torch::Tensor _flux1, _flux2, _flux3, _div, _imp;
};

TORCH_MODULE(Hydro);

/*void check_recon(torch::Tensor wlr, int nghost, int extend_x1, int extend_x2,
                 int extend_x3);
void check_eos(torch::Tensor w, int nghost);*/
}  // namespace snap

#undef ADD_ARG
