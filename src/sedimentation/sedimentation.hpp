#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/eos/equation_of_state.hpp>
#include <snap/utils/format.hpp>

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {

struct SedVelOptions {
  static SedVelOptions from_yaml(YAML::Node const& config);
  void report(std::ostream& os) const {
    os << "* radius = " << fmt::format("{}", radius()) << "\n"
       << "* density = " << fmt::format("{}", density()) << "\n"
       << "* const_vsed = " << fmt::format("{}", const_vsed()) << "\n"
       << "* grav = " << grav() << "\n"
       << "* a_diameter = " << a_diameter() << "\n"
       << "* a_epsilon_LJ = " << a_epsilon_LJ() << "\n"
       << "* a_mass = " << a_mass() << "\n"
       << "* min_radius = " << min_radius() << "\n"
       << "* upper_limit = " << upper_limit() << "\n";
  }

  //! radius and density of particles
  //! if specified, must be the same size of cloud particles
  ADD_ARG(std::vector<double>, radius) = {};
  ADD_ARG(std::vector<double>, density) = {};

  //! additional constant sedimentation velocity
  ADD_ARG(std::vector<double>, const_vsed) = {};

  ADD_ARG(double, grav) = 0.;

  //! default H2-atmosphere properties
  //! diameter of molecule [m]
  ADD_ARG(double, a_diameter) = 2.827e-10;

  //! Lennard-Jones potential [J]
  ADD_ARG(double, a_epsilon_LJ) = 59.7e-7;

  //! molecular mass of background atmosphere, default to H2 [kg]
  ADD_ARG(double, a_mass) = 3.34e-27;

  //! minimum radius of particles subject to sedimentation [m]
  ADD_ARG(double, min_radius) = 1.e-6;

  //! upper limit of sedimentation velocity [m/s]
  ADD_ARG(double, upper_limit) = 5.e3;
};

struct SedHydroOptions {
  void report(std::ostream& os) const { sedvel().report(os); }
  //! submodules options
  ADD_ARG(EquationOfStateOptions, eos);
  ADD_ARG(SedVelOptions, sedvel);
};

class SedVelImpl : public torch::nn::Cloneable<SedVelImpl> {
 public:
  //! particle radius and density
  //! 1D tensor of number of particles
  //! radius and density must have the same size
  torch::Tensor radius, density;

  //! options with which this `SedVel` was constructed
  SedVelOptions options;

  //! Constructor to initialize the layers
  explicit SedVelImpl(SedVelOptions const& options_) : options(options_) {
    reset();
  }
  void reset() override;

  //! Calculate sedimentation velocites
  /*!
   * \param dens    atmospheric density [kg/m^3]
   * \param pres    atmospheric pressure [Pa]
   * \param temp    atmospheric temperature [K]
   * \return        4D tensor of sedimentation velocities.
   *                The first dimension is the number of particles.
   */
  torch::Tensor forward(torch::Tensor dens, torch::Tensor pres,
                        torch::Tensor temp);
};
TORCH_MODULE(SedVel);

class SedHydroImpl : public torch::nn::Cloneable<SedHydroImpl> {
 public:
  //! cache
  torch::Tensor vsed;

  //! submodules
  EquationOfState peos = nullptr;
  SedVel psedvel = nullptr;

  //! options with which this `SedHydro` was constructed
  SedHydroOptions options;

  //! Constructor to initialize the layers
  explicit SedHydroImpl(SedHydroOptions const& options_) : options(options_) {
    reset();
  }
  void reset() override;

  //! Calculate sedimentation velocites
  /*!
   * \param hydro_w   hydro primitive variables
   * \param out       optional output tensor to store the result
   * \return          4D tensor of sedimentation flux (mass, momentum, energy).
   */
  torch::Tensor forward(torch::Tensor hydro_w,
                        torch::optional<torch::Tensor> out = torch::nullopt);
};
TORCH_MODULE(SedHydro);

}  // namespace snap

#undef ADD_ARG
