#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// kintera
#include <kintera/utils/format.hpp>

// snap
#include <snap/eos/equation_of_state.hpp>

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {

struct SedVelOptions {
  static SedVelOptions from_yaml(YAML::Node const& config);
  void report(std::ostream& os) const {
    os << "* particle_ids = " << fmt::format("{}", particle_ids()) << "\n"
       << "* radius = " << fmt::format("{}", radius()) << "\n"
       << "* density = " << fmt::format("{}", density()) << "\n"
       << "* const_vsed = " << fmt::format("{}", const_vsed()) << "\n"
       << "* grav = " << grav() << "\n"
       << "* a_diameter = " << a_diameter() << "\n"
       << "* a_epsilon_LJ = " << a_epsilon_LJ() << "\n"
       << "* a_mass = " << a_mass() << "\n"
       << "* upper_limit = " << upper_limit() << "\n";
  }
  //! \return species names
  std::vector<std::string> species() const;

  //! id of precipitating particles
  ADD_ARG(std::vector<int>, particle_ids) = {};

  //! radius and density of particles
  //! if specified, must be the same size of cloud particles in thermo
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

  //! upper limit of sedimentation velocity [m/s]
  ADD_ARG(double, upper_limit) = 5.e3;
};

struct SedHydroOptions {
  void report(std::ostream& os) const {
    os << "* hydro_ids = " << fmt::format("{}", hydro_ids()) << "\n";
    sedvel().report(os);
  }

  //! id of precipitating particles in hydro
  ADD_ARG(std::vector<int>, hydro_ids) = {};

  //! submodules options
  ADD_ARG(EquationOfStateOptions, eos);
  ADD_ARG(SedVelOptions, sedvel);
};

class SedVelImpl : public torch::nn::Cloneable<SedVelImpl> {
 public:
  //! particle radius and density
  //! 1D tensor of number of particles
  //! radius and density must have the same size
  torch::Tensor radius, density, const_vsed;

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

  //! particle indices in hydro
  torch::Tensor hydro_ids;

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
   * \param wr        hydro primitive variables at the right interface
   * \param out       optional output tensor to store the result
   * \return          4D tensor of sedimentation flux (mass, momentum, energy).
   */
  torch::Tensor forward(torch::Tensor wr,
                        torch::optional<torch::Tensor> out = torch::nullopt);
};
TORCH_MODULE(SedHydro);

}  // namespace snap

#undef ADD_ARG
