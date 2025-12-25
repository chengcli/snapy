#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}

namespace snap {

struct CoordinateOptionsImpl;
using CoordinateOptions = std::shared_ptr<CoordinateOptionsImpl>;

struct ConstGravityOptionsImpl;
using ConstGravityOptions = std::shared_ptr<ConstGravityOptionsImpl>;

//! Calculate the hydrostatic pressure field.
/*!
 * \param w hydro primitive variables
 * \param grav gravitational acceleration (positive)
 * \param dz cell sizes
 * \param ie index of the top cell
 */
torch::Tensor calc_hydrostatic_pressure(torch::Tensor w, double grav,
                                        torch::Tensor dz, int is, int ie);

//! Calculate the non-hydrostatic pressure field.
/*!
 * \param pres total pressure
 * \param psf hydrostatic pressure at cell interface
 * \param margin threshold for the pressure difference
 */
torch::Tensor calc_nonhydrostatic_pressure(torch::Tensor pres,
                                           torch::Tensor psf,
                                           double margin = 1.e-6);

struct PrimitiveProjectorOptionsImpl {
  static std::shared_ptr<PrimitiveProjectorOptionsImpl> create() {
    return std::make_shared<PrimitiveProjectorOptionsImpl>();
  }
  static std::shared_ptr<PrimitiveProjectorOptionsImpl> from_yaml(
      std::string const &filename, bool verbose = false);
  static std::shared_ptr<PrimitiveProjectorOptionsImpl> from_yaml(
      YAML::Node const &node);

  PrimitiveProjectorOptionsImpl() = default;
  void report(std::ostream &os) const {
    os << "-- primitive projector options --\n";
    os << "* type = " << type() << "\n"
       << "* pressure-margin = " << margin() << "\n";
  }

  //! choose from ["none", "temperature"]
  ADD_ARG(std::string, type) = "none";
  ADD_ARG(double, margin) = 1.e-6;

  //! submodule options
  ADD_ARG(ConstGravityOptions, grav) = nullptr;
  ADD_ARG(CoordinateOptions, coord) = nullptr;
};
using PrimitiveProjectorOptions =
    std::shared_ptr<PrimitiveProjectorOptionsImpl>;

class HydroImpl;

class PrimitiveProjectorImpl
    : public torch::nn::Cloneable<PrimitiveProjectorImpl> {
 public:
  //! Create and register a PrimitiveProjector module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the `PrimitiveProjector` module
   * \param[in] p     parent module for registering the created module
   * \param[in] name  name for the created module
   * \return created  `PrimitiveProjector` module
   */
  static std::shared_ptr<PrimitiveProjectorImpl> create(
      PrimitiveProjectorOptions const &opts, torch::nn::Module *p,
      std::string const &name = "proj");

  //! options with which this `PrimitiveProjector` was constructed
  PrimitiveProjectorOptions options;

  //! non-owning reference to parent
  HydroImpl const *phydro = nullptr;

  //! Constructor to initialize the layer
  PrimitiveProjectorImpl() : options(PrimitiveProjectorOptionsImpl::create()) {}
  explicit PrimitiveProjectorImpl(PrimitiveProjectorOptions options_,
                                  torch::nn::Module *p = nullptr);
  void reset() override;

  //! decompose the total pressure into hydrostatic and non-hydrostatic parts
  torch::Tensor forward(torch::Tensor w, torch::Tensor dz);

  void restore_inplace(torch::Tensor wlr);

 private:
  //! cache
  torch::Tensor _psf;
};
TORCH_MODULE(PrimitiveProjector);

}  // namespace snap

#undef ADD_ARG
