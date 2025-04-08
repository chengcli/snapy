#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/add_arg.h>

#include <snap/recon/reconstruct.hpp>
#include <snap/thermo/thermodynamics.hpp>

namespace snap {

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

struct PrimitiveProjectorOptions {
  PrimitiveProjectorOptions() = default;

  //! choose from ["none", "temperature"]
  ADD_ARG(std::string, type) = "none";
  ADD_ARG(double, margin) = 1.e-6;
  ADD_ARG(int, nghost) = 1;
  ADD_ARG(double, grav) = 0.;
  ADD_ARG(double, Rd) = 287.;

  //! thermodynamic model options
  ADD_ARG(ThermodynamicsOptions, thermo);
};

class PrimitiveProjectorImpl
    : public torch::nn::Cloneable<PrimitiveProjectorImpl> {
 public:
  //! options with which this `PrimitiveProjector` was constructed
  PrimitiveProjectorOptions options;

  //! submodules
  Thermodynamics pthermo = nullptr;

  //! Constructor to initialize the layer
  PrimitiveProjectorImpl() = default;
  explicit PrimitiveProjectorImpl(PrimitiveProjectorOptions options_);
  void reset() override;

  //! decompose the total pressure into hydrostatic and non-hydrostatic parts
  torch::Tensor forward(torch::Tensor w, torch::Tensor dz);

  void restore_inplace(torch::Tensor wlr);
};
TORCH_MODULE(PrimitiveProjector);

}  // namespace snap
