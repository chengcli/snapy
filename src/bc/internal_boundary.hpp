#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// clang-format off
#include <configure.h>
#include <add_arg.h>
// clang-format on

// fvm
#include "boundary_condition.hpp"

namespace canoe {
struct InternalBoundaryOptions {
  static constexpr int MAXRUN = 4;

  InternalBoundaryOptions() = default;

  ADD_ARG(int, nghost) = 1;
  ADD_ARG(int, max_iter) = 5;
  ADD_ARG(double, solid_density) = 1.e3;
  ADD_ARG(double, solid_pressure) = 1.e9;
};

class InternalBoundaryImpl : public torch::nn::Cloneable<InternalBoundaryImpl> {
 public:
  //! options with which this `InternalBoundary` was constructed
  InternalBoundaryOptions options;

  //! Constructor to initialize the layers
  InternalBoundaryImpl() = default;
  explicit InternalBoundaryImpl(InternalBoundaryOptions options);
  void reset() override;

  //! Mark the solid cells
  /*!
   * \param w primitive states
   * \param solid internal solid boundary in [0, 1]
   * \return primitive states with solid cells marked
   */
  torch::Tensor mark_solid(torch::Tensor w,
                           torch::optional<torch::Tensor> solid);

  //! Rectify the solid cells
  /*!
   * \param solid_in internal solid boundary in [0, 1]
   * \param total_num_flips total number of flips
   * \param bfuncs boundary functions
   * \return rectified internal solid boundary
   */
  torch::Tensor rectify_solid(torch::Tensor solid_in, int &total_num_flips,
                              std::vector<bfunc_t> const &bfuncs = {});

  //! Revise the left/right states
  /*!
   * \param wlr primitive left/right states
   * \param solid internal solid boundary in [0, 1]
   * \return revised primitive left/right states
   */
  torch::Tensor forward(torch::Tensor wlr, int dim,
                        torch::optional<torch::Tensor> solid);
};
TORCH_MODULE(InternalBoundary);

}  // namespace canoe
