#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include "bc.hpp"
#include "bc_func.hpp"

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {

struct InternalBoundaryOptions {
  static constexpr int MAXRUN = 4;

  static InternalBoundaryOptions from_yaml(const YAML::Node &root);
  InternalBoundaryOptions() = default;
  void report(std::ostream &os) const {
    os << "* MAXRUN = " << MAXRUN << "\n"
       << "* nghost = " << nghost() << "\n"
       << "* max_iter = " << max_iter() << "\n"
       << "* solid_density = " << solid_density() << "\n"
       << "* solid_pressure = " << solid_pressure() << "\n";
  }

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
   * \param w       primitive states
   * \param solid   internal solid boundary in [0, 1]
   */
  void mark_prim_solid_(torch::Tensor w, torch::Tensor solid);

  //! Mark the solid cells
  /*!
   * \param u       conserved states
   * \param solid   internal solid boundary in [0, 1]
   */
  void fill_cons_solid_(torch::Tensor u, torch::Tensor solid,
                        torch::Tensor fill);

  //! Rectify the solid cells
  /*!
   * \param solid_in internal solid boundary in [0, 1]
   * \param total_num_flips total number of flips
   * \param bfuncs boundary functions
   * \return rectified internal solid boundary
   */
  torch::Tensor rectify_solid(torch::Tensor solid_in, int &total_num_flips,
                              std::vector<bcfunc_t> const &bfuncs = {});

  //! Revise the left/right states
  /*!
   * \param wlr primitive left/right states
   * \param solid internal solid boundary in [0, 1]
   * \return revised primitive left/right states
   */
  torch::Tensor forward(torch::Tensor wlr, int dim, torch::Tensor solid);
};
TORCH_MODULE(InternalBoundary);

}  // namespace snap

#undef ADD_ARG
