#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/bc/bc_func.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/intg/integrator.hpp>
#include <snap/layout/distribute_info.hpp>
#include <snap/scalar/scalar.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

//! defined in output/output_type.hpp
struct OutputOptions;

//! \brief  container for parameters to initialize a MeshBlock
/*!
 * This struct holds all the options required to initialize a MeshBlock.
 * It can be initialized from a YAML input file using the `from_yaml` method,
 * or by setting the individual options manually.
 */
struct MeshBlockOptions {
  static MeshBlockOptions from_yaml(std::string input_file,
                                    DistributeInfo _dist = DistributeInfo());
  MeshBlockOptions() = default;
  void report(std::ostream& os) const {
    os << "* basename = " << basename() << "\n";
  }

  //! output
  ADD_ARG(std::string, basename) = "";
  ADD_ARG(std::vector<OutputOptions>, outputs);

  //! submodule options
  ADD_ARG(IntegratorOptions, intg);
  ADD_ARG(HydroOptions, hydro);
  ADD_ARG(ScalarOptions, scalar);

  //! boundary functions
  ADD_ARG(std::vector<bcfunc_t>, bfuncs);

  //! distributed meshblock info
  ADD_ARG(DistributeInfo, dist);
};

using Variables = std::map<std::string, torch::Tensor>;
class OutputType;

class MeshBlockImpl : public torch::nn::Cloneable<MeshBlockImpl> {
 public:
  //! options with which this `MeshBlock` was constructed
  MeshBlockOptions options;

  //! user output
  std::function<Variables(Variables const&)> user_output_callback;

  //! outputs
  std::vector<std::shared_ptr<OutputType>> output_types;

  //! current cycle number
  int cycle = 0;

  //! submodules
  Integrator pintg = nullptr;
  Hydro phydro = nullptr;
  Scalar pscalar = nullptr;

  //! Constructor to initialize the layers
  MeshBlockImpl() = default;
  explicit MeshBlockImpl(MeshBlockOptions const& options_);
  ~MeshBlockImpl() override;
  void reset() override;

  //! \brief return an index tensor for part of the meshblock
  /*!
   * \param offset: tuple of (x1_offset, x2_offset, x3_offset)
   * \param exterior: if true, return the exterior part (with ghost zones);
   *                  if false, return the interior part (without ghost zones)
   * \param extend_x1: number of cells to extend in the x1 direction
   * \param extend_x2: number of cells to extend in the x2 direction
   * \param extend_x3: number of cells to extend in the x3 direction
   * \return: vector of TensorIndex for each dimension
   */
  std::vector<torch::indexing::TensorIndex> part(
      std::tuple<int, int, int> offset, bool exterior = true, int extend_x1 = 0,
      int extend_x2 = 0, int extend_x3 = 0) const;

  //! initialize the variables
  /*!
   * \param vars: variables to initialize
   */
  Variables& initialize(Variables& vars);

  //! compute the maximum allowable time step
  /*!
   * \param vars: current variables
   * \return: maximum time step
   */
  double max_time_step(Variables const& vars);

  //! advance the variables by one time step
  /*!
   * \param vars: current variables
   * \param dt: time step
   * \param stage: current stage of the integrator
   */
  void forward(Variables& vars, double dt, int stage);

  //! make write outputs at the current time
  /*!
   * \param vars: current variables
   * \param current_time: current simulation time
   * \param final_write: if true, writing outputs as 'final' outputs
   */
  void make_outputs(Variables const& vars, double current_time,
                    bool final_write = false);

  //! print cycle info
  /*!
   * \param vars: current variables
   * \param time: current simulation time
   * \param dt: current time step
   */
  void print_cycle_info(Variables const& vars, double time, double dt) const;

  //! make final output and print diagnostics
  void finalize(Variables const& vars, double time);

  //! check if redo is needed
  /*!
   * \param vars: current variables
   * \return: > 0, redo is needed; 0, no redo; < 0, terminate simulation
   */
  int check_redo(Variables& vars);

 private:
  //! stage registers
  torch::Tensor _hydro_u0, _hydro_u1;
  torch::Tensor _scalar_s0, _scalar_s1;
};

TORCH_MODULE(MeshBlock);
}  // namespace snap

#undef ADD_ARG
