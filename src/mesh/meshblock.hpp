#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include <harp/integrator/integrator.hpp>

// snap
#include <snap/bc/bc_func.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/distribute_env.hpp>
#include <snap/layout/layout.hpp>
#include <snap/output/output_type.hpp>
#include <snap/scalar/scalar.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

//! \brief  container for parameters to initialize a MeshBlock
/*!
 * This struct holds all the options required to initialize a MeshBlock.
 * It can be initialized from a YAML input file using the `from_yaml` method,
 * or by setting the individual options manually.
 */
struct MeshBlockOptionsImpl {
  static std::shared_ptr<MeshBlockOptionsImpl> create() {
    return std::make_shared<MeshBlockOptionsImpl>();
  }
  static std::shared_ptr<MeshBlockOptionsImpl> from_yaml(
      std::string input_file);

  MeshBlockOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "* verbose = " << verbose() << "\n";
    os << "* basename = " << basename() << "\n";
  }

  //! verbose
  ADD_ARG(bool, verbose) = false;

  //! output
  ADD_ARG(std::string, basename) = "";
  ADD_ARG(std::vector<OutputOptions>, outputs);

  //! submodule options
  ADD_ARG(harp::IntegratorOptions, intg) = nullptr;
  ADD_ARG(HydroOptions, hydro) = nullptr;
  ADD_ARG(ScalarOptions, scalar) = nullptr;

  //! boundary functions
  ADD_ARG(std::vector<bcfunc_t>, bfuncs);

  //! distributed environment
  ADD_ARG(DistributeEnvOptions, dist) = nullptr;
  ADD_ARG(LayoutOptions, layout) = nullptr;
};
using MeshBlockOptions = std::shared_ptr<MeshBlockOptionsImpl>;

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
  harp::Integrator pintg = nullptr;
  Hydro phydro = nullptr;
  Scalar pscalar = nullptr;
  DistributeEnv pdist = nullptr;
  Layout playout = nullptr;

  //! Constructor to initialize the layers
  MeshBlockImpl() : options(MeshBlockOptionsImpl::create()) {}
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
   * \return: initial simulation time
   */
  double initialize(Variables& vars);

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

  //! exchange ghost zones
  void exchange(Variables& vars) {
    if (options->layout()->type() == "slab") {
      _slab_exchange(vars);
    } else {
      throw std::invalid_argument("MeshBlock::exchange: layout type " +
                                  options->layout()->type() +
                                  " not implemented");
    }
  }

 protected:
  /*!
   * \brief Initialize send and receive buffers for 2D domain decomposition
   *
   * Allocates torch::Tensor buffers for exchanging ghost zone data with
   * neighboring processes in a 2D slab decomposition. Buffers are sized
   * to match the ghost zone dimensions of the mesh block.
   */
  void _init_buffers_2d(Variables const& vars,
                        std::vector<std::string> const& names);

  //! Serialize function for 2D layout
  void _serialize_2d(Variables const& vars);

  //! Deserialize function for 2D layout
  void _deserialize_2d(Variables& vars) const;

  /*!
   * \brief Perform ghost zone exchange for slab layout
   *
   * Exchanges ghost zone data with neighboring processes using point-to-point
   * communication. This function serializes data, performs send/recv
   * operations, and deserializes received data into ghost zones.
   */
  void _slab_exchange(Variables& vars);

  //! initialize from restart file
  /*!
   * \param vars: variables to initialize
   * \return: simulation time from the restart file
   */
  double _init_from_restart(Variables& vars);

 private:
  //! clock and cycle at time start
  clock_t _time_start;
  int _cycle_start = 0;

  //! stage registers
  torch::Tensor _hydro_u0, _hydro_u1;
  torch::Tensor _scalar_s0, _scalar_s1;

  //! exchange buffers
  /*!
   * The first index indicates the rank
   * The second index indicates the variable group
   */
  std::vector<std::vector<torch::Tensor>> _send_bufs, _recv_bufs;

  //! buffer variable names
  std::vector<std::string> _buf_names;
};

TORCH_MODULE(MeshBlock);
}  // namespace snap

#undef ADD_ARG
