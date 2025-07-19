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
#include <snap/scalar/scalar.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

struct MeshBlockOptions {
  static MeshBlockOptions from_yaml(std::string input_file);
  MeshBlockOptions() = default;
  void report(std::ostream& os) const {
    os << "* lx1 = " << lx1() << "\n"
       << "* lx2 = " << lx2() << "\n"
       << "* lx3 = " << lx3() << "\n"
       << "* level = " << level() << "\n"
       << "* gid = " << gid() << "\n";
  }

  //! submodule options
  ADD_ARG(IntegratorOptions, intg);
  ADD_ARG(HydroOptions, hydro);
  ADD_ARG(ScalarOptions, scalar);

  //! boundary functions
  ADD_ARG(std::vector<bcfunc_t>, bfuncs);

  //! meshblock id
  ADD_ARG(int, lx1) = 0;
  ADD_ARG(int, lx2) = 0;
  ADD_ARG(int, lx3) = 0;
  ADD_ARG(int, level) = 0;
  ADD_ARG(int, gid) = 0;
};

class MeshBlockImpl : public torch::nn::Cloneable<MeshBlockImpl> {
 public:
  //! options with which this `MeshBlock` was constructed
  MeshBlockOptions options;

  //! user output variables
  torch::OrderedDict<std::string, torch::Tensor> user_out_var;

  //! submodules
  Integrator pintg = nullptr;
  Hydro phydro = nullptr;
  Scalar pscalar = nullptr;

  std::map<std::string, double> timer;

  //! Constructor to initialize the layers
  MeshBlockImpl() = default;
  explicit MeshBlockImpl(MeshBlockOptions const& options_);
  void reset() override;

  //! \brief return an index tensor for part of the meshblock
  std::vector<torch::indexing::TensorIndex> part(
      std::tuple<int, int, int> offset, bool exterior = true, int extend_x1 = 0,
      int extend_x2 = 0, int extend_x3 = 0) const;

  void initialize(torch::Tensor const& hydro_w,
                  torch::Tensor const& scalar_x = torch::Tensor());

  double max_time_step(torch::Tensor solid = torch::Tensor());

  int forward(double dt, int stage, torch::Tensor solid = torch::Tensor());

  void reset_timer() {
    for (auto& t : timer) {
      t.second = 0.0;
    }
  }

  void report_timer(std::ostream& stream) {
    phydro->report_timer(std::cout);
    for (const auto& t : timer) {
      stream << "meshblock[" << t.first << "] = " << t.second << " miliseconds"
             << std::endl;
    }
    reset_timer();
  }

 private:
  //! stage registers
  torch::Tensor _hydro_u0, _hydro_u1;
  torch::Tensor _scalar_v0, _scalar_v1;
};

TORCH_MODULE(MeshBlock);
}  // namespace snap

#undef ADD_ARG
