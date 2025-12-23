#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/forcing/forcing.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

struct ImplicitOptionsImpl {
  static std::shared_ptr<ImplicitOptionsImpl> create() {
    return std::make_shared<ImplicitOptionsImpl>();
  }
  static std::shared_ptr<ImplicitOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);
  static std::shared_ptr<ImplicitOptionsImpl> from_yaml(const YAML::Node& node);

  ImplicitOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- implicit hydro options --\n";
    os << "* type = " << type() << "\n"
       << "* scheme = " << scheme() << "\n";
  }

  int size() const {
    if ((scheme() >> 3) & 1) {  // full
      return 5;
    } else {
      return 3;
    }
  }

  ADD_ARG(std::string, type) = "none";
  ADD_ARG(int, scheme) = 0;
};
using ImplicitOptions = std::shared_ptr<ImplicitOptionsImpl>;

class HydroImpl;

class ImplicitHydroImpl : public torch::nn::Cloneable<ImplicitHydroImpl> {
 public:
  //! Create and register a ImplicitHydro module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts      options for creating the `ImplicitHydro` module
   * \param[in] p         parent module for registering the created module
   * \param[in] name      name for the created module
   * \return              created `ImplicitHydro` module
   */
  static std::shared_ptr<ImplicitHydroImpl> create(
      ImplicitOptions const& opts, torch::nn::Module* p,
      std::string const& name = "icorr");

  //! options with which this `ImplicitHydro` was constructed
  ImplicitOptions options;

  //! non-owning pointer to parent
  HydroImpl const* phydro = nullptr;

  //! Constructor to initialize the layer
  ImplicitHydroImpl() : options(ImplicitOptionsImpl::create()) {}
  explicit ImplicitHydroImpl(ImplicitOptions const& options,
                             torch::nn::Module* p = nullptr);
  void reset() override;

  //! corrector for the implicit hydro
  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor gamma,
                        double dt);
};
TORCH_MODULE(ImplicitHydro);

}  // namespace snap

#undef ADD_ARG
