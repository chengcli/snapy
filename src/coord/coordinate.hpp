#pragma once

// C/C++
#include <functional>
#include <iosfwd>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/mesh/mesh_functions.hpp>

#include "coordgen.hpp"

// arg
#include <snap/add_arg.h>

namespace snap {
using IndexRange = std::vector<torch::indexing::TensorIndex>;

struct EquationOfStateOptionsImpl;
using EquationOfStateOptions = std::shared_ptr<EquationOfStateOptionsImpl>;

class MeshBlockImpl;

struct CoordinateOptionsImpl {
  static std::shared_ptr<CoordinateOptionsImpl> create() {
    return std::make_shared<CoordinateOptionsImpl>();
  }

  static std::shared_ptr<CoordinateOptionsImpl> from_yaml(
      std::string const &filename);

  CoordinateOptionsImpl() = default;
  void report(std::ostream &os) const {
    os << "-- coordinate options --\n";
    os << "* type = " << type() << "\n"
       << "* x1min = " << x1min() << "\n"
       << "* x2min = " << x2min() << "\n"
       << "* x3min = " << x3min() << "\n"
       << "* x1max = " << x1max() << "\n"
       << "* x2max = " << x2max() << "\n"
       << "* x3max = " << x3max() << "\n"
       << "* nx1 = " << nx1() << "\n"
       << "* nx2 = " << nx2() << "\n"
       << "* nx3 = " << nx3() << "\n"
       << "* nghost = " << nghost() << "\n";
  }

  int nc1() const { return nx1() > 1 ? nx1() + 2 * nghost() : 1; }
  int nc2() const { return nx2() > 1 ? nx2() + 2 * nghost() : 1; }
  int nc3() const { return nx3() > 1 ? nx3() + 2 * nghost() : 1; }

  ADD_ARG(std::string, type) = "cartesian";
  ADD_ARG(double, x1min) = 0.;
  ADD_ARG(double, x2min) = 0.;
  ADD_ARG(double, x3min) = 0.;
  ADD_ARG(double, x1max) = 1.;
  ADD_ARG(double, x2max) = 1.;
  ADD_ARG(double, x3max) = 1.;
  ADD_ARG(int, nx1) = 1;
  ADD_ARG(int, nx2) = 1;
  ADD_ARG(int, nx3) = 1;
  ADD_ARG(int, nghost) = 1;
  ADD_ARG(int, interp_order) = 2;
};
using CoordinateOptions = std::shared_ptr<CoordinateOptionsImpl>;

class CoordinateImpl {
 public:
  //! Create and register a `Coordinate` module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the `Coordinate` module
   * \param[in] p     parent module for registering the created module
   * \param[in] name  name for the created module
   * \return          created `Coordinate` module
   */
  static std::shared_ptr<CoordinateImpl> create(
      CoordinateOptions const &opts, torch::nn::Module *p,
      std::string const &name = "coord");

  //! data
  torch::Tensor cosine_cell_kj, cosine_face2_kj, cosine_face3_kj;

  //! options with which this `Coordinate` was constructed
  CoordinateOptions options;

  //! non-owning reference to parent
  MeshBlockImpl const *pmb = nullptr;

  CoordinateImpl() : options(CoordinateOptionsImpl::create()) {}
  explicit CoordinateImpl(const CoordinateOptions &options_,
                          torch::nn::Module *p = nullptr);

  //! data
  torch::Tensor x1f, x2f, x3f;
  torch::Tensor x1v, x2v, x3v;
  torch::Tensor dx1f, dx2f, dx3f;

  virtual ~CoordinateImpl() = default;

  int il() const { return options->nx1() > 1 ? options->nghost() : 0; }

  int iu() const {
    return options->nx1() > 1 ? options->nghost() + options->nx1() - 1 : 0;
  }

  int jl() const { return options->nx2() > 1 ? options->nghost() : 0; }

  int ju() const {
    return options->nx2() > 1 ? options->nghost() + options->nx2() - 1 : 0;
  }

  int kl() const { return options->nx3() > 1 ? options->nghost() : 0; }

  int ku() const {
    return options->nx3() > 1 ? options->nghost() + options->nx3() - 1 : 0;
  }

  void print(std::ostream &stream) const;
  virtual void reset_coordinates(std::array<MeshGenerator, 3> meshgens);

  //! module methods
  virtual torch::Tensor center_width1() const;
  virtual torch::Tensor center_width1(int is, int ie) const {
    return center_width1().slice(2, is, ie);
  }

  virtual torch::Tensor center_width2() const;
  virtual torch::Tensor center_width2(int is, int ie) const {
    return center_width2().slice(1, is, ie);
  }

  virtual torch::Tensor center_width3() const;
  virtual torch::Tensor center_width3(int is, int ie) const {
    return center_width3().slice(0, is, ie);
  }

  virtual torch::Tensor face_area1() const;
  torch::Tensor face_area1(int is, int ie) const {
    return face_area1().slice(2, is, ie);
  }

  virtual torch::Tensor face_area2() const;
  torch::Tensor face_area2(int js, int je) const {
    return face_area2().slice(1, js, je);
  }

  virtual torch::Tensor face_area3() const;
  torch::Tensor face_area3(int ks, int ke) const {
    return face_area3().slice(0, ks, ke);
  }

  virtual torch::Tensor cell_volume() const;

  virtual torch::Tensor find_cell_index(torch::Tensor const &coords) const;

  virtual std::array<double, 3> vec_from_cartesian(
      std::array<double, 3> vec) const {
    return {vec[0], vec[1], vec[2]};
  }

  virtual void interp_ghost(torch::Tensor var,
                            std::tuple<int, int, int> const &) const {}

  //! project contravariant velocity to a local orthogonal frame at face 1
  virtual void prim2local1_(torch::Tensor const &prim) const {}

  //! project contravariant velocity to a local orthogonal frame at face 2
  virtual void prim2local2_(torch::Tensor const &prim) const {}

  //! project contravariant velocity to a local orthogonal frame at face 3
  virtual void prim2local3_(torch::Tensor const &prim) const {}

  //! project fluxes from local orthogonal frame to global contravariant frame
  //! at face 1
  virtual void flux2global1_(torch::Tensor const &flux) const {}

  //! project fluxes from local orthogonal frame to global contravariant frame
  //! at face 2
  virtual void flux2global2_(torch::Tensor const &flux) const {}

  //! project fluxes from local orthogonal frame to global contravariant frame
  //! at face 3
  virtual void flux2global3_(torch::Tensor const &flux) const {}

  //! fluxes -> flux divergence
  virtual torch::Tensor forward(torch::Tensor prim, torch::Tensor flux1,
                                torch::Tensor flux2, torch::Tensor flux3);
};
using Coordinate = std::shared_ptr<CoordinateImpl>;

class CartesianImpl : public torch::nn::Cloneable<CartesianImpl>,
                      public CoordinateImpl {
 public:
  using CoordinateImpl::forward;

  CartesianImpl() = default;
  explicit CartesianImpl(const CoordinateOptions &options_,
                         torch::nn::Module *p = nullptr)
      : CoordinateImpl(options_, p) {
    reset();
  }
  void reset() override;
  void pretty_print(std::ostream &stream) const override {
    stream << "Cartesian coordinate:" << std::endl;
    print(stream);
  }

  void reset_coordinates(std::array<MeshGenerator, 3> meshgens) override;
};
TORCH_MODULE(Cartesian);

class CylindricalImpl : public torch::nn::Cloneable<CylindricalImpl>,
                        public CoordinateImpl {
 public:
  using CoordinateImpl::forward;

  CylindricalImpl() = default;
  explicit CylindricalImpl(const CoordinateOptions &options_,
                           torch::nn::Module *p = nullptr)
      : CoordinateImpl(options_, p) {
    reset();
  }
  void reset() override {}
  void pretty_print(std::ostream &stream) const override {
    stream << "Cylindrical coordinate:" << std::endl;
    print(stream);
  }
};
TORCH_MODULE(Cylindrical);

}  // namespace snap

#undef ADD_ARG
