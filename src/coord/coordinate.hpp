#pragma once

// C/C++
#include <functional>
#include <iosfwd>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/layout/distribute_info.hpp>
#include <snap/mesh/mesh_functions.hpp>

#include "coordgen.hpp"

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}

namespace snap {
using IndexRange = std::vector<torch::indexing::TensorIndex>;

struct CoordinateOptions {
  static CoordinateOptions from_yaml(const YAML::Node &node,
                                     DistributeInfo dist = DistributeInfo());
  CoordinateOptions() = default;
  void report(std::ostream &os) const {
    os << "* type = " << type() << "\n"
       << "* eos_type = " << eos_type() << "\n"
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

  int64_t nc1() const { return nx1() > 1 ? nx1() + 2 * nghost() : 1; }
  int64_t nc2() const { return nx2() > 1 ? nx2() + 2 * nghost() : 1; }
  int64_t nc3() const { return nx3() > 1 ? nx3() + 2 * nghost() : 1; }

  ADD_ARG(std::string, type) = "cartesian";
  ADD_ARG(std::string, eos_type) = "ideal-gas";

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
};

class CoordinateImpl {
 public:
  //! options with which this `Coordinate` was constructed
  CoordinateOptions options;

  //! data
  torch::Tensor x1f, x2f, x3f;
  torch::Tensor x1v, x2v, x3v;
  torch::Tensor dx1f, dx2f, dx3f;

  virtual ~CoordinateImpl() = default;

  int is() const { return options.nx1() > 1 ? options.nghost() : 0; }

  int ie() const {
    return options.nx1() > 1 ? options.nghost() + options.nx1() - 1 : 0;
  }

  int js() const { return options.nx2() > 1 ? options.nghost() : 0; }

  int je() const {
    return options.nx2() > 1 ? options.nghost() + options.nx2() - 1 : 0;
  }

  int ks() const { return options.nx3() > 1 ? options.nghost() : 0; }

  int ke() const {
    return options.nx3() > 1 ? options.nghost() + options.nx3() - 1 : 0;
  }

  void print(std::ostream &stream) const;
  virtual void reset_coordinates(std::vector<MeshGenerator> meshgens);

  //! module methods
  virtual torch::Tensor center_width1() const;
  torch::Tensor center_width1(IndexRange const &r) const {
    auto len = r.size();
    auto all = torch::indexing::Slice();
    return center_width1().index({all, all, r[len - 1]});
  }

  virtual torch::Tensor center_width2() const;
  torch::Tensor center_width2(IndexRange const &r) const {
    auto len = r.size();
    auto all = torch::indexing::Slice();
    return center_width2().index({all, r[len - 2], all});
  }

  virtual torch::Tensor center_width3() const;
  torch::Tensor center_width3(IndexRange const &r) const {
    auto len = r.size();
    auto all = torch::indexing::Slice();
    return center_width3().index({r[len - 3], all, all});
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
  torch::Tensor cell_volume(IndexRange const &r) const {
    auto len = r.size();
    auto all = torch::indexing::Slice();
    return cell_volume().index({r[len - 3], r[len - 2], r[len - 1]});
  }

  virtual torch::Tensor find_cell_index(torch::Tensor const &coords) const;

  virtual std::array<double, 3> vec_from_cartesian(
      std::array<double, 3> vec) const {
    return {vec[0], vec[1], vec[2]};
  }

  virtual void vec_lower_(torch::Tensor &vel) const {}
  virtual void vec_raise_(torch::Tensor &vel) const {}

  virtual void prim2local1_(torch::Tensor &prim) const {}
  virtual void prim2local2_(torch::Tensor &prim) const {}
  virtual void prim2local3_(torch::Tensor &prim) const {}

  virtual void flux2global1_(torch::Tensor &flux) const {}
  virtual void flux2global2_(torch::Tensor &flux) const {}
  virtual void flux2global3_(torch::Tensor &flux) const {}

  //! fluxes -> flux divergence
  virtual torch::Tensor forward(torch::Tensor prim, torch::Tensor flux1,
                                torch::Tensor flux2, torch::Tensor flux3);

 protected:
  //! Disable default constructor
  CoordinateImpl() = default;
  explicit CoordinateImpl(const CoordinateOptions &options_);

 private:
  std::string name_() const { return "snap::CoordinateImpl"; }
};

using Coordinate = std::shared_ptr<CoordinateImpl>;

class CartesianImpl : public torch::nn::Cloneable<CartesianImpl>,
                      public CoordinateImpl {
 public:
  using CoordinateImpl::forward;

  CartesianImpl() = default;
  explicit CartesianImpl(const CoordinateOptions &options_)
      : CoordinateImpl(options_) {
    reset();
  }
  void reset() override;
  void pretty_print(std::ostream &stream) const override {
    stream << "Cartesian coordinate:" << std::endl;
    print(stream);
  }

  void reset_coordinates(std::vector<MeshGenerator> meshgens) override;
};
TORCH_MODULE(Cartesian);

class CylindricalImpl : public torch::nn::Cloneable<CylindricalImpl>,
                        public CoordinateImpl {
 public:
  using CoordinateImpl::forward;

  CylindricalImpl() = default;
  explicit CylindricalImpl(const CoordinateOptions &options_)
      : CoordinateImpl(options_) {
    reset();
  }
  void reset() override {}
  void pretty_print(std::ostream &stream) const override {
    stream << "Cylindrical coordinate:" << std::endl;
    print(stream);
  }
};
TORCH_MODULE(Cylindrical);

class SphericalPolarImpl : public torch::nn::Cloneable<SphericalPolarImpl>,
                           public CoordinateImpl {
 public:
  using CoordinateImpl::forward;

  SphericalPolarImpl() = default;
  explicit SphericalPolarImpl(const CoordinateOptions &options_)
      : CoordinateImpl(options_) {
    reset();
  }
  void reset() override {}
  void pretty_print(std::ostream &stream) const override {
    stream << "SphericalPolar coordinate:" << std::endl;
    print(stream);
  }
};
TORCH_MODULE(SphericalPolar);

class GnomonicEquiangleImpl
    : public torch::nn::Cloneable<GnomonicEquiangleImpl>,
      public CoordinateImpl {
 public:
  // geometry data
  torch::Tensor cosine_cell_kj, sine_cell_kj;
  torch::Tensor cosine_face2_kj, sine_face2_kj;
  torch::Tensor cosine_face3_kj, sine_face3_kj;
  torch::Tensor x_ov_rD_kji, y_ov_rC_kji;
  torch::Tensor dx2f_ang_kj, dx3f_ang_kj;
  torch::Tensor dx2f_ang_face3_kj, dx3f_ang_face2_kj;

  // metric data
  torch::Tensor g11, g22, g33, gi11, gi22, gi33, g12, g13, g23;

  GnomonicEquiangleImpl() = default;
  explicit GnomonicEquiangleImpl(const CoordinateOptions &options_)
      : CoordinateImpl(options_) {
    reset();
  }
  void reset() override;
  void pretty_print(std::ostream &stream) const override {
    stream << "GnomonicEquiangle coordinate:" << std::endl;
    print(stream);
  }

  torch::Tensor face_area1() const override;
  torch::Tensor face_area2() const override;
  torch::Tensor face_area3() const override;
  torch::Tensor cell_volume() const override;

  void vec_lower_(torch::Tensor &vel) const override;
  void vec_raise_(torch::Tensor &vel) const override;

  // no need to ortho-normal z-direction
  void prim2local2_(torch::Tensor &wlr) const override;
  void prim2local3_(torch::Tensor &wlr) const override;

  void flux2global1_(torch::Tensor &flux) const override;
  void flux2global2_(torch::Tensor &flux) const override;
  void flux2global3_(torch::Tensor &flux) const override;

  torch::Tensor forward(torch::Tensor prim, torch::Tensor flux1,
                        torch::Tensor flux2, torch::Tensor flux3) override;

 private:
  void _set_face2_metric() const;
  void _set_face3_metric() const;
};
TORCH_MODULE(GnomonicEquiangle);

IndexRange get_interior(torch::IntArrayRef const &shape, int nghost,
                        int extend_x1 = 0, int extend_x2 = 0,
                        int extend_x3 = 0);
}  // namespace snap

#undef ADD_ARG
