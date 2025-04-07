#pragma once

// C/C++
#include <functional>
#include <iosfwd>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// base
#include <add_arg.h>
#include <configure.h>

#include <input/parameter_input.hpp>

// fvm
#include <fvm/index.h>

#include <fvm/mesh/mesh_functions.hpp>

#include "coordgen.hpp"

namespace snap {
using IndexRange = std::vector<torch::indexing::TensorIndex>;

struct CoordinateOptions {
  CoordinateOptions() = default;
  explicit CoordinateOptions(ParameterInput pin);

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
};

class CoordinateImpl {
 public:
  //! options with which this `Coordinate` was constructed
  CoordinateOptions options;

  //! data
  torch::Tensor x1f, x2f, x3f;
  torch::Tensor x1v, x2v, x3v;
  torch::Tensor dx1f, dx2f, dx3f;

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

  virtual torch::Tensor vec_lower(torch::Tensor prim,
                                  int type = kPrimitive) const;
  virtual torch::Tensor vec_raise(torch::Tensor prim,
                                  int type = kPrimitive) const;

  virtual void vec_lower_inplace(torch::Tensor &prim) const {}
  virtual void vec_raise_inplace(torch::Tensor &prim) const {}

  virtual std::array<torch::Tensor, 3> vec_from_cartesian(
      std::array<double, 3> vec) const;

  virtual void prim2local_inplace(torch::Tensor &prim) const {}
  virtual void flux2global_inplace(torch::Tensor &flux) const {}

  //! fluxes -> flux divergence
  virtual torch::Tensor forward(torch::Tensor flux1, torch::Tensor flux2,
                                torch::Tensor flux3);

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

  torch::Tensor forward(torch::Tensor flux1, torch::Tensor flux2,
                        torch::Tensor flux3) override {
    return CoordinateImpl::forward(flux1, flux2, flux3);
  }
};
TORCH_MODULE(Cartesian);

class CylindricalImpl : public torch::nn::Cloneable<CylindricalImpl>,
                        public CoordinateImpl {
 public:
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

  torch::Tensor forward(torch::Tensor flux1, torch::Tensor flux2,
                        torch::Tensor flux3) override {
    return CoordinateImpl::forward(flux1, flux2, flux3);
  }
};
TORCH_MODULE(Cylindrical);

class SphericalPolarImpl : public torch::nn::Cloneable<SphericalPolarImpl>,
                           public CoordinateImpl {
 public:
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

  torch::Tensor forward(torch::Tensor flux1, torch::Tensor flux2,
                        torch::Tensor flux3) override {
    return CoordinateImpl::forward(flux1, flux2, flux3);
  }
};
TORCH_MODULE(SphericalPolar);

class CubedSphereImpl : public torch::nn::Cloneable<CubedSphereImpl>,
                        public CoordinateImpl {
 public:
  CubedSphereImpl() = default;
  explicit CubedSphereImpl(const CoordinateOptions &options_)
      : CoordinateImpl(options_) {
    reset();
  }
  void reset() override {}
  void pretty_print(std::ostream &stream) const override {
    stream << "CubedSphere coordinate:" << std::endl;
    print(stream);
  }

  //! cosine of the angle between the coordinate axis
  torch::Tensor mu1, mu2, mu3;

  torch::Tensor forward(torch::Tensor flux1, torch::Tensor flux2,
                        torch::Tensor flux3) override {
    return CoordinateImpl::forward(flux1, flux2, flux3);
  }
};
TORCH_MODULE(CubedSphere);

IndexRange get_interior(torch::IntArrayRef const &shape, int nghost,
                        int extend_x1 = 0, int extend_x2 = 0,
                        int extend_x3 = 0);
}  // namespace snap
