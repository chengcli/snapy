#pragma once

// C/C++
#include <memory>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/snap.h>

// arg
#include <snap/add_arg.h>

namespace snap {
struct InterpOptions {
  InterpOptions() = default;
  explicit InterpOptions(std::string name) { type(name); }
  void report(std::ostream& os) const {
    os << "* type = " << type() << "\n"
       << "* scale = " << (scale() ? "true" : "false") << "\n";
  }

  ADD_ARG(std::string, type) = "dc";
  ADD_ARG(bool, scale) = false;
};

class InterpImpl {
 public:
  //! options with which this `Interp` was constructed
  InterpOptions options;

  virtual int stencils() const { return 1; }

  virtual std::pair<torch::Tensor, torch::Tensor> forward(
      torch::Tensor w, int dim,
      torch::optional<torch::Tensor> wl = torch::nullopt,
      torch::optional<torch::Tensor> wr = torch::nullopt) {
    auto vec = w.sizes().vec();
    vec[dim] -= stencils() - 1;  // reduce size by stencils - 1

    auto wlv = wl.value_or(torch::empty(vec, w.options()));
    auto wrv = wr.value_or(torch::empty(vec, w.options()));

    left(w, dim, wlv);
    right(w, dim, wrv);

    wl = wlv;
    wr = wrv;
    return std::make_pair(wlv, wrv);
  }

  virtual void left(torch::Tensor w, int dim, torch::Tensor const& out) {
    forward(w, dim, out, torch::nullopt);
  }

  virtual void right(torch::Tensor w, int dim, torch::Tensor const& out) {
    forward(w, dim, torch::nullopt, out);
  }

 protected:
  //! Disable constructor
  InterpImpl() = default;
  explicit InterpImpl(InterpOptions const& options) : options(options) {}
};

using Interp = std::shared_ptr<InterpImpl>;

class DonorCellInterpImpl : public torch::nn::Cloneable<DonorCellInterpImpl>,
                            public InterpImpl {
 public:
  //! Constructor to initialize the layer
  DonorCellInterpImpl() = default;
  explicit DonorCellInterpImpl(InterpOptions const& options_)
      : InterpImpl(options_) {
    reset();
  }
  void reset() override {}
  using InterpImpl::forward;

  void left(torch::Tensor w, int dim, torch::Tensor const& out) override {
    out.copy_(w.slice(dim, 0, w.size(dim)));
  }
  void right(torch::Tensor w, int dim, torch::Tensor const& out) override {
    out.copy_(w.slice(dim, 0, w.size(dim) - 1));
  }
};
TORCH_MODULE(DonorCellInterp);

class PLMInterpImpl : public torch::nn::Cloneable<PLMInterpImpl>,
                      public InterpImpl {
 public:
  //! Constructor to initialize the layer
  PLMInterpImpl() = default;
  explicit PLMInterpImpl(InterpOptions const& options_) : InterpImpl(options_) {
    reset();
  }
  void reset() override {}

  int stencils() const override { return 3; }

  std::pair<torch::Tensor, torch::Tensor> forward(
      torch::Tensor w, int dim,
      torch::optional<torch::Tensor> wl = torch::nullopt,
      torch::optional<torch::Tensor> wr = torch::nullopt) override;
};
TORCH_MODULE(PLMInterp);

//! Colella & Woodward 1984, JCP
class PPMInterpImpl : public torch::nn::Cloneable<PPMInterpImpl>,
                      public InterpImpl {
 public:
  //! Constructor to initialize the layer
  PPMInterpImpl() = default;
  explicit PPMInterpImpl(InterpOptions const& options_) : InterpImpl(options_) {
    reset();
  }
  void reset() override {}

  int stencils() const override { return 5; }

  std::pair<torch::Tensor, torch::Tensor> forward(
      torch::Tensor w, int dim,
      torch::optional<torch::Tensor> wl = torch::nullopt,
      torch::optional<torch::Tensor> wr = torch::nullopt) override;
};
TORCH_MODULE(PPMInterp);

class Center3InterpImpl : public torch::nn::Cloneable<Center3InterpImpl>,
                          public InterpImpl {
 public:
  //! data
  torch::Tensor cm, cp;

  //! Constructor to initialize the layer
  Center3InterpImpl() {
    options.type("cp3");
    reset();
  }

  explicit Center3InterpImpl(InterpOptions const& options_)
      : InterpImpl(options_) {
    reset();
  }
  void reset() override;
  using InterpImpl::forward;

  int stencils() const override { return 3; }

  void left(torch::Tensor w, int dim, torch::Tensor const& out) override;
  void right(torch::Tensor w, int dim, torch::Tensor const& out) override;
};
TORCH_MODULE(Center3Interp);

class Weno3InterpImpl : public torch::nn::Cloneable<Weno3InterpImpl>,
                        public InterpImpl {
 public:
  //! data
  torch::Tensor cm, cp;

  //! Constructor to initialize the layer
  Weno3InterpImpl() {
    options.type("weno3");
    reset();
  }

  explicit Weno3InterpImpl(InterpOptions const& options_)
      : InterpImpl(options_) {
    reset();
  }
  void reset() override;
  using InterpImpl::forward;

  int stencils() const override { return 3; }

  void left(torch::Tensor w, int dim, torch::Tensor const& out) override;
  void right(torch::Tensor w, int dim, torch::Tensor const& out) override;
};
TORCH_MODULE(Weno3Interp);

class Center5InterpImpl : public torch::nn::Cloneable<Center5InterpImpl>,
                          public InterpImpl {
 public:
  //! data
  torch::Tensor cm, cp;

  //! Constructor to initialize the layer
  Center5InterpImpl() {
    options.type("cp5");
    reset();
  }

  explicit Center5InterpImpl(InterpOptions const& options_)
      : InterpImpl(options_) {
    reset();
  }
  void reset() override;
  using InterpImpl::forward;

  int stencils() const override { return 5; }

  void left(torch::Tensor w, int dim, torch::Tensor const& out) override;
  void right(torch::Tensor w, int dim, torch::Tensor const& out) override;
};
TORCH_MODULE(Center5Interp);

class Weno5InterpImpl : public torch::nn::Cloneable<Weno5InterpImpl>,
                        public InterpImpl {
 public:
  //! data
  torch::Tensor cm, cp;

  //! Constructor to initialize the layer
  Weno5InterpImpl() {
    options.type("weno5");
    reset();
  }

  explicit Weno5InterpImpl(InterpOptions const& options_)
      : InterpImpl(options_) {
    reset();
  }
  void reset() override;
  using InterpImpl::forward;

  int stencils() const override { return 5; }

  void left(torch::Tensor w, int dim, torch::Tensor const& out) override;
  void right(torch::Tensor w, int dim, torch::Tensor const& out) override;
};
TORCH_MODULE(Weno5Interp);
}  // namespace snap

#undef ADD_ARG
