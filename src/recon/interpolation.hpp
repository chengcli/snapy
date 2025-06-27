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

  ADD_ARG(std::string, type) = "dc";
  ADD_ARG(bool, scale) = false;
};

class InterpImpl {
 public:
  //! options with which this `Interp` was constructed
  InterpOptions options;

  virtual std::string print_name() const { return "Unknown"; }
  virtual int stencils() const { return 1; }
  virtual torch::Tensor forward(torch::Tensor w, int dim) {
    throw std::runtime_error("Not implemented");
  }

 protected:
  //! Disable constructor
  InterpImpl() = default;
  explicit InterpImpl(InterpOptions const& options) : options(options) {}

 private:
  std::string name_() const { return "snap::InterpImpl"; }
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
  std::string print_name() const override { return name(); }
  torch::Tensor forward(torch::Tensor w, int dim) override {
    auto vec = w.sizes().vec();
    vec.insert(vec.begin(), 2);
    return w.squeeze(-1).expand(vec);
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
  std::string print_name() const override { return name(); }

  torch::Tensor forward(torch::Tensor w, int dim) override;

  int stencils() const override { return 3; }

  void left(torch::Tensor w, int dim, torch::Tensor out) {
    out = forward(w, dim)[Index::ILT];
  }

  void right(torch::Tensor w, int dim, torch::Tensor out) {
    out = forward(w, dim)[Index::IRT];
  }
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
  std::string print_name() const override { return name(); }

  torch::Tensor forward(torch::Tensor w, int dim) override;

  int stencils() const override { return 5; }

  void left(torch::Tensor w, int dim, torch::Tensor out) {
    out = forward(w, dim)[Index::ILT];
  }

  void right(torch::Tensor w, int dim, torch::Tensor out) {
    out = forward(w, dim)[Index::IRT];
  }
};
TORCH_MODULE(PPMInterp);

class Center3InterpImpl : public torch::nn::Cloneable<Center3InterpImpl>,
                          public InterpImpl {
 public:
  //! data
  torch::Tensor cm, cp;

  //! Constructor to initialize the layer
  Center3InterpImpl() = default;
  explicit Center3InterpImpl(InterpOptions const& options_)
      : InterpImpl(options_) {
    reset();
  }
  void reset() override;
  std::string print_name() const override { return name(); }

  torch::Tensor forward(torch::Tensor w, int dim) override;

  int stencils() const override { return 3; }

  void left(torch::Tensor w, int dim, torch::Tensor out) const;
  void right(torch::Tensor w, int dim, torch::Tensor out) const;
};
TORCH_MODULE(Center3Interp);

class Weno3InterpImpl : public torch::nn::Cloneable<Weno3InterpImpl>,
                        public InterpImpl {
 public:
  //! data
  torch::Tensor c1m, c2m, c3m, c4m;
  torch::Tensor c1p, c2p, c3p, c4p;

  //! Constructor to initialize the layer
  Weno3InterpImpl() = default;
  explicit Weno3InterpImpl(InterpOptions const& options_)
      : InterpImpl(options_) {
    reset();
  }
  void reset() override;
  std::string print_name() const override { return name(); }

  torch::Tensor forward(torch::Tensor w, int dim) override;

  int stencils() const override { return 3; }

  void left(torch::Tensor w, int dim, torch::Tensor out) const;
  void right(torch::Tensor w, int dim, torch::Tensor out) const;
  torch::Tensor right_fallback(torch::Tensor w, int dim) const;
};
TORCH_MODULE(Weno3Interp);

class Center5InterpImpl : public torch::nn::Cloneable<Center5InterpImpl>,
                          public InterpImpl {
 public:
  //! data
  torch::Tensor cm, cp;

  //! Constructor to initialize the layer
  Center5InterpImpl() = default;
  explicit Center5InterpImpl(InterpOptions const& options_)
      : InterpImpl(options_) {
    reset();
  }
  void reset() override;
  std::string print_name() const override { return name(); }

  torch::Tensor forward(torch::Tensor w, int dim) override;

  int stencils() const override { return 5; }

  void left(torch::Tensor w, int dim, torch::Tensor out) const;
  torch::Tensor left_fallback(torch::Tensor w, int dim) const {
    return w.unfold(dim, stencils(), 1).matmul(cm);
  }

  void right(torch::Tensor w, int dim, torch::Tensor out) const;
  torch::Tensor right_fallback(torch::Tensor w, int dim) const {
    return w.unfold(dim, stencils(), 1).matmul(cp);
  }
};
TORCH_MODULE(Center5Interp);

class Weno5InterpImpl : public torch::nn::Cloneable<Weno5InterpImpl>,
                        public InterpImpl {
 public:
  //! data
  torch::Tensor c1m, c2m, c3m, c4m, c5m, c6m, c7m, c8m, c9m;
  torch::Tensor c1p, c2p, c3p, c4p, c5p, c6p, c7p, c8p, c9p;

  //! Constructor to initialize the layer
  Weno5InterpImpl() = default;
  explicit Weno5InterpImpl(InterpOptions const& options_)
      : InterpImpl(options_) {
    reset();
  }
  void reset() override;
  std::string print_name() const override { return name(); }

  torch::Tensor forward(torch::Tensor w, int dim) override;

  int stencils() const override { return 5; }

  void left(torch::Tensor w, int dim, torch::Tensor out) const;
  torch::Tensor left_fallback(torch::Tensor w, int dim) const;

  void right(torch::Tensor w, int dim, torch::Tensor out) const;
  torch::Tensor right_fallback(torch::Tensor w, int dim) const;
};
TORCH_MODULE(Weno5Interp);
}  // namespace snap

#undef ADD_ARG
