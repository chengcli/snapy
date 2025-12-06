#pragma once

// snap
#include "coordinate.hpp"

namespace snap {

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

  void vec_lower_(
      torch::Tensor &vel,
      std::vector<torch::indexing::TensorIndex> const &sub = {}) const override;
  void vec_raise_(
      torch::Tensor &vel,
      std::vector<torch::indexing::TensorIndex> const &sub = {}) const override;

  void contra_to_cart_(
      torch::Tensor vel,
      std::vector<torch::indexing::TensorIndex> const &sub = {}) const override;
  void cart_to_contra_(
      torch::Tensor vel,
      std::vector<torch::indexing::TensorIndex> const &sub = {}) const override;

  void prim2local1_(torch::Tensor &wlr) const override;
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

}  // namespace snap
