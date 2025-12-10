#pragma once

// snap
#include "coordinate.hpp"

namespace snap {

class SphericalPolarImpl : public torch::nn::Cloneable<SphericalPolarImpl>,
                           public CoordinateImpl {
 public:
  torch::Tensor coord_src1_i, coord_src1_j;
  torch::Tensor coord_src2_i, coord_src2_j;
  torch::Tensor coord_src3_j;

  SphericalPolarImpl() = default;
  explicit SphericalPolarImpl(const CoordinateOptions& options_,
                              torch::nn::Module* p = nullptr)
      : CoordinateImpl(options_, p) {
    reset();
  }
  void reset() override;
  void pretty_print(std::ostream& stream) const override {
    stream << "SphericalPolar coordinate:" << std::endl;
    print(stream);
  }

  torch::Tensor forward(torch::Tensor prim, torch::Tensor flux1,
                        torch::Tensor flux2, torch::Tensor flux3) override;
};
TORCH_MODULE(SphericalPolar);

}  // namespace snap
