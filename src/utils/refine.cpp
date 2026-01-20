#include "refine.hpp"

namespace snap {

torch::Tensor conservative_refine(torch::Tensor x) {
  auto opts_y = torch::nn::functional::InterpolateFuncOptions()
                    .scale_factor(std::vector<double>({2.0, 2.0, 1.0}))
                    .mode(torch::kTrilinear)
                    .align_corners(false);

  auto opts_x = torch::nn::functional::InterpolateFuncOptions()
                    .scale_factor(std::vector<double>({0.5, 0.5, 1.0}))
                    .mode(torch::kArea);

  auto opts_dy = torch::nn::functional::InterpolateFuncOptions()
                     .scale_factor(std::vector<double>({2.0, 2.0, 1.0}))
                     .mode(torch::kArea);

  // bilinear refine
  int dim = 0;
  while (x.dim() < 5) {
    ++dim;
    x = x.unsqueeze(0);
  }
  auto y1 = torch::nn::functional::interpolate(x, opts_y);

  // conservative coarsen
  auto x1 = torch::nn::functional::interpolate(y1, opts_x);

  // conservative correction
  auto dy = torch::nn::functional::interpolate(x - x1, opts_dy);
  auto y = y1 + dy;

  for (int i = 0; i < dim; ++i) y = y.squeeze(0);
  return y;
}

torch::Tensor conservative_coarsen(torch::Tensor x) {
  auto opts = torch::nn::functional::InterpolateFuncOptions()
                  .scale_factor(std::vector<double>({0.5, 0.5, 1.0}))
                  .mode(torch::kArea);
  int dim = 0;
  while (x.dim() < 5) {
    ++dim;
    x = x.unsqueeze(0);
  }

  auto y = torch::nn::functional::interpolate(x, opts);

  for (int i = 0; i < dim; ++i) y = y.squeeze(0);
  return y;
}

}  // namespace snap
