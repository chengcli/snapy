#include "refine.hpp"

namespace snap {

torch::Tensor conservative_refine(torch::Tensor x) {
  double scale_x3_y = x.size(-3) > 1 ? 2.0 : 1.0;
  double scale_x2_y = x.size(-2) > 1 ? 2.0 : 1.0;
  double scale_x3_x = x.size(-3) > 1 ? 0.5 : 1.0;
  double scale_x2_x = x.size(-2) > 1 ? 0.5 : 1.0;

  auto opts_y =
      torch::nn::functional::InterpolateFuncOptions()
          .scale_factor(std::vector<double>({scale_x3_y, scale_x2_y, 1.0}))
          .mode(torch::kTrilinear)
          .align_corners(false)
          .recompute_scale_factor(false);

  auto opts_x =
      torch::nn::functional::InterpolateFuncOptions()
          .scale_factor(std::vector<double>({scale_x3_x, scale_x2_x, 1.0}))
          .mode(torch::kArea)
          .recompute_scale_factor(false);

  auto opts_dy =
      torch::nn::functional::InterpolateFuncOptions()
          .scale_factor(std::vector<double>({scale_x3_y, scale_x2_y, 1.0}))
          .mode(torch::kArea)
          .recompute_scale_factor(false);

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
  double scale_x3 = x.size(-3) > 1 ? 0.5 : 1.0;
  double scale_x2 = x.size(-2) > 1 ? 0.5 : 1.0;

  auto opts = torch::nn::functional::InterpolateFuncOptions()
                  .scale_factor(std::vector<double>({scale_x3, scale_x2, 1.0}))
                  .mode(torch::kArea)
                  .recompute_scale_factor(false);
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
