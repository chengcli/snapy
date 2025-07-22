// C/C++
#include <iostream>

// snap
#include "pull_neighbors.hpp"

namespace snap {

torch::Tensor pull_neighbors2(const torch::Tensor& input) {
  TORCH_CHECK(input.dim() == 2, "Input must be 2D");
  TORCH_CHECK(input.is_floating_point(), "Must be float/double");

  auto X = input.clone();
  auto opts = torch::TensorOptions().dtype(X.dtype()).device(X.device());

  // 3×3 mean kernel (includes center): sum=9
  auto meanK = torch::ones({1, 1, 3, 3}, opts);
  const float meanNorm = 9.0f;

  // 3×3 dist kernel (zeros center): sum=8
  auto distK = torch::ones({1, 1, 3, 3}, opts);
  distK[0][0][1][1] = 0.0f;

  // zero‐pad of 1 on all sides: left, right, top, bottom
  auto pad2d = torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions(1));

  const double eps = 1e-10;

  while ((X < 0).any().item<bool>()) {
    // 1) mean over 3×3 (incl. center)
    auto X4 = X.unsqueeze(0).unsqueeze(0);
    auto Xp = pad2d->forward(X4);
    auto sum9 = at::conv2d(Xp, meanK);
    auto m9 = (sum9 / meanNorm).squeeze();

    // 2) excess only where X<0
    auto D = torch::where(X < 0, m9 - X, torch::zeros_like(X));

    // 3) fill negatives
    auto F = X + D;  // now >=0 everywhere

    // 4) neighbor‐sum for weighting (only 8 neighbors)
    auto F4 = F.unsqueeze(0).unsqueeze(0);
    auto Fp = pad2d->forward(F4);
    auto nsum = at::conv2d(Fp, distK).squeeze();
    auto invSum = 1.0 / (nsum + eps);

    // 5) build pull‐map: conv2d of (D * invSum)
    auto Dw = (D * invSum).unsqueeze(0).unsqueeze(0);
    auto Dp = pad2d->forward(Dw);
    auto pull = at::conv2d(Dp, distK).squeeze();

    // 6) subtract weighted pull from filled image
    X = F - (F * pull);
  }

  return X;
}

torch::Tensor pull_neighbors3(const torch::Tensor& input) {
  TORCH_CHECK(input.dim() == 3, "Input must be 3D");
  TORCH_CHECK(input.is_floating_point(), "Must be float/double");

  auto X = input.clone();
  auto opts = torch::TensorOptions().dtype(X.dtype()).device(X.device());

  // 3×3×3 mean kernel (includes center): sum=27
  auto meanK3 = torch::ones({1, 1, 3, 3, 3}, opts);
  const float meanNorm3 = 27.0f;

  // 3×3×3 dist kernel (zero center): sum=26
  auto distK3 = torch::ones({1, 1, 3, 3, 3}, opts);
  distK3[0][0][1][1][1] = 0.0f;

  // zero‐pad 1 on all six faces: {L,R,T,B,F,Bk}
  auto pad3d = torch::nn::ZeroPad3d(torch::nn::ZeroPad3dOptions(1));

  const double eps = 1e-10;

  while ((X < 0).any().item<bool>()) {
    // 1) 3×3×3 mean incl. center
    auto X5 = X.unsqueeze(0).unsqueeze(0);  // 1×1×D×H×W
    auto Xp3 = pad3d->forward(X5);
    auto s27 = at::conv3d(Xp3, meanK3);
    auto m27 = (s27 / meanNorm3).squeeze();

    // 2) excess only at negatives
    auto D = torch::where(X < 0, m27 - X, torch::zeros_like(X));

    // 3) fill negatives
    auto F = X + D;  // >=0 everywhere

    // 4) neighbor‐sum for weighting (26 neighbors)
    auto F5 = F.unsqueeze(0).unsqueeze(0);
    auto Fp3 = pad3d->forward(F5);
    auto nsum3 = at::conv3d(Fp3, distK3).squeeze();
    auto inv3 = 1.0 / (nsum3 + eps);

    // 5) pull‐map: conv3d of (D * inv3)
    auto Dw3 = (D * inv3).unsqueeze(0).unsqueeze(0);
    auto Dp3 = pad3d->forward(Dw3);
    auto pull3 = at::conv3d(Dp3, distK3).squeeze();

    // 6) subtract weighted pull
    X = F - (F * pull3);
  }

  return X;
}

}  // namespace snap
