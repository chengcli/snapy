// C/C++
#include <iostream>

// snap
#include "pull_neighbors.hpp"

namespace snap {

std::vector<int64_t> unravel_index(int64_t flat_index, at::IntArrayRef shape) {
  std::vector<int64_t> indices(shape.size());
  for (int64_t i = shape.size() - 1; i >= 0; --i) {
    int64_t size = shape[i];
    indices[i] = flat_index % size;
    flat_index /= size;
  }
  return indices;
}

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
  int max_iter = 5;

  while ((X < 0).any().item<bool>() && max_iter-- > 0) {
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

  TORCH_CHECK(
      max_iter > 0,
      "pull_neighbors2: Exceeded maximum iterations without convergence.");

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
  int max_iter = 5;

  while ((X < 0).any().item<bool>() && max_iter-- > 0) {
    // 1) 3×3×3 mean incl. center
    auto X5 = X.unsqueeze(0).unsqueeze(0);  // 1×1×D×H×W
    auto Xp3 = pad3d->forward(X5);
    auto s27 = at::conv3d(Xp3, meanK3);
    auto m27 = (s27 / meanNorm3).squeeze();

    // 2) excess only at negatives
    auto D = torch::where(X < 0, m27 - X, torch::zeros_like(X));

    // 3) fill negatives
    auto F1 = X + D;  // >=0 everywhere
    auto F2 = torch::where(X < 0, torch::zeros_like(X),
                           F1);  // zero weight for original negs

    // 4) neighbor‐sum for weighting (26 neighbors)
    auto F5 = F2.unsqueeze(0).unsqueeze(0);
    auto Fp3 = pad3d->forward(F5);
    auto nsum3 = at::conv3d(Fp3, distK3).squeeze();
    auto inv3 = 1.0 / (nsum3 + eps);

    // 5) pull‐map: conv3d of (D * inv3)
    auto Dw3 = (D * inv3).unsqueeze(0).unsqueeze(0);
    auto Dp3 = pad3d->forward(Dw3);
    auto pull3 = at::conv3d(Dp3, distK3).squeeze();

    // 6) subtract weighted pull
    X = F1 - (F2 * pull3);
  }

  TORCH_CHECK(
      max_iter > 0,
      "pull_neighbors3: Exceeded maximum iterations without convergence.");

  return X;
}

// Batched 3D: fix negatives with local, value‑weighted redistribution,
// including the center in the 3×3×3 mean, zero‑padding, per‑volume.
torch::Tensor pull_neighbors4(const torch::Tensor& input) {
  TORCH_CHECK(input.dim() == 4, "Expected 4D input [B,D,H,W]");
  TORCH_CHECK(input.is_floating_point(), "Must be float/double tensor");

  // singular case
  if (input.size(0) <= 0) {
    return input;  // return empty tensor
  }

  // Work in a [B,1,D,H,W] shape so we can use conv3d
  auto Xc = input.clone().unsqueeze(1);  // [B,1,D,H,W]
  auto opts = input.options();

  // 3×3×3 mean‑kernel (all ones): sum = 27
  auto meanK = torch::ones({1, 1, 3, 3, 3}, opts);
  const float meanNorm = 27.0f;

  // 3×3×3 dist‑kernel (ones but zero at center): sum = 26
  auto distK = torch::ones({1, 1, 3, 3, 3}, opts);
  distK[0][0][1][1][1] = 0.0f;

  // zero‑pad of 1 on all six faces
  auto pad3d = torch::nn::ZeroPad3d(torch::nn::ZeroPad3dOptions(1.));

  const double eps = 1e-10;
  int max_iter = 10;

  // iterate until no negatives remain anywhere in the batch
  while ((Xc < 0).any().item<bool>() && max_iter-- > 0) {
    std::cout << "pull_neighbors4: Iteration " << (10 - max_iter) << std::endl;
    std::cout << "xc min = " << Xc.min().item<double>()
              << ", max = " << Xc.max().item<double>() << std::endl;

    // find out the location of Xc minimum
    auto min_flat_index = torch::argmin(Xc).item<int64_t>();
    auto coords = unravel_index(min_flat_index, Xc.sizes());
    std::cout << "coord = " << coords << std::endl;

    // 1) compute 3×3×3 mean including center
    auto Xp = pad3d->forward(Xc);  // [B,1,D+2,H+2,W+2]
    auto sum27 = at::conv3d(Xp, meanK,
                            /*bias=*/c10::nullopt,
                            /*stride=*/{1, 1, 1},
                            /*padding=*/{0, 0, 0},
                            /*dilation=*/{1, 1, 1},
                            /*groups=*/1);  // [B,1,D,H,W]
    auto m27 = sum27 / meanNorm;            // [B,1,D,H,W]

    // 2) excess only at originally negative voxels
    auto zero = torch::zeros_like(Xc);
    auto D = torch::where(Xc < 0, m27 - Xc, zero);  // [B,1,D,H,W]

    // 3) fill negatives
    auto F1 = Xc + D;                          // now ≥0 everywhere
    auto F2 = torch::where(Xc < 0, zero, F1);  // zero weight for original negs

    // 4) sum of 26 neighbors for weighting
    auto Fp = pad3d->forward(F2);  // [B,1,D+2,H+2,W+2]
    auto nsum = at::conv3d(Fp, distK,
                           /*bias=*/c10::nullopt,
                           /*stride=*/{1, 1, 1},
                           /*padding=*/{0, 0, 0},
                           /*dilation=*/{1, 1, 1},
                           /*groups=*/1)
                    .squeeze(1);    // [B,D,H,W]
    auto inv = 1.0 / (nsum + eps);  // [B,D,H,W]

    // 5) build pull‑map: conv3d of (D * inv) over same dist‑kernel
    //    note: expand inv back into channel dim
    auto Dw = (D.squeeze(1) * inv).unsqueeze(1);  // [B,1,D,H,W]
    auto Dp = pad3d->forward(Dw);
    auto pull = at::conv3d(Dp, distK,
                           /*bias=*/c10::nullopt,
                           /*stride=*/{1, 1, 1},
                           /*padding=*/{0, 0, 0},
                           /*dilation=*/{1, 1, 1},
                           /*groups=*/1);  // [B,1,D,H,W]

    // 6) subtract weighted pull from filled
    Xc = F1 - (F2 * pull);  // [B,1,D,H,W]
  }

  TORCH_CHECK(
      max_iter > 0,
      "pull_neighbors4: Exceeded maximum iterations without convergence.");

  // drop the channel dim and return shape [B,D,H,W]
  return Xc.squeeze(1);
}

}  // namespace snap
