// C/C++
#include <iostream>

// snap
#include <torch/torch.h>

#include <iostream>
#include <snap/utils/pull_neighbors.hpp>

// Batched 3D: fix negatives with local, value‑weighted redistribution,
// including the center in the 3×3×3 mean, zero‑padding, per‑volume.
torch::Tensor fixNegativesWeighted4D(const torch::Tensor& input) {
  TORCH_CHECK(input.dim() == 4, "Expected 4D input [B,D,H,W]");
  TORCH_CHECK(input.is_floating_point(), "Must be float/double tensor");

  // Work in a [B,1,D,H,W] shape so we can use conv3d
  auto Xc = input.unsqueeze(1).clone();  // [B,1,D,H,W]
  auto opts = input.options();

  // 3×3×3 mean‑kernel (all ones): sum = 27
  auto meanK = torch::ones({1, 1, 3, 3, 3}, opts);
  const float meanNorm = 27.0f;

  // 3×3×3 dist‑kernel (ones but zero at center): sum = 26
  auto distK = torch::ones({1, 1, 3, 3, 3}, opts);
  distK[0][0][1][1][1] = 0.0f;

  // zero‑pad of 1 on all six faces
  auto pad3d = torch::nn::ZeroPad3d(torch::nn::ZeroPad3dOptions(1.));

  const double eps = 1e-6;

  // iterate until no negatives remain anywhere in the batch
  while ((Xc < 0).any().item<bool>()) {
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
    auto F = Xc + D;  // now ≥0 everywhere

    // 4) sum of 26 neighbors for weighting
    auto Fp = pad3d->forward(F);  // [B,1,D+2,H+2,W+2]
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
    Xc = F - (F * pull);  // [B,1,D,H,W]
  }

  // drop the channel dim and return shape [B,D,H,W]
  return Xc.squeeze(1);
}

// ---------------------------------------
// Example usage
// ---------------------------------------
int main() {
  // 2D example
  torch::Tensor img2d = torch::tensor(
      {{1.0f, -2.0f, 3.0f}, {4.0f, -5.0f, 6.0f}, {7.0f, 8.0f, -9.0f}});
  std::cout << "2D before:\n" << img2d << "\n\n";
  std::cout << "sum before: " << img2d.sum().item<float>() << "\n\n";
  auto out2 = snap::pull_neighbors2(img2d);
  std::cout << "2D after:\n" << out2 << "\n\n";
  std::cout << "sum after: " << out2.sum().item<float>() << "\n\n";

  // 3D example (3×3×3)
  torch::Tensor img3d = torch::tensor(
      {{{1.0f, -2.0f, 3.0f}, {4.0f, -5.0f, 6.0f}, {-7.0f, 8.0f, -9.0f}},
       {{-1.0f, 2.0f, -3.0f}, {4.0f, -5.0f, 6.0f}, {-7.0f, 8.0f, 9.0f}},
       {{1.0f, -2.0f, 3.0f}, {4.0f, -5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}}});
  std::cout << "3D before:\n" << img3d << "\n";
  std::cout << "sum before: " << img3d.sum().item<float>() << "\n";
  auto out3 = snap::pull_neighbors3(img3d);
  std::cout << "3D after:\n" << out3 << "\n";
  std::cout << "sum after: " << out3.sum().item<float>() << "\n\n";

  // Example: batch of 2 volumes (3×3×3)
  torch::Tensor batch3d = torch::tensor(
      {{{{1.0f, -2.0f, 3.0f}, {4.0f, -5.0f, 6.0f}, {-7.0f, 8.0f, -9.0f}},
        {{-1.0f, 2.0f, -3.0f}, {4.0f, -5.0f, 6.0f}, {-7.0f, 8.0f, -9.0f}},
        {{1.0f, -2.0f, 3.0f}, {4.0f, -5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}}},
       {{{-1.0f, 2.0f, -3.0f}, {4.0f, -5.0f, 6.0f}, {-7.0f, 8.0f, -9.0f}},
        {{1.0f, -2.0f, 3.0f}, {4.0f, -5.0f, 6.0f}, {-7.0f, 8.0f, -9.0f}},
        {{-1.0f, 2.0f, -3.0f},
         {4.0f, -5.0f, 6.0f},
         {7.0f, 8.0f, 9.0f}}}});  // shape [2,3,3,3]

  std::cout << "Before:\n" << batch3d << "\n\n";
  std::cout << "Sum before: " << batch3d[0].sum().item<float>() << "\n\n";
  std::cout << "Sum before: " << batch3d[1].sum().item<float>() << "\n\n";
  auto out = fixNegativesWeighted4D(batch3d);
  std::cout << "After:\n" << out << "\n";
  std::cout << "Sum after: " << out[0].sum().item<float>() << "\n";
  std::cout << "Sum after: " << out[1].sum().item<float>() << "\n";

  return 0;
}
