#include <torch/torch.h>

#include <iostream>

// Fix negatives by local, value‐weighted redistribution,
// but compute the fill‐mean over all 9 cells (including the center).
torch::Tensor fixNegativesWeightedIncludeCenter(const torch::Tensor& input) {
  TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");
  TORCH_CHECK(input.is_floating_point(), "Tensor must be float/double");

  // work on a copy
  torch::Tensor X = input.clone();

  // --- kernels ---
  auto opts = torch::TensorOptions().dtype(X.dtype()).device(X.device());
  // mean‐kernel: 3×3 of ones ⇒ includes center
  torch::Tensor meanKernel = torch::ones({1, 1, 3, 3}, opts);
  const float meanNorm = 9.0f;  // sum of meanKernel
  // dist‐kernel: 3×3 of ones but zero at center ⇒ only neighbors
  torch::Tensor distKernel = torch::ones({1, 1, 3, 3}, opts);
  distKernel[0][0][1][1] = 0.0f;

  // reflection padding layer
  auto pad = torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions(1));

  const double eps = 1e-6;

  // iterate until all entries ≥ 0
  while ((X < 0).any().item<bool>()) {
    // 1) compute 3×3 mean **including** center
    auto X4 = X.unsqueeze(0).unsqueeze(0);  // 1×1×H×W
    auto Xpad = pad->forward(X4);
    auto sum9 = at::conv2d(Xpad, meanKernel);  // 1×1×H×W
    auto mean9 = (sum9 / meanNorm).squeeze();  // H×W

    // 2) excess only at originally negative pixels
    //    D[i,j] = mean9[i,j] - X[i,j]   if X[i,j]<0, else 0
    auto D = torch::where(X < 0, mean9 - X, torch::zeros_like(X));

    // 3) fill negatives
    auto filled = X + D;  // now ≥0 everywhere

    // 4) compute neighbor‐sum of filled (for weighting)
    auto F4 = filled.unsqueeze(0).unsqueeze(0);
    auto Fpad = pad->forward(F4);
    auto neighSum = at::conv2d(Fpad, distKernel).squeeze();  // H×W

    // inv‐sum for normalization
    auto invSum = 1.0 / (neighSum + eps);

    // 5) prepare redistributed map: D[j] gets spread to its 8 neighbors
    //    in proportion to their filled‐value / neighbor‐sum[j]
    auto Dw = (D * invSum).unsqueeze(0).unsqueeze(0);
    auto Dpad = pad->forward(Dw);
    auto pullMap = at::conv2d(Dpad, distKernel).squeeze();  // H×W

    // 6) subtract from filled
    X = filled - (filled * pullMap);
  }

  return X;
}

int main() {
  torch::Tensor input = torch::tensor(
      {{1.0f, -2.0f, 3.0f}, {4.0f, -5.0f, 6.0f}, {7.0f, 8.0f, -9.0f}});

  std::cout << "Before:\n" << input << "\n\n";
  std::cout << "Sum before: " << input.sum().item<float>() << "\n\n";
  auto output = fixNegativesWeightedIncludeCenter(input);
  std::cout << "After:\n" << output << "\n";
  std::cout << "Sum after: " << output.sum().item<float>() << "\n";
  return 0;
}
