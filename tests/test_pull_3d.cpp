// C/C++
#include <iostream>

// snap
#include <torch/torch.h>

#include <iostream>
#include <snap/utils/pull_neighbors.hpp>

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
  auto out = snap::pull_neighbors4(batch3d);
  std::cout << "After:\n" << out << "\n";
  std::cout << "Sum after: " << out[0].sum().item<float>() << "\n";
  std::cout << "Sum after: " << out[1].sum().item<float>() << "\n";

  return 0;
}
