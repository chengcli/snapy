// torch
#include <torch/torch.h>

namespace snap {

torch::Tensor pull_neighbors2(const torch::Tensor& input);
torch::Tensor pull_neighbors3(const torch::Tensor& input);
torch::Tensor pull_neighbors4(const torch::Tensor& input);

}  // namespace snap
