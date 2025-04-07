#include <torch/torch.h>

char const* serialize(const torch::Tensor& tensor) {
  VectorStream vs(tensor.numel() * tensor.element_size());
  std::ostream os(&vsbuf);
  torch::save(tensor, os);
  return vs.buffer();
}

torch::Tensor deserialize(const char* data, size_t size) {
  std::istringstream is(std::string(data, size));
  return torch::load(is);
}
