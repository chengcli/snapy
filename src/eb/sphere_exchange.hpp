#pragma once

#include <torch/types.h>

#include <array>
#include <memory>

namespace snap {
class MeshBlockImpl;
class LayoutImpl;

// Owner-cell exchange for mapped embedded boundaries. Each canonical cell
// has one slot, including the three distinct cells around a cube vertex.
class SphereExchange {
 public:
  explicit SphereExchange(const MeshBlockImpl& block);
  torch::Tensor gather(const torch::Tensor& owned) const;
  torch::Tensor take(const torch::Tensor& global) const;
  const torch::Tensor& adjacency() const { return adjacent_; }
  const torch::Tensor& back_faces() const { return back_; }

 private:
  std::shared_ptr<LayoutImpl> layout_;
  std::array<int64_t, 3> shape_;
  std::vector<torch::Tensor> local_indices_;
  torch::Tensor adjacent_, back_;
  int64_t count_;
};
}  // namespace snap
