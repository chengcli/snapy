#pragma once

#include <torch/types.h>

#include <snap/eb/sphere_exchange.hpp>

namespace snap {
struct EBGeometry;
struct SphereStencil {
  SphereExchange exchange;
  const MeshBlockImpl* block;
  std::array<int64_t, 3> lo, hi;
  torch::Tensor centers, neighbors, coeff, near;
  SphereStencil(const MeshBlockImpl& b, const EBGeometry& geometry, int width);
  torch::Tensor owned(torch::Tensor t, int dir = 0, int side = 0) const;
};

class SphereReconstruction {
 public:
  SphereReconstruction(const SphereStencil& stencil, const torch::Tensor& state,
                       bool velocity);
  torch::Tensor faces(const torch::Tensor& regular, int dim) const;
  torch::Tensor wall() const;

 private:
  const SphereStencil& s_;
  torch::Tensor state_, values_, slopes_, low_, high_;
  bool velocity_;
  torch::Tensor evaluate(const torch::Tensor& points, int side) const;
};
}  // namespace snap
