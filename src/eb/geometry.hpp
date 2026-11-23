#pragma once

#include <torch/types.h>

#include <array>
#include <memory>

namespace snap {

struct CutCellStencil;

// Cartesian metrics for a fixed, single-valued embedded boundary. Component
// indices are x1,x2,x3. The trailing spatial dimensions are always z,y,x.
struct EBGeometry {
  torch::Tensor volume_fraction, cell_centroid;
  std::array<torch::Tensor, 3> face_fraction, face_centroid;
  torch::Tensor wall_area_vector, wall_centroid, flags;
  // Geometry-only reconstruction weights shared by hydro and scalar stages.
  mutable std::shared_ptr<CutCellStencil> reconstruction_stencil;

  // Positive vertex values are fluid. The zero isosurface is reconstructed
  // linearly on a conforming tetrahedral subdivision of each Cartesian cell.
  static EBGeometry from_level_set(const torch::Tensor &phi,
                                   const torch::Tensor &x1f,
                                   const torch::Tensor &x2f,
                                   const torch::Tensor &x3f);

  // Physical centroids and outward wall area vectors. Each geometric face
  // has one shared entry, including the two ends of a coordinate axis.
  static EBGeometry from_metrics(
      const torch::Tensor &volume_fraction, const torch::Tensor &cell_centroid,
      const std::array<torch::Tensor, 3> &face_fraction,
      const std::array<torch::Tensor, 3> &face_centroid,
      const torch::Tensor &wall_area_vector, const torch::Tensor &wall_centroid,
      const torch::Tensor &x1f, const torch::Tensor &x2f,
      const torch::Tensor &x3f);

  EBGeometry to(torch::Device device) const;
};

}  // namespace snap
