#pragma once

#include <torch/types.h>

#include <array>
#include <memory>

namespace snap {

struct CutCellStencil;

// Static, single-valued embedded-boundary metrics. Arrays follow logical
// grid order [x3,x2,x1]. Centroid and area-vector components are physical XYZ.
struct EBGeometry {
  torch::Tensor volume_fraction, cell_centroid;
  std::array<torch::Tensor, 3> face_fraction, face_centroid;
  torch::Tensor wall_area_vector, wall_centroid, flags;
  // Mapped fields use physical XYZ coordinates. The face vector is the
  // integral of the unit normal oriented toward increasing r, alpha or beta.
  int panel = -1;
  torch::Tensor background_volume;
  std::array<torch::Tensor, 3> background_face_area, face_area_vector;

  static EBGeometry from_cubed_sphere_level_set(const torch::Tensor& phi,
                                                const torch::Tensor& x1f,
                                                const torch::Tensor& x2f,
                                                const torch::Tensor& x3f,
                                                int panel);
  static EBGeometry from_cubed_sphere_metrics(
      const torch::Tensor& volume_fraction, const torch::Tensor& cell_centroid,
      const std::array<torch::Tensor, 3>& face_fraction,
      const std::array<torch::Tensor, 3>& face_centroid,
      const std::array<torch::Tensor, 3>& face_area_vector,
      const torch::Tensor& wall_area_vector, const torch::Tensor& wall_centroid,
      const torch::Tensor& x1f, const torch::Tensor& x2f,
      const torch::Tensor& x3f, int panel);

  // Geometry-only reconstruction weights shared by hydro and scalar stages.
  mutable std::shared_ptr<CutCellStencil> reconstruction_stencil;

  // Positive vertex values are fluid. The zero isosurface is reconstructed
  // linearly on a conforming tetrahedral subdivision of each Cartesian cell.
  static EBGeometry from_level_set(const torch::Tensor& phi,
                                   const torch::Tensor& x1f,
                                   const torch::Tensor& x2f,
                                   const torch::Tensor& x3f);

  // Physical centroids and outward wall area vectors. Each geometric face
  // has one shared entry, including the two ends of a coordinate axis.
  static EBGeometry from_metrics(
      const torch::Tensor& volume_fraction, const torch::Tensor& cell_centroid,
      const std::array<torch::Tensor, 3>& face_fraction,
      const std::array<torch::Tensor, 3>& face_centroid,
      const torch::Tensor& wall_area_vector, const torch::Tensor& wall_centroid,
      const torch::Tensor& x1f, const torch::Tensor& x2f,
      const torch::Tensor& x3f);

  EBGeometry to(torch::Device device) const;
};

}  // namespace snap
