#pragma once

#include <torch/torch.h>

#include <array>
#include <vector>

namespace snap {
struct EBGeometry;

//! Geometry and least-squares stencils for weighted state redistribution.
//! Spatial tensor axes are (z,y,x); geometry arguments below use (x,y,z).
class WeightedStateRedistribution {
 public:
  WeightedStateRedistribution(const EBGeometry& geometry,
                              const torch::Tensor& background_volume,
                              std::array<bool, 3> periodic = {},
                              std::array<double, 3> periods = {},
                              std::array<int64_t, 3> owned_lo = {0, 0, 0},
                              std::array<int64_t, 3> owned_hi = {-1, -1, -1},
                              torch::Tensor adjacency = {});

  //! Redistribute a provisional state [components,z,y,x] (or [z,y,x]).
  //! Nonparticipants and cells outside the owned box retain their input.
  //! When hydro_rows is supplied, limit reconstructed hydro and tracer states
  //! jointly. EOS floor energies are per unit species mass, dry species first.
  //! bad receives a scalar device flag for an inadmissible mean or fluid cell.
  torch::Tensor apply(
      const torch::Tensor& state, int hydro_rows = 0,
      torch::Tensor* bad = nullptr,
      const torch::Tensor& minimum_specific_internal_energy = {}) const;

 private:
  std::vector<int64_t> shape_;
  torch::Tensor q_cells_, q_weight_, owner_rows_, slope_rows_;
  std::array<torch::Tensor, 3> ls_weight_, slope_offset_, member_offset_;
  torch::Tensor fluid_, targets_, edge_rows_, edge_weight_;
};
}  // namespace snap
