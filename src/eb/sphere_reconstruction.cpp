#include "sphere_reconstruction.hpp"

#include <snap/eb/sphere_transport.hpp>
#include <snap/mesh/meshblock.hpp>

namespace snap {
using Tensor = torch::Tensor;

Tensor SphereStencil::owned(Tensor t, int dir, int side) const {
  for (int d = 0; d < 3; ++d) {
    int shift = (dir == d + 1) ? side : 0;
    t = t.slice(-1 - d, lo[d] + shift, hi[d] + shift);
  }
  return t;
}

SphereStencil::SphereStencil(const MeshBlockImpl& b, const EBGeometry& g,
                             int width)
    : exchange(b), block(&b) {
  for (int d = 0; d < 3; ++d) {
    int n = g.volume_fraction.size(-1 - d);
    lo[d] = n == 1 ? 0 : b.pcoord->options->nghost();
    hi[d] = n - lo[d];
  }
  std::vector<Tensor> fields{g.volume_fraction};
  for (int k = 0; k < 3; ++k) fields.push_back(g.cell_centroid[k]);
  for (int side = 0; side < 6; ++side) {
    auto f = g.face_fraction[side / 2];
    fields.push_back(f.narrow(-1 - side / 2, side % 2,
                              g.volume_fraction.size(-1 - side / 2)));
  }
  auto data = exchange.gather(owned(torch::stack(fields)));
  centers = data.slice(0, 1, 4);
  auto adj = exchange.adjacency();
  auto ids = torch::arange(adj.size(1), adj.options()).expand_as(adj);
  // Use topology for detecting every ordinary stencil touching the boundary.
  auto connected = torch::where(adj.ge(0), adj, ids);
  near = data[0].ne(1);
  for (int i = 0; i <= width; ++i)
    near =
        near | near.index_select(0, connected.flatten()).reshape_as(adj).any(0);
  neighbors = torch::where((adj >= 0) & data.slice(0, 4).gt(0), adj, ids);
  auto dx = centers.index_select(1, neighbors.flatten()).reshape({3, 6, -1}) -
            centers.unsqueeze(1);
  auto dist2 = dx.square().sum(0, true);
  auto unit = dx / torch::where(dist2 > 0, dist2, 1.);
  auto metric = torch::bmm(unit.permute({2, 0, 1}), dx.permute({2, 1, 0}));
  // Map is static; use double CPU SVD on backends without linalg_pinv.
  auto inverse =
      torch::linalg_pinv(metric.to(torch::kCPU, torch::kFloat64), 1e-11, true)
          .to(metric);
  coeff = torch::bmm(inverse, unit.permute({2, 0, 1}))
              .permute({1, 2, 0})
              .contiguous();
}

SphereReconstruction::SphereReconstruction(const SphereStencil& stencil,
                                           const Tensor& state, bool velocity)
    : s_(stencil), state_(state), velocity_(velocity) {
  if (velocity) {
    state_ = state.clone();
    auto v = state_.narrow(0, IVX, 3);
    v.copy_(sphere_velocity_to_cart(*s_.block, v));
  }
  values_ = s_.exchange.gather(s_.owned(state_));
  auto adjacent = values_.index_select(1, s_.neighbors.flatten())
                      .reshape({state.size(0), 6, -1});
  low_ = torch::minimum(values_, adjacent.amin(1));
  high_ = torch::maximum(values_, adjacent.amax(1));
  slopes_ = ((adjacent - values_.unsqueeze(1)).unsqueeze(1) *
             s_.coeff.to(values_.options()))
                .sum(2);
}

// True owner or its open-face neighbor evaluated at this physical point.
Tensor SphereReconstruction::evaluate(const Tensor& points, int side) const {
  auto select = [&](Tensor x) {
    if (side >= 0) x = x.index_select(1, s_.neighbors[side]);
    return s_.exchange.take(x);
  };
  auto delta =
      points.to(values_.options()) - select(s_.centers).to(values_.options());
  auto q = select(values_);
  for (int d = 0; d < 3; ++d) q = q + delta[d] * select(slopes_.select(1, d));
  return torch::clamp(q, select(low_), select(high_));
}

Tensor SphereReconstruction::faces(const Tensor& regular, int dim) const {
  int d = 3 - dim, dir = d + 1;
  const auto& g = *s_.block->eb_geometry();
  auto result = regular.clone();
  auto native = [&](Tensor q, int side) {
    if (velocity_) {
      auto full = state_.clone();
      s_.owned(full, dir, side).copy_(q);
      auto m = full.narrow(0, IVX, 3);
      m.copy_(sphere_cart_to_velocity(*s_.block, m, dir));
      return s_.owned(full, dir, side);
    }
    return q;
  };
  for (int side = 0; side <= 1; ++side) {
    int slot = 2 * d + side;
    auto p = s_.owned(g.face_centroid[d], dir, side);
    auto direct = evaluate(p, -1);
    auto peer = evaluate(p, slot);
    auto nb = s_.exchange.adjacency()[slot];
    auto other = s_.near.index_select(0, nb.clamp_min(0));
    auto affected =
        s_.exchange.take((s_.near | ((nb >= 0) & other)).unsqueeze(0));
    auto has_peer = s_.exchange.take(nb.ge(0).unsqueeze(0));
    auto donor = s_.owned(result[side ? ILT : IRT], dir, side);
    auto opposite = s_.owned(result[side ? IRT : ILT], dir, side);
    donor.copy_(torch::where(affected, native(direct, side), donor));
    opposite.copy_(
        torch::where(affected & has_peer, native(peer, side), opposite));
  }
  return result;
}

Tensor SphereReconstruction::wall() const {
  auto wall = state_.clone();
  s_.owned(wall).copy_(
      evaluate(s_.owned(s_.block->eb_geometry()->wall_centroid), -1));
  return wall;
}
}  // namespace snap
