#include "sphere_exchange.hpp"

#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/meshblock.hpp>

namespace snap {
SphereExchange::SphereExchange(const MeshBlockImpl& block)
    : layout_(block.get_layout()) {
  auto const& c = *block.options->coord();
  auto const& l = *layout_->options;
  auto n = c.global_nx2(), nr = c.global_nx1();
  TORCH_CHECK(
      l.type() == "cubed-sphere" && l.pz() == 1 && c.global_nx3() == n,
      "embedded sphere requires square panels and an undivided radial column");
  shape_ = {c.nx3(), c.nx2(), nr};
  count_ = int64_t(6) * n * n * nr;
  auto id = [=](int p, int b, int a, int r) {
    return (((int64_t(p) * n + b) * n + a) * nr + r);
  };
  auto adj = torch::full({6, count_}, -1, torch::kLong);
  auto back = torch::full_like(adj, -1);
  auto edges = adj.accessor<int64_t, 2>(), backs = back.accessor<int64_t, 2>();
  for (int p = 0; p < 6; ++p)
    for (int b = 0; b < n; ++b)
      for (int a = 0; a < n; ++a)
        for (int r = 0; r < nr; ++r) {
          auto mine = id(p, b, a, r);
          for (int side = 0; side < 6; ++side) {
            int f = p, y = b, x = a, z = r, opposite = side ^ 1;
            int sign = side % 2 ? 1 : -1;
            if (side < 2)
              z += sign;
            else if (side < 4)
              x += sign;
            else
              y += sign;
            if (z < 0 || z >= nr) continue;
            if (x < 0 || x >= n || y < 0 || y >= n) {
              auto e = CS_FACE_EDGES[p][side - 2];
              int k = side < 4 ? b : a;
              if (e.rev) k = n - 1 - k;
              f = e.nface;
              opposite = 2 + e.nside;
              x = e.nside == SIDE_L ? 0 : e.nside == SIDE_R ? n - 1 : k;
              y = e.nside == SIDE_B ? 0 : e.nside == SIDE_T ? n - 1 : k;
            }
            edges[side][mine] = id(f, y, x, z);
            backs[side][mine] = opposite;
          }
        }
  auto device = block.pcoord->x1v.device();
  adjacent_ = adj.to(device);
  back_ = back.to(device);
  // Local block zero owns the process collective; other blocks pass their
  // owned tensor directly to its worker through the mesh-scoped mailbox.
  for (int i = 0; i < l.blocks_per_process(); ++i) {
    auto [ax, by, p] =
        layout_->loc_of(l.global_block_rank(l.process_rank(), i));
    auto ids = torch::empty({shape_[0] * shape_[1] * shape_[2]}, torch::kLong);
    auto ptr = ids.data_ptr<int64_t>();
    for (int b = 0; b < shape_[0]; ++b)
      for (int a = 0; a < shape_[1]; ++a)
        for (int r = 0; r < nr; ++r)
          *ptr++ = id(p, by * shape_[0] + b, ax * shape_[1] + a, r);
    local_indices_.push_back(ids.to(device));
  }
}

torch::Tensor SphereExchange::gather(const torch::Tensor& owned) const {
  TORCH_CHECK(
      owned.dim() == 4 && owned.sizes().slice(1).equals(shape_),
      "sphere gather expects [channels,local_x3,local_x2,x1] owned cells");
  auto const& l = *layout_->options;
  int root = l.global_block_rank(l.process_rank(), 0);
  constexpr int input_tag = 21800, result_tag = 21801;
  if (l.rank() != root) {
    layout_->post_to_local_block(root, owned, input_tag);
    return layout_->take_from_local_block(root, result_tag);
  }
  auto global = torch::zeros({owned.size(0), count_}, owned.options());
  for (int i = 0; i < l.blocks_per_process(); ++i) {
    auto part =
        i == 0 ? owned : layout_->take_from_local_block(root + i, input_tag);
    global.index_copy_(1, local_indices_[i], part.flatten(1));
  }
  if (layout_->has_process_group()) {
    // Every slot has a single owner. A sum here only assembles disjoint data.
    std::vector<torch::Tensor> values{global};
    layout_->comm->allreduce(values, c10d::ReduceOp::SUM);
  }
  for (int i = 1; i < l.blocks_per_process(); ++i)
    layout_->post_to_local_block(root + i, global, result_tag);
  return global;
}

torch::Tensor SphereExchange::take(const torch::Tensor& global) const {
  TORCH_CHECK(global.dim() == 2 && global.size(1) == count_,
              "invalid global sphere field");
  auto const& l = *layout_->options;
  auto i = l.local_block_index(l.rank());
  return global.index_select(1, local_indices_[i])
      .reshape({global.size(0), shape_[0], shape_[1], shape_[2]});
}
}  // namespace snap
