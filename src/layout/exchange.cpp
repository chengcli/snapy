// snap
#include "exchange.hpp"
#include <snap/mesh/meshblock.hpp>

namespace snap {

void init_buffers_2d(MeshBlockImpl const* block,
                     torch::Tensor const& hydro_u,
                     std::vector<torch::Tensor>& send_bufs,
                     std::vector<torch::Tensor>& recv_bufs) {
  // Initialize vectors to size 9 with empty tensors
  send_bufs.clear();
  recv_bufs.clear();
  send_bufs.resize(9);
  recv_bufs.resize(9);

  // Iterate over all 2D neighbor directions
  for (int x3_offset = -1; x3_offset <= 1; ++x3_offset) {
    for (int x2_offset = -1; x2_offset <= 1; ++x2_offset) {
      // Skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;

      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);
      int bid = get_buffer_id(x3_offset, x2_offset, 0);

      // Get the part indices for this neighbor direction
      auto part = block->part(offset, false);  // false = interior part
      
      // Get shape by applying indices to tensor
      auto part_tensor = hydro_u.index(part);
      
      // Allocate send and receive buffers with same shape
      send_bufs[bid] = torch::empty_like(part_tensor);
      recv_bufs[bid] = torch::empty_like(part_tensor);
    }
  }
}

void serialize_2d(MeshBlockImpl const* block,
                  torch::Tensor& hydro_u,
                  std::vector<torch::Tensor>& send_bufs) {
  // Iterate over all 2D neighbor directions
  for (int x3_offset = -1; x3_offset <= 1; ++x3_offset) {
    for (int x2_offset = -1; x2_offset <= 1; ++x2_offset) {
      // Skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;

      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);
      int bid = get_buffer_id(x3_offset, x2_offset, 0);

      // Only serialize if buffer exists
      if (send_bufs[bid].defined()) {
        // Get the interior part for this direction
        auto part = block->part(offset, false);  // false = interior part
        
        // Copy data from mesh to send buffer
        send_bufs[bid].copy_(hydro_u.index(part));
      }
    }
  }
}

void deserialize_2d(MeshBlockImpl const* block,
                    torch::Tensor& hydro_u,
                    std::vector<torch::Tensor> const& recv_bufs) {
  // Iterate over all 2D neighbor directions
  for (int x3_offset = -1; x3_offset <= 1; ++x3_offset) {
    for (int x2_offset = -1; x2_offset <= 1; ++x2_offset) {
      // Skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;

      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);
      int bid = get_buffer_id(x3_offset, x2_offset, 0);

      // Only deserialize if buffer exists
      if (recv_bufs[bid].defined()) {
        // Get the exterior (ghost zone) part for this direction
        auto part = block->part(offset, true);  // true = exterior part (ghost zones)
        
        // Copy data from receive buffer to mesh ghost zones
        hydro_u.index_put_(part, recv_bufs[bid]);
      }
    }
  }
}

}  // namespace snap
