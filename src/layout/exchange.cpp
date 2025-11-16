// snap
#include "exchange.hpp"
#include <snap/layout/cubed_layout.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/layout/distribute_info.hpp>
#include <snap/layout/slab_layout.hpp>
#include <snap/mesh/meshblock.hpp>

// torch distributed
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#ifdef USE_NCCL
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif

// C++
#include <cstdlib>
#include <stdexcept>

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

InitDistResult init_dist(
    std::string const& backend,
    std::string const& init_method,
    std::string const& layout_type,
    int px, int py, int pz,
    bool periodic_x1,
    bool periodic_x2,
    bool periodic_x3,
    std::string const& device_type,
    int local_rank) {
  
  InitDistResult result;
  
  // Initialize process group using environment variables
  // Note: In C++, we use c10d directly instead of torch.distributed
  // The process group should be initialized externally or via Python
  // This function focuses on setting up the layout and ranks
  
  // Get world size and rank from environment (set by torch.distributed or mpirun)
  const char* world_size_env = std::getenv("WORLD_SIZE");
  const char* rank_env = std::getenv("RANK");
  
  if (!world_size_env || !rank_env) {
    throw std::runtime_error(
        "WORLD_SIZE and RANK environment variables must be set. "
        "Initialize torch.distributed from Python or use mpirun/torchrun.");
  }
  
  int world_size = std::atoi(world_size_env);
  int rank = std::atoi(rank_env);
  
  // Determine device
  if (device_type == "cuda") {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("CUDA requested but not available");
    }
    
    int ngpu = torch::cuda::device_count();
    if (local_rank < 0) {
      const char* local_rank_env = std::getenv("LOCAL_RANK");
      local_rank = local_rank_env ? std::atoi(local_rank_env) : (rank % std::max(1, ngpu));
    }
    
    torch::cuda::set_device(local_rank);
    result.device = torch::Device(torch::kCUDA, local_rank);
  } else {
    result.device = torch::Device(torch::kCPU);
  }
  
  // Create layout based on type
  result.layout_type = layout_type;
  result.info = DistributeInfo();
  result.info.gid(rank);
  
  if (layout_type == "slab") {
    // Slab layout: 2D decomposition
    if (pz != 1) {
      throw std::invalid_argument("px1 (pz) must be 1 for slab layout");
    }
    if (px * py != world_size) {
      throw std::invalid_argument(
          "px2*px3 (" + std::to_string(px) + "*" + std::to_string(py) + 
          ") != world_size (" + std::to_string(world_size) + ")");
    }
    
    auto layout = std::make_shared<SlabLayout>(px, py, periodic_x3, periodic_x2);
    auto loc = layout->loc_of(rank);
    
    result.info.nb3(px);
    result.info.nb2(py);
    result.info.lx3(loc.first);
    result.info.lx2(loc.second);
    
    // Build ranks array for 2D
    result.ranks.resize(9);
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        if (dx != 0 || dy != 0) {
          int bid = get_buffer_id(dx, dy, 0);
          result.ranks[bid] = layout->neighbor_rank(loc.first, loc.second, dx, dy);
        }
      }
    }
    result.ranks[get_buffer_id(0, 0, 0)] = rank;
    
    result.layout = layout;
    
  } else if (layout_type == "cubed") {
    // Cubed layout: 3D decomposition
    if (px * py * pz != world_size) {
      throw std::invalid_argument(
          "px1*px2*px3 (" + std::to_string(px) + "*" + std::to_string(py) + 
          "*" + std::to_string(pz) + ") != world_size (" + std::to_string(world_size) + ")");
    }
    
    auto layout = std::make_shared<CubedLayout>(px, py, pz, periodic_x3, periodic_x2, periodic_x1);
    auto loc = layout->loc_of(rank);
    
    result.info.nb3(px);
    result.info.nb2(py);
    result.info.nb1(pz);
    result.info.lx3(std::get<0>(loc));
    result.info.lx2(std::get<1>(loc));
    result.info.lx1(std::get<2>(loc));
    
    // Build ranks array for 3D
    result.ranks.resize(27);
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          if (dx != 0 || dy != 0 || dz != 0) {
            int bid = get_buffer_id(dx, dy, dz);
            result.ranks[bid] = layout->neighbor_rank(
                std::get<0>(loc), std::get<1>(loc), std::get<2>(loc), dx, dy, dz);
          }
        }
      }
    }
    result.ranks[get_buffer_id(0, 0, 0)] = rank;
    
    result.layout = layout;
    
  } else if (layout_type == "cubed_sphere") {
    // Cubed sphere layout
    if (pz != 1) {
      throw std::invalid_argument("px1 (pz) must be 1 for cubed_sphere layout");
    }
    if (px != py) {
      throw std::invalid_argument("px2 must equal px3 for cubed_sphere layout");
    }
    if (6 * px * py != world_size) {
      throw std::invalid_argument(
          "6*px2*px3 (6*" + std::to_string(px) + "*" + std::to_string(py) + 
          ") != world_size (" + std::to_string(world_size) + ")");
    }
    
    auto layout = std::make_shared<CubedSphereLayout>(px);
    auto loc = layout->loc_of(rank);
    
    result.info.nb3(px);
    result.info.nb2(py);
    result.info.face(std::get<0>(loc));
    result.info.lx3(std::get<1>(loc));
    result.info.lx2(std::get<2>(loc));
    
    // Build ranks array for 2D (cubed sphere is 2D on each face)
    result.ranks.resize(9);
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        if (dx != 0 || dy != 0) {
          int bid = get_buffer_id(dx, dy, 0);
          result.ranks[bid] = layout->neighbor_rank(
              std::get<0>(loc), std::get<1>(loc), std::get<2>(loc), dx, dy);
        }
      }
    }
    result.ranks[get_buffer_id(0, 0, 0)] = rank;
    
    result.layout = layout;
    
  } else {
    throw std::invalid_argument("Unknown layout type: " + layout_type);
  }
  
  return result;
}

void slab_exchange(MeshBlockImpl const* block,
                   torch::Tensor& hydro_u,
                   std::vector<int> const& ranks,
                   std::vector<torch::Tensor>& send_bufs,
                   std::vector<torch::Tensor>& recv_bufs) {
  // Serialize data into send buffers
  serialize_2d(block, hydro_u, send_bufs);
  
  // Note: For C++ implementation, we need access to the process group
  // which is typically initialized from Python via torch.distributed.
  // 
  // The recommended approach is to call this function from Python where
  // torch.distributed handles the communication, OR to pass a process
  // group reference as a parameter.
  //
  // For now, we provide the serialization/deserialization logic,
  // and the communication should be handled at a higher level.
  //
  // Example Python usage:
  //   serialize_2d(block, hydro_u, send_bufs)
  //   # Use torch.distributed for communication
  //   ops = []
  //   for r in range(1, len(ranks)):
  //       if send_bufs[r] is not None:
  //           ops.append(dist.P2POp(dist.isend, send_bufs[r], ranks[r]))
  //           ops.append(dist.P2POp(dist.irecv, recv_bufs[r], ranks[r]))
  //   if ops:
  //       reqs = dist.batch_isend_irecv(ops)
  //       for r in reqs: r.wait()
  //   deserialize_2d(block, hydro_u, recv_bufs)
  
  // For a complete C++ implementation without Python, uncomment and adapt:
  /*
  // Get default process group (requires torch.distributed initialization)
  auto pg = c10d::ProcessGroupGloo::createProcessGroupGloo(...);
  
  std::vector<c10::intrusive_ptr<c10d::Work>> works;
  
  for (size_t r = 1; r < ranks.size(); ++r) {
    if (ranks[r] >= 0 && send_bufs[r].defined() && recv_bufs[r].defined()) {
      // Send operation
      std::vector<at::Tensor> send_tensors = {send_bufs[r]};
      auto send_work = pg->send(send_tensors, ranks[r], 0);
      works.push_back(send_work);
      
      // Receive operation
      std::vector<at::Tensor> recv_tensors = {recv_bufs[r]};
      auto recv_work = pg->recv(recv_tensors, ranks[r], 0);
      works.push_back(recv_work);
    }
  }
  
  // Wait for all operations to complete
  for (auto& work : works) {
    work->wait();
  }
  */
  
  // Deserialize received data into ghost zones
  deserialize_2d(block, hydro_u, recv_bufs);
}

}  // namespace snap
