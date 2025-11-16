#pragma once

// C/C++
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// torch
#include <torch/torch.h>

namespace snap {

// Forward declarations
class MeshBlockImpl;
struct DistributeInfo;
class SlabLayout;
class CubedLayout;
class CubedSphereLayout;

/*!
 * \brief Calculate buffer ID from directional offsets
 * 
 * Converts 3D directional offsets into a linear buffer index.
 * For 2D layouts (dz=0), returns index in range [0,8].
 * For 3D layouts, returns index in range [0,26].
 * 
 * \param dx offset in x3 direction (-1, 0, or 1)
 * \param dy offset in x2 direction (-1, 0, or 1)
 * \param dz offset in x1 direction (-1, 0, or 1)
 * \return linear buffer index
 */
inline int get_buffer_id(int dx, int dy, int dz = 0) {
  return (dx % 3 + 3) % 3 + ((dy % 3 + 3) % 3) * 3 + ((dz % 3 + 3) % 3) * 9;
}

/*!
 * \brief Initialize send and receive buffers for 2D domain decomposition
 * 
 * Allocates torch::Tensor buffers for exchanging ghost zone data with
 * neighboring processes in a 2D slab decomposition. Buffers are sized
 * to match the ghost zone dimensions of the mesh block.
 * 
 * \param block pointer to MeshBlock containing grid information
 * \param hydro_u tensor containing hydro variables [nhydro, nc3, nc2, nc1]
 * \param send_bufs output vector of send buffers (size 9)
 * \param recv_bufs output vector of receive buffers (size 9)
 */
void init_buffers_2d(MeshBlockImpl const* block,
                     torch::Tensor const& hydro_u,
                     std::vector<torch::Tensor>& send_bufs,
                     std::vector<torch::Tensor>& recv_bufs);

/*!
 * \brief Serialize mesh data into send buffers
 * 
 * Copies data from the interior boundaries of the mesh into send buffers
 * for each neighbor direction. Used in 2D domain decomposition.
 * 
 * \param block pointer to MeshBlock containing grid information
 * \param hydro_u tensor containing hydro variables [nhydro, nc3, nc2, nc1]
 * \param send_bufs vector of send buffers to fill
 */
void serialize_2d(MeshBlockImpl const* block,
                  torch::Tensor& hydro_u,
                  std::vector<torch::Tensor>& send_bufs);

/*!
 * \brief Deserialize received data into mesh ghost zones
 * 
 * Copies data from receive buffers into the ghost zones of the mesh
 * for each neighbor direction. Used in 2D domain decomposition.
 * 
 * \param block pointer to MeshBlock containing grid information
 * \param hydro_u tensor containing hydro variables [nhydro, nc3, nc2, nc1]
 * \param recv_bufs vector of receive buffers to read from
 */
void deserialize_2d(MeshBlockImpl const* block,
                    torch::Tensor& hydro_u,
                    std::vector<torch::Tensor> const& recv_bufs);

/*!
 * \brief Initialize distributed computing environment
 * 
 * Initializes the process group for distributed communication and sets up
 * layout information including neighbor ranks for ghost zone exchanges.
 * 
 * \param backend "gloo" for CPU, "nccl" for CUDA
 * \param init_method initialization method (e.g., "env://", "tcp://...")
 * \param layout_type "slab", "cubed", or "cubed_sphere"
 * \param px processes in x3 direction
 * \param py processes in x2 direction
 * \param pz processes in x1 direction
 * \param periodic_x1 periodic boundary in x1
 * \param periodic_x2 periodic boundary in x2
 * \param periodic_x3 periodic boundary in x3
 * \param device_type "cpu" or "cuda"
 * \param local_rank local rank for CUDA device selection (-1 for auto)
 * \return InitDistResult structure with layout, ranks, device, and info
 */
struct InitDistResult {
  std::shared_ptr<void> layout;  // Will be SlabLayout, CubedLayout, or CubedSphereLayout
  std::vector<int> ranks;
  torch::Device device;
  DistributeInfo info;
  std::string layout_type;
};

InitDistResult init_dist(
    std::string const& backend,
    std::string const& init_method,
    std::string const& layout_type,
    int px, int py, int pz,
    bool periodic_x1 = false,
    bool periodic_x2 = false,
    bool periodic_x3 = false,
    std::string const& device_type = "cpu",
    int local_rank = -1);

/*!
 * \brief Perform ghost zone exchange for slab layout
 * 
 * Exchanges ghost zone data with neighboring processes using point-to-point
 * communication. This function serializes data, performs send/recv operations,
 * and deserializes received data into ghost zones.
 * 
 * \param block pointer to MeshBlock containing grid information
 * \param hydro_u tensor containing hydro variables [nhydro, nc3, nc2, nc1]
 * \param ranks vector of neighbor ranks
 * \param send_bufs vector of send buffers
 * \param recv_bufs vector of receive buffers
 */
void slab_exchange(MeshBlockImpl const* block,
                   torch::Tensor& hydro_u,
                   std::vector<int> const& ranks,
                   std::vector<torch::Tensor>& send_bufs,
                   std::vector<torch::Tensor>& recv_bufs);

}  // namespace snap
