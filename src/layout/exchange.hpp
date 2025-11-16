#pragma once

// C/C++
#include <tuple>
#include <vector>

// torch
#include <torch/torch.h>

namespace snap {

// Forward declarations
class MeshBlockImpl;

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

}  // namespace snap
