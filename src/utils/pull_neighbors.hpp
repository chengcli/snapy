// torch
#include <torch/torch.h>

namespace snap {

//! \brief Pull neighboring values in 2D
//!
//! \param[in] input Input tensor containing values to pull from neighbors
//! \return Tensor with neighboring values pulled in 2D
torch::Tensor pull_neighbors2(const torch::Tensor& input);

//! \brief Pull neighboring values in 3D
//!
//! \param[in] input Input tensor containing values to pull from neighbors
//! \return Tensor with neighboring values pulled in 3D
torch::Tensor pull_neighbors3(const torch::Tensor& input);

//! \brief Pull neighboring values in 4D
//!
//! \param[in] input Input tensor containing values to pull from neighbors
//! \return Tensor with neighboring values pulled in 4D
torch::Tensor pull_neighbors4(const torch::Tensor& input);

}  // namespace snap
