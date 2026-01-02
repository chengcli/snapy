#include <torch/torch.h>

//! \brief Serialize a PyTorch tensor to a byte buffer
//!
//! Converts a PyTorch tensor into a serialized byte representation
//! that can be transmitted or stored.
//!
//! \param[in] tensor The tensor to serialize
//! \return Pointer to serialized data buffer
//!
//! \note The returned buffer is managed by the VectorStream object
char const* serialize(const torch::Tensor& tensor) {
  VectorStream vs(tensor.numel() * tensor.element_size());
  std::ostream os(&vs);
  torch::save(tensor, os);
  return vs.buffer();
}

//! \brief Deserialize a byte buffer to a PyTorch tensor
//!
//! Reconstructs a PyTorch tensor from its serialized byte representation.
//!
//! \param[in] data Pointer to serialized data
//! \param[in] size Size of serialized data in bytes
//! \return Deserialized PyTorch tensor
torch::Tensor deserialize(const char* data, size_t size) {
  std::istringstream is(std::string(data, size));
  return torch::load(is);
}
