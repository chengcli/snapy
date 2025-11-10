// to be included in athena_array.hpp

namespace snap {
template <typename T>
void AthenaArray<T>::InitFromTensor(torch::Tensor const& tensor, int dim) {
  if (dim == 4) {
    initFromTensor4D(tensor, 0, tensor.size(0));
  } else {
    throw std::invalid_argument("Only 4D tensor is supported");
  }
}

template <typename T>
void AthenaArray<T>::InitFromTensor(torch::Tensor const& tensor, int dim,
                                    int index, int nvar) {
  if (dim == 4) {
    initFromTensor4D(tensor, index, nvar);
  } else {
    throw std::invalid_argument("Only 4D tensor is supported");
  }
}

template <typename T>
void AthenaArray<T>::CopyFromTensor(torch::Tensor const& tensor) {
  if (tensor.dim() == 3) {
    copyFromTensor3D(tensor);
  } else {
    throw std::invalid_argument("Only 3D is supported");
  }
}

template <typename T>
void AthenaArray<T>::initFromTensor4D(torch::Tensor const& tensor, int index,
                                      int nvar) {
  nx1_ = tensor.size(3);
  nx2_ = tensor.size(2);
  nx3_ = tensor.size(1);
  nx4_ = nvar;
  nx5_ = 1;
  nx6_ = 1;

  int64_t str1 = 1;
  int64_t str2 = nx1_;
  int64_t str3 = nx2_ * nx1_;
  int64_t str4 = nx3_ * nx2_ * nx1_;

  if (tensor.device().is_cpu()) {  // tensor on CPU
    pdata_ = tensor.data_ptr<T>() + index * str4;
    state_ = DataStatus::shallow_slice;
  } else {                // tensor on device
    DeleteAthenaArray();  // clear existing memory
    pdata_ = new T[nvar * str4];

    // create a temporary tensor holder
    torch::Tensor tmp = torch::from_blob(pdata_, {nvar, nx3_, nx2_, nx1_},
                                         {str4, str3, str2, str1}, nullptr,
                                         torch::dtype(tensor.dtype()));
    auto tmp1 = tensor.slice(0, index, index + nvar).to(torch::kCPU);
    tmp.copy_(tmp1);
    state_ = DataStatus::allocated;
  }
}

template <typename T>
void AthenaArray<T>::copyFromTensor3D(torch::Tensor const& tensor) {
  nx1_ = tensor.size(2);
  nx2_ = tensor.size(1);
  nx3_ = tensor.size(0);
  nx4_ = 1;
  nx5_ = 1;
  nx6_ = 1;

  int64_t str1 = 1;
  int64_t str2 = nx1_;
  int64_t str3 = nx2_ * nx1_;
  int64_t str4 = nx3_ * nx2_ * nx1_;

  DeleteAthenaArray();  // clear existing memory
  pdata_ = new T[str4];

  // create a temporary tensor holder
  torch::Tensor tmp =
      torch::from_blob(pdata_, {nx3_, nx2_, nx1_}, {str3, str2, str1}, nullptr,
                       torch::dtype(tensor.dtype()));
  auto tmp1 = tensor.to(torch::kCPU);
  tmp.copy_(tmp1);
  state_ = DataStatus::allocated;
}
}  // namespace snap
