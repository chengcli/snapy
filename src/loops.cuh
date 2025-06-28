// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/ReduceOpsUtils.h>

namespace snap {
namespace native {

template <typename scalar_t, typename func_t>
__global__ void elementwise_kernel(int64_t numel, func_t f) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  int idx = x + y * blockDim.x * gridDim.x + z * blockDim.x * blockDim.y * gridDim.x * gridDim.y;

  // Shared memory allocation
  extern __shared__ unsigned char memory[];
  scalar_t *smem = reinterpret_cast<scalar_t *>(memory);

  if (idx < numel) {
    f(idx, smem);
  }
}

template <int Arity, typename func_t>
void gpu_kernel(at::TensorIterator& iter, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();

  at::native::launch_legacy_kernel<128, 1>(numel,
      [=] __device__ (int idx) {
      auto offsets = offset_calc.get(idx);
      f(data.data(), offsets.data());
    });
}

template <typename scalar_t, int Arity, typename func_t>
void stencil_kernel(at::TensorIterator& iter, int dim, int buffers, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.input().numel();

  TORCH_INTERNAL_ASSERT(numel >= 0 && numel <= std::numeric_limits<int32_t>::max());
  if (numel == 0) {
    return;
  }

  ////// prepare to launch elementwise kernel  /////
  int len[3] = {1, 1, 1};
  auto ndim = iter.input().dim();
  len[3 + dim - ndim] = at::native::ensure_nonempty_size(iter.input(), dim);

  dim3 block(len[2], len[1], len[0]);

  // get dimensions
  for (int i = 1; i < ndim; ++i)
    len[3 + i - ndim] = at::native::ensure_nonempty_size(iter.input(), i);

  // number of variables
  int nvar = at::native::ensure_nonempty_size(iter.output(), 0);

  dim3 grid(len[2] / block.x, len[1] / block.y, len[0] / block.z);
  size_t shared = (len[3 + dim - ndim] * nvar + buffers) * sizeof(scalar_t);

  auto stream = at::cuda::getCurrentCUDAStream();
  printf("block: %d %d %d, grid: %d %d %d\n",
      block.x, block.y, block.z, grid.x, grid.y, grid.z);

  printf("numel: %ld\n", numel);

  elementwise_kernel<scalar_t><<<grid, block, shared, stream>>>(numel,
      [=] __device__ (int idx, scalar_t *smem) {
      auto offsets = offset_calc.get(idx);
      f(data.data(), offsets.data(), smem);
    });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ////// kernel launched /////
}

} // namespace native
} // namespace snap
