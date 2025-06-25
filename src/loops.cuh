// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

namespace snap {
namespace native {

template <typename func_t>
__global__ void elementwise_kernel(int numel, func_t f) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  int idx = x + y * blockDim.x * gridDim.x + z * blockDim.x * blockDim.y * gridDim.x * gridDim.y;

  if (idx < numel) {
    f(idx);
  }
}

template <typename scalar_t, int Arity, typename func_t>
void gpu_kernel(at::TensorIterator& iter, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  at::detail::Array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();
  constexpr int unroll_factor = sizeof(scalar_t) >= 4 ? 2 : 4;

  at::native::launch_legacy_kernel<128, unroll_factor>(numel,
      [=] __device__ (int idx) {
      auto offsets = offset_calc.get(idx);
      f(data.data, offsets.data);
    });
}

template <typename scalar_t, int Arity, typename func_t>
void stencil_kernel(at::TensorIterator& iter, int dim, int buffers, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  at::detail::Array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();

  TORCH_INTERNAL_ASSERT(numel >= 0 && numel <= std::numeric_limits<int32_t>::max());
  if (numel == 0) {
    return;
  }

  ////// prepare to launch elementwise kernel  /////
  int len[3] = {1, 1, 1};
  len[dim - 1] = at::native::ensure_nonempty_size(iter.output(), dim);

  dim3 block(len[2], len[1], len[0]);

  // get dimensions
  len[2] = at::native::ensure_nonempty_size(iter.output(), 3);
  len[1] = at::native::ensure_nonempty_size(iter.output(), 2);
  len[0] = at::native::ensure_nonempty_size(iter.output(), 1);

  // get stencil size
  int stencil = at::native::ensure_nonempty_size(iter.input(1), dim);

  dim3 grid(len[2] / block.x, len[1] / block.y, len[0] / block.z);
  size_t shared = (len[dim - 1] + buffers * stencil) * sizeof(scalar_t);

  auto stream = at::cuda::getCurrentCUDAStream();

  elementwise_kernel<func_t><<<grid, block, shared, stream>>>(numel,
      [=] __device__ (int idx) {
      auto offsets = offset_calc.get(idx);
      f(data.data, offsets.data);
    });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ////// kernel launched /////
}

} // namespace native
} // namespace snap
