// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>

// fvm
#include <fvm/loops.cuh>

using namespace canoe;

template <typename T>
inline void __host__ __device__ test1_impl(T* out, T* a, int out_dim_stride,
                                           int in_dim_stride, int nhydro) {
  for (int i = 0; i < nhydro; i++) {
    out[0 * out_dim_stride] += a[i * in_dim_stride];
    out[1 * out_dim_stride] += a[i * in_dim_stride] * a[i * in_dim_stride];
    out[2 * out_dim_stride] +=
        a[i * in_dim_stride] * a[i * in_dim_stride] * a[i * in_dim_stride];
  }
}

template <int N>
void call_test_cuda(at::TensorIterator& iter, int dim) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "test1_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.input(0), dim);
    auto in_dim_stride = at::native::ensure_nonempty_stride(iter.input(0), dim);
    auto out_dim_stride =
        at::native::ensure_nonempty_stride(iter.output(), dim);

    native::gpu_kernel<scalar_t, N>(
        iter, [=] GPU_LAMBDA(char* const data[N], unsigned int strides[N]) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto a = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          test1_impl(out, a, out_dim_stride, in_dim_stride, nhydro);
          printf("out = %f, a = %f\n", *out, *a);
        });
  });
}

void call_test_cpu(at::TensorIterator& iter, int dim) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "test1_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.input(0), dim);
    auto in_dim_stride = at::native::ensure_nonempty_stride(iter.input(0), dim);
    auto out_dim_stride =
        at::native::ensure_nonempty_stride(iter.output(), dim);

    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto a = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        test1_impl(out, a, out_dim_stride, in_dim_stride, nhydro);
        printf("i = %d, out = %f, a = %f\n", i, *out, *a);
      }
    });
  });
}

TEST(gpu_kernel, test1) {
  auto a = torch::randn({3, 1, 5, 4}).to(torch::kCUDA);
  std::cout << "a = " << a << std::endl;

  auto out = torch::zeros_like(a);

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), /*squash_dims=*/0)
                  .add_output(out)
                  .add_input(a)
                  .build();

  call_test_cuda<2>(iter, 0);

  std::cout << "out = " << out << std::endl;
}

TEST(cpu_kernel, test1) {
  auto a = torch::randn({3, 1, 5, 4});
  std::cout << "a = " << a << std::endl;

  auto out = torch::zeros_like(a);

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), /*squash_dims=*/0)
                  .add_output(out)
                  .add_input(a)
                  .build();

  call_test_cpu(iter, 0);

  std::cout << "out = " << out << std::endl;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
