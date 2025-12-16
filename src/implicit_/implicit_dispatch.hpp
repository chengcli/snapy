#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using iterator_fn = void (*)(at::TensorIterator &iter);
using dim_iterator_fn = void (*)(at::TensorIterator &iter, int dim);

using vic_solve_fn = void (*)(at::TensorIterator &iter, double dt, int il,
                              int iu);

using alloc_eigen_fn = void (*)(c10::ScalarType dtype, char *&a, char *&b,
                                char *&c, char *&delta, char *&corr, int ncol,
                                int nlayer);

using free_eigen_fn = void (*)(char *&a, char *&b, char *&c, char *&delta,
                               char *&corr);

DECLARE_DISPATCH(iterator_fn, call_roe_average);
DECLARE_DISPATCH(dim_iterator_fn, call_eigen_system);
DECLARE_DISPATCH(dim_iterator_fn, call_flux_jacobian);

DECLARE_DISPATCH(vic_solve_fn, vic_solve3);
DECLARE_DISPATCH(vic_solve_fn, vic_solve5);

DECLARE_DISPATCH(alloc_eigen_fn, alloc_eigen3);
DECLARE_DISPATCH(alloc_eigen_fn, alloc_eigen5);
DECLARE_DISPATCH(free_eigen_fn, free_eigen);

}  // namespace at::native
