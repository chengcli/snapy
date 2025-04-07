#pragma once

// torch
#include <c10/core/ScalarType.h>

namespace snap {
template <int N>
void alloc_eigen_cuda(char*& a, char*& b, char*& c, char*& delta, char*& corr,
                      int ncol, int nlayer, c10::ScalarType dtype);

template <int N>
void alloc_eigen_cpu(char*& a, char*& b, char*& c, char*& delta, char*& corr,
                     int ncol, int nlayer, c10::ScalarType dtype);

void free_eigen_cuda(char*& a, char*& b, char*& c, char*& delta, char*& corr);
void free_eigen_cpu(char*& a, char*& b, char*& c, char*& delta, char*& corr);
}  // namespace snap
