#pragma once

// C/C++
#include <cstdarg>
#include <cstdio>

// dispatch
#include <configure.h>

template <typename T, int N, typename... Arg>
void DISPATCH_MACRO printf_vector(char const* fmt,
                                  Eigen::Vector<T, N> const& vec, Arg... args) {
  printf(fmt, std::forward<Arg>(args)...);
  printf("(");
  for (int i = 0; i < N; i++) {
    printf("%g", vec(i));
    if (i != N - 1) printf(", ");
  }
  printf(")\n");
}

template <typename T, int N, typename... Arg>
void DISPATCH_MACRO
printf_matrix(char const* fmt,
              Eigen::Matrix<T, N, N, Eigen::RowMajor> const& mat, Arg... args) {
  // printf(fmt, std::forward<Arg>(args)...);
  printf("(\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%12g", mat(i, j));
      if (j != N - 1) printf(", ");
    }
    if (i != N - 1) printf(";\n");
  }
  printf(")\n");
}
