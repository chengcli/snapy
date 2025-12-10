#pragma once

// base
#include <configure.h>

namespace snap {

template <typename T>
void DISPATCH_MACRO coord_vec_raise_impl(T *v2, T *v3, T cth) {
  T v = (*v2);
  T w = (*v3);
  T sth2 = 1. - cth * cth;

  (*v2) = v / sth2 - w * cth / sth2;
  (*v3) = -v * cth / sth2 + w / sth2;
}

template <typename T>
void DISPATCH_MACRO coord_vec_lower_impl(T *v2, T *v3, T cth) {
  T v = (*v2);
  T w = (*v3);
  (*v2) = v + w * cth;
  (*v3) = w + v * cth;
}

}  // namespace snap
