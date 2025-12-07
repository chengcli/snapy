#pragma once

#define SOURCE(j, i) source[(j) * stride_x2 + (i) * stride_x1]
#define TARGET(j, i) target[(j) * stride_x2 + (i) * stride_x1]

namespace snap {

//! 1D linear interpolate from source to fill target ghost cell
/*!
 * source is a 3D array with shape (nx3, nx2, nx1)
 * to access source data, use SOURCE(j,i) macro defined above
 *
 * u_src is an array of length nghost*N giving the fractional
 * source coordinates along the source edge line for each
 * target ghost cell to be filled.
 * to access u_src, use u_src[n*N + j] for ghost depth n and
 * along-edge index j.
 *
 * target is a 3D array with shape (nx3, nx2, nx1) to be filled
 * with interpolated ghost values; use TARGET(j,i) macro
 * defined above to access target data.
 *
 * Both source and target share the same strides stride_x2
 * and stride_x1 for the 2nd and 1st dimensions.
 */
template <typename T>
void cs_interp_LR(T* target, const T* source, int N, int nghost, T* u_src,
                  int stride_x2, int stride_x1) {
  for (int n = 0; n < nghost; ++n)
    for (int j = 0; j < N; ++j) {
      T u = u_src[n * N + j];
      int i0 = (int)floor(u);
      int i1 = i0 + 1;
      T w1 = u - (T)i0;
      T w0 = 1.0 - w1;

      T v0 = SOURCE(i0, n);
      T v1 = SOURCE(i1, n);
      TARGET(j, n) = w0 * v0 + w1 * v1;
    }
}

template <typename T>
void cs_interp_BT(T* target, const T* source, int N, int nghost, T* u_src,
                  int stride_x2, int stride_x1) {
  for (int n = 0; n < nghost; ++n)
    for (int j = 0; j < N; ++j) {
      T u = u_src[n * N + j];
      int i0 = (int)floor(u);
      int i1 = i0 + 1;
      T w1 = u - (T)i0;
      T w0 = 1.0 - w1;

      T v0 = SOURCE(n, i0);
      T v1 = SOURCE(n, i1);
      TARGET(n, j) = w0 * v0 + w1 * v1;
    }
}

}  // namespace snap

#undef SOURCE
#undef TARGET
