#pragma once

#define SRC_LR(j, i) source[(j) * (N * stride_x1) + (i) * stride_x1]
#define SRC_BT(j, i) source[(j) * (nghost * stride_x1) + (i) * stride_x1]
#define TARGET(j, i) target[(j) * stride_x2 + (i) * stride_x1]

namespace snap {

//! 1D linear interpolate from source to fill target ghost cell
/*!
 * source is a 3D array with either shape
 * LR: (nghsot, N, nx1), or
 * BT: (N, nghost, nx1) depending on the direction of interpolation.
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
void cs_interp_LR(T *target, const T *source, int N, int nghost, T *u_src,
                  int stride_x2, int stride_x1) {
  /*T u = u_src[n * N + j];
  int i0 = (int)floor(u);
  int i1 = i0 + 1;
  T w1 = u - (T)i0;
  T w0 = 1.0 - w1;

  T v0 = SRC_LR(n, i0);
  T v1 = SRC_LR(n, i1);*/
  // TARGET(n, j) = w0 * v0 + w1 * v1;
  *target = 3.;
}

template <typename T>
void cs_interp_BT(T *target, const T *source, int N, int nghost, T *u_src,
                  int stride_x2, int stride_x1) {
  for (int n = 0; n < nghost; ++n)
    for (int j = 0; j < N; ++j) {
      T u = u_src[n * N + j];
      int i0 = (int)floor(u);
      int i1 = i0 + 1;
      T w1 = u - (T)i0;
      T w0 = 1.0 - w1;

      T v0 = SRC_BT(i0, n);
      T v1 = SRC_BT(i1, n);
      TARGET(j, n) = w0 * v0 + w1 * v1;
    }
}

template <typename T>
void cs_lonlat_to_ab(int *nP, T *alpha, T *beta, T lon, T lat) {
  // Translate from RLL coordinates to XYZ space
  T xx, yy, zz, pm;

  xx = cos(lon) * cos(lat);
  yy = sin(lon) * cos(lat);
  zz = sin(lat);

  pm = std::max(fabs(xx), std::max(fabs(yy), fabs(zz)));

  // Check maxmality of the x coordinate
  if (pm == fabs(xx)) {
    if (xx > 0) {  // +X
      (*nP) = 0;
    } else {  // -X
      (*nP) = 2;
    }
  }

  // Check maximality of the y coordinate
  if (pm == fabs(yy)) {
    if (yy > 0) {  // +Y
      (*nP) = 1;
    } else {  // -Y
      (*nP) = 4;
    }
  }

  // Check maximality of the z coordinate
  if (pm == fabs(zz)) {
    if (zz > 0) {  // +Z
      (*nP) = 3;
    } else {  // -Z
      (*nP) = 5;
    }
  }

  // Panel assignments
  double sx, sy, sz;
  switch ((*nP)) {
    case 0:
      sx = yy;
      sy = zz;
      sz = xx;
      break;
    case 1:
      sx = -xx;
      sy = zz;
      sz = yy;
      break;
    case 2:
      sx = -yy;
      sy = zz;
      sz = -xx;
      break;
    case 3:
      sx = yy;
      sy = -xx;
      sz = zz;
      break;
    case 4:
      sx = xx;
      sy = zz;
      sz = -yy;
      break;
    case 5:
      sx = yy;
      sy = xx;
      sz = -zz;
      break;
  }

  // Convert to gnomonic coordinates
  (*alpha) = atan2(sx, sz);
  (*beta) = atan2(sy, sz);
}

}  // namespace snap

#undef SRC_LR
#undef SRC_BT
#undef TARGET
