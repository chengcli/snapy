// snap
#include "spherical_polar.hpp"

#include <snap/snap.h>

#include <snap/eos/equation_of_state.hpp>

#include "spherical_utils.hpp"

#define SQR(x) ((x) * (x))

namespace snap {

void SphericalPolarImpl::reset() {
  // register buffers defined in the base class
  register_buffer("x1f", x1f);
  register_buffer("x2f", x2f);
  register_buffer("x3f", x3f);
  register_buffer("cosine_cell_kj", cosine_cell_kj);
}

torch::Tensor SphericalPolarImpl::forward(torch::Tensor prim,
                                          torch::Tensor flux1,
                                          torch::Tensor flux2,
                                          torch::Tensor flux3) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1, DIMC = 0 };

  auto div = CoordinateImpl::forward(prim, flux1, flux2, flux3);
  bool use_x2_fluxes = options->nx2() > 1;

  int si = is();
  int ei = ie() + 1;
  int sj = js();
  int ej = je() + 1;
  int sk = ks();
  int ek = ke() + 1;

  // src_1 = < M_{theta theta} + M_{phi phi} ><1/r>
  auto m_ii = prim[IDN] * (SQR(prim[IVY]) + SQR(prim[IVZ]));

  if (options->eos()->type() == "shallow-water") {
    // m_ii += 2.0*(iso_cs*iso_cs)*prim[IDN];
  } else {
    m_ii += 2.0 * prim[IPR];
  }

  div[IVX] += coord_src1_i * m_ii;

  // src_2 = -< M_{theta r} ><1/r>
  div[IVY] -=
      coord_src2_i *
      (face_area1(si, ei) * flux1[IVY].slice(DIM1, si, ei) +
       face_area1(si + 1, ei + 1) * flux1[IVY].slice(DIM1, si + 1, ei + 1));

  // src_3 = -< M_{phi r} ><1/r>
  div[IVZ] -=
      coord_src2_i *
      (face_area1(si, ei) * flux1[IVZ].slice(DIM1, si, ei) +
       face_area1(si + 1, ei + 1) * flux1[IVZ].slice(DIM1, si + 1, ei + 1));

  // src_2 = < M_{phi phi} ><cot theta/r>
  auto m_pp = prim[IDN] * SQR(prim[IVZ]);
  if (options->eos()->type() == "shallow-water") {
    // m_pp += (iso_cs*iso_cs)*prim[IDN];
  } else {
    m_pp += prim[IPR];
  }

  div[IVY] += coord_src1_i * coord_src1_j * m_pp;

  // src_3 = -< M_{phi theta} ><cot theta/r>
  if (use_x2_fluxes) {
    div[IVZ] -=
        coord_src1_i * coord_src2_j *
        (face_area2(sj, ej) * flux2[IVZ].slice(DIM2, sj, ej) +
         face_area2(sj + 1, ej + 1) * flux2[IVZ].slice(DIM2, sj + 1, ej + 1));
  } else {
    auto m_ph = prim[IDN] * prim[IVZ] * prim[IVY];
    div[IVZ] -= coord_src1_i * coord_src3_j * m_ph;
  }

  return div;
}

}  // namespace snap

#undef SQR
