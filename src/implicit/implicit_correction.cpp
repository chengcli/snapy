// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "implicit.hpp"

namespace snap {

ImplicitCorrectionImpl::ImplicitCorrectionImpl(ImplicitOptions options_)
    : options(options_) {
  reset();
}

void ImplicitCorrectionImpl::reset() {
  pihc = register_module("vic", ImplicitHydro(options));
}

torch::Tensor ImplicitCorrectionImpl::forward(torch::Tensor du, torch::Tensor w,
                                              torch::Tensor wlr[3], double dt) {
  if (options.scheme() == 0) {  // null operation
    return du;
  }

  //// -------- Vertical direction --------- ////
  auto [a, b, c] = pihc->forward(w, wlr[2], 3);
  auto delta = torch::zeros_like(a.select(-1, 0));

  int m = option.size();
  auto Dt = torch::eye(m, w.options()) / dt;
  auto Phi = torch::zeros({m, m}, w.options());

  Phi[IVX][IDN] = options.grav();
  Phi[m - 1][IVX] = options.grav();

  auto Bnd = torch::eye(m, w.options());
  Bnd[IVX][IVX] = -1.;

  int is = pihc->peos->pcoord->is();
  int ie = pihc->peos->pcoord->ie();

  a.slice(d, is, ie) += Dt - Phi;

  //// --------- Fix boundary condition ---------- ////
  a.select(2, is) += b.select(2, is).matmul(Bnd);
  a.select(2, ie) += c.select(2, ie).matmul(Bnd);

  //// -------- Solve block-tridiagonal matrix --------- ////
  auto du0 = du.clone();
  std::vector<int64_t> vec = {3, 0, 1, 2};
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(du.sizes(), /*squash_dims=*/{0, 3})
                  .add_output(du)
                  .add_input(w)
                  .add_owned_input(a.permute(vec))
                  .add_owned_input(b.permute(vec))
                  .add_owned_input(c.permute(vec))
                  .add_owned_input(delta.permute(vec))
                  .build();

  if (options.scheme() == 1) {
    at::native::vic_solve3(du.device().type(), iter, dt, is, ie);
  } else if (options.scheme() == 9) {
    at::native::vic_solve5(du.device().type(), iter, dt, is, ie);
  } else {
    TORCH_CHECK(false, "Unknown implicit scheme");
  }
  return du - du0;
}

}  // namespace snap
