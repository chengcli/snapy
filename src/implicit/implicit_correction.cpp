// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "implicit.hpp"
#include "implicit_dispatch.hpp"

namespace snap {

ImplicitCorrectionImpl::ImplicitCorrectionImpl(ImplicitOptions options_)
    : options(options_) {
  reset();
}

void ImplicitCorrectionImpl::reset() {
  pvic = register_module("vic", ImplicitHydro(options));
}

torch::Tensor ImplicitCorrectionImpl::forward(torch::Tensor du, torch::Tensor w,
                                              torch::Tensor gammar, double dt) {
  if (options.scheme() == 0) {  // null operation
    return torch::zeros_like(du);
  }

  auto vec = du.sizes().vec();
  vec.insert(vec.begin(), 2);

  //// -------- Vertical direction --------- ////
  auto wlr = torch::empty(vec, w.options());
  wlr[ILT] = w;
  wlr[IRT] = w.roll(-1, 3);

  auto gammal = gammar.roll(-1, 2);
  auto gamma = 0.5 * (gammar + gammal);

  auto [a, b, c, corr] = pvic->forward(w, gamma, wlr, 3);

  auto delta = torch::zeros_like(corr);

  int m = options.size();
  auto Dt = torch::eye(m, w.options()) / dt;
  auto Phi = torch::zeros({m, m}, w.options());

  Phi[Index::IVX][Index::IDN] = options.grav();
  Phi[m - 1][Index::IVX] = options.grav();

  auto Bnd = torch::eye(m, w.options());
  Bnd[Index::IVX][Index::IVX] = -1.;

  int is = pvic->pcoord->is();
  int ie = pvic->pcoord->ie();

  a.slice(2, is, ie + 1) += Dt - Phi;

  //// --------- Fix boundary condition ---------- ////
  a.select(2, is) += b.select(2, is).matmul(Bnd);
  a.select(2, ie) += c.select(2, ie).matmul(Bnd);

  //// -------- Solve block-tridiagonal matrix --------- ////
  int nc1 = pvic->pcoord->options.nc1();
  int nc2 = pvic->pcoord->options.nc2();
  int nc3 = pvic->pcoord->options.nc3();

  std::vector<int64_t> vec1 = {nc3, nc2, nc1, -1};

  auto du0 = du.clone();
  std::vector<int64_t> vec2 = {3, 0, 1, 2};
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(du.sizes(), /*squash_dims=*/{0, 3})
                  .add_output(du)
                  .add_input(w)
                  .add_owned_input(a.view(vec1).permute(vec2))
                  .add_owned_input(b.view(vec1).permute(vec2))
                  .add_owned_input(c.view(vec1).permute(vec2))
                  .add_owned_input(delta.view(vec1).permute(vec2))
                  .add_owned_input(corr.view(vec1).permute(vec2))
                  .build();

  if ((options.scheme() >> 3) & 1) {  // full
    at::native::vic_solve5(du.device().type(), iter, dt, is, ie);
  } else {  // partial
    at::native::vic_solve3(du.device().type(), iter, dt, is, ie);
  }

  return du - du0;
}

}  // namespace snap
