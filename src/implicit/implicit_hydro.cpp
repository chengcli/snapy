// yaml
#include <yaml-cpp/yaml.h>

// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "implicit.hpp"
#include "implicit_dispatch.hpp"

namespace snap {

ImplicitHydroImpl::ImplicitHydroImpl(ImplicitOptions options_)
    : options(options_) {
  reset();
}

void ImplicitHydroImpl::reset() {
  // set up equation of state
  peos = register_module_op(this, "eos", options.eos());

  // register buffers
  wroe = register_buffer("wroe", torch::empty_like(peos->get_buffer("W")));
  groe = register_buffer("groe", torch::empty_like(wroe[IDN]));
  croe = register_buffer("croe", torch::empty_like(wroe[IDN]));
}

torch::Tensor ImplicitHydroImpl::diffusion_matrix(torch::Tensor wlr,
                                                  torch::Tensor elr, int dim) {
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(wroe.sizes(), /*squash_dims=*/0)
                  .add_output(wroe)
                  .add_input(wlr)
                  .add_input(elr)
                  .build();

  // IPR index is specific enthalpy + ke
  at::native::call_roe_average(wroe.device().type(), iter, dim);

  auto vec = groe.sizes().vec();
  vec.push_back(options.size());
  vec.push_back(options.size());

  auto Rmat = torch::empty(vec, torch::kFloat64);
  auto Rimat = torch::empty(vec, torch::kFloat64);
  auto EV = torch::empty(vec, torch::kFloat64);

  groe = peos->compute("W->A", {wroe});
  auto ke = peos->compute("W->K", {wroe});

  // specific enthalpy + ke -> pressure
  wroe[IPR] = (wroe[IPR] * wroe[IDN] - ke) * groe / (groe + 1.);
  croe = peos->compute("WA->L", {wroe, groe});
  auto ie = peos->compute("W->I", {wroe});

  iter = at::TensorIteratorConfig()
             .resize_outputs(false)
             .check_all_same_dtype(true)
             .add_output(Rmat)
             .add_output(Rimat)
             .add_output(EV)
             .add_input(wroe)
             .add_input(ie)
             .add_input(croe);

  at::native::call_eigen_system(wroe.device().type(), iter, dim);
  auto result = Rmat.matmul(EV.abs()).matmul(Rimat);

  if (options.scheme() == 1) {  // partial matrix
    // 0, 1, 4 are Indices for a 3x3 submatrix
    auto sub = torch::tensor(
        {0, 1, 4}, torch::dtype(torch::kLong).device(result.device()));
    return result.index_select(-2, sub).index_select(-1, sub);
  } else {  // full matrix
    return result;
  }
}

torch::Tensor ImplictHydro::flux_jacobian(torch::Tensor w, int dim) {
  auto gamma = peos->compute("W->A", {w});
  auto cs = peos->compute("WA->L", {w, gamma});

  auto vec = gamma.sizes().vec();
  vec.push_back(options.size());
  vec.push_back(options.size());

  auto dfdq = torch::empty(vec, torch::kFloat64);

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .add_output(dfdq)
                  .add_input(w)
                  .add_input(gamma)
                  .add_input(cs);

  // calculate flux jacobian
  at::native::call_flux_jacobian(dfdq.device().type(), iter, dim);

  if (options.scheme() == 1) {  // partial matrix
    auto sub = torch::tensor({0, 1, 4},
                             torch::dtype(torch::kLong).device(dfdq.device()));
    return dfdq.index_select(-2, sub).index_select(-1, sub);
  } else {
    return dfdq;
  }
}

torch::Tensor ImplicitHydroImpl::forward(torch::Tensor w, torch::Tensor wlr,
                                         torch::Tensor elr, int dim) {
  auto A = diffusion_matrix(wlr, elr, dim);
  auto B = flux_jacobian(w, dim);

  //// ------------ Assemble tridiagonal system ------------ ////
  auto pcoord = peos->pcoord;
  int xs = pcoord->is();
  int xe = pcoord->ie();

  auto aleft = pcoord->face_area1(xs, xe).unsqueeze(-1).unsqueeze(-1);
  auto aright = pcoord->face_area1(xs + 1, xe + 1).unsqueeze(-1).unsqueeze(-1);
  auto vol =
      pcoord->cell_volume().slice(-1, xs, xe).unsqueeze(-1).unsqueeze(-1);

  int d = dim - 1;

  auto vec = w[IDN].sizes().vec();
  int m = option.size();
  vec.push_back(m);
  vec.push_back(m);

  auto a = torch::zeros(vec, torch::kFloat64);
  auto b = torch::zeros(vec, torch::kFloat64);
  auto c = torch::zeros(vec, torch::kFloat64);

  a.slice(d, xs, xe) =
      (A.slice(d, xs, xe) * aleft + A.slice(d, xs + 1, xe + 1) * aright +
       (aright - aleft) * B.slice(d, xs, xe)) /
      (2. * vol);

  b.slice(d, xs, xe) =
      -(A.slice(d, xs - 1, xe - 1) + B.slice(d, xs - 1, xe - 1)) * aleft /
      (2. * vol);

  c.slice(d, xs, xe) =
      -(A.slice(d, xs + 1, xe + 1) - B.slice(d, xs + 1, xe + 1)) * aright /
      (2. * vol);

  return std::make_tuple(a, b, c);
}

}  // namespace snap
