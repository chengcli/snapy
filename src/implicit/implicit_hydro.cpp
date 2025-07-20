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
                  .add_input(wlr[ILT])
                  .add_input(wlr[IRT])
                  .add_input(elr)
                  .build();

  // IPR index is specific enthalpy + ke
  at::native::call_roe_average(wroe.device().type(), iter);

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

  vec = {2, 3, 4, 0, 1};
  iter = at::TensorIteratorConfig()
             .resize_outputs(false)
             .check_all_same_dtype(true)
             .declare_static_shape(Rmat.sizes(),
                                   /*squash_dims=*/{wroe.dim(), wroe.dim() + 1})
             .add_output(Rmat)
             .add_output(Rimat)
             .add_output(EV)
             .add_owned_input(wroe.unsqueeze(0).permute(vec))
             .add_owned_input(ie.unsqueeze(0).unsqueeze(0).permute(vec))
             .add_owned_input(croe.unsqueeze(0).unsqueeze(0).permute(vec));

  at::native::call_eigen_system(wroe.device().type(), iter, dim);
  auto result = Rmat.matmul(EV.abs()).matmul(Rimat);

  if ((options.scheme() >>> 3) & 1) {  // full matrix
    return result;
  } else {  // partial matrix
    auto sub = torch::tensor(
        {IDN, IVX, IPR}, torch::dtype(torch::kLong).device(result.device()));
    return result.index_select(-2, sub).index_select(-1, sub);
  }
}

torch::Tensor ImplictHydro::flux_jacobian(torch::Tensor w, torch::Tensor gamma,
                                          torch::Tensor cs, int dim) {
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

  if ((options.scheme() >>> 3) & 1) {  // full matrix
    return dfdq;
  } else {  // partial matrix
    auto sub = torch::tensor({IDN, IVX, IPR},
                             torch::dtype(torch::kLong).device(dfdq.device()));
    return dfdq.index_select(-2, sub).index_select(-1, sub);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ImplicitHydroImpl::forward(torch::Tensor w, torch::Tensor wlr, int dim) {
  auto vec = w.sizes().vec();
  vec[0] = 2;

  auto elr = torch::empty(vec, w.options());
  elr[ILT] = peos->compute("W->I", {wlr[ILT]});
  elr[IRT] = peos->compute("W->I", {wlr[IRT]});

  auto A = diffusion_matrix(wlr, elr, dim);
  auto B = flux_jacobian(w, dim);

  //// ------------ Assemble tridiagonal system ------------ ////
  auto pcoord = peos->pcoord;
  int xs, xe;

  torch::Tensor aleft, aright, vol;

  switch (dim) {
    case 3:
      xs = pcoord->is();
      xe = pcoord->ie();

      aleft = pcoord->face_area1(xs, xe).unsqueeze(-1).unsqueeze(-1);
      aright = pcoord->face_area1(xs + 1, xe + 1).unsqueeze(-1).unsqueeze(-1);
      vol = pcoord->cell_volume().slice(2, xs, xe).unsqueeze(-1).unsqueeze(-1);
      break;
    case 2:
      xs = pcoord->js();
      xe = pcoord->je();

      aleft = pcoord->face_area2(xs, xe).unsqueeze(-1).unsqueeze(-1);
      aright = pcoord->face_area2(xs + 1, xe + 1).unsqueeze(-1).unsqueeze(-1);
      vol = pcoord->cell_volume().slice(1, xs, xe).unsqueeze(-1).unsqueeze(-1);
      break;
    case 1:
      xs = pcoord->ks();
      xe = pcoord->ke();

      aleft = pcoord->face_area3(xs, xe).unsqueeze(-1).unsqueeze(-1);
      aright = pcoord->face_area3(xs + 1, xe + 1).unsqueeze(-1).unsqueeze(-1);
      vol = pcoord->cell_volume().slice(0, xs, xe).unsqueeze(-1).unsqueeze(-1);
      break;
    default:
      TORCH_CHECK(false, "Wrong dimension");
  }

  auto a = torch::zeros_like(A);
  auto b = torch::zeros_like(A);
  auto c = torch::zeros_like(A);
  auto corr = torch::zeros_like(A.select(-1, 0));

  int d = dim - 1;
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

  return std::make_tuple(a, b, c, corr);
}

}  // namespace snap
