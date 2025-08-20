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
  // set up coordinate
  pcoord = register_module_op(this, "coord", options.coord());
}

torch::Tensor ImplicitHydroImpl::diffusion_matrix(torch::Tensor wlr,
                                                  torch::Tensor gamma,
                                                  int dim) {
  int nc1 = pcoord->options.nc1();
  int nc2 = pcoord->options.nc2();
  int nc3 = pcoord->options.nc3();

  auto wroe = torch::empty({5, nc3, nc2, nc1}, wlr.options());

  auto iter1 = at::TensorIteratorConfig()
                   .resize_outputs(false)
                   .check_all_same_dtype(true)
                   .declare_static_shape(wroe.sizes(), /*squash_dims=*/0)
                   .add_output(wroe)
                   .add_owned_input(wlr[ILT])
                   .add_owned_input(wlr[IRT])
                   .add_owned_input(gamma.unsqueeze(0))
                   .build();

  at::native::call_roe_average(wroe.device().type(), iter1);

  auto Rmat = torch::empty({nc3, nc2, nc1, 25}, wlr.options());
  auto Rimat = torch::empty({nc3, nc2, nc1, 25}, wlr.options());
  auto EV = torch::empty({nc3, nc2, nc1, 25}, wlr.options());

  std::vector<int64_t> vec = {3, 0, 1, 2};
  auto iter2 = at::TensorIteratorConfig()
                   .resize_outputs(false)
                   .check_all_same_dtype(true)
                   .declare_static_shape(wroe.sizes(), /*squash_dims=*/0)
                   .add_owned_output(Rmat.permute(vec))
                   .add_owned_output(Rimat.permute(vec))
                   .add_owned_output(EV.permute(vec))
                   .add_input(wroe)
                   .add_owned_input(gamma.unsqueeze(-1))
                   .build();

  at::native::call_eigen_system(wroe.device().type(), iter2, dim);

  // resize 25 -> 5x5
  Rmat = Rmat.view({nc3, nc2, nc1, 5, 5});
  Rimat = Rimat.view({nc3, nc2, nc1, 5, 5});
  EV = EV.view({nc3, nc2, nc1, 5, 5});

  auto result = Rmat.matmul(EV.abs()).matmul(Rimat);

  if ((options.scheme() >> 3) & 1) {  // full matrix
    return result;
  } else {  // partial matrix
    auto sub = torch::tensor(
        {IDN, IVX, IPR}, torch::dtype(torch::kLong).device(result.device()));
    return result.index_select(-2, sub).index_select(-1, sub);
  }
}

torch::Tensor ImplicitHydroImpl::flux_jacobian(torch::Tensor w,
                                               torch::Tensor gamma, int dim) {
  int nc1 = pcoord->options.nc1();
  int nc2 = pcoord->options.nc2();
  int nc3 = pcoord->options.nc3();

  auto dfdq = torch::empty({nc3, nc2, nc1, 25}, w.options());

  std::vector<int64_t> vec = {1, 2, 3, 0};
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(dfdq.sizes(), /*squash_dims=*/3)
                  .add_output(dfdq)
                  .add_owned_input(w.permute(vec))
                  .add_owned_input(gamma.unsqueeze(0).permute(vec))
                  .build();

  // calculate flux jacobian
  at::native::call_flux_jacobian(dfdq.device().type(), iter, dim);

  // resize 25 -> 5x5
  dfdq = dfdq.view({nc3, nc2, nc1, 5, 5});

  if ((options.scheme() >> 3) & 1) {  // full matrix
    return dfdq;
  } else {  // partial matrix
    auto sub = torch::tensor({IDN, IVX, IPR},
                             torch::dtype(torch::kLong).device(dfdq.device()));
    return dfdq.index_select(-2, sub).index_select(-1, sub);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ImplicitHydroImpl::forward(torch::Tensor w, torch::Tensor gamma,
                           torch::Tensor wlr, int dim) {
  auto A = diffusion_matrix(wlr, gamma, dim);
  auto B = flux_jacobian(w, gamma, dim);

  //// ------------ Assemble tridiagonal system ------------ ////
  int xs, xe;

  torch::Tensor aleft, aright, vol;

  switch (dim) {
    case 3:
      xs = pcoord->is();
      xe = pcoord->ie() + 1;

      aleft = pcoord->face_area1(xs, xe).unsqueeze(-1).unsqueeze(-1);
      aright = pcoord->face_area1(xs + 1, xe + 1).unsqueeze(-1).unsqueeze(-1);
      vol = pcoord->cell_volume().slice(2, xs, xe).unsqueeze(-1).unsqueeze(-1);
      break;
    case 2:
      xs = pcoord->js();
      xe = pcoord->je() + 1;

      aleft = pcoord->face_area2(xs, xe).unsqueeze(-1).unsqueeze(-1);
      aright = pcoord->face_area2(xs + 1, xe + 1).unsqueeze(-1).unsqueeze(-1);
      vol = pcoord->cell_volume().slice(1, xs, xe).unsqueeze(-1).unsqueeze(-1);
      break;
    case 1:
      xs = pcoord->ks();
      xe = pcoord->ke() + 1;

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
