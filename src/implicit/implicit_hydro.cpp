// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/hydro.hpp>

#include "implicit_dispatch.hpp"
#include "implicit_hydro.hpp"

namespace snap {

ImplicitHydroImpl::ImplicitHydroImpl(ImplicitOptions const& options_,
                                     torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void ImplicitHydroImpl::reset() {
  TORCH_CHECK(phydro, "[ImplicitHydro] phydro is nullptr");
}

torch::Tensor ImplicitHydroImpl::forward(torch::Tensor du, torch::Tensor w,
                                         torch::Tensor gamma, double dt) {
  if (options->scheme() == 0) {  // null operation
    return torch::zeros_like(du);
  }

  auto pcoord = phydro->pcoord;
  auto cos_theta = pcoord->cosine_cell_kj;
  auto sin_theta = torch::sqrt(1.0 - cos_theta * cos_theta);

  auto du0 = du.clone();

  /// (1) Project to local orthonormal frame
  w[IVY] += w[IVZ] * cos_theta;
  w[IVZ] *= sin_theta;

  // coord_vec_raise_(du.narrow(0, IVX, 3), cos_theta);
  // pcoord->prim2local1_(du);

  //// -------- Solve block-tridiagonal matrix --------- ////
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();

  int is = pcoord->is();
  int ie = pcoord->ie();

  int m = options->size();
  auto a = torch::zeros({1, nc3, nc2, nc1 * m * m}, w.options());
  auto b = torch::zeros_like(a);
  auto c = torch::zeros_like(a);
  auto delta = torch::zeros({1, nc3, nc2, nc1 * m}, w.options());

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(du.sizes(), /*squash_dims=*/{0, 3})
                  .add_output(du)
                  .add_input(w)
                  .add_owned_input(gamma.unsqueeze(0))
                  .add_owned_input(pcoord->face_area1().unsqueeze(0))
                  .add_owned_input(pcoord->cell_volume().unsqueeze(0))
                  .add_input(a)
                  .add_input(b)
                  .add_input(c)
                  .add_input(delta)
                  .build();

  if ((options->scheme() >> 3) & 1) {
    at::native::vic_solve_full(du.device().type(), iter, dt,
                               options->grav()->grav1(), is, ie, 0);
  } else {
    at::native::vic_solve_partial(du.device().type(), iter, dt,
                                  options->grav()->grav1(), is, ie, 0);
  }

  /// (3) De-project from local orthonormal frame
  w[IVZ] /= sin_theta;
  w[IVY] -= w[IVZ] * cos_theta;
  // pcoord->flux2global1_(du);

  return du - du0;
}

std::shared_ptr<ImplicitHydroImpl> ImplicitHydroImpl::create(
    ImplicitOptions const& opts, torch::nn::Module* p,
    std::string const& name) {
  TORCH_CHECK(opts != nullptr, "ImplicitHydro options is nullptr");
  TORCH_CHECK(p != nullptr, "Parent module is nullptr");
  return p->register_module(name, ImplicitHydro(opts, p));
}

}  // namespace snap
