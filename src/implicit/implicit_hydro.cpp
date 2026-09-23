// yaml
#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "implicit_dispatch.hpp"
#include "implicit_hydro.hpp"

namespace snap {

ImplicitOptions ImplicitOptionsImpl::from_yaml(const std::string& filename,
                                               bool /*verbose*/) {
  auto config = YAML::LoadFile(filename);
  if (!config["integration"]) return nullptr;
  auto intg = config["integration"];
  auto op =
      intg["implicit-scheme"] ? from_yaml(intg["implicit-scheme"]) : nullptr;
  if (!op) {
    TORCH_CHECK(!intg["implicit-advection-cfl"] && !intg["shear-cfl"],
                "integration/implicit-advection-cfl and shear-cfl bound an "
                "implicit direction: set implicit-scheme or remove them");
    return nullptr;
  }
  {
    op->advection_cfl(
        config["integration"]["implicit-advection-cfl"].as<double>(1.0));
    TORCH_CHECK(op->advection_cfl() > 0.,
                "integration/implicit-advection-cfl must be > 0, got ",
                op->advection_cfl(), " (use a large value to relax the bound)");
    op->shear_cfl(config["integration"]["shear-cfl"].as<double>(0.0));
    TORCH_CHECK(op->shear_cfl() >= 0.,
                "integration/shear-cfl must be >= 0, got ", op->shear_cfl(),
                " (0 switches the bound off)");
  }
  return op;
}

ImplicitOptions ImplicitOptionsImpl::from_yaml(const YAML::Node& node) {
  int s = node.as<int>();
  // scheme 0 == "none": an implicit object that does nothing. Treat it as if
  // the `implicit-scheme` key were absent (return nullptr) so `implicit-scheme:
  // 0` is a true explicit spelling that also runs at nb1>1, instead of tripping
  // the nb1 guard on a phantom no-op object. picorr != null now faithfully
  // means "implicit is active".
  if (s == 0) return nullptr;
  auto op = ImplicitOptionsImpl::create();
  op->scheme(s);
  return op;
}

std::string ImplicitOptionsImpl::type() const {
  switch (scheme()) {
    case 0:
      return "none";
      break;
    case 1:
      return "vic-partial";
      break;
    case 9:
      return "vic-full";
      break;
    default:
      TORCH_CHECK(false, "Unsupported implicit scheme");
  }
}

ImplicitHydroImpl::ImplicitHydroImpl(ImplicitOptions const& options_,
                                     torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void ImplicitHydroImpl::reset() {
  TORCH_CHECK(phydro, "[ImplicitHydro] Parent Hydro is null");
  _a = register_buffer("a", torch::empty({0}, torch::kFloat64));
  _b = register_buffer("b", torch::empty({0}, torch::kFloat64));
  _c = register_buffer("c", torch::empty({0}, torch::kFloat64));
  _delta = register_buffer("delta", torch::empty({0}, torch::kFloat64));
  _du0 = register_buffer("du0", torch::empty({0}, torch::kFloat64));
  _corr = register_buffer("corr", torch::empty({0}, torch::kFloat64));
  _mass_corr = register_buffer("mass_corr", torch::empty({0}, torch::kFloat64));
  _clamp_residual =
      register_buffer("clamp_residual", torch::zeros({1}, torch::kFloat64));
}

void ImplicitHydroImpl::ensure_workspace(torch::Tensor const& w) {
  auto pcoord = phydro->pmb->pcoord;
  int nx1 = pcoord->options->nx1();
  int nx2 = pcoord->options->nx2();
  int nx3 = pcoord->options->nx3();
  int m = options->size();

  auto abc_shape = std::vector<int64_t>{1, nx3, nx2, nx1 * m * m};
  auto delta_shape = std::vector<int64_t>{1, nx3, nx2, nx1 * m};

  auto needs_reset = [&](torch::Tensor const& t,
                         std::vector<int64_t> const& shape) {
    return !t.defined() || t.sizes().vec() != shape ||
           t.scalar_type() != w.scalar_type() || t.device() != w.device();
  };

  auto maybe_resize = [&](torch::Tensor& t, std::vector<int64_t> const& shape) {
    if (needs_reset(t, shape)) {
      t.set_(torch::empty(shape, w.options()));
    }
  };

  maybe_resize(_a, abc_shape);
  maybe_resize(_b, abc_shape);
  maybe_resize(_c, abc_shape);
  maybe_resize(_delta, delta_shape);
  maybe_resize(_du0, w.sizes().vec());
  maybe_resize(_corr, w.sizes().vec());
  maybe_resize(_mass_corr, w.sizes().vec());
}

torch::Tensor ImplicitHydroImpl::forward(torch::Tensor du, torch::Tensor w,
                                         torch::Tensor gamma, double dt) {
  if (options->scheme() == 0) {  // null operation
    if (_corr.sizes() != du.sizes() ||
        _corr.scalar_type() != du.scalar_type() ||
        _corr.device() != du.device()) {
      _corr.set_(torch::zeros_like(du));
    } else {
      _corr.zero_();
    }
    return _corr;
  }

  TORCH_CHECK(phydro->options->grav(),
              "[ImplicitHydro] forcing does not have const-gravity");

  auto pcoord = phydro->pmb->pcoord;
  auto interior = phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false));
  auto cos_theta = pcoord->cosine_cell_kj;
  auto sin_theta = torch::sqrt(1.0 - cos_theta * cos_theta);

  /*if (torch::isnan(du.index(interior)).any().item<bool>()) {
    TORCH_CHECK(false, "[ImplicitHydro] NaN encountered before implicit solve");
  }*/

  ensure_workspace(w);
  _du0.copy_(du);
  _mass_corr.zero_();

  /// (1) Project to local orthonormal frame
  w[IVY] += w[IVZ] * cos_theta;
  w[IVZ] *= sin_theta;

  coord_vec_raise_(du.narrow(0, IVX, 3), cos_theta);
  pcoord->prim2local1_(du);

  //// -------- Solve block-tridiagonal matrix --------- ////
  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(true)
          .declare_static_shape(du.index(interior).sizes(),
                                /*squash_dims=*/{0, 3})
          .add_owned_output(du.index(interior))
          .add_owned_output(_mass_corr.index(interior))
          .add_owned_input(w.index(interior))
          .add_owned_input(gamma.unsqueeze(0).index(interior))
          .add_owned_input(
              pcoord->face_area1().unsqueeze(0).contiguous().index(interior))
          .add_owned_input(
              pcoord->cell_volume().unsqueeze(0).contiguous().index(interior))
          .add_input(_a)
          .add_input(_b)
          .add_input(_c)
          .add_input(_delta)
          .build();

  // Linearize the FULL gravity: du always carries it (body force + rho_grav
  // sum to grav1); scaling by non_hydrostatic() drops the gravity coupling
  // and destabilizes the solve at dt >> dt_acoustic whenever nh < 1.
  auto grav1 = phydro->options->grav()->grav1();

  if ((options->scheme() >> 3) & 1) {
    at::native::vic_assemble_full(du.device().type(), iter, dt, grav1, 0);
    at::native::vic_solve_full(du.device().type(), iter, dt, grav1, 0);
    at::native::vic_redistribute_full(du.device().type(), iter, dt, grav1, 0);
  } else {
    // Match the full-VIC pipeline: assemble coefficients, run the column
    // solve + reductions, then apply the per-cell redistribution map.
    at::native::vic_assemble_partial(du.device().type(), iter, dt, grav1, 0);
    at::native::vic_solve_partial(du.device().type(), iter, dt, grav1, 0);
    at::native::vic_redistribute_partial(du.device().type(), iter, dt, grav1,
                                         0);
  }

  // only the availability clamp breaks sum_ch MASS*VOL == M(i) - M(i+1)
  {
    int is = pcoord->il();
    int ie = pcoord->iu() + 1;
    int nyc = du.size(0) - ICY;
    auto cell = _mass_corr[IDN].clone();
    for (int n = 0; n < nyc; ++n) cell += _mass_corr[ICY + n];
    auto M = _mass_corr[IVX];
    auto Ml = M.slice(-1, is, ie);
    auto Mu = M.slice(-1, is + 1, ie + 1);
    auto lhs = (cell * pcoord->cell_volume()).slice(-1, is, ie);
    auto rhs = Ml - Mu;
    // per cell: a column max would hide a binding in a low-flux layer
    auto scale = torch::maximum(Ml.abs(), Mu.abs()).clamp_min(1.e-300);
    _clamp_residual.copy_(torch::maximum(
        _clamp_residual, ((lhs - rhs).abs() / scale).max().detach()));
  }

  /// (3) De-project from local orthonormal frame
  w[IVZ] /= sin_theta;
  w[IVY] -= w[IVZ] * cos_theta;
  pcoord->flux2global1_(du);

  // The implicit matrix linearizes gravity as cell-centred work,
  // dt*grav1*du[IVX]. Replace that contribution with work derived from the
  // same face mass transfer exported by the VIC redistribution.  MASS[IVX]
  // is the mass moved through the face below each cell during this stage;
  // the closed top face lives in the first outer ghost cell and remains zero.
  // Using the actually redistributed dry/species increments also keeps the
  // energy update consistent if constituent availability limiting is active.
  if (grav1 != 0.) {
    int is = pcoord->il();
    int ie = pcoord->iu() + 1;
    int ny = du.size(0) - ICY;

    auto mass_du = _mass_corr[IDN].slice(-1, is, ie).clone();
    if (ny > 0) {
      mass_du += _mass_corr.narrow(0, ICY, ny).sum(0).slice(-1, is, ie);
    }

    auto face_mass = _mass_corr[IVX];
    auto phi_face = -grav1 * pcoord->x1f;
    auto phi_cell = -grav1 * pcoord->x1v;
    auto volume = pcoord->cell_volume();
    auto potential_flux_div =
        (phi_face.slice(0, is + 1, ie + 1) *
             face_mass.slice(-1, is + 1, ie + 1) -
         phi_face.slice(0, is, ie) * face_mass.slice(-1, is, ie)) /
        volume.slice(-1, is, ie);
    auto face_gravity_work =
        -potential_flux_div - phi_cell.slice(0, is, ie) * mass_du;
    auto matrix_gravity_work = dt * grav1 * du[IVX].slice(-1, is, ie);
    du[IPR].slice(-1, is, ie) += face_gravity_work - matrix_gravity_work;
  }

  _corr.copy_(du);
  _corr.sub_(_du0);

  /*if (torch::isnan(du.index(interior)).any().item<bool>()) {
    TORCH_CHECK(false, "[ImplicitHydro] NaN encountered after implicit solve");
  }*/

  return _corr;
}

std::shared_ptr<ImplicitHydroImpl> ImplicitHydroImpl::create(
    ImplicitOptions const& opts, torch::nn::Module* p,
    std::string const& name) {
  TORCH_CHECK(p != nullptr, "[ImplicitHydro] Parent module is nullptr");
  TORCH_CHECK(opts != nullptr, "[ImplicitHydro] Options pointer is nullptr");

  return p->register_module(name, ImplicitHydro(opts, p));
}

}  // namespace snap
