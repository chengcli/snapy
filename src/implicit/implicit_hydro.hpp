#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/forcing/forcing.hpp>
#include <snap/hydro/gravity_work_radial.hpp>

// arg
#include <snap/utils/cache_lifecycle.hpp>

#include <snap/add_arg.h>

namespace snap {

struct ImplicitOptionsImpl {
  static std::shared_ptr<ImplicitOptionsImpl> create() {
    return std::make_shared<ImplicitOptionsImpl>();
  }
  static std::shared_ptr<ImplicitOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);
  static std::shared_ptr<ImplicitOptionsImpl> from_yaml(const YAML::Node& node);

  ImplicitOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- implicit hydro options --\n";
    os << "* type = " << type() << "\n"
       << "* scheme = " << scheme() << "\n"
       << "* advection_cfl = " << advection_cfl() << "\n"
       << "* shear_cfl = " << shear_cfl() << "\n";
  }

  int size() const {
    if ((scheme() >> 3) & 1) {  // full
      return 5;
    } else {
      return 3;
    }
  }

  std::string type() const;
  ADD_ARG(int, scheme) = 0;
  ADD_ARG(double, advection_cfl) = 1.0;
  ADD_ARG(double, shear_cfl) = 0.0;
};
using ImplicitOptions = std::shared_ptr<ImplicitOptionsImpl>;

class HydroImpl;

class ImplicitHydroImpl : public torch::nn::Cloneable<ImplicitHydroImpl> {
 public:
  //! Create and register a ImplicitHydro module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts      options for creating the `ImplicitHydro` module
   * \param[in] p         parent module for registering the created module
   * \param[in] name      name for the created module
   * \return              created `ImplicitHydro` module
   */
  static std::shared_ptr<ImplicitHydroImpl> create(
      ImplicitOptions const& opts, torch::nn::Module* p,
      std::string const& name = "icorr");

  //! options with which this `ImplicitHydro` was constructed
  ImplicitOptions options;

  //! non-owning pointer to parent
  HydroImpl const* phydro = nullptr;

  //! Constructor to initialize the layer
  ImplicitHydroImpl() : options(ImplicitOptionsImpl::create()) {}
  explicit ImplicitHydroImpl(ImplicitOptions const& options,
                             torch::nn::Module* p = nullptr);
  void reset() override;

  //! \brief Drop the grid-only factors cached from the coordinate.
  /*!
   * Both the radial-exact stencils of #308 (_rx_*) and the forward_masked
   * geometry of #309 (_g_*) are built from the coordinate's buffers, which
   * `Module::to` converts with `set_data`: identity and version are unchanged,
   * and a float64 -> float32 -> float64 round trip restores the dtype as well,
   * so neither key sees the conversion. See src/utils/cache_lifecycle.hpp.
   */
  void invalidate_cached_grid() {
    _rx_key = X1GridKey();
    _rx_lo.reset();
    _rx_mid.reset();
    _rx_hi.reset();
    _rx_work = CorrectedPEFactors();
    _g_key = GeoKey();
    _g_area.reset();
    _g_volume.reset();
    _g_work_lo.reset();
    _g_work_hi.reset();
    _g_mask0.reset();
    _g_phi.reset();
    _g_dp_lo.reset();
    _g_dp_hi.reset();
  }

  //! the coordinate is a sibling module, so a conversion requested on this
  //! module alone still has to drop the caches built from its grid
  SNAP_INVALIDATE_CACHES_ON_TO(torch::nn::Cloneable<ImplicitHydroImpl>,
                               invalidate_cached_grid())

  //! corrector for the implicit hydro
  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor gamma,
                        double dt);
  torch::Tensor forward_masked(torch::Tensor du, torch::Tensor w,
                               torch::Tensor gamma, double dt,
                               torch::Tensor solid);

  torch::Tensor correction() const { return _corr; }
  torch::Tensor mass_correction() const { return _mass_corr; }

  //! running max relative break of the VIC telescoping identity (clamp meter)
  torch::Tensor clamp_residual() const { return _clamp_residual; }

  //! 1 if the dry-gas clamp emptied a cell since the last reset, else 0
  torch::Tensor dry_clamp_step() const { return _dry_clamp_step; }
  void reset_dry_clamp_step() { _dry_clamp_step.zero_(); }

  //! Latched until the next step, including later RK stages. A failure
  //! found after the solve is latched on the device; this reads it back.
  bool solve_failed() const {
    return _solve_failed || (_failed_dev.defined() && _failed_dev.item<bool>());
  }
  void reset_solve_failure() {
    _solve_failed = false;
    if (_failed_dev.defined()) _failed_dev.zero_();
  }

  //! solve_failed() as a 0-d tensor (1 or 0) with `opts`, without a host read
  torch::Tensor solve_failed_tensor(torch::TensorOptions const& opts) const {
    auto flag = torch::full({}, _solve_failed ? 1. : 0., opts);
    return _failed_dev.defined() ? torch::maximum(flag, _failed_dev.to(opts))
                                 : flag;
  }

 private:
  void ensure_workspace(torch::Tensor const& w);
  bool _solve_failed = false;
  torch::Tensor _failed_dev;  // 0-d bool: a failure found after the solve

  torch::Tensor _a, _b, _c, _delta, _du0, _corr, _mass_corr, _clamp_residual,
      _dry_clamp_step;

  //! SNAP_GRAVITY_WORK_RADIAL_EXACT: the grid-only stencil rows and work
  //! factors, built once per grid (#291)
  X1GridKey _rx_key;
  torch::Tensor _rx_lo, _rx_mid, _rx_hi;
  CorrectedPEFactors _rx_work;

  // grid-only factors of forward_masked, rebuilt only
  // when a coordinate face tensor, the device, dtype, shape or grav1 changes
  //! The face tensors are held by reference, as every sibling cache does
  //! (HydroImpl::FaceMoment2/CellVolume, X1GridKey): a raw TensorImpl address
  //! can be handed back by the allocator for a different tensor, with the
  //! version counter of a fresh tensor back at 0, and the key would then
  //! compare equal to a stale entry.
  struct GeoKey {
    torch::Tensor f1, f2, f3;
    int64_t v1 = -1, v2 = -1, v3 = -1;
    torch::Device dev = torch::kCPU;
    c10::ScalarType st = torch::kDouble;
    std::vector<int64_t> shape;
    double grav1 = 0.;
    bool operator==(GeoKey const& o) const {
      return f1.defined() && o.f1.defined() && f1.is_same(o.f1) &&
             f2.is_same(o.f2) && f3.is_same(o.f3) && v1 == o.v1 &&
             v2 == o.v2 && v3 == o.v3 && dev == o.dev && st == o.st &&
             shape == o.shape && grav1 == o.grav1;
    }
  };
  GeoKey _g_key;
  torch::Tensor _g_area, _g_volume, _g_work_lo, _g_work_hi, _g_mask0, _g_phi,
      _g_dp_lo, _g_dp_hi;
};
TORCH_MODULE(ImplicitHydro);

}  // namespace snap

#undef ADD_ARG
