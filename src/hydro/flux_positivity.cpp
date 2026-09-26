// C/C++
#include <limits>

// snap
#include <snap/snap.h>

#include "flux_positivity.hpp"

namespace snap {

torch::Tensor flux_positivity_theta(torch::Tensor const& u,
                                    torch::Tensor const& flux1,
                                    torch::Tensor const& flux2,
                                    torch::Tensor const& flux3,
                                    Coordinate const& pcoord, double dt) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };

  auto out = torch::zeros_like(u);

  // Accumulate the outgoing A*F over exactly the faces the divergence uses
  // (lower faces il..iu+1 per dimension; see CoordinateImpl::divergence).
  // For cell i, the upper face is index i+1 and drains the cell where the
  // flux is positive; the lower face is index i and drains it where the flux
  // is negative.
  if (flux1.defined()) {
    int il = pcoord->il(), iu = pcoord->iu();
    int nf = iu - il + 2;  // faces il..iu+1
    auto af = pcoord->face_area1(il, il + nf) * flux1.slice(DIM1, il, il + nf);
    out.slice(DIM1, il, iu + 1) +=
        af.slice(DIM1, 1, nf).relu() + af.slice(DIM1, 0, nf - 1).neg().relu();
  }

  if (flux2.defined()) {
    int jl = pcoord->jl(), ju = pcoord->ju();
    int nf = ju - jl + 2;
    auto af = pcoord->face_area2(jl, jl + nf) * flux2.slice(DIM2, jl, jl + nf);
    out.slice(DIM2, jl, ju + 1) +=
        af.slice(DIM2, 1, nf).relu() + af.slice(DIM2, 0, nf - 1).neg().relu();
  }

  if (flux3.defined()) {
    int kl = pcoord->kl(), ku = pcoord->ku();
    int nf = ku - kl + 2;
    auto af = pcoord->face_area3(kl, kl + nf) * flux3.slice(DIM3, kl, kl + nf);
    out.slice(DIM3, kl, ku + 1) +=
        af.slice(DIM3, 1, nf).relu() + af.slice(DIM3, 0, nf - 1).neg().relu();
  }

  // theta = min(1, avail / (dt*out)) where out > 0; 1 elsewhere (in
  // particular in all ghost cells, whose outflow is not accumulated above --
  // their true factors arrive via the caller's ghost fill).
  // Stop 4096 ulp short of zero: an exact-zero target rounds negative.
  double eps = u.scalar_type() == torch::kFloat
                   ? std::numeric_limits<float>::epsilon()
                   : std::numeric_limits<double>::epsilon();
  double margin = 4096. * eps;
  auto avail = u.relu() * pcoord->cell_volume() * (1. - margin);
  auto drain = out.mul_(dt);
  return torch::where(drain > 0.,
                      (avail / drain.clamp_min(1e-300)).clamp_max(1.0),
                      torch::ones_like(u));
}

void flux_positivity_scale_(torch::Tensor const& theta,
                            torch::Tensor const& flux1,
                            torch::Tensor const& flux2,
                            torch::Tensor const& flux3,
                            Coordinate const& pcoord) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };

  if (flux1.defined()) {
    int il = pcoord->il(), iu = pcoord->iu();
    auto f = flux1.slice(DIM1, il, iu + 2);          // faces il..iu+1
    auto th_lo = theta.slice(DIM1, il - 1, iu + 1);  // donor when f > 0
    auto th_hi = theta.slice(DIM1, il, iu + 2);      // donor when f <= 0
    f.mul_(torch::where(f > 0., th_lo, th_hi));
  }

  if (flux2.defined()) {
    int jl = pcoord->jl(), ju = pcoord->ju();
    auto f = flux2.slice(DIM2, jl, ju + 2);
    auto th_lo = theta.slice(DIM2, jl - 1, ju + 1);
    auto th_hi = theta.slice(DIM2, jl, ju + 2);
    f.mul_(torch::where(f > 0., th_lo, th_hi));
  }

  if (flux3.defined()) {
    int kl = pcoord->kl(), ku = pcoord->ku();
    auto f = flux3.slice(DIM3, kl, ku + 2);
    auto th_lo = theta.slice(DIM3, kl - 1, ku + 1);
    auto th_hi = theta.slice(DIM3, kl, ku + 2);
    f.mul_(torch::where(f > 0., th_lo, th_hi));
  }
}

void flux_positivity_carry_(torch::Tensor const& theta,
                            torch::Tensor const& hspec,
                            torch::Tensor const& vel,
                            torch::Tensor const& flux1,
                            torch::Tensor const& flux2,
                            torch::Tensor const& flux3,
                            Coordinate const& pcoord) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  int ny = theta.size(0);

  auto carry = [&](torch::Tensor const& flux, int dim, int lo, int hi) {
    if (!flux.defined()) return;
    auto f = flux.narrow(0, ICY, ny).slice(dim, lo, hi + 2);  // faces lo..hi+1
    auto up = f > 0.;  // the donor is the lower cell
    auto donor = [&](torch::Tensor const& x) {
      return torch::where(up, x.slice(dim, lo - 1, hi + 1),
                          x.slice(dim, lo, hi + 2));
    };
    auto dm = (1. - donor(theta)) * f;
    flux[IPR].slice(dim - 1, lo, hi + 2) -= (dm * donor(hspec)).sum(0);
    flux.narrow(0, IVX, 3).slice(dim, lo, hi + 2) -=
        (dm * up).sum(0) * vel.slice(dim, lo - 1, hi + 1) +
        (dm * up.logical_not()).sum(0) * vel.slice(dim, lo, hi + 2);
  };

  carry(flux1, DIM1, pcoord->il(), pcoord->iu());
  carry(flux2, DIM2, pcoord->jl(), pcoord->ju());
  carry(flux3, DIM3, pcoord->kl(), pcoord->ku());
}

}  // namespace snap
