// snap
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
  auto avail = u.relu() * pcoord->cell_volume();
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
    auto f = flux1.slice(DIM1, il, iu + 2);            // faces il..iu+1
    auto th_lo = theta.slice(DIM1, il - 1, iu + 1);    // donor when f > 0
    auto th_hi = theta.slice(DIM1, il, iu + 2);        // donor when f <= 0
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

}  // namespace snap
