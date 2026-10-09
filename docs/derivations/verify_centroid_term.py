"""Is there a SECOND O(dr^2) term on a curved horizontal face, beyond the
metric-weighted covariance?

The code's x2 energy flux is kappa * pbar * mbar/rhobar, where every barred
quantity is the CELL value, i.e. the cell's VOLUME average, which for a linear
profile sits at the volume centroid r_v (r^2 weight).  The exact FV flux is the
AREA average over the face, whose weight is r, with area centroid r_c.
r_v != r_c, so there is a centroid mismatch of order dr^2/r.

Clean test state (so the discrete state is the EXACT volume average and no IC
error pollutes anything):
    rho = const            -> cell value exact, <m>/<rho> = u exactly
    p   = linear in r      -> cell value = p(r_v) exactly
    u   = linear in r      -> cell value = u(r_v) exactly
Then
    exact  = <p u>_A              (weight r, quadrature)
    code   = p(r_v) u(r_v)        (weight r^2 centroid)
    claim  = sigma^2 p' u' - (r_v - r_c) * d_r(p u)|_{r_c}
with sigma^2 = (h^2/12)(1 - h^2/(12 rbar^2)) and r_c, r_v the closed forms.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

NG = 80
xg, wg = leggauss(NG)


def wavg(f, rm, rp, n):
    """area/volume average of f with weight r^n on [rm, rp], by quadrature"""
    r = 0.5 * (rp + rm) + 0.5 * (rp - rm) * xg
    wt = wg * r ** n
    return float(np.sum(wt * f(r)) / np.sum(wt))


def centroid(rm, rp, n):
    r = 0.5 * (rp + rm) + 0.5 * (rp - rm) * xg
    wt = wg * r ** n
    return float(np.sum(wt * r) / np.sum(wt))


def rc_closed(rm, rp):          # area centroid, weight r
    rbar = 0.5 * (rm + rp)
    return (3 * rbar ** 2 + (rp - rm) ** 2 / 4.0) / (3 * rbar)


def rv_closed(rm, rp):          # volume centroid, weight r^2 (snapy x1v)
    return 0.75 * (rp ** 4 - rm ** 4) / (rp ** 3 - rm ** 3)


def sigma2_closed(rm, rp):      # area-weighted 2nd central moment, weight r
    h = rp - rm
    rbar = 0.5 * (rm + rp)
    return h ** 2 / 12.0 * (1.0 - h ** 2 / (12.0 * rbar ** 2))


# linear p and u, constant rho
P0, PSL = 3.0e4, -4.1e3        # p  = P0 + PSL*(r-1)
U0, USL = 0.37, 0.58           # u  = U0 + USL*(r-1)
p = lambda r: P0 + PSL * (r - 1.0)
u = lambda r: U0 + USL * (r - 1.0)

print("closed-form centroids vs quadrature (rbar=1.4, h=0.2):")
rm, rp = 1.4 - 0.1, 1.4 + 0.1
print("  r_c quad=%.14f closed=%.14f" % (centroid(rm, rp, 1), rc_closed(rm, rp)))
print("  r_v quad=%.14f closed=%.14f" % (centroid(rm, rp, 2), rv_closed(rm, rp)))
print("  r_v - r_c = %.6e   h^2/(12 rbar) = %.6e"
      % (rv_closed(rm, rp) - rc_closed(rm, rp), 0.04 / (12 * 1.4)))
print()

for rbar in (1.4, 20.0):
    print("=== rbar = %g ===" % rbar)
    print("  %-9s %-14s %-14s %-14s %-11s %-11s %-9s" %
          ("h", "exact-code", "cov only", "cov+centroid", "resid/cov",
           "resid full", "ratio"))
    prev = None
    for h in (0.2, 0.1, 0.05, 0.025, 0.0125):
        rm, rp = rbar - h / 2, rbar + h / 2
        rc, rv = rc_closed(rm, rp), rv_closed(rm, rp)
        s2 = sigma2_closed(rm, rp)
        exact = wavg(lambda r: p(r) * u(r), rm, rp, 1)
        code = p(rv) * u(rv)
        truth = exact - code
        cov = s2 * PSL * USL
        dpu = PSL * u(rc) + p(rc) * USL           # d_r(p u) at r_c
        full = cov - (rv - rc) * dpu
        r1 = abs(truth - cov) / abs(truth)
        r2 = abs(truth - full) / abs(truth)
        ratio = "" if prev is None else "%.2f" % (prev / r2)
        print("  %-9g %-14.6e %-14.6e %-14.6e %-11.3e %-11.3e %-9s"
              % (h, truth, cov, full, r1, r2, ratio))
        prev = r2
    print("  size of centroid term / covariance term at h=0.2: %.3f"
          % abs((rv_closed(rbar - .1, rbar + .1) - rc_closed(rbar - .1, rbar + .1))
                * (PSL * u(rbar) + p(rbar) * USL)
                / (sigma2_closed(rbar - .1, rbar + .1) * PSL * USL)))
    print()
