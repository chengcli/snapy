"""The EXACT curved O(dr^2) correction, general profiles.

Claim (derived):  with kappa = gamma/(gamma-1),
    dF = kappa [ sigma_c^2 * p * (ln(p/rho))' * u'   -   delta * d_r(p u) ]
    sigma_c^2 = area-weighted 2nd central moment (weight r)  = (h^2/12)(1-h^2/(12 rbar^2))
    delta     = r_v - r_c  (volume centroid minus area centroid) = h^2/(12 rbar) + O(h^4)
everything evaluated at r_c.

exact = kappa <p u>_A              (area average, weight r)
code  = kappa * pbar * mbar/rhobar, every bar = VOLUME average (weight r^2),
        which is what a finite-volume cell actually stores.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

GAMMA = 1.4
K = GAMMA / (GAMMA - 1.0)
NG = 90
xg, wg = leggauss(NG)

# generic smooth, strongly varying profiles; nothing linear, rho not constant
p = lambda r: 3.1e4 * np.exp(-(r - 1.0) / 0.37) * (1.0 + 0.11 * np.sin(2.3 * r))
rho = lambda r: 0.83 * np.exp(-(r - 1.0) / 0.52) * (1.0 + 0.07 * np.cos(1.7 * r))
u = lambda r: 0.29 * np.sin(3.1 * r) + 0.13 * r
m = lambda r: rho(r) * u(r)


def wavg(f, rm, rp, n):
    r = 0.5 * (rp + rm) + 0.5 * (rp - rm) * xg
    wt = wg * r ** n
    return float(np.sum(wt * f(r)) / np.sum(wt))


def rc_closed(rm, rp):
    rbar = 0.5 * (rm + rp)
    return (3 * rbar ** 2 + (rp - rm) ** 2 / 4.0) / (3 * rbar)


def rv_closed(rm, rp):
    return 0.75 * (rp ** 4 - rm ** 4) / (rp ** 3 - rm ** 3)


def sigma2_c(rm, rp):
    h, rbar = rp - rm, 0.5 * (rm + rp)
    return h ** 2 / 12.0 * (1.0 - h ** 2 / (12.0 * rbar ** 2))


def d(f, r, e=1e-7):
    return (f(r + e) - f(r - e)) / (2 * e)


for rbar in (1.4, 6.0, 50.0):
    print("=== rbar = %g ===" % rbar)
    print("  %-8s %-14s %-14s %-14s %-11s %-11s %-6s" %
          ("h", "exact-code", "cov only", "cov+centroid", "rel cov",
           "rel full", "ratio"))
    prev = None
    for h in (0.2, 0.1, 0.05, 0.025, 0.0125):
        rm, rp = rbar - h / 2, rbar + h / 2
        rc, rv, s2 = rc_closed(rm, rp), rv_closed(rm, rp), sigma2_c(rm, rp)
        delta = rv - rc
        exact = K * wavg(lambda r: p(r) * u(r), rm, rp, 1)
        code = K * wavg(p, rm, rp, 2) * wavg(m, rm, rp, 2) / wavg(rho, rm, rp, 2)
        truth = exact - code
        lnt = lambda r: np.log(p(r) / rho(r))
        cov = K * s2 * p(rc) * d(lnt, rc) * d(u, rc)
        cent = -K * delta * d(lambda r: p(r) * u(r), rc)
        r1 = abs(truth - cov) / abs(truth)
        r2 = abs(truth - cov - cent) / abs(truth)
        ratio = "" if prev is None else "%.2f" % (prev / r2)
        print("  %-8g %-14.6e %-14.6e %-14.6e %-11.3e %-11.3e %-6s"
              % (h, truth, cov, cov + cent, r1, r2, ratio))
        prev = r2
    rm, rp = rbar - 0.05, rbar + 0.05
    rc, rv = rc_closed(rm, rp), rv_closed(rm, rp)
    cov = sigma2_c(rm, rp) * p(rc) * d(lambda r: np.log(p(r) / rho(r)), rc) * d(u, rc)
    cent = -(rv - rc) * d(lambda r: p(r) * u(r), rc)
    print("  |centroid| / |covariance| at h=0.1 : %.3f" % abs(cent / cov))
    print("  delta quad-vs-closed  : %.12e  vs  %.12e"
          % (0.75 * (rp**4 - rm**4) / (rp**3 - rm**3) - (2/3) * (rp**3 - rm**3) / (rp**2 - rm**2),
             (rp - rm) ** 2 / (12 * 0.5 * (rm + rp))))
    print()
