"""Energy-row covariance under the FAVRE cell values (ideal_moist_impl.h:40/:44).

Cell storage, per volume:   rho, rho*u, rho*q, I (+p -> rho*h = I + p)
Cell primitives, per mass:  u = <rho u>/<rho>,  q = <rho q>/<rho>,
                            h = <rho h>/<rho>   (because rho h = I + p is itself
                            a volume-averaged conserved-like density)
So u, q AND h are all Favre. The solver's energy flux is rho_bar * h_bar * u_bar
= <rho u> * <rho h> / <rho>, while the exact FV flux is <rho u h>_A.

Discrepancy = <rho>_A * ( {u h} - {u}{h} )  = rho * sigma^2 * u' h'   (Favre cov)

Candidate A (Xi)   : rho * s2 * D1[h] * D1[u]
Candidate B (coded): (I + p) * d1 ln(p/rho) * d1 u      [hydro_forward.cpp:131-133]
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

NG = 90
xg, wg = leggauss(NG)

rho = lambda r: 0.83 * np.exp(-(r - 1.) / 0.52) * (1 + 0.07 * np.cos(1.7 * r))
p   = lambda r: 3.1e4 * np.exp(-(r - 1.) / 0.37) * (1 + 0.11 * np.sin(2.3 * r))
u   = lambda r: 0.29 * np.sin(3.1 * r) + 0.13 * r


def make(kappa_var):
    """kappa = (I+p)/p ; constant for a dry ideal gas, varying with composition."""
    if kappa_var:
        kap = lambda r: 3.5 * (1 + 0.18 * np.sin(2.0 * r))
    else:
        kap = lambda r: 3.5 * np.ones_like(np.asarray(r, dtype=float))
    h = lambda r: kap(r) * p(r) / rho(r)        # specific enthalpy
    return kap, h


def avg(f, rm, rp, n):
    r = .5 * (rp + rm) + .5 * (rp - rm) * xg
    w = wg * r ** n
    return float(np.sum(w * f(r)) / np.sum(w))


def rc(rm, rp):
    rb = .5 * (rm + rp); return (12 * rb * rb + (rp - rm) ** 2) / (12 * rb)


def rv(rm, rp):
    return .75 * (rp ** 4 - rm ** 4) / (rp ** 3 - rm ** 3)


def s2(rm, rp):
    h_, rb = rp - rm, .5 * (rm + rp)
    return h_ * h_ / 12. * (1 - h_ * h_ / (12 * rb * rb))


def d(f, r, e=1e-7):
    return (f(r + e) - f(r - e)) / (2 * e)


for kappa_var in (False, True):
    kap, h = make(kappa_var)
    tag = "kappa VARIES (moist / composition gradient)" if kappa_var else "kappa CONSTANT (dry ideal gas)"
    for geom, nface, ncell in (("CARTESIAN (uniform weight, delta=0)", 0, 0),
                               ("CURVED (face r, cell r^2)", 1, 2)):
        print(f"=== {tag} | {geom} ===")
        prev_a = prev_b = None
        for hh in (0.2, 0.1, 0.05, 0.025, 0.0125):
            rbar = 1.4
            rm, rp = rbar - hh / 2, rbar + hh / 2
            C = rc(rm, rp) if nface else .5 * (rm + rp)
            S = s2(rm, rp) if nface else hh * hh / 12.
            D = (rv(rm, rp) - rc(rm, rp)) if nface else 0.0
            rhou = lambda r: rho(r) * u(r)
            rhoh = lambda r: rho(r) * h(r)
            rhouh = lambda r: rho(r) * u(r) * h(r)
            exact = avg(rhouh, rm, rp, nface)
            code = avg(rhou, rm, rp, ncell) * avg(rhoh, rm, rp, ncell) / avg(rho, rm, rp, ncell)
            truth = exact - code
            cent = -D * d(rhouh, C)
            A = S * rho(C) * d(h, C) * d(u, C) + cent               # Xi
            B = S * (kap(C) * p(C)) * d(lambda r: np.log(p(r) / rho(r)), C) * d(u, C) + cent  # coded
            ra = abs(truth - A) / abs(truth)
            rb_ = abs(truth - B) / abs(truth)
            oa = "" if prev_a is None else f"{prev_a/ra:5.2f}"
            ob = "" if prev_b is None else f"{prev_b/rb_:5.2f}"
            print(f"  h={hh:<7g} truth {truth:+.6e}  relA(Xi) {ra:.3e} [{oa}]"
                  f"   relB(coded) {rb_:.3e} [{ob}]")
            prev_a, prev_b = ra, rb_
        # the algebraic difference A - B at the finest level
        rm, rp = rbar - 0.0125 / 2, rbar + 0.0125 / 2
        C = rc(rm, rp) if nface else .5 * (rm + rp); S = s2(rm, rp) if nface else 0.0125**2/12.
        dif = S * d(u, C) * (rho(C) * d(h, C) - kap(C) * p(C) * d(lambda r: np.log(p(r)/rho(r)), C))
        pred = S * d(u, C) * d(kap, C) * p(C)
        print(f"  A - B = {dif:+.6e}   predicted s2*d1u*p*d1(kappa) = {pred:+.6e}")
        print()
