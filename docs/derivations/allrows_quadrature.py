"""Per-row O(dr^2) correction on a curved x2/x3 face, by quadrature.

General rule derived for a face flux the code forms as a product of CELL values:
    <a b ...>_A  -  abar bbar ...  =  sigma_c^2 * (sum over distinct pairs of
                                      first derivatives)  -  delta * d_1(flux)
with sigma_c^2 the AREA-weighted 2nd central moment (weight r) and
delta = r_v - r_c the volume-minus-area centroid offset.
Exact = area average (weight r). Code = product of volume averages (weight r^2).
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

NG = 90
xg, wg = leggauss(NG)

# generic smooth profiles, nothing linear, nothing constant
rho = lambda r: 0.83 * np.exp(-(r - 1.) / 0.52) * (1 + 0.07 * np.cos(1.7 * r))
p   = lambda r: 3.1e4 * np.exp(-(r - 1.) / 0.37) * (1 + 0.11 * np.sin(2.3 * r))
un  = lambda r: 0.29 * np.sin(3.1 * r) + 0.13 * r          # face-normal velocity
ut  = lambda r: 0.21 * np.cos(2.2 * r) - 0.08 * r          # a tangential velocity
qv  = lambda r: 0.011 * (1 + 0.6 * np.sin(1.9 * r))        # vapour mixing ratio
qc  = lambda r: 0.004 * (1 + 0.5 * np.cos(2.6 * r))        # condensate


def avg(f, rm, rp, n):
    r = .5 * (rp + rm) + .5 * (rp - rm) * xg
    w = wg * r ** n
    return float(np.sum(w * f(r)) / np.sum(w))


def rc(rm, rp):
    rb = .5 * (rm + rp)
    return (12 * rb * rb + (rp - rm) ** 2) / (12 * rb)


def rv(rm, rp):
    return .75 * (rp ** 4 - rm ** 4) / (rp ** 3 - rm ** 3)


def s2(rm, rp):
    h, rb = rp - rm, .5 * (rm + rp)
    return h * h / 12. * (1 - h * h / (12 * rb * rb))


def d(f, r, e=1e-7):
    return (f(r + e) - f(r - e)) / (2 * e)


# row: (name, exact integrand, code product factors, covariance pairs)
ROWS = [
    ("mass            rho*un",
     lambda r: rho(r) * un(r), (rho, un), [(rho, un)]),
    ("momentum-normal rho*un*un + p",
     lambda r: rho(r) * un(r) ** 2 + p(r), (rho, un, un), None),
    ("momentum-tang   rho*un*ut",
     lambda r: rho(r) * un(r) * ut(r), (rho, un, ut), [(rho, un), (rho, ut), (un, ut)]),
    ("tracer vapour   rho*un*qv",
     lambda r: rho(r) * un(r) * qv(r), (rho, un, qv), [(rho, un), (rho, qv), (un, qv)]),
    ("tracer condens. rho*un*qc",
     lambda r: rho(r) * un(r) * qc(r), (rho, un, qc), [(rho, un), (rho, qc), (un, qc)]),
]

for rbar in (1.4, 6.0):
    print(f"================ rbar = {rbar} ================")
    for name, exact_f, factors, pairs in ROWS:
        print(f"  {name}")
        prev = None
        for h in (0.2, 0.1, 0.05, 0.025, 0.0125):
            rm, rp = rbar - h / 2, rbar + h / 2
            C, Vc, S, D = rc(rm, rp), rv(rm, rp), s2(rm, rp), rv(rm, rp) - rc(rm, rp)
            exact = avg(exact_f, rm, rp, 1)                 # area average, weight r
            code = np.prod([avg(f, rm, rp, 2) for f in factors])   # volume averages
            if name.startswith("momentum-normal"):
                code = code + avg(p, rm, rp, 2)             # + pbar
            truth = exact - code
            # covariance part
            if pairs is None:   # rho*un*un : pairs (rho,un) twice and (un,un)
                cov = S * (2 * d(rho, C) * d(un, C) * un(C) + rho(C) * d(un, C) ** 2)
            else:
                cov = 0.0
                for a, b in pairs:
                    others = [f for f in factors if f is not a and f is not b]
                    # product of the remaining factors at the centroid
                    rest = np.prod([f(C) for f in others]) if others else 1.0
                    cov += S * d(a, C) * d(b, C) * rest
            cent = -D * d(exact_f, C)
            rel_cov = abs(truth - cov) / abs(truth)
            rel_full = abs(truth - cov - cent) / abs(truth)
            ratio = "" if prev is None else f"{prev/rel_full:5.2f}"
            print(f"    h={h:<7g} truth {truth:+.5e}  rel(cov only) {rel_cov:.3e}"
                  f"  rel(cov+centroid) {rel_full:.3e}  ratio {ratio}")
            prev = rel_full
    print()
