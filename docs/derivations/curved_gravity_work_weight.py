#!/usr/bin/env python3
"""Replica of snapy's x1 face-form gravity work on a spherical-polar radial grid.

Per steradian: A(r) = r^2, V = (r+^3 - r-^3)/3, r_c = x1v = 3/4 (r+^4 - r-^4)/(r+^3 - r-^3)
(src/coord/spherical_polar.cpp radial_centers). grav1 < 0, phi = -grav1 r.

Weights per unit volume on (F+, F-), cell work = grav1 (a F+ + b F-):
  face     a = A+ (r+ - r_c)/V,       b = A- (r_c - r-)/V      (snapy 8cea3ae, hydro_forward.cpp:676-692)
  exact    a = (r_c - r-)/h,          b = (r+ - r_c)/h         (unique 2-point weights exact for F in {1, r}
                                                                 on the r^2 measure)
  cons(s)  a = A+ (s+ - r_c)/V,       b = A- (r_c - s-)/V      (face form with face potential at s_f)

Checks
 1. sympy: per-face weight sum minus the discrete-PE requirement A_f (r_c,i+1 - r_c,i), exact closed form.
 2. numpy: closed-wall column, one step: (dE + dPE) / dt, relative to the step's sum |work|, and relative
    to E+PE with E = PE (order of magnitude of a real column).
 3. numpy: booked work vs exact r^2-measure cell average on a smooth profile, h-ladder at fixed R and
    R-ladder at fixed h/H; reports the error with and without the common Cartesian (h^2/12) F'' piece.
Usage: python curved_gravity_work_weight.py   (prints every number in curved-gravity-work-weight.md)
"""

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss

OUT = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)

# ---------------------------------------------------------------- 1. sympy closed forms
R, h = sp.symbols("R h", positive=True)
def cell(rm, rp):
    V = (rp**3 - rm**3) / 3
    rc = sp.Rational(3, 4) * (rp**4 - rm**4) / (rp**3 - rm**3)
    return V, rc
Vi, rci = cell(R - h, R)        # cell below face f (face at r = R)
Vj, rcj = cell(R, R + h)        # cell above
need = R**2 * (rcj - rci)       # sum the two cells must book on F_f for E+PE to telescope
S_face = R**2 * (R - rci) + R**2 * (rcj - R)
S_exact = Vi * (rci - (R - h)) / h + Vj * ((R + h) - rcj) / h
say("== 1. per-face weight sum S_f minus need A_f (r_c,i+1 - r_c,i), uniform h, face at R (per sr, x grav1 F_f) ==")
say("face :", sp.simplify(S_face - need))
dex = sp.simplify(S_exact - need)
say("exact:", dex)
say("exact, series in h:", sp.series(dex, h, 0, 6).removeO().expand())
say("exact, relative to need, series:", sp.series(sp.simplify(dex / need), h, 0, 5).removeO().expand())
# leading error terms of the face form (face - exact average), F = F0 + F1 s + F2 s^2/2 about rbar
s, F0, F1, F2, rb = sp.symbols("s F0 F1 F2 rbar")
Fs = F0 + F1 * s + F2 * s**2 / 2
rm, rp = rb - h / 2, rb + h / 2
V = sp.integrate((rb + s)**2, (s, -h / 2, h / 2))
rc = sp.integrate((rb + s)**3, (s, -h / 2, h / 2)) / V
avg = sp.integrate((rb + s)**2 * Fs, (s, -h / 2, h / 2)) / V
Fp, Fm = Fs.subs(s, h / 2), Fs.subs(s, -h / 2)
face = (rp**2 * (rp - rc) * Fp + rm**2 * (rc - rm) * Fm) / V
exw = ((rc - rm) * Fp + (rp - rc) * Fm) / h
say("== leading error terms (booked - exact average) / grav1 ==")
say("face :", sp.series(sp.simplify(face - avg), h, 0, 4).removeO().expand())
say("exact:", sp.series(sp.simplify(exw - avg), h, 0, 5).removeO().expand())
# conservative family s_f = r_f + lam h^2 / r_f: leading F and F' coefficients
lam = sp.symbols("lam")
sp_, sm_ = rp + lam * h**2 / rp, rm + lam * h**2 / rm
cons = (rp**2 * (sp_ - rc) * Fp + rm**2 * (rc - sm_) * Fm) / V
ce = sp.series(sp.simplify(cons - avg), h, 0, 4).removeO().expand()
say("cons(lam):", sp.collect(ce, [F0, F1, F2]))
say("  F0 coeff zero at lam =", sp.solve(sp.expand(ce).coeff(F0), lam),
    "; F1 coeff zero at lam =", sp.solve(sp.expand(ce).coeff(F1), lam))

# ---------------------------------------------------------------- numpy grid helpers
def grid(r0, L, n):
    rf = r0 + L * np.arange(n + 1) / n
    rm, rp = rf[:-1], rf[1:]
    V = (rp**3 - rm**3) / 3.
    rc = 0.75 * (rp**4 - rm**4) / (rp**3 - rm**3)
    return rf, rm, rp, V, rc

def weights(kind, rf, rm, rp, V, rc, lam=0.):
    hh = rp - rm
    if kind == "face":
        return rp**2 * (rp - rc) / V, rm**2 * (rc - rm) / V
    if kind == "exact":
        return (rc - rm) / hh, (rp - rc) / hh
    if kind == "cons":
        sf = rf + lam * hh.mean()**2 / rf
        return rp**2 * (sf[1:] - rc) / V, rm**2 * (rc - sf[:-1]) / V
    raise ValueError(kind)

# ---------------------------------------------------------------- 2. one step, closed column
say("== 2. closed-wall column, one step: (dE + dPE)/dt  [grav1 = -10, F_f random interior, F = 0 at walls] ==")
rng = np.random.default_rng(42)
g1 = -10.
for (r0, H, nz) in [(5., 1., 64), (1000., 1., 64), (5., 1., 256), (1.0, 1., 32)]:
    L = 3 * H
    rf, rm, rp, V, rc = grid(r0, L, nz)
    Ff = np.zeros(nz + 1); Ff[1:-1] = rng.standard_normal(nz - 1)
    dM = -(rp**2 * Ff[1:] - rm**2 * Ff[:-1])            # per unit dt
    dPE = np.sum(-g1 * rc * dM)
    PE = np.sum(-g1 * rc * V)                           # rho = 1 column
    for kind, lamv in [("face", 0.), ("exact", 0.), ("cons", 1 / 6), ("cons", -1 / 6)]:
        a, b = weights(kind, rf, rm, rp, V, rc, lamv)
        W = g1 * (a * Ff[1:] + b * Ff[:-1])             # per unit volume per unit dt
        dE = np.sum(W * V)
        d = dE + dPE
        say("  R=%-6g h/R=%.1e %-5s%-6s (dE+dPE)/sum|W V| = %+.3e   /(E+PE) = %+.3e"
            % (r0, L / nz / r0, kind, "" if kind != "cons" else "%+.3f" % lamv,
               d / np.sum(np.abs(W * V)), d / (2 * PE)))

# ---------------------------------------------------------------- 3. convergence on a smooth profile
say("== 3. booked work vs exact r^2-measure cell average, F = exp(-(r-r0)/H) sin(pi (r-r0)/L), L = 4H ==")
xg, wg = leggauss(12)
def Ffun(r, r0, H, L): return np.exp(-(r - r0) / H) * np.sin(np.pi * (r - r0) / L)
def F2fun(r, r0, H, L):
    k = np.pi / L; z = r - r0; e = np.exp(-z / H)
    return e * ((1 / H**2 - k**2) * np.sin(k * z) - 2 * k / H * np.cos(k * z))
def errs(r0, H, nz, kind):
    L = 4 * H
    rf, rm, rp, V, rc = grid(r0, L, nz)
    Ff = Ffun(rf, r0, H, L)
    a, b = weights(kind, rf, rm, rp, V, rc)
    booked = a * Ff[1:] + b * Ff[:-1]
    rq = 0.5 * (rp + rm)[:, None] + 0.5 * (rp - rm)[:, None] * xg[None, :]
    avg = (0.5 * (rp - rm)[:, None] * wg[None, :] * rq**2 * Ffun(rq, r0, H, L)).sum(1) / V
    e = booked - avg
    hh = L / nz
    cart = hh**2 / 12. * F2fun(rc, r0, H, L)            # common Cartesian trapezoid piece
    return np.max(np.abs(e)), np.max(np.abs(e - cart))
say("  h-ladder at fixed R = 5H (H = 1): max|e| and max|e - (h^2/12)F''|; slope = log2 ratio")
for kind in ["face", "exact"]:
    prev = None
    for nz in [16, 32, 64, 128, 256]:
        e, ec = errs(5., 1., nz, kind)
        sl = "" if prev is None else "  slopes %.2f %.2f" % (np.log2(prev[0] / e), np.log2(prev[1] / ec))
        say("   %-5s nz %4d  h=%.4f  %.3e  %.3e%s" % (kind, nz, 4. / nz, e, ec, sl)); prev = (e, ec)
say("  R-ladder at fixed h/H = 1/16: max|e - (h^2/12)F''| (face should scale ~ h^2/R, exact ~ h^4-level)")
for kind in ["face", "exact"]:
    for r0 in [5., 50., 500., 5000.]:
        e, ec = errs(r0, 1., 64, kind)
        say("   %-5s R/H %6g  %.3e  x R = %.3e" % (kind, r0, ec, ec * r0))
