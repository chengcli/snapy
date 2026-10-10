#!/usr/bin/env python3
"""Option F on a gnomonic-equiangle (cubed-sphere) x1 grid: the centroid term (sec 12 of
curved-gravity-work-weight.md; prints every number quoted there).

Per steradian (the gnomonic solid angle of a column is common to A and V and cancels):
  A = r^2, V = (r+^3 - r-^3)/3, x1v = rbar = (r+ + r-)/2 (gnomonic_equiangle.cpp), and the r^2
  centroid r_c = rbar + delta, delta = rbar h^3 / (6 V) (not x1v, unlike spherical-polar).
Forms, per unit volume and time (g1 = grav1 < 0, phi = -g1 r):
  face  W = g1 [A+ (r+ - x1v) F+ + A- (x1v - r-) F-] / V      (eq. 1 with phi_c = phi(x1v))
  face+H  face minus g1 div(A H)/V, H_f = (x1v_i - x1v_{i-1})/12 (m_i - m_{i-1}), H = 0 at walls
  lift  face + g1 var s[rhodot]               (the spherical-polar form with only the gate lifted)
  fix   lift + g1 delta rhodot                (eq. 12.5, what the code books on this grid)
var = <(r - r_c)^2>_V, s = the slope at x1v of the quadratic through the cell values at x1v.
Checks:
  1. sympy: delta, and the cell PE expansion (12.2) to O(h^4) with x1v = rbar;
  2. one step, closed column, random interior F: each form against its own functional
     (P_lift without, P_fix with the delta term), and fix against P_lift;
  3. booked work against the exact r^2 cell average on a smooth closed-wall profile: wall and interior
     cells, nz 16 -> 256, at R = 5H and at R = 60H (the six-panel shell of the #300 run);
  4. R-ladder at nz 64: lift's interior error times R is flat (an h^2/R term), fix is at the h^4 floor;
  5. the slope nodes: var (s on x1v - s on r_c) is O(h^4).
"""
import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss

xg, wg = leggauss(16)
G1 = -10.

# ------------------------------------------------------------------ 1. sympy
print("== 1. sympy: delta and the cell PE with x1v = rbar ==")
R, h, s_ = sp.symbols("R h s", positive=True)
c = sp.symbols("c0:5")
rho = sum(c[k] * s_**k for k in range(5))          # rho(r), s = r - R, R = rbar
rp, rm = R + h / 2, R - h / 2
Vr = sp.simplify((rp**3 - rm**3) / 3)
rc = sp.integrate((R + s_)**3, (s_, -h / 2, h / 2)) / Vr
delta = sp.simplify(rc - R)
print("  delta = r_c - rbar =", sp.simplify(delta - R * h**3 / (6 * Vr)), "+ rbar h^3/(6 V)  (closed form)")
print("  delta series:", sp.series(delta, h, 0, 5).removeO())
var = sp.integrate((R + s_)**2 * (R + s_ - rc)**2, (s_, -h / 2, h / 2)) / Vr
g1 = sp.Symbol("g1")
pe = sp.integrate((R + s_)**2 * rho * (-g1) * (R + s_), (s_, -h / 2, h / 2)) / Vr   # <rho phi>_V
rbar_cell = sp.integrate((R + s_)**2 * rho, (s_, -h / 2, h / 2)) / Vr              # cell value
drho_c = sp.diff(rho, s_).subs(s_, rc - R)                                          # rho'(r_c)
form = rbar_cell * (-g1 * R) - g1 * delta * rbar_cell - g1 * var * drho_c            # (12.2) per unit V
lift = rbar_cell * (-g1 * R) - g1 * var * drho_c                                      # without delta
d_form = sp.series(sp.simplify(pe - form), h, 0, 5).removeO()
d_lift = sp.series(sp.simplify(pe - lift), h, 0, 4).removeO()
print("  <rho phi>_V - [rho phi(x1v) - g1 delta rho - g1 var rho'(r_c)] =", sp.factor(d_form))
print("  <rho phi>_V - [rho phi(x1v) - g1 var rho'(r_c)]               =", sp.factor(d_lift))


# ------------------------------------------------------------------ helpers
def grid(r0, L, n):
    rf = r0 + L * np.arange(n + 1) / n
    rm, rp = rf[:-1], rf[1:]
    hh, rb = rp - rm, 0.5 * (rp + rm)
    V = rb * rb * hh + hh**3 / 12.
    dl = rb * hh**3 / (6. * V)                     # r_c - rbar
    var = (rb**2 * hh**3 / 12. + hh**5 / 80.) / V - dl**2
    g = dict(rf=rf, rm=rm, rp=rp, A=rf**2, V=V, x1v=rb, rc=rb + dl, off=dl, var=var, n=n)
    g["D"] = slope_matrix(g["x1v"])
    return g


def slope_matrix(x):
    n = len(x)
    D = np.zeros((n, n))
    for i in range(n):
        ks = [i - 1, i, i + 1] if 0 < i < n - 1 else ([0, 1, 2] if i == 0 else [n - 3, n - 2, n - 1])
        for a in ks:
            others = [b for b in ks if b != a]
            den = np.prod([x[a] - x[b] for b in others])
            num = sum(np.prod([x[i] - x[c] for c in others if c != b]) for b in others)
            D[i, a] = num / den
    return D


def rhodot(g, Ff):
    return -(g["A"][1:] * Ff[1:] - g["A"][:-1] * Ff[:-1]) / g["V"]


def work(kind, g, Ff, avg=None):
    A, V, x = g["A"], g["V"], g["x1v"]
    W = G1 * (A[1:] * (g["rp"] - x) * Ff[1:] + A[:-1] * (x - g["rm"]) * Ff[:-1]) / V
    if kind == "face+H":
        m = avg / G1
        H = np.zeros(g["n"] + 1)
        H[1:-1] = (x[1:] - x[:-1]) / 12. * (m[1:] - m[:-1])
        return W - G1 * (A[1:] * H[1:] - A[:-1] * H[:-1]) / V
    if kind in ("lift", "fix"):
        W = W + G1 * g["var"] * (g["D"] @ rhodot(g, Ff))
    if kind == "fix":
        W = W + G1 * g["off"] * rhodot(g, Ff)
    return W


def P(g, rho, with_delta):
    out = g["V"] * (rho * (-G1 * g["x1v"]) - G1 * g["var"] * (g["D"] @ rho))
    if with_delta:
        out = out - G1 * g["V"] * g["off"] * rho
    return np.sum(out)


# ------------------------------------------------------------------ 2. conservation
print("== 2. closed column, random interior F, one unit-dt step: (dE + dP)/sum|W V| ==")
rng = np.random.default_rng(42)
for r0 in (5., 60., 1000.):
    g = grid(r0, 3., 64)
    Ff = np.zeros(65)
    Ff[1:-1] = rng.standard_normal(63)
    rd, rho = rhodot(g, Ff), np.exp(-(g["x1v"] - r0))
    for kind in ("lift", "fix"):
        W = work(kind, g, Ff)
        dE, s = np.sum(W * g["V"]), np.sum(np.abs(W * g["V"]))
        own = P(g, rho + rd, kind == "fix") - P(g, rho, kind == "fix")
        other = P(g, rho + rd, kind != "fix") - P(g, rho, kind != "fix")
        print("  R/H %-6g %-4s  against its own P %+.2e   against the other P %+.2e"
              % (r0, kind, (dE + own) / s, (dE + other) / s))


# ------------------------------------------------------------------ 3. convergence
def Ffun(r, r0, L, H=1.):
    return np.exp(-(r - r0) / H) * np.sin(np.pi * (r - r0) / L)


def errs(r0, n, kind, L=4.):
    g = grid(r0, L, n)
    Ff = Ffun(g["rf"], r0, L)
    rq = 0.5 * (g["rp"] + g["rm"])[:, None] + 0.5 * (g["rp"] - g["rm"])[:, None] * xg[None, :]
    avg = G1 * (0.5 * (g["rp"] - g["rm"])[:, None] * wg[None, :] * rq**2 * Ffun(rq, r0, L)).sum(1) / g["V"]
    e = (work(kind, g, Ff, avg) - avg) / abs(G1)
    return np.max(np.abs(e[2:-2])), np.max(np.abs(e[[0, 1, -2, -1]]))


print("== 3. max|W - g1<F>_V|/|g1|, F = exp(-(r-r0)/H) sin(pi (r-r0)/4H), closed walls: "
      "interior | the 2 cells at each wall ==")
for r0 in (5., 60.):
    for kind in ("face", "face+H", "lift", "fix"):
        prev = None
        for n in (16, 32, 64, 128, 256):
            ei, ew = errs(r0, n, kind)
            sl = "" if prev is None else "  orders %.2f %.2f" % (np.log2(prev[0] / ei), np.log2(prev[1] / ew))
            print("  R/H %-3g %-6s nz %4d  interior %.3e  wall %.3e%s" % (r0, kind, n, ei, ew, sl))
            prev = (ei, ew)

# ------------------------------------------------------------------ 4. R-ladder
print("== 4. R-ladder, nz 64: interior | wall error, and lift's interior error x R ==")
for r0 in (5., 50., 500., 5000.):
    li, lw = errs(r0, 64, "lift")
    fi, fw = errs(r0, 64, "fix")
    print("  R/H %6g  lift %.3e | %.3e (interior x R %.3e)   fix %.3e | %.3e" % (r0, li, lw, li * r0, fi, fw))

# ------------------------------------------------------------------ 5. slope nodes
print("== 5. var * |s on x1v - s on r_c| for rho = exp(-(r - r0)), max over cells / max|var s| ==")
for r0 in (5., 60.):
    prev = None
    for n in (16, 32, 64, 128):
        g = grid(r0, 4., n)
        rq = 0.5 * (g["rp"] + g["rm"])[:, None] + 0.5 * (g["rp"] - g["rm"])[:, None] * xg[None, :]
        rho = (0.5 * (g["rp"] - g["rm"])[:, None] * wg[None, :] * rq**2 * np.exp(-(rq - r0))).sum(1) / g["V"]
        a = g["var"] * (g["D"] @ rho)
        b = g["var"] * (slope_matrix(g["rc"]) @ rho)
        d = np.max(np.abs(a - b)) / np.max(np.abs(a))
        print("  R/H %-3g nz %4d  %.3e%s" % (r0, n, d, "" if prev is None else "  order %.2f" % np.log2(prev / d)))
        prev = d
