"""Exact checks behind docs/derivations/x1-centroid-spherical.md (SNAP_X1_CENTROID_EXACT), in sympy.

On a spherical-polar x1 grid of rational faces (r0 = 5, dr = 1/8, ten cells, so r/dr = 40; and a stretched
grid), with exact rational arithmetic:
  1. the five-point r^2-mean -> plain-mean conversion C is exact for every polynomial of degree <= 4, at the
     centred interior window and at both one-sided windows of a clamped wall, and NOT exact at degree 5
     (so its error is O(dr^5) times the fifth derivative);
  2. four-point Gauss-Legendre integrates every integrand the code forms (x^n r^2 with n <= 4; r L_j with L_j
     a quintic) exactly, so the code's weights are the exact ones;
  3. the pressure source S_i = (2/V_i) int r p~ dr, p~ the quintic through six faces, satisfies
     A_{i+1/2} p_{i+1/2} - A_{i-1/2} p_{i-1/2} - V_i S_i = int r^2 p~' dr exactly (integration by parts),
     and a constant pressure gives zero net force, with V_i = (r_{i+1/2}^3 - r_{i-1/2}^3)/3;
  4. the rest balance: for a density rho(r) of degree <= 4 and p = -g int rho dr (degree <= 5), the face
     pressures built by the hydrostatic scan from the CONVERTED cell densities, plus any constant offset c,
     give a net x1 force -(flux difference)/V + S - g <rho>_{r^2} = 0 exactly in every cell.
Every check is an exact identity; the script raises on the first failure.

  python3 docs/derivations/verify_x1_centroid.py
"""
import sympy as sp

r, x = sp.symbols("r x", real=True)
Q = sp.Rational
GX = [sp.sqrt(Q(3, 7) - Q(2, 7) * sp.sqrt(Q(6, 5))), sp.sqrt(Q(3, 7) + Q(2, 7) * sp.sqrt(Q(6, 5)))]
GW = [Q(1, 2) + sp.sqrt(30) / 36, Q(1, 2) - sp.sqrt(30) / 36]
GAUSS4 = [(-GX[1], GW[1]), (-GX[0], GW[0]), (GX[0], GW[0]), (GX[1], GW[1])]


def check(cond, what):
    assert cond, what
    print("ok:", what)


def r2mean(f, a, b):
    return sp.integrate(f * r**2, (r, a, b)) / sp.integrate(r**2, (r, a, b))


def plainmean(f, a, b):
    return sp.integrate(f, (r, a, b)) / (b - a)


def gauss4(f, a, b):
    """four-point Gauss-Legendre of f(r) over [a, b], as the code forms it"""
    return sum(w * f.subs(r, (a + b) / 2 + (b - a) / 2 * xq) for xq, w in GAUSS4) * (b - a) / 2


def conv_weights(xf, i, lo, hi):
    """the code's conversion weights for cell i: window clamped to [lo, hi], moments of x = (r - c)/h"""
    s = max(lo, min(hi - 4, i - 2))
    c, h = (xf[i] + xf[i + 1]) / 2, xf[i + 1] - xf[i]
    M = sp.Matrix(5, 5, lambda n, k: r2mean(((r - c) / h) ** n, xf[s + k], xf[s + k + 1]))
    rhs = sp.Matrix([1, 0, Q(1, 12), 0, Q(1, 80)])
    return s, list(M.LUsolve(rhs))


def lagrange(X, j, t):
    return sp.prod([(t - X[m]) / (X[j] - X[m]) for m in range(len(X)) if m != j])


GRIDS = {
    "uniform r0=5 dr=1/8": [Q(5) + Q(k, 8) for k in range(11)],
    "stretched": [Q(5) + Q(k, 8) + Q(k * k, 400) for k in range(11)],
}

for name, xf in GRIDS.items():
    nc = len(xf) - 1
    # 1. conversion exact to degree 4, not 5; interior and both clamped walls
    for i in (0, 1, nc // 2, nc - 2, nc - 1):
        s, wt = conv_weights(xf, i, 0, nc - 1)
        for deg in range(6):
            f = (r - Q(53, 10)) ** deg
            got = sum(wt[k] * r2mean(f, xf[s + k], xf[s + k + 1]) for k in range(5))
            err = sp.nsimplify(got - plainmean(f, xf[i], xf[i + 1]))
            if deg <= 4:
                check(err == 0, "%s: conversion exact, cell %d (window %d..%d), degree %d" % (name, i, s, s + 4, deg))
            else:
                check(err != 0, "%s: conversion NOT exact at degree 5, cell %d" % (name, i))
        check(sp.simplify(sum(wt) - 1) == 0, "%s: conversion weights sum to 1, cell %d" % (name, i))

    # 2. Gauss-4 is exact for the code's integrands
    a, b = xf[3], xf[4]
    c, h = (a + b) / 2, b - a
    for n in range(5):
        f = ((r - c) / h) ** n * r**2
        check(sp.simplify(gauss4(f, a, b) - sp.integrate(f, (r, a, b))) == 0,
              "%s: Gauss-4 exact for x^%d r^2" % (name, n))
    X = [(xf[3 + j] - c) / h for j in range(6)]
    for j in range(6):
        f = r * lagrange(X, j, (r - c) / h)
        check(sp.simplify(gauss4(f, a, b) - sp.integrate(f, (r, a, b))) == 0,
              "%s: Gauss-4 exact for r L_%d" % (name, j))

    # 3. integration by parts and constant pressure, at an interior and both one-sided windows
    P = sp.symbols("P0:%d" % (nc + 1))
    for i in (0, nc // 2, nc - 1):
        s = max(0, min(nc - 5, i - 2))
        a, b = xf[i], xf[i + 1]
        V = (b**3 - a**3) / 3
        pt = sum(P[s + j] * lagrange([xf[s + k] for k in range(6)], j, r) for j in range(6))
        S = 2 / V * sp.integrate(r * pt, (r, a, b))
        check(sp.expand(pt.subs(r, a) - P[i]) == 0 and sp.expand(pt.subs(r, b) - P[i + 1]) == 0,
              "%s: p~ passes through both faces of cell %d" % (name, i))
        ibp = b**2 * P[i + 1] - a**2 * P[i] - V * S - sp.integrate(r**2 * sp.diff(pt, r), (r, a, b))
        check(sp.expand(ibp) == 0, "%s: A p | - V S = int r^2 p~' dr, cell %d" % (name, i))
        const = {P[k]: 1 for k in range(nc + 1)}
        check(sp.simplify((b**2 - a**2) - V * S.subs(const)) == 0,
              "%s: constant pressure exerts no force, cell %d" % (name, i))

    # 4. rest balance from converted r^2-mean densities, a scan and an arbitrary anchor offset
    g, cst = Q(7, 3), sp.Symbol("c")
    rho = 3 - (r - xf[0]) + Q(1, 2) * (r - xf[0]) ** 2 - Q(1, 5) * (r - xf[0]) ** 3 + Q(1, 9) * (r - xf[0]) ** 4
    rbar = [r2mean(rho, xf[i], xf[i + 1]) for i in range(nc)]
    plain = []
    for i in range(nc):
        s, wt = conv_weights(xf, i, 0, nc - 1)
        plain.append(sum(wt[k] * rbar[s + k] for k in range(5)))

    def worst_force(scan_rho):
        pf = [None] * (nc + 1)
        pf[nc] = cst
        for i in range(nc - 1, -1, -1):  # top-down scan: the hydrostatic step
            pf[i] = pf[i + 1] + g * (xf[i + 1] - xf[i]) * scan_rho[i]
        worst = 0
        for i in range(nc):
            s = max(0, min(nc - 5, i - 2))
            a, b = xf[i], xf[i + 1]
            V = (b**3 - a**3) / 3
            pt = sum(pf[s + j] * lagrange([xf[s + k] for k in range(6)], j, r) for j in range(6))
            S = 2 / V * sp.integrate(r * pt, (r, a, b))
            force = -(b**2 * pf[i + 1] - a**2 * pf[i]) / V + S - g * rbar[i]
            worst = max(worst, abs(sp.nsimplify(sp.expand(force))))
        return worst

    check(worst_force(plain) == 0,
          "%s: rest column of r^2 means is exactly balanced in every cell (any offset c)" % name)
    check(worst_force(rbar) != 0, "%s: without the conversion the same column is not balanced" % name)

print("all checks passed")
