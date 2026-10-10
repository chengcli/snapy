#!/usr/bin/env python3
"""Option F: face-form gravity work plus the work implied by a corrected discrete PE functional.

P[rho] = sum_i V_i [rho_i phi(r_v,i) - g1 var_i s_i[rho]],  g1 = grav1 < 0, phi = -g1 r,
  var_i = <(r - r_v)^2>_V (exact r^2-measure variance of r in cell i), s_i[rho] = sum_k d_ik rho_k the slope
  of the quadratic through the cell values at r_v (interior: i-1, i, i+1; wall cells: one-sided 0,1,2).
  P_i is the exact cell PE  g1-free form  int rho phi dV  to O(h^4): rho phi integrates to m phi(r_v) - g1 V cov,
  cov = var * rho' + O(h^4).
Work: W_i V_i := -dP_i/dt - Delta_i(A F phi(r_f)) = W_face,i V_i + g1 V_i var_i s_i[rhodot],
  rhodot_k = -(A_+ F_+ - A_- F_-)_k / V_k (the x1 mass tendency). Sum_i W_i V_i + dP/dt = 0 identically (closed walls).
Checks (prints every number quoted in sections 7-9 of curved-gravity-work-weight.md):
  1. one step, closed column, random F: (dE + dP), (dE + dPE_d), each / sum|W V|; spherical R = H, 5H, 1000H, and Cartesian.
  2. booked work vs exact r^2-measure cell average on smooth closed-wall profiles: interior and wall cells, h- and R-ladders.
  3. Cartesian: option F minus face form (the change it makes there).
  4. settling check for the ablation: conservation defect and booked-work error of A, B, C, D, F on a closed column
     carrying the smooth profile (column sums).
  5. F and face form with the cp3/cp5/weno5 curvature flux H on top (F + H double counts).
  6. cell-by-cell errors, wall cells included, Cartesian and spherical.
  7. Cartesian: F against today's face+H, by region, and the uniform-grid closed forms.
  8. F against the r-bar potential form (phi_c at rbar, H on r^2 m): conservation, cell and column errors,
     accuracy of each conserved functional.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

xg, wg = leggauss(16)
G1 = -10.


def grid(r0, L, n, cart=False):
    rf = r0 + L * np.arange(n + 1) / n
    rm, rp = rf[:-1], rf[1:]
    if cart:
        A = np.ones_like(rf)
        V = rp - rm
        rv = 0.5 * (rp + rm)
        var = (rp - rm)**2 / 12.
    else:
        A = rf**2
        V = (rp**3 - rm**3) / 3.
        rv = 0.75 * (rp**4 - rm**4) / (rp**3 - rm**3)
        hh, rb = rp - rm, 0.5 * (rp + rm)
        dl = rb * hh**3 / (6. * V)  # r_v - rbar
        var = (rb**2 * hh**3 / 12. + hh**5 / 80.) / V - dl**2  # <(r - r_v)^2>_V, cancellation-free
    return dict(rf=rf, rm=rm, rp=rp, A=A, V=V, rv=rv, var=var, n=n)


def slope_matrix(g):
    """d_ik: derivative at r_v,i of the quadratic through (r_v,k, rho_k), k in the 3-point stencil."""
    n, x = g["n"], g["rv"]
    D = np.zeros((n, n))
    for i in range(n):
        ks = [i - 1, i, i + 1] if 0 < i < n - 1 else ([0, 1, 2] if i == 0 else [n - 3, n - 2, n - 1])
        for a in ks:
            # d/dx of Lagrange basis l_a at x_i
            others = [b for b in ks if b != a]
            den = np.prod([x[a] - x[b] for b in others])
            num = sum(np.prod([x[i] - x[c] for c in others if c != b]) for b in others)
            D[i, a] = num / den
    return D


def rhodot(g, Ff):
    return -(g["A"][1:] * Ff[1:] - g["A"][:-1] * Ff[:-1]) / g["V"]


def work(kind, g, Ff, lam=0.):
    """booked work per unit volume per unit time (grav1 included)"""
    rf, rm, rp, A, V, rv = g["rf"], g["rm"], g["rp"], g["A"], g["V"], g["rv"]
    h = rp - rm
    if kind in ("face", "F", "cons"):
        sf = rf if kind != "cons" else rf + lam * h.mean()**2 / rf
        W = G1 * (A[1:] * (sf[1:] - rv) * Ff[1:] + A[:-1] * (rv - sf[:-1]) * Ff[:-1]) / V
        if kind == "F":
            W = W + G1 * g["var"] * (g["D"] @ rhodot(g, Ff))
        return W
    if kind == "exact":
        return G1 * ((rv - rm) / h * Ff[1:] + (rp - rv) / h * Ff[:-1])
    raise ValueError(kind)


def P(g, rho):
    return np.sum(g["V"] * (rho * (-G1 * g["rv"]) - G1 * g["var"] * (g["D"] @ rho)))


def PEd(g, rho):
    return np.sum(g["V"] * rho * (-G1 * g["rv"]))


def mk(r0, L, n, cart=False):
    g = grid(r0, L, n, cart)
    g["D"] = slope_matrix(g)
    return g


# ------------------------------------------------------------------ 1. conservation, one step
print("== 1. closed column, random interior F, one unit-dt step: defect / sum|W V| ==")
rng = np.random.default_rng(42)
for (r0, cart, n) in [(5., False, 64), (1000., False, 64), (1., False, 32), (5., True, 64)]:
    g = mk(r0, 3., n, cart)
    Ff = np.zeros(n + 1); Ff[1:-1] = rng.standard_normal(n - 1)
    rd = rhodot(g, Ff)
    rho = np.exp(-(g["rv"] - r0))
    for kind in ["face", "F"]:
        W = work(kind, g, Ff)
        dE = np.sum(W * g["V"]); s = np.sum(np.abs(W * g["V"]))
        dP = P(g, rho + rd) - P(g, rho)          # P, PE_d linear in rho
        dPd = PEd(g, rho + rd) - PEd(g, rho)
        print("  %-4s R=%-6g %-4s  (dE+dP)/sum|WV| %+.3e   (dE+dPE_d)/sum|WV| %+.3e   (dE+dP)/(E+P) %+.3e"
              % ("cart" if cart else "sph", r0, kind, (dE + dP) / s, (dE + dPd) / s, (dE + dP) / (2 * P(g, rho))))


# ------------------------------------------------------------------ 2. convergence on a smooth closed-wall profile
def Ffun(r, r0, L, H=1.):
    return np.exp(-(r - r0) / H) * np.sin(np.pi * (r - r0) / L)


def errs(r0, n, kind, cart=False, L=4., lam=0.):
    g = mk(r0, L, n, cart)
    Ff = Ffun(g["rf"], r0, L)
    W = work(kind, g, Ff, lam)
    rm, rp = g["rm"], g["rp"]
    rq = 0.5 * (rp + rm)[:, None] + 0.5 * (rp - rm)[:, None] * xg[None, :]
    wq = rq**2 if not cart else np.ones_like(rq)
    avg = G1 * (0.5 * (rp - rm)[:, None] * wg[None, :] * wq * Ffun(rq, r0, L)).sum(1) / g["V"]
    e = (W - avg) / abs(G1)
    return np.max(np.abs(e[2:-2])), np.max(np.abs(e[[0, 1, -2, -1]])), g, W, avg


print("== 2. max|W - g1<F>_V|/|g1|, F = exp(-(r-r0)/H) sin(pi (r-r0)/4H), interior cells | 2 wall cells each side ==")
for cart, r0 in [(False, 5.), (True, 5.)]:
    for kind in ["face", "F"]:
        prev = None
        for n in [16, 32, 64, 128, 256]:
            ei, ew, *_ = errs(r0, n, kind, cart)
            sl = "" if prev is None else "  slopes %.2f %.2f" % (np.log2(prev[0] / ei), np.log2(prev[1] / ew))
            print("  %-4s R=%g %-4s nz %4d  interior %.3e  wall %.3e%s"
                  % ("cart" if cart else "sph", r0, kind, n, ei, ew, sl)); prev = (ei, ew)
print("  R-ladder, nz 64 (h = H/16), option F interior | wall; face interior x R")
for r0 in [5., 50., 500., 5000.]:
    ei, ew, *_ = errs(r0, 64, "F")
    fi, *_ = errs(r0, 64, "face")
    print("   R/H %6g  F %.3e | %.3e   face x R %.3e" % (r0, ei, ew, fi * r0))

# ------------------------------------------------------------------ 3. Cartesian change
print("== 3. Cartesian: option F - face form, nz 64, same profile: max|dW|/max|W| ==")
g = mk(5., 4., 64, True); Ff = Ffun(g["rf"], 5., 4.)
d = work("F", g, Ff) - work("face", g, Ff)
print("  interior %.3e   wall cells %.3e   (relative to max|W| %.3e)"
      % (np.max(np.abs(d[2:-2])) / np.max(np.abs(work("face", g, Ff))),
         np.max(np.abs(d[[0, 1, -2, -1]])) / np.max(np.abs(work("face", g, Ff))), np.max(np.abs(work("face", g, Ff)))))

# ------------------------------------------------------------------ 4. settling check, column sums
print("== 4. column sums on the smooth profile (closed walls), per option: ==")
print("   sum(W - g1<F>)V / sum|g1<F>|V  = booked-work error;  (sum W V + dPE_d)/sum|g1<F>|V = PE_d defect")
for r0 in [5., 1000.]:
    for n in [32, 64]:
        for kind, lam in [("face", 0.), ("exact", 0.), ("cons", 1 / 6), ("cons", -1 / 6), ("F", 0.)]:
            ei, ew, g, W, avg = errs(r0, n, kind, lam=lam)
            Ff = Ffun(g["rf"], r0, 4.)
            dPd = np.sum(g["V"] * rhodot(g, Ff) * (-G1 * g["rv"]))
            nrm = np.sum(np.abs(avg) * g["V"])
            lab = {"face": "A face", "exact": "B exact", "F": "F"}.get(kind, "C" if lam > 0 else "D")
            print("   R/H %6g nz %3d %-8s work err %+.4e   PE_d defect %+.4e   cell max|err|/max|g1<F>| %.3e"
                  % (r0, n, lab, np.sum((W - avg) * g["V"]) / nrm, (np.sum(W * g["V"]) + dPd) / nrm,
                     max(ei, ew) * abs(G1) / np.max(np.abs(avg))))

# ------------------------------------------------------------------ 5. interaction with the cp3/cp5/weno5 curvature flux
# hydro_forward.cpp subtracts grav1 * div(A H)/V, H_f = (x1v_i - x1v_{i-1})/12 (m_i - m_{i-1}), H = 0 at walls,
# m = the cell mass flux rho*v (here: the exact cell average <F>_V). It is a divergence, so it keeps E + PE_d.
def errs_H(r0, n, kind, cart=False, L=4.):
    ei, ew, g, W, avg = errs(r0, n, kind, cart, L)
    m = avg / G1
    H = np.zeros(n + 1); H[1:-1] = (g["rv"][1:] - g["rv"][:-1]) / 12. * (m[1:] - m[:-1])
    W = W - G1 * (g["A"][1:] * H[1:] - g["A"][:-1] * H[:-1]) / g["V"]
    e = (W - avg) / abs(G1)
    return np.max(np.abs(e[2:-2])), np.max(np.abs(e[[0, 1, -2, -1]])), W
print("== 5. with the cp3/cp5/weno5 curvature flux H on top: max|W - g1<F>_V|/|g1|, interior | wall ==")
for cart, r0 in [(False, 5.), (True, 5.)]:
    for kind in ["face", "F"]:
        prev = None
        for n in [32, 64, 128, 256]:
            ei, ew, _ = errs_H(r0, n, kind, cart)
            sl = "" if prev is None else "  slopes %.2f %.2f" % (np.log2(prev[0] / ei), np.log2(prev[1] / ew))
            print("  %-4s R=%g %-4s+H nz %4d  interior %.3e  wall %.3e%s"
                  % ("cart" if cart else "sph", r0, kind, n, ei, ew, sl)); prev = (ei, ew)
print("  R-ladder nz 64, face+H interior error x R^2 (a -h^2/(6 R^2) F residue gives a constant):")
for r0 in [5., 50., 500.]:
    ei, ew, _ = errs_H(r0, 64, "face")
    print("   R/H %6g  face+H interior %.3e  x R^2 %.3e" % (r0, ei, ei * r0**2))
g = mk(5., 4., 64, True); Ff = Ffun(g["rf"], 5., 4.)
_, _, WH = errs_H(5., 64, "face", True)
print("  Cartesian nz 64: max|W_F - W_face+H| / max|W| = %.3e (both remove h^2/12 F'')"
      % (np.max(np.abs(work("F", g, Ff) - WH)) / np.max(np.abs(WH))))

# ------------------------------------------------------------------ 6. wall cells one by one
print("== 6. |W - g1<F>_V|/|g1| cell by cell, nz 16/32/64/128 (slope = log2 ratio), same profile, closed walls ==")
print("   first = cell 0 (on the lower wall), second = cell 1, last = cell n-1 (on the upper wall),")
print("   interior = max over cells 2..n-3")
for cart in (True, False):
    for kind in ("face+H", "F"):
        prev = None
        for n in (16, 32, 64, 128):
            if kind == "F":
                *_, g, W, avg = errs(5., n, "F", cart)
            else:
                *_, W = errs_H(5., n, "face", cart)
                _, _, g, _, avg = errs(5., n, "face", cart)
            e = np.abs(W - avg) / abs(G1)
            row = np.array([e[0], e[1], e[-2], e[-1], np.max(e[2:-2])])
            sl = "" if prev is None else "  slopes " + " ".join("%.2f" % v for v in np.log2(prev / row))
            print("  %-4s R=5 %-6s nz %4d  first %.2e second %.2e second-last %.2e last %.2e interior %.2e%s"
                  % ("cart" if cart else "sph", kind, n, *row, sl))
            prev = row

# ------------------------------------------------------------------ 7. Cartesian: F against today's face+H, by region
print("== 7. Cartesian, F - (face+H), max|dW| / max|W|: interior (cells 2..n-3) | first cell | last cell ==")
prev = None
for n in (16, 32, 64, 128):
    _, _, g, WF, _ = errs(5., n, "F", True)
    *_, WH = errs_H(5., n, "face", True)
    d, s = np.abs(WF - WH), np.max(np.abs(WH))
    row = np.array([np.max(d[2:-2]), d[0], d[-1]]) / s
    sl = "" if prev is None else "  slopes " + " ".join("%.2f" % v for v in np.log2(prev / row))
    print("  nz %4d  interior %.2e  first %.2e  last %.2e%s" % (n, *row, sl)); prev = row
print("  uniform-grid closed forms (sec 8): interior W/g1 = (F+ + F-)/2 - (F[i+3/2] - F[i+1/2] - F[i-1/2] + F[i-3/2])/24,")
print("  wall cell W/g1 = (19 F[1/2] - 5 F[3/2] + F[5/2])/24; check against the matrix form on random F:")
g = mk(0., 1., 12, True); Ff = np.zeros(13); Ff[1:-1] = np.random.default_rng(7).standard_normal(11)
W = work("F", g, Ff) / G1
Wi = 0.5 * (Ff[3:-2] + Ff[2:-3]) - (Ff[4:-1] - Ff[3:-2] - Ff[2:-3] + Ff[1:-4]) / 24.
print("  interior max diff %.1e   wall cell diff %.1e"
      % (np.max(np.abs(W[2:-2] - Wi)), abs(W[0] - (19 * Ff[1] - 5 * Ff[2] + Ff[3]) / 24.)))
print("  upper wall cell W/g1 = (19 F[n-3/2] - 5 F[n-5/2] + F[n-7/2])/24: diff %.1e"
      % abs(W[-1] - (19 * Ff[-2] - 5 * Ff[-3] + Ff[-4]) / 24.))

# ------------------------------------------------------------------ 8. F against the r-bar potential form (derivation sec 9)
# r-bar form: face form with phi_c = g1-potential at rbar = (r+ + r-)/2, and the curvature flux applied to r^2 m:
# A_f H_f = (x1v_i - x1v_{i-1})/12 (rbar_i^2 m_i - rbar_{i-1}^2 m_{i-1}), zero at the walls; it conserves
# E + P_rbar, P_rbar = sum V rho (-g1 rbar). F conserves E + P, P of eq. (6). m = <F>_V as in sec 5.
def work_rbar(g, Ff, m):
    rf, rm, rp, A, V, rv = g["rf"], g["rm"], g["rp"], g["A"], g["V"], g["rv"]
    rb = 0.5 * (rp + rm)
    W = G1 * (A[1:] * (rf[1:] - rb) * Ff[1:] + A[:-1] * (rb - rf[:-1]) * Ff[:-1]) / V
    AH = np.zeros(g["n"] + 1); AH[1:-1] = (rv[1:] - rv[:-1]) / 12. * (rb[1:]**2 * m[1:] - rb[:-1]**2 * m[:-1])
    return W - G1 * (AH[1:] - AH[:-1]) / V


def Prbar(g, rho):
    return np.sum(g["V"] * rho * (-G1 * 0.5 * (g["rp"] + g["rm"])))


print("== 8. F vs the r-bar potential form (phi_c at rbar + curv on r^2 m), spherical ==")
print("  8a. one step, closed column, random F and m: defect against each form's own functional / sum|W V|")
rng = np.random.default_rng(42)
for r0 in (1., 5., 1000.):
    g = mk(r0, 3., 64)
    Ff = np.zeros(65); Ff[1:-1] = rng.standard_normal(63); m = rng.standard_normal(64)
    rho, rd = np.exp(-(g["rv"] - r0)), rhodot(g, Ff)
    Wb, WF = work_rbar(g, Ff, m), work("F", g, Ff)
    print("   R/H %6g  rbar: (dE+dP_rbar) %+.1e  (dE+dP) %+.1e | F: (dE+dP) %+.1e  (dE+dP_rbar) %+.1e"
          % (r0, (np.sum(Wb * g["V"]) + Prbar(g, rd)) / np.sum(np.abs(Wb * g["V"])),
             (np.sum(Wb * g["V"]) + P(g, rd)) / np.sum(np.abs(Wb * g["V"])),
             (np.sum(WF * g["V"]) + P(g, rd)) / np.sum(np.abs(WF * g["V"])),
             (np.sum(WF * g["V"]) + Prbar(g, rd)) / np.sum(np.abs(WF * g["V"]))))


def errs_rbar(r0, n, L=4.):
    _, _, g, _, avg = errs(r0, n, "face", False, L)
    W = work_rbar(g, Ffun(g["rf"], r0, L), avg / G1)
    return np.abs(W - avg) / abs(G1), g, W, avg


print("  8b. cell error |W - g1<F>_V|/|g1|, same smooth profile as sec 2, R = 5H: first | second | last | interior")
for kind in ("rbar", "F"):
    prev = None
    for n in (16, 32, 64, 128, 256):
        if kind == "F":
            *_, g, W, avg = errs(5., n, "F"); e = np.abs(W - avg) / abs(G1)
        else:
            e, *_ = errs_rbar(5., n)
        row = np.array([e[0], e[1], e[-1], np.max(e[2:-2])])
        sl = "" if prev is None else "  slopes " + " ".join("%.2f" % v for v in np.log2(prev / row))
        print("   %-4s nz %4d  first %.2e second %.2e last %.2e interior %.2e%s" % (kind, n, *row, sl)); prev = row
print("  8c. column-summed work error sum(W - g1<F>)V / sum|g1<F>|V (what a column eps diagnostic sees), R = 5H | 1000H")
for n in (16, 32, 64, 128, 256):
    out = []
    for r0 in (5., 1000.):
        e, g, Wb, avg = errs_rbar(r0, n)
        *_, WF, _ = errs(r0, n, "F")
        nrm = np.sum(np.abs(avg) * g["V"])
        out += [np.sum((Wb - avg) * g["V"]) / nrm, np.sum((WF - avg) * g["V"]) / nrm]
    print("   nz %4d  R=5: rbar %+.3e  F %+.3e   R=1000: rbar %+.3e  F %+.3e" % (n, *out))
print("  8d. accuracy of each conserved functional: |P - int rho phi dV| / |int rho phi dV|, rho = exp(-(r-r0)), R = 5H, L = 4H")
for n in (16, 32, 64, 128, 256):
    g = mk(5., 4., n)
    rq = 0.5 * (g["rp"] + g["rm"])[:, None] + 0.5 * (g["rp"] - g["rm"])[:, None] * xg[None, :]
    wq = 0.5 * (g["rp"] - g["rm"])[:, None] * wg[None, :] * rq**2
    rho = (wq * np.exp(-(rq - 5.))).sum(1) / g["V"]          # exact cell averages
    ex = (wq * np.exp(-(rq - 5.)) * (-G1 * rq)).sum()
    print("   nz %4d  P_d (x1v) %.2e   P_rbar %.2e   P (F) %.2e"
          % (n, abs(PEd(g, rho) - ex) / ex, abs(Prbar(g, rho) - ex) / ex, abs(P(g, rho) - ex) / ex))
print("  8e. P_rbar - int rho phi dV against the pure wall term g1 h^2/12 [r^2 rho] (top minus bottom), same rho")
for n in (32, 64, 128, 256):
    g = mk(5., 4., n); h = 4. / n
    rq = 0.5 * (g["rp"] + g["rm"])[:, None] + 0.5 * (g["rp"] - g["rm"])[:, None] * xg[None, :]
    wq = 0.5 * (g["rp"] - g["rm"])[:, None] * wg[None, :] * rq**2
    rho = (wq * np.exp(-(rq - 5.))).sum(1) / g["V"]
    ex = (wq * np.exp(-(rq - 5.)) * (-G1 * rq)).sum()
    wall = G1 * h**2 / 12. * (9.**2 * np.exp(-4.) - 5.**2)
    print("   nz %4d  P_rbar - exact %+.6e   wall term %+.6e   ratio %.5f" % (n, Prbar(g, rho) - ex, wall, (Prbar(g, rho) - ex) / wall))
