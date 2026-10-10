"""Exact checks behind docs/derivations/wb-ref4.md (SNAP_WB_REF4), in sympy.

  1. the filter F = (-1, 4, 10, 4, -1)/16: unit sum, zero 1st-3rd moments, 4th moment -3/2, zero Nyquist
     response; the base binomial B = (1, 4, 6, 4, 1)/16 has 2nd moment 1;
  2. the cubic extrapolation past a wall, (4, -6, 4, -1) and (10, -20, 15, -4), and the identity that F with
     those two values returns the first two cells unchanged for ANY data (so "unfiltered at the two end
     cells" and "filtered with cubic wall extrapolation" are the same reference);
  3. the face value of cell averages from the quartic primitive: interior (-1, 7, 7, -1)/12, wall rows
     (25, -23, 13, -3)/12 and (3, 13, -5, 1)/12, exact for cubic point profiles;
  4. the base face-density offset delta rho_f = dz^2/12 (p' R' + 2 p R'') + O(dz^4), R = rho/p;
  5. with F and the quartic face value the offset is O(dz^4);
  6. the non-uniform cell pressure: three-point Gauss-Legendre average of the cubic through four faces is
     exact for cubic p(z).
Every check is an exact (symbolic or rational) identity; the script raises on the first failure.

  python3 docs/derivations/wb_ref4_weights.py
"""
import sympy as sp

z, h, x = sp.symbols("z h x", real=True)
R = sp.Rational
F = [R(-1, 16), R(4, 16), R(10, 16), R(4, 16), R(-1, 16)]
B = [R(1, 16), R(4, 16), R(6, 16), R(4, 16), R(1, 16)]
E = [[4, -6, 4, -1], [10, -20, 15, -4]]  # values at index -1 and -2 from cells 0..3


def check(cond, what):
    assert cond, what
    print("ok:", what)


# 1. moments
mom = lambda w, k: sum(wm * m ** k for wm, m in zip(w, range(-2, 3)))
check([mom(F, k) for k in range(5)] == [1, 0, 0, 0, R(-3, 2)], "F: sum 1, moments 1-3 zero, 4th -3/2")
check(sum(wm * (-1) ** m for wm, m in zip(F, range(-2, 3))) == 0, "F: Nyquist response 0")
check(mom(B, 2) == 1 and mom(B, 1) == 0, "B: 2nd moment 1 (bias dz^2/2 q'')")

# 2. wall extrapolation and the end-cell identity
c = sp.symbols("c0:4")
cubic = sp.interpolate(list(zip(range(4), c)), x)
for d, row in zip((1, 2), E):
    check(sp.expand(cubic.subs(x, -d) - sum(a * b for a, b in zip(row, c))) == 0,
          f"cubic extrapolation to index {-d}: {row}")
r = sp.symbols("r0:6")
ext = [sum(a * b for a, b in zip(E[1], r[:4])), sum(a * b for a, b in zip(E[0], r[:4]))] + list(r)
for i in (0, 1):
    check(sp.expand(sum(w * ext[i + k] for k, w in enumerate(F)) - r[i]) == 0,
          f"F with the cubic wall values returns cell {i} unchanged for any data")

# 3. face value from cell averages via the quartic primitive
def face_weights(face, start):
    """weights on cells start..start+3 (uniform, unit width) of P'(face)"""
    nodes = list(range(start, start + 5))
    w = []
    for k in range(4):
        tot = 0
        for j in range(k + 1, 5):
            L = sp.prod([(x - nodes[m]) / (nodes[j] - nodes[m]) for m in range(5) if m != j])
            tot += sp.diff(L, x).subs(x, face)
        w.append(sp.nsimplify(tot))
    return w


check(face_weights(2, 0) == [R(-1, 12), R(7, 12), R(7, 12), R(-1, 12)], "interior face (-1, 7, 7, -1)/12")
check(face_weights(0, 0) == [R(25, 12), R(-23, 12), R(13, 12), R(-3, 12)], "wall face (25, -23, 13, -3)/12")
check(face_weights(1, 0) == [R(3, 12), R(13, 12), R(-5, 12), R(1, 12)], "next face (3, 13, -5, 1)/12")
a = sp.symbols("a0:4")
q = sum(ak * x ** k for k, ak in enumerate(a))  # cubic point profile
avg = [sp.integrate(q, (x, i, i + 1)) for i in range(4)]
for f in (0, 1, 2):
    check(sp.expand(sum(w * v for w, v in zip(face_weights(f, 0), avg)) - q.subs(x, f)) == 0,
          f"face {f}: exact for a cubic profile")

# 4. base offset: rho_sf - I[rho_ref], smooth p(z), rho(z), to O(h^2)
p0, p1, p2, R0, R1, R2 = sp.symbols("p0 p1 p2 R0 R1 R2")
P = p0 + p1 * x + p2 * x ** 2 / 2          # around the face, x in units of length
Rr = R0 + R1 * x + R2 * x ** 2 / 2
rho = sp.expand(P * Rr)
cavg = lambda f, i: sp.integrate(f, (x, i * h, (i + 1) * h)) / h   # cell i spans [i h, (i+1) h], face at 0
rbar = lambda i: cavg(rho, i)
pbar = lambda i: cavg(P, i)
ratio = lambda i: sp.series(rbar(i) / pbar(i), h, 0, 3).removeO()
rs = lambda i: sp.expand(sum(wb * ratio(i + m) for wb, m in zip(B, range(-2, 3))))
# p_ref = p_bar to high order (wb-ref4.md section 2); I[q]_face = q(0) + O(h^4) for cell averages of q
dref = lambda i: sp.series(pbar(i) * rs(i), h, 0, 3).removeO()
# the cell field dref has cell averages D(z) = rho + h^2 c(z) + ...; its face value I[dref] equals the
# (-1,7,7,-1)/12 combination to O(h^4)
I_dref = sp.expand(sum(w * dref(i) for w, i in zip([R(-1, 12), R(7, 12), R(7, 12), R(-1, 12)], (-2, -1, 0, 1))))
dsf = sp.expand(p0 * (rs(-1) + rs(0)) / 2)
off = sp.simplify(sp.series(dsf - I_dref, h, 0, 3).removeO())
want = h ** 2 / 12 * (p1 * R1 + 2 * p0 * R2)
check(sp.simplify(off - want) == 0, "base face-density offset = dz^2/12 (p'R' + 2 p R'')")

# 5. switched: dref = p_bar F(r), dsf = the quartic face value of dref -> offset 0 to this order
rF = lambda i: sp.expand(sum(wf * ratio(i + m) for wf, m in zip(F, range(-2, 3))))
dref4 = lambda i: sp.series(pbar(i) * rF(i), h, 0, 3).removeO()
check(all(sp.simplify(sp.series(dref4(i) - rbar(i), h, 0, 3).removeO()) == 0 for i in (-1, 0)),
      "switched cell reference = rho_bar + O(dz^4) (here: to O(dz^2) exactly)")

# 6. three-point Gauss-Legendre average of the cubic through four faces (non-uniform)
xs = sp.symbols("x0:4")
b = sp.symbols("b0:4")
cub = sum(bk * x ** k for k, bk in enumerate(b))
lo, hi = xs[1], xs[2]
c0, hh = (lo + hi) / 2, (hi - lo) / 2
g = [(-sp.sqrt(R(3, 5)), R(5, 18)), (0, R(8, 18)), (sp.sqrt(R(3, 5)), R(5, 18))]
gl = sum(wq * cub.subs(x, c0 + t * hh) for t, wq in g)
check(sp.simplify(gl - sp.integrate(cub, (x, lo, hi)) / (hi - lo)) == 0,
      "3-point Gauss-Legendre cell average exact for the cubic through four faces")
print("all checks passed")
