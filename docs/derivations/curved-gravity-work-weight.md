# Face-form gravity work on a radial grid: exact weights versus discrete E+PE conservation

Scope: the x1 face-form gravity work (`gravity-work: face`) on a spherical-polar grid, explicit path
(`src/hydro/hydro_forward.cpp`, the `face_gravity_work` block) and implicit path (`src/implicit/implicit_hydro.cpp`,
`work_lo`/`work_hi`, which carry the same weights times 1/2 for the face average of the two cell momenta).
Base: 8cea3ae. Every closed form below is checked by `curved_gravity_work_weight.py` (sympy + numpy;
`python docs/derivations/curved_gravity_work_weight.py` prints every number quoted; "replica" below).

**Result.** The exact $r^2$-measure weights remove both $O(h^2/\bar r)$ error terms, but they do **not** conserve
discrete E+PE: the per-face defect is $h^3/3$ (relative $h^2/(3R^2)$), the same order as the terms they remove.
No two-point weight choice that keeps the face form's conservation of E+PE$_d$ removes both terms; one or the other, not both.

## 1. Setup

Per steradian (the polar and azimuthal factors are common to $A$ and $V$ and cancel):
$$A(r) = r^2,\qquad V_i = \tfrac13\left(r_{i+1/2}^3 - r_{i-1/2}^3\right),\qquad
r_{c,i} = \frac{\int_i r^3\,dr}{\int_i r^2\,dr} = \frac34\,\frac{r_{i+1/2}^4 - r_{i-1/2}^4}{r_{i+1/2}^3 - r_{i-1/2}^3},$$
where $r_c$ is `x1v` (`radial_centers` in `src/coord/spherical_polar.cpp`), the volume centroid. Gravity is
$g_1 = $ `grav1` $< 0$ and the potential is $\phi = -g_1 r$, so $\langle\phi\rangle_{V_i} = \phi(r_{c,i})$ exactly.
$F_f$ is the x1 mass-flux density at face $f$ and $G = A F = r^2 F$ the mass flow per steradian. Write
$r_\pm$ for a cell's faces, $h = r_+ - r_-$, $\bar r = (r_+ + r_-)/2$, $\delta = r_c - \bar r$.

Mass: $V_i\,\dot\rho_i = -(A_+F_+ - A_-F_-)$. Discrete potential energy: $\mathrm{PE} = \sum_i \rho_i\,\phi(r_{c,i})\,V_i$
(the exact $\int\rho\phi\,dV$ for piecewise-constant $\rho$).

Exact gravity work per unit volume: $g_1\langle F\rangle_V = \frac{g_1}{V}\int_{r_-}^{r_+} r^2F\,dr$.

## 2. The face-form booking

The code books, per unit volume and time,
$$W_i = \frac{1}{V_i}\Big[\phi_{c,i}\,(A_+F_+ - A_-F_-) - (A_+\phi_+F_+ - A_-\phi_-F_-)\Big]
      = \frac{g_1}{V_i}\Big[(r_+ - r_c)\,r_+^2F_+ + (r_c - r_-)\,r_-^2F_-\Big]. \tag{1}$$
The second form follows from $\phi_c - \phi_\pm = g_1(r_\pm - r_c)$.

**Leading error.** Expand $F = F_0 + F_1 s + F_2 s^2/2$, $s = r - \bar r$, and integrate exactly (replica §1):
$$\frac{W - g_1\langle F\rangle_V}{g_1} = \underbrace{\frac{h^2}{12}F''}_{\text{Cartesian}}
  + \underbrace{\frac{h^2}{6\bar r}F' - \frac{h^2}{6\bar r^2}F}_{\text{curvature}} + O(h^4). \tag{2}$$
The curvature pair is the term the radial ablation found (with its sign: face minus exact). The Cartesian term is the trapezoid
error present on every grid.

Equivalent form in $G = r^2F$: $\;W - g_1\langle F\rangle_V = \frac{g_1}{V}\big[-\delta h\,G' + \frac{h^3}{12}G''\big] + O(h^5)$,
with $\delta = h^2/(6\bar r) + O(h^4)$. So (1) is **exact for $G$ = const**, a steady divergence-free radial mass
flow ($F \propto r^{-2}$): $W = g_1 G h / V = g_1\langle F\rangle_V$ for any $h$.

## 3. Exact replacement weights

A two-point booking $W = g_1(a F_+ + b F_-)$ that is exact for $F \in \{1, r\}$ on the $r^2$ measure needs
$a + b = 1$ and $a r_+ + b r_- = \langle r\rangle_V = r_c$. The solution is unique:
$$a = \frac{r_c - r_-}{h},\qquad b = \frac{r_+ - r_c}{h}. \tag{3}$$
Note that (3) puts the weight $(r_c - r_-)$ on $F_+$, where (1) puts $(r_+ - r_c)r_+^2/V$ on $F_+$: the roles
of the two offsets are swapped. Its error (replica §1, $F$ expanded to $s^4$; the odd orders vanish):
$$\frac{W_{\rm exact} - g_1\langle F\rangle_V}{g_1} = \frac{h^2}{12}F''
  + h^4\Big(-\frac{F''}{360\,\bar r^2} + \frac{F'''}{360\,\bar r} + \frac{F''''}{480}\Big) + O(h^6).\tag{4}$$
Both curvature terms of (2) are gone; the Cartesian term stays. Cartesian limit: $r_c = \bar r$, so
$a = b = 1/2$, and (1) gives $(h/2)\cdot A/V = 1/2$ as well, so (3) reduces to the old weights there.

## 4. Discrete E+PE conservation

**Lemma (what conservation requires).** Let every cell book a two-point work $W_i V_i = g_1(\alpha_i F_{i+1/2} +
\beta_i F_{i-1/2})$. With closed walls ($F = 0$ on the two wall faces) the E+PE change per unit time is
$$\frac{d}{dt}(E + \mathrm{PE}) = \sum_i W_iV_i + \sum_i \phi_{c,i}\,V_i\dot\rho_i
 = g_1\sum_{f\ \rm interior} F_f\Big[\underbrace{\alpha_i + \beta_{i+1}}_{S_f} - A_f\,(r_{c,i+1} - r_{c,i})\Big],$$
using $\sum_i\phi_{c,i}V_i\dot\rho_i = \sum_f A_fF_f(\phi_{c,i+1} - \phi_{c,i})$ (cell $i$ below face $f$, cell $i+1$ above),
and $\phi_{c,i+1} - \phi_{c,i} = -g_1(r_{c,i+1} - r_{c,i})$. The interior $F_f$ are independent, so E+PE is
conserved for every flux field **iff** $S_f = A_f(r_{c,i+1} - r_{c,i})$ on every interior face. The other energy
fluxes telescope to the walls and drop out.

Equivalently: conserving two-point bookings are exactly the face form with an arbitrary face potential,
$W_iV_i = \phi_{c,i}(A_+F_+ - A_-F_-) - (A_+\phi(s_+)F_+ - A_-\phi(s_-)F_-)$, where $s_f$ (the split point of
the face's total weight between the two cells) is free. The code uses $s_f = r_f$.

**Face form (1):** $S_f = A_f(r_f - r_{c,i}) + A_f(r_{c,i+1} - r_f) = A_f(r_{c,i+1} - r_{c,i})$. Conserved
identically, for any grid (replica: `face : 0`).

**Exact weights (3):** $S_f = \frac{V_i}{h_i}(r_{c,i} - r_{i-1/2}) + \frac{V_{i+1}}{h_{i+1}}(r_{i+3/2} - r_{c,i+1})$.
For uniform $h$ and the face at $r_f = R$ the exact closed form is
$$S_f - A_f(r_{c,i+1} - r_{c,i}) = \frac{h^3\,(3R^4 - R^2h^2 + h^4/6)}{9R^4 - 3R^2h^2 + h^4}
 = \frac{h^3}{3} + O(h^5/R^2) \neq 0, \tag{5}$$
relative to the required value: $h^2/(3R^2) + h^4/(18R^4)$. Hence the E+PE change per unit time is
$$\frac{d}{dt}(E+\mathrm{PE}) = g_1\,\frac{h^3}{3}\sum_f F_f + O(h^5) = g_1\,\frac{h^2}{3}\int F\,dr + O(h^4)
\quad\text{(per steradian)},$$
nonzero whenever the column-integrated vertical mass flux density is nonzero. **The exact weights do not keep the
face form's conservation.** The defect is $O(h^2/R^2)$ of the work, the same order as the curvature terms in (2)
that they remove.

**No conserving two-point booking removes both curvature terms.** Take the conserving family with
$s_f = r_f + \lambda h^2/r_f$ (any smooth $O(h^2)$ shift has this leading form locally). Replica §1:
$$\frac{W_\lambda - g_1\langle F\rangle_V}{g_1} = \frac{h^2}{12}F'' + \Big(\lambda + \frac16\Big)\frac{h^2}{\bar r}F'
 + \Big(\lambda - \frac16\Big)\frac{h^2}{\bar r^2}F + O(h^4).$$
$\lambda = 1/6$ removes the $F$ term and doubles the $F'$ term; $\lambda = -1/6$ removes the $F'$ term and doubles
the $F$ term; no $\lambda$ removes both. For a general shift $s_f = r_f + h^2\varepsilon(r_f)$ the two conditions are
$\varepsilon = -1/(6r)$ and $\varepsilon' + 2\varepsilon/r = +1/(6r^2)$, which contradict each other
($-1/(6r)$ gives $-1/(6r^2)$). Only $\lambda = 0$ (the current code) is exact for divergence-free flow ($G$ = const),
because only a constant shift keeps $s_+ - s_- = h$.

**Wider stencils (leading order, a remark not a proof).** Any booking that equals the exact cell average up to the
common Cartesian term books the exact total work, and
$\sum_i g_1\!\int_i G\,dr + \sum_i\phi_{c,i}\dot M_i = -g_1\sum_i\int_i (r - r_{c,i})\,G'\,dr$, whose curvature part is
$+g_1\frac{h^2}{6}\int G'/r\,dr = g_1\frac{h^2}{6}\int F\,dr$ for closed walls. This is a column integral, not a sum of
local face differences, so no local telescoping correction of the *weights* cancels it (the potential is held at
$\mathrm{PE}_d$ here). The Cartesian part, $(h^2/12)[G']_{\rm walls}$, is a pure wall term, so it is no real conflict:
a wall closure plus a modified discrete PE removes it with exact conservation (the corrected-PE work behind
`SNAP_GRAVITY_WORK_RADIAL_EXACT`, and an independent four-point booking that is $O(h^4)$ in every cell including the
walls). Only the face form with $\mathrm{PE}_d$ pays the trapezoid error to conserve.

## 5. Numbers (replica §2, §3)

One step, closed column, random interior $F_f$, $g_1 = -10$, column $3H$, $\rho = 1$:

| $R/H$ | $h/R$ | face: $\Delta(E{+}PE)/\sum\lvert WV\rvert$ | exact (3) | cons $\lambda=\pm1/6$ | exact, $\Delta(E{+}PE)/(E{+}PE)$ |
|---|---|---|---|---|---|
| 5 | 9.4e-3 | 3.9e-15 | **-1.3e-6** | 3.9e-15 | -4.1e-8 |
| 5 | 2.3e-3 | -9.8e-16 | **-5.7e-8** | -1.0e-15 | -2.5e-9 |
| 1000 | 4.7e-5 | -6.5e-13 | **+1.2e-10** | -6.5e-13 | +2.8e-14 |
| 1 | 9.4e-2 | 3.1e-15 | **+2.0e-4** | 3.2e-15 | +2.2e-5 |

The face form and both $\lambda$ variants stay at the floor of these sums ($\le 4\times10^{-15}$ for $R \le 5H$; $6.5\times10^{-13}$
at $R = 1000H$, where the $r^2$-weighted terms span a wide range: a cancellation floor, not shown to be round-off); the exact weights fail a 1e-14 relative gate
at every $R$ tested, by both measures (at $R = 1000H$: 2.8e-14 of $E{+}PE$, 1.2e-10 of the step's work).

Convergence against the exact $r^2$-measure cell average, $F = e^{-(r-r_0)/H}\sin(\pi(r-r_0)/4H)$:
- $h$-ladder at $R = 5H$, $\max|e - \tfrac{h^2}{12}F''|$: face 1.18e-3 → 6.27e-6 over nz 16 → 256, slope 1.75 → 1.97
  ($h^2$); exact 2.42e-5 → 3.86e-10, slope 3.96 → 4.00 ($h^4$).
- $R$-ladder at $h/H = 1/16$: face $\times R$ = 4.74e-4, 4.77e-4, 4.60e-4 at $R/H$ = 5, 50, 500 ($h^2/R$); exact at the
  $h^4$ floor 4e-8.
So the exact weights pass oracle (e) and fail oracle (c).

## 6. Options (not chosen here)

| option | curvature terms in (2) | E+PE per step | divergence-free flow exact |
|---|---|---|---|
| A. current face form ($\lambda = 0$) | both present | round-off | yes |
| B. exact weights (3) | both removed | defect $g_1(h^2/3)\int F\,dr$, rel $h^2/(3R^2)$ | no ($O(h^2/R^2)$) |
| C. conserving, $\lambda = +1/6$ | $F$ term removed, $F'$ doubled | round-off | no |
| D. conserving, $\lambda = -1/6$ | $F'$ term removed, $F$ doubled | round-off | no |
| E. B plus a global E+PE fixer (as gravity-work-fixer does for cell) | both removed | round-off after the fix | no |

Hypothesis (unverified): the ablation's sign surprise (shift +1.05 / +1.19 against the predicted -1.07 / -1.23)
comes from option B's defect, which is the same order as the term removed. The check that settles it: rerun the
ablation arm with option D (conserving, $F'$ term only) and with C; if each moves the excess by its own share of
the predicted term, the defect explains the rest.
