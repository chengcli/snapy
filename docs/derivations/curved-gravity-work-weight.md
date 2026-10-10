# Face-form gravity work on a radial grid: exact weights versus discrete E+PE conservation

Scope: the x1 face-form gravity work (`gravity-work: face`) on a spherical-polar grid, explicit path
(`src/hydro/hydro_forward.cpp`, the `face_gravity_work` block) and implicit path (`src/implicit/implicit_hydro.cpp`,
`work_lo`/`work_hi`, which carry the same weights times 1/2 for the face average of the two cell momenta).
Base: 8cea3ae (the pre-rebase head of #293, merged to main as aea71ed). Every closed form below is checked by `curved_gravity_work_weight.py` (sympy + numpy;
`python docs/derivations/curved_gravity_work_weight.py` prints every number quoted; "replica" below).

**Result.** The exact $r^2$-measure weights remove both $O(h^2/\bar r)$ error terms, but they do **not** conserve
discrete E+PE: the per-face defect is $h^3/3$ (relative $h^2/(3R^2)$), the same order as the terms they remove.
No two-point weight choice that keeps the face form's conservation of E+PE$_d$ removes both terms; one or the other, not both.

**Option F (§7).** Conserving a corrected discrete PE, $P = \mathrm{PE}_d - g_1\sum_iV_i\sigma_i^2 s_i[\rho]$ (exact
to $O(h^4)$), instead of $\mathrm{PE}_d$ resolves the conflict: the work $W = W_{\rm face} + g_1\sigma^2 s[\dot\rho]$ is
$O(h^4)$-exact and conserves E+$P$ to round-off by construction. It changes Cartesian too and replaces (not adds to)
the cp3/cp5/weno5 curvature flux. §8 is the Cartesian case written to be ported on its own: there F removes the
$O(h)$ wall-cell error that face+H leaves, and leaves the interior unchanged to $O(h^4)$. §9 compares F with the $\bar r$ potential form; §10 lists
every place that computes a PE under the switch `SNAP_GRAVITY_WORK_RADIAL_EXACT`, which selects F.

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
$\mathrm{PE}_d$ here; §7 lifts that). The Cartesian part, $(h^2/12)[G']_{\rm walls}$, is a pure wall term, so it is no real conflict:
a wall closure plus a modified discrete PE removes it with exact conservation (§§7–8, the corrected-PE work behind
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

## 6. Two-point options (superseded by option F, §7)

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

## 7. Option F: a corrected discrete PE and the work it implies

The lemma of §4 is about two-point bookings measured against $\mathrm{PE}_d = \sum_i V_i\rho_i\phi(r_{c,i})$. That
functional is itself only $O(h^2)$-accurate: an exact work conserves the exact PE, not $\mathrm{PE}_d$. Option F
changes the functional, not the weights. Checked by `optionF_replica.py` (numpy;
`python docs/derivations/optionF_replica.py` prints every number quoted; "F-replica" below; entries at the
round-off or cancellation floor, $\lesssim 10^{-12}$, differ in their leading digits between numpy builds).

**The functional.** Expand $\rho$ about the centroid inside cell $i$, $\rho = \rho(r_c) + \rho'(r - r_c) +
\tfrac12\rho''(r - r_c)^2 + \dots$. With $\phi = -g_1 r$,
$$\int_i\rho\phi\,dV = -g_1\Big[r_c M_i + \int_i\rho\,(r - r_c)\,dV\Big]
 = V_i\big[\rho_i\phi(r_c) - g_1\,\sigma_i^2\,\rho'(r_c)\big] + O(h^4 V),$$
where $M_i = \rho_iV_i$ (exact), $\sigma_i^2 = \langle(r - r_c)^2\rangle_{V_i}$ (the $r^2$-measure variance; the
third central moment is $O(h^4/R)$). Replace $\rho'$ by $s_i[\rho]$, the slope at $r_{c,i}$ of the quadratic through
the cell values at the centroids (interior: cells $i-1, i, i+1$; the two cells at each closed wall: the one-sided
stencil $0,1,2$ or $n-3,n-2,n-1$). The cell values sit $O(h^2)$ off the point values by a smooth amount, so
$s_i = \rho'(r_c) + O(h^2)$, and
$$P[\rho] = \sum_i V_i\big[\rho_i\,\phi(r_{c,i}) - g_1\,\sigma_i^2\,s_i[\rho]\big] = \int\rho\phi\,dV + O(h^4).\tag{6}$$
On a uniform centroid spacing this is a $\kappa$ form ($\kappa_i \propto \sigma_i^2/(r_{c,i+1} - r_{c,i-1})$);
the quadratic stencil keeps it $O(h^4)$ on the non-uniform centroid spacing as well. Per steradian, cancellation-free:
$\sigma^2 = (\bar r^2h^3/12 + h^5/80)/V - \delta^2$, $\delta = r_c - \bar r = \bar r h^3/(6V)$.

**The work.** Define $W_iV_i := -\dot P_i - (A_+\phi_+F_+ - A_-\phi_-F_-)$ with $\dot\rho$ from the discrete continuity
$V\dot\rho = -(A_+F_+ - A_-F_-)$. Since $P$ is linear in $\rho$,
$$W_i = W_{{\rm face},i} + g_1\,\sigma_i^2\,s_i[\dot\rho]. \tag{7}$$
The first term is (1); the second couples faces $i-3/2 \dots i+3/2$ (wall cells: the first four faces).

**Conservation (exact, by construction).** $\sum_iW_iV_i + \dot P = -\sum_i\Delta_i(A\phi F) = 0$ for closed walls,
for every flux field and every grid. E+P is conserved to round-off; E+PE$_d$ is not (it changes by
$-g_1\sum_iV_i\sigma_i^2s_i[\dot\rho]$, an $O(h^2)$ amount, which is the point: $\mathrm{PE}_d$ is the wrong target).
With the switch on, `gravity_work_defect()` and every E+PE$_d$ check therefore measure the wrong invariant; check E+P.

**Accuracy.** The exact cell budget is $g_1V_i\langle F\rangle = -\dot{\mathrm{PE}}_i - \Delta_i(A\phi F)$, so
$W_iV_i - g_1V_i\langle F\rangle = \frac{d}{dt}(\mathrm{PE}_i - P_i) = O(h^4V)$ by (6). Both curvature terms of (2)
and the Cartesian term go: to leading order $\sigma^2s[\dot\rho] = -\tfrac{h^2}{12}(F'' + 2F'/r - 2F/r^2)$, which is
minus (2).

**Properties.**
1. Exact to $O(h^4)$, both $h^2/R$ terms and the $h^2F''/12$ term removed, interior and wall cells (table below).
2. Cartesian changes. There $\sigma^2 = h^2/12$ and (7) removes the trapezoid term $h^2F''/12$ that the face form
   keeps; E+P is conserved and E+PE$_d$ is not. Change at nz 64: $1.6\times10^{-3}$ of $\max|W|$ (interior),
   $1.9\times10^{-3}$ (wall cells). F applies on Cartesian too, under the same switch; the Cartesian case is
   written out on its own in §8.
3. Closed walls need nothing beyond the one-sided slope stencil: conservation does not use the wall fluxes' value
   (they are zero), and the wall cells converge at $h^{4.4-4.6}$. At an internal block edge the same one-sided
   stencil makes $P$ a sum of per-block functionals; the faces shared by two blocks still telescope, so global
   E+P stays exact. **Limit: block seams.** A column split in x1 therefore conserves its own $P$, not the
   one-block $P$: the seam cells take the one-sided slope where one block takes the centred one, so the states
   differ there. Taking the centred slope at a seam needs the neighbour's $\Delta\rho$ in the ghost cells,
   and that is not available without a new exchange: the ghost $\Delta\rho$ comes from the ghost-face mass
   fluxes, which the reconstruction computes only on the owned faces, and the logged `pe=` reads ghost
   $\rho$ that need not be current when the diagnostics run. All three sites (explicit work, VIC work, logged
   `pe=`) call the same `corrected_pe_work`, so the booked work and the logged $P$ agree per block and E+P
   stays at round-off on a split column. Measured (`tests/test_x1_seam_split.cpp`, isothermal spherical
   column with a seam density bump, $t$ fixed): max relative 1-vs-2-block gap
   $8.0\times10^{-7}$, $6.3\times10^{-8}$, $2.3\times10^{-9}$ at $n_z$ = 32, 64, 128 (order 3.7, 4.8,
   i.e. $O(h^4)$, as $\sigma^2 \sim h^2$ times the $O(h^2)$ difference of the two slopes); switch off:
   0, bit for bit, so the whole gap is the one-sided seam slope; E+P drift on the split column $\le 5\times10^{-15}$. In 2-D/3-D, x2/x3 fluxes book no gravity work and move mass between columns at
   the same radius; on spherical-polar $V = \Omega(\theta,\varphi)V_r(r)$ and $\sigma^2$, $s$ depend on $r$ only, so
   $\sum_{\rm columns}P$ depends only on each shell's total mass, which they conserve: E+P stays exact (§8.6 for Cartesian).
4. Numbers below.

**Interaction with the cp3/cp5/weno5 curvature flux.** `hydro_forward.cpp` already subtracts
$g_1\,\Delta(AH)/V$, $H_f = (r_{c,i} - r_{c,i-1})(m_i - m_{i-1})/12$, zero at physical boundaries, for these
reconstructions. It is a divergence (keeps E+PE$_d$) and removes $\tfrac{h^2}{12}(r^{-2}(r^2F')')
= \tfrac{h^2}{12}(F'' + 2F'/r)$. So, on curved grids, face+H leaves $-\tfrac{h^2}{6\bar r^2}F$ in the interior,
and, because $H = 0$ at the walls, its wall cells are only first order ($h^1$, Cartesian and spherical alike).
F already removes the $F''$ term, so **F must replace H, not add to it**: F+H is back to $O(h^2)$ (double count).

**Numbers** (F-replica §1, §2, §5; $g_1 = -10$; closed walls).

| one step, random interior $F_f$ | face: $\Delta(E{+}P)/\sum\lvert WV\rvert$ | face: $\Delta(E{+}PE_d)$ | F: $\Delta(E{+}P)$ | F: $\Delta(E{+}PE_d)$ | F: $\Delta(E{+}P)/(E{+}P)$ |
|---|---|---|---|---|---|
| sph $R = 5H$ | 8.0e-3 | -3.0e-15 | **-1.3e-15** | -7.3e-3 | -2.0e-16 |
| sph $R = 1000H$ | 3.5e-3 | 5.3e-13 | **-5.7e-14** | -3.2e-3 | -4.6e-17 |
| sph $R = H$ | -2.1e-2 | 9.0e-16 | **1.3e-15** | 1.9e-2 | 1.2e-15 |
| Cartesian | 5.8e-3 | -3.5e-15 | **3.7e-15** | -5.2e-3 | 5.9e-16 |

Oracle (c) measured against $P$: F passes at round-off ($\le 1.2\times10^{-15}$ of E+P); the face form does not
conserve $P$ (it conserves $\mathrm{PE}_d$).

| $\max\lvert W - g_1\langle F\rangle\rvert/\lvert g_1\rvert$, $R = 5H$, nz 16 → 256 | interior | slope | wall cells | slope |
|---|---|---|---|---|
| face (1) | 3.2e-3 → 2.5e-5 | 1.4 → 1.95 | 5.8e-3 → 2.5e-5 | 1.9 → 2.0 |
| F (7) | 3.5e-5 → 5.4e-10 | 3.99 → 4.00 | 5.4e-5 → 1.8e-10 | 4.6 → 4.4 |
| face + H (cp3/cp5/weno5 today), nz 32 → 256 | 2.0e-5 → 3.3e-7 | 2.0 | 8.0e-3 → 1.0e-3 | 1.0 |
| F + H (double count) | 1.2e-3 → 2.5e-5 | 1.9 | 6.5e-3 → 9.9e-4 | 0.96 |
| Cartesian F | 6.9e-5 → 1.2e-9 | 4.00 | 1.3e-4 → 1.9e-9 | 4.0 |

$R$-ladder at nz 64 ($h = H/16$), F interior | wall: 1.4e-7 | 9.0e-8, 2.8e-7 | 4.8e-7, 2.9e-7 | 5.0e-7, 3.0e-7 | 5.0e-7
at $R/H$ = 5, 50, 500, 5000 (flat: no $1/R$ term left), against face $\times R$ = 1.8e-3, 2.1e-2, 0.21, 2.1 (the
face form's interior error at this $h$ is dominated by the $h^2F''/12$ term, which does not fall with $R$).

**Settling check (can a column-integrated diagnostic tell the options apart?).** Same profile,
closed walls, column sums normalised by $\sum\lvert g_1\langle F\rangle\rvert V$ (F-replica §4):

| $R/H$, nz | A face | B exact | C ($\lambda = +1/6$) | D ($\lambda = -1/6$) | F |
|---|---|---|---|---|---|
| 5, 32: work error | +1.420e-3 | +1.292e-3 | +1.420e-3 | +1.420e-3 | -7.3e-6 |
| 5, 32: PE$_d$ defect | 1e-16 | **-1.29e-4** | 3e-16 | 1e-16 | -1.43e-3 |
| 1000, 64: work error | +5.249e-4 | +5.249e-4 | +5.249e-4 | +5.249e-4 | -8.0e-7 |
| 1000, 64: PE$_d$ defect | 3e-14 | -1.3e-9 | 3e-14 | 3e-14 | -5.26e-4 |

A, C and D book the same column total (they differ only in how it is split between cells), so a column-integrated
diagnostic cannot separate them; only B differs, and by exactly its E+PE$_d$ defect. At $R = 5H$, nz 32 that defect is
9% of the column work error. So if the ablation's $(d)$ arm used the exact weights, the shift it saw includes B's
non-conservation; a C or D arm would show the column total of A. The real ablation's $\varepsilon$ harness was not
rebuilt here. F is the only option whose column total is right ($-7\times10^{-6}$ against $+1.4\times10^{-3}$): its
PE$_d$ "defect" is the $O(h^2)$ error of PE$_d$ itself, equal and opposite to A's work error.

## 8. Option F on a Cartesian $x_1$ grid, self-contained ($R \to \infty$)

This section can be read and ported without §§1–7. Every identity is checked in `optionF_replica.py` (§1, §6, §7) and in `tests/test_gravity_work_radial_exact.py` (closed columns in the code).

**8.1 Grid and notation.** One column, $x_1 = z$, cells $i = 0,\dots,n-1$ with faces $z_{i-1/2} < z_{i+1/2}$,
width $h_i = z_{i+1/2} - z_{i-1/2}$, centre $z_i = (z_{i-1/2} + z_{i+1/2})/2$ (the volume centroid, `x1v`). All
quantities are per unit horizontal area. $\rho_i$ is the cell-average total density (dry plus every condensate
and tracer that carries mass, the same sum the code's x1 mass flux uses). $F_{i+1/2}$ is the x1 mass flux through
face $i+1/2$ as the Riemann solver returns it; closed walls mean $F_{-1/2} = F_{n-1/2} = 0$. Gravity is
$g_1 = $ `grav1` $< 0$, the potential $\phi(z) = -g_1 z$. $E$ is the total energy density (the code's `IPR` row).

**8.2 Discrete continuity.** The x1 part of the mass update is
$$h_i\,\dot\rho_i = -(F_{i+1/2} - F_{i-1/2}). \tag{8.1}$$
In the code $\dot\rho_i\,dt$ is the stage's x1 density increment, `-dt * vertical_mass_div`.

**8.3 What the code books today (face form).** `hydro_forward.cpp` books, per cell,
$$W^{\rm face}_i h_i = g_1\big[(z_{i+1/2} - z_i)F_{i+1/2} + (z_i - z_{i-1/2})F_{i-1/2}\big],$$
which is the same as
$$W^{\rm face}_i h_i = -h_i\dot\rho_i\,\phi(z_i) - \big(\phi_{i+1/2}F_{i+1/2} - \phi_{i-1/2}F_{i-1/2}\big),
\qquad \phi_{i\pm1/2} = \phi(z_{i\pm1/2}). \tag{8.2}$$
(Insert (8.1) and $\phi = -g_1z$: the right side is $-g_1z_i(F_+ - F_-) + g_1(z_+F_+ - z_-F_-) =
g_1[(z_+ - z_i)F_+ + (z_i - z_-)F_-]$. The code's form is `dt*(phi_cell*div - div(phi_face*F))` with
`div` $= (F_+ - F_-)/h_i$, the same expression divided by $h_i$.)
Summing (8.2) over the column, the bracket telescopes to $\phi F$ at the two walls, which is zero, so
$\sum_iW^{\rm face}_ih_i + \tfrac{d}{dt}\mathrm{PE}_d = 0$ with $\mathrm{PE}_d = \sum_ih_i\rho_i\phi(z_i)$: the face form
conserves $E + \mathrm{PE}_d$.
*Its error.* The exact cell work is $g_1\langle F\rangle_i = \tfrac{g_1}{h_i}\int_i F\,dz$. On a uniform grid
$W^{\rm face}_i = g_1(F_{i+1/2} + F_{i-1/2})/2$, the trapezoid rule, so
$W^{\rm face}_i - g_1\langle F\rangle_i = g_1\tfrac{h^2}{12}F''(z_i) + O(h^4)$ in every cell, walls included.
*With the cp3/cp5/weno5 curvature flux H* (also in `hydro_forward.cpp`):
$W^{\rm face+H}_i = W^{\rm face}_i - g_1(H_{i+1/2} - H_{i-1/2})/h_i$, $H_{i+1/2} = \tfrac{z_{i+1} - z_i}{12}(m_{i+1} - m_i)$,
$m = \rho v_1$ at the cell centres, $H = 0$ on the walls. In the interior this subtracts $g_1\tfrac{h^2}{12}F''$ and
the result is $O(h^4)$. In a wall cell only one H face acts: $H_{1/2}/h \approx \tfrac{h}{12}F'(z_{\rm wall})$ is left,
an $O(h)$ error, $\approx g_1\tfrac{h}{12}F'(z_{\rm wall})$ (F-replica §6: $1.65\times10^{-2}$ at nz 16 for $F' = \pi/4$,
$h = 1/4$, where $h F'/12 = 1.64\times10^{-2}$).

**8.4 The exact potential energy of a cell, and its discrete stand-in $P$.** Let $\rho(z)$ be smooth and expand about
$z_i$: $\rho = \rho(z_i) + \rho'(z - z_i) + \tfrac12\rho''(z - z_i)^2 + \tfrac16\rho'''(z - z_i)^3 + \dots$ Then, with
$\int_i(z - z_i)\,dz = 0$, $\int_i(z - z_i)^2dz = h^3/12$, $\int_i(z - z_i)^4dz = h^5/80$,
$$\int_i\rho\phi\,dz = -g_1\Big[z_i\int_i\rho\,dz + \int_i\rho\,(z - z_i)\,dz\Big]
= h_i\Big[\rho_i\phi(z_i) - g_1\tfrac{h_i^2}{12}\rho'(z_i)\Big] - g_1\tfrac{h_i^5}{480}\rho'''(z_i) + \dots \tag{8.3}$$
so the exact cell PE is $h_i[\rho_i\phi(z_i) - g_1\sigma_i^2\rho'(z_i)] + O(h^5)$ with $\sigma_i^2 = h_i^2/12$, the variance
of $z$ over the cell. $\mathrm{PE}_d$ drops the $\sigma^2\rho'$ term and is therefore only $O(h^2)$ accurate per unit
length; that is the whole reason the face form, which conserves $\mathrm{PE}_d$ exactly, keeps an $O(h^2)$ work error.
Replace $\rho'(z_i)$ by a slope built from the cell averages,
$$s_i[\rho] = \text{derivative at } z_i \text{ of the quadratic through } (z_k,\rho_k),\ k \in S_i,\qquad
S_i = \{i-1,i,i+1\}\ (0 < i < n-1),\ S_0 = \{0,1,2\},\ S_{n-1} = \{n-3,n-2,n-1\}. \tag{8.4}$$
On a uniform grid: $s_i = (\rho_{i+1} - \rho_{i-1})/(2h)$ inside, $s_0 = (-3\rho_0 + 4\rho_1 - \rho_2)/(2h)$,
$s_{n-1} = (3\rho_{n-1} - 4\rho_{n-2} + \rho_{n-3})/(2h)$; the non-uniform weights are in `centroid_slope`
(`src/hydro/gravity_work_radial.hpp`). Cell averages differ from point values by $\tfrac{h^2}{24}\rho''$, a smooth
amount, so $s_i = \rho'(z_i) + O(h^2)$ (interior and wall alike), and $\sigma^2 s_i = \sigma^2\rho' + O(h^4)$. Define
$$P[\rho] = \sum_i h_i\Big[\rho_i\,\phi(z_i) - g_1\,\sigma_i^2\,s_i[\rho]\Big]
 = \int\rho\phi\,dz + O(h^4). \tag{8.5}$$
$P$ is linear in $\rho$ and needs no data outside the column (no ghost cells): that is the whole wall closure.

**8.5 The work that conserves $E + P$.** Define the booked work by the same recipe as (8.2), with $P$ in place of
$\mathrm{PE}_d$: $W_ih_i := -\dot P_i - (\phi_{i+1/2}F_{i+1/2} - \phi_{i-1/2}F_{i-1/2})$, where $P_i$ is the $i$-th
summand of (8.5). Because $P_i$ is linear in $\rho$, $\dot P_i = h_i[\dot\rho_i\phi(z_i) - g_1\sigma_i^2s_i[\dot\rho]]$,
and comparing with (8.2),
$$W_i = W^{\rm face}_i + g_1\,\sigma_i^2\,s_i[\dot\rho],\qquad \dot\rho \text{ from (8.1)}. \tag{8.6}$$
This is all the switch adds (`corrected_pe_work`; explicit: in `hydro_forward.cpp` on `-dt*vertical_mass_div`;
VIC: in `implicit_hydro.cpp` on the solve's own density change `du - du0`, rows IDN and the condensates, partly inside
the operator, §10). It changes
only the energy row: mass and momentum are untouched, so a state at rest ($F \equiv 0$, $\dot\rho = 0$) gets exactly
zero from it.
*Uniform-grid closed forms* (substitute (8.1) into (8.4); F-replica §7 checks them against the matrix form to
$8\times10^{-16}$):
$$\frac{W_i}{g_1} = \frac{F_{i+1/2} + F_{i-1/2}}{2} - \frac{F_{i+3/2} - F_{i+1/2} - F_{i-1/2} + F_{i-3/2}}{24}
\quad(0 < i < n-1),$$
$$\frac{W_0}{g_1} = \frac{19F_{1/2} - 5F_{3/2} + F_{5/2}}{24},\qquad
\frac{W_{n-1}}{g_1} = \frac{19F_{n-3/2} - 5F_{n-5/2} + F_{n-7/2}}{24}.$$
(Cell 1 and cell $n-2$ use the interior form with the wall flux $F_{\mp1/2} = 0$.)
Hand check of the wall form: for $F = z, z^2, z^3$ on the cell $[0,h]$ with $F(0) = 0$ it gives $h/2$, $h^2/3$, $h^3/4$,
the exact averages; for $z^4$ it gives $5h^4/6$ against $h^4/5$, so the wall cell is exact through cubics, error
$O(h^4)$. Interior: expanding about $z_i$, the trapezoid is $F + \tfrac{h^2}{8}F''$, the 4-face difference is
$2h^2F''$, so $W_i/g_1 = F + \tfrac{h^2}{24}F'' + O(h^4) = \langle F\rangle_i + O(h^4)$.

**8.6 Why $E + P$ is conserved (exactly, not to truncation error).** Sum (8.6)·$h_i$ over the column using the
definition in 8.5: $\sum_iW_ih_i + \dot P = -\sum_i(\phi_{i+1/2}F_{i+1/2} - \phi_{i-1/2}F_{i-1/2}) =
-(\phi_{n-1/2}F_{n-1/2} - \phi_{-1/2}F_{-1/2}) = 0$. Nothing else is used: not the stencil weights, not the grid
spacing, not the size of $F$. The energy fluxes other than gravity work telescope as before. So in exact arithmetic
$E + P$ is constant per stage; every RK stage is $u \leftarrow a u_0 + b u + c\,dt\,\dot u$ and $E + P$ is linear in $u$,
so it is constant per step too, to round-off. $E + \mathrm{PE}_d$ is not conserved any more: it changes by
$-g_1\sum_ih_i\sigma_i^2s_i[\Delta\rho]$ per stage, an $O(h^2)$ amount that is the error of $\mathrm{PE}_d$ itself.
*Horizontal directions.* In 2-D/3-D, x2/x3 fluxes move mass between columns at the same level $i$ and book no
gravity work. $\sigma_i^2$, $h_i$ and the stencil (8.4) are the same in every column and $s$ is linear, so
$\sum_{\rm columns}P = \sum_ih_i[\phi(z_i)\bar M_i - g_1\sigma_i^2s_i[\bar M]]$ with $\bar M_i$ the level's total mass,
which x2/x3 fluxes conserve (periodic or closed lateral boundaries): $E + P$ stays exact.
*Several blocks in x1.* Each block uses its own one-sided stencil at its x1 ends, so $P = \sum_{\rm blocks}P_b$; at a
shared face both blocks use the same $F$ and $\phi$, the $\phi F$ terms cancel between them, and global $E + P$ is exact.
*Rest.* The added term is proportional to $\dot\rho$; at discrete hydrostatic rest it is zero to round-off and the
momentum balance is not touched.

**8.7 Accuracy.** The exact cell budget is $g_1h_i\langle F\rangle_i = -\tfrac{d}{dt}\mathrm{PE}^{\rm exact}_i -
(\phi F)\big|_{i-1/2}^{i+1/2}$; subtract the definition of $W_i$: $W_ih_i - g_1h_i\langle F\rangle_i =
\tfrac{d}{dt}(\mathrm{PE}^{\rm exact}_i - P_i) = O(h^5)$ by (8.5), i.e. $O(h^4)$ per unit length in every cell, wall
cells included. Measured (F-replica §6, $|W - g_1\langle F\rangle|/|g_1|$, profile $F = e^{-z}\sin(\pi z/4)$ on $[0,4]$,
closed walls, nz 16 → 128):

| Cartesian | first cell (wall) | slope | last cell (wall) | slope | interior | slope |
|---|---|---|---|---|---|---|
| face + H (today, cp3/cp5/weno5) | 1.65e-2 → 2.05e-3 | 1.00 | 3.04e-4 → 3.75e-5 | 1.00 | 3.7e-5 → 1.0e-8 | 4.00 |
| F (8.6) | 1.29e-4 → 3.08e-8 | 4.02 | 1.46e-6 → 5.42e-10 | 3.96 | 6.9e-5 → 1.85e-8 | 4.00 |
| face (plm, no H) | — | 2 | — | 2 | — | 2 |

(The last cell is small because $F$ has decayed by $e^{-4}$ there.) The change F makes against today's face+H on a
Cartesian weno5 deck (F-replica §7, $\max|\Delta W|/\max|W|$): interior $1.2\times10^{-4} \to 3.2\times10^{-8}$
(slope 4.00, so the interior is unchanged to $O(h^4)$), first wall cell $6.2\times10^{-2} \to 7.7\times10^{-3}$ and
last wall cell $1.2\times10^{-3} \to 1.4\times10^{-4}$ (slope 1: the change is the removal of face+H's $O(h)$ wall error).
Against plm decks (no H) the change is the $O(h^2)$ trapezoid term in every cell.

**8.8 Porting checklist.** (i) The sum of mass rows that the x1 mass flux carries, as $\dot\rho$. (ii) $\sigma^2 = h^2/12$
per cell. (iii) The 3-point slope (8.4), one-sided at both x1 ends of every block. (iv) Add $g_1\sigma^2s[\Delta\rho]$ to
the energy row after the face work, where $\Delta\rho$ is that stage's x1 density change (implicit solvers: the solved
change, and the term inside the implicit operator, §10). (v) Turn H off (F already removes the $F''$ term; F + H is $O(h^2)$ again). (vi) Test: closed column, any
flow, $\Delta(E + P)$ per step at round-off with $P$ from (8.5).

## 9. F against the $\bar r$ potential form

A smaller alternative keeps the face form and the curvature flux but moves the potential: $\phi_c = -g_1\bar r$,
$\bar r = (r_{i+1/2} + r_{i-1/2})/2$, with the cp3/cp5/weno5 curvature flux applied to $r^2m$,
$A_fH_f = \tfrac{1}{12}(r_{c,i} - r_{c,i-1})(\bar r_i^2m_i - \bar r_{i-1}^2m_{i-1})$, zero at the walls. It is a
divergence plus a face form, so it conserves $E + P_{\bar r}$, $P_{\bar r} = \sum_iV_i\rho_i\phi(\bar r_i)$. Checked in
F-replica §8; spherical, closed walls, the profile of §5, $R = 5H$ unless noted.

| | $\bar r$ form | F |
|---|---|---|
| conserved functional, one step, random $F_f$ and $m$, / $\sum\lvert WV\rvert$ | $E + P_{\bar r}$: 5e-15 ($R = H, 5H$), $\lesssim$ 2e-12 ($R = 1000H$) | $E + P$: 6e-15, $\lesssim$ 2e-12 |
| interior max cell error, nz 64 / 256 | 1.1e-7 / 5.4e-10 ($h^{3.9}$) | 1.4e-7 / 5.4e-10 ($h^{4.0}$) |
| first (wall) cell, nz 64 / 256 | **4.0e-3 / 1.0e-3 ($h^{1.0}$)** | 9.0e-8 / 1.8e-10 ($h^{4.4}$) |
| last (wall) cell, nz 64 / 256 | **7.6e-5 / 1.9e-5 ($h^{1.0}$)** | 4.5e-9 / 2.0e-11 |
| column-summed work error, nz 64, $R = 5H$ / $1000H$ | **+3.39e-4 / +5.25e-4** | -4.7e-7 / -8.0e-7 |
| same, nz 256 | +2.1e-5 / +3.3e-5 ($h^2$) | -1.9e-9 / -3.4e-9 ($h^4$) |
| conserved PE against $\int\rho\phi\,dV$, nz 64 / 256 | $P_{\bar r}$: 3.5e-5 / 2.2e-6 ($h^2$) | $P$: 3.0e-8 / 1.3e-10 ($h^4$) |

($R = 1000H$ conservation is the cancellation floor of §5, the same for both forms. For reference, $\mathrm{PE}_d$ is
5.3e-5 / 3.3e-6 off at nz 64 / 256.)

*Why the $\bar r$ form's wall cells stay $O(h)$.* Its $H$ is zero at the walls, as today's is, so its wall cells are
face+H's: the first cell is 1.02e-3 for both at nz 256 (F-replica §5, §8b). Its column total equals the face form's at
$R = 1000H$ (+5.249e-4 for both, §7 settling table) and is close to it at $R = 5H$ (+3.39e-4 against +3.55e-4). The
interior error is $O(h^4)$, the cell-by-cell and column errors are not.

*Why the two forms cannot be equivalent.* A conservative booking conserves $E$ plus its own functional, so its column
work error is the time derivative of that functional's error. $P_{\bar r} - \int\rho\phi\,dV$ is a pure wall term,
$g_1\tfrac{h^2}{12}[r^2\rho]$ (top minus bottom) $+ O(h^4)$ (F-replica §8e: ratio 0.99998 at nz 64, 1.00000 at nz 256),
so the $\bar r$ form's $O(h^2)$ column error sits in its wall cells. Closing $H$ at the walls would break conservation
unless the PE changes as well, and the PE that makes the walls $O(h^4)$ is $P$ of (6): F is the $\bar r$ idea plus the
wall closure. F's $P$ is not of the form $\sum_iV_i\rho_i\phi_{c,i}$, so the §4 lemma (which bounds those) does not
apply to it.

*Scope.* The $\bar r$ form is spherical only and leaves Cartesian unchanged, so it cannot remove the Cartesian $O(h)$
wall error of face+H (§8.7). Adding its $H$ on top of F would remove the $F''$ term twice, as F + H does (§7). F is the
form implemented.

## 10. Every place that computes a PE, under `SNAP_GRAVITY_WORK_RADIAL_EXACT`

F conserves $E + P$, not $E + \mathrm{PE}_d$, so every site that computes or checks a PE has to agree with the switch.
Searched: `src/`, `python/`, `tests/`.

| site | PE used | under the switch |
|---|---|---|
| explicit face work, `hydro_forward.cpp` | $P$ | the booked work (7); the reference |
| implicit face work: the matrix's work rows and the projection/clamp work in `implicit_hydro.cpp` | $\mathrm{PE}_d$ increments | consistent: the energy row books $g_1\sigma^2\tilde s[\delta\rho_{\rm raw} - \Delta\rho_0]$ (below) and the post-solve term the rest, so the total is $g_1\sigma^2s[\Delta\rho]$ with $\Delta\rho$ the solved change `du - du0`, which includes the redistributed and projected mass; $\Delta(E + P)$ per step $\le 4\times10^{-16}$ on the implicit cases of the test |
| gravity-work fixer (`gwfix_stage` in `hydro_forward.cpp`, the implicit `epe` lambda) and its `fixgrav=` printout | $\mathrm{PE}_d$ | consistent by construction: the fixer runs only with `gravity-work: cell` (`hydro.cpp`), the switch acts only with `gravity-work: face` (`radial_exact_work()`); they never meet |
| cycle diagnostics `pe=` (`print_cycle_diagnostics`, `meshblock.cpp`) | $P$ | logs $P$ via `corrected_pe_work` with the same per-block one-sided slope stencil, so the logged `ie=` + `pe=` is the conserved $E + P$ (it logged $\mathrm{PE}_d$ before; the test's check 5 failed on that at $\sim2\times10^{-5}$ relative and passes at $\le 3\times10^{-14}$) |
| netCDF outputs | none | no PE field is written |
| E+PE$_d$ oracles in other tests (`test_horizontal_flux_covariance`, `test_gravity_work_fixer`, `test_forcing.cpp`) | $\mathrm{PE}_d$ | correct for plain face work: the explicit ones, and the values `test_flux_covariance_rows` recorded with it, run with the switch set to 0 in their ctest entries; the switch-on oracle is `tests/test_gravity_work_radial_exact.py` ($E + P$ per step, and the logged `ie=` + `pe=` against it) |
| face-work energy oracles of `test_implicit_face_work_operator` and `test_implicit_stratified_solid` | $\mathrm{PE}_d$, or $P$ with the switch on | they read the switch as `hydro.cpp` does (a gnomonic-equiangle face block keeps the plain face work) and then measure $E + P$ on the final state; ctest runs each twice, with the switch `=0` and `=1` (#296) |

**The work inside the implicit operator (#296).** Booked only after the solve, the implicit part of (7) is explicit,
and it is not small where it matters: for a grid-scale density change the slope term is not $O(h^2)$ below the face
work (by the closed forms of §8.5, a quarter of it for a face-flux wave of wavelength $3h$). A discretely balanced 11.3H rest column (`test_implicit_gravity_tall_column.py`,
implicit-scheme 9) then grew from round-off: max $w$ $1.3\times10^{-7}$ m/s at step 13 for vertical acoustic Courant
197 and step 9 for 250, Cartesian and spherical-polar alike, against $\le 4.9\times10^{-9}$ with the switch off.
Dropping only the post-solve term made every rung pass at the switch-off level; dropping only the explicit term did not.
So the energy row now carries the term. With $\delta\rho$ the solve's total-mass unknown and $\Delta\rho_0$ the explicit
density change it starts from (`du0`), the row gains $-g_1\sigma_i^2\tilde s_i[\delta\rho]/\Delta t$ and its right side
$-g_1\sigma_i^2\tilde s_i[\Delta\rho_0]/\Delta t$, so it books $g_1\sigma^2\tilde s[\delta\rho - \Delta\rho_0]$, the work of
the mass the solve moves. $\tilde s$ is $s$ of (8.4) where the matrix can hold it: the interior 3-point stencil is
tridiagonal as it is, and at each end the one-sided slope's third weight is lumped onto the neighbour (it still
annihilates a constant). After the solve the term $g_1\sigma^2(s[\Delta\rho] - \tilde s[\delta\rho_{\rm raw} -
\Delta\rho_0])$ is added, $\Delta\rho$ the solved change after redistribution and clamps, so the booked total and the
E + P identity are what they were: only how much of the work the solve sees changes. Measured on a CPU build:
every rung of the tall column passes in both geometries with the switch on, max $w$ $4.69$–$4.89\times10^{-9}$, at or
below its switch-off rung, settled to $\le 7.4\times10^{-13}$ at step 40; $\Delta(E + P)$ per step
$\le 3.9\times10^{-16}$ of $E + P$ there, on the implicit cases of `test_gravity_work_radial_exact.py`, and on its
column under partial VIC (implicit-scheme 1); switch-off states and switch-on explicit states are bitwise unchanged.
The same defect failed `test_implicit_face_work_operator` with the switch on (rest column at Courant 197: max $w$
0.37 m/s; non-finite at steps 16-22 at Courant 197 and 657, full and partial VIC) and the face-work tall columns of
`test_implicit_stratified_solid` (nz 120/140/150 rolled back after 13/11/9 of 300 steps); with the term in the
operator they run to the end (rest columns $\le 4.9\times10^{-9}$ m/s at Courant 197 and 657; the stratified columns
$\le 4.4\times10^{-6}$ m/s, the level cell work reaches on them). Their remaining switch-on
"failures" were $E + \mathrm{PE}_d$ oracles, which measure the wrong invariant here: on the same runs $E + P$ closes,
e.g. the spherical implicit energy check gives $E + \mathrm{PE}_d = -0.166$ but $E + P = 3.5\times10^{-14}$
(scale 2119), and the moving tall columns drift by $\le 4.1\times10^{-15}$ in $E + P$ against up to
$3.9\times10^{-9}$ in $E + \mathrm{PE}_d$.

## 11. The default reference at the walls, F on by default with `gravity-work: face`, and the remaining error

### 11.1 The wall closure of the default x1 reference

With `SNAP_WB_REF4` off, the default well-balanced reference repeated the wall cell's $r = \rho/p$ in the ghosts
past a clamped physical wall, which makes the face density reference first order at the three faces next to each
wall. It now continues $r$ past the wall (`wb-ref-wall.md` §4): linearly, $r_{-k} = r_0 + k(r_0 - r_1)$, where $r$
falls away from the wall, and ln-linearly, $r_{-k} = r_0(r_0/r_1)^k$, where it rises. Neither form alone is
bounded:
- linear in $r$ everywhere reaches zero, $r_{-3} = r_0(1 - 3d)$ with $d = (r_1 - r_0)/r_0$, and goes negative past
  $d = 1/3$; `test_straka_redo` (CFL 1.6, nx1 64) then fails at 31.67 s with $p < 0$ below the top wall;
- ln-linear everywhere grows without bound next to a nearly empty cell: in `test_face_floor`'s unresolved column it
  gives a dipped-face mass flux of $3.36\times10^{-7}$, against $2.83\times10^{-8}$ for the linear form.

Taking per column the branch that stays closer to $r_0$ keeps the continuation in $[r_0(r_0/r_1)^k, r_0(1+k)]$. The
observed order at faces 1-2 of both walls is 1.81-2.12 (nz 32 to 64), and every interior face and cell is bit for
bit as before. The tests that pinned values of the old wall reference were re-pinned with it.

### 11.2 Why F is on by default with `gravity-work: face`

With `gravity-work: face` on a Cartesian or spherical-polar grid and $g_1 \ne 0$, `SNAP_GRAVITY_WORK_RADIAL_EXACT`
is on unless it is set to 0/false/off/no, which remains for A/B runs. Plain face work is first order in the two
x1 wall cells (§8), and it shows:
- in the dry Cartesian onset box with `SNAP_FLUX_COVARIANCE` and `SNAP_WB_REF4` on (cell-average initial state),
  the one-step $\varepsilon_{\rm eff}\,n_z^2$ at $n_z$ 16, 32, 64 is $+0.0306$, $+0.0158$, $+0.0080$ without F and
  $-0.00092$, $-0.00020$, $-0.00004$ with F;
- on a coarse polytrope the kinetic energy runs away to $3.1\times10^{-4}$ without F, $5.4\times10^{-5}$ with F, and
  $8.5\times10^{-5}$ with `gravity-work: cell`.

A gnomonic-equiangle (cubed-sphere) grid has no form for F: there the setup warns once and the plain face work is
kept. `gravity-work: cell`, the default, is unchanged bit for bit. The explicit E+PE$_d$ oracles of §10 that test the
plain face form run with the switch set to 0.

### 11.3 The remaining error

With the wall closure and F, the convective onset (linear growth) rate keeps a second-order truncation error
that face and cell work share, so it does not separate the two forms.

> **TABLE PLACEHOLDER** -- convergence table nz 16-128, fit $c_2\,\Delta z^2 + c_3\,\Delta z^3$: to be filled by the
> lead.
