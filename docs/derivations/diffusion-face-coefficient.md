# The x1 face coefficient of the diffusive fluxes with an x1 profile

**Scope.** `forcing/diffusion` with `nu_scale_x1` or `kappa_scale_x1` set. The
kinematic coefficients are multiplied by a dimensionless x1 profile $s(x_1)$, so
the coefficient that multiplies the gradient in the flux is a product of two
cell fields,

$$
c = s\,q, \qquad q = \rho \ \text{(viscosity, } \mu = \nu s \rho\text{)}, \qquad
q = \rho c_v \ \text{(conduction, } k = \kappa s \rho c_v\text{)} .
$$

The flux through the x1 face $i-\tfrac12$ between cells $a = i-1$ and $b = i$
is $F = -c_{i-1/2}\,(T_b - T_a)/\Delta x$ (and the same with $v$ in place of
$T$ for the shear stress); the tendency is
$-(F_{i+1/2} - F_{i-1/2})/\Delta x$ on a uniform Cartesian mesh.

**Base.** `src/forcing/diffusion.cpp:173-193` at `aea71ed` (`chengcli/snapy`
`main`), `face_scaled_coefficient`. The wall faces of a reflecting x1 boundary
already extrapolate the product $s q$ linearly from the two nearest active
cells; only the interior faces are at issue. On x2 and x3 faces both cells share
one x1 position, so $s_a = s_b$ and the two forms below coincide.

## 1. Product of means against mean of products

The base takes the product of the two face averages,

$$
c^{\rm PM}_{i-1/2} = \frac{s_a + s_b}{2}\,\frac{q_a + q_b}{2}
= \frac{s_a q_a + s_b q_b}{2} - \frac{(s_b - s_a)(q_b - q_a)}{4} ,
$$

so it differs from the mean of the products,

$$
c^{\rm MP}_{i-1/2} = \frac{s_a q_a + s_b q_b}{2} ,
$$

by the covariance term $-\tfrac14\,\Delta s\,\Delta q$.

**A constant dynamic coefficient.** Let $s\,q = C$ be uniform (a stratified
column with $s = C/q$: the convection decks whose dynamic viscosity and
conductivity are constant, $\nu = \nu_t/\rho_0(z)$, put $s = 1/\rho_0$). Then
$c^{\rm MP} = C$ exactly on every face, while

$$
c^{\rm PM}_{i-1/2} = \frac{C}{4}\left(\frac1{q_a} + \frac1{q_b}\right)(q_a + q_b)
= C\left[1 + \frac{(q_b - q_a)^2}{4\,q_a q_b}\right] .
$$

With $\Delta q \simeq q'\Delta x$ the excess is
$e_{i-1/2} = \tfrac14 (\Delta x\, q'/q)^2 + O(\Delta x^3)$: second order, always
positive, largest where the relative gradient $q'/q$ is largest. For a linear $T$
(conductive equilibrium, $F_0 = -C\,T'$ uniform) the tendency of cell $i$ is

$$
\dot E_i = -\frac{F_0}{\Delta x}\,(e_{i+1/2} - e_{i-1/2})
\simeq -F_0\, e'(x_i) = O(\Delta x^2)
$$

inside. In the cell next to a wall one face is the extrapolated wall face, whose
coefficient is exact ($e = 0$), so there

$$
\dot E_{\rm wall} = \pm\frac{F_0}{\Delta x}\, e_{\rm first\ interior\ face}
= O(\Delta x) .
$$

For a density that falls upward, $q'/q$ is largest at the top, so the top cell
takes the largest spurious heating (with $F_0$ upward, the top wall face carries
less flux than the face below it); the column builds a thin stable layer there.

**A smooth variable coefficient.** $c^{\rm MP}$ is the trapezoidal average of
$c(x)$ over the face, so $c^{\rm MP} = c(x_{i-1/2}) + O(\Delta x^2)$; the
centred difference of $T$ is second order too, and the face flux converges at
second order. $c^{\rm PM}$ differs from it by $\tfrac14 s' q' \Delta x^2$, also
second order: on a smooth problem with no special balance both forms converge
at the same rate, and the difference shows only where a balance should be exact.

## 2. The fix

On x1 faces `face_scaled_coefficient(value, scale, ...)` now returns
`face_coefficient(value * scale, ...)`: the same two-cell average and wall
extrapolation that the unscaled coefficient already uses, applied to the
product. On x2 and x3 faces the two forms agree only up to round-off
($\tfrac12(a s + b s)$ against $\tfrac12(a + b)\,s$), so those faces keep the
base expression and stay bit for bit. A profile of ones gives the coefficient of no profile bit for bit
($q \times 1 = q$). Without a profile the code path is not entered, so every
case that sets neither `nu_scale_x1` nor `kappa_scale_x1` is unchanged.

## 3. Checks

- `docs/derivations/diffusion_face_coefficient.py`: a numpy transcription of
  the x1 operator (interior average, wall extrapolation, centred gradient).
  Constant $C$ on $\rho = (1.25 - x)^{3/2}$, $x \in [0, 1]$, $T = 300 + 50x$:
  mean of products gives $|\dot E| \le 10^{-12}$; product of means gives
  $|\dot E|_{\max}$ = 18.3, 11.2, 6.24, 3.31 at nx = 16, 32, 64, 128 (order
  0.7, 0.84, 0.92, the wall cell) and 5.6, 2.1, 0.67, 0.19 inside (order 1.40,
  1.66, 1.81). Smooth $s = 1 + \tfrac12\cos 3x$, $T = 300 + 50\sin 2x$: face
  flux error order 1.91, 1.96, 1.98 for both forms.
- `tests/test_diffusion_x1_scale.cpp`,
  `constant_dynamic_coefficient_column_has_no_tendency`: the same column through
  `Diffusion::forward`, conduction and shear, every interior cell
  $|\dot u| < 10^{-12}\,|F|/\Delta x$ at nx1 = 16 and 64. Fails on the base.
- `smooth_profile_flux_converges_at_second_order`: the face fluxes recovered
  from the tendency by summing up from the lower wall, against the exact flux,
  order $> 1.85$ from nx1 = 32 to 64 to 128, conduction and shear. Passes on
  both forms, as §1 predicts.
