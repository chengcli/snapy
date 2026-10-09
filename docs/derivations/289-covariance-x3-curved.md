# Issue #289, item 2: the $O(\Delta x_1^2)$ covariance term in the horizontal energy flux, on curved grids and on $x_3$

**Scope.** Issue [#289](https://github.com/chengcli/snapy/issues/289) item 2: derive
the face average of the horizontal energy flux *with the face metric* for
spherical-$x_1$ and cubed-sphere grids, extend it to the $x_3$ direction, and
state the discrete form that is implemented.

**Base commit.** Everything quoted from the source below was read at
`117e449a620bd7fd50abe19239ab660a56d5d4cb` (`chengcli/snapy` `main`), confirmed
on 2026-10-09 by both

```
git ls-remote https://github.com/chengcli/snapy main
gh api repos/chengcli/snapy/commits/main --jq .sha
```

Every `file:line` below is a line number **at that commit**.

**Provenance.** Derived from the formula stated in #289 and from the snapy
source only. No other implementation or derivation of this term was read.

---

## 1. Starting definitions

### 1.1 The finite-volume update

For a conserved density $U$ with flux $\mathbf{f}$, the finite-volume balance over
cell $(k,j,i)$ is exact:

$$
\frac{\mathrm{d}}{\mathrm{d}t}\Big(V_{kji}\,\bar U_{kji}\Big)
= -\oint_{\partial\Omega_{kji}} \mathbf{f}\cdot\hat{\mathbf{n}}\,\mathrm{d}A ,
\qquad
\bar U_{kji} \equiv \frac{1}{V_{kji}}\int_{\Omega_{kji}} U\,\mathrm{d}V .
$$

snapy evaluates exactly this shape, in `CoordinateImpl::divergence`
(`src/coord/coordinate.cpp:461-505`):

```cpp
  if (flux2.defined() > 0) {                                  // :492
    dflx.slice(DIM2, sj, ej) +=
        face_area2(sj + 1, ej + 1) * flux2.slice(DIM2, sj + 1, ej + 1) -
        face_area2(sj, ej) * flux2.slice(DIM2, sj, ej);
  }
  if (flux3.defined() > 0) {                                  // :498
    dflx.slice(DIM3, sk, ek) +=
        face_area3(sk + 1, ek + 1) * flux3.slice(DIM3, sk + 1, ek + 1) -
        face_area3(sk, ek) * flux3.slice(DIM3, sk, ek);
  }
  return dflx / vol;                                          // :504
```

and the tendency is $-\mathrm{d}t$ times that divergence on the interior
(`src/hydro/hydro_forward.cpp:418`, `:427-429`). So the semi-discrete energy
balance is

$$
\frac{\mathrm{d}\bar E_{kji}}{\mathrm{d}t}
= -\frac{1}{V_{kji}}\Big[
A^{(2)}_{j+\frac12}F^{(2)}_{j+\frac12} - A^{(2)}_{j-\frac12}F^{(2)}_{j-\frac12}
+ A^{(3)}_{k+\frac12}F^{(3)}_{k+\frac12} - A^{(3)}_{k-\frac12}F^{(3)}_{k-\frac12}
+ (x_1\ \text{terms})\Big].
\tag{1.1}
$$

The identity (1.1) holds **only if** $A^{(2)}F^{(2)}$ is the true surface
integral, i.e. if

$$
\boxed{\;F^{(2)}_{j+\frac12}
= \big\langle f_E^{(2)} \big\rangle_{A}
\equiv \frac{1}{A^{(2)}_{j+\frac12}}\int_{S_{j+1/2}} f_E^{(2)}\,\mathrm{d}A \;}
\tag{1.2}
$$

— the **area-weighted average over the face** of the point flux. This is the
definition #289 starts from:

> The finite-volume flux through it is the x1 average over the face of the point
> flux gamma/(gamma-1) p u. The code instead evaluates the flux from x1-averaged
> states.

### 1.2 The flux that is actually evaluated

The energy flux normal to an $x_2$ face is $f_E^{(2)}=(E+p)\,u_n$ with
$E+p = \rho h + \tfrac12\rho|\mathbf{u}|^2$ and $h=e+p/\rho$ the specific
enthalpy. LMARS writes it as (`src/riemann/lmars_impl.h:28-31`, `:61`, `:76`)

```cpp
  hl += 0.5 * (SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ])) +
        wli[IPR] / wli[IDN];                                  // :28-29
  ...
  FLX(IPR) = ubar * wli[IDN] * hl;                            // :61
```

i.e. $F^{(2)} = \bar u_n\,\bar\rho\,\bar h$, every factor taken from the
**reconstructed face state**, where $\bar u_n$ is LMARS' acoustically corrected
normal velocity (`lmars_impl.h:42`) and the normal row is selected by
`ivx = IPR - dim` (`lmars_impl.h:20`): for `dim == 2` that is `IVY`, for
`dim == 1` (the $x_3$ direction, `DIM3 = 1`) it is `IVZ`
(`src/snap.h:38-40`). Keeping the enthalpy part and writing
$\rho h = \gamma/(\gamma-1)\,p$ for an ideal gas, the code evaluates

$$
F^{(2)}_{\rm code} = \frac{\gamma}{\gamma-1}\,\bar p\,\frac{\bar m}{\bar\rho},
\qquad m\equiv\rho u_n ,
\tag{1.3}
$$

whereas (1.2) demands $\frac{\gamma}{\gamma-1}\langle p\,u_n\rangle_A$. The
missing piece is

$$
\Delta F \;\equiv\; \frac{\gamma}{\gamma-1}\Big[\langle p u_n\rangle_A
- \bar p\,\frac{\bar m}{\bar\rho}\Big].
\tag{1.4}
$$

### 1.3 The face measure $\mathrm{d}A$, per grid

Write the general curvilinear area element of a surface $x^2=\text{const}$ as
$\mathrm{d}A_2=\sqrt{g}\,\sqrt{g^{22}}\,\mathrm{d}x^1\mathrm{d}x^3$. Rather than
re-deriving it per grid, read the *discrete* measure off snapy's own
`face_area2()`/`face_area3()`, which is what (1.1) actually uses.

**(a) Cartesian** (`src/coord/coordinate.cpp:429-435`):

```cpp
torch::Tensor CoordinateImpl::face_area2() const {            // :429
  return dx3f.outer(dx1f).unsqueeze(1).expand({-1, x2f.size(0), -1});
}
torch::Tensor CoordinateImpl::face_area3() const {            // :433
  return dx2f.outer(dx1f).unsqueeze(0).expand({x3f.size(0), -1, -1});
}
```

so $\mathrm{d}A_2 = \mathrm{d}x_1\,\mathrm{d}x_3$ and
$\mathrm{d}A_3=\mathrm{d}x_1\,\mathrm{d}x_2$: the $x_1$ weight is **uniform**,

$$
w_2(x_1)=w_3(x_1)=1 .
\tag{1.5}
$$

**(b) Spherical-polar**, $(x_1,x_2,x_3)=(r,\theta,\phi)$
(`src/coord/spherical_polar.cpp:185-198`):

```cpp
torch::Tensor SphericalPolarImpl::face_area2() const {        // :185
  auto radial = 0.5 * (x1f.slice(0, 1, options->nc1() + 1).square() -
                       x1f.slice(0, 0, options->nc1()).square());
  return radial.unsqueeze(0).unsqueeze(1) *
         x2f.sin().abs().unsqueeze(0).unsqueeze(2) *
         dx3f.unsqueeze(1).unsqueeze(2);
}
torch::Tensor SphericalPolarImpl::face_area3() const {        // :193
  auto radial = 0.5 * (x1f.slice(0, 1, options->nc1() + 1).square() -
                       x1f.slice(0, 0, options->nc1()).square());
  return (radial.unsqueeze(0).unsqueeze(1) * dx2f.unsqueeze(0).unsqueeze(2))
      .expand({x3f.size(0), -1, -1});
}
```

The `radial` factor is exactly $\tfrac12(r_+^2-r_-^2)=\int_{r_-}^{r_+} r\,\mathrm{d}r$.
Hence

$$
\mathrm{d}A_2 = r\,\sin\theta_{j+\frac12}\,\mathrm{d}r\,\mathrm{d}\phi,
\qquad
\mathrm{d}A_3 = r\,\mathrm{d}r\,\mathrm{d}\theta,
\qquad\Longrightarrow\qquad
w_2(r)=w_3(r)=r .
\tag{1.6}
$$

Note that the $r^2$ of the *$x_1$* face (`spherical_polar.cpp:178-183`) and of
the cell volume (`:200-211`) does **not** appear here: an $x_2$ or $x_3$ face
integrates $\mathrm{d}r$ against exactly **one** power of $r$, the one that comes
from the transverse arc length.

**(c) Cubed sphere** (gnomonic-equiangular), $(x_1,x_2,x_3)=(r,\alpha,\beta)$
(`src/coord/gnomonic_equiangle.cpp:197-203`):

```cpp
torch::Tensor GnomonicEquiangleImpl::face_area2() const {     // :197
  return (x1v * dx1f).unsqueeze(0).unsqueeze(1) * dx3f_ang_face2_kj;
}
torch::Tensor GnomonicEquiangleImpl::face_area3() const {     // :201
  return (x1v * dx1f).unsqueeze(0).unsqueeze(1) * dx2f_ang_face3_kj;
}
```

Here `x1v` is the arithmetic mid-radius on this grid (`:36-37`), so

$$
\texttt{x1v}\cdot\texttt{dx1f} = \tfrac12(r_-+r_+)(r_+-r_-)
= \tfrac12\big(r_+^2-r_-^2\big) = \int_{r_-}^{r_+} r\,\mathrm{d}r ,
\tag{1.7}
$$

**the same radial measure as (1.6), exactly** — not to leading order. The
companion factors `dx3f_ang_face2_kj` (`:119`) and `dx2f_ang_face3_kj` (`:105`)
are the great-circle *angles* subtended by the face's transverse edge; they are
built from $\tan x_2$, $\tan x_3$ and the panel geometry and carry the
non-orthogonality $g_{23}=\cos\vartheta$ (`:83`,
`_set_face2_metric` at `:233-245`) and $\sqrt{g}\propto$ `sine_face2_kj`
(`:84`). Crucially **they do not depend on $r$.** So, with $\psi$ denoting the
transverse coordinate,

$$
\mathrm{d}A_2 = r\,W_2(\psi)\,\mathrm{d}r\,\mathrm{d}\psi,
\qquad \partial_r W_2 = 0 ,
\tag{1.8}
$$

where $W_2$ holds every angular, $\sqrt{g}$ and non-orthogonal factor.

### 1.4 The face-average operator and its moments

For any in-face coordinate $\xi$ with $x_1$-weight $w$, define

$$
\langle a\rangle_A \equiv \frac{\int a\,w\,\mathrm{d}x_1}{\int w\,\mathrm{d}x_1},
\qquad
x_1^{c} \equiv \frac{\int x_1 w\,\mathrm{d}x_1}{\int w\,\mathrm{d}x_1}
\ \ (\textbf{area centroid}),
\qquad
\zeta \equiv x_1-x_1^{c} .
\tag{1.9}
$$

Moments: $\mu_0=1$, and **by the choice of $x_1^c$**

$$
\mu_1 \equiv \langle\zeta\rangle_A = 0,
\qquad
\sigma_1^2 \equiv \mu_2 = \langle\zeta^2\rangle_A,
\qquad
\mu_3 = \langle\zeta^3\rangle_A .
\tag{1.10}
$$

$\mu_1=0$ is the only property of $x_1^c$ used below; everything else follows.

---

## 2. Every expansion, written out

### 2.1 One factor

Assume $a\in C^4$ across the face (§4, A1). Taylor about $x_1^c$:

$$
a(\zeta) = a_c + a'_c\zeta + \tfrac12 a''_c\zeta^2
+ \tfrac16 a'''_c\zeta^3 + O(\zeta^4),
$$

so, applying $\langle\cdot\rangle_A$ term by term and using $\mu_1=0$:

$$
\langle a\rangle_A
= a_c + a'_c\underbrace{\mu_1}_{=0} + \tfrac12 a''_c\,\sigma_1^2
+ \tfrac16 a'''_c\,\mu_3 + O(\mu_4).
\tag{2.1}
$$

### 2.2 Size of $\mu_3$ and $\mu_4$ — why they drop out

For the **Cartesian** weight (1.5) the measure is symmetric about the face
mid-point, so $\mu_3=0$ **exactly**.

For the **curved** weight $w=r$ (1.6)/(1.8), put $r=\bar r+s$ with
$\bar r=\tfrac12(r_-+r_+)$, $h=r_+-r_-$, $s\in[-h/2,h/2]$. Normalisation:

$$
\int_{-h/2}^{h/2}(\bar r+s)\,\mathrm{d}s = \bar r h .
$$

First moment about the mid-point:

$$
\langle s\rangle = \frac{1}{\bar r h}\int_{-h/2}^{h/2}s(\bar r+s)\,\mathrm{d}s
= \frac{1}{\bar r h}\Big(0 + \frac{h^3}{12}\Big)
= \frac{h^2}{12\,\bar r}\;\equiv\;\delta ,
\tag{2.2}
$$

so $x_1^c = \bar r+\delta$ and $\zeta=s-\delta$. Second moment about the
mid-point:

$$
\langle s^2\rangle = \frac{1}{\bar r h}\int_{-h/2}^{h/2}s^2(\bar r+s)\,\mathrm{d}s
= \frac{1}{\bar r h}\Big(\bar r\frac{h^3}{12}+0\Big) = \frac{h^2}{12},
\qquad
\langle s^3\rangle = \frac{1}{\bar r h}\int s^3(\bar r+s)\mathrm{d}s
= \frac{h^4}{80\,\bar r}.
$$

Hence

$$
\mu_3 = \langle(s-\delta)^3\rangle
= \langle s^3\rangle - 3\delta\langle s^2\rangle + 3\delta^2\langle s\rangle-\delta^3
= \frac{h^4}{80\bar r} - \frac{h^4}{48\bar r} + 2\delta^3
= -\frac{h^4}{120\,\bar r} + O\!\Big(\frac{h^6}{\bar r^3}\Big).
\tag{2.3}
$$

So $\mu_3 = O(h^4)$ and $\mu_4 = O(h^4)$: both enter (2.1) at the **same order as
the remainder we drop**, and neither contributes to the $O(h^2)$ term we are
after. This is the one place where the asymmetry of the curved measure could
have mattered, and it does not.

### 2.3 Two factors — the covariance identity

Apply (2.1) to the product $ab$, using
$(ab)''=a''b+2a'b'+ab''$ and
$(ab)'''=a'''b+3a''b'+3a'b''+ab'''$:

$$
\langle ab\rangle_A = a_cb_c
+ \tfrac12\sigma_1^2\big(a''b+2a'b'+ab''\big)_c
+ \tfrac16\mu_3\big(a'''b+3a''b'+3a'b''+ab'''\big)_c
+ O(h^4).
\tag{2.4}
$$

And multiplying the two single-factor expansions:

$$
\langle a\rangle_A\langle b\rangle_A
= \Big(a_c+\tfrac12\sigma_1^2a''+\tfrac16\mu_3a'''\Big)
  \Big(b_c+\tfrac12\sigma_1^2b''+\tfrac16\mu_3b'''\Big)
$$
$$
= a_cb_c
+ \tfrac12\sigma_1^2\big(a''b+ab''\big)_c
+ \tfrac16\mu_3\big(a'''b+ab'''\big)_c
+ \underbrace{\tfrac14\sigma_1^4\,a''b''}_{O(h^4)}
+ O(h^4).
\tag{2.5}
$$

Subtract (2.5) from (2.4). **The $a''b$ and $ab''$ terms cancel identically**,
and so do the $a'''b$ and $ab'''$ terms; what is left of the $\mu_3$ group is
$\tfrac12\mu_3(a''b'+a'b'')=O(h^4)$ by (2.3). Therefore

$$
\boxed{\;\langle ab\rangle_A - \langle a\rangle_A\langle b\rangle_A
= \sigma_1^2\,a'\,b' + O(h^4)\;}
\tag{2.6}
$$

This single identity **is** the whole generalisation. The metric enters in
exactly two ways and no others: through the value of $\sigma_1^2$, and through
the point $x_1^c$ at which $a'$ and $b'$ are evaluated.

### 2.4 The quotient $\bar m/\bar\rho$

$u_n=m/\rho$, and the code forms $\bar m/\bar\rho = \langle m\rangle/\langle\rho\rangle$,
not $\langle u_n\rangle$. Apply (2.6) with $a=m$, $b=1/\rho$:

$$
\langle u_n\rangle_A = \big\langle m\cdot\tfrac1\rho\big\rangle_A
= \langle m\rangle_A\Big\langle\tfrac1\rho\Big\rangle_A
+ \sigma_1^2\,m'\Big(\tfrac1\rho\Big)'
= \langle m\rangle_A\Big\langle\tfrac1\rho\Big\rangle_A
- \sigma_1^2\,\frac{m'\rho'}{\rho^2}.
\tag{2.7}
$$

Now the difference between "average of the reciprocal" and "reciprocal of the
average". With $(1/\rho)''=2\rho'^2/\rho^3-\rho''/\rho^2$, (2.1) gives

$$
\Big\langle\tfrac1\rho\Big\rangle_A
= \frac1{\rho_c} + \tfrac12\sigma_1^2\Big(\frac{2\rho'^2}{\rho^3}-\frac{\rho''}{\rho^2}\Big)
+ O(h^4),
$$

while

$$
\frac1{\langle\rho\rangle_A}
= \frac{1}{\rho_c+\tfrac12\sigma_1^2\rho''+O(h^4)}
= \frac1{\rho_c}\Big(1-\tfrac12\sigma_1^2\frac{\rho''}{\rho_c}\Big)+O(h^4)
= \frac1{\rho_c}-\tfrac12\sigma_1^2\frac{\rho''}{\rho^2}+O(h^4).
$$

Subtracting, **the $\rho''$ terms cancel** and only the squared first derivative
survives:

$$
\Big\langle\tfrac1\rho\Big\rangle_A - \frac1{\langle\rho\rangle_A}
= \sigma_1^2\,\frac{\rho'^2}{\rho^3} + O(h^4).
\tag{2.8}
$$

Insert (2.8) into (2.7). Because the whole expression is already $O(\sigma_1^2)$
we may replace $\langle m\rangle_A\to m$, $\langle\rho\rangle_A\to\rho$ inside it
(the correction is $O(h^4)$):

$$
\langle u_n\rangle_A - \frac{\bar m}{\bar\rho}
= m\,\sigma_1^2\frac{\rho'^2}{\rho^3} - \sigma_1^2\frac{m'\rho'}{\rho^2}
= -\sigma_1^2\,\frac{\rho'}{\rho^2}\Big(m'-\frac{m\rho'}{\rho}\Big).
$$

Since $u_n'=(m/\rho)'=m'/\rho-m\rho'/\rho^2$, i.e. $\rho\,u_n' = m'-m\rho'/\rho$,

$$
\boxed{\;\langle u_n\rangle_A - \frac{\bar m}{\bar\rho}
= -\,\sigma_1^2\,\frac{\rho'}{\rho}\,u_n' + O(h^4)\;}
\tag{2.9}
$$

### 2.5 Assembling $\Delta F$

Apply (2.6) to $p\,u_n$, then use $\bar p=\langle p\rangle_A$ and (2.9):

$$
\langle p\,u_n\rangle_A - \bar p\,\frac{\bar m}{\bar\rho}
= \Big[\langle p\rangle_A\langle u_n\rangle_A + \sigma_1^2 p'u_n'\Big]
  - \langle p\rangle_A\frac{\bar m}{\bar\rho}
= \langle p\rangle_A\Big[\langle u_n\rangle_A-\frac{\bar m}{\bar\rho}\Big]
  + \sigma_1^2 p'u_n'
$$
$$
= -\sigma_1^2\,p\,\frac{\rho'}{\rho}\,u_n' + \sigma_1^2\,p'\,u_n'
= \sigma_1^2\,u_n'\Big(p'-\frac{p\rho'}{\rho}\Big)
= \sigma_1^2\,p\,u_n'\Big(\frac{p'}{p}-\frac{\rho'}{\rho}\Big),
$$

that is

$$
\boxed{\;
\Delta F = \frac{\gamma}{\gamma-1}\,\sigma_1^2\;p\;
\big[\ln(p/\rho)\big]'\;u_n'
\;=\;\frac{\gamma}{\gamma-1}\,\sigma_1^2\;p\;(\ln T)'\;u_n'
\;}
\tag{2.10}
$$

the last step using $p=\rho R T$ with $R$ uniform across the face. This
reproduces #289's formula

> `Delta F = gamma/(gamma-1) (dz²/12) p (ln T)_z u_z`

with $\sigma_1^2$ in place of $\mathrm{d}z^2/12$. Note what is **absent**: no
term $\propto u_n p'$ and none $\propto u_n\rho'$ survives. They cancel in two
places — the $p''$ and $\rho''$ second-derivative groups cancel between
$\langle ab\rangle$ and $\langle a\rangle\langle b\rangle$ in (2.4)–(2.5), and the
$\rho''$ group cancels again between $\langle 1/\rho\rangle$ and
$1/\langle\rho\rangle$ in (2.8). This is #289's
"the u rho_z and u p_z cross terms cancel identically", shown rather than
asserted.

### 2.6 $\sigma_1^2$: Cartesian

With $w=1$ the centroid is the mid-point and

$$
\sigma_1^2 = \frac{1}{\Delta x_1}\int_{-\Delta x_1/2}^{+\Delta x_1/2}\zeta^2\,\mathrm{d}\zeta
= \frac{1}{\Delta x_1}\cdot\frac{2}{3}\Big(\frac{\Delta x_1}{2}\Big)^3
= \frac{\Delta x_1^{\,2}}{12}.
\tag{2.11}
$$

This is where #289's $\mathrm{d}z^2/12$ comes from: it is the second central
moment of a *uniform* face measure, nothing more.

### 2.7 $\sigma_1^2$: the curved horizontal faces — exactly

With $w(r)=r$ (true for spherical-polar *and* cubed-sphere, (1.6)–(1.8)), all
three moments are elementary and **exact**:

$$
r_+^3-r_-^3 = h\big(3\bar r^2+\tfrac{h^2}{4}\big),\qquad
r_+^2-r_-^2 = 2\bar r h,\qquad
r_+^4-r_-^4 = 2\bar r h\big(2\bar r^2+\tfrac{h^2}{2}\big),
$$

(from $r_\pm^2=\bar r^2\pm\bar r h+h^2/4$ and $r_+r_-=\bar r^2-h^2/4$), hence

$$
x_1^c = r_c = \frac{\int r\cdot r\,\mathrm{d}r}{\int r\,\mathrm{d}r}
= \frac{\tfrac13(r_+^3-r_-^3)}{\tfrac12(r_+^2-r_-^2)}
= \frac{2}{3}\,\frac{r_+^3-r_-^3}{r_+^2-r_-^2}
= \frac{3\bar r^2+h^2/4}{3\bar r}
= \bar r + \frac{h^2}{12\bar r},
\tag{2.12}
$$

$$
\langle r^2\rangle_A = \frac{\tfrac14(r_+^4-r_-^4)}{\tfrac12(r_+^2-r_-^2)}
= \frac{r_+^2+r_-^2}{2} = \bar r^2+\frac{h^2}{4},
\tag{2.13}
$$

$$
\sigma_1^2 = \langle r^2\rangle_A - r_c^2
= \Big(\bar r^2+\frac{h^2}{4}\Big)
- \frac{\big(3\bar r^2+h^2/4\big)^2}{9\bar r^2}
= \Big(\bar r^2+\frac{h^2}{4}\Big)
- \Big(\bar r^2+\frac{h^2}{6}+\frac{h^4}{144\bar r^2}\Big),
$$

$$
\boxed{\;
\sigma_1^2 = \frac{h^2}{12} - \frac{h^4}{144\,\bar r^2}
= \frac{h^2}{12}\left(1-\frac{h^2}{12\,\bar r^2}\right),
\qquad h=\Delta r,\ \ \bar r=\tfrac12(r_-+r_+)
\;}
\tag{2.14}
$$

(2.14) is an identity, not a truncation, and the right-hand form is free of the
cancellation that the middle form would suffer at $h\ll\bar r$.

**How much does the metric change the coefficient? Almost nothing, and here is
why.** The relative shift is $h^2/(12\bar r^2) = (\Delta r/\bar r)^2/12$: at
$\Delta r/r=0.1$ it is $8\times10^{-4}$ of $\Delta F$, and $\Delta F$ is itself
$O(h^2)$, so the metric's contribution to the flux is $O(h^4)$ — the same order
as everything else dropped. This is the precise content of #289's
"adds O(dr²) terms": they are $O((\Delta r/r)^2)$ *relative* to the term. The
implementation nevertheless uses the exact (2.14), because it costs nothing and
makes the statement exact rather than asymptotic.

The *evaluation point* also moves, from $\bar r$ to $r_c=\bar r+h^2/(12\bar r)$,
and neither equals snapy's `x1v`: on the spherical grid `x1v` is the **volume**
centroid (`src/coord/spherical_polar.cpp:17-21`, `:61`)

$$
r_v = \frac{\int r\cdot r^2\mathrm{d}r}{\int r^2\mathrm{d}r}
= \frac34\frac{r_+^4-r_-^4}{r_+^3-r_-^3} = \bar r+\frac{h^2}{6\bar r}+O(h^4),
\tag{2.15}
$$

i.e. twice as far out as $r_c$. Both differ from $r_c$ by $O(h^2/r)$, and since
they only ever multiply a quantity that is already $O(h^2)$, using `x1v` to
locate the derivatives in (2.10) perturbs $\Delta F$ by $O(h^4)$. The
implementation uses `x1v`.

### 2.8 Why $\sqrt{g}$ and $g_{23}\neq0$ do not appear

Take the cubed-sphere measure (1.8), $\mathrm{d}A_2=r\,W_2(\psi)\,\mathrm{d}r\,\mathrm{d}\psi$
with $\partial_r W_2=0$. The two-dimensional face average is

$$
\langle a\rangle_A
= \frac{\displaystyle\int\!\!\int a(r,\psi)\,r\,W_2(\psi)\,\mathrm{d}r\,\mathrm{d}\psi}
       {\displaystyle\int\!\!\int r\,W_2(\psi)\,\mathrm{d}r\,\mathrm{d}\psi}.
\tag{2.16}
$$

Once the $\psi$-covariance is dropped (§2.9) every factor is a function of $r$
alone, and then

$$
\langle a\rangle_A
= \frac{\Big(\int W_2\,\mathrm{d}\psi\Big)\int a(r)\,r\,\mathrm{d}r}
       {\Big(\int W_2\,\mathrm{d}\psi\Big)\int r\,\mathrm{d}r}
= \frac{\int a(r)\,r\,\mathrm{d}r}{\int r\,\mathrm{d}r},
$$

so **$W_2$ cancels identically between numerator and denominator**. Everything
non-orthogonal and every $\sqrt{g}$ lives in $W_2$. Hence:

> On the cubed sphere the non-orthogonal metric $g_{23}=\cos\vartheta$ and the
> $\sqrt{g}$ factors do not change $\sigma_1^2$ at all. The only metric effect
> on the coefficient is the radial weight $w(r)=r$, which is identical to
> spherical-polar, so (2.14) serves both grids.

What the cubed sphere *does* change is the **frame**: the normal velocity in
(2.10) must be the face-local orthonormal component that the energy flux
carries. `prim2local2_`/`prim2local3_`
(`src/coord/gnomonic_equiangle.cpp:270-308`) produce it, and
`flux2global2_`/`flux2global3_` (`:325-380`) rotate **only** the `IVY`/`IVZ`
momentum rows — `IDN` and `IPR` are never touched — so the energy flux is a
scalar under that rotation and $\Delta F$ may be added to it directly.

### 2.9 The transverse in-face covariance, and why it is dropped

An $x_2$ face is spanned by $x_1$ **and** $x_3$. For a separable measure
$w(x_1)W(x_3)$ with both first moments removed,
$\langle\zeta_1\zeta_3\rangle_A=\langle\zeta_1\rangle\langle\zeta_3\rangle=0$, so the
two directions do not mix and the corrections simply add:

$$
\Delta F = \frac{\gamma}{\gamma-1}\Big[
\sigma_1^2\,p\,(\ln T)_{,1}\,u_{n,1}
+ \sigma_\perp^2\,p\,(\ln T)_{,\perp}\,u_{n,\perp}\Big].
\tag{2.17}
$$

Only the $x_1$ piece is kept. The reason is the $\epsilon$ scaling: the
background is stratified in $x_1$ only, so $(\ln T_0)_{,1}=O(1)$ — set by the
lapse rate, **not** by $\epsilon=\nabla-\nabla_{\rm ad}$ — while
$(\ln T)_{,\perp}$ and $u_{n,\perp}$ are both perturbation quantities. The
$x_1$ piece is therefore **linear** in the perturbation amplitude and the
transverse piece is **quadratic**; for the linear onset problem #289 is about,
the transverse piece does not contribute at all. Dropping it is a deliberate
truncation (§4, A5), not an identity.

### 2.10 Why the term behaves like a spurious stratification

With $u_{n,1}$ carrying the mode's vertical structure and $(\ln T_0)_{,1}$ the
background lapse rate, the divergence of (2.10) acts on the linear problem
exactly like an added stable stratification. #289 reports
$\epsilon_{\rm spur} = -(\beta m^2/12)\,\mathrm{d}z^2 \approx -0.24/n_z^2$ for the
first mode, with the measured one-step tendency at $\epsilon=0$, $n_z=64$ giving
$\epsilon_{\rm spur}n_z^2=-0.240$ against $-0.241$ from the formula, and hence a
growth-rate error that collapses on $\epsilon n_z^2$. On a curved grid the same
argument goes through with $\mathrm{d}z^2/12\to\sigma_1^2$ from (2.14), i.e. with
a correction of relative size $(\Delta r/\bar r)^2/12$ to $\epsilon_{\rm spur}$.

---

## 3. The $x_3$ direction

Nothing in §2 used which horizontal direction the face normal points along; it
used (i) the face measure's $x_1$ weight and (ii) the fact that $x_1$ is the
stratified direction. Both hold for $x_3$ faces too. Running the same derivation
for a face of constant $x_3$, spanned by $x_1$ and $x_2$:

| face normal | in-face coordinates | covariance kept along | $x_1$ weight $w$ | $\sigma_1^2$ |
|---|---|---|---|---|
| $x_1$ | $x_2,x_3$ | none — both perturbation-quadratic | — | — |
| $x_2$ | $x_1,x_3$ | $x_1$ | Cart. $1$; sph./CS $r$ | (2.11) / (2.14) |
| $x_3$ | $x_1,x_2$ | $x_1$ | Cart. $1$; sph./CS $r$ | (2.11) / (2.14) |

* **Cartesian $x_3$ face**: $\mathrm{d}A_3=\mathrm{d}x_1\mathrm{d}x_2$
  (`coordinate.cpp:433-435`), uniform $x_1$ weight,
  $\sigma_1^2=\Delta x_1^2/12$ by (2.11).
* **Spherical-polar $x_3$ face**: $\mathrm{d}A_3=r\,\mathrm{d}r\,\mathrm{d}\theta$
  (`spherical_polar.cpp:193-198`), $w=r$, $\sigma_1^2$ by (2.14) — *the same
  coefficient as the $x_2$ face*.
* **Cubed-sphere $x_3$ face**: `face_area3 = (x1v*dx1f) * dx2f_ang_face3_kj`
  (`gnomonic_equiangle.cpp:201-203`), again $\int r\,\mathrm{d}r$ times an
  $r$-independent angular factor by (1.7), $w=r$, $\sigma_1^2$ by (2.14).

So the $x_3$ term is (2.10) verbatim, with $u_n$ the $x_3$-face normal velocity
(`IVZ` after `prim2local3_`; `lmars_impl.h:20` with `dim == 1`,
`lmars.cpp:56-57`) and the **same** $\sigma_1^2$. The $x_1$ flux receives no
correction from this mechanism: its in-face coordinates are $x_2$ and $x_3$ and
both of those covariances are perturbation-quadratic.

### 3.1 Conservation: does it telescope?

Yes, **by construction rather than by cancellation**. $\Delta F$ is added to the
face flux $F^{(2)}$/$F^{(3)}$ *before* the divergence (1.1), so the sums
telescope exactly as the flux they correct does. Summing $V\,\mathrm{d}\bar E/\mathrm{d}t$
over an $x_2$ column leaves only the two end faces, therefore

* periodic $x_2$/$x_3$: the two ends are the same face, the contribution is zero
  to round-off;
* impermeable wall: $u_n\equiv0$ there, so $\Delta F\equiv0$ at that face and
  there is nothing to cancel;
* mass and momentum are untouched — $\Delta F$ enters the `IPR` row only.

The one requirement is that the two cells sharing a face compute the **same**
$\Delta F$. It is built from the face states and from $x_1$ differences of them,
which are single-valued at a shared $x_2$/$x_3$ face because those ghosts are
exchanged before reconstruction; and the switch is read once per process (§5), so
two ranks cannot disagree about whether a shared face carries the term.

### 3.2 The discrete balanced state

$\Delta F$ carries $u_n'$ as a factor, evaluated as a **difference of face-state
normal velocities**. In a state at rest every one of those is exactly zero, so
the difference is exactly zero in floating point and $\Delta F\equiv0$ bitwise.
The tendency of a discretely balanced hydrostatic state is therefore unchanged
bit for bit, at any stratification, and the term can neither repair nor spoil
well-balancing. A second exact zero comes from the other factor: an isothermal
state has $[\ln(p/\rho)]'=0$, so $\Delta F\equiv0$ there even in motion. Both are
what #289 means by "It is zero at rest and zero for an isothermal state", and
both are checked (§6).

---

## 4. Assumptions, and the order retained

* **A1 — Smoothness.** $p,\rho,m\in C^4$ across the face, and $\rho>0$. Used for
  (2.1) and for bounding $\mu_3,\mu_4$. Across a shock the expansion is void;
  the term is then a bounded $O(h^2)$ perturbation of a flux that the limiter
  already dominates, but no claim is made for it there.
* **A2 — Ideal gas for the $\ln T$ step.** The derivation produces
  $[\ln(p/\rho)]'$; writing it as $(\ln T)'$ needs $p=\rho RT$ with $R$ uniform
  across the face. The implementation evaluates $\ln(p/\rho)$, so it does not
  rely on this; only the *name* $\ln T$ does.
* **A3 — Enthalpy flux only.** $f_E^{(2)}=(E+p)u_n$ also has the kinetic part
  $\tfrac12\rho|\mathbf{u}|^2u_n$, whose covariance is smaller by $O(M^2)$.
  Dropped. #289's regime is $M\sim10^{-4}\!-\!10^{-3}$, so this is a relative
  $10^{-8}\!-\!10^{-6}$.
* **A4 — Reconstructed face state read as the face average.** The reconstructed
  $w_{L},w_{R}$ are taken to represent $\langle\cdot\rangle_A$ over that face
  with the face's **area** measure. In Cartesian the transverse weight is uniform
  and the reading is unambiguous. On a curved grid it is a choice at exactly the
  order in question, and §7 (H3) states what is left over under the other
  reading.
* **A5 — Transverse covariance dropped.** §2.9. Exact only to leading order in
  perturbation amplitude.
* **A6 — Uniform vs non-uniform $x_1$ spacing.** $\sigma_1^2$ is computed
  **per cell** from that cell's own $(r_-,r_+)$, so non-uniform $x_1$ is handled
  exactly by (2.11)/(2.14); no assumption of uniform $\Delta x_1$ enters the
  coefficient. The *derivatives* in (2.10) are taken by a centred difference on
  `x1v`, which is $O(h^2)$-accurate on a uniform grid and $O(h)$ on a stretched
  one. Since $\Delta F$ is already $O(h^2)$, that leaves the flux error at
  $O(h^3)$ in the worst case — still a strict improvement on the $O(h^2)$ error
  being removed.
* **Order retained.** Exactly the $O(h^2)$ covariance. Dropped, all $O(h^4)$ or
  smaller: the $\tfrac14\sigma_1^4a''b''$ term in (2.5); the $\mu_3$ and $\mu_4$
  groups via (2.3); the $O(h^4)$ residue of (2.14); the $O(h^2/r)$ offset between
  $r_c$ and `x1v` multiplied into an $O(h^2)$ term; and the replacement of
  $\langle m\rangle,\langle\rho\rangle$ by point values inside (2.9).

---

## 5. The final discrete form, exactly as coded

Implemented in `HydroImpl::_flux_covariance` (`src/hydro/hydro_forward.cpp`),
called once per horizontal direction from `HydroImpl::forward`. In the code's own
names, for an $x_2$ face at index $(k,j,i)$:

```cpp
auto wbar = 0.5 * (wl + wr);                    // the face state
auto p    = wbar[IPR];
auto lnt  = (p / wbar[IDN]).log();              // ln(p/rho) = ln(R T)
auto enth = peos->compute("W->I", {wbar}) + p;  // I + p = rho*h = gamma/(gamma-1) p
if (dim == 2) pcoord->prim2local2_(wbar); else pcoord->prim2local3_(wbar);
auto un   = wbar[dim == 2 ? IVY : IVZ];         // face-normal velocity, local frame

auto x1v  = pcoord->x1v.to(p.device(), p.scalar_type());
auto s2   = pcoord->face_moment2_x1().to(p.device(), p.scalar_type())
                   .unsqueeze(0).unsqueeze(1);  // sigma_1^2, one value per x1 cell
auto dx1  = x1v.narrow(0, 2, n1 - 2) - x1v.narrow(0, 0, n1 - 2);
auto dlnt = (lnt.narrow(-1, 2, n1 - 2) - lnt.narrow(-1, 0, n1 - 2)) / dx1;
auto dun  = (un.narrow(-1, 2, n1 - 2)  - un.narrow(-1, 0, n1 - 2))  / dx1;

auto dflx = torch::zeros_like(p);
dflx.narrow(-1, 1, n1 - 2) = enth.narrow(-1, 1, n1 - 2)
                           * s2.narrow(-1, 1, n1 - 2) * dlnt * dun;
```

and at the call sites, `_flux2[IPR] += dcov` / `_flux3[IPR] += dcov` immediately
after `priemann->forward(...)`. In symbols, for interior $x_1$ index $i$:

$$
\Delta F_{kji} = \big(I+p\big)_{kji}\;\sigma^2_{1,i}\;
\frac{\ln(p/\rho)_{k j, i+1}-\ln(p/\rho)_{k j, i-1}}
     {\texttt{x1v}_{i+1}-\texttt{x1v}_{i-1}}\;
\frac{(u_n)_{k j, i+1}-(u_n)_{k j, i-1}}
     {\texttt{x1v}_{i+1}-\texttt{x1v}_{i-1}} .
\tag{5.1}
$$

Correspondence with (2.10): $(I+p)=\rho h=\frac{\gamma}{\gamma-1}p$ exactly for an
ideal gas, so no $\gamma$ has to be introduced and the form carries over to any
EOS that supplies an internal energy (see §7, H4);
$\sigma^2_{1,i}=$ `face_moment2_x1()`, which is `dx1f.square()/12.` in
`CoordinateImpl` (eq. 2.11) and `radial_face_moment2_(x1f, nc1)` in
`SphericalPolarImpl` and `GnomonicEquiangleImpl` (eq. 2.14).

Only Cartesian is covered by the base value. `CylindricalImpl::reset()` is empty
at `117e449` and `src/coord/cylindrical.cpp_` is not compiled, so cylindrical is
a stub there; it inherits the uniform coefficient and would need its own
override, derived by §2.7 from whichever power of $r$ its own `face_area2` and
`face_area3` turn out to carry, before the term should be trusted on it.

**Why `wbar` is projected on a private copy.** `LmarsSolverImpl::forward` calls
`prim2local2_(wl)`/`(wr)` in place (`src/riemann/lmars.cpp:60-61`), so reading
the solver's inputs *after* the call would also give the local-frame normal
velocity — but `roe` and `plume_roe` do not project at all, so that side effect
is not a contract. `_flux_covariance` therefore runs **before** the Riemann call
on `0.5*(wl+wr)` and projects that copy itself; the result is added to the flux
after the call. `_set_face2_metric()` is idempotent, so the extra call does not
disturb the solver.

### 5.1 The environment switch

```cpp
bool HydroImpl::flux_covariance() {              // src/hydro/hydro.cpp
  static const bool on = [] {
    auto v = get_env("SNAP_FLUX_COVARIANCE", "0");
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return !(v.empty() || v == "0" || v == "false" || v == "off" || v == "no");
  }();
  return on;
}
```

* **Name:** `SNAP_FLUX_COVARIANCE`. **Default: OFF.** Unset, or any of
  `0`, `false`, `off`, `no`, or empty (case-insensitive) leaves the term out and
  the code path bit-identical to `117e449`.
* Read **once** into a function-local `static`, for the reason in §3.1: every
  block of every rank must make the same choice or a shared face stops being
  single-valued. `get_env` is the tree's existing helper
  (`src/layout/layout.hpp:38-41`).

### 5.2 Stencil and ghost-cell requirements

* **Stencil:** $\{i-1,\,i,\,i+1\}$ along $x_1$, on the face states and on `x1v`.
  Three points, one direction, no transverse stencil.
* **Ghosts:** needs $n_{\rm ghost}\ge1$ on $x_1$. Interior cells run
  $i\in[\texttt{il},\texttt{iu}]=[n_{\rm ghost},\,n_{\rm ghost}+n_{x_1}-1]$, so
  $i\pm1$ is always in range. `dflx` is left zero at $i=0$ and $i=n_{c_1}-1$,
  which are $x_1$ ghosts and are never read by the interior update
  (`hydro_forward.cpp:427-429`).
* **Faces covered:** $j\in[\texttt{jl},\texttt{ju}{+}1]$ and
  $k\in[\texttt{kl},\texttt{ku}{+}1]$ — exactly the faces `divergence` consumes.
  Outside that range `_apply_inplace` replicates the L/R states
  (`src/recon/reconstruct.cpp:59-64`), and those faces are not used.
* **Guards:** returns undefined (no term) when $n_{c_1}<3$ — an unresolved $x_1$
  axis has no vertical gradient to correct — and for the `shallow-water` EOS,
  which carries no internal-energy row.
* **Device:** pure device-agnostic `torch` ops; `x1v` and `face_moment2_x1()` are
  explicitly cast to the field's device and dtype, so the CPU and CUDA paths run
  the same arithmetic on the same values.
* **Interaction with the other stages:** the positivity limiter rescales only the
  species rows and carries their enthalpy, so it never rescales $\Delta F$; and
  the vertical implicit solve does not form the $x_2$/$x_3$ fluxes, so the term
  applies unchanged under any `implicit-scheme` (#289 item 4).

---

## 6. What each test checks, and what a wrong term would look like

**(T1) Switch OFF reproduces the base commit, on a short run.**
Checks that the whole change is inert by default. The diff is purely additive and
gated, so the expectation is a **bitwise** identical state. A non-bitwise result
would localise the leak: either the gate itself, or the two new calls the term
makes that touch shared state — `prim2local2_`, which `set_`s the
`g22`/`g23`/`g33` metric buffers through `_set_face2_metric`
(`gnomonic_equiangle.cpp:233-245`), and `peos->compute("W->I", ...)`. If OFF were
not bitwise, this term could not be reviewed as opt-in at all.

**(T2) $\epsilon=0$ tendency on a balanced spherical column, OFF vs ON, at two
resolutions.** The background is *exactly adiabatic*
($\epsilon=\nabla-\nabla_{\rm ad}=0$), so $(\ln T)_{,1}\neq0$ and the term is
"armed", and the state is at rest and discretely hydrostatic. By §3.2 the
prediction is $\Delta F\equiv0$ **bitwise**, hence an identical
$\max|\mathrm{d}E/\mathrm{d}t|$ and identical momentum and energy residuals. This
is the sharpest cheap falsification available:

* if ON were *different* from OFF here, the velocity factor is not the face-normal
  difference — e.g. a $p'$-only term, or the metric moment added as a source
  rather than as a covariance — i.e. the term is not (2.10);
* if ON were *worse* than OFF, the term is breaking the discrete hydrostatic
  balance, which §3.2 says it cannot;
* the momentum and mass residuals must be *identical*, not merely small, because
  $\Delta F$ enters the `IPR` row only (§3.1).

**(T3) Cell-by-cell comparison against (5.1) on a predictable state.** T2 is a
test of a zero; this one tests the *value*. The state is built constant in $x_2$
and linear in the $x_3$ index, so the reconstruction returns the analytic face
value exactly (`interp_cp3` reproduces a linear profile exactly,
`src/recon/interp_simple.hpp:73-76`), and (5.1) can be evaluated independently in
Python — including $\sigma_1^2$ recomputed from `x1f` by (2.14) — and compared
with the measured $(\mathrm{d}u_{\rm on}-\mathrm{d}u_{\rm off})/\mathrm{d}t$. On
the spherical wedge $A^{(2)}$ varies with $\theta$, so an $x_2$-independent
$\Delta F$ still has a non-zero divergence, which is what makes the metric
observable. Failure modes it separates: a wrong **coefficient** (using
$\Delta r^2/12$ instead of (2.14), or $/24$, or the volume weight $r^2$) shows as
a near-constant relative offset; a wrong **stencil** shows as an error localised
at the $x_1$ ends; a wrong **frame** on the cubed sphere shows as an error of
relative size $g_{23}$.

**(T4) Resolution scaling $n_z: 32\to64$.** $\sigma_1^2\propto\Delta r^2$, so the
$\Delta F$-induced tendency must fall by a factor $4$. A factor $2$ or $8$ means
the order is wrong, independently of any coefficient.

**(T5) CUDA point.** The term is device-agnostic `torch` arithmetic on tensors
already resident on the device, and no `.cu` kernel is involved, so CPU and CUDA
must agree to float64 round-off on the same deck. A discrepancy beyond that
would mean a host-only buffer (`x1v`, `face_moment2_x1()`) was silently
host-resident or a dtype promotion differed between the paths — which is exactly
what the explicit `.to(device, dtype)` casts in §5 exist to prevent.

**(T6) Formula-level quadrature check** (independent of snapy):
evaluate $\langle pu\rangle_A-\bar p\,\bar m/\bar\rho$ by high-order quadrature
with weight $w=1$ and $w=r$ on generic smooth profiles and compare with (2.10).
The residual must be $O(h^4)$ absolute, i.e. $O(h^2)$ *relative* to a term that
is itself $O(h^2)$, falling by $4\times$ per halving. If it fell by $2\times$,
(2.10) would be missing part of the $O(h^2)$ covariance; if the closed form
(2.14) disagreed with the quadrature moment, $\sigma_1^2$ would be wrong.

---

## 7. Derived, versus hypothesis

**Derived here, with the algebra above:** the metric-weighted covariance identity
(2.6); the quotient correction (2.9); $\Delta F$ (2.10); the Cartesian
coefficient (2.11); the exact curved coefficient (2.14) and its Cartesian limit;
the cancellation of $\sqrt{g}$ and $g_{23}$ (2.8)→(2.16); the $x_3$ extension
(§3); exact telescoping (§3.1); bitwise preservation of a balanced state (§3.2).

**Hypotheses, labelled as such:**

* **H1.** That dropping the transverse in-face covariance (§2.9) stays negligible
  once the flow is turbulent rather than a linear mode. Untested here; that is
  #289 item 5's question.
* **H2.** That $\Delta F$ is the *whole* $O(\mathrm{d}z^2)$ error of the
  horizontal energy flux. #289's own open item 5 asks this: after the Cartesian
  fix, `cell` keeps about $-1\%$ where `face` goes to about $0$.
* **H3. A separate $O(\Delta r^2)$ curvilinear term exists and is not this one.**
  If the stored cell value is read as a *volume*-weighted average (weight
  $r^2\mathrm{d}r$) rather than as the face-area average of A4 (weight
  $r\,\mathrm{d}r$), the two differ at the same order by a **linear** term: their
  centroids are $\bar r+h^2/(6\bar r)$ (2.15) and $\bar r+h^2/(12\bar r)$ (2.12),
  so the mismatch is $\simeq\big(h^2/(12\bar r)\big)\,\partial_r a$ for every
  advected quantity. That is the standard curvilinear-reconstruction error: it
  affects the mass and momentum fluxes too, it has none of the
  $(\ln T)_{,1}u_{n,1}$ spurious-stratification structure, and it is **out of
  scope for #289**. It deserves its own issue. Nothing here fixes it and nothing
  here makes it worse.
* **H4.** The $(I+p)$ form of §5 is exact for an ideal gas. For a genuinely
  non-ideal EOS the exact covariance would need that EOS's own expansion, and
  $\ln(p/\rho)$ would no longer be $\ln T$ up to a constant.
