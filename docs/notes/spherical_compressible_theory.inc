Independent compressible spherical-shell reference
--------------------------------------------------

This reference treats the fully compressible, nonrotating ideal-gas equations
in a spherical shell. Gravity is prescribed, constant in magnitude, and
directed inward. Dynamic viscosity :math:`\mu` and Fourier conductivity
:math:`K` are constant; bulk viscosity is zero under the Stokes hypothesis.
Both boundaries are impermeable, stress free, and held at fixed temperature.
There is no perturbation of the gravitational field. The radial calculation
uses NumPy/SciPy independently of snapy's spatial and temporal operators.

The thermodynamic linearization has the same physical content as the planar
problem in `Alboussiere and Ricard (2017), JFM 817, 264--305
<https://doi.org/10.1017/jfm.2017.108>`_, sections 5 and 8.1; an accessible
`author manuscript <https://arxiv.org/pdf/1607.02193>`_ gives the equations.
The spherical equations and numerical values below are derived and computed
here, rather than transcribed from a published spherical-shell table.

Shell geometry and normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\eta=r_i/r_o` be the **radius ratio**. Increasing :math:`\eta`
towards one makes the shell thinner relative to its radius. Use

.. math::

   d=r_o-r_i=T_m=\rho_m=R=1,\qquad
   r_i=\frac{\eta}{1-\eta},\quad
   r_o=\frac{1}{1-\eta},\quad
   r_m=\frac{r_i+r_o}{2},\quad z=r-r_m\in[-1/2,1/2].

Here :math:`T_m` and :math:`\rho_m` are evaluated at the radial midpoint;
neither is a volume average. Define

.. math::

   a=\frac{T_i-T_o}{T_m}=1,\qquad
   {\cal D}=\frac{gd}{c_pT_m}=\frac12,\qquad
   \gamma=\frac53,\quad c_p=\frac{\gamma}{\gamma-1}=\frac52,
   \quad g=c_p{\cal D}=\frac54.

The conductive solution obeys :math:`r^{-2}(r^2 T_b')'=0`:

.. math::

   B=a r_i r_o,\qquad A=1-\frac{B}{r_m},\qquad
   T_b=A+\frac{B}{r}
      =1-\frac{Bz}{r r_m}.

This fixes :math:`T_b(r_m)=1` and :math:`T_i-T_o=a` simultaneously. In
particular,

.. math::

   T_i=1+\frac{a}{1+\eta},\qquad
   T_o=1-\frac{a\eta}{1+\eta},\qquad
   \frac{T_i+T_o}{2}=1+\frac{a(1-\eta)}{2(1+\eta)}.

Thus using wall temperatures :math:`1\pm a/2` would define a different
finite-thickness background. Hydrostatic balance and the ideal-gas law give

.. math::

   p_b'=-g\rho_b,\qquad \rho_b=\frac{p_b}{T_b},\qquad
   p_b=\exp(-gJ),\qquad
   J(r)=\int_{r_m}^r\frac{ds}{T_b(s)}.

For :math:`A\ne0`, the exact integral is

.. math::

   J=\frac{z}{A}-\frac{B}{A^2}
       \log\left(1+\frac{Az}{r_m}\right).

At :math:`A=0`, use :math:`J=z+z^2/(2r_m)`. The implementation evaluates
an equivalent expression using ``log1p`` and a short series near zero to
avoid cancellation. Since :math:`J(r_m)=0`, both :math:`p_b(r_m)` and
:math:`\rho_b(r_m)` equal one. Useful derivatives are

.. math::

   T_b'=-\frac{B}{r^2},\qquad
   \rho_b'=\frac{\rho_b}{T_b}(-g-T_b'),\qquad
   H(r)=-T_b'-{\cal D}=\frac{B}{r^2}-{\cal D}.

The Rayleigh number uses the integrated superadiabatic temperature drop,
:math:`\int_{r_i}^{r_o}H\,dr=a-{\cal D}`, and midpoint diffusivities:

.. math::

   Ra_{\rm sa}=\frac{g(a-{\cal D})}{\nu_m\chi_m},\qquad
   \nu_m=\frac{\mu}{\rho_m}=\mu,\qquad
   \chi_m=\frac{K}{\rho_m c_p}=\frac{K}{c_p},\qquad Pr=\frac{\nu_m}{\chi_m}.

For arbitrary positive :math:`Pr`, the reference sets

.. math::

   \chi_m=\sqrt{\frac{g(a-{\cal D})}{Pr\,Ra_{\rm sa}}},\qquad
   \mu=Pr\,\chi_m,\qquad K=c_p\chi_m.

All tabulated cases use :math:`Pr=1`. Local diffusivities vary with density
even though :math:`\mu,K` are constant. Also, :math:`K/(\rho c_v)` is not
the thermal diffusivity used in this Rayleigh number. The local midpoint
driving :math:`H(r_m)=a[1-1/(4r_m^2)]-{\cal D}` differs from
:math:`a-{\cal D}` at finite curvature.

Vector harmonics and viscous operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose :math:`\Delta_\Omega Y_{\ell m}=-L Y_{\ell m}` with
:math:`L=\ell(\ell+1)`, :math:`s=\sqrt L`, and expand a poloidal velocity as

.. math::

   \boldsymbol u=w(r)Y_{\ell m}\boldsymbol e_r
      +U(r)\frac{\nabla_\Omega Y_{\ell m}}{s}.

The operator :math:`\nabla_\Omega` differentiates on the **unit sphere**.
The coefficient multiplying its unnormalized gradient is :math:`V=U/s`;
there is no additional radial factor in the velocity definition. This is
the radial/spheroidal decomposition used in the `SHTns documentation
<https://nschaeff.bitbucket.io/shtns/vsh.html>`_, with its spheroidal
coefficient equal to :math:`U/s`. All scalar perturbations multiply the same
:math:`Y_{\ell m}`. Spherical symmetry makes the eigenvalues independent
of the azimuthal order :math:`m`.

With :math:`\partial=d/dr`, define the scalar radial Laplacian and the
velocity divergence amplitude by

.. math::

   S=\partial^2+\frac{2}{r}\partial-\frac{L}{r^2},\qquad
   \delta=\partial w+\frac{2w}{r}-\frac{sU}{r}.

For constant :math:`\mu`,
:math:`\nabla\!\cdot\boldsymbol\tau/\mu=
\nabla^2\boldsymbol u+\nabla(\nabla\!\cdot\boldsymbol u)/3`.
Its horizontal and radial coefficients are respectively

.. math::

   F_h&=SU+\frac{2s}{r^2}w+\frac{s}{3r}\delta,\\
   F_r&=Sw-\frac{2}{r^2}w+\frac{2s}{r^2}U+\frac13\partial\delta.

The vector curvature terms and the gradient-of-divergence contribution are
both needed. Treating Cartesian velocity components as independent scalar
Laplacians in this radial representation would omit these couplings.

Fully compressible growth and neutral pencils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write physical temperature and pressure perturbation amplitudes as
:math:`\theta,q`, with time dependence :math:`\exp(\sigma t)`. The ideal-gas
density perturbation is :math:`\rho'=(q-\rho_b\theta)/T_b`. The four equations
are

.. math::

   \sigma\left(\frac{q}{T_b}-\frac{\rho_b\theta}{T_b}\right)
      &=-\rho_b\delta-\rho_b'w,\\
   \sigma\rho_b U&=-\frac{s}{r}q+\mu F_h,\\
   \sigma\rho_b w&=-\partial q-\frac{g}{T_b}q
                 +\frac{g\rho_b}{T_b}\theta+\mu F_r,\\
   \sigma(\rho_b c_p\theta-q)&=\rho_b c_p H(r)w+K S\theta.

The last equation follows from the ideal-gas entropy equation; the
background entropy gradient supplies :math:`H(r)`. Density and pressure
time derivatives are retained, so acoustic as well as thermal/convective
modes are present. Viscous heating is quadratic in perturbations and
vanishes from this linearization.

At each wall, the boundary conditions are

.. math::

   w=0,\qquad \partial U-\frac{U}{r}=0,\qquad \theta=0.

The full tangential traction coefficient is
:math:`\mu(\partial U-U/r+s w/r)`; impermeability reduces it to the
condition shown above. Replacing it with :math:`\partial U=0` would impose
the planar condition at finite curvature.

For stationary onset set :math:`\sigma=0` and introduce
:math:`\Theta=g\theta/\mu`, :math:`Q=q/\mu`. The generalized Rayleigh
eigenproblem becomes

.. math::

   0&=\rho_b\delta+\rho_b'w,\\
   0&=-\frac{s}{r}Q+F_h,\\
   0&=-\partial Q-\frac{g}{T_b}Q+\frac{\rho_b}{T_b}\Theta+F_r,\\
   S\Theta&=-Ra_{\rm sa}\,\rho_b\frac{H(r)}{a-{\cal D}}w.

The mass-flux constraint in this stationary pencil follows from the full
continuity equation at :math:`\sigma=0`; it does not invoke an anelastic
approximation. The numerical growth convention is
:math:`\widehat\sigma=\sigma d^2/\chi_m=\sigma/\chi_m`, with positive
real part indicating instability. The code returns both
:math:`\widehat\sigma` and the physical :math:`\sigma`.

The onset search covers poloidal degrees :math:`\ell\ge1`. Toroidal
perturbations decouple from buoyancy; the neutral rigid rotation at
:math:`\ell=1` is excluded from the convection criterion. The separate
:math:`\ell=0` radial acoustic/thermal sector is not an overturning mode.

Mixed-degree discretization and pressure verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``spherical_compressible_reference.py`` uses Chebyshev degree :math:`n` for
:math:`U,w,\theta` on :math:`n+1` Lobatto nodes, and degree :math:`n-2` for
pressure. Pressure unknowns are Chebyshev coefficients; their values and
derivatives are evaluated at the velocity nodes. Continuity is enforced at
the :math:`n-1` interior nodes. The two momentum equations and thermal
equation are imposed in the interior, with their endpoint rows replaced by
the six wall conditions. This gives a square four-field pencil of dimension
:math:`4(n+1)-2`. No additional pressure boundary condition is imposed.

An equal-degree pressure representation was found to admit an unresolved
pressure branch whose apparent damping approaches zero as radial degree
increases. For example, at :math:`\eta=0.5,\ell=3,Ra=820,n=48`, that branch
has :math:`\widehat\sigma=-0.35312131` and a matrix residual of
:math:`9.4\times10^{-17}`, yet its pressure oscillates strongly and its
off-grid momentum residual is approximately one. It is not a physical
slowly decaying mode. The mixed-degree formulation removes this artifact
and gives :math:`\widehat\sigma=-0.4296686893` consistently for
:math:`n=24,32,48,64`.

As a separate check, every selected growth eigenvector is interpolated and
differentiated analytically on at least 257 Gauss points between the
collocation nodes. The four differential equations are evaluated there
independently of the assembled matrices. For each equation the residual is
the maximum absolute imbalance divided by the sum of the maximum absolute
values of its individual terms. A candidate is rejected if any such
residual exceeds :math:`5\times10^{-6}`; rejected candidates are recorded.
Selection proceeds in decreasing real growth rate among finite eigenvalues,
without an arbitrary growth-rate cutoff. No candidate needed rejection in
the final mixed-degree radius-ratio sweep. The original equal-degree
artifact remains a regression test of the off-grid diagnostic.

The neutral solver also rescales the low-wavenumber eigenvalue as
:math:`\lambda=Ra\,k_m^2`, where :math:`k_m=s/r_m`, and balances field and
equation scales. This is an exact algebraic transformation that prevents
very large low-degree neutral Rayleigh numbers from being misclassified as
infinite eigenvalues in the thin-shell limit.

Reference curve and planar limit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each radius ratio the search samples the continuous neutral curve from
:math:`\ell=1` to :math:`k_m=16`, refines every sampled local minimum, and
then evaluates adjacent integer degrees. Small degree ranges, up to 128
degrees, are checked exhaustively. The high-wavenumber tail must be increasing
and more than ten times the minimum. The JSON records the complete sampled
bracket and integer candidates. This is a numerical global-bracket audit,
not a mathematical proof over unbounded degrees.

The following values use :math:`n=48`; the fractional differences refer to
the independently recomputed planar threshold. All values use the same
:math:`a,{\cal D},\gamma,Pr` and midpoint units defined above.

.. csv-table:: Fully compressible stationary onset reference
   :header: "Radius ratio", "Critical degree", "Midpoint wavenumber", "Critical Ra", "Difference from plane (%)"
   :widths: 15, 15, 24, 24, 22

   0.5, 3, 2.30940108, 865.98306270, 24.393066
   0.6, 4, 2.23606798, 797.10854448, 14.499671
   0.7, 6, 2.28732025, 755.76183511, 8.560474
   0.8, 10, 2.33068633, 728.60159059, 4.659074
   0.9, 21, 2.26254582, 708.31171683, 1.744560
   0.95, 43, 2.23062186, 701.60935176, 0.781807
   0.98, 111, 2.25250260, 698.15664028, 0.285847
   0.99, 224, 2.25627581, 697.13665548, 0.139332
   0.9999, 22561, 2.25626281, 696.17612966, 0.001359
   Planar, Continuous, 2.25631961, 696.16666977, 0

As :math:`\eta\to1` with :math:`d=1`, :math:`r_m\to\infty`. Holding
:math:`z=r-r_m` and :math:`k_m=\sqrt{\ell(\ell+1)}/r_m` fixed gives

.. math::

   T_b\to1-az,\quad
   \rho_b\to(1-az)^{g/a-1},\quad
   H\to a-{\cal D},\quad
   S\to\partial_z^2-k_m^2,\quad
   \delta\to\partial_z w-k_m U.

The curvature terms vanish, :math:`\partial U-U/r=0` tends to
:math:`\partial_z U=0`, and the spacing between permitted midpoint
wavenumbers tends to zero. Both pencils therefore approach the matched
**compressible planar** equations. Compressibility remains finite in this
limit; the target is :math:`696.16667`, rather than the Boussinesq value
:math:`27\pi^4/4=657.51136`. At :math:`\eta=0.9999` the remaining relative
difference from the compressible plane is about :math:`1.36\times10^{-5}`.

Validation evidence and profile export
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The completed reference calculation verifies all nine radius ratios at
:math:`n=24,32,48`. Across these runs, adjacent radial-resolution estimates
of :math:`Ra_c` differ by less than :math:`2\times10^{-9}`, and the selected
integer degree is unchanged. The leading physical growth rate is negative
at :math:`0.98Ra_c`, zero to within :math:`1.5\times10^{-11}` in thermal
diffusion units at :math:`Ra_c`, and positive at :math:`1.02Ra_c`.
The supercritical growth rates converge to within :math:`8\times10^{-12}`.

Neutral and growth matrix residuals are below :math:`3\times10^{-17}` and
:math:`2\times10^{-16}`, respectively. The independent off-grid differential
residuals for the three threshold-neighboring growth rates are below
:math:`10^{-9}`. A broader scan of integer angular degrees at
:math:`0.98Ra_c`, including acoustic eigenvalues, finds no positive growth
on its sampled degrees; its off-grid residuals are below
:math:`3.1\times10^{-8}`. The scan is a numerical check of the stationary
interpretation, not an exhaustive theorem excluding every possible mode.

Additional checks cover the analytic hydrostatic integral (including
:math:`A=0`), the conductive temperature derivative, wall traction, and the
vector identity
:math:`\nabla\!\cdot\boldsymbol\tau/\mu=(4/3)\nabla\Delta\phi`
for :math:`\boldsymbol u=\nabla\phi`. Profile convergence at
:math:`\eta=0.5,\ell=3,Ra=820,900` uses :math:`n=24,32,48,64` and 128
independent radial centers. All four physical fields agree to better than
:math:`6\times10^{-12}\chi_m`, including pressure. These are CPU spectral
reference checks; they make no claim about GPU or cubed-sphere solver error.

Reproduce the reference and its assertions with::

   OPENBLAS_NUM_THREADS=1 python docs/notes/spherical_compressible_reference.py

The output ``spherical_reference_results.json`` contains the bracket audits,
resolution levels, growth checks, residuals, pressure-artifact regression,
and neutral-mode samples for each radius ratio. Reusable Python functions
are ``background``, ``neutral``, ``critical``, ``growth``, ``eigenmode``, and
``export_profile``. For example, export a physical decaying mode with::

   OPENBLAS_NUM_THREADS=1 python docs/notes/spherical_compressible_reference.py \
     --export-profile /tmp/shell-profile.txt --eta .5 --ell 3 --ra 820 --nr 12

The headerless file has exactly ``nr`` ascending rows at
:math:`z_i=(i+1/2)/N_r-1/2`, with columns ``z,w,U,Tpert,ppert``. It uses the
growth eigenvector at the requested physical Rayleigh number, rather than
a neutral eigenvector at a different Rayleigh number. Every field is scaled
by the same factor so that :math:`\max_i|w_i|=\chi_m` on the exported
centers. A runner must multiply **all four perturbations** by its small seed
amplitude. The coefficient of :math:`\nabla_\Omega Y` is :math:`U/\sqrt L`.

A read-only audit of the existing planar calculation confirmed that its
quoted :math:`n=48` growth rates at :math:`Ra=680,695,710` agree with a
mixed-degree pressure calculation to about :math:`2\times10^{-11}`.
The equal-degree planar pencil also develops a spurious leading branch at
:math:`n=64,Ra=680`; its reported rate there would be :math:`-0.13922552`
instead of the resolved physical :math:`-0.17793503`. The published
:math:`n=48` comparison and measured planar neutral curve are unaffected.
No planar source or result file was changed by this spherical reference work.
