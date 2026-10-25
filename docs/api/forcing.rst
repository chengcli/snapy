Forcing Module
==============

.. module:: snapy

Classes
-------

ConstGravityOptions
~~~~~~~~~~~~~~~~~~~

.. class:: ConstGravityOptions

   Constant gravity forcing configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> ConstGravityOptions

      Load ConstGravityOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: ConstGravityOptions loaded from file
      :rtype: ConstGravityOptions

   .. method:: g() -> float
               g(value: float) -> ConstGravityOptions

      Get or set the gravitational acceleration.

      :return: Gravitational acceleration
      :rtype: float

CoriolisOptions
~~~~~~~~~~~~~~~

.. class:: CoriolisOptions

   Coriolis forcing configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> CoriolisOptions

      Load CoriolisOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: CoriolisOptions loaded from file
      :rtype: CoriolisOptions

   .. method:: omega() -> float
               omega(value: float) -> CoriolisOptions

      Get or set the rotation rate.

      :return: Rotation rate
      :rtype: float

DiffusionOptions
~~~~~~~~~~~~~~~~

.. class:: DiffusionOptions

   Isotropic hydro diffusion configuration for Cartesian and equiangular
   cubed-sphere meshes. Cubed-sphere diffusion requires the cubed-sphere
   layout, ``nb1=1``, reflecting radial hydrodynamic boundaries, at least
   ``max(4, nghost)`` active cells in every direction, and ``nghost >= 2``.
   Calls on participating blocks must run concurrently, as for mesh exchange.

   .. method:: nu_iso() -> float
               nu_iso(value: float) -> DiffusionOptions

      Get or set the viscosity: kinematic by default, or the dynamic
      viscosity ``mu`` when :py:meth:`dynamic` is true.

   .. method:: kappa_iso() -> float
               kappa_iso(value: float) -> DiffusionOptions

      Get or set the thermal coefficient: a diffusivity in units of length
      squared per time by default, or the conductivity ``k`` when
      :py:meth:`dynamic` is true. In the default (kinematic) form the
      conductive energy flux is ``-rho * cv * kappa_iso * grad(T)``, where
      ``cv`` is the local equation-of-state mixture specific heat at constant
      volume; in the dynamic form it is ``-k * grad(T)``, with no face density.

   .. method:: dynamic() -> bool
               dynamic(value: bool) -> DiffusionOptions

      Read ``nu_iso`` as ``mu`` and ``kappa_iso`` as ``k``. Default ``False``,
      which leaves the previous path bit-identical.

   .. method:: temperature_inner() -> float
               temperature_inner(value: float) -> DiffusionOptions
               temperature_outer() -> float
               temperature_outer(value: float) -> DiffusionOptions

      Temperature in kelvin at the inner/outer x1 wall, supported on Cartesian
      and cubed-sphere meshes with reflecting hydrodynamic boundaries. Positive
      values prescribe the conductive boundary value independently of the
      hydrodynamic ghost state. Zero preserves the default: insulated on the
      cubed sphere, or the existing ghost-based heat flux on Cartesian meshes.

   .. method:: wall_velocity() -> str
               wall_velocity(value: str) -> DiffusionOptions

      Cubed-sphere radial viscous wall: ``"stress-free"`` (default) or
      ``"no-slip"``. Both are stationary and impermeable.

   .. method:: implicit_radial() -> bool
               implicit_radial(value: bool) -> DiffusionOptions

      Default ``False``. With ``True``, use a second-order L-stable SDIRK solve
      for radial heat conduction, Strang split around each full hydro step.
      Requires the dry ``ideal-gas`` EOS. Horizontal conduction and all
      viscosity remain explicit and retain timestep restrictions. Step retries
      restore the state before the first half-step. This option is independent
      of the acoustic vertical implicit correction. The solver rejects a
      nonpositive temperature; reduce the timestep for such sharp transients.

   The new wall and radial implicit options use the same names under
   ``forcing/diffusion`` in YAML. The prescribed wall temperatures also work
   on Cartesian meshes. The no-slip override and radial implicit solve remain
   specific to the cubed sphere. Defaults preserve the Cartesian behavior.
   See :doc:`../notes/cubed_sphere_diffusion_convection` for the
   metric, coefficient conventions and validation results.

   .. method:: nu_scale_x1_table() -> torch.Tensor
               nu_scale_x1_table(value: torch.Tensor) -> DiffusionOptions
               kappa_scale_x1_table() -> torch.Tensor
               kappa_scale_x1_table(value: torch.Tensor) -> DiffusionOptions

      An x1 profile multiplying the kinematic ``nu_iso`` (or ``kappa_iso``):
      a ``(2, n)`` float64 tensor of x1 knots (strictly increasing) and
      scale values (finite, strictly positive), ``n >= 2``. It is linear
      between knots, held beyond the ends, and interpolated onto each
      block's x1 cell centres, so it works with any number of blocks along
      x1. Undefined by default (a uniform coefficient). The time-step bound
      uses the largest scale.

   .. method:: nu_scale_x1() -> torch.Tensor
               nu_scale_x1(value: torch.Tensor) -> DiffusionOptions
               kappa_scale_x1() -> torch.Tensor
               kappa_scale_x1(value: torch.Tensor) -> DiffusionOptions

      The same profile given per cell: a 1-D float64 tensor with one value
      per x1 cell centre of the block, ghosts included (``nc1`` values).
      Refused when x1 is split into more than one block, since every block
      shares these options; use the table there.

   A profile is refused together with ``dynamic=True``, when both the table
   and the per-cell form are given, and when it is set, replaced or changed,
   or ``dynamic`` is set to true beside it, after the MeshBlock is built (at
   the next ``forward`` or ``max_time_step``, which compare the profile with a
   copy taken at build, so a write through a NumPy array behind
   ``torch.from_numpy`` or through ``.data`` is caught too). An inference tensor (made under
   ``torch.inference_mode()``) is refused, since its in-place changes cannot be
   detected; pass an ordinary tensor.
