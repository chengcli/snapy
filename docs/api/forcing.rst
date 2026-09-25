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

   Isotropic hydro diffusion configuration for Cartesian meshes.

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
