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

   Constant isotropic hydro diffusion configuration for Cartesian meshes.

   .. method:: nu_iso() -> float
               nu_iso(value: float) -> DiffusionOptions

      Get or set kinematic viscosity.

   .. method:: kappa_iso() -> float
               kappa_iso(value: float) -> DiffusionOptions

      Get or set thermal diffusivity in units of length squared per time.
      The conductive energy flux is
      ``-rho * cv * kappa_iso * grad(T)``, where ``cv`` is the local
      equation-of-state mixture specific heat at constant volume.

ScalarHyperdiffusionOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: ScalarHyperdiffusionOptions

   Fourth-order horizontal scalar diffusion configuration for a cubed sphere.

   .. method:: damping_time() -> float
               damping_time(value: float) -> ScalarHyperdiffusionOptions

      Get or set the grid-scale e-folding time.

   .. method:: fields() -> list[str]
               fields(value: list[str]) -> ScalarHyperdiffusionOptions

      Get or set the selected ``vel1`` and vapor/cloud species names.
