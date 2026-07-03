Hydrodynamics Module
====================

The hydro module handles hydrodynamic calculations.

.. module:: snapy

Classes
-------

Hydro
~~~~~

.. class:: Hydro

   Hydrodynamics implementation.

   .. method:: __init__(options: HydroOptions = None, block = None)

      Construct a Hydro module.

      :param options: Hydrodynamics configuration options
      :type options: HydroOptions, optional
      :param block: Parent block module
      :type block: optional

   .. method:: forward(*args) -> torch.Tensor

      Forward pass through the module.

      :return: Updated tensor
      :rtype: torch.Tensor

   .. method:: max_time_step(*args) -> float

      Calculate maximum stable time step.

      :return: Maximum stable time step
      :rtype: float

   .. attribute:: options

      Hydrodynamics configuration options.

      :type: HydroOptions

HydroOptions
~~~~~~~~~~~~

.. class:: HydroOptions

   Hydrodynamics configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> HydroOptions

      Load HydroOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: HydroOptions loaded from file
      :rtype: HydroOptions

   .. method:: verbose() -> bool
               verbose(value: bool) -> HydroOptions

      Get or set verbose flag.

      :return: Verbose flag
      :rtype: bool

   .. method:: disable_flux_x1() -> bool
               disable_flux_x1(value: bool) -> HydroOptions

      Get or set disable flux x1 flag.

      :return: Disable flux x1 flag
      :rtype: bool

   .. method:: disable_flux_x2() -> bool
               disable_flux_x2(value: bool) -> HydroOptions

      Get or set disable flux x2 flag.

      :return: Disable flux x2 flag
      :rtype: bool

   .. method:: disable_flux_x3() -> bool
               disable_flux_x3(value: bool) -> HydroOptions

      Get or set disable flux x3 flag.

      :return: Disable flux x3 flag
      :rtype: bool

   .. method:: fused_recon_riemann() -> bool
               fused_recon_riemann(value: bool) -> HydroOptions

      Get or set the resolved fused reconstruction/Riemann CUDA path status.
      For options loaded from YAML, the ``FUSED`` environment variable is the
      selection control: ``FUSED=OFF`` disables the path, ``FUSED=ON`` requires
      it and errors if unsupported, and ``FUSED=AUTO`` or an unset ``FUSED``
      enables it only when the configuration is supported.

      :return: Resolved fused reconstruction/Riemann status
      :rtype: bool

   .. method:: eos() -> EquationOfStateOptions
               eos(value: EquationOfStateOptions) -> HydroOptions

      Get or set equation of state options.

      :return: Equation of state options
      :rtype: EquationOfStateOptions

   .. method:: recon1() -> ReconstructOptions
               recon1(value: ReconstructOptions) -> HydroOptions

      Get or set reconstruction options for dimension 1.

      :return: Reconstruction options
      :rtype: ReconstructOptions

   .. method:: recon23() -> ReconstructOptions
               recon23(value: ReconstructOptions) -> HydroOptions

      Get or set reconstruction options for dimensions 2 and 3.

      :return: Reconstruction options
      :rtype: ReconstructOptions

   .. method:: riemann() -> RiemannSolverOptions
               riemann(value: RiemannSolverOptions) -> HydroOptions

      Get or set Riemann solver options.

      :return: Riemann solver options
      :rtype: RiemannSolverOptions

   .. method:: icorr() -> ImplicitOptions
               icorr(value: ImplicitOptions) -> HydroOptions

      Get or set implicit correction options.

      :return: Implicit correction options
      :rtype: ImplicitOptions
