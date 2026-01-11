Implicit Solver Module
======================

.. module:: snapy

Classes
-------

ImplicitOptions
~~~~~~~~~~~~~~~

.. class:: ImplicitOptions

   Implicit solver configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> ImplicitOptions

      Load ImplicitOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: ImplicitOptions loaded from file
      :rtype: ImplicitOptions

   .. method:: enabled() -> bool
               enabled(value: bool) -> ImplicitOptions

      Get or set whether implicit correction is enabled.

      :return: Enabled flag
      :rtype: bool

   .. method:: max_iter() -> int
               max_iter(value: int) -> ImplicitOptions

      Get or set the maximum number of iterations.

      :return: Maximum iterations
      :rtype: int

   .. method:: tolerance() -> float
               tolerance(value: float) -> ImplicitOptions

      Get or set the convergence tolerance.

      :return: Tolerance
      :rtype: float
