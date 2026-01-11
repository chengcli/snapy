Riemann Solver Module
=====================

.. module:: snapy

Classes
-------

RiemannSolverOptions
~~~~~~~~~~~~~~~~~~~~

.. class:: RiemannSolverOptions

   Riemann solver configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> RiemannSolverOptions

      Load RiemannSolverOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: RiemannSolverOptions loaded from file
      :rtype: RiemannSolverOptions

   .. method:: type() -> str
               type(value: str) -> RiemannSolverOptions

      Get or set the Riemann solver type.

      :return: Solver type (e.g., "hllc", "roe", "hll")
      :rtype: str
