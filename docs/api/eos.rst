Equation of State Module
========================

.. module:: snapy

Classes
-------

EquationOfStateOptions
~~~~~~~~~~~~~~~~~~~~~~

.. class:: EquationOfStateOptions

   Equation of state configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> EquationOfStateOptions

      Load EquationOfStateOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: EquationOfStateOptions loaded from file
      :rtype: EquationOfStateOptions

   .. method:: type() -> str
               type(value: str) -> EquationOfStateOptions

      Get or set the equation of state type.

      :return: EOS type (e.g., "ideal", "real")
      :rtype: str

   .. method:: gamma() -> float
               gamma(value: float) -> EquationOfStateOptions

      Get or set the adiabatic index (gamma).

      :return: Adiabatic index
      :rtype: float
