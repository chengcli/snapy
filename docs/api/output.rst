Output Module
=============

.. module:: snapy

Classes
-------

OutputOptions
~~~~~~~~~~~~~

.. class:: OutputOptions

   Output configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> OutputOptions

      Load OutputOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: OutputOptions loaded from file
      :rtype: OutputOptions

   .. method:: dt() -> float
               dt(value: float) -> OutputOptions

      Get or set the output time interval.

      :return: Output interval
      :rtype: float

   .. method:: format() -> str
               format(value: str) -> OutputOptions

      Get or set the output format.

      :return: Output format (e.g., "netcdf", "hdf5")
      :rtype: str
