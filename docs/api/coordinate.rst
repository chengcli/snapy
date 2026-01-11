Coordinate Module
=================

.. module:: snapy

Classes
-------

CoordinateOptions
~~~~~~~~~~~~~~~~~

.. class:: CoordinateOptions

   Coordinate system configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> CoordinateOptions

      Load CoordinateOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: CoordinateOptions loaded from file
      :rtype: CoordinateOptions

   .. method:: type() -> str
               type(value: str) -> CoordinateOptions

      Get or set the coordinate system type.

      :return: Coordinate type (e.g., "cartesian", "spherical")
      :rtype: str
