Boundary Module
===============

.. module:: snapy

Classes
-------

BoundaryFuncOptions
~~~~~~~~~~~~~~~~~~~

.. class:: BoundaryFuncOptions

   Options for boundary functions.

   .. method:: dir() -> int

      Get the boundary direction.

      :return: Direction index
      :rtype: int

InternalBoundaryOptions
~~~~~~~~~~~~~~~~~~~~~~~

.. class:: InternalBoundaryOptions

   Internal boundary configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> InternalBoundaryOptions

      Load InternalBoundaryOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: InternalBoundaryOptions loaded from file
      :rtype: InternalBoundaryOptions
