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
