Reconstruction Module
=====================

.. module:: snapy

Classes
-------

ReconstructOptions
~~~~~~~~~~~~~~~~~~

.. class:: ReconstructOptions

   Reconstruction scheme configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> ReconstructOptions

      Load ReconstructOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: ReconstructOptions loaded from file
      :rtype: ReconstructOptions

   .. method:: type() -> str
               type(value: str) -> ReconstructOptions

      Get or set the reconstruction type.

      :return: Reconstruction type (e.g., "plm", "ppm", "weno")
      :rtype: str
