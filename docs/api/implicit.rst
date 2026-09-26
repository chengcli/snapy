Implicit Solver Module
======================

.. module:: snapy

Classes
-------

ImplicitOptions
~~~~~~~~~~~~~~~

.. class:: ImplicitOptions

   Implicit solver configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> ImplicitOptions | None

      Load ImplicitOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: ImplicitOptions loaded from file, or ``None`` when the file sets
               no implicit scheme (``integration/implicit-scheme`` absent or 0;
               ``implicit-advection-cfl`` or ``shear-cfl`` then raise)
      :rtype: ImplicitOptions or None

   .. method:: scheme() -> int
               scheme(value: int) -> ImplicitOptions

      Get or set the implicit scheme. From YAML it is ``integration/implicit-scheme``;
      ``0`` there means no implicit correction at all (``from_yaml`` returns ``None``).

      :return: ``0`` (none), ``1`` (vertical implicit correction, 3x3 blocks) or
               ``9`` (vertical implicit correction, 5x5 blocks)
      :rtype: int

   .. method:: type() -> str

      Name of the current scheme (read-only).

      :return: ``"none"``, ``"vic-partial"`` or ``"vic-full"``; any other
               ``scheme`` raises
      :rtype: str

   .. method:: advection_cfl() -> float
               advection_cfl(value: float) -> ImplicitOptions

      Get or set the advective Courant number of the x1 time-step bound when
      the implicit scheme has removed the acoustic one there:
      ``dt <= advection_cfl * dx1 / |v1|``. It applies only when the grid has
      more than one cell in x2 or x3; a 1-D column keeps the acoustic bound.
      From YAML it is ``integration/implicit-advection-cfl``; it must be finite
      and > 0.

      :return: Advective Courant number (default 1.0)
      :rtype: float

   .. method:: shear_cfl() -> float
               shear_cfl(value: float) -> ImplicitOptions

      Get or set the Courant number of an extra time-step bound at each x1 face
      across which a horizontal wind component ``v`` jumps by at least the
      face sound speed ``cs``: ``dt <= shear_cfl * cs * dx / (|v_lo| * |v_hi|)``,
      with ``dx`` the cell width in that horizontal direction. From YAML it is
      ``integration/shear-cfl``; it must be finite and >= 0, and 0 switches the
      bound off.

      :return: Shear Courant number (default 0.0)
      :rtype: float
