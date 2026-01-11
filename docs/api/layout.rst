Layout Module
=============

.. module:: snapy

Classes
-------

LayoutOptions
~~~~~~~~~~~~~

.. class:: LayoutOptions

   Domain layout configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> LayoutOptions

      Load LayoutOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: LayoutOptions loaded from file
      :rtype: LayoutOptions

   .. method:: type() -> str
               type(value: str) -> LayoutOptions

      Get or set the layout type.

      :return: Layout type (e.g., "slab", "cubed", "cubed_sphere")
      :rtype: str

SlabLayout
~~~~~~~~~~

.. class:: SlabLayout

   2D slab layout for domain decomposition.

   .. method:: __init__(nb3: int, nb2: int, periodic_x3: bool = False, periodic_x2: bool = False)

      Initialize a SlabLayout.

      :param nb3: Number of blocks in x3 direction
      :type nb3: int
      :param nb2: Number of blocks in x2 direction
      :type nb2: int
      :param periodic_x3: Periodic in x3 direction
      :type periodic_x3: bool, optional
      :param periodic_x2: Periodic in x2 direction
      :type periodic_x2: bool, optional

   .. method:: loc_of(rank: int) -> tuple

      Get the location of a rank.

      :param rank: Rank number
      :type rank: int
      :return: Location tuple (x3, x2)
      :rtype: tuple[int, int]

   .. method:: neighbor_rank(x3: int, x2: int, dx3: int, dx2: int, dx1: int = 0) -> int

      Get the neighbor rank.

      :param x3: Current x3 position
      :type x3: int
      :param x2: Current x2 position
      :type x2: int
      :param dx3: Offset in x3 direction
      :type dx3: int
      :param dx2: Offset in x2 direction
      :type dx2: int
      :param dx1: Offset in x1 direction
      :type dx1: int, optional
      :return: Neighbor rank or -1 if no neighbor
      :rtype: int

CubedLayout
~~~~~~~~~~~

.. class:: CubedLayout

   3D cubed layout for domain decomposition.

   .. method:: __init__(nb3: int, nb2: int, nb1: int, periodic_x3: bool = False, periodic_x2: bool = False, periodic_x1: bool = False)

      Initialize a CubedLayout.

      :param nb3: Number of blocks in x3 direction
      :type nb3: int
      :param nb2: Number of blocks in x2 direction
      :type nb2: int
      :param nb1: Number of blocks in x1 direction
      :type nb1: int
      :param periodic_x3: Periodic in x3 direction
      :type periodic_x3: bool, optional
      :param periodic_x2: Periodic in x2 direction
      :type periodic_x2: bool, optional
      :param periodic_x1: Periodic in x1 direction
      :type periodic_x1: bool, optional

   .. method:: loc_of(rank: int) -> tuple

      Get the location of a rank.

      :param rank: Rank number
      :type rank: int
      :return: Location tuple (x3, x2, x1)
      :rtype: tuple[int, int, int]

   .. method:: neighbor_rank(x3: int, x2: int, x1: int, dx3: int, dx2: int, dx1: int) -> int

      Get the neighbor rank.

      :param x3: Current x3 position
      :type x3: int
      :param x2: Current x2 position
      :type x2: int
      :param x1: Current x1 position
      :type x1: int
      :param dx3: Offset in x3 direction
      :type dx3: int
      :param dx2: Offset in x2 direction
      :type dx2: int
      :param dx1: Offset in x1 direction
      :type dx1: int
      :return: Neighbor rank or -1 if no neighbor
      :rtype: int

CubedSphereLayout
~~~~~~~~~~~~~~~~~

.. class:: CubedSphereLayout

   Cubed sphere layout for domain decomposition.

   .. method:: __init__(nb_per_face: int)

      Initialize a CubedSphereLayout.

      :param nb_per_face: Number of blocks per face dimension
      :type nb_per_face: int

   .. method:: loc_of(rank: int) -> tuple

      Get the location of a rank.

      :param rank: Rank number
      :type rank: int
      :return: Location tuple (face, x3, x2)
      :rtype: tuple[int, int, int]

   .. method:: neighbor_rank(face: int, x3: int, x2: int, dx3: int, dx2: int, dx1: int = 0) -> int

      Get the neighbor rank.

      :param face: Current face
      :type face: int
      :param x3: Current x3 position
      :type x3: int
      :param x2: Current x2 position
      :type x2: int
      :param dx3: Offset in x3 direction
      :type dx3: int
      :param dx2: Offset in x2 direction
      :type dx2: int
      :param dx1: Offset in x1 direction
      :type dx1: int, optional
      :return: Neighbor rank or -1 if no neighbor
      :rtype: int

DistributeInfo
~~~~~~~~~~~~~~

.. class:: DistributeInfo

   Information about distributed domain decomposition.

   .. method:: gid(value: int = None) -> int

      Get or set global rank ID.

      :param value: Value to set
      :type value: int, optional
      :return: Global rank ID
      :rtype: int

   .. method:: nb3(value: int = None) -> int

      Get or set number of blocks in x3.

      :param value: Value to set
      :type value: int, optional
      :return: Number of blocks
      :rtype: int

   .. method:: nb2(value: int = None) -> int

      Get or set number of blocks in x2.

      :param value: Value to set
      :type value: int, optional
      :return: Number of blocks
      :rtype: int

   .. method:: nb1(value: int = None) -> int

      Get or set number of blocks in x1.

      :param value: Value to set
      :type value: int, optional
      :return: Number of blocks
      :rtype: int

   .. method:: lx3(value: int = None) -> int

      Get or set local x3 position.

      :param value: Value to set
      :type value: int, optional
      :return: Local x3 position
      :rtype: int

   .. method:: lx2(value: int = None) -> int

      Get or set local x2 position.

      :param value: Value to set
      :type value: int, optional
      :return: Local x2 position
      :rtype: int

   .. method:: lx1(value: int = None) -> int

      Get or set local x1 position.

      :param value: Value to set
      :type value: int, optional
      :return: Local x1 position
      :rtype: int

   .. method:: face(value: int = None) -> int

      Get or set face number (for cubed sphere).

      :param value: Value to set
      :type value: int, optional
      :return: Face number
      :rtype: int
