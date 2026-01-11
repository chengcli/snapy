Exchange: Distributed Communication
===================================

The exchange module provides utilities for distributed communication in parallel simulations.

.. module:: snapy.exchange

Functions
---------

.. function:: init_dist(args, periodic_x1: bool = False, periodic_x2: bool = False, periodic_x3: bool = False) -> tuple

   Initialize distributed environment.

   :param args: Arguments containing device, layout, and process counts
   :param periodic_x1: Periodic boundary in x1 direction
   :type periodic_x1: bool, optional
   :param periodic_x2: Periodic boundary in x2 direction
   :type periodic_x2: bool, optional
   :param periodic_x3: Periodic boundary in x3 direction
   :type periodic_x3: bool, optional
   :return: Tuple of (layout, ranks, device, info)
   :rtype: tuple

.. function:: init_buffers_2d(layout, rank: int, block: MeshBlock, block_vars: dict) -> tuple

   Initialize communication buffers for 2D decomposition.

   :param layout: Domain layout
   :param rank: Process rank
   :type rank: int
   :param block: Mesh block
   :type block: snapy.MeshBlock
   :param block_vars: Dictionary of block variables
   :type block_vars: dict[str, torch.Tensor]
   :return: Tuple of (send_buffers, recv_buffers)
   :rtype: tuple

.. function:: serialize_2d(block: MeshBlock, block_vars: dict, send_bufs: list) -> None

   Serialize data into send buffers for 2D decomposition.

   :param block: Mesh block
   :type block: snapy.MeshBlock
   :param block_vars: Dictionary of block variables
   :type block_vars: dict[str, torch.Tensor]
   :param send_bufs: List of send buffers
   :type send_bufs: list

.. function:: deserialize_2d(block: MeshBlock, block_vars: dict, recv_bufs: list) -> None

   Deserialize data from receive buffers for 2D decomposition.

   :param block: Mesh block
   :type block: snapy.MeshBlock
   :param block_vars: Dictionary of block variables
   :type block_vars: dict[str, torch.Tensor]
   :param recv_bufs: List of receive buffers
   :type recv_bufs: list

.. function:: slab_exchange(block: MeshBlock, block_vars: dict, ranks: list, send_bufs: list, recv_bufs: list) -> None

   Exchange halo data for slab layout.

   :param block: Mesh block
   :type block: snapy.MeshBlock
   :param block_vars: Dictionary of block variables
   :type block_vars: dict[str, torch.Tensor]
   :param ranks: List of neighbor ranks
   :type ranks: list[int]
   :param send_bufs: List of send buffers
   :type send_bufs: list
   :param recv_bufs: List of receive buffers
   :type recv_bufs: list

.. function:: get_buffer_id(dx: int, dy: int, dz: int = 0) -> int

   Get buffer ID for a given offset.

   :param dx: Offset in x direction
   :type dx: int
   :param dy: Offset in y direction
   :type dy: int
   :param dz: Offset in z direction
   :type dz: int, optional
   :return: Buffer ID
   :rtype: int

Examples
--------

Initialize distributed environment::

    from snapy import exchange
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layout", default="slab")
    parser.add_argument("--px3", type=int, default=2)
    parser.add_argument("--px2", type=int, default=2)
    parser.add_argument("--px1", type=int, default=1)
    args = parser.parse_args()

    layout, ranks, device, info = exchange.init_dist(
        args,
        periodic_x3=True,
        periodic_x2=True
    )

Set up communication buffers::

    send_bufs, recv_bufs = exchange.init_buffers_2d(
        layout, rank, block, block_vars
    )

Exchange halo data::

    exchange.slab_exchange(
        block, block_vars, ranks, send_bufs, recv_bufs
    )

Distributed Layouts
-------------------

The exchange module supports three layout types:

**Slab Layout**
   2D domain decomposition in the x2-x3 plane. Suitable for simulations with vertical structure.

**Cubed Layout**
   3D domain decomposition. Suitable for fully 3D simulations.

**Cubed Sphere Layout**
   Domain decomposition on a cubed sphere. Suitable for global atmospheric simulations.

Communication Pattern
---------------------

The exchange pattern follows these steps:

1. **Serialize**: Copy data from block variables to contiguous send buffers
2. **Exchange**: Use PyTorch distributed P2P operations to send/receive data
3. **Deserialize**: Copy data from receive buffers to block ghost cells

This pattern minimizes communication overhead and maximizes GPU utilization.
