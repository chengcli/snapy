Quick Start Guide
=================

This guide will help you get started with Snapy quickly.

Basic Example
-------------

Here's a minimal example showing how to set up and run a simulation:

.. code-block:: python

    import snapy
    import torch

    # Load configuration from YAML file
    options = snapy.MeshBlockOptions.from_yaml("config.yaml")

    # Create a mesh block
    block = snapy.MeshBlock(options)

    # Initialize variables
    vars, time = block.initialize({})

    # Time integration loop
    dt = 0.01
    for step in range(100):
        # Forward integration
        vars = block.forward(dt, 0, vars)

        # Generate outputs
        if step % 10 == 0:
            block.make_outputs(vars, time)

        time += dt

Configuration
-------------

Simulations are configured using YAML files. Here's a basic example:

.. code-block:: yaml

    mesh:
      nx1: 100
      nx2: 100
      nx3: 1
      x1min: 0.0
      x1max: 1.0
      x2min: 0.0
      x2max: 1.0

    time:
      cfl: 0.8
      tlim: 1.0

    hydro:
      gamma: 1.4
      riemann: hllc

    output:
      dt: 0.1

Working with Distributed Simulations
-------------------------------------

For parallel simulations using distributed computing:

.. code-block:: python

    import torch.distributed as dist
    from snapy import exchange

    # Initialize distributed environment
    layout, ranks, device, info = exchange.init_dist(
        args,
        periodic_x3=True,
        periodic_x2=True
    )

    # Create mesh block with distributed info
    options = snapy.MeshBlockOptions.from_yaml("config.yaml")
    block = snapy.MeshBlock(options)

    # Initialize communication buffers
    send_bufs, recv_bufs = exchange.init_buffers_2d(
        layout, rank, block, block_vars
    )

    # Exchange halo data
    exchange.slab_exchange(
        block, block_vars, ranks, send_bufs, recv_bufs
    )

Next Steps
----------

* See :doc:`user_guide/index` for detailed documentation
* Check out :doc:`examples` for more complete examples
* Explore the :doc:`api/index` for API reference
