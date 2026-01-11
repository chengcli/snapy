Examples
========

This section provides complete working examples of Snapy simulations.

Basic Examples
--------------

1D Shock Tube (Sod Problem)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classic 1D shock tube test:

.. code-block:: python

    import snapy
    import torch
    
    # Configuration
    options = snapy.MeshBlockOptions.from_yaml("shock.yaml")
    block = snapy.MeshBlock(options)
    
    # Initial conditions: Sod shock tube
    nx1 = 100
    hydro_w = torch.zeros((5, 1, 1, nx1), dtype=torch.float64)
    
    # Left state (high pressure)
    hydro_w[snapy.kIDN, :, :, :nx1//2] = 1.0
    hydro_w[snapy.kIPR, :, :, :nx1//2] = 1.0
    
    # Right state (low pressure)
    hydro_w[snapy.kIDN, :, :, nx1//2:] = 0.125
    hydro_w[snapy.kIPR, :, :, nx1//2:] = 0.1
    
    # Initialize
    vars = {"hydro_w": hydro_w}
    vars, time = block.initialize(vars)
    
    # Time integration
    max_time = 0.2
    while time < max_time:
        dt = block.max_time_step(vars)
        vars = block.forward(dt, 0, vars)
        time += dt
    
    # Output
    block.make_outputs(vars, time, final_write=True)

2D Cold Bubble (Straka)
~~~~~~~~~~~~~~~~~~~~~~~~

Straka cold bubble convection test:

.. code-block:: python

    import snapy
    import torch
    import numpy as np
    
    options = snapy.MeshBlockOptions.from_yaml("straka.yaml")
    block = snapy.MeshBlock(options)
    
    # Domain parameters
    nx1, nx2, nx3 = 64, 128, 1
    xc, zc = 5000.0, 3000.0  # Bubble center
    rx, rz = 4000.0, 2000.0  # Bubble radii
    dtheta = -15.0            # Temperature perturbation (K)
    
    # Base state
    T0 = 300.0
    p0 = 100000.0
    g = 9.81
    Rd = 287.0
    gamma = 1.4
    
    hydro_w = torch.zeros((5, nx3, nx2, nx1), dtype=torch.float64)
    
    # Create grid
    x = torch.linspace(0, 10000, nx2)
    z = torch.linspace(0, 6400, nx1)
    
    # Hydrostatic base state
    for i in range(nx1):
        T = T0
        p = p0 * np.exp(-g * z[i] / (Rd * T))
        rho = p / (Rd * T)
        
        hydro_w[snapy.kIDN, :, :, i] = rho
        hydro_w[snapy.kIPR, :, :, i] = p
    
    # Add cold bubble perturbation
    for j in range(nx2):
        for i in range(nx1):
            r = np.sqrt(((x[j] - xc) / rx)**2 + ((z[i] - zc) / rz)**2)
            if r <= 1.0:
                dT = dtheta * np.cos(0.5 * np.pi * r)**2
                T = T0 + dT
                p = hydro_w[snapy.kIPR, 0, j, i]
                hydro_w[snapy.kIDN, 0, j, i] = p / (Rd * T)
    
    # Initialize and run
    vars = {"hydro_w": hydro_w}
    vars, time = block.initialize(vars)
    
    max_time = 900.0
    output_interval = 30.0
    next_output = output_interval
    
    while time < max_time:
        dt = block.max_time_step(vars)
        vars = block.forward(dt, 0, vars)
        time += dt
        
        if time >= next_output:
            block.make_outputs(vars, time)
            next_output += output_interval
            print(f"Time: {time:.1f} s")

Advanced Examples
-----------------

Distributed Simulation
~~~~~~~~~~~~~~~~~~~~~~

Multi-GPU simulation with domain decomposition:

.. code-block:: python

    import torch
    import torch.distributed as dist
    from snapy import exchange
    import snapy
    import argparse
    
    def main():
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--layout", default="slab")
        parser.add_argument("--px3", type=int, default=2)
        parser.add_argument("--px2", type=int, default=2)
        parser.add_argument("--px1", type=int, default=1)
        args = parser.parse_args()
        
        # Initialize distributed
        layout, ranks, device, info = exchange.init_dist(
            args, periodic_x3=True, periodic_x2=True
        )
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Load configuration
        options = snapy.MeshBlockOptions.from_yaml("config.yaml")
        block = snapy.MeshBlock(options)
        
        # Initialize
        vars, time = block.initialize({})
        
        # Communication buffers
        send_bufs, recv_bufs = exchange.init_buffers_2d(
            layout, rank, block, vars
        )
        
        # Time integration
        max_time = 10.0
        output_interval = 0.5
        next_output = output_interval
        
        while time < max_time:
            # Exchange halos
            exchange.slab_exchange(block, vars, ranks, send_bufs, recv_bufs)
            
            # Time step
            dt = block.max_time_step(vars)
            vars = block.forward(dt, 0, vars)
            time += dt
            
            # Output
            if time >= next_output:
                if rank == 0:
                    print(f"Time: {time:.3f}")
                block.make_outputs(vars, time)
                next_output += output_interval
        
        # Cleanup
        dist.destroy_process_group()
    
    if __name__ == "__main__":
        main()

Run with::

    torchrun --nproc_per_node=4 distributed_sim.py \
             --device=cuda --layout=slab --px3=2 --px2=2

Moist Convection
~~~~~~~~~~~~~~~~

Simulation with moisture and phase changes:

.. code-block:: python

    import snapy
    import torch
    from kintera import ThermoOptions, ThermoX
    
    # Load thermodynamics
    thermo_opts = ThermoOptions.from_yaml("earth_moist.yaml")
    thermo = ThermoX(thermo_opts)
    
    # Load simulation configuration
    options = snapy.MeshBlockOptions.from_yaml("earth_moist.yaml")
    block = snapy.MeshBlock(options)
    
    # Initial conditions with moisture
    nx1, nx2, nx3 = 64, 128, 1
    nspecies = len(thermo_opts.species())
    
    hydro_w = torch.zeros((5 + nspecies, nx3, nx2, nx1), 
                          dtype=torch.float64)
    
    # Base state (temperature, pressure, humidity)
    T0 = 300.0
    p0 = 100000.0
    rh0 = 0.8  # 80% relative humidity
    
    # Set initial state with moisture
    # (implementation depends on your moisture model)
    
    # Initialize and run
    vars = {"hydro_w": hydro_w}
    vars, time = block.initialize(vars)
    
    max_time = 3600.0  # 1 hour
    while time < max_time:
        dt = block.max_time_step(vars)
        vars = block.forward(dt, 0, vars)
        time += dt

Custom Physics
--------------

Adding Custom Forcing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import snapy
    import torch
    
    def custom_forcing(block, vars, dt):
        """Apply custom forcing term."""
        rho = vars["hydro_w"][snapy.kIDN, :, :, :]
        vel1 = vars["hydro_w"][snapy.kIV1, :, :, :]
        
        # Example: Add damping layer at top
        nx1 = vel1.shape[-1]
        z_damp_start = nx1 * 0.8
        
        for i in range(int(z_damp_start), nx1):
            # Rayleigh damping coefficient
            z_frac = (i - z_damp_start) / (nx1 - z_damp_start)
            damp_coef = 0.1 * z_frac**2
            
            # Apply damping
            vars["hydro_w"][snapy.kIV1, :, :, i] *= (1.0 - damp_coef * dt)
            vars["hydro_w"][snapy.kIV2, :, :, i] *= (1.0 - damp_coef * dt)
            vars["hydro_w"][snapy.kIV3, :, :, i] *= (1.0 - damp_coef * dt)
        
        return vars
    
    # Apply in time loop
    while time < max_time:
        dt = block.max_time_step(vars)
        vars = block.forward(dt, 0, vars)
        vars = custom_forcing(block, vars, dt)
        time += dt

Configuration Files
-------------------

Example YAML configurations are provided in the ``examples/`` directory:

* ``shock.yaml``: 1D shock tube
* ``straka.yaml``: 2D cold bubble
* ``earth_moist.yaml``: Moist convection

These serve as templates for your own simulations.

More Examples
-------------

Additional examples can be found in the repository:

* ``examples/plume.py``: Thermal plume rising
* ``examples/explosion.py``: Blast wave propagation
* ``examples/topo_example.py``: Flow over topography
* ``examples/example_crm_dist.py``: Cloud-resolving model (distributed)

Each example includes both Python code and corresponding YAML configuration files.
