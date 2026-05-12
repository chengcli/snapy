Running Simulations
===================

This guide covers how to run Snapy simulations, from initialization to time integration.

Basic Simulation Loop
---------------------

A typical simulation follows this pattern:

.. code-block:: python

    import snapy
    import torch

    # Load configuration
    options = snapy.MeshBlockOptions.from_yaml("config.yaml")

    # Create mesh block
    block = snapy.MeshBlock(options)

    # Initialize variables
    vars, time = block.initialize({})

    # Main time integration loop
    dt = 0.01
    max_time = 10.0
    output_interval = 0.1
    next_output = output_interval

    while time < max_time:
        # Calculate time step
        dt = min(dt, block.max_time_step(vars))
        dt = min(dt, max_time - time)

        # Forward integration (single step)
        vars = block.forward(dt, 0, vars)
        time += dt

        # Output
        if time >= next_output:
            block.make_outputs(vars, time)
            next_output += output_interval
            print(f"Time: {time:.3f}, dt: {dt:.6f}")

    # Final output
    block.make_outputs(vars, time, final_write=True)
    block.finalize()

Multi-Stage Time Integration
-----------------------------

For Runge-Kutta schemes with multiple stages:

.. code-block:: python

    import snapy

    options = snapy.MeshBlockOptions.from_yaml("config.yaml")
    block = snapy.MeshBlock(options)
    vars, time = block.initialize({})

    # RK3 integration
    num_stages = 3
    dt = 0.01

    while time < max_time:
        dt = block.max_time_step(vars)

        # Multi-stage integration
        for stage in range(num_stages):
            vars = block.forward(dt, stage, vars)

        time += dt

        if time >= next_output:
            block.make_outputs(vars, time)
            next_output += output_interval

Initial Conditions
------------------

Setting Custom Initial Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can set custom initial conditions before starting the simulation:

.. code-block:: python

    import snapy
    import torch

    options = snapy.MeshBlockOptions.from_yaml("config.yaml")
    block = snapy.MeshBlock(options)

    # Get domain size
    nx3, nx2, nx1 = 128, 128, 64
    nvar = 5  # density, vel1, vel2, vel3, pressure

    # Create initial condition tensor
    hydro_w = torch.zeros((nvar, nx3, nx2, nx1), dtype=torch.float64)

    # Set uniform density
    hydro_w[snapy.kIDN, :, :, :] = 1.0

    # Set uniform pressure
    hydro_w[snapy.kIPR, :, :, :] = 1.0

    # Set velocity field (e.g., shear flow)
    x2 = torch.linspace(0, 1, nx2)
    hydro_w[snapy.kIV2, :, :, :] = x2[None, None, :, None]

    # Initialize with custom state
    vars = {"hydro_w": hydro_w}
    vars, time = block.initialize(vars)

Loading from File
~~~~~~~~~~~~~~~~~

Load initial conditions from a TorchScript file:

.. code-block:: python

    import torch

    # Load saved state
    module = torch.jit.load("initial_state.pt")
    hydro_w = module.hydro_w

    # Initialize simulation
    vars = {"hydro_w": hydro_w}
    vars, time = block.initialize(vars)

Adaptive Time Stepping
----------------------

The ``max_time_step`` method calculates the maximum stable time step based on the CFL condition:

.. code-block:: python

    # Automatic time step control
    dt_max = 0.1  # Maximum allowed time step

    while time < max_time:
        # Get CFL-limited time step
        dt_cfl = block.max_time_step(vars)

        # Use the minimum of CFL limit and user limit
        dt = min(dt_cfl, dt_max)

        # Don't overshoot end time
        dt = min(dt, max_time - time)

        vars = block.forward(dt, 0, vars)
        time += dt

Monitoring and Diagnostics
---------------------------

Monitor simulation progress:

.. code-block:: python

    import snapy

    def print_diagnostics(vars, time):
        """Print diagnostic information."""
        rho = vars["hydro_w"][snapy.kIDN, :, :, :]
        press = vars["hydro_w"][snapy.kIPR, :, :, :]

        print(f"Time: {time:.3f}")
        print(f"  Density: min={rho.min():.3e}, max={rho.max():.3e}")
        print(f"  Pressure: min={press.min():.3e}, max={press.max():.3e}")

    # Main loop with diagnostics
    diag_interval = 1.0
    next_diag = diag_interval

    while time < max_time:
        dt = block.max_time_step(vars)
        vars = block.forward(dt, 0, vars)
        time += dt

        if time >= next_diag:
            print_diagnostics(vars, time)
            next_diag += diag_interval

Custom Output Functions
-----------------------

Register custom output callbacks:

.. code-block:: python

    def user_output(vars):
        """Return custom derived output fields."""
        # Calculate derived quantities
        rho = vars["hydro_w"][snapy.kIDN, :, :, :]
        vel1 = vars["hydro_w"][snapy.kIV1, :, :, :]
        vel2 = vars["hydro_w"][snapy.kIV2, :, :, :]
        vel3 = vars["hydro_w"][snapy.kIV3, :, :, :]

        # Kinetic energy
        ke = 0.5 * rho * (vel1**2 + vel2**2 + vel3**2)

        return {"kinetic_energy": ke}

    # Register callback
    block.set_user_output_func(user_output)

Register custom forcing callbacks:

.. code-block:: python

    def user_forcing(vars, dt, stage):
        rho = vars["hydro_u"][snapy.kIDN, :, :, :]
        hydro_du = torch.zeros_like(vars["hydro_u"])
        hydro_du[snapy.kIDN] = 0.01 * dt * rho
        return {"hydro_du": hydro_du}

    block.set_user_forcing_func(user_forcing)

GPU Acceleration
----------------

Run on GPU by configuring the device:

.. code-block:: python

    import snapy
    import torch

    # Set default device to GPU
    device = torch.device("cuda:0")
    torch.set_default_device(device)

    # Load configuration
    options = snapy.MeshBlockOptions.from_yaml("config.yaml")
    block = snapy.MeshBlock(options)

    # All tensors will be on GPU
    vars, time = block.initialize({})

    # Run simulation on GPU
    while time < max_time:
        dt = block.max_time_step(vars)
        vars = block.forward(dt, 0, vars)
        time += dt

Restart from Checkpoint
-----------------------

Save and restart simulations:

.. code-block:: python

    import snapy
    import torch

    # Save checkpoint
    def save_checkpoint(vars, time, cycle):
        checkpoint = {
            "vars": vars,
            "time": time,
            "cycle": cycle
        }
        torch.save(checkpoint, f"checkpoint_{cycle:05d}.pt")

    # Load checkpoint
    def load_checkpoint(filename):
        checkpoint = torch.load(filename)
        return checkpoint["vars"], checkpoint["time"], checkpoint["cycle"]

    # Restart simulation
    if restart:
        vars, time, cycle = load_checkpoint("checkpoint_00100.pt")
    else:
        vars, time = block.initialize({})
        cycle = 0

    # Continue simulation
    while time < max_time:
        dt = block.max_time_step(vars)
        vars = block.forward(dt, 0, vars)
        time += dt
        cycle += 1

        # Save checkpoint every 100 cycles
        if cycle % 100 == 0:
            save_checkpoint(vars, time, cycle)
