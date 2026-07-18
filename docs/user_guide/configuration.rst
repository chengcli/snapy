Configuration
=============

Snapy simulations are configured using YAML files that specify mesh parameters, time integration settings, physical models, and output options.

Basic Configuration
-------------------

A minimal configuration file looks like this:

.. code-block:: yaml

    mesh:
      nx1: 100        # Grid points in x1 direction
      nx2: 100        # Grid points in x2 direction
      nx3: 1          # Grid points in x3 direction (1 for 2D)
      x1min: 0.0      # Domain minimum in x1
      x1max: 1.0      # Domain maximum in x1
      x2min: 0.0      # Domain minimum in x2
      x2max: 1.0      # Domain maximum in x2

    time:
      cfl: 0.8        # CFL number for time step control
      tlim: 1.0       # Simulation end time

    hydro:
      gamma: 1.4      # Adiabatic index
      riemann: hllc   # Riemann solver type

    output:
      dt: 0.1         # Output interval

Mesh Configuration
------------------

The mesh section defines the computational domain:

.. code-block:: yaml

    mesh:
      # Grid dimensions
      nx1: 100
      nx2: 100
      nx3: 50

      # Domain extent
      x1min: 0.0
      x1max: 10.0
      x2min: 0.0
      x2max: 10.0
      x3min: 0.0
      x3max: 5.0

      # Boundary conditions
      bc_x1: periodic
      bc_x2: periodic
      bc_x3: outflow

Coordinate Systems
~~~~~~~~~~~~~~~~~~

Snapy supports multiple coordinate systems:

* **Cartesian**: Standard rectangular coordinates
* **Cylindrical**: (r, θ, z) coordinates
* **Spherical**: (r, θ, φ) coordinates

.. code-block:: yaml

    coordinate:
      type: cartesian  # or cylindrical, spherical

Time Integration
----------------

Configure time stepping and integration schemes:

.. code-block:: yaml

    time:
      cfl: 0.8              # CFL number (0 < CFL < 1)
      tlim: 100.0           # Simulation end time
      nlim: 10000           # Maximum number of steps
      integrator: rk3       # Integration scheme (rk2, rk3, rk4)

Hydrodynamics
-------------

Hydrodynamic solver configuration:

.. code-block:: yaml

    hydro:
      # Equation of state
      gamma: 1.4            # Adiabatic index
      eos: ideal            # EOS type (ideal, real)

      # Riemann solver
      riemann: hllc         # Solver type (hllc, roe, hll)

      # Reconstruction
      reconstruction: plm    # Scheme (plm, ppm, weno)

      # Flux calculation
      disable_flux_x1: false
      disable_flux_x2: false
      disable_flux_x3: false

Physical Forcing
----------------

Add gravitational and Coriolis forcing:

.. code-block:: yaml

    hydro:
      gravity:
        enabled: true
        g: 9.81             # Gravitational acceleration
        direction: -1       # Direction (-1 for downward in x1)

      coriolis:
        enabled: true
        omega: 7.27e-5      # Rotation rate (rad/s)
        latitude: 45.0      # Latitude (degrees)

Add constant isotropic viscosity and heat conduction on a Cartesian mesh:

.. code-block:: yaml

    forcing:
      diffusion:
        nu_iso: 0.0         # Kinematic viscosity
        kappa_iso: 0.0      # Thermal diffusivity (length^2 / time)

Diffusion is integrated explicitly and contributes a parabolic time-step
limit. Curved coordinates, anisotropic coefficients, and spatially varying
coefficients are not supported. Heat conduction uses the energy flux
``-rho * cv * kappa_iso * grad(T)``, where ``cv`` is the local mixture
specific heat supplied by the equation of state. An EOS without a positive
reference specific heat at constant volume cannot enable heat conduction.

Cubed-Sphere Scalar Hyperdiffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply fourth-order horizontal diffusion to vertical velocity and selected
moisture fields on a cubed sphere:

.. code-block:: yaml

    forcing:
      scalar-hyperdiffusion:
        damping-time: 3600.0
        fields: [vel1, H2O, H2O(l), H2O(l,p)]

``vel1`` selects vertical velocity. All other entries must exactly match a
configured non-dry vapor or cloud species name; precipitating species are
selected by their cloud species name. Diffusion is horizontal in ``x2`` and
``x3`` only.

The strength is set by ``damping-time`` rather than a coefficient with units
of length to the fourth power per time. At each radial level, snapy derives
the coefficient so that the metric-aware discrete grid-scale reference mode
has the requested e-folding time.

For a scalar :math:`\phi`, the second-order intermediate operator is

.. math::

   \mathcal{L}_\rho(\phi) =
   \rho^{-1}\nabla_h\mathbin{\cdot}(\rho\nabla_h\phi).

The fourth-order tendency is obtained from
:math:`-K_4\mathcal{L}_\rho(\mathcal{L}_\rho\phi)`. The intermediate
Laplacian is exchanged across block and panel ghost zones before the second
application. Species partial densities are updated conservatively. For
``vel1``, radial momentum and total energy receive consistent conservative
fluxes. The explicit stability limit is included in the hydro time-step
estimate.

Implicit Correction
-------------------

For stiff problems, enable implicit correction:

.. code-block:: yaml

    hydro:
      implicit:
        enabled: true
        max_iter: 100
        tolerance: 1e-6
        method: newton      # Newton-Raphson method

Output Configuration
--------------------

Control simulation output:

.. code-block:: yaml

    output:
      # Basic output settings
      dt: 0.1               # Output interval
      format: netcdf        # Output format (netcdf, hdf5)

      # Variables to output
      variables:
        - density
        - velocity
        - pressure
        - temperature

      # Output file settings
      basename: simulation
      directory: output/

Advanced Output
~~~~~~~~~~~~~~~

Configure multiple output streams:

.. code-block:: yaml

    output:
      streams:
        - id: 1
          dt: 0.1
          variables: [density, velocity, pressure]

        - id: 2
          dt: 1.0
          variables: [temperature, energy]

        - id: 3
          dt: 10.0
          variables: [diagnostics]

Loading Configuration
---------------------

Load configuration in Python:

.. code-block:: python

    import snapy

    # Load from YAML file
    options = snapy.MeshBlockOptions.from_yaml("config.yaml")

    # Create mesh block
    block = snapy.MeshBlock(options)

You can also configure individual components:

.. code-block:: python

    # Configure hydrodynamics
    hydro_opts = snapy.HydroOptions.from_yaml("config.yaml")
    hydro_opts.riemann().type("hllc")
    hydro_opts.eos().gamma(1.4)

    # Configure EOS separately
    eos_opts = snapy.EquationOfStateOptions()
    eos_opts.type("ideal")
    eos_opts.gamma(1.4)
    hydro_opts.eos(eos_opts)
