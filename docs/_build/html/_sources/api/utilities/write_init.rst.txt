write_init: Write Initial Conditions
====================================

The write_init module provides utilities for writing initial conditions from NetCDF files to PyTorch TorchScript format.

.. module:: snapy.write_init

Functions
---------

.. function:: save_tensors(tensor_map: dict, filename: str) -> None

   Save a dictionary of tensors to a TorchScript file.

   :param tensor_map: Dictionary mapping names to tensors
   :type tensor_map: dict[str, torch.Tensor]
   :param filename: Output filename
   :type filename: str

.. function:: read_hydro(ymlfile: str, inpfile: str) -> torch.Tensor

   Read hydrodynamic variables from NetCDF file.

   :param ymlfile: Path to YAML configuration file
   :type ymlfile: str
   :param inpfile: Path to input NetCDF file
   :type inpfile: str
   :return: Tensor containing hydrodynamic variables
   :rtype: torch.Tensor

Examples
--------

Read initial conditions from NetCDF and save to TorchScript::

    from snapy.write_init import read_hydro, save_tensors
    
    # Read from NetCDF
    w = read_hydro('config.yaml', 'initial_state.nc')
    
    # Save to TorchScript format
    data = {'hydro_w': w}
    save_tensors(data, 'initial_state.pt')

Using from command line::

    python -m snapy.write_init

Data Format
-----------

The module expects NetCDF files with the following variables:

* ``rho``: Density (kg/m³)
* ``vel1``: Velocity in x1 direction (m/s)
* ``vel2``: Velocity in x2 direction (m/s)
* ``vel3``: Velocity in x3 direction (m/s)
* ``press``: Pressure (Pa)
* Species mole fractions (if applicable)

The output TorchScript file contains a tensor with shape::

    (nvar, nx3, nx2, nx1)

where:
    * ``nvar`` = 4 + number of species
    * Dimensions are in the order (x3, x2, x1)

Variable Order
--------------

The variables are ordered as:

0. Density (rho)
1. Velocity in x1 direction (vel1)
2. Velocity in x2 direction (vel2)
3. Velocity in x3 direction (vel3)
4. Pressure (press)
5+. Species mole fractions (if applicable)
