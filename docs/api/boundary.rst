Boundary Module
===============

.. module:: snapy

Classes
-------

BoundaryFuncOptions
~~~~~~~~~~~~~~~~~~~

.. class:: BoundaryFuncOptions

   Options for boundary functions.

   .. method:: dir() -> int

      Get the boundary direction.

      :return: Direction index
      :rtype: int

InternalBoundaryOptions
~~~~~~~~~~~~~~~~~~~~~~~

.. class:: InternalBoundaryOptions

   Internal boundary configuration options.

   .. staticmethod:: from_yaml(filename: str, verbose: bool = False) -> InternalBoundaryOptions

      Load InternalBoundaryOptions from a YAML file.

      :param filename: Path to YAML file
      :type filename: str
      :param verbose: Enable verbose output
      :type verbose: bool, optional
      :return: InternalBoundaryOptions loaded from file
      :rtype: InternalBoundaryOptions

External outflow and extrapolation
---------------------------------

For ``ideal-gas``, ``ideal-moist``, and ``moist-mixture``, ``outflow`` is a
characteristic radiation condition about the fixed initial state. Outgoing
acoustic and advected perturbations are copied from the nearest interior cell;
incoming perturbations are zero. Mode speeds use the current outward normal
velocity and EOS sound speed, including backflow and supersonic flow. Zero-speed
modes are retained. Each ghost cell keeps its own initial background, so an
unperturbed stratified profile is preserved. All perturbations, including tracer
mixing ratios, are reduced together if necessary to preserve admissibility.
There is no sponge layer.

Initialize valid density, pressure, composition, velocity, and tracer mixing
ratios throughout the grid, including all ghost layers, before initialization.
The background is captured as independent copies before physical boundary fills.
To launch a pulse relative to a uniform background, initialize the uniform state
first, then add the pulse to the evolving state.

Use ``extrapolation`` for the previous nearest-interior-cell copy condition.
``shallow-water`` outflow keeps this copy behavior. Auxiliary scalar fields such
as positivity factors also use copying; transported tracers use the hydro
characteristic context. Other EOS types are rejected for characteristic outflow.

Restart files now include ``boundary_reference_w`` and, when tracers are
present, ``boundary_reference_r``. Characteristic outflow rejects old restart
files without these fields: initialize afresh or select ``extrapolation``.
The references are restored on the state device and dtype.

The C++ callback signature is unchanged, but ``BoundaryFuncOptions`` now carries
EOS, coordinate, and reference tensors. Rebuild compiled custom callbacks.
Characteristic outflow faces operate on primitives and convert only their
ghosts back to conserved variables. Other face callbacks receive the requested
representation. Consecutive radiating faces share a conversion; their ghosts
are committed before the next nonradiating callback. Faces are filled in x1, x2, x3 order; internal process faces and
collapsed dimensions are skipped.

For manual Python boundary fills, use
``block.apply_boundaries(vars, hydro, tracers=None, primitive=False)`` with the
initialized variables map. Pass transported tracers so hydro and tracer
perturbations are limited together. Set ``primitive=True`` when supplying hydro
primitives and tracer mixing ratios instead of conserved variables.

The acoustic regression uses a Gaussian pulse of density amplitude ``1e-4``
on 128 cells with HLLC and RK3. The recorded reflected characteristic amplitude
fractions (relative to the initial outgoing amplitude) are:

============  ==================  ==================
Scheme        outflow             extrapolation
============  ==================  ==================
PLM           2.38333e-5          7.74746e-7
WENO5         2.92124e-5          9.64716e-6
============  ==================  ==================

Both orientations give the same values to the shown precision. These uniform
1D cases verify the 5% reflection bound; they do not demonstrate an improvement
over extrapolation. The test prints fresh measurements on each run.
