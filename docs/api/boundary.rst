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

EmbeddedBoundaryGeometry
~~~~~~~~~~~~~~~~~~~~~~~~

Install stationary Cartesian obstacles before initialization. Supply a level
set at all block vertices, including ghost vertices. Positive values denote
fluid. Tensor spatial axes are (x3, x2, x1). For a two-dimensional obstacle,
extrude the level set through both vertices of the inactive axis.

.. code-block:: python

   z, y, x = (block.buffer("coord." + name) for name in ("x3f", "x2f", "x1f"))
   zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
   phi = xx.square() + yy.square() - radius**2
   geometry = snapy.EmbeddedBoundaryGeometry.from_level_set(phi, x, y, z)
   block.set_embedded_boundary(geometry)

Alternatively, use ``from_metrics`` with the fluid volume fraction, fluid
centroid, three staggered face fractions and centroids, outward wall area
vector, wall centroid, and the three coordinate face arrays. Centroids are in
physical coordinates. Vector components are ordered (x1, x2, x3). The flag
values are 0 for regular, 1 for cut or wall-adjacent, and 2 for covered.

On gnomonic-equiangle cubed-sphere blocks, construct the mapped geometry with
``from_cubed_sphere_level_set(phi, x1f, x2f, x3f, panel)``. Supply radius in
length units and the two angular face axes in radians. Panel IDs 0 through 5
denote +X, +Y, -X, +Z, -Y, -Z respectively. Give the level set the same
physical value on shared vertices, including edges joining different panels.
The radius must stay positive and both angular coordinates must stay strictly
between -pi/2 and pi/2 through the ghost faces.

Mapped centroids and wall vectors use global Cartesian XYZ. The three
``face_area_vector`` arrays store open vector areas pointing in the direction of
increasing radius, first panel angle, or second panel angle. Each vector is
the integral of the surface normal over the aperture. The scalar area is
``background_face_area[d] * face_fraction[d]``. To supply mapped metrics
directly, use ``from_cubed_sphere_metrics`` with these vector areas after
``face_centroid``, and the panel ID after the three coordinate axes.

Cubed-sphere transport evaluates the Riemann flux on each open face and the
pressure force on each stationary wall in a common Cartesian frame. Weighted
state redistribution uses fluid volumes and physical centroids across panel
edges and cube corners, including for species and passive tracers. Enable
flux in every direction with more than one global cell. A single radial
layer retains pressure forces on both spherical surfaces. Set
``conserve-axial-momentum: off``, ``flux-covariance: false``, and
``x1-mass-covariance: false`` for mapped embedded boundaries.

Each block's exchange and reconstruction caches span the entire six-panel
mesh. Stage exchanges also gather fields over all owned cells into global
tensors. For a panel with ``n`` cells on each angular axis and ``nr`` radial
cells, each gathered field has ``6 * n * n * nr`` entries per component on
each participating block. Account for this replicated memory and communication
when choosing the resolution and number of blocks per process.

Embedded boundaries use explicit total-state dynamics with the
ideal-gas, ideal-moist, or moist-mixture EOS. Nonzero gravity and diffusion
require wall-specific energy or diffusive fluxes and are rejected at setup. Assign
geometry to every block of a decomposed mesh, including blocks containing
only fluid. Restarts store geometry and resume the accepted state. Input
decks and checkpoints containing the former Boolean solid mask must be
recreated with the geometry interface.

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
