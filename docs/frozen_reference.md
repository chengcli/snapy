# Frozen x1 reconstruction reference

`dynamics/wb-density-ref: frozen` captures the `smooth5` reference once during
fresh mesh initialization, after primitive ghost exchange and before the first
integration stage. It freezes all four reconstruction targets: lower-face
pressure `psf_lo`, cell pressure `pref`, lower-face density `dsf`, and cell
density `dref`. In the #250 D3 terminology these are the face and volume
pressure/density reference targets; the code calls the volume density `dref`,
not `dsv`.

The primitive and conserved pressure/density fields continue to evolve. At
every reconstruction the current primitives have the *initial* cell targets
subtracted; WENO reconstructs those perturbations and adds the *initial* face
targets. Gravity, thermodynamics, numerical fluxes, and positivity fallbacks
are unchanged. Thus this is an a-priori reference after initialization, not a
method that fixes the physical pressure or density. Initial anomalies are part
of the target; this option does not infer an unperturbed background.

The four older options (`smooth5`, `isentrope`, `none`, `local_polytrope`) keep
their existing path and defaults. `frozen` stores four additional scalar
fields per cell in `Variables["wb_frozen_x1"]`. Normal restart output includes
this tensor. Restart initialization restores it and rejects missing or
incompatible reference data rather than anchoring at restart time. Changing
mesh topology or the state device/dtype after initialization requires a new
compatible initialization; no remapping of the fixed target is implemented.
Direct users of `HydroImpl`, outside `Mesh`/`MeshBlock` initialization, must call
`initialize_wb_reference(vars)` before advancing. Normal `Mesh` initialization
does this automatically. The inherited study branch's restrictions on x1
partitioning still apply.

`test_face_floor.release` includes a frozen-reference regression that checks
initial equivalence with smooth5, invariance after both pressure and density
are changed, nonzero prognostic response, and restoring/rejecting restart
reference data. `test_hydro_options.release` checks selection through YAML.
