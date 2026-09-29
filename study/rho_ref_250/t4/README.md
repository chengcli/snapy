# #250 T4: robustness of the x1 density reference

Study-only harness for chengcli/snapy#250, T4. It re-creates the local T4
harness (93c445f, never pushed; not recoverable) from the issue text. Not
meant for main. It touches no library code: `t4_driver.cpp` reads the
production references through `HydroImpl::_hydro_ref_x1` (a derived-class
member pointer) and otherwise runs the normal `Mesh` loop.

- `t4_driver.cpp`: 2-D column (x1 up, x2 periodic), analytic background
  (`isothermal`, `polytrope`, `inversion`, `tropopause`), cosine temperature
  anomaly at fixed pressure, optional geometric x1 stretch `q`, optional
  single-cell fault injection, optional timed loop of reference calls. Writes
  `ref.*.bin` (t = 0 references per block), `state.*.bin` (final interior
  primitives), `summary.<rank>.json`.
- `run_t4.py`: the matrix (groups `anomaly`, `positivity`, `fault`,
  `stretch`, `seam`, `cost`) for the four `dynamics/wb-density-ref` forms.
  Multi-rank runs use `torchrun --standalone --no-python`, Gloo backend;
  `nb1 > 1` needs `layout: cubed` (slab refuses pz > 1).
- `analyze_t4.py`: tables. Seam jump = max |dsf_A(top face) - dsf_B(bottom
  face)| / rho(seam cell) at every shared x1 face; decomposition checks
  compare the final state bit for bit with the one-rank run of the same case,
  form and device.

```
bash build_t4.sh ../../../build
python run_t4.py --bin $PWD/../../../build/bin --out RUNDIR
python analyze_t4.py RUNDIR --json t4.json > t4_tables.md
```

`--bin` must be absolute (runs execute inside their own directories).
