# #250 T3: nonlinear cases with published references

Study-only files for chengcli/snapy#250 (which density reference `rho_ref`
the well-balanced x1 reconstruction should use). Not meant for main.

- `straka.yaml`: Straka et al. (1993) density current, the control
  (dry neutral background, where isentrope is exact).
- `straka_t3.cpp`: `examples/straka.cpp` plus K lap(theta), applied by
  operator splitting after each step. snapy's `kappa_iso` is Fourier
  conduction of T. On this dry-adiabatic background it heats the column
  (2.5 K in 900 s at 100 m), so it is off here and theta is diffused as
  Straka specifies. Build it with `bash build_driver.sh <snapy build>`.
- `bryan.yaml`: Bryan & Fritsch (2002) saturated neutral moist bubble, the
  discriminating case. Run with `examples/bryan.cpp` unchanged.
- Forms: smooth5, isentrope, none, and local_polytrope. local_polytrope is ported from
  cshsgy/snapy@612976c `src/hydro/hydro_rho_ref_study.cpp:36-62`.
- `run_t3.py`: case x resolution x `dynamics/wb-density-ref` matrix,
  plus the unperturbed background (`dT = 0`) for the initial-state residual
  and CPU spot checks. Grids wider than 1024 cells along x2 are split into two
  x2 blocks in one process, because the CUDA WENO5 stencil block holds one
  thread per cell.
- `analyze_t3.py`, `make_table.py`, `plot_t3.py`: metrics against the
  published values (sources in the `analyze_t3.py` docstring), tables and figure.

```
bash build_driver.sh ../../build
python run_t3.py --bin ../../build/bin --out RUNDIR --cpu-spot
python analyze_t3.py RUNDIR --json t3.json && python make_table.py t3.json
python plot_t3.py RUNDIR 50 100 t3.png
```
