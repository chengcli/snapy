# Flat-bottom JW06 cubed-sphere validation

The case uses a reflecting spherical shell from 113 m to 30 km above the JW06 reference radius. Its latitude-dependent pressure, temperature and jets are evaluated at geometric height in FP64. The disturbed run adds the published 1 m/s zonal wind perturbation. The pressure diagnostic samples the lowest cell centre. The 120 s comparisons below test early evolution. The development of the baroclinic wave over several days requires a longer integration. [Jablonowski and Williamson (2006), equations 1–11](https://www.gfdl.noaa.gov/wp-content/uploads/files/user_files/pjp/qj_jablonowski_williamson_2006.pdf)

The benchmark uses the dry ideal-moist EOS, HLLC, WENO5, RK3, constant gravity and traditional Coriolis forcing. All six panels run on one device in one process. FP64 uses conventional total-state fluxes. FP32 uses anomaly storage and split fluxes. Both use the same physical options, timestep and accepted-step checks.

## Tensor copies in the flux rotation and diagnostic counters

The x2 and x3 gnomonic flux rotations now retain the two intermediate
momentum rows and round each to the flux dtype before lowering its index.
The diagnostic register reduces each posted mask or count vector into one
int64 slot. The stage still publishes counts at its flush, and a rejected
step discards its pending counts.

A matched Nsight Systems capture on dart10 GPU 1 covered one complete C48/L26
FP32 RK3 step after ten warm-up steps. The CUDA runtime recorded 1,818
device-to-device copies totaling 1,385.2 MiB on the baseline and 1,602
totaling 1,257.2 MiB with these changes. ATen copy kernels fell from 3,390
to 3,264 executions. Their summed durations were 30.75 and 28.71 kernel-ms
across six overlapping block streams. The elapsed times below were measured
with tracing disabled.

Each GPU pair compared FP32 on the prior draft (`5f8bc5f4`) and `ab2575bf`
on the same Quadro RTX 4000. The binaries used the same Python binding,
deck and CUDA kernels. Run order alternated across nine pairs per grid.
The table gives medians for 100 accepted RK3 steps after ten warm-up steps.

| Grid | Host, GPU | Prior (s) | Updated (s) | Time saved | Updated wins | Peak allocated MiB, prior → updated |
|---|---|---:|---:|---:|---:|---:|
| C48/L26 | dart11, GPU 1 | 11.561 | 11.356 | 1.8% | 9/9 | 469.8 → 414.6 |
| C96/L26 | dart10, GPU 1 | 36.427 | 35.787 | 1.8% | 8/9 | 1,631.6 → 1,425.8 |

On the development CPU, C48 eager FP32 medians were 22.949 s before and
21.661 s after across three alternating pairs. The existing C48 precision
gate on dart10 GPU 0 also passed with the updated code in both modes. FP32
took 9.825 s and conventional FP64 took 11.159 s
(9/9 FP32 wins).

For the revision comparison, run the following command against each build,
using `--resolution 96` for C96. The CPU measurements used
`--device cpu --execution eager`.

```sh
python examples/jw06.py --resolution 48 --mode fp32 --device cuda --execution cuda-graph --warmup 10 --steps 100 --dt 1
```

At C16/L8, the reconstructed primitive fields after three accepted GPU steps
were bitwise equal to the prior build for both six and 24 blocks. The final
120 s GPU pressure differences between FP32 and FP64 were 9.46e-5 Pa RMS
(disturbed) and 9.28e-5 Pa RMS (balanced). The corresponding C48–C96
differences rounded to 0.532 Pa RMS.

The local release build passed all 12 selected CTest entries. On dart10,
nine native tests passed across the register, CUDA graph save/rollback and
horizontal flux rounding cases. The science checks and FP64 speed gate used
the following commands.

```sh
ctest --test-dir /tmp/snapy-mailbox-compare/build -R 'test_(coordinate|flux_register|face_floor|graph_monitor_amounts|riemann|cubed_sphere_exchange|cubed_sphere_seams).release|test_cubed_sphere_fp32_rest_python|test_jw06.*python|test_fp32_(face_floor_counts|horizontal_sweep)_python' --output-on-failure
python tests/test_jw06.py --device cuda --subdivide 2
python tests/run_jw06_science.py --device cuda --seconds 120
python tests/run_jw06_science.py --device cuda --seconds 120 --balanced
python tests/bench_jw06.py --resolution 48
```

## Optimized GPU speed gate

The release build of `8bc11ceb` ran on GPU 0 of each Quadro RTX 4000 host.
The two hosts used identical native libraries. Each pair compares 100 complete
accepted RK3 steps after ten warm-up steps. Both modes used strict CUDA graph
replay. Precision order alternated across nine pairs per host and resolution.

| Grid | Host | FP64 median, s | FP32 median, s | Wall time saved | FP32 wins |
|---|---|---:|---:|---:|---:|
| C48/L26 | dart10 | 11.427 | 10.034 | 12.2% | 9/9 |
| C48/L26 | dart11 | 11.403 | 10.076 | 11.6% | 9/9 |
| C96/L26 | dart10 | 39.797 | 34.199 | 14.1% | 9/9 |
| C96/L26 | dart11 | 39.984 | 34.331 | 14.1% | 9/9 |

```sh
python tests/bench_jw06.py --resolution 48
python tests/bench_jw06.py --resolution 96
```

## Correctness

The CPU and both Quadro RTX 4000 hosts passed the six-block and 24-block comparisons, including a check that subdivision preserves the panel solution. The test compares both split FP64 and FP32 against conventional FP64. On CUDA it also compares FP32 graph replay to eager execution with zero tolerance. The independent FP32 rest test measured a maximum velocity of 4.13e-11 times the sound speed after 20 cycles.

```sh
python tests/test_jw06.py --device cpu
python tests/test_jw06.py --device cpu --subdivide 2
python tests/test_cubed_sphere_cell_volume.py --split-precision float32
python tests/test_jw06.py --device cuda
python tests/test_jw06.py --device cuda --subdivide 2
python tests/test_cubed_sphere_cell_volume.py --device cuda --split-precision float32
```

The science driver compares C48 FP32 and FP64 pressure at identical model times, and restricts the C96 FP64 field to C48 with cell-volume weights. At 120 s on CPU the disturbed run gave a precision difference of 9.46e-5 Pa RMS and a resolution difference of 0.532 Pa RMS. The corresponding unperturbed values were 9.29e-5 Pa and 0.532 Pa.

```sh
python tests/run_jw06_science.py --device cpu --seconds 120
python tests/run_jw06_science.py --device cpu --seconds 120 --balanced
```

## Throughput before the optimization

Each GPU sample times 100 complete accepted RK3 steps at dt = 1 s after 10 warm-up steps. The executable synchronizes CUDA at both timing boundaries and checks graph replay and final state validity. The two modes run in separate processes on the same GPU, in alternating order across nine pairs. Acceptance requires seven FP32 wins and a lower FP32 median on each host. Both hosts ran identical native libraries and Python bindings with Torch 2.10.0+cu128 and Kintera 2.6.14. The feature commit is `2213d8b3` and the benchmark replay check is `a2c7828b`.

On the development CPU one C48/L26 matched pair took 21.27 s in FP64 and 25.48 s in FP32. CPU timing is informational.

```sh
for mode in fp64 fp32; do
  python examples/jw06.py --mode "$mode" --steps 100 --warmup 10 --dt 1
done
python tests/bench_jw06.py
python tests/bench_jw06.py --resolution 96
```

| C48/L26, seconds per 100 steps | dart10 | dart11 |
|---|---:|---:|
| FP64 graph, nine-pair median | 11.343 | 11.417 |
| FP32 split graph, nine-pair median | 13.555 | 13.606 |
| FP32 wins | 0/9 | 0/9 |
| FP64 split graph, one control sample | 15.533 | 15.642 |
| FP64 eager, one control sample | 11.988 | 12.058 |
| FP32 split eager, one control sample | 14.331 | 14.349 |

| C96/L26, seconds per 100 steps | dart10 | dart11 |
|---|---:|---:|
| FP64 graph, nine-pair median | 39.764 | 40.014 |
| FP32 split graph, nine-pair median | 47.353 | 47.620 |
| FP32 wins | 0/9 | 0/9 |

FP32 took 19.5% longer on dart10 and 19.2% longer on dart11 at C48. Narrowing the split algorithm recovered some time relative to split FP64. Conventional FP64 still had the lowest elapsed time on both GPUs.

```sh
python examples/jw06.py --mode split64 --device cuda --execution cuda-graph --warmup 10 --steps 100 --dt 1
for mode in fp64 fp32; do
  python examples/jw06.py --mode "$mode" --device cuda --execution eager --warmup 10 --steps 100 --dt 1
done
```

For the 120 s GPU science checks, dart10 ran the disturbed case and dart11 ran the unperturbed case. The FP32 precision differences were 9.43e-5 Pa and 9.27e-5 Pa RMS, respectively. The corresponding FP64 resolution differences were both 0.532 Pa RMS.

```sh
python tests/run_jw06_science.py --device cuda --seconds 120
python tests/run_jw06_science.py --device cuda --seconds 120 --balanced
```

## Comparison with Kratos

Wang reports a 7.2× full-single and 5.5× mixed-precision speedup over double precision on the RTX 4090 for 2-D double Mach reflection. Kratos keeps conserved state in double precision in mixed mode and uses single precision for reconstruction and the Riemann solve. Its GPU implementation reuses face data in shared memory. Snapy's JW06 gate changes the stored representation as well as arithmetic precision. Its split-FP64 control measures the extra cost of that algorithm on the same atmosphere and GPU. The timings here give the combined effect of those changes. [Wang (2025), sections 3.2–3.3 and Tables 1–2](https://arxiv.org/html/2501.02317v2)

A post-benchmark Nsight Systems capture on dart10 covered one accepted C48 RK3 step after ten warm-up steps, with CUDA graph node tracing. Durations below sum individual kernel execution intervals across six block streams. Overlap and profiler instrumentation make those sums different from uninstrumented wall time.

| Per profiled RK3 step | Conventional FP64 | Split FP64 | Split FP32 |
|---|---:|---:|---:|
| GPU kernel executions | 18,932 | 21,782 | 23,996 |
| HLLC kernels, summed ms | 8.05 | 8.99 | 1.85 |
| Radial hydrostatic reference kernels, summed ms | 16.23 | 40.56 | 40.22 |
| ATen copy kernels, summed ms | 14.41 | 18.54 | 33.91 |
| All GPU kernels, summed ms | 217.38 | 284.23 | 261.01 |

All three paths executed 54 HLLC kernels. FP32 reduced their summed time by a factor of 4.35 relative to conventional FP64. Both split paths still used the paired radial reference kernel. Its summed time and the additional copy work account for much of the difference in the kernel totals. The copy category covers kernels with `copy_kernel` or `direct_copy` in their symbols and includes data movement within device memory.

## Four-field radial reference

For a uniform column of 6–128 cells that fits in 32 KiB of scratch and has no radial neighbour,
the CPU and CUDA operators read background density and pressure in FP64 and
their stored increments in FP32. Each column evaluates the original nonlinear
pressure scan once, using paired FP64 arithmetic for the anchor, pressure floor
and reference stencils. It writes the five totals and four pressure components
directly. On CUDA the workers for a column share the face pressures and smoothed
density-to-pressure ratios in shared memory. Stretched, vertically decomposed
and longer columns continue to use the general paired operator.

The horizontal cubed-sphere solve exchanges total face pressure before forming
the shared gauge. The radial reconstruction forms the pressure and density
differences in FP64 before narrowing its working state. The conserved state
and restart representation retain their configured precision.

The direct-operator regression compares each returned field with the packed
operator, including pressure increments below one FP64 ulp and a pressure floor.
The JW06 timing record also reports the peak CUDA allocation in bytes, measured
from the end of warm-up through the last timed synchronization.

The kernel computes each cell's hydrostatic pressure drop and each
extrapolated wall sample once per column. The scan still accumulates from the
upper anchor in the original order. Dry FP32 HLLC reads the total pressure and
density for its thermodynamics and retains the residual pressure for the flux.
A regression with a nonzero internal-energy offset checks the face energy,
adiabatic index and sound speed against the same EOS evaluated on total states.

The local CPU regression command selects the native reference and HLLC oracles,
the split-column exchange, reconstruction floors, perturbation fluxes, restart
and JW06 checks.

```sh
ctest --test-dir /tmp/snapy-mailbox-compare/build -R 'test_(eos|riemann|hydro_ref_x1|split_reference).release|test_cubed_sphere_fp32_rest_python|test_jw06.*python|test_fp32_(split|face_floor_counts|restart|horizontal_sweep)_python|test_perturbation_flux.*python|test_x1_seam_split.release' --output-on-failure
```

All 15 tests passed. One C48 CPU sample took 21.38 s in conventional FP64 and
23.30 s in optimized FP32, compared with the earlier 25.48 s FP32 sample.

On dart10 the native oracle ran on both CPU and CUDA. Fourteen reference
tests and all four dry-face tests passed. The two FP64 cases of the FP32-only
wall-stencil test were skipped. At 120 model seconds the final GPU pressure
differences were 9.46e-5 Pa RMS for the disturbed case on dart10 and 9.28e-5
Pa RMS for the unperturbed case on dart11. Both resolution differences were
0.532 Pa RMS. The six- and 24-block tests also compare strict graph execution
with eager execution on the same initial state.

### Effect of the changes

On dart11's second GPU, two C48 samples per variant ran in forward and reverse
order. The initial four-field implementation took 12.60 and 12.90 s per 100
steps. Reusing the pressure drops and extended wall stencil took 11.39 and
11.44 s. Removing dry face thermodynamic staging alone took 12.28 and 12.57 s.
Enabling both took 11.27 and 11.31 s.

On dart10's second GPU, three alternating old/new FP32 pairs per resolution
used the same Python binding and deck. Median C48 time fell from 14.879 s to
11.266 s. Median C96 time fell from 51.066 s to 36.417 s. These comparisons
use the saved pre-optimization libraries and the production build of
`8bc11ceb`.

| One traced C48 RK3 step, dart10 GPU 1 | Conventional FP64 | Previous FP32 | Optimized FP32 |
|---|---:|---:|---:|
| Kernel executions | 18,932 | 23,996 | 22,862 |
| Radial reference, summed kernel-ms | 15.91 | 41.72 | 10.53 |
| HLLC, summed kernel-ms | 7.78 | 1.88 | 1.90 |
| ATen copies, summed kernel-ms | 13.90 | 33.97 | 30.48 |

All three traces contain 18 radial evaluations and 54 HLLC solves. The largest
reduction came from the radial operator. Kernel times are summed across
overlapping block streams. The gate measures synchronized wall time with
profiling disabled.

| Peak CUDA allocation after warm-up, MiB | Conventional FP64 | Previous FP32 | Optimized FP32 |
|---|---:|---:|---:|
| C48/L26 | 462.1 | 467.3 | 469.8 |
| C96/L26 | 1,572.7 | 1,639.0 | 1,631.6 |

The split solver stores conserved fields in FP32 and retains FP64 reference
fields and work buffers. Its measured peak allocation exceeded conventional
FP64 at both sizes.

Additional C48 controls on dart10 GPU 1 took 17.30 s for split FP64 with
graph replay, 19.64 s for eager FP32 and 19.01 s for eager conventional
FP64. These were single samples while GPU 0 ran the gate. The paired speed gate uses graph replay.
