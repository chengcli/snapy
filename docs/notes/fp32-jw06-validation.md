# Flat-bottom JW06 cubed-sphere validation

The case uses a reflecting spherical shell from 113 m to 30 km above the JW06 reference radius. Its latitude-dependent pressure, temperature and jets are evaluated at geometric height in FP64. The disturbed run adds the published 1 m/s zonal wind perturbation. The pressure diagnostic samples the lowest cell centre. The 120 s comparisons below test early evolution. The development of the baroclinic wave over several days requires a longer integration. [Jablonowski and Williamson (2006), equations 1–11](https://www.gfdl.noaa.gov/wp-content/uploads/files/user_files/pjp/qj_jablonowski_williamson_2006.pdf)

The benchmark uses the dry ideal-moist EOS, HLLC, WENO5, RK3, constant gravity and traditional Coriolis forcing. All six panels run on one device in one process. FP64 uses conventional total-state fluxes. FP32 uses anomaly storage and split fluxes. Both use the same physical options, timestep and accepted-step checks.

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

## Throughput

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
