# Draft: CUDA test policy for CPU-only builds with a visible GPU

A CPU-only Snapy build can see a CUDA-capable PyTorch installation and a GPU. Runtime availability checks then select CUDA paths whose Snapy kernels were not built. This draft separates the build-capability policy from the wall-corner fix in PR #265; it proposes no implementation decision.

## Reproduction and measured main base

- Repository: `chengcli/snapy`; current `origin/main` checked at task start: `5eeb9b6761ae484a98b3993aae18314fc58cf862`.
- This is **not** `e630d916da3237b36c7f7db2349fe68fab9599ee`, the PR #265 baseline used in the earlier controlled comparison.
- CPU-only build: `CUDA=OFF`, `UCX=OFF`, `FULL_TESTS=OFF`, `PNETCDF=OFF`, `BUILD_TESTS=ON`, `BUILD_TESTING=OFF` (dependency tests disabled).
- Python 3.11, Torch 2.10.0+cu128, Kintera **2.5.13**, NVIDIA GeForce RTX 5090 visible as device 0; `torch.cuda.is_available()` is true.
- `CUDA_VISIBLE_DEVICES=0`, `OMP_NUM_THREADS=1`, `BACKEND=gloo`; the same existing CPU build directory and dependency environment were used for the PR baseline, blanket-guard comparison, and this main check. No packages were installed for this check.
- Full `.release` CTest set ran serially after rebuilding main; default timeout 180 seconds, with individual test properties retained.

```sh
cmake -S . -B build-cpu -DCUDA=OFF -DUCX=OFF -DFULL_TESTS=OFF \
  -DPNETCDF=OFF -DBUILD_TESTS=ON -DBUILD_TESTING=OFF
cmake --build build-cpu -j 8
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 BACKEND=gloo \
  ctest --test-dir build-cpu -j 1 --timeout 180 --output-on-failure -R '\.release$'
```

Measured main `.release` result: **28 passed, 24 failed, 0 skipped, 2 disabled, 54 registered**. Counts are from the completed CTest JUnit output and failure-name log.

## Failing executables: main versus the PR baseline

**The earlier 25-failure count is for `e630d91`, not current main.** Main does not contain or register `test_wb_wall_corner.release`; that test was introduced by PR #265. The complete prior 25-name list is retained below, with current-main results distinguished rather than claiming an absent test failed.

| `.release` executable | Current main | PR baseline `e630d91` |
| --- | --- | --- |
| `test_cloud_parent_slots.release` | FAIL | FAIL |
| `test_condensate_conservation.release` | FAIL | FAIL |
| `test_coordinate.release` | FAIL | FAIL |
| `test_cubed_sphere_exchange.release` | FAIL | FAIL |
| `test_diffusion.release` | FAIL | FAIL |
| `test_diffusion_moist.release` | FAIL | FAIL |
| `test_eos.release` | FAIL | FAIL |
| `test_face_floor.release` | FAIL | FAIL |
| `test_fix_vapor_volume.release` | FAIL | FAIL |
| `test_flux_positivity_carry.release` | FAIL | FAIL |
| `test_forcing.release` | FAIL | FAIL |
| `test_hydro_options.release` | FAIL | FAIL |
| `test_hydro_ref_x1.release` | FAIL | FAIL |
| `test_parentless_cloud.release` | FAIL | FAIL |
| `test_parentless_cloud_nb1.release` | FAIL | FAIL |
| `test_radiating_boundary.release` | FAIL | FAIL |
| `test_reconstruct.release` | FAIL | FAIL |
| `test_rectify.release` | FAIL | FAIL |
| `test_riemann.release` | FAIL | FAIL |
| `test_scalar.release` | FAIL | FAIL |
| `test_sedimentation_guards.release` | FAIL | FAIL |
| `test_two_cards_species.release` | FAIL | FAIL |
| `test_wb_wall_corner.release` | not registered (PR-only test) | FAIL |
| `test_weno.release` | FAIL | FAIL |
| `test_weno5_cuda_line.release` | FAIL | FAIL |

A representative failure excerpt, verbatim:

```text
DispatchStub: missing kernel for cuda
```

## Quantified effect of the blanket guard

The controlled comparison on the same machine kept the environment and CMake caches unchanged between `e630d916da3237b36c7f7db2349fe68fab9599ee` and `49879c54e9e32e9bc4f4edbd5da78498b27ff9a7` (`test/265-wb-wall-corner`). The latter is retained as a reference; its repo-wide guards are not proposed for PR #265.

- All **25** failing `.release` executables on the PR baseline became green under `49879c5`; all 53 enabled C++ executables passed there. The current-main failures above are the corresponding non-wall-corner cases. The blanket patch itself was not applied to main in this check.
- In addition to skipping failing CUDA cases, the shared fixture skips **92 GoogleTest cases that already passed** in a CPU-only Snapy build with a GPU visible. Those cases use available Torch/device operations without necessarily requiring Snapy CUDA kernels.
- **102** completed failing GoogleTest cases moved to skipped. The baseline parentless-cloud subprocess also aborted before reaching later existing cases; four of those become skipped and two CPU cases run successfully under the blanket guard.
- `test_rectify.release` is a separate behavior change: it moves **fail → pass by selecting CPU**, not by reporting a skip.
- Thus there were no newly failing tests, but the blanket patch did not satisfy a strict allowance of only fail → skip plus the new passing wall-corner case.

All 92 were also confirmed passing in the current-main CPU-only rerun. They are grouped below; float and double parameterizations count separately.

| Binary | Previously passing cases changed to skipped |
| --- | ---: |
| `test_coordinate.release` | 20 |
| `test_diffusion.release` | 22 |
| `test_diffusion_moist.release` | 4 |
| `test_diffusion_x1_scale.release` | 14 |
| `test_eos.release` | 4 |
| `test_internal_boundary.release` | 2 |
| `test_plm.release` | 10 |
| `test_radiating_boundary.release` | 2 |
| `test_reconstruct.release` | 4 |
| `test_refine.release` | 2 |
| `test_two_cards_species.release` | 4 |
| `test_weno.release` | 4 |
| **Total** | **92** |

## Distinct Python-package failures

The complete CTest suites in the earlier controlled comparison also ran Python tests that imported **installed Snapy 2.10.8 and its installed extension**, rather than either rebuilt checkout. Kintera remained 2.5.13 and `SNAPY_TEST_PYTHONPATH` was unset. The `.release`-only main rerun above does not include these Python tests.

| Commit / build | Passed | Failed | Skipped | Disabled | Registered |
| --- | ---: | ---: | ---: | ---: | ---: |
| `e630d91` CPU | 43 | 47 | 1 | 3 | 94 |
| `e630d91` CUDA | 68 | 22 | 1 | 3 | 94 |
| `49879c5` CPU | 68 | 16 | 7 | 3 | 94 |
| `49879c5` CUDA | 68 | 22 | 1 | 3 | 94 |

Both CUDA runs had the same **22 Python failures**. The blanket CPU configuration skips six CUDA-labelled Python tests, leaving **16** failures. These are distinct from missing native CUDA kernels in the CPU-only C++ executables. Representative Python output, verbatim:

```text
RuntimeError: zeros: Dimension size must be non-negative.
KeyError: 'scalar.positivity_hits'
```

Prepending the CMake build directory and source Python directory to `PYTHONPATH` still selected the installed package: these build directories have no importable checkout-built Python extension. A matching Python package installation or a path to an already-built package is a separate validation concern; no package/environment repair was made during these measurements.

## Why 94 differs from the earlier 98/99 totals

The PR baseline and blanket-guard builds register **94** CTests with `FULL_TESTS=OFF` and `UCX=OFF`; **3** are disabled, leaving 91 enabled. `FULL_TESTS=ON` adds five (`test_mesh_multi_block.release`, `test_exchange_decomp`, `test_shallow_xy_decomp`, `test_shallow_splash_decomp`, `test_shallow_splash_ucx_cuda_decomp`). UCX adds two (`test_parentless_cloud_nb1_mp_gloo`, `test_exchange_ucx`), and CUDA with UCX adds `test_exchange_ucx_cuda`.

- CPU enabled count: **94 − 3 + 5 + 2 = 98**.
- CUDA enabled count: **94 − 3 + 5 + 2 + 1 = 99**.
- Current main registers **93** tests under the measured reduced configuration, one fewer than the PR baseline because its wall-corner test is absent.

This arithmetic explains the broader enabled-test denominators from the registration rules; the original 98/99 logs were unavailable, so their historical configuration and Python package selection have not been independently confirmed. Registered, disabled, skipped, and passed counts should be reported separately.

## Open policy question: blanket or targeted skipping?

- **Blanket build-capability guard:** gives a uniform meaning to `CUDA=OFF`, prevents accidental entry into unbuilt kernels, and is simple to maintain. It also suppresses the 92 currently passing Torch/device cases and broadens the change beyond the tests that fail.
- **Targeted guards:** skip only tests or paths that require Snapy CUDA kernels, retaining working Torch/device coverage in CPU-only builds. They require identifying those dependencies and maintaining that mapping as test bodies change.
- **Automatic device-selection programs:** `test_rectify.release` raises a separate reporting choice between selecting CPU and explicitly skipping a CUDA-specific check.

Which policy should the suite use, and should Torch-only CUDA coverage remain active when Snapy itself is built without CUDA? This draft records the tradeoffs and leaves the decision open.
