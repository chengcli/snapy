# Gate 0: requested four-advection-arm raw dump

Baseline Snapy: `5eeb9b6761ae484a98b3993aae18314fc58cf862`. Kintera: `4dc613d04f24621b3119d343c5c7c9b93628895b`, exact tag **v2.5.15**, imported version **2.5.15**. Study plan: `9b4a61f8ebdde95b5f2dce10388de7633fd55f9c` (unchanged).

## Definition and interpretation

Line numbers below refer to PLAN.md at 9b4a61f8.

- L393 defines Gate 0: “build current main 5eeb9b6 ... against kintera v2.5.15 (4dc613d ...). Record the six moist-mixture counts, M_n and T.” Pass requires all six moist-mixture cases red with energy residual relative 1.0 at every limited face, and ideal-moist arms green. Count differences are recorded, not an abort. L82 preserves the same baseline/version requirement.
- L433-440 identify the carry target and case grid, including advection LMARS/HLLC. The four requested advection arms are interpreted as **ideal-moist/moist-mixture × LMARS/HLLC**. Each arm compares limiter false/true with exactly the unmodified baseline helper and base YAML. The CUDA-device versions are added only by the external diagnostic wrapper.
- L179-185 identify the base-card naming and one-forward fixture. Use `tests/test_flux_positivity_carry.cpp` and `tests/test_flux_positivity_carry.yaml` at the baseline: N=6, two ghosts, dt=1, one forward, initial rho=1, p=1e5, velocity=(2,3,0), vapor/cloud fractions=(0.01,0.02); no proposed second u0 card is used in this Gate 0 run.
- **F_off/F_on are limiter-off/on, NOT heat-capacity-off/on.** PLAN L166 requires `use_nasa9_cp=false` and `use_h2_cp=false` in every run, failing a cell if either is on. L159 explains that the existing limiter bool also controls cell repair; its independent debug knob is only proposed. For this untouched baseline measurement we retain `forward_once(false/true)` from source lines 37-43,169-177. We do not implement that future knob or enable cp overrides. No cp-on heat-capacity comparison is claimed.
- M_n means species molar mass in kg/mol, dry/vapor/cloud, queried from the EOS `species_weight(n)` (ideal_moist.cpp:54-56; moist_mixture.cpp:33-35). PLAN L337-342 defines O3's YAML masses and nominal 28.97e-3/18.015e-3; L376 gives relative mass tolerance 2e-5. The measured values match those nominal values at printed precision. Composition-table verification beyond that nominal check is not claimed.
- PLAN L363-375 lists the tolerances. The unchanged `expect_carried` uses the stricter energy tolerance `1e-12*max(abs(F_off[IPR]),abs(expected_dE))`, momentum analogously, exact dry flux equality, and column tolerance `1e-12 + 1e-12*sum(abs(du_off))` (test source:134-160). Its ideal-moist assertions pass. For moist-mixture the probe requires each limited-face carry residual to exceed that energy tolerance and its ratio to abs(expected_dE) to be within 1e-12 of 1; all measured ratios are exactly 1. The probe additionally checks column totals for every dumped conserved row. This baseline carry oracle uses W->E, as PLAN L167 acknowledges; this is not an independent O3 validation.
- PLAN gives no Gate 0 output directory; use the requested fallback `study/236/gate0/`.

**Scope:** all eight requested arm/device runs completed. Their baseline signatures pass below. At the original advection publication this was **not a claim that full six-case Gate 0 was complete** (the CPU follow-up below now supplies the missing cases): settling, x2, donors and mixed-flux cases have not been newly measured here. No A/G implementation, full CTest, O3/mutation gate or sign-off is claimed.

## Per-arm/device results

M_n columns are dry / vapor / cloud in kg/mol. “Expected carry red” is a successful baseline-gap observation, not a repaired carry test. Every run's two cp flags were off in BOTH F_off and F_on.

| Arm | Device/build | M_n | cp flags | Limited faces | Max relative carry-energy residual | Gate 0 advection verdict | Tolerance used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ideal-moist/lmars | cpu | 0.02897 / 0.018015 / 0.018015 | off/off, both F arms | 5 | 2.9414032233720378e-16 | PASS (carry green) | E/M: 1e-12 scaled; masses: 2e-5 relative |
| ideal-moist/hllc | cpu | 0.02897 / 0.018015 / 0.018015 | off/off, both F arms | 5 | 0 | PASS (carry green) | E/M: 1e-12 scaled; masses: 2e-5 relative |
| moist-mixture/lmars | cpu | 0.02897 / 0.018015 / 0.018015 | off/off, both F arms | 5 | 1 | PASS (expected carry red) | E/M: 1e-12 scaled; masses: 2e-5 relative |
| moist-mixture/hllc | cpu | 0.02897 / 0.018015 / 0.018015 | off/off, both F arms | 5 | 1 | PASS (expected carry red) | E/M: 1e-12 scaled; masses: 2e-5 relative |
| ideal-moist/lmars | cuda | 0.02897 / 0.018015 / 0.018015 | off/off, both F arms | 5 | 2.9414032233720378e-16 | PASS (carry green) | E/M: 1e-12 scaled; masses: 2e-5 relative |
| ideal-moist/hllc | cuda | 0.02897 / 0.018015 / 0.018015 | off/off, both F arms | 5 | 0 | PASS (carry green) | E/M: 1e-12 scaled; masses: 2e-5 relative |
| moist-mixture/lmars | cuda | 0.02897 / 0.018015 / 0.018015 | off/off, both F arms | 5 | 1 | PASS (expected carry red) | E/M: 1e-12 scaled; masses: 2e-5 relative |
| moist-mixture/hllc | cuda | 0.02897 / 0.018015 / 0.018015 | off/off, both F arms | 5 | 1 | PASS (expected carry red) | E/M: 1e-12 scaled; masses: 2e-5 relative |

Every arm has 5 limited faces (3..7), or 10 limited species-face entries. Each dump covers all **7 faces il..iu+1 = 2..8**, including both boundaries, with all **6 requested rows** and both fluxes printed by `fprintf(...,"%.17g",...)`. Total: 8 files × 42 data rows = **336 data rows / 672 flux numbers**, plus metadata and residual/budget comments. `IVZ` is zero in the fixture and is checked by the unchanged ideal-moist helper, but is not an extra requested dump row.

| Arm/device | T (K) | Max absolute carry-energy residual |
| --- | --- | --- |
| ideal-moist/lmars/cpu | 353.34734862459703 | 7.2759576141834259e-12 |
| ideal-moist/hllc/cpu | 353.34734862459703 | 0 |
| moist-mixture/lmars/cpu | 353.34734862459709 | 24736.348815998896 |
| moist-mixture/hllc/cpu | 353.34734862459709 | 24736.348815998812 |
| ideal-moist/lmars/cuda | 353.34734862459703 | 7.2759576141834259e-12 |
| ideal-moist/hllc/cuda | 353.34734862459703 | 0 |
| moist-mixture/lmars/cuda | 353.34734862459709 | 24736.348815998896 |
| moist-mixture/hllc/cuda | 353.34734862459709 | 24736.348815998812 |

CPU/CUDA raw-flux comparisons use PLAN L371-375 absolute/relative tolerances (energy 1e-9/1e-13, momentum 1e-12/1e-13, species 1e-15/1e-13; same species tolerance applied to dry).

| Arm | Max absolute CPU/CUDA flux difference | Within tolerance |
| --- | --- | --- |
| ideal-moist/hllc | 1.4551915228366852e-10 | True |
| ideal-moist/lmars | 1.1641532182693481e-10 | True |
| moist-mixture/hllc | 0 | True |
| moist-mixture/lmars | 0 | True |

## Build configuration, commands and problems

Existing Python environment `/home/chengcli/pyenv` (Python 3.11.13, Torch 2.10.0+cu128); GCC 11.5.0, CMake 3.31.8; CUDA toolkit 12.9.86. GPU 0 is an RTX 5090; both GPUs were visible, including during CPU runs. Runtime used `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1`.

Build invocation was `bash configure.sh cpu` and `bash configure.sh cuda` using the exact script saved in BUILD-CONFIG.txt. The source checkout was created with `git worktree add --detach /data00/ai_workspace/worker2/gate0-w5a149772/source 5eeb9b6761ae484a98b3993aae18314fc58cf862`.

Both builds are Release, tests enabled, examples/FULL_TESTS/UCX/PNETCDF disabled, NETCDF enabled; CPU `CUDA=OFF`, CUDA `CUDA=ON`. Requested architecture 120; repository-generated flags expand to 60,61,70,75,80,86,89,90,120. ABI=1. `BUILD-CONFIG.txt` preserves exact configure/build commands and effective CUDA flags. Both builds completed `test_flux_positivity_carry.release`, including the full required Snapy libraries, in separate new build directories.

Dependency commands:

```sh
source /home/chengcli/pyenv/bin/activate
python -m pip download --no-deps kintera==2.5.15 -d gate0-w5a149772
# Failed: configured index has no 2.5.15. Built exact tagged source instead.
git clone --no-checkout https://github.com/chengcli/kintera.git gate0-w5a149772/kintera
git -C gate0-w5a149772/kintera checkout --detach 4dc613d04f24621b3119d343c5c7c9b93628895b
cmake -S gate0-w5a149772/kintera -B gate0-w5a149772/kintera/build -DCUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build gate0-w5a149772/kintera/build -j 6
# From the isolated Kintera checkout:
MAX_JOBS=4 CUDA_HOME=/usr/local/cuda-12.9 python -m pip wheel --no-build-isolation --no-deps . -w ../wheels
python -m pip install --no-deps --target ../packages ../wheels/kintera-2.5.15-cp311-cp311-linux_x86_64.whl
```

The wheel was built while the CUDA library was still compiling; its CPU libraries/headers/bindings are from the exact tagged source. After CUDA linking finished, the same build's `libkintera_cuda_release.so` was copied into the isolated package's lib directory. Both Snapy configurations resolve that package via per-run PYTHONPATH. Library hashes are recorded. No shared site-packages or other workers' environments/builds were modified.

Diagnostic procedure: an external C++ translation unit includes the untouched baseline carry test source, adds `TEST(Gate0,dump)`, calls `forward_once` for the EOS/solver/device selection, and emits the requested rows. It reuses `expect_carried` for ideal-moist and checks the expected missing carry for moist-mixture. It is compiled with each build's exact carry-test compile command and linked with that build's carry-test libraries. Compile/link commands are saved in probe build/link logs. No Snapy source file was edited; no instrumentation is committed to main or the study source tree. The one-off probe and runner remain in the worker's isolated workspace.

The first CPU probe link happened before libsnap_release.so was ready and failed with `cannot find ../lib/libsnap_release.so`; a link-only retry after the baseline build completed passed. Both baseline builds passed without source fixes. Committed build logs have ANSI color escapes and trailing whitespace removed; diagnostic values and failure messages are preserved. CUDA emitted pre-sm_75 deprecation and unsigned-comparison warnings, retained in logs. All eight final probe runs exited 0; no run was skipped.

Run command for each of four EOS/solver pairs, once per build/device (fresh process, cwd build-DEVICE/tests):

```sh
GATE_EOS=ideal-moist GATE_RIEMANN=lmars GATE_DEVICE=cpu \
GATE_OUTPUT=<absolute dump path> CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
../gate0-probe.release --gtest_filter=Gate0.dump
# Repeat EOS={ideal-moist,moist-mixture}, solver={lmars,hllc}, device={cpu,cuda}.
```

Probe SHA-256: `ca685c07ff44e3c33531336730548fb89b0bbcdaa62fab81031b6fbf9d1f310f`. Elapsed wall time from isolated workspace creation to result assembly: **1112.3 seconds (18.54 minutes)**, excluding initial plan/environment inspection. Eight numerical subprocesses together: **66.130 seconds**.

## Dump and log files

- [ideal-moist-lmars-cpu.csv](ideal-moist-lmars-cpu.csv) — [ideal-moist-lmars-cpu.log](ideal-moist-lmars-cpu.log)
- [ideal-moist-hllc-cpu.csv](ideal-moist-hllc-cpu.csv) — [ideal-moist-hllc-cpu.log](ideal-moist-hllc-cpu.log)
- [moist-mixture-lmars-cpu.csv](moist-mixture-lmars-cpu.csv) — [moist-mixture-lmars-cpu.log](moist-mixture-lmars-cpu.log)
- [moist-mixture-hllc-cpu.csv](moist-mixture-hllc-cpu.csv) — [moist-mixture-hllc-cpu.log](moist-mixture-hllc-cpu.log)
- [ideal-moist-lmars-cuda.csv](ideal-moist-lmars-cuda.csv) — [ideal-moist-lmars-cuda.log](ideal-moist-lmars-cuda.log)
- [ideal-moist-hllc-cuda.csv](ideal-moist-hllc-cuda.csv) — [ideal-moist-hllc-cuda.log](ideal-moist-hllc-cuda.log)
- [moist-mixture-lmars-cuda.csv](moist-mixture-lmars-cuda.csv) — [moist-mixture-lmars-cuda.log](moist-mixture-lmars-cuda.log)
- [moist-mixture-hllc-cuda.csv](moist-mixture-hllc-cuda.csv) — [moist-mixture-hllc-cuda.log](moist-mixture-hllc-cuda.log)

Dependency/build/probe logs, `BUILD-CONFIG.txt`, `environment.log`, `version.log` and `kintera-libraries.sha256` preserve setup and failure evidence. PLAN.md is byte-identical to 9b4a61f8, including all protected blocks.

## CPU follow-up: remaining four Gate 0 cases

Baseline/version and tolerances are unchanged: Snapy 5eeb9b6, Kintera v2.5.15/4dc613d, the existing CPU-only Release build. PLAN.md remains byte-identical to 6f9e0adc/9b4a61f8 in this data commit. F_off/F_on again mean **limiter-off/limiter-on**, not cp-off/on; both heat-capacity flags are false for both arms.

All eight new runs match the expected result: four ideal-moist tests green (exit 0), four moist-mixture tests red (exit 1), with relative carry-energy residual exactly 1 at EVERY limited face in moist-mixture. These are the original named GoogleTest bodies; the external diagnostic copy overrides only `dynamics.equation-of-state.type` after each test's YAML edits and attaches a dump listener. No Snapy checkout source was edited. In conjunction with the earlier LMARS/HLLC advection runs, all six CPU cases now have the expected baseline signature. The four non-advection CUDA cases were not requested and remain unmeasured.

| Case | Config | Pass/fail vs expected | Limited faces | M_n (dry/vapor/cloud kg/mol) | cp flags (NASA9/H2) | Device |
| --- | --- | --- | --- | --- | --- | --- |
| settling | ideal-moist | PASS (green) | 5 test-slice / 5 whole grid | 0.02897 / 0.018015 / 0.018015 | off/off in both arms | CPU |
| settling | moist-mixture | PASS (expected red; exit 1) | 5 test-slice / 5 whole grid | 0.02897 / 0.018015 / 0.018015 | off/off in both arms | CPU |
| x2 | ideal-moist | PASS (green) | 5 test-slice / 30 whole grid | 0.02897 / 0.018015 / 0.018015 | off/off in both arms | CPU |
| x2 | moist-mixture | PASS (expected red; exit 1) | 5 test-slice / 30 whole grid | 0.02897 / 0.018015 / 0.018015 | off/off in both arms | CPU |
| donors | ideal-moist | PASS (green) | 4 test-slice / 4 whole grid | 0.02897 / 0.018015 / 0.018015 | off/off in both arms | CPU |
| donors | moist-mixture | PASS (expected red; exit 1) | 4 test-slice / 4 whole grid | 0.02897 / 0.018015 / 0.018015 | off/off in both arms | CPU |
| mixed | ideal-moist | PASS (green) | 5 test-slice / 5 whole grid | 0.02897 / 0.018015 / 0.018015 | off/off in both arms | CPU |
| mixed | moist-mixture | PASS (expected red; exit 1) | 5 test-slice / 5 whole grid | 0.02897 / 0.018015 / 0.018015 | off/off in both arms | CPU |

Each on/off (and mixed-case bare) arm is a separate single hydro forward: dt=1, step count=1, final time=1 from initial time=0. It is not a multistep evolution. Each process starts with `test_flux_positivity_carry.yaml` at main 5eeb9b6. Runtime `GATE_EOS=ideal-moist` or `moist-mixture` sets `dynamics.equation-of-state.type` on every helper-loaded card; the test's other edits are retained. LMARS is the base-card solver throughout these four cases. The existing B2 debug knob is not implemented; this is the unchanged baseline limiter boolean comparison.

- Settling: original test lines 184-191; cloud const-vsed=-2, velocity=(0,3,0), 6x1x1.
- Along x2: lines 197-209; 6x6x1, reflecting x2 boundaries, velocity=(0,2,3).
- Donors: lines 219-239; original six-cell density and three-velocity arrays, no sedimentation; the original assertions require two distinguishable upward and two downward limited faces.
- Mixed: lines 253-336; original six-cell nonuniform arrays, cloud const-vsed=-1; the additional unlimited bare arm has no sedimentation and determines the settling increment for the original energy/momentum oracle. One forward per bare/off/on arm.

Dumps retain the requested row names and %.17g F_off/F_on values; extra axis/i/j/k columns identify geometry. All x1 faces il..iu+1 are dumped over every interior transverse row. For along-x2 **both x1 and x2 faces** are dumped: 42 x1 faces plus 42 x2 faces (each with six conserved rows), including jl..ju+1 for every interior x1 column. Its limited-face count is 5 on the original test's sampled line and 30 over the six-column slab; x1 has zero limited faces. Other cases have seven x1 faces. New data totals: 2*504 + 6*42 = **1260 rows / 2520 flux values**. No raw numeric dump was postprocessed.

Compile/link reused the existing CPU target's compile and link commands with the external diagnostic translation unit; no library rebuild or environment change was needed. Compilation passed. The process command was:

```sh
GATE_EOS=<ideal-moist|moist-mixture> GATE_OUTPUT=<absolute CSV> OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  gate0-followup.release --gtest_filter=flux_positivity.<test-name>
```

The four exact test-name suffixes and their raw dump/log pairs are:

- `withheld_settling_mass_keeps_its_energy_and_momentum`, ideal-moist: [settling-ideal-moist-cpu.csv](settling-ideal-moist-cpu.csv), [settling-ideal-moist-cpu.log](settling-ideal-moist-cpu.log).
- `withheld_settling_mass_keeps_its_energy_and_momentum`, moist-mixture: [settling-moist-mixture-cpu.csv](settling-moist-mixture-cpu.csv), [settling-moist-mixture-cpu.log](settling-moist-mixture-cpu.log).
- `withheld_mass_keeps_its_energy_and_momentum_along_x2`, ideal-moist: [x2-ideal-moist-cpu.csv](x2-ideal-moist-cpu.csv), [x2-ideal-moist-cpu.log](x2-ideal-moist-cpu.log).
- `withheld_mass_keeps_its_energy_and_momentum_along_x2`, moist-mixture: [x2-moist-mixture-cpu.csv](x2-moist-mixture-cpu.csv), [x2-moist-mixture-cpu.log](x2-moist-mixture-cpu.log).
- `withheld_mass_keeps_its_donors_energy_and_momentum`, ideal-moist: [donors-ideal-moist-cpu.csv](donors-ideal-moist-cpu.csv), [donors-ideal-moist-cpu.log](donors-ideal-moist-cpu.log).
- `withheld_mass_keeps_its_donors_energy_and_momentum`, moist-mixture: [donors-moist-mixture-cpu.csv](donors-moist-mixture-cpu.csv), [donors-moist-mixture-cpu.log](donors-moist-mixture-cpu.log).
- `withheld_mixed_flux_keeps_each_parts_energy_and_momentum`, ideal-moist: [mixed-ideal-moist-cpu.csv](mixed-ideal-moist-cpu.csv), [mixed-ideal-moist-cpu.log](mixed-ideal-moist-cpu.log).
- `withheld_mixed_flux_keeps_each_parts_energy_and_momentum`, moist-mixture: [mixed-moist-mixture-cpu.csv](mixed-moist-mixture-cpu.csv), [mixed-moist-mixture-cpu.log](mixed-moist-mixture-cpu.log).

The moist-mixture failure logs are intentionally retained verbatim apart from trailing whitespace. They demonstrate missing carry, not a run crash. The diagnostic copy and compile helper remain outside the repository.
