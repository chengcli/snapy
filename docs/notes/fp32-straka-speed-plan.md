# Straka GPU throughput target: 2× FP64

## Target and measurement contract

Target optimized FP32 throughput at least **2× the FP64 baseline**, on the same
RTX 5090 and the existing 64×256, 9000-step, dt=0.1 Straka experiment. The measured
FP64 median is 63.4482 s: the initial acceptance target is **≤31.7241 s**. This is
also a 2.21× improvement over current installed FP32 (70.2006 s). If the FP64
implementation changes, repeat its measurement and retain both comparisons.
This is a goal, not a predicted result from the HLLC fix.

Keep HLLC, WENO5, RK3, resolution, forcing, scalar transport, timestep schedule,
and validation/output schedule identical. Do not meet the target by lowering
accuracy, skipping physical work, hiding rejected steps or excluding validation
from only one side. Run three or more sequential alternating pairs on one GPU;
report median, range, variation, hashes, end-to-end time and separately kernel
intervals. No CPU timing contributes to this target. Preserve numerical gates
and report fields/budgets at 300/600/900 s. No regridding work.

## H1: true FP32 HLLC arithmetic and safe acoustic differences

Replace implicitly double literals with T-typed constants; assert representative
intermediate types at compile time. For float, form acoustic wave/flow differences
directly, honoring the clipped-wave branches separately. Retain FP64 arithmetic
order and shifted-pressure semantics. Remove the unused middle velocity estimate.

Validate shocks, rarefactions, both boost signs, all directions, species flux and
large pressure backgrounds against double precision. Exercise ordinary states
and extreme boosted states where float subtraction erases an acoustic speed.
Run existing Riemann, hydro, restart and implicit suites plus Straka. On an
updated GPU binary inspect SASS: avoid unintended double arithmetic in the float
kernel, then time the kernel and full case. This stage cannot deliver 2× alone:
HLLC is only ~1.4% of currently summed FP32 GPU kernel duration.

## H2: fuse high-frequency work around existing kernels

Use existing traces to rank total per-step cost and launch counts. Target:

1. Fuse face pressure gauge operations and dry face EOS preparation with the
   HLLC wrapper/kernel; keep sensitive pressure differences in FP64 locally.
2. Fuse flux divergence, gravity/source increments and RK update where dependency
   order permits. Preserve shared-face conservation and source rounding contracts.
3. Cache immutable geometry/background coefficients with explicit invalidation;
   eliminate redundant whole-state conversions and temporary face arrays.

Keep a tested fallback for unsupported geometry, covariance or moist cases.
Measure each change independently. A working launch-count goal is ≤500 kernels
per step versus the current ~1953, but only measured runtime establishes success.
Retained scratch can increase resident memory; track CUDA peak allocation rather
than claiming memory savings from registered buffers alone.

## H3: reduce host dispatch and synchronization

After buffers and shapes stabilize, assess CUDA graph capture for safe regional
or full-stage execution. Inventory host scalar reads, retry decisions, CFL checks
and dynamic allocations first. Keep failure detection and retry semantics intact;
do not bypass them to enable capture. Compare graph and eager paths numerically.
If capture is unsuitable, batch independent operations and fuse reductions or
validation to reduce CPU dispatch. End-to-end target remains 3.525 ms/step; host
and device budgets must be measured as overlapping timelines, not simply added.

## H4: prove scope, accuracy and throughput

Re-run the exact fixed-grid Straka benchmark with the optimized GPU binary.
Require the 2× target across repeated trials with numerical acceptance; record
any regression, not just favorable runs. Sweep larger grids separately to locate
launch-limited versus device-work-limited regimes. A larger-grid speedup does not
satisfy the fixed-grid target. Recheck moist/Jupiter, restart and multi-block
regressions before enabling changes beyond the validated dry Cartesian path.

An optional FP64 evolving-anomaly/RK accumulation plus FP32 compute experiment
addresses accuracy and the Kratos precision-policy distinction. It is separate
from the throughput target and must not silently change existing storage/restart
contracts. No kintera API changes are currently required.

## Execution status

H1 implementation and local validation are complete. GPU validation is pending.
H2–H4 are planned.
The current installed dungeon3 binary predates all optimizations. An isolated
GPU rebuild is still pending the earlier no-recompilation clarification; existing
GPU measurements must not be presented as optimized results.

### H1 local validation completed

The shared CPU/CUDA HLLC implementation is fixed. Compile-time assertions prevent
representative float intermediates from silently becoming double. The new test
covers 120 combinations of direction, shock/expansion/equal state, signed boost
and reference pressure, including ±1e8 boosts and a transported species.

Commands and results:

```sh
PYTHONPATH=/tmp/snapy-straka-opt-python cmake --build build --parallel 4 --target snap_release test_riemann.release
ctest --test-dir build --output-on-failure -L 'fp32-(foundations|science|acceptance)'
PYTHONPATH=/tmp/snapy-straka-opt-python python -u tests/run_fp32_straka.py \
  --precision float32 --device cpu --output /tmp/straka-hllc-fix-900
```

Build passed; **40/40 CTests passed** (158.18 s). Straka completed **9000 steps**,
with no rejected/nonfinite steps and relative mass drift **1.1391e-9**, below the
existing reduced-case 2e-5 engineering mass limit. At 900 s, theta differs from
the old CPU FP32 run by **0.0005673 K RMS / 0.01535 K maximum**; changing arithmetic
precision is not expected to preserve bitwise trajectories. These differences
are observations, not a new scientific-equivalence claim. [Correctness data and
binary hashes](fp32-m6-data/straka-hllc-fix.json) are retained.

Google clang-format and `git diff --check` passed. No kintera changes. GPU SASS
verification and throughput for the new binary remain unmeasured; CPU runtime
is not used to infer GPU gains. H2/H3 remain planned, not implemented.
