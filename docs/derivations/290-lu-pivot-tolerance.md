# LU pivot rejection and VIC retry (#290)

`ludcmp` returns the original success parity (+1 or -1), or zero on failure.
For each input row retain its original magnitude

\[
s_i = \max_k |a_{ik}|,\qquad v_i = 1/s_i.
\]

The existing scaled partial-pivot selection uses `v_i * abs(pivot)`. Carry
`s_i` with row permutations, and reject the selected pivot, including the last
pivot, when

\[
|p_j|/s_j \le \tau_N,\qquad \tau_N = 8N\epsilon_T.
\]

Here `epsilon_T` is `numeric_limits<T>::epsilon()`. A dot product and its
subtraction use at most about `2N` elementary rounded operations. With IEEE
unit roundoff `u = epsilon_T/2`, their usual accumulation factor is
`gamma_(2N) = 2Nu/(1-2Nu)`, approximately `N*epsilon_T`. The factor eight gives
a conservative guard band around that rounding scale for these small systems.
It rejects cancellation-sized pivots before division. Multiplying a row by a
finite positive factor scales its pivot and original row magnitude together,
so the ratio is dimensionless; an absolute tolerance would depend on the
units of a VIC row.

This is a pivot safety guard, not a condition-number estimate or a certified
rank/forward-error bound. Large elimination growth can exceed this simple
rounding scale; acceptance does not guarantee accurate evolution. In
particular it does not solve #286's coarse-column accuracy/fixer problem.

| dtype | N=3 | N=5 |
|---|---:|---:|
| float | 2.86102294921875e-6 | 4.76837158203125e-6 |
| double | 5.329070518200751e-15 | 8.881784197001252e-15 |

Every input coefficient, row-scale reciprocal, elimination/pivot-selection
result, and returned factor must be finite. The sweeps also reject nonfinite
solved coefficients and right-hand-side solutions. Both 3x3 inverse paths
factor a copy to check pivots, retaining their original inverse arithmetic on
success; the 5x5 paths retain their original arithmetic as well.

The current CPU and CUDA dispatches share the same sweep and failed-column
sentinel. A failed sweep skips backward substitution. The host module checks
the solution before redistribution, and checks its inputs and final results.
On failure it restores the correction inputs, returns a zero correction, and
latches the failure through subsequent RK stages. It reports rank, column,
step, stage, and retry. The sixth globally reduced `check_redo` cause restores
the saved step input, including scalar state, on retry and on terminal VIC
failure. Stage zero resets the latch for the next attempt. The gravity fixer
also defers a rejected step to redo instead of applying its thermal correction.

The shared code expresses the intended CPU/CUDA policy. Independent device
RED/GREEN and the full CUDA CTest gate remain Zoey's agent's responsibility;
CPU evidence alone does not establish their outcome.

## Float32 VIC conditioning limitation

VIC permits float32 through `AT_DISPATCH_FLOATING_TYPES` in the CPU and
CUDA dispatches. The guard is substantially stricter in float32: poorly
conditioned columns around condition number 1e6 and above can be refused.
Xi supplied the independent review's random 3x3 conditioning probe: **166/5000
rejections at condition number 1e6** and **4915/5000 at 1e8**, while its double
probe found no rejections in 5000 matrices per level from 1e4 through 1e14.
These are attributed independent measurements, not measurements by this
implementer, and not a universal condition-number cutoff. Row-scaled pivot
ratios depend on matrix structure and scaling as well as condition number.
A rejected float32 VIC column enters the same redo/stop policy; reducing the
time step is not guaranteed to cure conditioning due to variable scales.
Use float64 when float32 conditioning prevents progress.
