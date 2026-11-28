Execution: eager and CUDA graphs
================================

On a CUDA device ``snapy`` runs each stage as captured CUDA graphs by default,
and eagerly wherever a case cannot be captured; on CPU and MPS it runs
eagerly. Eager execution is the reference: a graph replay gives the same bits.

Selecting the execution
-----------------------

The deck selects it under ``integration``::

  integration:
    execution: auto           # auto | eager | cuda-graph
    execution-strict: false   # true: refuse a case that cannot be captured

The two key names are provisional until the V3 naming is confirmed.

- ``auto`` (default, also when the key is absent): ``cuda-graph`` when the
  block is on a CUDA device, ``eager`` otherwise. With
  ``execution-strict: true`` it means ``cuda-graph``.
- ``eager``: every stage runs eagerly, on any device. ``execution-strict:
  true`` with ``eager`` is refused when the options are read.
- ``cuda-graph``: a case that can be captured runs each stage as a graph; any
  other case runs eagerly, and each reason is logged once per block (a
  warning). With ``execution-strict: true`` such a case is refused before its
  first launch, and the error lists the reasons.
- A ``Mesh`` runs its blocks eagerly; ``execution-strict: true`` is refused
  there (graphs for several blocks come with a later milestone).

Capability matrix
-----------------

Numerical support is the eager columns; graph support is separate.

.. csv-table::
   :header: "Case", "eager FP64", "eager FP32", "graph FP64", "graph FP32"

   "Cartesian dry, 1 block", "yes", "yes", "yes (below)", "yes (below)"
   "Cartesian, N blocks, UCX ranks", "yes", "yes", "no (G4)", "no (G4)"
   "kintera moist", "yes", "partly", "no (G2)", "no (G2)"
   "vertical implicit correction", "yes", "Cartesian ideal-moist", "yes (below)", "yes (below)"
   "cubed sphere", "yes", "no (G5)", "no (G6)", "no (G6)"
   "user TorchScript forcing", "yes", "yes", "no", "no"
   "radiation (pyharp/pydisort)", "not on the step path", "no", "no", "no"

CPU and MPS run eager only. The graph rows that say yes cover the dry Straka
case: a CUDA device, Cartesian geometry, one species, no reactions, the EOS
limiter off, an ideal-moist EOS, the hllc Riemann solver, rk3, one block,
no solid mask and constant gravity. Its vertical implicit solve reads the
block's device timestep.

CUDA graphs support
``const-gravity``, ``coriolis``, ``diffusion``, ``body-heat``, ``bot-heat``,
``top-cool``, ``relax-bot-comp``, ``relax-bot-temp``, ``relax-bot-velo``,
``top-sponge-lyr`` and ``bot-sponge-lyr``, individually or together.
Capture eligibility follows each registered source's ``capturable``
declaration in both source phases. A noncapturable built-in (including
``plume-forcing``) makes the entire block eager; in a Mesh this makes every
block eager. Strict execution refuses the configuration before launch.
This is a registry-level guarantee for plume: the current EOS factory does
not implement ``plume-eos``, so an end-to-end plume deck cannot be constructed.
A ``plume-forcing`` entry under another EOS does not register a plume source.
User TorchScript forcings run in an eager segment between graphs when strict
execution is disabled, including when built-ins and radial diffusion are present.

The implicit radial half-steps and their flux-register posts read the device
timestep. Changing a finite, nonnegative timestep, including between RK
stages or returning to a previous value, does not invalidate their graphs.
A zero radial timestep is a no-op; negative or nonfinite values are refused.
Previously, radial splitting keyed both advance and combine on the host dt,
so every stage's compute segments recaptured on every dt change, including
the middle stage, which performs neither radial half-step.

``MeshBlock.execution_capability(vars)`` returns a block's row for its own
configuration and variables: ``eager_fp64``/``eager_fp32`` (its precision),
``graph_fp64``/``graph_fp32`` (whether it can be captured) and
``eager_reasons`` (why its stages run eagerly).

What a graph depends on
-----------------------

A stage's graph is captured again when any of these changes: the device, the
shapes, strides or dtypes of the state, the state store's generation, the
configuration generation or the stage. Ordinary forcing kernels read ``dt``
from a device scalar at replay. A segment that reads the host timestep
also includes ``dt`` in its capture key. The store's generation changes
when its buffers move: ``to()``, a restart, or a variable that the caller
replaced between steps (it is copied into the block's fixed storage).
``bump_config_generation()`` recaptures after a change the graph cannot see,
such as ``implicit-fallback-fp64`` or a live forcing scalar (``type: 123``
Coriolis ``omega*``, heating ``flux``/``dTdt``, relaxation ``tau``, diffusion
``nu_iso``/``kappa_iso``
and wall temperatures). These host options are baked into captures; after
mutating them, call ``block.bump_config_generation()`` before advancing.
This does not rebuild module caches: for example, ``type: xyz`` Coriolis
precomputes rotation fields during construction; construct a new block to
change those options.
The generation belongs to the executor: for a Mesh it invalidates all blocks'
compute segments, which recapture as their stages next execute. Even a
block-local invalidation aligns recapture of the corresponding stage/segment
across all Mesh blocks so their panel exchanges rendezvous correctly.
Diffusion profiles are immutable after construction and remain refused if
changed, even with a generation bump; construct a new block to change them.
A restart never serializes graphs.

Capture warms each stage up eagerly first, with the step's ``dt``, and then
restores the state, the registered flux-register storage and ``hydro_w``.
Retries stay on the host after the replay: ``check_redo`` restores the
step's backup, and a smaller ``dt`` reuses the graphs unless another capture
key component changed.

``MeshBlock.execution_report()`` counts captures, replays, eager stages and
held graphs.

Benchmark
---------

``tools/bench_execution.py`` (run from ``tests/``) times the four
precision/execution combinations of the dry Straka case on one GPU, with the
run order alternated between repeats, and reports the wall time per step, its
spread, the captures and replays, the peak memory and the speed-up over the
eager run of the same precision.
