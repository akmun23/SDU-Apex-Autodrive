# IPOPT recovery and performance work — 2026-10-28

## Current job and scope

The active all-day continuation is
`sdu-apex-ipopt-optimized-20261028.service`, output
`optimized_practice_v354_durable_optimized_20261028`.
It starts from the complete saved factor-2 primal, not the original path CSV.
Remaining factors are **4,16,64,256,1024,1e12**, 6000 iterations each.
The last, very wide numerical trust box allows much larger geometry changes;
physical dynamics, footprint clearance, speed, steering and positive-throttle
bounds remain hard. It is not permission to accept an infeasible trajectory.
No CPU/wall cutoff is imposed. CPU 15, one BLAS thread, low OS priority keep
runtime development independent. Sleep inhibition is bound to this service.

```bash
python3 tools/watch_ipopt.py optimized_practice_v354_durable_optimized_20261028 --unit sdu-apex-ipopt-optimized-20261028.service
```

This changes the offline optimizer, not production odometry, MPC weights or
the installed runtime raceline. An OCP-feasible target is not a demonstrated
Unity lap or a controller tracking certificate. The ten-lap racing goal is
not declared achieved by an optimizer objective.

## What happened before power loss

The original `optimized_practice_v347_all_day_20261028` **did continue**.
Its first time stage (factor 2) reached iteration 6000 and returned
`Maximum_Iterations_Exceeded`, but its final source-constraint residual was
0.0003611643894670933, below the 0.001 gate. The complete saved primal predicts
4.940067996783238 s. The code accepted it as a feasible restart and entered
factor 4; power loss occurred at iteration 1048, console objective 4.9388517 s
and primal residual 0.000793. No primal vector was saved for that interrupted
stage, so console numbers cannot recover those 1048 iterations.

Raw factor-2 checkpoint SHA256:
`bb9fc985db4ec3eacf0a2cb94fbcef4f011a7274a0839d0e39b3583b89c1418c`.
The original output remains untouched.

## Implemented corrections

1. **Preserve exact full-stage restart inputs.** The existing raw-stage seed
   converter transferred all differential states and both control arrays,
   but its modification marker triggered a steady-force throttle re-guess.
   That raised the initial residual to 0.096227. `run_ipopt.py` now preserves
   a complete, fraction-1, unconverted, unclipped raw-stage primal. Partial
   blends, coordinate conversion, speed replacement and explicit reprojection
   retain their normal initialization behavior. Every path still undergoes
   the complete current source-constraint audit. Corrected recovery reproduces
   the saved **0.0003611643894670933 exactly**, and skips unnecessary Phase I.
2. **Compile reusable source-force kernels.** The unchanged source equations
   are lowered once to a compact scalar CasADi SX function, reused from the MX
   OCP at knots, midpoints and seed evaluation. Exact derivatives are reused
   rather than expanding the same large force expression everywhere. The
   kernel and its derivatives compile with `-O3 -ffp-contract=off`; there is
   no fast-math, learned correction, changed curve or new model parameter.
   TemporaryDirectory owns the generated kernel files, avoiding competing
   cleanup by CasADi derivative objects.
3. **Retain sampled feasible incumbents separately from search restarts.** A
   faster sampled point is eligible only after a full current-constraint audit
   passes 0.001. It can survive an infeasible last iteration and be used for
   final selection. It is labeled `Sampled_Feasible_Incumbent`, not silently
   called IPOPT-converged. A faster terminal point inside the existing 0.005
   near-feasible restart gate can still seed the next stage; it is not confused
   with a certified output or discarded merely because a slower feasible
   incumbent exists.
4. **Durable mid-stage checkpoints.** At sampled iterations (default stride
   25), save the latest primal and best sampled feasible primal after 120 s
   has elapsed, and also at stage exit. Atomic replacement, file fsync and
   directory fsync protect completed writes. Archives contain all states,
   controls, phase lengths, primal decision vector, constraint multipliers,
   iteration, objective and measured constraint violation. A latest checkpoint
   may be infeasible; only the separate best-feasible checkpoint is marked
   feasible. Restarts must still recheck the current source equations. The
   sampling interval adds up to a stride of iterations to the nominal write
   interval; this is not a guarantee against hardware/storage failure.
5. **Cache checkpoint evaluation.** Bounds/parameters are fixed within a
   trust stage; cache them once and evaluate checkpoint channels together.
   This removes repeated reconstruction of the huge Opti initialization map.
   Checkpoint errors are counted and printed rather than silently destroying
   an otherwise valid optimization.
6. **Batch export and cache shared channels.** Stack node/midpoint expressions
   into matrix evaluations. Twenty export sites no longer issue thousands of
   scalar `Opti.value()` calls. Forces/slips used by multiple report and CSV
   fields are evaluated once per immutable node list. Shapes and output
   channels are preserved; no diagnostic or feasibility check is removed.
7. **Honest status and diagnostics.** An inactive service overrides a stale
   `running` status file in the dashboard. Remaining factors and iteration
   count are explicit launcher options, validated before creating output.
   `advance_to_next_trust_stage` replaces the misleading failed-stage-only
   continuation field. Stage reports retain evaluation timers and terminal
   status/residual alongside selected sampled-incumbent data. An IPOPT setup
   exception is no longer masked by an unconditional `opti.stats()` call on an
   unsolved stack. The viewer distinguishes derivative setup from iterations.

## Measurements and tests

Old factor-2 stage, 6000 iterations, total IPOPT time 21451.643 s:

| Evaluation | Mean cost | Approximate share of total time |
| --- | ---: | ---: |
| Lagrangian Hessian | 2.71 s | 76% |
| Constraint Jacobian | 530.83 ms | 15% |
| Constraints | 46.66 ms | 3.6% |

This motivates optimizing expression/derivative evaluation before adding more
iterations or changing physical equations.

Reusable benchmark:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv/bin/python SDU_Apex_IPOPT_PhysX_MinTime_v2/benchmark_source_equation_kernels.py --seed optimized_practice_v348_power_recovery_20261028/recovered_factor_2_seed.npz --nodes 32 --jit
```

Recorded 32-node kernel benchmark, medians over 15 repeats:

| Evaluation | Inline MX | Scalar SX | Compiled scalar SX |
| --- | ---: | ---: | ---: |
| Forces | 1.368 ms | 0.242 ms | 0.133 ms |
| Jacobian | 12.472 ms | 3.430 ms | 1.535 ms |
| Exact Hessian | 48.602 ms | 6.391 ms | 1.194 ms |

Maximum differences: forces 0, Jacobian 3.41e-13, Hessian 2.18e-11.
These are kernel timings, not a 40x whole-optimizer speedup claim.
Whole-size compiled-kernel screen v351 measured Hessian 1.34 s and Jacobian
388 ms in its first ten-iteration stage. Its deliberately every-iteration
checkpoint test cost about 0.74 s per callback, motivating the cached
checkpoint evaluator and normal stride 25. The old 6000-iteration stage and
short screen have different iterate distributions; a controlled full-run
speedup remains to be measured.

Tests cover force values, sparsity, Jacobians and exact Hessians across 45
speed/steering combinations (including zero speed and either side of 5 m/s),
interpreted and compiled kernels, batched shapes/model binding, atomic
checkpoint behavior, infeasible/latest versus feasible/best separation,
nonmutating reconstruction of the complete Opti primal, and near-feasible
restart separation. Existing source-overlay/restart tests also pass. Pytest
plugin auto-loading is disabled for these tests: the optimizer venv lacks
`lark` required by the unrelated host ROS launch-testing plugin.

## Experiments and known pitfalls

- v348 initial recovery was stopped when the throttle re-guess was found;
  its recovered seed file is still the input artifact for later jobs.
- v349 successfully rechecked the exact saved residual and entered factor 4.
  It was superseded by v354 after the source-kernel and checkpoint tests;
  no new certified stage-end result had been exported from v349.
- v350 exposed an Opti API mistake in sampled-incumbent reconstruction:
  `opti.x` is a concatenation, not an assignable symbol. Fixed by reconstructing
  assignments for each actual decision symbol, with shape/parameter tests.
- v351 passed both time stages, selected the correct sampled feasible primal
  instead of the infeasible terminal iterate, and saved checkpoints. It then
  exposed the scalar-export bottleneck. A SIGUSR1 stack dump located execution
  in repeated `Opti.value()` export calls. This short test was stopped and
  superseded by the batched-export screen; the all-day continuation was not
  stopped merely to run an unrelated test.
- v352 tested whole-oracle compilation together with a global CasADi `cse`
  option. CasADi 3.8.1 propagates that option to MX gradient generation, which
  rejects it. That invalid option is removed, not retained as a fallback.
  The scalar SX function's valid CSE option remains.
- v353 tested whole-oracle JIT without that invalid option, plus batched export.
  GCC `-O1` exceeded 20 GiB compiler RSS and induced heavy swap before any
  iteration. This separate test was stopped; it was not a solver failure.
  v355 tests lower-overhead `-O0 -ffp-contract=off` compilation on CPU 14.
  Whole-oracle compilation is **not** enabled in the main all-day job until
  correctness, memory and sustained cost pass. Its startup cost must be
  amortized over an actual run.

## Next checks

1. Confirm v354 passes the source audit and reaches numerical iterations with
   periodic durable saves and no checkpoint errors.
2. Finish the short whole-oracle compile/export screen. Check identical initial
   feasibility, numerical iteration parity, complete CSV/checkpoint export,
   timing and memory. Promote compilation only if its measured net gain warrants
   replacing the running job from a durable checkpoint.
3. Inspect stage timing and retained feasible improvements. Investigate numerical
   conditioning/continuation when objective improvement stalls; do not change
   the physics or loosen the 0.001 feasibility gate to manufacture progress.
4. Once a genuinely faster feasible target exists, validate it with the actual
   simulator/controller workflow before claiming faster racing performance.
