# MPC controller / RTI source boundaries — 31 October 2026

## Purpose and safety contract

Refactor the two largest C++ modules into distinct responsibilities, keeping **one** production control algorithm, **one** native-source-model dynamics/Jacobian path, and exact independently evaluated footprint and corridor checks. This phase is **not** a weight or fallback-policy tuning exercise; the 49.5 s practice result is only diagnostic. Strict publication budget is **20 ms** with target worst-case **15–17 ms** once repeatability is qualified. Failure to publish an unsafe command is fail-closed, not a valid completed race.

## Controller source boundaries (fully built)

- `f1tenth_mpc/src/mpc_controller_node.cpp` (5 lines): ROS component registration, no algorithm or duplicate class definition.
- `f1tenth_mpc/src/mpc_controller_node_impl.hpp` (~2,422 lines): the one private node class, members and retained inline lifecycle/control coordination. Header-only controller-local helpers now have external **inline** linkage in `controller_detail` to avoid anonymous-namespace ODR conflicts across translation units. No new public class/library API.
- `f1tenth_mpc/src/mpc_controller_diagnostics.cpp` (~936 lines): the sole existing JSON diagnostic method, retaining schema version 4, field sequence, cached source outputs, ROS diagnostics queue timing and failure reporting. It no longer obscures the main source.
- `f1tenth_mpc/src/mpc_controller_recovery.cpp` (~1,663 lines): existing prior-plan, reacquisition, full-horizon viability, quality-aware prefix and candidate-prefix recovery. All exact command-time revalidation and physical acceptance checks are **unaltered**.
- `mpc_diagnostic_worker.{hpp,cpp}` remains the previous validated bounded, asynchronous publisher interface; the failed experiment that extracted its remaining methods was **not** reapplied.

## RTI source boundaries (fully built)

- `f1tenth_mpc/src/mpc_rti.cpp` (~3,265 lines): the existing exact source plant rollout, trajectory progress tracking, nonlinear objective/geometry validation, nominal seed, and R1/R2 cycle coordination.
- `f1tenth_mpc/src/mpc_rti_qp.cpp` (~594 lines): **sole** full-state per-stage Jacobian/QP builder, objective and convex corridor approximation; the original code was moved, not copied. No hidden reduced-horizon models or surrogate physics.
- The attempt to move the 732-line RTI pass into `mpc_rti_pass.cpp` was **reverted** (see below). The unique pass therefore remains within `mpc_rti.cpp` until separately qualified. No obsolete pass implementation remains.
- `f1tenth_mpc/src/mpc_rti_internal.hpp` (25 lines): private declarations shared between QP builder and exact rollout. Only the minimum helper linkage necessary for this QP split has changed; `mpc_rti.h` and the public ABI are unchanged.
- CMake explicitly compiles exactly one source file for each responsibility; legacy `.c` code remains deleted.

## Validation evidence before final Unity run

- Release ROS build succeeds; `colcon test` **212 tests, 0 errors, 0 failures**.
- Archived full-state 30-stage MPC replays: **28/28**, exact first steering/throttle and identical critical solver, line-search and feasibility statuses, after both extractions.
- Live validation stage 1 (controller diagnostics and recovery split): `practice_cpp_controller_diag_recovery_split_20261031_01` — **10 scored laps**, zero collisions, zero publication misses, **one** accepted recovery, maximum callback **17.928 ms**, 49.450 s (lap time is diagnostic).
- Live validation stage 2 (plus RTI QP builder split): `practice_cpp_controller_qp_source_split_20261031_01` — **10 scored laps**, zero collisions, zero publication misses, **three** accepted recoveries, maximum callback **18.102 ms**, 49.720 s (diagnostic).
- Live validation stage 3 (plus RTI solve-pass split): `practice_cpp_controller_rti_pass_split_20261031_01` — **FAILED** at zero scored laps. Fail-closed exact source/footprint corridor rejection at stage 0, solver reported an iteration-limited iterate, no command was issued, and maximum callback was 17.108 ms. There were zero collisions and zero deadline misses, but the race did not finish. Despite 212 passing tests and 28/28 frozen replay parity, the *entire* solve-pass extraction was reverted to its exact previous source and header; the previous stage-2 (controller + QP) layout was rebuilt and all 212 tests passed. This does not establish causation, but it disqualifies the split.

## Follow-up recovery strategy extraction rejected and rolled back

A subsequent experiment split the recovery file into a 560-line retained-plan/reacquisition core, a 642-line `mpc_controller_viability.cpp`, and a 461-line `mpc_controller_prefix_selection.cpp`. The only change was out-of-line member-method placement, with no change to safety gates or command policy. The Release build, 212 tests and 28/28 frozen difficult-state replays all passed. However, `practice_cpp_recovery_strategies_split_20261031_01` **failed before a scored lap** with an invalid-model rejection at no defined stage, 3 accepted recovery actions, 1 missed publication deadline and **21.326 ms** maximum callback. A valid full-distance race is required for acceptance. The separate strategy units were therefore **deleted** and the original `mpc_controller_recovery.cpp` was restored byte-for-byte from its pre-experiment snapshot, as was CMake. The restored controller/QP baseline rebuilt and passed 212 tests.

**This does not prove causation.** Other process load and source-state sensitivity may contribute. The restored baseline completed a further Unity run, `practice_cpp_controller_qp_baseline_repeat_20261031_02`: **ten scored laps, zero collisions, zero missed deadlines, zero accepted recoveries, maximum callback 18.367 ms, 49.477 s total**. The same retained controller/QP source layout therefore has two independent successful full-distance races (QP split and baseline repeat). These are not five-repeat qualification and do not meet the preferred 17 ms worst case.

## Failure-state fixtures and incidental ADMM-tail finding

The full ROS bags for both failed experimental runs were decoded and the final 20 complete MPC/Unity hidden-plant state rows exported without numeric transformation beyond the existing replay schema. Each file includes a `.json` metadata sidecar and record-index window:

- `live_runs/mpc_collision_margin_experiment_20261030/practice_cpp_controller_rti_pass_split_20261031_01_failure_window.csv` — source-state corridor rejection at stage 0; 20/20 rows exported
- `live_runs/mpc_collision_margin_experiment_20261030/practice_cpp_recovery_strategies_split_20261031_01_failure_window.csv` — terminal `invalid_model` rejection; 20/20 rows exported

These fixtures do **not** preserve the live ADMM warm-start variables or retained-plan memory; replay parity cannot prove that the original callback failure is deterministically reproducible. Check recorded QP snapshot/solver-state capture separately before changing fallback.

A direct review of the successful complete-race cycle detail identified the **rare ADMM-tail contributor**: in `practice_cpp_controller_qp_source_split_20261031_01` the maximum ADMM time was **14.100 ms**, with **9** adaptive rho changes and **10** quadratic factorizations in that cycle. In the repeat baseline the maximum ADMM time was **13.989 ms**, with **6** rho changes and **7** factorizations; the same run had a distinct 13.420 ms ADMM cycle with **5** changes and **6** factorizations. The failed viability split had 15.955 ms maximum ADMM time and one 21.326 ms callback deadline miss. This supports repeated re-factorization under adaptive rho as a target for tail-latency reduction, but no solver-policy change or factorization approximation is justified by these observations alone. Further instrumentation corrected this priority: for the two worst **successful-race** ADMM cycles, solver wall-time/thread-CPU pairs were **14,100 / 14,095 µs** and **13,989 / 13,980 µs**. These peaks consumed actual CPU rather than waiting on OS scheduling. The slow cycles used only **92** and **62** solver iterations, respectively, so this is *not* simply a large iteration count. In a separate profiling-enabled frozen-QP replay of four recorded hairpin QPs (150 iterations each), factorization time was **0.68–1.05 ms**, factored RHS **0.80–0.89 ms**, polygon/projection/residual **0.22–0.24 ms**, adaptive rho update only **~5 µs**, total QP median **1.83–2.21 ms**. **Those fixtures do not reproduce the 14 ms tails** and cannot establish whether special polygon projection/constraint activity or state-dependent Riccati arithmetic causes the live problem. Capture QP fixtures from the actual slow-cycle source states, enable the existing breakdown counters, and optimize the observed expensive path with frozen numerical parity *before* changing adaptive-rho frequency, solver tolerances or safety acceptance.

## Final source/provenance validation

- `tools/mpc_qp_fixture.py` now fingerprints the **actual** RTI/QP files and internal interface, plus a separate controller-runtime source group; a five-line component registration stub is insufficient provenance on its own.
- `tools/benchmark_mpc_replay.py` records build flags for the new QP translation unit; `tools/verify_runtime_topic_policy.py` audits the relocated node/header/diagnostics/recovery implementation for restricted subscriptions. Static topic policy **passes**.
- **212/212** ROS C++ tests and **22/22** selected Python replay-manifest/source-provenance tests pass; **28/28** archived difficult source-state MPC replays show zero critical status and first-action drift. `git diff --check` passes. All old `f1tenth_mpc/src/*.c` numerical implementations remain deleted.
- An unrelated optimizer change was committed concurrently as `a8958bde3` while the MPC refactor was running. The MPC source work here remains in the dirty working tree and was **not** committed or merged by this work; do not reset/rebase over unrelated changes.

## Next actions

Keep the validated controller/RTI-QP source boundaries. Further extraction of viability and quality-prefix methods should wait for baseline repeatability evidence, and be isolated with a fail-closed callback regression rather than merged directly. Independently identify redundant strategy predicates and exact rollout calls, but **do not** remove genuine validation. Re-run frozen QP cases, 28/28 full-state replay, the complete 212-test suite, and a full-distance Unity race after any structural change. Benchmark ADMM/rollout tail latency separately before claiming a speedup. **Five complete repeated races and cross-map validation are still missing.** All work remains in the pre-dirty local checkout, not committed or merged; unrelated optimizer/IPOPT modifications were not edited.
