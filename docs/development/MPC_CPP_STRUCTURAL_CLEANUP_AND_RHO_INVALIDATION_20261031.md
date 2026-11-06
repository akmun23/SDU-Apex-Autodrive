# MPC structural cleanup and Riccati factor ownership — 31 October 2026

## Scope and priority

The 49.5-second practice time is **not** an acceptance gate for these changes. The immediate goal is a single understandable and efficient production C++ implementation, reliable complete races under the **20 ms** command deadline, ideally **15–17 ms** maximum callback, with zero collision and no unchecked fallback. Behavioral/weight/lap-time tuning comes only after robust timing. The 30-stage 750 ms PhysX-source prediction, full vehicle footprint, hard corridor/world geometry, both RTI/SQP passes, all feedback and terminal checks, and the configured solver convergence/adaptive-penalty policies were not relaxed.

## Completed structural changes

1. **ROS diagnostics moved out of the 5,239-line controller node** to `f1tenth_mpc/src/mpc_diagnostic_worker.hpp` and `mpc_diagnostic_worker.cpp`. The controller is now **4,956 lines**; its asynchronous ROS diagnostic queue, timing recorder and worker implementation live in a dedicated component. Its queue replacement behavior, low-priority worker and all timing counters are preserved. CMake builds the worker with the node; there is one, not two, publisher implementations.
2. **Riccati responsibilities separated into independent C++ translation units:** `riccati_solver.cpp` (ADMM convergence, projection, adaptive rho, warm start), `riccati_factorization.cpp` (the one quadratic factorization and the single Eigen factored-RHS interface), `riccati_scaling.cpp` (QP transformation, output inverse-transform and public solve entry points). The original 1,581-line solver is now a ~964-line ADMM core, ~340-line factorization module and ~286-line scaling module. `riccati_solver_internal.hpp` declares only the two private cross-module operations and profiling/debug access. Public `riccati_solver.h` and the C-compatible replay ABI remain stable. Debug state remains owned by the ADMM module instead of being a writable external global.
3. **Small RTI/control-flow cleanup:** removed repeated nominal-build duration/seed copies in separate success and failure branches, and a redundant unchanged first-action assignment. Eliminated an always-true warm-start ternary and stale comments. Old `.c` solver and source-plant translation units from the previous migration remain deleted; no legacy implementation was reinstated.
4. **Source fingerprints corrected** in `tools/mpc_qp_fixture.py` and `tools/benchmark_mpc_replay.py`, so the validation artifacts include all three Riccati units and their private interface rather than hashing only the now-smaller `riccati_solver.cpp`.

## Incidental numerical hot-loop cleanup found during extraction

`riccati_solver_primal_pass()` previously compared **773 floating-point ADMM penalties** (31×23 state, 30×2 input) before *every* iteration, even when adaptation had not changed any penalty. The factorization is allocated exclusively for the current fixed QP. The only mutable quadratic terms are the penalty vectors, changed inside the ADMM core at the explicit rho-adaptation event. The ADMM core now sets `factorization.valid=0` **only when the adaptation actually changes those vectors**, forcing the same recomputation on the following primal iteration. The private factorization step checks this validity flag rather than scanning unmodified arrays. Public factored solves remain independent and still permit changing affine dynamics `d` without recomputing quadratic factors.

This is a redundant-work removal, not a new numerical method or changed adaptation schedule. All 16 saved QPs × cold/recorded configurations had **exactly identical status, iterations, primal/dual residuals, objective and equality diagnostics** against the immediately preceding modular solver. The entire MPC package has **212 tests passed, zero errors or failures**. All **28** archived source-plant full-state RTI/SQP replays matched critical statuses and first steering/throttle outputs exactly at the stored precision. Benchmark: **80 interleaved pairs**, pinned CPU8 with optimizer yielded, median new/old QP ratio **0.98497** (~1.5% reduction), 2,174.57 vs 2,151.01 µs median across the sampled QPs; not a guaranteed latency reduction across all operating regimes, and *not* a solution to the rare 13 ms ADMM tails. Frozen benchmark artifacts:

- `live_runs/mpc_collision_margin_experiment_20261030/cpp_riccati_module_clean_bench_20261031.json` — modular split alone, new/old 1.00711, effectively neutral
- `live_runs/mpc_collision_margin_experiment_20261030/cpp_riccati_explicit_rho_bench_20261031.json` — explicit invalidation, new/old 0.98497
- `live_runs/mpc_collision_margin_experiment_20261030/cpp_riccati_explicit_rho_exact_parity_20261031.json` — sixteen exact field parity records
- `/tmp/sdu_mpc_rho_explicit_fullreplay.log` — 28/28 full MPC source-state parity

## Live regression

- `practice_cpp_riccati_diagnostics_modular_20261031_01`: **ten scored laps completed**, zero collisions, zero publication misses, **two** recovery actions, maximum callback **18.054 ms**, 49.559 seconds ten-lap time (diagnostic only). This validates the structural split *before* the explicit penalty invalidation optimization.
- `practice_cpp_rho_invalidation_final_20261031_01` (**final source after explicit rho invalidation**) completed **ten scored laps**, zero collisions, zero publication misses, **one** accepted recovery, maximum callback **18.075 ms**, total 49.540 s (diagnostic only). The callback still exceeds the preferred 17 ms maximum, so this is not a robust timing qualification.

## Optional profiling build regression fixed

After separating translation units, a separate `SDU_APEX_MPC_RICCATI_PROFILE=ON` build found two pre-existing direct debug-object references in `riccati_scaling.cpp`, which are unreachable in the normal production build. Replaced them with the same private `riccati_debug_last_mutable()` interface used in factorization. The profiling-enabled `mpc_core` and linked `mpc_qp_fixture_replay` build successfully in an isolated CMake directory. The final normal Release ROS package also builds and retains **212 passing tests**. The correction affects profiling-only counter writes, not release arithmetic or controller policy.

Final-source live cycle details: **2,231** callback records; callback p50 8.476 ms, p95 16.720 ms, p99 17.382 ms and max 18.075 ms. Exact nonlinear rollout p99 8.977 ms (max 11.397 ms); ADMM p99 4.357 ms (max 5.930 ms). These are sample observations, not bounds or proof of tail elimination.

## Still open

**Reliable sub-17 ms tail and repeated full-distance races are not qualified.** The dominant remaining callback costs are exact nonlinear source-plant rollouts/backtracking and unusual long-iteration ADMM episodes; a 1.5% per-QP microbenchmark gain does not solve either worst-case. Do not change weights/fallback policy for performance until repeated full races demonstrate timing/robustness. No optimizer source, production racing YAML, simulation physics or hard margins were edited here. Work is in the existing, pre-dirty working tree, not committed or merged.
