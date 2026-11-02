# MPC C++20 core migration and dead-code cleanup — 31 October 2026

## Objective and scope

Migrate the **whole maintained MPC package** to C++, delete superseded C translation units after validation, and retain one physical model / one production solver. This is *not* a new controller policy or a proof of runtime speedup: changing a `.c` suffix to `.cpp` does not itself produce faster machine code. The existing C++ Unity/source-derived plant, exact 30 × 25 ms predictions, two SQP passes, 23-state/2-input Riccati QP, commanded throttle/steering, strict physical wall/corridor tests, and 40 Hz actuator bridge remain intact. Production racing YAML, hard margins, optimizer campaign and protected five-run practice baseline remain unchanged.

## Implementation completed in working tree

- Removed the five formerly maintained production C files in `f1tenth_mpc/src`: `mpc_reference.c`, `vehicle_model.c`, `mpc_linearization.c`, `mpc_rti.c`, `riccati_solver.c`. Their maintained replacements are the identically named `.cpp` files. All sources under `f1tenth_mpc/src` are now `.cpp`; there are zero `.c` files in the MPC source directory. These are actual source-tree replacements, not an extra optional C++ backend. The original 5 C versions are saved **outside** the checkout in `/tmp/sdu_mpc_before_cpp_migration_20261031` solely for local independent comparisons; Git history retains earlier versions.
- `f1tenth_mpc/CMakeLists.txt`: `LANGUAGES CXX`, C++20, Eigen3 fixed-size Riccati kernels, OpenMP CXX; removed C compiler requirement. Public header C ABI remains deliberately stable for replay/tools even though implementation is pure C++.
- Updated offline replay and fixture source fingerprints for the actual `.cpp` source paths: `tools/mpc_qp_fixture.py`, `tools/benchmark_mpc_replay.py`, and incidental audit comments. Otherwise fixture source identifiers would refer to deleted source files.
- Removed 255 lines of permanently compiled-out `MPC_ENABLE_FD_ORACLE` finite-difference Jacobian and its no-op config switch. Kept production source-derived derivatives, known-step sensitivities, and full-state affine API needed by tests. The full-affine calculation now shares one implementation; the reduced MPC QP still avoids this unnecessary full-state calculation entirely.
- Eliminated one unconditional whole-horizon previous nominal copy: keep snapshot only when caller aliases output and prior input; added explicit test that compares aliased and unaliased full predicted states, controls and progress.
- Removed duplicate horizon control copies around nonlinear candidate rollout, while retaining necessary post-validation controls. Consolidated success/failure Jacobian worker statistics into one shared block, removed duplicate conditional handling and unnecessary temporary durations. Converted old C compound literals/aggregate initialization to C++ aggregate equivalents.
- Removed dead, default-OFF, closed-loop-rejected *experimental R2 nominal rollout reuse* along with its unused configuration switch and its telemetry field from ROS and the replay tool. **Kept both full R1 and R2 SQP passes**; any later rollout-reuse optimization must be independently revalidated before it re-enters production.

The five replaced C sources contained **7,538 lines** before migration; their maintained C++ counterparts now total **7,196 lines**, a **342-line / 4.5% net reduction**. This reduction is separate from earlier Riccati C/Eigen consolidation already present in the starting checkout. It is not a reduction of the entire ROS package, which also includes large controller, physics, and test files; further architectural splitting requires a separate measured refactor.

## Correctness validation

- `source /opt/ros/jazzy/setup.bash; colcon build --packages-select f1tenth_mpc --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release`: PASS after the final migration and cleanup.
- `colcon test --packages-select f1tenth_mpc` and `colcon test-result --test-result-base build/f1tenth_mpc --verbose`: **211 tests; zero errors, zero failures, zero skipped**. Includes new aliased warm-start nominal regression.
- Eight genuine saved IROS QP fixtures × two initialization modes (`recorded`, `cold`): **16/16** same QP status and iterations, zero differences in reported primal/dual/objective/equality values. Bench p50 medians: original compiled C-wrapper ~2293 µs, migrated C++ ~2266 µs. This ~1% difference is not evidence of systematic speedup. Measured on CPU8, background optimizer workload present.
- 18 + 10 difficult saved IROS full-state replays (both SQP passes, exact nonlinear 750 ms physical checks), against archived pre-migration executable: **28/28 no critical status/feasibility/SQP-decision difference, zero first steering or throttle delta at reported precision**. Repeated after final FD dead-code removal and affine deduplication. Script `/tmp/sdu_cpp_migration_full_mpc_parity.py` and report `live_runs/mpc_collision_margin_experiment_20261030/cpp_core_migration_fullreplay_parity_20261031.json`. This is saved-state parity, not a proof of closed-loop robustness.
- `git diff --check`: PASS.

## Independent Unity practice closed loop (11 crossings: warmup + ten scored)

| Run | Outcome | Scored time | Max callback | Deadline misses | Recoveries |
| --- | --- | ---: | ---: | ---: | ---: |
| `practice_cpp_core_migration_20261031_01` | Rejected at physical corridor@0 before scored lap | — | 17.277 ms | 0 | 0 |
| `practice_cpp_core_migration_20261031_02` | Completed ten scored laps; no collision or shutdown | 49.737 s | 18.283 ms | 0 | 3 |
| `practice_cpp_core_finalclean_20261031_01` | **Final-source run** completed ten scored laps; no collision or shutdown | **50.137 s** | **19.252 ms** | **0** | **18** |

The first trial had competing background static analysis and ended because both the candidate and the prior-plan short prefix failed exact physical corridor validation; the controller failed closed rather than issuing unsafe commands. A subsequent trial without that competing analysis completed ten scored laps, but neither this single completion nor the first failure satisfies the acceptance gate. The final-source run after FD-oracle removal and the full-affine helper deduplication also completed, with no publication deadline misses, but had 18 recoveries, a 50.137 s ten-lap time, and 19.252 ms maximum callback. **This is still not repeated full live qualification:** one earlier migration trial rejected before a lap and worst measured callbacks remain above 17 ms. Lap time is recorded as a diagnostic, not a gate for source cleanup and timing improvements (updated user priority on 31 October).

### Observed timing on the completed run (2,240 callbacks)

| Measured phase | p50 | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: |
| callback to publication gate | 8.342 ms | 16.668 ms | 17.292 ms | 18.283 ms |
| Jacobian wall time | 2.414 ms | 5.308 ms | — | 5.395 ms |
| exact nonlinear rollout (all candidate/backtrack work) | 3.151 ms | 8.200 ms | 8.973 ms | 10.002 ms |
| ADMM solver | 0.592 ms | 3.595 ms | 4.743 ms | 13.005 ms |

The slowest callback of the **second** trial (18.283 ms) was an accepted viability-recovery cycle with 13.005 ms ADMM. Other 17.8–18.2 ms callbacks were two-pass SQP cases with roughly 7–10 ms cumulative exact rollout and multiple attempted steps. On the **final-source** 2,281-callback run: publication p50 8.943 ms, p95 16.604 ms, p99 17.297 ms, max 19.252 ms. Final-source exact rollout p99 9.082 ms (max 11.557 ms); ADMM p99 5.388 ms (max 13.309 ms). The slowest callback was an accepted feasible action with 13.309 ms ADMM. This identifies actual variability: **nonlinear rollout/backtracking and rare high-iteration ADMM**, rather than C-vs-C++ syntax. Keep exact independent acceptance tests, source-plant state channels, per-point Jacobians, full 750 ms horizon, all hard body constraints and R2. Never skip safety tests for speed.

## Acceptance and next engineering work (not yet fulfilled)

1. **Performance hardening before behaviour tuning:** instrument number and cause of source-plant propagations per callback, line-search candidate repeats, QP factors, and recovery budget; evaluate reuse only for the *identical, unmodified* state+command sequence with explicit cache scope and independent validation. Treat jitter and rare ADMM max-iteration recovery as a separate hard latency issue. Aim worst observed <=17 ms under representative load, hard publication 20 ms, not just a lower average.
2. **Reliability:** investigate first-stage corridor infeasibility and recursive viability before physical boundaries: optimize constraints against full world vehicle footprint, not only Frenet corridor and post-hoc rejection; add bounded terminal continuation to normal SQP. An already unreachable state may have no physically safe command; never promise unconditional safety or disable fail-closed handling.
3. **Robust acceptance:** independently achieve repeated complete ten-scored-lap Unity practice and IROS runs with no missed 20 ms command gates, collision, or latched controller failure, ideally <=17 ms worst observed callback under representative load. **Do not gate numerical cleanup on a 49.5 s total:** lap time is diagnostic until the timing/reliability phase is complete. Preserve the protected working build until the replacement qualifies.
4. **Only then** reconfigure lateral/heading and other weights independently, test solver convergence quality, improve fallback quality and recover practice lap time; there is no fixed lap-time target during numerical cleanup. No weight tuning, braking policy changes, margin loosening, or horizon shortening were made during this migration.

## Reproduction and working-tree caution

This work is in the user's **local working tree** on branch `mpc/next-engineering-handoff-20261021`, baseline HEAD `43312ea19`. It is deliberately not committed or merged. The working tree already had substantial uncommitted MPC source and diagnostic changes before this migration; do not reset them. The active PPO/IPOPT optimizer campaign was preserved and yielded only for isolated Unity trials. To reproduce offline replay after build: `cmake --build build/f1tenth_mpc --target mpc_offline_qp_tools -j4 && python3 /tmp/sdu_cpp_migration_full_mpc_parity.py`, subject to temporary archived baseline executable still existing. Prefer recording a permanent tracked baseline procedure for future comparisons instead of depending indefinitely on `/tmp` files.
