# MPC Riccati rare-tail latency: captured hard states and scoped FTZ/DAZ

Date: 2026-10-31. Status: **five-run practice qualification PASSED**, IROS cross-map and the preferred 15–17 ms worst-case callback target **not yet qualified**.

## Reproducible evidence and interpretation

- Two previous clean 10-scored-lap practice bags showed **14.100 ms** and **13.989 ms** max ADMM wall time with 9 and 6 adaptive rho changes, respectively. Their thread CPU time was essentially the same, so this was mostly actual arithmetic, not OS descheduling. The hardest calls had only 92 and 62 iterations, so iteration count alone was not responsible.
- Using the **exact recorded ROS diagnostic source state, 22-dimensional hidden PhysX state, command queues and actual 4800-point practice raceline**, six slow source-state rows were exported from those two bags. Each was replayed through the original production RTI/QP builder with the same tracked weight and trust-region overlay. Files and manifest: `live_runs/mpc_collision_margin_experiment_20261030/recorded_live_slow_state_qps_20261031/`. The six subdirectories contain a frozen `qp_000000.fqp`, source-state CSV, replay output and logs.
- **Important limitation:** the ROS bag did not contain a byte-for-byte copy of the live RTI nominal, solver's mutable warm-start ADMM state or original per-stage QP matrices. The reconstruction re-seeds the nominal and solver independently; these are QPs *reconstructed from real hard Unity source states*, **not byte-identical live QP snapshots**. The live symptom and reconstructed slow case are related, but their numerical or wall-time identity is not established.
- A production-solver profiling build replayed these six QPs on an isolated pinned CPU. One source-state QP had **12.366 ms median** (recorded-seed replay), with **9.346 ms factorization**, 1.401 ms factored RHS, 0.303 ms projection, and 15 factorizations. Other cases had ~0.89–4.38 ms median. An inspection of input QP matrices found no directly stored float32 subnormals; the slowdown arises from arithmetic/intermediate behavior rather than an obviously subnormal input matrix element.

## Root cause experiment

On x86, denormal/subnormal intermediate floats can trigger slow floating-point assists during dense Riccati matrix operations. An A/B experiment using a **temporary process-level MXCSR FTZ+DAZ preload** shortened the worst reconstructed QP by **~4.6×** without any solver-status or iteration difference. That temporary preload was only a test and was **not deployed**.

## Production code change

`f1tenth_mpc/src/riccati_scaling.cpp` now uses a compact RAII `ScopedRiccatiDenormalMode` to enable SSE FTZ (bit 15) and DAZ (bit 6) **only for the duration of a Riccati ADMM solve**, and to restore the caller thread's original MXCSR on *every* return, including null-input errors and solver time budget expiration. Non-x86/SSE targets use unchanged arithmetic. This keeps ROS synchronization, source-plant rollout and diagnostics outside the modified floating-point mode. All QP/corridor constraints, rho adaptation, factorization recursions, exact native rollout, tolerances, publication gate, and fallback policies remain unchanged. Only values smaller than the minimum normal float32 magnitude are affected by the mode. A standalone GTest checked MXCSR restoration after valid and error returns.

## Before/after pinned production QP results

Control: one pinned CPU (CPU8), temporarily yield unrelated IPOPT optimizer only during paired measurements, five interleaved A/B pairs per case/seed, 70 solves per executable invocation; **60 matched A/B pairs** total. Stored at `recorded_live_slow_state_qps_20261031/scoped_ftz_production_ab.json`.

| Real-state diagnostic | Original median QP | Scoped FTZ/DAZ | Speedup |
|---|---:|---:|---:|
| practice split diag0612 | 3.512 ms | 1.946 ms | 1.78× |
| practice split diag0815 | 11.250 ms | 2.406 ms | **4.66×** |
| practice split diag0614 | 4.268 ms | 1.826 ms | 2.33× |
| baseline repeat diag0812 | 0.887 ms | 0.875 ms | 1.01× |
| baseline repeat diag2184 | 1.183 ms | 1.166 ms | 1.01× |
| baseline repeat diag0411 | 1.154 ms | 1.136 ms | 1.02× |

These are the recorded-seed comparisons. Cold-seed comparisons appear in the same JSON and are similar. Numerical statuses and iteration counts were identical in all 60 pairs.

- **28/28** QP fixture cases (8 previous + 6 reconstructed, each with cold and recorded starts): exact match for all numerical result fields before and after scoped FTZ; benchmark-time fields excluded. Evidence: `live_runs/mpc_collision_margin_experiment_20261030/ftz_production_frozen_qp_parity_20261031.json`.
- **28/28** saved difficult full-state source-plant MPC replays: exact first steering/throttle and no critical status/line-search changes. Evidence: `live_runs/mpc_collision_margin_experiment_20261030/ftz_full_replay_parity_20261031.json`.
- **214/214** ROS C++ tests pass after adding the scoped MXCSR restoration regression test. `git diff --check` clean.
- A standalone GTest confirmed successful and early-error returns leave the calling thread's MXCSR exactly as set before the Riccati entry. It has now been integrated permanently at `f1tenth_mpc/test/test_riccati_denormal_scope.cpp`, with a dedicated CMake GTest target. The test-only change was made **after** all five races; the production library source was not modified during qualification. The final Release C++ suite passes **214/214** tests.

## Closed-loop qualification

The run series `practice_riccati_scoped_ftz_20261031_01` to `_05` uses 11 lap crossings (one warmup + ten scored) and stops early at any collision, incomplete race, or publication deadline miss. Evidence is under `live_runs/mpc_collision_margin_experiment_20261030/scoped_ftz_practice_five_run_summary_20261031.json` and individual run's full bag/cycle report.

- **Run 01:** 10/10 scored laps, 0 physical collisions, 0 missed publication deadlines, 17 accepted recovery actions, worst callback **17.890 ms**, 50.767 s. Worst ADMM **7.038 ms**, versus 13.989 ms on a previous full-distance baseline. ADMM p99 4.847 ms versus prior 4.711 ms: the improvement is **worst-case-tail**, not p99 or mean. The first race had more recoveries and slower lap time; do not attribute those differences solely to FTZ without repeatability evidence.
- **Run 02:** 10/10 laps, 0 collisions, 0 deadline misses, 11 recoveries, 18.094 ms max callback, 50.261 s, max ADMM 7.679 ms.
- **Run 03:** 10/10 laps, 0 collisions, 0 deadline misses, 0 recoveries, 18.245 ms max callback, 49.475 s, max ADMM 6.612 ms.
- **Run 04:** 10/10 laps, 0 collisions, 0 deadline misses, 4 recoveries, 18.212 ms max callback, 50.090 s, max ADMM 6.101 ms.
- **Run 05:** 10/10 laps, 0 collisions, 0 deadline misses, 4 recoveries, 18.293 ms max callback, 49.769 s, max ADMM **7.890** ms.

**All five races were complete and clean:** 50 scored laps, zero collisions, zero missed publication deadlines, 36 accepted recovery actions, fastest run 49.475 s, mean 10-lap time **50.072** s, maximum callback 18.293 ms. The previous two baseline complete races had maximum ADMM times ~14.100/13.989 ms; the five new races each stayed below **7.890** ms. This indicates substantial reduction in rare solver tail but is not a strictly paired identical-state live counterfactual. The p99 callback remains ~17.3 ms, so the preferred sub-17 ms worst-case has **not** been reached.

## Next bottlenecks and guardrails

1. Verify multiple full-distance Unity races with zero publication misses, safety violations and dropped actuator commands. The preferred callback cap is **15–17 ms**; 20 ms is a hard budget, not a goal.
2. If FTZ qualifies, the major remaining callback work is **full native nonlinear rollout** (baseline p99 ~8.96 ms) and per-stage **Jacobian generation** (baseline p99 ~5.35 ms). Investigate duplicate same-state evaluations or cheaper *mathematically identical* derivatives and geometry caching, never a surrogate physical model or omitted safety validation.
3. Only after measured repeated improvement consider revising behavior/weights/recovery. Never silently relax corridor constraints or increase unsafe command publication when the QP remains infeasible.
4. Source splitting has already separated diagnostics, recovery and the unique 30-stage QP builder. Another mechanical file split cannot alone make execution faster; remove old code only after passing all numerical/closed-loop tests. Avoid changes to unrelated optimizer processes or sim.
