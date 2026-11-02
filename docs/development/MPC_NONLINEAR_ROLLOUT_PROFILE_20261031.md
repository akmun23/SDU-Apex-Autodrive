# MPC nonlinear rollout hot-path audit — 31 October 2026

## Outcome and acceptance

**The validated production MPC remains the scoped Riccati FTZ/DAZ release** (`docs/development/MPC_RICCATI_SUBNORMAL_TAIL_20261031.md`): five consecutive 10-scored-lap practice races, zero collisions, zero deadline misses, worst callback 18.293 ms, worst ADMM 7.890 ms. There is **no separately qualified nonlinear source-plant arithmetic optimization** in this audit. We tested three small candidates independently, rejected changes without a measured significant gain or complete closed-loop race, deleted the speculative code, and rebuilt the exact retained physics executable. Source dynamics, tire forces, map projection, nonlinear acceptance, line search, fallback and any published commands are unchanged.

## Correct cost model: callback total vs one exact rollout

One full candidate horizon is 30 × 25 ms = 750 internal **1 ms Unity/PhysX-derived steps**. The expensive callback metric `exact_rollout_us` accumulates **multiple distinct candidate re-integrations** when the original QP's full step fails the source corridor or the exact nonlinear objective merit check. The algorithm evaluates up to four damped candidate controls (α = 0.5, 0.25, 0.125, 0.0625), each separately validated against the full plant, full footprint/corridor and source-state limits. It stops at the **first** independently exact-valid, improving trial; it can also certify and retain the nominal when no trial qualifies. This expensive work is **not duplicate validation of identical dynamics**. The same physical state cannot generally be reused between different controls or line-search alphas.

Five 10-lap validated FTZ practice bags: **11,279 total callbacks**, of which **11,247 contain solver rollout timings**. The telemetry `line_search_trials` has the following distribution. Metrics are accumulated per callback, not individual source-step costs.

| Backtracking trials | Solver callbacks | Share | Median accumulated exact rollout | p99 | Maximum |
|---:|---:|---:|---:|---:|---:|
| 0 | 3,728 | 33.15% | 0.585 ms | 3.584 ms | 7.409 ms |
| 1 | 1,274 | 11.33% | 2.324 ms | 6.935 ms | 9.384 ms |
| 2 | 1,688 | 15.01% | 3.720 ms | 8.306 ms | 10.021 ms |
| 3 | 1,746 | 15.52% | 5.186 ms | 9.285 ms | 10.143 ms |
| 4 | 2,811 | **24.99%** | **5.952 ms** | **9.552 ms** | **10.752 ms** |

Across five races the final controller classified 7,518 cycles as `accepted_line_search`, 3,381 as `accepted_optimal`, 312 as `accepted_feasible`, 35 as `accepted_viability_recovery`, plus startup/other. 6,761 cycles had the exact nonlinear merit trigger. The median accumulated exact rollout time was 4.550 ms for `accepted_line_search`, versus 0.578 ms for `accepted_optimal`. Therefore the dominant prospective improvement is **fewer safely necessary complete rollout trials**, not eliminating an unexplained 10 ms single-rollout hotspot.

Raw aggregate and full distribution: `live_runs/mpc_collision_margin_experiment_20261030/nonlinear_rollout_trial_analysis_20261031.json`. Full trace source: each of the five `live_runs/practice_riccati_scoped_ftz_20261031_0{1..5}/mpc_cycle_detail_diagnostics.jsonl` files.

## Experiments run, with explicit rollbacks

1. **Transient Frenet update at exactly zero path curvature**. The loaded-frame MPC projection replaces the transient local lateral/heading tracking components after the 1 ms plant integration. Algebraically bypassing `hypotf`/`atan2f` at curvature zero passed 214 ROS tests and 28/28 difficult full-state replays with zero first-control drift; however the paired physical rollout measurements were effectively unchanged (~1.8–5 ms depending on hard state). Restored `f1tenth_mpc/src/vehicle_model.cpp` byte-for-byte from the pre-experiment file. No fast-path branch remains.
2. **Global denormal FTZ/DAZ for the whole source plant**. Six recorded hard-state offline replays showed no reliable reduction in nonlinear rollout duration, though solver results remained numerically identical. No global floating-point mode was deployed. Retained FTZ remains tightly scoped **only** to the Riccati ADMM solve.
3. **GCC/Clang `-fno-math-errno` for `unity_source_plant.cpp` only**. Nine interleaved paired timings on each of six hard source-state QPs showed approximately 2.5–4.6% faster physical rollout and no checked numerical/result differences. **Closed-loop rejected**: first Unity run `practice_nativeplant_matherrno_20261031_01` stopped after eight scored laps from `rejected_nonlinear_rollout` (corridor at stage 0). No collision and zero publication deadline misses, but incomplete race. The flag was **fully removed** from CMake; the original production binary was rebuilt and 214/214 C++ tests passed. The failure does not prove compiler causation; nonetheless the optimization was *not qualified*. Raw A/B data: `recorded_live_slow_state_qps_20261031/matherrno_nativeplant_ab_20261031.json`; failure ROS bag/run report under `live_runs/practice_nativeplant_matherrno_20261031_01`.
4. **Skip `sqrt(max(1, combined_slip_demand²))` when demand² ≤ 1**. This algebraically preserves the primal and twenty-lane automatic differentiation for that branch. Tests and 28/28 difficult replays matched; nine interleaved A/B trials for six hard states showed no consistent timing benefit. Exact source file restored and the branch deleted. Raw A/B data: `recorded_live_slow_state_qps_20261031/utilization_sqrt_nativeplant_ab_20261031.json`.
5. **Skip candidate-only tracking Jacobian**. Existing nominal QP uses the 3×6 tracking Jacobian and must keep it. A 40,000-sample production helper benchmark measured **0.332 µs with Jacobian** vs **0.307 µs without**, just 0.025 µs/stage. The total saving would be a few microseconds across several 30-stage rollouts and would change the auxiliary candidate-state validity flags. No code changed.

No experiments involved relaxation of the 750 ms full exact source model, substitute physics, reduction of 1 ms integration steps, or skipping corridor certification.

## Retained post-run observability improvement

`tools/export_mpc_run_report.py` now also writes existing `solver.line_search_trials` and `solver.line_search_us` into per-cycle CSV. This uses the **already-published diagnostic payload**; it adds zero work to the 200 Hz control callback. All 17 `tools/test_export_mpc_run_report.py` tests passed.

## What to try next — explicitly a **separate behavior experiment**

- Quantify per-corner recurrence of the accepted α and test an **offline** warm-start of the line-search alpha ordering against archived hard-source states. Any alpha must still undergo the full exact source rollout, footprint/corridor and objective-decrease gate **before** publication. A last-cycle alpha hint is not a safety certificate. Search order can alter steering/throttle and lap time; never enable by default based on single-step simulation alone.
- Prefer low-variance reductions in merit-triggered backtracking from better SQP convergence/Jacobians over skipping safety checks. Prove that any new trial-order or curvature-aware search policy decreases **number of complete source rollouts** with no increase in incomplete laps, collisions, deadline misses or fallback chains. Re-run 28 difficult full-state replay cases, the 214 C++ tests, 6 reconstructed hard-state QPs, **five full practice races** and separate IROS competition-map tests.
- Most of the non-line-search exact source-plant step is unavoidable 1 ms native dynamics; a 4× reduction without changing semantics was not demonstrated. The validated FTZ source remains the current release until a better variant passes all gates. Current target remains <20 ms hard budget and preferably <17 ms maximum callback; the five FTZ practice races reached at most **18.293 ms**, not <17 ms.

## Source state and integrity

`mpc_vehicle_model_step` and source plant are restored exactly to the pre-experiment snapshots. No unqualified CMake flag is retained. After rollback, the production saved-state replay executable compared **byte-for-byte** with `/tmp/sdu_before_utilization_sqrt_replay_20261031` (which predates the unqualified experiments); 214/214 package C++ tests, all 28 prior full-state parity comparisons and `git diff --check` passed. The complete user optimizer and active monitoring processes were not modified. The only new checked-in-runtime-independent source change from this audit is the post-run CSV reporting addition. Work remains uncommitted in the existing dirty checkout; never reset unrelated work.
