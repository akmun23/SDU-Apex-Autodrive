# Balanced MPC racing validation — 30 October 2026

## Non-negotiable requirements

- **Practice**: keep `f1tenth_mpc/config/mpc_practice_fastline.yaml` unchanged until an independent candidate repeatedly completes **one warmup plus ten scored laps**, with no collisions, fatal faults, or publication overruns; required total below 49.8 s; stricter average below 4.95 s (equivalent to ten-lap total below 49.5 s). These thresholds are NOT equivalent.
- **IROS competition**: follow `live_runs/optimizer_iros_adaptive_regions_v6_20261030/exports/e75d1f49f552ec98/source_command_raceline_with_witness.csv` at its **uncapped native speeds**, not a 5/6.5 m/s artificial global ceiling. The source witness is certified to 7.075 s in native PhysX and has approximately 2.98 cm extra minimum footprint clearance above its requested 5 cm clearance. It is NOT a proven 7.075 s autonomous Unity lap.
- Real-world physical footprint and source-plant rollout validations remain mandatory. Do NOT trade wall penetration for a fast score.
- Lower heading/lateral tracking error is not by itself an improvement: high weights can force needless deceleration. Evaluate lap-time total, variability, actual speed undershoot, steering/throttle oscillations, minimum body clearance, recoveries, and publication timing together.
- Avoid overlapping Unity trials with the concurrently running optimizer. Use `tools.optimizer_mpc_validation.yield_optimizer` for exclusive ownership and automatic resume. Preserve unrelated staged/uncommitted work.

## Verified closed-loop IROS experiments

- New 7.075s raceline, 5 m/s cap, physical corridor 2 cm: ten scored laps, 9.47849 s mean, no collisions/faults/deadline violations. This demonstrates driveability at capped speed, NOT the desired competition performance.
- 6.5 m/s cap with optional curvature+braking speed envelope, 8 m/s^2 lateral acceleration and 5 m/s^2 deceleration: first run ten scored laps at 8.45269 s mean, zero misses/collisions, **independent repeat failed after warmup**. Do not promote. In the failure, car exceeded 9 m/s while reference was 6.5 because the ceiling is a cost target, not a hard physical cap.
- Uncapped 7.075s source witness with 2 cm corridor: failed at station ~34.32m (progress 77.30m) in first/second lap transition with source-plant corridor@0; independent world-pose footprint audit reported an obstacle-region overlap (~-0.0227m map clearance). MPC ADMM residual was 0.488, capped at 150 iterations, and no revalidated prefix remained. This is a real safety/control deficit; changing the speed reference alone did not cure it.

## Verified practice measurements

All totals below represent ten scored laps, excluding warmup:

| Setup | Total [s] | Mean [s] | Deadline misses | Collision | Notes |
|---|---:|---:|---:|---|---|
| Original 2cm fastline, prior build | 49.602 | 4.9602 | 1 | no | under 49.8 total, misses stricter 4.95 mean |
| Restored no-recovery repeat | 49.997 | 4.9997 | 0 | no | startup variation, not target |
| Default viability recovery | 50.471 | 5.0471 | 0 | no | preserved normal config |
| Stronger tracking 7500/2500, speed weight 50 | 51.004 | 5.1004 | 2 | no | speed gets worse |
| Optional 2cm/0.015rad tracking tolerance | 50.964 | 5.0964 | 0 | no | speed undershoot rose 0.179->0.274m/s, steering step p95 0.0505->0.0794 rad |
| Weight_u 42, unchanged other terms, no recovery | 49.884 | 4.9884 | 1 | no | improves total 0.113s relative 49.997 baseline, but not target or clean |
| Weight_u 45, underspeed_curvature_gain_m 0.6, no recovery | **49.578** | **4.9578** | **0** | **no** | first clean under-49.8 run; **repeatability pending** |

Source profiles and run logs live under `live_runs/mpc_collision_margin_experiment_20261030/`, exact run outputs under `live_runs/<run_id>/`, and `mpc_run_report.json` + `mpc_cycle_detail.csv` inside run folders. The 45/.6 profile is `live_runs/mpc_collision_margin_experiment_20261030/practice_speed_priority_45_curvature_discount.yaml`; Unity run `practice_speed_priority_curvature_20261030_01`. Its independent `_02` began as this memo was written; use actual report rather than inferring results.

## Code created, default-disabled experiments

- `f1tenth_mpc/include/mpc_speed_envelope.hpp`: cyclic curvature & braking speed envelope precomputed at trajectory load. Disabled by default (`curvature_speed_envelope_lateral_accel_mps2=0`); invalidates native source witness when speeds change. It has one successful and one failed 6.5m/s IROS run; not a demonstrated solution.
- `f1tenth_mpc/include/mpc_tracking_tolerance.h`: optional convex squared dead-zone for lateral/heading error, with local QP target, exact nonlinear merit and zero-tolerance backward-compatible behavior. Exposed as `tracking_lateral_tolerance_m` and `tracking_heading_tolerance_rad`, both 0 by default. The 2cm/0.015rad Unity experiment was **slower**, so do NOT enable by default.
- `f1tenth_mpc/test/test_mpc_tracking_tolerance.cpp`: zero-tolerance identity, inside dead-zone, exact derivative, independent heading/lateral checks. Total package **185 tests, zero failures** (ROS Jazzy build).
- `tools/audit_mpc_race_quality.py`: summarizes full Unity race time and control quality, and `--enforce-repeatability` demands at least 3 distinct, clean, faster-than-both-threshold runs. `tools/test_audit_mpc_race_quality.py`: two passing tests including prevention of counting duplicate runs.

## Next engineering sequence

1. Finish and report independent ten-lap repeats for the 45/.6 curvature-weighted practice candidate. Do not commit or promote it on a single run. If it meets only the 49.8s threshold but not 49.5s, state that precisely.
2. Do not destroy the protected practice profile when modifying competition; run cross-map regression after any core MPC source change.
3. On competition, diagnose source witness mismatch vs actual initial condition, actuator delay, ADMM first-step residual, and receding-horizon wall clearance. The native 7.075s witness is an **open-loop** optimized trajectory, not a closed-loop guarantee. Focus recursive feasibility, source-plant 750ms horizon and anticipatory physical clearance before speed-weight tweaks.
4. Promote only after repeated independent full competition races and three clean ten-lap practice trials below both strict thresholds (unless user relaxes the 4.95 mean goal).

## Independent repeated practice result (updated)

The under-49.8s candidate **weight_u=45.0, underspeed_curvature_gain_m=0.6, viability disabled** was tested three times with the **same** YAML (`run_id` suffix `_01`, `_02`, `_03`):

| Independent run | Ten scored lap time | Deadline misses | Result |
|---|---:|---:|---|
| `_01` | **49.578s** (mean 4.9578s) | 0 | completed cleanly, under 49.8s but not under 4.95s mean |
| `_02` | **49.397s** (mean 4.9397s) | 0 | completed cleanly, under BOTH speed thresholds |
| `_03` | incomplete | 0 before fault | MPC fault `rejected_nonlinear_rollout; corridor@0`, at 97.351m progress (lap station ~17.560m); body-u 6.422m/s, e_y +0.3314m, heading error 0.0193rad, no collision-free full-horizon/candidate prefix. Independent occupancy map reconstructed vehicle footprint with -0.0227m map clearance (within obstacle region) |

The second clean result does **not** establish repeatability, because the third trial failed physically. Do NOT promote this configuration or use the two fastest lap times as proof of robustness. Full-viability-recovery variant launched separately and completed all scored laps but reported two 20ms publication misses; also not immediately promotable. Another independent test of minimal near-wall recovery is running from a separate validator, so never run overlapping live simulations.

The map geometry auditor's third-failure output is `live_runs/mpc_collision_margin_experiment_20261030/practice_45_gain06_repeat03_failure_geometry.json`. The profiler's first two runs summary is `live_runs/mpc_collision_margin_experiment_20261030/practice_45_gain06_repeats.json`.

## Failure-chain root cause and solver audit (12:29 local)

**Where/when**: fastest practice config `_03`, lap station 17.560m, progress 97.351m, 6.422m/s, e_y=+0.331m, map signed footprint clearance -0.0227m. Fatal exact nonlinear corridor rejection at stage 0. The actual scored run has a physical collision at approx 30.818s; fault cycle was 14.266ms and maximum recorded callback-to-gate 19.167ms, zero deadline misses. This is **not a deadline-caused failure**. Uncapped IROS witness at station 34.32m similarly failed with physical map-boundary penetration and ADMM residual 0.488.

**How**: in preceding 1.5s: 31 full-horizon acceptances, **28 short candidate/retained-plan prefix publications**. In progress interval 94-95m **11/11 actions were fallback**, 95-96m 8/9, 96-97m 7/7, 97-97.4m 2/3. In this span +e_y grew from approx 0.18m to 0.33m while speed rose from approx 4 to 6.5m/s; most fallback commands had positive throttle (~0.18-0.31). The replay confirms the short ~25-75ms validator is NOT a recursive viability certificate. Fast `_02` (49.397s, no faults) had only **two fallback actions over the entire race**, versus **37 fallback actions** on failed `_03` before abort. Difference supports fallback chain as failure mechanism, not proof of sole causality.

**Why solver failed**: last practice cycle ran 2 SQP passes with 242 total ADMM iterations, returned time_budget with primal residual 0.647, dual 0.217 and rejected full-horizon corridor at stage 0. IROS failed with 150 iterations, cap and residual 0.488/0.123. Both ran under the publication budget but had poor numerics/feasibility; constrained SQP linearization near boundary and nominal controls inherited from a repeatedly recycled short prefix are likely major contributors. Need replay per-QP residual trajectory and station-specific oracle before asserting exact numerical root cause.

**Cold-start facts**: `try_publish_quality_aware_prefix` already invokes `mpc_rti_memory_reset_solver_state()` on each selected fallback and preserves only a *composed nominal control warm start* (`compose_candidate_prefix_warm_start` / `copy_retained_plan_warm_start`). Thus “cold-start ADMM after prefix” is ALREADY ACTIVE and insufficient. A FULL nominal cold start is separate and may destroy useful source-witness seeds, increase computation and destabilize. Do not indiscriminately clear both memories or claim it will solve the issue.

**Timing/CPU**: accepted-cycle callback-to-gate median in failed fast `_03` 10.511ms, p99 18.414ms, max 19.167ms, and in successful `_02` median 9.598ms, p99 18.635ms, max 19.577ms. Nominal build medians ~0.7ms and QP+Jacobian stage wall medians ~2.2ms, p99 ~4.6ms. Evidence of isolated >20ms overruns on other configurations exists, but _03 failure was not timing. Hard real-time **guarantee cannot be made** from median/p99 tests on non-real-time Linux; requires preallocation, WCET tests under CPU contention, guaranteed reserve before ADMM and non-blocking publish.

**Fallback utility and repair requirement**: retain a safety-validated prefix for **isolated** solver misses; do not permit long chains where new action only passes 25-75ms, lacks a 750ms viable successor, and the state drifts toward a wall. A future quality gate should require (a) validated **full-horizon** continuation or a proven controlled safety invariant/terminal set, (b) non-decreasing sufficient wall reserve adjusted for speed and model uncertainty, (c) explicit divergence trend and fallback-chain limit, and (d) no positive acceleration when a predicted unsafe barrier is approaching, verified with the *actual source plant* and map—not merely clamp throttle and hope. If no physically safe plan exists, flag a genuine unrecoverable state; do not claim a fallback fix. The current 16-candidate tapered recovery and optional steady bridge are not reliably repeatable; must test independently.

Replay and geometry artifacts: `live_runs/mpc_collision_margin_experiment_20261030/practice_fast03_failure_chain.json`, `practice_45_gain06_repeat03_failure_geometry.json`, `practice_45_gain06_repeats.json`, `iros_uncapped_witness01_geometry.json`.
