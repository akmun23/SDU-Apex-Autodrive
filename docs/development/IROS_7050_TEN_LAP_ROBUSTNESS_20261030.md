# IROS 2026 record-chasing MPC: 7.050 s native raceline and ten-lap robustness audit

**Date:** 2026-10-30. **Status: experimental, ten-lap objective NOT met.** No competition/practice profile promoted. The optimizer's independent process and its existing uncommitted files were preserved.

## Official target, verified against organizers' leaderboard
Source: https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-iros-2026/ (Results, Phase 1 / Phase 2)
- IROS 2026 competition winner: Siga Siga Racing Team, **71.82s ten-lap race, 0 collisions, best lap 7.16s**.
- Competition second: NTU DeepSpeed, 72.26s race / best lap 7.19s.
- IROS 2026 practice qualification leader: NTU DeepSpeed, **49.83s ten laps, best lap 4.94s**.
- For full comparison, aim for >=10 **uninterrupted** actual simulator laps with no collision/control abort/missed 20ms publication, **race sum <71.82s** and **best lap <7.16s**; report standing-start ten-lap sum and after-warmup sum separately. A source-plant witness time, single lap or partial 10-lap run does not establish a record.

## Exact offline source witnesses
- `exports/83d2ebd3f02d1be7/source_command_raceline_with_witness.csv`: native PhysX **7.050000s**, 7050 samples, minimum sampled **extra** full-body clearance **+0.03471m above 0.05m**, max requested speed 11.871m/s, source-plant passing status. **Selected for initial robust MPC validation**.
- The chronologically latest 7.050s source export `exports/d823d0f07140067a` has just **+0.000389m** extra margin above the required 5cm. Its steering/throttle changes are smoother but it is less robust to localization/controller disagreement; it has also been tested as a distinct reference.
- Previous 7.075s source `exports/e75d1f49f552ec98` has ~+0.03070m extra margin; earlier achieved **one actual 8.1267s Unity lap** at the native speed profile, then failed on lap 2.
- **None** of the offline source witness exports claims a Unity closed-loop 7.05s lap or continuous swept-body certificate.

## Controlled full-source-speed ten-lap *requests* (one warmup + 10 scored target)
All runs used the source witness itself with configured max 16m/s (higher than the requested speeds), no artificial 5m/s or 6.5m/s limit; full body corridor, ROS localization, native source plant, and exact nonlinear rollout. A recorded `rejected_nonlinear_rollout` with no collision is *still a failed race*. Path progress here excludes initial unwrapped position; full loop about 43m.

| Run directory under `live_runs/` | Single changed factor vs weight4.5/5000/1700/speed45 critical policy | Progress advanced | Full laps | Deadline misses | Outcome |
|---|---|---:|---:|---:|---|
| iros_7050_margin11_critical_20261030_01 | high-clearance 7.050s baseline | 34.92m | 0 | 0 | corridor@14 |
| iros_7050_margin11_critical_20261030_02 | identical independent repeat | 18.73m | 0 | 0 | nonlinear rollout rejection |
| iros_7050_margin11_critical_20261030_03 | identical independent repeat | 23.27m | 0 | 0 | nonlinear rollout rejection |
| iros_7050_latest11_critical_20261030_01 | latest low-extra-margin 7.050s source export | 23.04m | 0 | 0 | nonlinear rollout rejection |
| iros_7050_margin11_mapcert_20261030_01 | full additional map world/body sweep every stage | 18.83m | 0 | **1** | map check too expensive, max 20.781ms |
| iros_7050_margin11_maprescue_20261030_01 | fast map check only on Frenet candidate rejection | 18.45m | 0 | 0 | no benefit in this trial |
| iros_7050_prefixcap4_20261030_01 | bounded exact-valid short-prefix count 4 instead of 2 | 18.60m | 0 | 0 | no benefit |
| iros_7050_prefixcap6_20261030_01 | bounded exact-valid short-prefix count 6 instead of 2 | 20.99m | 0 | 0 | no benefit, stage-zero rejection |
| iros_7050_overspeed35_20261030_01 | overspeed penalty from 20 to 35 | 35.31m | 0 | 0 | marginal improvement in this single run |
| iros_7050_overspeed50_20261030_01 | overspeed penalty from 20 to 50 | 18.33m | 0 | 0 | **physical simulator collision**, reject |
| iros_7075_critical_11laps_20261030_01 | prior 7.075s source, ten-lap requested | aborted early | 0 | 0 | rejected nonlinear rollout |

**Verdict:** exactly identical high-clearance 7.050s critical policy: **0 out of 3 independent full-lap completions**. It is NOT robust nor a record; other one-off candidate perturbations are not apples-to-apples repetitions.

## Engineering delivered under opt-in flags, not promoted
1. `f1tenth_mpc/include/mpc_world_footprint.hpp` (new): immutable occupancy-derived 2D exact Euclidean obstacle-cell-center transform and conservative lower-bound map distance, 0.510 x 0.273m rear-axle-centered body perimeter, densely sampled interpolated pose sweep. Treat unknown cells and map boundaries as unsafe; support ROS origin yaw/resolution.
2. `f1tenth_mpc/src/mpc_rti.c` + `include/mpc_rti.h`: optional physical occupancy sweep callback on exact nonlinear source-model rollout and objective validation. Ordinary QP/Frenet approximation unchanged.
3. `f1tenth_mpc/src/mpc_controller_node.cpp`: optional latched ROS OccupancyGrid subscription, one-time precomputation, allocation-free immutable grid in control callback, fail-closed until map loads, explicit extra uncertainty allowance.
4. Opt-in `world_map_authority_enabled` with `world_map_additional_uncertainty_m`; optional `world_map_rejection_only` to check the physical map only when conservative Frenet bound rejects. **Both new modes default disabled.** In map-authoritative mode all predicted source-step transitions are checked; in rejection-only mode existing fast Frenet remains binding on normal cycles and physical map is permitted to rescue *only* conservative false rejections.
5. `f1tenth_mpc/test/test_mpc_world_footprint.cpp`: synthetic unknown occupancy, in/out of map, real obstacle in body sweep and rotated map origin; compiler and full C++ suite **199 tests, 0 failures** in current ROS/Jazzy checkout.
6. Existing gated `viability_stage_targeted_recovery_enabled` and `viability_proactive_after_fallback_only` remain experimental; accepted full-horizon source-state risk scoring was corrected. All commands remain subject to exact physical actuator envelope and final source-plant rollout. No selected baseline or production config overwritten.
7. Full map sweeps were **not runtime-safe** at worst measured 20.781ms; opt-in map rescue kept <20ms in one run but did not solve the 7.05s track. DO NOT promote either variant yet. Interpolated endpoint sweep is **not** an exact continuous Unity internal-substep certificate; model uncertainty must remain accounted for.

## Key causal diagnosis
- A QP solve may converge (ADMM residual) and still predict a physically unsafe body/corridor at stage 14–27. Current MPC shuts down if no 750ms exact-valid horizon and no remaining bounded 75ms short-prefix gate.
- Several failed 7.050s cycles had 2/2 short-prefix actions just published; the next cycle was required to reacquire a full 750ms horizon and refused to check another otherwise potentially valid short action. Increasing the count to 4 or 6 worsened these individual IROS trials; repeated short-prefix replay previously led to collisions. **Never remove the gate without independently proven successor viability.**
- Earlier guarded practice `practice_stageaware_guarded_weight4p5_20261030_02` rejected at stage zero even though independent map audit measured about **5.03cm** minimum sampled body clearance along the ~9.2cm interpolated transition, above the *experimental* 2cm physical allowance. This is evidence of a false conservative Frenet rejection; requires cross-check with native physical swept substeps before changing live authority.
- At least two separate IROS faults were confirmed **genuine future physical map overlaps**. They must NOT be bypassed by weakening the corridor or allowing unchecked commands.
- Better record goals require preserving feasible **future** maneuvering (recursive viability) with the real queued controls, not just maximizing current speed. Both standing-start time and 40Hz source witness/200Hz actuator delay matter. Even assuming the 8.1267s standing-start lap followed by nine ideal 7.050s laps, the 10-lap sum would be ~71.577s—only ~0.243s below the official 71.82s record. Thus the 7.050s witness is barely fast enough without major tracking overhead; an even faster physically safe raceline may ultimately be needed.

## Immediate next engineering priority
1. Produce reproducible saved-state native source-plant replay at 7.05 IROS **future wall failure** (especially p~74m): determine legal control response, QP constraint inconsistency, actual reachable terminal set and earliest actuation that can preserve steering authority. Compare candidate, prior and witness in the **same exact state** (including the two-slot command queue).
2. Make deterministic short-cost full-horizon recovery a *lexicographically feasible* objective before tracking cost: if nominal is invalid, prioritize source-backed wall clearance and a safe terminal successor/backup, then restore time-optimal tracking. Exact wall, friction/steering limits and publication budget remain hard.
3. Make mapped body/sweep certificate more efficient and source-substep-faithful; validate fail-closed against the genuine IROS wall overlaps and practice false-corridor-reject state, then re-evaluate map authority on both maps under 20ms worst case. Do not claim map-pose interpolation equals continuous PhysX.
4. Demand **five independent full ten-lap competitions**, no collisions, no fatal faults, no deadline misses, all <71.82s if seeking to beat the official ten-lap result; separately capture fastest valid lap <7.16s. Also retain practice 10-lap regression <49.83s. No deployment promotion on a partial race.

## Artifacts
- Individual run dirs `live_runs/iros_7050_*`, with `watcher.log`, `mpc_run_report.json`, `mpc_cycle_detail.csv`, `mpc_cycle_detail_diagnostics.jsonl` and copied source profile.
- Experiment scripts and run JSONs: `live_runs/mpc_collision_margin_experiment_20261030/run_7050_*.py`; independent repeats in `iros_7050_identical_baseline_repeats.json`.
- Prior recovery analysis: `live_runs/mpc_collision_margin_experiment_20261030/FULL_HORIZON_RECURSIVE_FEASIBILITY_20261030.md`.
- Source export reports/CSVs: `live_runs/optimizer_iros_adaptive_regions_v6_20261030/exports/{83d2ebd3f02d1be7,d823d0f07140067a,e75d1f49f552ec98}/`.
