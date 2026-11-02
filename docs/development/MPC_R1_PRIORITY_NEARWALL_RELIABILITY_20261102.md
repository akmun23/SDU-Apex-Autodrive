# R1 solver priority and near-wall objective experiments — 2026-11-02

## Problem and provenance

All tests used the live Unity/ROS 2 30×25 ms exact source-plant MPC, two eligible SQP passes and the unchanged 20 ms publication gate. Race durations are **ten scored laps excluding the warmup lap**; each execution is an independent simulator start. The historical five-race qualified 49.1622 s mean practice profile remains protected. `f1tenth_mpc/config/mpc_practice.yaml` is experimental, not its replacement.

An earlier captured braking-envelope trial aborted at practice progress 277.786626m because an R1 Jacobian consumed 10.915 ms wall time while the controller reserved a speculative R2 time window, causing `deadline_before_admm`: zero ADMM iterations after only 13.041 ms callback wall time. A worker accumulated 2.317 ms of CPU work over 10.889 ms of stage wall time, indicating scheduling/preemption, not equivalent sustained CPU load. The optimizer never got an executable primary solution.

## Implemented primary solver priority

`mpc_rti_solve_cycle()` now grants R1 the actual cycle deadline, preserving its mandatory 3 ms post-solve validation/publication reserve. R2 remains implemented, relinearizes about the exact accepted candidate, and runs only when **measured** remaining cycle time can pay for nominal construction, Jacobian, ADMM and exact validation. Former code truncated the R1 deadline in advance using a guessed R2 reserve. This removal does not change solver tolerance, hard corridor limits, actuation, source physics, or emergency-collision policy.

Original 20 mm corridor, source-informed SQP, source nominal profile:

| Independent race | 10 scored laps (s) | Maximum publication gate (ms) | Nonfull recovery actions | Result |
| --- | ---: | ---: | ---: | --- |
| `forensic_r1_priority_source_practice_20261102_01` | 48.605 | 18.211 | 0 | completed |
| `forensic_r1_priority_source_practice_20261102_02` | 48.611 | 17.808 | 0 | completed |
| `forensic_r1_priority_source_practice_20261102_03` | — | — | — | **ABORT**, first-lap source corridor at stage 0 |

Prior source-informed original R1/R2 deadline run `normal_sqp_source_directional_20261101_01`: 48.961 s, one accepted viability recovery and one retained-plan action, maximum 18.260 ms gate. These are separate independent races, not seed-paired or causal timing proofs.

At failed race03 progress 27.560969 m, physical state `[e_y=-0.04472m,e_psi=0.00630rad,u=2.7858m/s,v=0.0242m/s,r=-3.3685rad/s]`, predicted first source step at progress 27.635811m had `e_y=-0.0432653m`, while physical 20 mm-margin lower corridor bound was `-0.0427518m` (**0.5135 mm violation**). R1 and R2 both completed, final ADMM primal/dual residuals 0.00672/0.01719; no accepted full-horizon or retained-plan prefix. Earlier certified-state prediction at the same progress differed in e_y by ~5.46mm. This is a nominal/reference/safety-robustness failure, **not** a solver deadline, ADMM nonconvergence or reported collision. Exact failure JSON `live_runs/mpc_forensics_20261102/r1_primary_practice_03_rejection.json`.

A naive hard-margin experiment increased both hard corridor margins from 20 mm to 25 mm while leaving zero nonlinear tolerance and full source speed. It completed 48.761 s and 48.791 s races, then aborted on its third independent start near 27.407 m, again at corridor stage 0 with 1.728 mm violation. **Increasing the hard margin alone failed repeatability**; retain the zero-tolerance original safety contract, do not promote the stricter 25 mm profile.

## Opt-in frozen near-wall lateral objective target

New `near_wall_tracking_reserve_m` defaults to **0**. For each physical source nominal state after stage 0, compute exact footprint corridor slack with `mpc_rti_compute_corridor_slack`, excluding any nonlinear-rollout tolerance. If either side's slack is below the reserve, shift the frozen lateral objective target away from that side by the deficit (bounded to ±reserve). This is a *soft objective gradient* intended to stop the normal MPC systematically consuming the last few millimeters of available clearance; all existing hard physical constraints remain untouched. The same schedule is supplied to both SQP passes and to exact nonlinear objective scoring. Regression `MpcRtiObjective.NearWallBiasIsBoundedFrozenAndInactiveAwayFromWall` verifies both wall signs, no bias when far from walls, and exact opt-in-off parity. Targeted tests and full ROS colcon release build passed.

Candidate overlay `live_runs/mpc_forensics_20261102/practice_source_nearwall_25mm.yaml`: source-informed SQP plus objective-only reserve 25mm, original 20mm hard corridor margin and 0 nonlinear tolerance.

First independent race `forensic_nearwall25_practice_20261102_01` completed ten scored laps in **48.629 s**, zero nonfull actions, zero publication misses, max gate **17.946 ms**, p99 **16.783 ms**. Subsequent independent repeats must be recorded before drawing a reliability conclusion.

## Remaining release blockers

1. Five consecutive independent practice races, ten scored laps each, **all <49.0 s**, zero collision/unpublished MPC rejection, deadline misses and preferably zero nonfull actions.
2. Reproduce the stage0 corridor infeasibility with true plant state plus actuator-event timing; quantify source model vs Unity uncertainty and require proactive clearance before the queued control reaches an unrecoverable boundary.
3. Validate R1 priority improvement under diverse worker-preemption loads, worst-case under 20ms, preferably ≤15–17ms. Do not hide missed publication deadlines.
4. Resolve locally LP-infeasible competition cases and source-nonlinear corner mismatch without weakening the 50 mm safety clearance, then five strict full-speed competition ten-scored-lap races. The earlier ~101s competition series allowed an extra 20mm nonlinear tolerance and had 1/5 deadline failure, so is NOT qualified.
5. Preserve the historical qualified practice profile and all forensic data until replacement passes the specified gates. Do not reset or discard other staged/unstaged work.

## Completed five-race qualification — source-informed R1-priority near-wall objective

The overlay `live_runs/mpc_forensics_20261102/practice_source_nearwall_25mm.yaml` used near-wall soft reference reserve 25 mm, original **20 mm hard footprint corridor**, zero nonlinear corridor tolerance, source-informed SQP and ordinary production 20ms publication constraint. No actuator or collision limit was loosened.

| Independent race | Ten scored laps (s) | Max publication gate (ms) | Nonfull actions | Result |
| --- | ---: | ---: | ---: | --- |
| `forensic_nearwall25_practice_20261102_01` | 48.6290007 | 17.9455 | 0 | PASS |
| `forensic_nearwall25_practice_20261102_02` | 48.6919999 | 18.2152 | 0 | PASS |
| `forensic_nearwall25_practice_20261102_03` | 48.6190000 | 18.2792 | 0 | PASS |
| `forensic_nearwall25_practice_20261102_04` | 48.6649995 | 17.9869 | 0 | PASS |
| `forensic_nearwall25_practice_20261102_05` | 48.7640009 | 17.9817 | 0 | PASS |

**Exact mean 48.6738002s**; all five <49s, total 50/50 scored laps, 0 reported collisions, 0 rejected unpublished cycles, 0 20ms gate misses, 0 accepted retained-plan/short-prefix/viability-recovery actions. Strict epoch-aware qualification tool `tools/qualify_mpc_race_series.py` emitted `live_runs/mpc_forensics_20261102/practice_nearwall25_5x10_qualification.json` with `all_qualification_gates_pass:true` and `identical_image_profile_and_trajectory:true`. Maximum publication gate **18.279205ms**, versus the hard 20ms ceiling. This is **practice-qualified on the tested image, exact track and hardware**, not a guarantee of worst-case scheduling under other loads.

The exact package release build completed and `colcon test --packages-select f1tenth_mpc` reported **249 tests, 0 errors, 0 failures, 0 skipped**.

Minimum predicted corridor slack from the first three near-wall runs was about 24.56 / 29.85 / 26.32 mm beyond the hard footprint margin, versus 16.46 mm in one former fast but unqualified original-margin run. This is descriptive evidence, not a new mathematical constraint or guaranteed reserve. A small number of late source-regime variations can still violate future recursive feasibility; particularly stage0 actuator-queued conditions remain outside full control. The qualification did not verify full Unity-model fidelity or strict cross-track competition margins.

For production deployment preserve the overlay and complete artifact manifest alongside the installed binary; DO NOT assume that changing the default config file or rebuilding with different source/settings is the same qualified executable. The historically qualified October practice profile remains archived rather than removed.

## Competition speed transfer and the still-open recursive-feasibility blocker

The competition experiment deliberately used the native 6.925s source-witness line, **hard corridor 50mm**, **nonlinear tolerance 0**, nearwall soft objective 40mm, source-informed SQP, six Jacobian workers (to reduce oversubscription of the six physical cores allocated by `cpuset 0-11`). Overlay `live_runs/mpc_forensics_20261102/competition_source_nearwall_40mm_strict.yaml`.

- Two-lap screen `forensic_nearwall40_competition_strict_20261102_01` completed the warmup plus **one scored 7.0417s lap**, no collision, no missed publication gate, max gate **18.773945ms**, 13 retained-plan and 3 validated short-prefix actions. A speed result, **not robust / fallback-free competition qualification**.
- Ten-scored-lap attempt `forensic_nearwall40_competition_strict_20261102_10lap_01` failed after the warmup near global progress 100.419m. Next exact 25ms physical source step at progress 100.485m crossed the **upper corridor bound by 4.538mm** (`e_y=.014992m`, upper `.010454m`); the measured state already held queued actuator steering/throttle. Both SQP passes attempted with 0 exact accepted plans and all retained-prefix revalidations failed. Last solver `admm_stop_reason=time_budget`, primal/dual residual `.04458/.00674`, callback 14.73ms. This is genuine local **stage0 shortfall**, not a 20ms missed publication deadline or Unity collision. Full fault `live_runs/mpc_forensics_20261102/competition_nearwall40_long_run_01_rejection.json`.
- Stronger near-wall reserve **80mm** at identical strict geometry failed **earlier**, near competition progress 59.122m with next-step `e_y=-.240832m` vs lower `-.224957m` (15.874mm deficit), steering saturated to 0.523599 rad, 300 total iterations, primal residual `.4743`. Fault `competition_nearwall80_long_run_01_rejection.json`. Simply escalating a geometric soft bias does not solve the nonlinear entry dynamics.
- Curvature-aware loaded-reference speed cap `ay=6m/s²`, braking `6m/s²`, 40mm soft reserve, otherwise strict geometry failed at progress 80.329m (stage0 upper corridor deviation ~2.555mm) before a complete scored lap. The speed envelope reduced 5627/6925 waypoints; logged `source witness invalidated=true` because transformed speed is no longer the original PhysX-certified optimizer command witness. Fault `competition_nearwall40_ay6_rejection.json`. Do not present this as a faster certified source reference or a reliable competition controller.

**Interpretation:** The practice result is qualified, but competition physical dynamic margin near high curvature is still not recursively feasible in the current normal formulation. Multiple failures occur at stage0 of the *upcoming 25ms source step*, when the actuator FIFO already fixes steering. More ADMM iterations, harder cost weights or after-the-fact margin shifting are not capable of correcting an already-queued physically unavoidable violation. Priority is earlier physical state-and-command-queue-aware viability constraints, source-vs-Unity one-step error analysis in those high-curvature regimes, then strict full-distance race qualification. Do not change the verified practice profile to cover up this independent competition defect.

### Cross-cycle physical forecast discrepancies (do not equate with proven model error)

At failure epochs, compare the previous certified plan state closest in path progress to the newly repropagated **first next-step** source state. The following are *forecast-to-new-estimate differences* spanning controller updates; they can include plant integration, localization, sensor timestamp, control FIFO and unmodeled Unity effect. They are NOT a controlled one-step source-vs-Unity identification residual:

| Failure | Forecast/current stage e_y difference | Heading difference | Speed difference | Other evidence |
| --- | ---: | ---: | ---: | --- |
| Practice original 20mm margin, 27.636m | -5.462mm | -0.00273rad | +0.0395m/s | corridor lower shortfall .514mm |
| Competition 40mm nearwall objective, 100.485m | +5.366mm | -0.00932rad | -0.0607m/s | corridor upper shortfall 4.538mm; retained-plan cursor 2 |
| Competition 40mm nearwall plus ay6, 80.391m | +6.331mm | -0.00171rad | +0.0027m/s | corridor upper shortfall 2.555mm; cursor 4 |
| Competition 80mm nearwall, 59.221m | -177.66mm | not comparable to current near wall | — | **retained-plan cursor 23**, large stale-forecast deviation over ~23 prediction increments |

The next engineering test must synchronize *actual source-plant issued-command events*, wheel/suspension hidden state, map pose and sensor stamps and replay the exact previously projected next step. Before introducing new robust bounds, separate actual dynamic-model error from stale nominal-state/actuation-epoch mismatch. The replay should quantify position, yaw and body footprint clearance jointly, not just lateral tracking error. A large cursor 23 retained-plan episode is unacceptable as evidence of normal robust convergence: optimize successor feasibility earlier instead of accumulating stale fallback commands.
