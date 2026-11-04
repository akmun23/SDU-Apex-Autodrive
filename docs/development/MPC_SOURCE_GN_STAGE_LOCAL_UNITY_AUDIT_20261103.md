# Source-MPC Gauss–Newton and stage-local source Jacobians: implementation and Unity trials — 2026-11-03

## Scope

Goal: single source-derived 60-control / 30-stage nonlinear SQP for fast 40 Hz racing, keeping all 26 source-state coordinates, 22 hidden physical states, four actuator FIFO slots, exact 25 ms interval source integration, frozen original racing objective, hard physical footprint/rate rows and independent full-horizon nonlinear certification. The **20 ms ROS end-to-end budget**, inside the 25 ms command interval, remains unchanged.

These changes are **experimental opt-ins** and the established practice/competition controller defaults have NOT been replaced. Do not present five observed lap completions as five repeatable full races or Unity/source model parity certification.

## Work implemented

1. Full-source **Gauss–Newton PSD Hessian**, including steering/throttle inter-stage cross terms and residual Jacobians of 26-source physical state, trajectory tracking, speed, wheel slip, actuator queue, steering and throttle. The exact original objective gradient and exact nonlinear acceptance remain authoritative. No hard trust region is reintroduced. Lateral tire-slip reserve GN curvature remains outside the implemented sum-of-squares set; source objective gradients and acceptance still include original terms.
2. **Stage-local exact source differentiation**: at each of 30 nominal physical states, differentiate the ORIGINAL 25 ms Unity/PhysX-derived source plant with respect to 26 source-state coordinates and two commands. Evaluate each of the 30 local 26+2 physical Jacobians independently on available CPU cores; assemble the full 60-control source sensitivities causally through 26-coordinate local A/B matrices, with the original 3x6 incoming tracking-progress Jacobians, wheel slip and actuator FIFO. No reduced plant substitution. The direct full-horizon source-forward method remains available for parity checks; opt-in stage-local source flag controls selection.
3. Live source-mode **single publisher isolation**: a rejected native-source horizon cannot publish via legacy source reacquisition, short-prefix, viability recovery or retained-plan policies. The simulator collision stop remains separate. Deadline expiration in native source mode is not relabeled a solver failure: no late/unchecked command is issued, and the next coherent 40 Hz epoch reoptimizes.
4. **Actual command epoch regression guard**: a live callback with a candidate command epoch earlier than the last issued native source optimizer epoch is superseded, not solved from an unrelated cold reference seed. Run 05 showed an accepted epoch followed by an older epoch 0.354 ms backwards, causing a huge artificial Phase-I violation.
5. Preserve profiling on derivative deadline failure; diagnostic JSON exposes derivative wall time, sum of dual-plant worker time, tangent scoring worker time, completed sensitivity intervals, and optional near-feasible corrective QP counts.
6. Source-exact **necessary-condition prefilter**: a trial whose exact source objective/corridor/rate residual already makes acceptance mathematically impossible does not consume an additional full 30-stage strict verifier. Every potentially accepted trial still receives independent strict source revalidation.
7. Experimental near-feasible phase-I **constraint projection**, using the original exact-source footprint and rate constraints, with at most two additional QPs/exact rollouts subject to remaining budget; rank-one source-measured Broyden secant corrects local constraint Jacobian after large nonlinear steps. No relaxed physical hard bounds or alternate controller.

## Frozen-case numerical agreement and QP-cost fidelity

Saved 8 initially feasible normal-source states, 1 SQP iteration:
- Previous diagonal curvature: median |true racing cost minus QP predicted cost change| **14.053**, maximum **436.602**, median actual/predicted cost reduction **0.13**.
- Full-source Gauss–Newton: median absolute prediction error **3.837**, maximum **20.726**, median actual/predicted cost reduction **0.681**. Improved prediction error in 8/8 and actual nonlinear racing cost in 6/8; all eight stayed strictly source feasible.
- Model ratio on one GN case still **0.113**: this is not a perfect second-order cost model.
- Six high-slip hard-corner cases remain NONSMOOTH at sticky/contact branch transitions. Earlier full-source forward-vs-source central FD tests reported worst scaled gradient mismatch **1.697**; parallelization and local 26+2 tangent propagation do not resolve physical branch nonsmoothness or Unity/source mismatch.
- Direct versus stage-local source differentiation, 14 cases (six hard corners + eight moving):
  - maximum absolute 60-column objective-gradient discrepancy **0.73044** (on high-curvature nonsmooth case),
  - maximum full physical wall-J entry difference **0.00011921 m/normalized command**,
  - maximum rate-J difference **5.96e-9**, equivalent feasible/rejected status on all 14 cases.
- Six-core median single-solve benchmark: original direct full-source derivative **9.858 ms**, stage-local **7.365 ms** (not a live timing guarantee).
- Source-only 60 sequential 25 ms epochs with absolute **19 ms** C++ solver deadline, six P cores, strict independent 30-stage source validation: **60/60**, median **7.821 ms**, p95 **13.891 ms**, worst **16.055 ms** (early stage-local test). These are **not** complete Unity races.

## Unity runtime qualification results and remaining failures

All runs used the fixed unchanged practice trajectory `f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800.csv`, a six-P-core parallel source pilot, original physical model, ROS 20 ms end-to-end budget, strict full-horizon source certificate, and no legacy recovery actions.

| Run | Changes / outcome | Evidence |
|---|---|---|
| 01 | Invalid first overlay, 25 ms end-to-end budget rejected by constructor; corrected immediately | `live_runs/mpc_source_gn_pilot_20261103_01` |
| 02 | Parallel full-horizon sensitivities, not stage-local: 0 published commands; 2522 absolute-deadline deferrals | `live_runs/mpc_source_gn_pilot_20261103_02` |
| 03 | Instrumented startup: derivative wall time ~17.3 ms, sum of source workers ~31.1 ms; could not enter QP before deadline | `live_runs/mpc_source_gn_telemetry_20261103_03` |
| 04 | Stage-local: 77 source-certified publishes; failed to issue a late command near end of lap. No collision | `live_runs/mpc_source_stage_local_pilot_20261103_04` |
| 05 | 481 full-horizon source-certified publications and **two laps** (5.6558 s, 5.0468 s); a regressing control epoch caused cold seed and 1.416 mm future margin deficit at stage28; no collision | `live_runs/mpc_source_stage_filter_pilot_20261103_05` |
| 06 | Small-residual source projection: 101 accepted, one stage29 5.06 mm future deficit despite a source-accepted correction; no collision | `live_runs/mpc_source_stage_projection_pilot_20261103_06` |
| 07 | Continuous near-wall reserve cost 0.09m/12000 proved worse: early stage29 violation at 22.1m and 7.34m/s. **REJECTED EXPERIMENT; overlay removed**, original practice settings kept | `live_runs/mpc_source_clearance_pilot_20261103_07` |
| 08 | Original corridor weights, epoch guard, source corrections: **five consecutive completed laps**, 5.6928 / 4.9308 / 4.9208 / 4.9168 / 4.8758 seconds. 1,066 full-horizon commands, median end-to-end publish 8.52 ms, P95 15.31 ms, maximum 19.99 ms, seven deadline deferrals without legacy fallback and zero collisions. Requested sixth lap failed after a 3-stage warm start shift, stage24 remaining 4.04 cm horizon deficit; not fully race qualified | `live_runs/mpc_source_epoch_guard_pilot_20261103_08` |
| 09 | Expanded near-feasible 26-state source correction: 106 commands, stage29 1.47 cm residual after correction. No lap; not qualified | `live_runs/mpc_source_extended_phase1_pilot_20261103_09` |

Run 08 report: `/tmp/mpc_source_epoch08_report.json`, `/tmp/mpc_source_epoch08_report_diagnostics.jsonl`.
Run 05 report: `/tmp/mpc_source_stage05_report.json`, `/tmp/mpc_source_stage05_report_diagnostics.jsonl`.
Pure 14-case derivative comparison: `/tmp/mpc_source_secant_14case_ab_20261103.json`.
Cost/GN comparison: `/tmp/mpc_source_gn_ab_20261103.json`.
Source 60-epoch stage-local test: `/tmp/mpc_stage_local_source_closed_loop_60_20261103/results.json`.

## Remaining hard acceptance gates

1. Source Phase-I future-horizon feasibility when 2–3 command stages are shifted after deadline skips. One SQP pass may reduce a large footprint violation by >99% yet leave 1–4cm at stages 24–30. Without a valid complete horizon the native source mode currently withholds command (correctly: no false safety certificate); the supervisor regards that as a fault. Fix this **inside the normal nonlinear source SQP**, not by hiding failures in a second fallback controller.
2. Qualify source stage-local A/B derivative chaining across contact/slip branch changes and measure the discrepancy against actual Unity transitions, not just the same mathematical source plant. The high-slip source/Unity model accuracy gap remains.
3. Prove worst-case end-to-end 20ms across all phases of practice and competition with ROS contention and no repeated deadline misses; the 25ms scheduling interval is NOT the budget.
4. Run repeated complete 6-lap pilots, then repeated 10-lap fast practice races and the full competition map before disabling old controller or removing superseded implementations.
5. Add source-specific end-to-end cost model agreement diagnostics (QP predicted quadratic cost change versus independently scored accepted nonlinear source cost), including sparse corner wall defect; do not let good median GN ratio hide rare severe nonlinear branch discrepancies.

## Reproduction

```bash
cmake -S f1tenth_mpc -B build/f1tenth_mpc \
  -DSDU_APEX_MPC_PARALLEL_SOURCE_SENSITIVITIES=ON
cmake --build build/f1tenth_mpc -j6
(cd build/f1tenth_mpc && ctest -j6 --output-on-failure)
```

The explicitly selected pilot config is `f1tenth_mpc/config/mpc_source_gn_live_pilot.yaml`. The development Docker build needs `SDU_APEX_MPC_PARALLEL_SOURCE_SENSITIVITIES=true`; six physical P-core worker affinity can be selected by `OMP_NUM_THREADS=6 OMP_PROC_BIND=spread OMP_PLACES='{0},{2},{4},{6},{8},{10}'` with appropriate container cpuset. This CPU configuration is a measured host-specific experimental setup, not a portable production default.

**Release status:** experimental single-source-SQP, genuinely able to complete five practice laps, still NOT fully robust or Unity-model-error qualified. The immutable established practice profile is unchanged.
