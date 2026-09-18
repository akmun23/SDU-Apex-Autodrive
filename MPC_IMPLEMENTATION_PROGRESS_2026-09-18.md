# MPC implementation continuation note — 2026-09-18

## Current objective and constraints

Continue the 2026-09-18 MPC coding-agent handoff and finish implementation before attempting to make the car drive. The current goal is a complete, internally consistent MPC with evidence-backed offline readiness; live driving is not yet assumed safe or required.

- Do not edit Unity, simulator physics, scenes, or simulator behavior.
- Do not use timing jitter as a rejection or launch blocker. Keep timing as diagnostic evidence. Investigate delay if a collision occurs.
- Keep runtime inputs within the competition-legal interface; simulator truth remains offline-only.
- Preserve 40 Hz / 25 ms prediction, N=30 (0.75 s), adaptive rho, and the existing 100-iteration cap. Do not tune weights before the model/diagnostic work is complete.
- When a later live test is justified, use the ICRA image in batchmode (not `-no-graphics`). Do not start a GUI.
- The accepted working raceline is `f1tenth_planning/trajectories/autodrive_mintime_exact/autodrive_mintime_raceline.csv`.

## Work completed in this session

1. Read the supplied coding-agent handoff and carried forward its ordered priorities: isolate the obsolete solver, build rejection diagnostics, verify CT2 semantics, analyze existing applied-command replay before gathering more data, implement adaptive same-sample RTI, instrument adaptive rho, and only then optimize/tune.
2. Added an MPC-only AMCL warm-up delay in `sdu_apex_autodrive/launch/controller.launch.py`. Default is 2 seconds from the AMCL process-start event; AMCL keeps processing scans during the delay. `mpc_start_delay_sec:=0` disables it. Pure Pursuit/FTG do not use this delay. If localization is explicitly disabled, MPC launches without waiting for AMCL. Launch behavior has only had Python syntax validation, not a live launch test.
3. Removed `f1tenth_mpc/src/mpc.c` from production `mpc_core`. It is now built into a `BUILD_TESTING`-only compatibility library, used by the old regression test/benchmark; `mpc.h` is excluded from installation. Updated `f1tenth_mpc/README.md` to describe that boundary.
4. Began the RTI implementation in `f1tenth_mpc/include/mpc_rti.h`, `src/mpc_rti.c`, `include/riccati_solver.h`, and `src/riccati_solver.c`:
   - Added R1/R2/adaptive mode and candidate/trigger/solver telemetry structures.
   - Added an exact nonlinear candidate objective/corridor-slack scorer, R1 pass helper, and adaptive-trigger calculations.
   - Added rho start/change-count fields and enabled Riccati profiling in the core build.

## Important: unfinished implementation state

The new R1/R2 helper code is **not wired into `mpc_rti_solve_cycle()` yet**. The build reports the two expected `-Wunused-function` warnings for `solve_rti_pass` and `adaptive_rti2_trigger_mask`. No R2 pass currently runs, none of the new per-pass fields are populated/published, and the YAML/node do not yet select RA or configure trigger thresholds. Treat adaptive R2 and its telemetry as unimplemented until those tasks are completed and tested.

There are no new tests for these helper functions, R2 fallback/selection, the AMCL launch delay, or rho telemetry. The passing suite below exercises the pre-existing R1 path only.

## Validation performed

- `git diff --check`: passed.
- `python3 -m py_compile sdu_apex_autodrive/launch/controller.launch.py`: passed.
- Humble container build: `colcon build --packages-select f1tenth_mpc --merge-install`: passed, with the two unused-helper warnings described above.
- `ctest --test-dir build/f1tenth_mpc --output-on-failure`: 9/9 passed.
- No simulator or controller launch was started for this session. The only active container observed was the development container `sdu_apex_autodrive`.
- No current-session ROS 2/Python test suite beyond the MPC CTest suite was run.

## Earlier evidence to retain (not new validation)

- Pure Pursuit ran the accepted mintime raceline in ICRA batchmode. Initial complete lap splits were about 12.20, 12.09, and 12.26 seconds; later laps slowed. Four collision-count increments occurred. One was close to a long response-arrival interval; the other three had ordinary intervals. Existing packet logs cannot prove which applied command/physics frame caused each collision. Do not describe the PP run as collision-free acceptance.
- An earlier MPC authority probe remained stationary: RTI rejected nonlinear rollout at stage 24, followed by residual rejections. This was not a timing rejection. No further MPC live run has been made after the instruction to finish implementation/offline readiness first.

## Next actions, in order

1. Finish integrating the new pass helper into `mpc_rti_solve_cycle()`:
   - Run R1 once, using the existing one-time warm-start shift.
   - For R2/RA, construct the second nominal from R1's exact nonlinear candidate states, controls, and progress; resample references from that progress. Do not call `mpc_rti_build_nominal()` again or shift the horizon.
   - Snapshot/restore solver and nominal memory so an invalid R2 falls back to a valid R1 without losing R1's warm start.
   - Compare only feasible candidates by nonlinear objective; use minimum corridor slack as a near-tie-breaker. Preserve R1 when R2 is infeasible/worse.
   - Populate R1/R2 status, trigger mask, objectives, slacks, actions, iterations, per-pass/total solve duration, selected candidate, `u*abs(r)`, rho changes and factorization telemetry.
2. Review the just-added configuration validation with the new RA/R2 path; keep R1 compatibility intact. Add deterministic tests for R1 parity, same-sample R2 (no double shift), forced R2, R1 fallback, candidate selection, triggers, and telemetry.
3. Add node/YAML parameters for `rti_refinement_mode` (initial production mode RA only after tests), trigger thresholds, and diagnostic serialization. Keep timing out of all reject/stop logic. `rti2_budget_skipped` must not become a timing-based rejection gate.
4. Implement the handoff's live rejection analyzer (CSV/JSON/Markdown, stratified bins, >=95% rejected-cycle categorization) and emit all required raw quantities from shadow diagnostics. Do not adjust thresholds or weights before this report exists; derive RA thresholds from recorded development data and verify against an untouched holdout.
5. Audit CT2 timestamp semantics and run the existing applied-command replay/decomposition before requesting new data. Retain user-visible distinction between source time, ROS command time, callback/synchronization steady time, and offline scoring truth.
6. Complete static build/tests for affected packages and verify launch arguments in the Humble container. Only then decide whether a controlled live MPC test is warranted. Do not claim MPC driving readiness without a collision-free multi-lap result.

## Worktree handling

The worktree was already broadly modified before this session; retain all user changes. In particular, preserve the user's deletion of `f1tenth_planning/trajectories/autodrive_mintime_exact_v13.zip` and all simulator-trace artifacts. Do not reset/clean the repository. Current-session edits are concentrated in the seven MPC/launch files listed in this note; broader localization, bridge, actuator, and PP edits are pre-existing task work and must also be preserved.
