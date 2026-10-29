# IROS 2026 MPC ten-lap recovery — 2026-10-29

## Verified result: one successful full-length trial, not yet repeatable

The first **supervised** competition-map MPC test completing warmup + **ten scored laps** is:

- `live_runs/iros_mpc_recovery_5mps_20261029_02/`
- 11 official laps, no reported collision or fatal MPC rejection, supervisor `SCREEN_COMPLETE: lap limit`, exit code 0, odometry distance 472.61 m
- Bag decoding: 10 scored laps totalling **96.991799 s**, **9.699180 s/lap mean**, fastest **9.567200 s**, slowest **9.840300 s**; warmup **11.988200 s**
- Controller-state cross-track RMSE **3.039 cm**, heading RMSE **0.03380 rad**, speed RMSE **0.3255 m/s**
- 4,374 measured MPC gate cycles; p50 **9.613 ms**, p95 **16.734 ms**, maximum **19.789 ms**, **zero >=20 ms**; 21 published retained-plan and 3 published prefix actions, none fatal
- Offline 750 ms replay with actual issued commands: endpoint XY RMSE **1.29 cm**, maximum **14.64 cm** and 52/4352 overlapping windows above 5 cm; this is not complete Unity physical parity
- Source-only reference: `live_runs/optimizer_iros_conditioned_20261029/exports/f8c5b636134d8b64/source_command_raceline_with_witness.csv` (7.400 s **source command duration**, not achieved Unity lap).
- Trial MPC overlay: `f1tenth_mpc/config/mpc_iros_recovery_5mps.yaml`, byte-identical to the recording's `mpc_overlay.yaml`, and separate from competition/practice defaults.
- All existing 5 cm *hard* wall-clearance constraints, controller fault/early-abort gates and permitted sensor streams are retained. Speed ceiling 5.0 m/s, 3.0 m/s startup ramp over one lap.

This establishes **one working conservative baseline**, not ten-lap repeatability across independent runs, not fast IROS qualification, and not proof of the source trajectory's advertised 7.4 s duration.

## Root-cause fix: launch parameter precedence

`sdu_apex_autodrive/launch/controller.launch.py` previously appended a hard-coded final `max_speed_mps: 16.0` composable-node parameter after the MPC YAML and overlay. That final parameter **silently overrode** `max_speed_mps: 5.0` in an otherwise correctly logged overlay. This was confirmed directly in the failed `iros_mpc_recovery_5mps_20261029_01` diagnostic: active speed ceiling 16 m/s and corridor fault after warmup.

Fix: the optional `controller_max_speed` launch argument defaults to empty, which preserves the final YAML's speed ceiling. **Only an explicitly supplied numeric** launch argument overrides the YAML; validate finite and range (0,16]. Verified with `tools/test_controller_track_profile.py` (3/3 passing) and fresh competition Docker build. The subsequent full-length trial's MPC diagnostics show the 3-to-5 m/s ceiling taking effect.

## Higher speed tests with identical weights/reference/physics

The controller is not yet fully reliable at higher speed:

| Run | Speed ceiling | Outcome |
| --- | ---: | --- |
| `iros_mpc_recovery_5mps_20261029_02` | 5.0 m/s | **Passed warmup + 10 scored laps** |
| `iros_mpc_recovery_6mps_20261029_01` | 6.0 m/s | After warmup, nonlinear corridor rejection at stage 2, progress ~105.79 m |
| `iros_mpc_recovery_7mps_20261029_01` | 7.0 m/s | After warmup, nonlinear corridor rejection at stage 7, progress ~124.25 m |
| `iros_mpc_recovery_5p5mps_20261029_01` | 5.5 m/s | After warmup, nonlinear corridor rejection at stage 2, progress ~117.38 m; measured plant speed ~5.94 m/s (above nominal ceiling due dynamics) |
| `iros_mpc_recovery_5mps_20261029_03` | 5.0 m/s | Independent repeat: warmup + **one** scored lap only, then nonlinear corridor rejection at stage 1, progress ~167.39 m; lateral error ~-0.40 m and rack near 0.52 rad |

Both higher-speed test failures were **exact MPC safety-gate rejections**, not proof of actual vehicle collision. Do **not** weaken collision constraints to force completion or interpret one 5 m/s pass as proof of a general stability bound. In the 6 m/s failure, controller lateral error was only about 3 cm; the predicted corridor becomes infeasible ahead of the car. At 7 m/s, errors accumulated and repeated retained-plan actions preceded failure. This supports corner-specific feasibility and recovery investigation, not simply another uniform weight increase.

## Reproduction

From repository root, when no other simulator trial is active:

```bash
SDU_APEX_IROS_TRAJECTORY_FILE=live_runs/optimizer_iros_conditioned_20261029/exports/f8c5b636134d8b64/source_command_raceline_with_witness.csv \
SDU_APEX_IROS_MPC_PARAMETER_FILE=f1tenth_mpc/config/mpc_iros_recovery_5mps.yaml \
SDU_APEX_IROS_TOTAL_LAPS=11 \
SDU_APEX_IROS_WALL_TIMEOUT_S=900 \
bash tools/run_iros_trial.sh UNIQUE_NEW_RUN_ID
```

Competition-image builds are normally enabled by the launcher. For a CPU-uncontended comparison, **pause the optimizer using its existing workspace-specific cgroup lease via `tools/optimizer_mpc_validation.py` / `yield_optimizer`**, then thaw it after trial/decoding. Never run uncoordinated simultaneous Unity simulations.

Use the original capture `live_runs/iros_mpc_recovery_5mps_20261029_02` for results; an unrecorded or uncapped trial is not evidence.

## Next engineering priority

1. Independent 5.0 m/s repeat **failed**; do not call the successful first trial consistently reproducible. Examine the new failure at progress 167.39 m and run several independently supervised trials before promoting any setting as reliable.
2. Inspect why 6.0 m/s becomes infeasible at progress 105–106 m when lateral error is only ~3 cm, including the QP linearized footprint constraint, actual 75 ms prefix corridor, command latency, actuator saturation and feasible steering authority.
3. Develop a **corner-conditioned, anticipation-aware speed envelope** rather than a uniform global cap. It must be physically realizable with the same steering/throttle and delay model, leave room beyond the 5 cm hard clearance for tracking errors, and be independently checked.
4. Raise speed using the same ten-lap supervised criterion, keeping the 5 m/s result recoverable. Only report new IROS lap-time improvement on full Unity/MPC completion.
5. The faster 7.125 s **source-only** optimizer witness is not a driven competition result. No change to the ongoing independent offline optimizer is needed just to preserve this baseline.
