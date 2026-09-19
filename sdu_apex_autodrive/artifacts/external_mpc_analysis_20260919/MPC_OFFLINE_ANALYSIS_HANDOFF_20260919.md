# MPC offline-analysis handoff — 2026-09-19

This folder is a curated evidence package for external analysis of the AutoDRIVE
MPC failure. It was assembled from existing simulator captures and offline
replays. No simulator, Unity, physics, scene, sensor, or timing-source code was
modified while collecting this package.

The package intentionally distinguishes runtime-legal inputs from simulator
truth. `/odom`, `/ekf_odom`, `/current_map_pose`, `/cmd/speed`, IMU, encoders,
and the accepted raceline are the inputs available to the production controller.
`gt_odom.csv`, bridge packet fields, collision counters, and simulator pose
fields are included only for offline diagnosis and must not be used by the
controller.

## Question to solve

The exact raceline does not command speeds near 16 m/s. Why did MPC previously
produce a high target-speed command, and why does MPC still fail to complete a
lap after the target-speed correction?

## Reference raceline

File: `01_reference/raceline_autodrive_mintime_exact.csv`

Observed profile:

| Quantity | Value |
|---|---:|
| Samples | 313 |
| Approximate lap length | 53.106 m |
| Minimum speed | 2.3505 m/s |
| Median speed | 6.1116 m/s |
| p95 speed | 9.7284 m/s |
| Maximum speed | 10.1607 m/s |
| Maximum acceleration | +2.9791 m/s² |
| Maximum braking | -7.6049 m/s² |
| Samples above 12 m/s | 0 |
| Samples above 14 m/s | 0 |

Using the identified Unity-native longitudinal response model and the raceline
acceleration profile, the inverse target-speed feedforward has an approximate
maximum of 10.52 m/s. This is still well below 16 m/s and is consistent with
the raceline. A 16 m/s target was therefore not coming from the raceline.

## Implemented correction before these tests

The working tree contains the P0-A correction from the latest MPC handoff:

- raceline acceleration is parsed and interpolated;
- the longitudinal target state is given a direct raceline-derived reference;
- the target reference uses the accepted Unity-native inverse response model;
- the target-speed state has a configurable direct cost, currently `20.0`;
- fallback speed selection is capped by the local raceline speed instead of
  taking the maximum of observed and requested speed;
- diagnostics expose raceline speed, acceleration, rate, and target feedforward.

Relevant implementation files are:

- `f1tenth_mpc/src/mpc_rti.c`
- `f1tenth_mpc/src/mpc_controller_node.cpp`
- `f1tenth_mpc/src/mpc_reference.c`
- `f1tenth_mpc/include/mpc_rti.h`
- `f1tenth_mpc/include/mpc_reference.h`
- `f1tenth_mpc/config/mpc_autodrive.yaml`

The build passed and all 9 MPC package tests passed after this correction.
The production settings remain N30 at `dt=0.025 s`, 100 solver iterations,
adaptive rho, and Riccati prefactorization.

## Dataset descriptions

### 1. Clean PP exact-raceline baseline

Folder: `02_pp_exact_baseline/`

Source: `pp_new_mintime_exact_20260918`.

- 9982 `/odom` rows, 9982 `/ekf_odom` rows, and 9975 `/current_map_pose`
  rows.
- 9982 `/cmd/speed` rows.
- 9982 bridge packets.
- Recorded wall duration: 256.894 s.
- `source_duplicate_or_reverse_count=0`.
- `source_gaps_over_30ms=0` in the stored timing report.
- The timing report has no Unity source-time fields for this run; ROS headers
  use the bridge host receive clock according to the manifest.
- `gt_odom.csv` and bridge packet fields are diagnostic-only truth.

This is the clean reference for comparing the legal state and controller
behavior. The separate 2026-09-19 PP authority check on the same exact
raceline also completed one lap at the 3 m/s test cap in approximately 20.02 s
with zero collisions; that check was not recorded with the telemetry recorder,
so its result is recorded in `live_test_summary.csv` rather than represented as
a raw per-cycle capture.

### 2. Clean PP plus MPC shadow holdout

Folder: `03_pp_mpc_shadow_clean/`.

Source: `pp_mpc_shadow_icra_20260918`.

The existing replay README reports 13030 synchronized cycles, all accepted,
covering approximately 16.27 raceline laps and reaching 8.59 m/s odometry
speed. The MPC output stayed inside its configured envelope; target-speed max
was 8.85 m/s and minimum predicted corridor slack was 0.2173 m.

This proves bounded offline MPC outputs at states visited by PP. It does not
prove MPC command authority because the MPC output was not applied back to the
simulator.

### 3. Partial MPC authority capture

Folder: `04_mpc_authority_partial/`.

Source: `mpc_authority_4mps_attempt2_20260918`.

This is a short, incomplete authority capture with 490 bridge/state cycles and
485 map-pose rows. It is useful for checking the authority-run sensor and
command sequence but is not a complete lap and must not be treated as a clean
holdout.

### 4. Solver and recovery diagnostics

Folder: `05_mpc_replay_diagnostics/`.

Important observations from these files:

- The live rejection diagnostic aggregate contains 1288 solver-residual
  rejects (99.536% of rejected cycles) and 6 missing synchronized-state
  rejects.
- The clean recovery-policy replay contains 13030 cycles, 13030 accepted, and
  zero active recovery cycles. It proves normal-operation parity, not recovery
  on the original failing state.
- The dimensionless-scaling diagnostic produced 4127 rejects on the same
  clean holdout and is not a production candidate.
- The recovery schedule CSV is stage-expanded so an external reviewer can
  determine whether an `e_y` issue is measured-state violation, first-step
  reachability, seed re-entry failure, or solver failure.

## Newest live observations

The newest tests were deliberately run in batchmode without `-no-graphics`.
The telemetry recorder was not enabled, so these results are summary evidence,
not raw CSV traces. See `live_test_summary.csv`.

### MPC with a 16 m/s global ceiling

- The command sample was approximately 5.30 m/s, not 16 m/s.
- The earlier target-speed runaway was not reproduced after the correction.
- The vehicle completed zero laps and recorded 14 collisions.

### MPC with a 3 m/s ceiling

- `/cmd/speed` arrived at approximately 39.94 Hz.
- The command sample was approximately 2.28 m/s.
- The MPC target reference in diagnostics was approximately 2.30 m/s.
- The vehicle completed zero laps and recorded 20 collisions.
- This demonstrates that the current authority failure is not explained by an
  MPC command running to 16 m/s.

### PP with the same exact raceline and 3 m/s ceiling

- One clean lap was completed in approximately 20.02 s.
- Collision count was zero.

## Current interpretation

The evidence separates the problem into two different failures:

1. **High target-speed generation was a formulation/fallback problem.** The
   raceline maximum is 10.16 m/s, while the old MPC target state had no direct
   raceline-speed cost and could float toward the 16 m/s global ceiling. The
   target-state reference, inverse feedforward, and fallback cap address this
   specific defect.

2. **The remaining authority failure is MPC-specific and lateral/state-path
   related.** It remains at a 3 m/s cap, where the target speed is correct. PP
   succeeds on the same raceline and simulator setup, so the raceline is not
   inherently infeasible and the map is not globally unusable. The live MPC
   run showed nonlinear, corridor, residual, and recovery/fallback activity.

The most important unproven possibility is startup/state handoff. The current
`mpc_start_delay_sec` delays MPC relative to AMCL process start; it does not
wait for an explicit AMCL global-lock/stability condition. A fixed delay can
therefore let MPC project an early provisional map pose or consume a pose that
is still being corrected. The exact live first-failure state was not captured
in a per-cycle telemetry CSV, so this is a hypothesis to test rather than a
proven root cause.

## Questions for external analysis

Please analyze the package in this order:

1. Compare PP baseline and MPC shadow legal state streams by header/arrival
   time. Quantify `/odom` to `/current_map_pose` skew and jumps in `e_y` and
   `e_psi` after projection onto the exact raceline.
2. On `live_shadow_mpc_cycles.csv`, classify rejected cycles by state age,
   `e_y`, `e_psi`, `u`, `v`, `r`, curvature, corridor slack, nonlinear failure
   stage, residual, and R1/R2 selection.
3. Verify whether the first solver/recovery failure follows an AMCL correction,
   a control-time extrapolation, a path-projection segment change, or an
   actuator/model mismatch.
4. Compare `recovery_policy_cycles.csv` and
   `recovery_policy_schedule_summary.csv` to determine whether the recovery
   policy was exercised only synthetically or on a physically reachable
   violation.
5. Check the longitudinal target state independently: target speed must follow
   the local raceline profile with only the small inverse-model headroom, never
   the global 16 m/s ceiling.
6. Use `gt_odom.csv` only to label the cause offline. Do not feed it into an
   estimator, controller, replay state, or runtime decision.

## What is not proven

- MPC authority has not completed a lap.
- The exact 2026-09-19 authority failure does not have a raw per-cycle CSV
  capture.
- The `e_y` recovery mechanism has not been exercised on the strongest real
  authority failure; the clean replay has no active recovery cycles.
- Timing averages near 40 Hz do not by themselves prove that message source
  epochs are aligned; the event and runtime-state streams must be compared.
- No conclusion should be drawn from the simulator truth fields about what the
  controller is allowed to use online.

## Suggested next experiment

Enable the telemetry recorder for one controlled MPC authority run at the 3
m/s cap using the exact raceline. Record `/odom`, `/ekf_odom`,
`/current_map_pose`, `/cmd/speed`, MPC diagnostics, IMU, encoders, bridge
timing, and simulator truth only as offline diagnostics. Require the recorder
to retain data through the first collision or first solver rejection. Then
compare the first failure against the clean PP run before changing weights,
AMCL covariance, or actuator limits.

The package file list and provenance are in `DATASET_MANIFEST.csv`.
