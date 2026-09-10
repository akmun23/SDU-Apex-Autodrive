# AMCL / odometry acceptance campaign

All runs below are real graphical AutoDRIVE ICRA-compete simulator runs. The
Unity player was started with `-batchmode` and without `-no-graphics`. Camera
publication was disabled in the bridge. Ground truth appears only in the
recorder and monitor for offline scoring.

The native LiDAR, odometry, encoder, and IMU streams measured about 19--20 Hz;
steering commands measured about 40 Hz. The recorder preserves source event
timestamps, so duplicated recorder rows must not be interpreted as extra
sensor samples.

## Retained runs

- `candidate_a`: production AMCL at 0.5 m/s; collision-free slow baseline.
- `candidate_c`: 4 cm local AMCL correction gate at 0.5 m/s; collision-free,
  but not promoted to production.
- `control_a_2mps`: production AMCL at 2 m/s before the odometry turn-entry
  fix; collision after launch odometry lost about 1.5 m.
- `ekf_cov_2mps_retry`: production AMCL with the fitted EKF XY process-noise
  density; collision after the same launch odometry failure.
- `turn_entry_fix_2mps`: current turn-entry fix at 2 m/s; odom/EKF absolute
  error stayed about 5--12 cm before collision, while delayed steering
  feedback and the raceline's start bend caused the vehicle to leave the path.

Each run contains `amcl_vs_ground_truth.csv`, the raw telemetry CSV, the
controller log, and the simulator log. `localization_score.json` was produced
with the telemetry-aware scorer and is the acceptance result for that run.

The 2 m/s runs are failures, not passes. A future change must be validated at
the real driving speed and must complete the track collision-free; a 0.5 m/s
run remains useful only as a diagnostic comparison.

## Continuous-correction acceptance runs

The active AMCL configuration has no lifetime correction budget. Both
per-scan innovation caps are disabled with `0.0`, and full 2-D scan
corrections are enabled; association radius and cluster weight are the only
remaining local-match quality checks.

- `acceptance_3mps_cross_track_gain025_weight090`: rejected after 13.5 m;
  this was the last cross-track-only comparison and is retained as evidence
  that the projection was too restrictive for full localization.
- `acceptance_3mps_full_xy_gain025_weight090`: real graphical 3 m/s run,
  167.1 m before collision. Current map pose p95 error was 0.139 m and
  median error 0.038 m with no pose jumps. The collision snapshot had about
  0.04 m current-map error, while the actuator sent -0.488 rad for a PP
  request of -0.256 rad; this is an actuation/PP tracking failure, not an
  AMCL correction-budget rejection. The GT/current-map/AMCL overlay is saved
  as `map_raceline_gt_current_map_amcl_pre_collision.png`.

- `acceptance_3mps_gain050_raycast_local_bag`: raycast verification enabled for
  local tracking, with no correction caps. The graphical 3 m/s run travelled
  185.4 m before collision. Source-time scoring gave current-map p50/p95/max
  errors of 5.1/25.4/44.6 cm with zero pose jumps; raw AMCL was
  16.6/38.5/63.9 cm, while odometry was 55.4/128.5/158.3 cm. This is a
  meaningful localization improvement, but the run is not an acceptance pass
  because the simulator reported a collision. The overlay is saved as
  `acceptance_3mps_gain050_raycast_local_bag/map_gt_current_map_amcl_pp_collision.png`.

- `acceptance_4mps_gain050_raycast_local_bag`: same continuous-correction
  configuration at a 4 m/s PP cap. The car reached the inner hairpin at an
  actual peak of about 4 m/s and collided after 25.0 m. Source-time
  current-map p50/p95/max errors were 8.6/42.5/49.6 cm; odometry reached
  170.1 cm p95. The collision occurred with current-map error about 4 cm but
  PP steering saturated, so this is a rejected high-speed control/trajectory
  run, not evidence for restoring an AMCL correction cap. Its raw data is
  retained for the PP investigation.

## Startup refinement and lookahead acceptance

- `acceptance_startup_refinement_stationary`: stationary startup diagnostic.
  The first LiDAR scan refined 2234/2500 particles and produced the saved
  map-frame start `(0.801, 3.158, -1.571)`. The monitor reported zero position
  and yaw error while the vehicle was stationary. This verifies the startup
  matcher fix for the previous `0.805 m > 0.750 m` anchor rejection.

- `acceptance_3mps_lookahead075_refined`: real graphical 3 m/s run using the
  startup refinement and 0.75--1.10 m PP lookahead. It travelled 379.8 m
  (~7.3 track lengths) for 90 s without collision or reset. Current-map
  error was 2.96 cm median, 10.65 cm p95, and 22.65 cm maximum with no pose
  jumps. The complete monitor CSV, raw allowed-sensor telemetry, raw ROS bag,
  and score are retained.

- `acceptance_4mps_lookahead075_refined_retry`: real graphical 4 m/s run using
  the same settings. It travelled 268.4 m (~5.2 track lengths) for 45 s
  without collision or reset, reaching 4.02 m/s. Current-map error was
  7.37 cm median, 24.39 cm p95, and 36.79 cm maximum with no current-map
  jumps. PP cross-track error p95 was 9.97 cm and neither commanded nor
  feedback steering reached the 0.52 rad limit in this run. The map overlay
  is `map_gt_current_map_raceline.png`.

The tested PP lookahead and startup scan refinement are now the production
defaults in `f1tenth_control/config/path_tracking_autodrive.yaml` and
`f1tenth_localization/config/gpu_amcl_cpp_params.yaml`. The 4 m/s run is a
control/localization milestone, not a claim that the complete estimator is
within a fixed absolute 5 cm or 2% bound at every instantaneous sample.
