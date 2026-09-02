# AutoDRIVE diagnostic calibration

This package contains an offline diagnostic suite for the official
AutoDRIVE RoboRacer simulator. It is separate from the competition runtime.

## Current competition-track runtime defaults

The unified controller launch currently defaults to
`f1tenth_planning/maps/autodrive_compete_2026.yaml` and
`f1tenth_planning/trajectories/icra_2025_raceline.csv`. The map is the verified
competition-track geometry retained from the historical simulator map; it was
checked against the current official ICRA simulator before being promoted.
The simulator must be running and its UI connection state must read
`Connected!` before live tuning or recording begins.

For mapping experiments, `config/compete_mapping.yaml` enables the five-lap
saver. `config/slam_params.yaml` allows the measured bridge timestamp jitter
(`transform_timeout: 1.0`, `tf_buffer_duration: 60.0`). Mapping outputs must be
inspected for closure before replacing the verified runtime map.

`calibration_full_suite.yaml` runs, in one batch on the open-ground scene:

- isolated direct-throttle steps from `0.00` through the complete normalized
  forward range `1.00`;
- full-throttle release and passive deceleration;
- closed-loop speed steps through the requested full simulator envelope and
  back to zero. Use `calibration_speed_full_envelope.yaml` for the long-dwell
  high-speed test.

The recorder stores the allowed runtime sensors plus simulator ground truth
(`/autodrive/roboracer_1/ips`, `/autodrive/roboracer_1/odom`, and
`/autodrive/roboracer_1/collision_count`). Ground truth is diagnostic-only and
is never subscribed to by the racing controller or localization nodes.

Run inside the Humble workspace container with the bridge started before the
simulator application:

```bash
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
source /workspace/install/setup.bash
ros2 launch sdu_apex_autodrive test_suite.launch.py \
  test:=full_suite \
  calibration_params:=/workspace/install/share/sdu_apex_autodrive/config/calibration_full_suite.yaml \
  output_dir:=/workspace/src/sdu_apex_autodrive/artifacts/calibration/raw/open_scene_full_suite
```

The simulator application may then be started with its normal `-ip 127.0.0.1
-port 4567` arguments. The generated CSV is analyzed offline:

The application can render its scene while disconnected. For this simulator
release, open its hamburger menu and activate the `Disconnected` control; wait
until it reads `Connected` before starting or trusting a recording. A recorder
started before that connection only reaches its startup timeout and contains no
telemetry rows.

```bash
ros2 run sdu_apex_autodrive analyze_calibration \
  /workspace/src/sdu_apex_autodrive/artifacts/calibration/raw/open_scene_full_suite/full_suite_<timestamp>.csv \
  --feedforward-output \
  /workspace/src/sdu_apex_autodrive/artifacts/calibration/derived/ground_truth_feedforward.csv \
  --response-output \
  /workspace/src/sdu_apex_autodrive/artifacts/calibration/derived/ground_truth_response.csv \
  --acceleration-map-output \
  /workspace/src/sdu_apex_autodrive/artifacts/calibration/derived/throttle_speed_acceleration.csv
```

The analyzer reports truth-based speed, acceleration, response, collision,
encoder-versus-truth, motion-regime, and sensor-cadence metrics. Recorder rows
are written at a faster timer rate than native simulator telemetry; when
`gt_odom_event_count` is present, fit and error metrics use one row per unique
ground-truth odometry event and report the discarded duplicate count. Truth
distance validation uses source timestamps and the documented 22.88 m/s speed
envelope, so legitimate high-speed steps are not rejected by a fixed
low-speed displacement threshold. For a saved regime report, add:

```bash
--regime-output \
/workspace/src/sdu_apex_autodrive/artifacts/calibration/derived/odom_regimes.csv
```

The production feed-forward table includes the full normalized throttle range.
The acceleration map keeps the measured throttle-to-acceleration relationship
across speed bins, while the runtime controller uses only the allowed IMU and
encoder/odometry inputs. Provenance for retained calibration files is recorded
in `artifacts/calibration/MANIFEST.yaml`.
The `calibration_speed_full_envelope.yaml` profile holds each high-speed
target for six seconds so reaching the body-speed envelope is not confused
with a short transient.
