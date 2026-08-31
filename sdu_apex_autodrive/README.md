# AutoDRIVE diagnostic calibration

This package contains an offline diagnostic suite for the official
AutoDRIVE RoboRacer simulator. It is separate from the competition runtime.

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
encoder-versus-truth, and sensor-cadence metrics. The production feed-forward
table includes the full normalized throttle range. The acceleration map keeps
the measured throttle-to-acceleration relationship across speed bins, while
the runtime controller uses only the allowed IMU and encoder/odometry inputs.
The `calibration_speed_full_envelope.yaml` profile holds each high-speed
target for six seconds so reaching the body-speed envelope is not confused
with a short transient.
