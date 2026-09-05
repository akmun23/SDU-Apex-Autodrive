# AutoDRIVE calibration and actuator boundary

This package contains the finite diagnostic calibration suite and the
production actuator boundary for the AutoDRIVE RoboRacer stack. The open-world
recorder may record simulator ground truth for fitting, but truth is never
consumed by runtime odom, EKF, AMCL, planning, or controllers.

## One finite odom-identification test

Use the official open-world simulator image
autodriveecosystem/autodrive_roboracer_sim:2026-iros-explore. The scene must
be visibly flat and contain no track. identification_grid starts the bridge,
sensor-only odom, local EKF, and recorder; it deliberately does not start AMCL.

The grid in config/calibration_identification_grid.yaml covers seven speed
bands, twelve throttle values, and steering probes at four speeds. It logs
commands and feedback, IMU, both cumulative encoders, timestamp-derived wheel
speeds, wheel/body slip, odom diagnostics, EKF, timestamped simulator odom,
IPS, collision count, reset epoch, and measured source rates.

Every point is isolated. The recorder waits for throttle settling, applies zero
throttle for braking, and resets only after encoder, IMU, odom, and diagnostic
ground-truth stop evidence agree. Encoder freeze while the ground-truth car is
moving remains valid slip data.

Start the official simulator first, make the flat no-track scene visible, and
wait until its UI says Connected!. This avoids a calibration startup race:

~~~bash
cd /home/akselmo/Documents/GitHub/SDU-Apex-Autodrive
docker compose up -d workspace simulator
docker exec -it autodrive_roboracer_sim bash
cd /home/autodrive_simulator
./AutoDRIVE\ Simulator.x86_64 -screen-width 1280 -screen-height 720 \
  -screen-fullscreen 0 -ip 127.0.0.1 -port 4567
~~~

In a second terminal, start exactly one bridge and wait for the simulator UI to
report Connected!:

~~~bash
docker exec -d sdu_apex_autodrive bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/autodrive_devkit/install/setup.bash
  source /workspace/install/setup.bash
  exec ros2 run autodrive_roboracer autodrive_bridge \
    --ros-args -r __node:=autodrive_bridge
'
~~~

Then start the one finite test. `start_bridge:=false` prevents a second bridge
from racing the verified one:

~~~bash
docker exec -it sdu_apex_autodrive bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/autodrive_devkit/install/setup.bash
  source /workspace/install/setup.bash
  ros2 launch sdu_apex_autodrive test_suite.launch.py \
    test:=identification_grid \
    start_bridge:=false \
    calibration_params:=/workspace/install/share/sdu_apex_autodrive/config/calibration_identification_grid.yaml \
    output_dir:=/workspace/src/sdu_apex_autodrive/artifacts/calibration/raw/pending_identification_grid
'
~~~

Do not start AMCL in this open scene.

## Offline analysis

The canonical odometry identification corpus is
artifacts/calibration/raw/identification_grid_full_20260904/identification_grid_20260904_113534.csv.
The independent holdout validation corpus is
artifacts/calibration/raw/identification_grid_fusion_candidate_20260904/identification_grid_20260904_123232.csv.
Fit only on the first file and score only on the second; do not merge them.
Write all derived outputs for a new run under one dated directory:

~~~bash
ros2 run sdu_apex_autodrive analyze_calibration INPUT.csv \
  --feedforward-output derived/DATE/feedforward.csv \
  --response-output derived/DATE/response.csv \
  --acceleration-map-output derived/DATE/acceleration_map.csv \
  --acceleration-envelope-output derived/DATE/acceleration_envelope.csv \
  --longitudinal-fit-output derived/DATE/longitudinal_fit.csv \
  --wheel-speed-map-output derived/DATE/wheel_speed_map.csv \
  --slip-model-output derived/DATE/slip_model.csv \
  --frozen-encoder-model-output derived/DATE/frozen_encoder_brake_model.csv \
  --steering-output derived/DATE/steering.csv \
  --regime-output derived/DATE/regime.csv
~~~

The analyzer deduplicates rows by timestamped simulator odom event and reports
native rates from message source timestamps. It must not assume a fixed 40 or
50 Hz sensor rate. The regime fitter is
`sdu_apex_autodrive/sdu_apex_autodrive/scripts/fit_regime_sensor_fusion_model.py`.
It trains separate accelerating, steady, decelerating, and frozen-encoder
models from causal IMU/encoder features, including a timestamp-derived
9-sample encoder-travel window. Ground truth is used only for offline labels
and targets. The slip model is an offline diagnostic fit using the official
(S_x=(r\omega-v_x)/v_x) definition; it informs wheel confidence and model
selection, not a direct truth correction. Runtime never consumes ground truth,
throttle, or AMCL.

## Controller abstractions

Pure Pursuit, Stanley, and FTG publish through the controller boundary. Pure
Pursuit publishes /cmd/speed; actuator_interface converts speed to normalized
throttle using feed-forward and speed feedback. MPC publishes
/cmd/acceleration; the same boundary converts it using the inverse acceleration
model and filtered IMU feedback. Only one controller and one actuator interface
may be active in a racing run.

The speed controller has three operating phases. While the target error is
normally at least 1.5 m/s it uses the maximum configured forward throttle to
accelerate quickly. As the target is approached, it predicts the short-term speed rise
from the filtered IMU and the calibrated throttle model; at the target minus
0.15 m/s, or when that boundary is predicted within 0.25 s, it hands off to
the feed-forward throttle mapped to the requested target speed. It then holds
that operating point instead of returning to boost while the target remains
nearby. A target-relative overspeed guard commands passive coast at the larger
of the 0.15 m/s hold deadband or 10% of the target, bounded by the configured
coast threshold. This limits both overshoot and the subsequent undershoot
correction without requiring an unavailable active brake channel.

The acceleration controller uses the measured speed-dependent reachable
envelope rather than assuming every acceleration is available everywhere:
approximately 5.5 m/s² at standstill, 3.18 at 8 m/s, 2.04 at 12 m/s, 0.96 at
16 m/s, 0.53 at 18–22 m/s, and near zero at the 23 m/s limit. Requests above
that envelope are bounded before the throttle inverse is applied. MPC uses the
same envelope in its per-horizon acceleration bounds.

The current odom candidate is in
f1tenth_localization/config/sensor_odometry.yaml. It uses IMU and encoders only;
IMU acceleration alpha=0.90, wheel correction gain=0.10, published-velocity
filter alpha=1.0 for both acceleration and deceleration, and the causal
sensor-fusion model are enabled. The model is reproducibly fitted by
`sdu_apex_autodrive/sdu_apex_autodrive/scripts/fit_regime_sensor_fusion_model.py`
from the open-world identification corpus and compiled into the localization
node. It contains separate accelerating, steady, decelerating, and frozen
models and 24 causal features. The frozen-encoder fallback is the bounded
prior `decel = min(12.0, 5.5 + 0.27 * speed)` after a quiet-IMU hold; it is a
motion prediction, not a stop/reset condition.

The promoted held-out live validation reached a median relative speed error
below 2% in every measured nonzero speed bin: 0.967% at 1-3 m/s, 0.386% at
3-5, 0.360% at 5-10, 0.177% at 10-15, 0.264% at 15-20, and 0.198% at
20-23 m/s. This does not satisfy a strict every-sample 98% requirement: the
same p95 relative errors were 17.244%, 4.312%, 2.316%, 1.326%, 1.472%, and
1.410%. Long-distance integrated pose error (distance >=50 m) had 0.535%
median and 1.931% p95 relative error; short-distance/reset transients remain
the main position limitation. The complete metrics are in
`artifacts/calibration/derived/latest_regime_sensor_fusion_validation_promoted_20260903/`.
After that held-out check, the production header was regenerated from all
nine usable recordings (32,473 deduplicated examples). The saved
leave-one-recording-out aggregate is intentionally more pessimistic because
the historical recordings have different distributions; it is a robustness
warning, not a replacement for the independent live result.
The promoted run reached 22.882 m/s with zero collisions. Native sensor,
odom, EKF, and simulator-truth streams were about 11.49 Hz by their source
timestamps; commands were about 49.98 Hz. These measured rates include jitter
and are not assumed constants. The public `/odom` remains a single output;
model regime and window values are exposed through diagnostics.
The speed/acceleration slew-limit boundary was subsequently corrected so the
speed loop and acceleration loop use independent configured limits. The
existing live controller CSVs predate the boost-to-hold handoff and therefore
remain baseline evidence only; one fresh finite open-world speed verification
is required before final tuning is accepted.

## Track runtime

AMCL is excluded from open-ground identification. Track operation uses
controller.launch.py with:

~~~text
map:       f1tenth_planning/maps/autodrive_compete_2026.yaml
raceline:  f1tenth_planning/trajectories/icra_2025_raceline.csv
~~~

AMCL independently corrects map-frame position from the map and official
LiDAR. The EKF predicts from odom and applies delayed AMCL corrections without
feeding AMCL directly into sensor odom. IPS and simulator truth remain
diagnostic-only.

Active evidence and the recoverable cleanup archive are documented in
artifacts/calibration/MANIFEST.yaml and the repository-level
IMPLEMENTATION_STATUS.md.
