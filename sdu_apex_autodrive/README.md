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

For the numeric timing gate, use the simulator's documented no-rendering mode
instead; it disables all rendering, including vehicle-camera rendering, at the
simulator rather than filtering the image after transmission:

~~~bash
cd /home/autodrive_simulator
./AutoDRIVE\ Simulator.x86_64 -batchmode -nographics \
  -ip 127.0.0.1 -port 4567 -logFile /tmp/autodrive-nographics.log
~~~

When using the locally built simulator image, the equivalent wrapper mode is
`AUTODRIVE_SIM_RENDER_MODE=nographics autodrive-headless-simulator`.

In a second terminal, start exactly one 40 Hz command-pacing bridge and wait
for the simulator UI to report Connected!. The official bridge alone can
publish at a slower image-processing-limited cadence, so it is not the
acceptance path for this stack:

~~~bash
docker exec -d sdu_apex_autodrive bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/autodrive_devkit/install/setup.bash
  source /workspace/install/setup.bash
  exec ros2 run sdu_apex_autodrive autodrive_bridge_40hz \
    --ros-args -r __node:=autodrive_bridge
'
~~~

The recorder may remain at its configured 50 Hz timer rate, but the native
simulator sensor/event stream used for acceptance must be 40 Hz without bursty
delivery. `test_suite.launch.py` starts only non-actuating sensor odometry for
preflight, then requires the timing/data gate to pass before it starts any
actuator or calibration process. The gate checks bridge-side packet arrivals,
request timing, raw IMU/encoders/LiDAR, simulator odometry, `/odom`, and
`/odom/diagnostics`. It rejects duplicate timestamps, bursts, long gaps,
missing packets, non-finite values, and implausible encoder jumps. A monitor
continues during the finite run and aborts it if timing degrades.

Camera decode and ROS camera publication are disabled by default for the
numeric odometry/timing path. The API also sends the source-side setting to the
patched Unity simulator, which omits camera capture and serialization before
the packet is sent. The legacy prebuilt simulator image cannot apply that
setting and will continue to include its compiled camera field until replaced
with the patched player in `simulator/`. `publish_camera:=true` opts back into
camera handling for a run that needs `/autodrive/roboracer_1/front_camera`.

### Prebuilt-image camera-off workaround

When Unity source/build access is unavailable, the supplied IL2CPP player can
be run with a reversible scene-file override that removes the serialized
front-camera assignment. This changes the serialized player scene and must not
be treated as competition-equivalent without organizer approval.

Generate the override from the exact image tag being used:

~~~bash
cd /home/akselmo/Documents/GitHub/SDU-Apex-Autodrive
./simulator/tools/prepare_camera_disabled_level0.sh \
  autodriveecosystem/autodrive_roboracer_sim:2026-iros-explore \
  /tmp/autodrive-level0-no-camera
~~~

Start it with the optional Compose overlay. A real display is required for
this workaround; the Unity `-nographics` player remains capped at the observed
20 Hz even after camera removal:

~~~bash
export AUTODRIVE_SIM_LEVEL0=/tmp/autodrive-level0-no-camera
docker compose \
  -f docker-compose.yml \
  -f simulator/docker-compose.camera-off.yml \
  up -d workspace simulator
docker compose exec simulator bash -lc '
  cd /home/autodrive_simulator
  exec taskset -c 0-11 "./AutoDRIVE Simulator.x86_64" \
    -batchmode -ip 127.0.0.1 -port 4567 \
    -logFile /tmp/autodrive-camera-off.log
'
~~~

Because this legacy binary cannot provide Unity simulation-time/frame
metadata, its timing-only preflight must explicitly use legacy mode:

~~~bash
docker compose exec workspace bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/autodrive_devkit/install/setup.bash
  source /workspace/install/setup.bash
  ros2 launch sdu_apex_autodrive test_suite.launch.py \
    test:=timing_only start_bridge:=false \
    require_simulation_metadata:=false
'
~~~

The default remains `require_simulation_metadata:=true` for a rebuilt Unity
player with the source patch. Legacy mode still enforces 40 Hz, 25 ms-like
source intervals, no bursts/gaps, finite sensor values, and cross-stream
coherence.

The simulator container already has unlimited CPU and memory cgroup limits and
GPU access. `AUTODRIVE_SIMULATOR_IMAGE` selects a rebuilt player image; the
compose file also provides 2 GB shared memory for Unity graphics. More Docker
resources alone do not remove synchronous Unity main-thread capture or packet
serialization cost.

To perform only the non-driving timing/data preflight, use:

~~~bash
docker exec -it sdu_apex_autodrive bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/autodrive_devkit/install/setup.bash
  source /workspace/install/setup.bash
  ros2 launch sdu_apex_autodrive test_suite.launch.py \
    test:=timing_only
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

## Deterministic odometry observer

The active `/odom` implementation is a pure deterministic observer. It
assembles exact source-timestamp packets from both encoders and the IMU before
updating; it never forward-fills, interpolates, or fabricates a sensor value.
The observer uses the frozen wheel-speed map in
`config/wheel_speed_map.csv`, scalar longitudinal propagation, a gated wheel
update, and a 2-D body-frame RK2 turn mode with the fixed lever-arm correction.
Ground truth, throttle, camera, LiDAR, AMCL, and learned models are not runtime
inputs.

The observer constants are frozen in
`f1tenth_localization/config/sensor_odometry.yaml`. Diagnostics are version 2
and publish source stamp, timing state, wheel-gate state, turn state, reset
epoch, packet-drop count, and packet-coherence-fault count.

The canonical historical fit/test/blind recordings are:

- `artifacts/calibration/raw/identification_grid_fit_40hz_20260905_final/identification_grid_20260905_144304.csv`
- `artifacts/calibration/raw/identification_grid_test_40hz_20260905/identification_grid_20260905_151313.csv`
- `artifacts/calibration/raw/identification_grid_validation_40hz_20260905/identification_grid_20260905_155143.csv`

Replay uses exact packet reconstruction and has a Python reference plus a
standalone C++ replay binary:

~~~bash
PYTHONPATH=sdu_apex_autodrive python3 -m \
  sdu_apex_autodrive.scripts.validate_odometry_observer INPUT.csv \
  --cpp-replay /tmp/odometry_observer_replay
~~~

For new model development, keep fitting and scoring offline and separate from
the runtime observer. Write any derived outputs for a new run under one dated
directory:

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
50 Hz sensor rate. Ground truth is allowed only for offline labels and targets;
runtime never consumes ground truth, throttle, AMCL, or camera data.

## Controller abstractions

Pure Pursuit, Stanley, and FTG publish through the controller boundary. Pure
Pursuit publishes /cmd/speed; actuator_interface converts speed to normalized
throttle using feed-forward and speed feedback. MPC publishes
/cmd/acceleration; the same boundary integrates that request into a speed
trajectory and uses the validated speed loop. This avoids closing throttle on
the simulator's bursty acceleration derivative. Only one controller and one
actuator interface may be active in a racing run.

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

The acceleration command path bounds the requested positive acceleration by
the measured speed-dependent reachable envelope rather than assuming every
acceleration is available everywhere: approximately 5.5 m/s² at standstill,
3.18 at 8 m/s, 2.04 at 12 m/s, 0.96 at 16 m/s, 0.53 at 18–22 m/s, and near
zero at the 23 m/s limit. It integrates the bounded request into a speed
trajectory and reuses the speed-loop feed-forward and feedback. Negative
acceleration coasts because the competition actuator has no active brake
channel. MPC uses the same envelope in its per-horizon acceleration bounds.

The observer replay matched the C++ implementation to floating-point precision
on the fit, test, blind validation, and fresh motion capture. The three
historical replays had p95 relative speed error of 0.7208%, 0.7218%, and
0.7422%, respectively. The fresh no-reset motion capture passed at 40 Hz,
contained 1,081 complete packets with zero incomplete or duplicate packets,
and had every moving replay sample within 2% in the validation report.

The old learned C++ runtime headers are removed from the active localization
include path. Offline model-development scripts and artifacts remain separate
so new models can be developed without changing the deterministic runtime.

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

For map-frame localization without starting a controller or a second bridge,
use the dedicated launch while the official bridge is already running:

~~~bash
ros2 launch sdu_apex_autodrive localization.launch.py start_bridge:=false
~~~

The launch defaults to the compete map and ICRA raceline above. It does not
upsample simulator data: AMCL consumes each real LiDAR scan at the native
player cadence.
