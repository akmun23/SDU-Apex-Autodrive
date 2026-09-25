# AutoDRIVE startup guide

This guide covers the official practice and compete simulator images, the
development stack, track mapping, and competition-image rehearsals. The
controller must always receive a map and raceline generated for the same
simulator track.

The normal controller is MPC. The development launch also supports Pure
Pursuit and FTG explicitly.

## 1. Start the development stack before the simulator

The simulator waits for the ROS bridge to listen on port 4567 before starting
Unity, so the bridge must be running first. Start the matching development
stack, recorder, and run watcher before the simulator. For the practice MPC,
follow sections 2 and 5, then start the simulator as the final step. The
batchmode helper starts Xvfb so Unity retains a graphics display; it uses
`-batchmode` and never uses `-no-graphics`:

```bash
SDU_APEX_SIM_TRACK=practice SDU_APEX_SIM_MODE=batchmode \
  ./tools/start_simulator.sh
```

Keep the terminal open. Stop the simulator with `Ctrl-C`. The helper does not
delete or replace existing containers. Docker commands automatically use the
isolated rootless daemon when its socket is available; an explicit
`DOCKER_HOST` still takes precedence.

For the other packaged course, select its image explicitly:

```bash
SDU_APEX_SIM_TRACK=compete SDU_APEX_SIM_MODE=batchmode \
  ./tools/start_simulator.sh
```

The `gui` mode is available only when a desktop display and readable
`XAUTHORITY` are explicitly supplied. Do not use `-no-graphics` for the
batchmode run.

The official competition guidance distinguishes the practice/qualification
track from the later released track, so they require separately mapped maps
and racelines. See the [official track guidance](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-icra-2026/),
the [practice simulator image](https://hub.docker.com/layers/autodriveecosystem/autodrive_roboracer_sim/2026-icra-practice/images/sha256-264e3946dc4ff199d667caa65b3bebc1829822acf201a3a33e7aa406bc574441),
and the [compete simulator image](https://hub.docker.com/layers/autodriveecosystem/autodrive_roboracer_sim/2026-icra-compete/images/sha256-0511fb6d6db7f31ec0d20787c0498eb05f3bd45ae6b9683165e8680b61e9b159).

## 2. Start the development stack

For the practice run, first build the development image against the matching
API image. This keeps the expensive rebuild separate from starting the stack,
so the recorder can be started before the controller:

```bash
source tools/docker_env.sh
docker build --network=host \
  --build-arg AUTODRIVE_API_IMAGE=autodriveecosystem/autodrive_roboracer_api@sha256:2447bb8466e631b10412096ddde1b183c806ae328aee6383f9e373a0d46772cc \
  -t sdu-apex-autodrive:practice-dev .
```

For the compete image, use the matching API digest and a distinct image tag:

```bash
AUTODRIVE_API_IMAGE=autodriveecosystem/autodrive_roboracer_api@sha256:ce081910948c3f30898322358d682b79cf165aa287a3dc27128dbacae99178c7 \
SDU_APEX_IMAGE=sdu-apex-autodrive:dev-compete \
./tools/start_dev.sh
```

The helper builds when `AUTODRIVE_REBUILD=1` (the default), mounts the source,
runs the topic-policy check, then starts the legal bridge, odometry,
localization, controller, and actuator. MPC is the default. For an existing
matching image, set `AUTODRIVE_REBUILD=0`.

For this practice track, start MPC with the map and raceline as a matching
pair:

```bash
AUTODRIVE_REBUILD=0 \
SDU_APEX_IMAGE=sdu-apex-autodrive:practice-dev \
SDU_APEX_MAP_YAML="$PWD/f1tenth_planning/maps/autodrive_practice_20260924_b.yaml" \
SDU_APEX_TRAJECTORY_FILE="$PWD/f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv" \
./tools/start_dev.sh
```

To select another maintained development controller:

```bash
SDU_APEX_CONTROLLER=pure_pursuit ./tools/start_dev.sh
SDU_APEX_CONTROLLER=ftg ./tools/start_dev.sh
```

Optional development settings:

```bash
SDU_APEX_WITH_RVIZ=true ./tools/start_dev.sh
SDU_APEX_MPC_PUBLISH_DIAGNOSTICS=true ./tools/start_dev.sh
```

Diagnostics are for development only and should remain disabled for normal
performance measurements.

Stop the ROS stack with `Ctrl-C`, or from another terminal run:

```bash
./tools/stop_autodrive.sh
```

Do not start a second stack before stopping the first one. The development
launch rejects duplicate runtime nodes.

## 3. Map and race on each track

Mapping uses simulator world pose and lap count only in this development
workflow. FTG drives the car; MPC and a race line are not used to collect the
map. First build and start the development mapper against the matching API
base. This launch contains the bridge, SLAM, FTG mapping controller, and
actuator:

```bash
AUTODRIVE_API_IMAGE=autodriveecosystem/autodrive_roboracer_api@sha256:2447bb8466e631b10412096ddde1b183c806ae328aee6383f9e373a0d46772cc \
SDU_APEX_IMAGE=sdu-apex-autodrive:dev-practice \
SDU_APEX_LAUNCH=mapping SDU_APEX_MAP_NAME=icra_practice \
./tools/start_dev.sh
```

In two more terminals, start the debug recorder and then the collision watcher.
Only after all three are running, start the simulator in a fourth terminal:

```bash
SDU_APEX_IMAGE=sdu-apex-autodrive:dev-practice \
SDU_APEX_RUN_ID=map_icra_practice ./tools/record_debug_bag.sh
```

```bash
./tools/watch_sim_run.sh sdu_apex_autodrive_dev sdu_apex_sim_practice \
  rec_map_icra_practice collision 1800
```

```bash
SDU_APEX_SIM_TRACK=practice SDU_APEX_SIM_MODE=batchmode \
  ./tools/start_simulator.sh
```

The mapper counts five simulator laps, writes per-lap snapshots and the final
map under `live_runs/maps/`, then requests actuator neutral. Wait for the
`5-lap map saved successfully` log, stop the mapping stack with `Ctrl-C`, and
then stop the watcher with `Ctrl-C`; its cleanup stops the simulator and
finalizes the bag. If a collision occurs, the watcher stops the simulator and
finalizes the bag immediately; then stop the mapping stack. Never use
post-collision samples for map or model validation.

For the other official track, use its matching API image, with distinct
image/container names and a distinct map name. Start its mapper, recorder,
and watcher before starting its simulator, using `SDU_APEX_SIM_TRACK=compete`.
```bash
AUTODRIVE_API_IMAGE=autodriveecosystem/autodrive_roboracer_api@sha256:ce081910948c3f30898322358d682b79cf165aa287a3dc27128dbacae99178c7 \
SDU_APEX_IMAGE=sdu-apex-autodrive:dev-compete \
SDU_APEX_LAUNCH=mapping SDU_APEX_MAP_NAME=icra_compete \
./tools/start_dev.sh
```

Use matching image and container names in the recorder/watcher commands.
`start_dev.sh` builds by default; set `AUTODRIVE_REBUILD=0` only when that
exact development image is already built.

Generate a matching minimum-time raceline from each saved map with the map
optimizer, using an environment that has the optimizer requirements installed:

```bash
MINTIME_MAP="$PWD/live_runs/maps/icra_practice.yaml" \
MINTIME_TRACK_NAME=icra_practice_mintime \
MINTIME_OUTPUT="$PWD/f1tenth_planning/trajectories/icra_practice" \
python3 f1tenth_planning/scripts/optimize_trajectory.py
```

Promote each validated map YAML and its referenced image into
`f1tenth_planning/maps/` with a unique name. Keep its resulting raceline under
`f1tenth_planning/trajectories/`. Then race with both paths explicitly paired:

```bash
SDU_APEX_MAP_YAML="$PWD/f1tenth_planning/maps/icra_practice.yaml" \
SDU_APEX_TRAJECTORY_FILE="$PWD/f1tenth_planning/trajectories/icra_practice/icra_practice_mintime_raceline.csv" \
AUTODRIVE_API_IMAGE=autodriveecosystem/autodrive_roboracer_api@sha256:2447bb8466e631b10412096ddde1b183c806ae328aee6383f9e373a0d46772cc \
SDU_APEX_IMAGE=sdu-apex-autodrive:dev-practice \
./tools/start_dev.sh
```

`map_yaml` and `trajectory_file` must be supplied together; the launch rejects
a mixed map/raceline pair.

## 4. Exact competition-container rehearsal

Build the self-contained image without mounting the source tree:

```bash
AUTODRIVE_MAP_REL=maps/icra_practice.yaml \
AUTODRIVE_TRAJECTORY_REL=trajectories/icra_practice/icra_practice_mintime_raceline.csv \
AUTODRIVE_API_IMAGE=autodriveecosystem/autodrive_roboracer_api@sha256:2447bb8466e631b10412096ddde1b183c806ae328aee6383f9e373a0d46772cc \
SDU_APEX_IMAGE=sdu-apex-autodrive:competition-practice \
./tools/build_competition.sh
```

With the simulator already running, start the fixed MPC-only launch:

```bash
SDU_APEX_IMAGE=sdu-apex-autodrive:competition-practice \
./tools/run_competition.sh
```

This path uses the map and raceline baked into the image and
`competition.launch.py`. Build a separate image for each track. It does not
start RViz, a shadow controller, a simulator-truth input, a parameter overlay,
or a recorder; ROS output is discarded and Docker logging is disabled.

The competition launch starts MPC only after the AMCL warm-up. The runtime
nodes consume simulator LiDAR, IMU, encoders, steering, and throttle feedback,
plus the team-derived `/odom`, `/ekf_odom`, and `/current_map_pose` topics.

## 5. Record a real run

Recording all topics is allowed for debugging and offline analysis, but the
active controller, odom, EKF, and AMCL must not subscribe to simulator truth,
IPS, simulator odom, lap, collision, reset, result, or absolute `/tf` topics.

The debug bag records only the LiDAR/IMU/encoder inputs, odometry, EKF and AMCL
outputs/diagnostics, commands and actuator feedback, bridge timing, and
simulator ground truth/lap/collision data needed for offline comparisons. It
does not record the camera or large particle-cloud streams. Simulator truth,
IPS, lap, and collision data are debug-bag-only and must never be subscribed to
by the controller, odometry, EKF, or AMCL nodes. Create only a run parent
directory; `ros2 bag record` must create the final `run` directory itself.

```bash
SDU_APEX_RUN_ID=real_run_$(date +%Y%m%d_%H%M%S) \
./tools/record_debug_bag.sh
```

Start the recorder before the controller can move the car. For 1 warmup + 10
racing + 1 extra lap, start the collision watcher with target lap 12. It stops
the simulator and finalizes the recorder on the first collision or after the
target count:

```bash
./tools/watch_sim_run.sh sdu_apex_autodrive_dev sdu_apex_sim_practice rec_<run-id> 12 360
```

Never use post-collision samples for
estimator or MPC conclusions. Acceptance requires zero collisions, measured
40 Hz delivery of the sensor/bridge data during driving, and each of the ten
racing laps below 11 seconds; the warmup and extra lap are not timed as racing
laps.

## 6. Preflight and useful checks

Run the static topic-boundary check before launching a stack:

```bash
python3 tools/verify_runtime_topic_policy.py
```

Useful live checks from another ROS shell are:

```bash
docker exec -it sdu_apex_autodrive_dev bash
```

Then, inside that shell:

```bash
ros2 topic hz /autodrive/roboracer_1/lidar
ros2 topic hz /odom
ros2 topic echo --once /current_map_pose
```

The bridge target is 40 Hz. A sustained lower rate or a growing AMCL scan
queue indicates a transport or processing problem and should be investigated
before changing localization or MPC parameters.

Mapping remains a separate development launch. Only the fixed competition
image uses the selected map/raceline pair; controller/localization runtime
inputs remain unchanged.
