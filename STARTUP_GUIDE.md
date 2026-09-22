# AutoDRIVE startup guide

This guide covers the validated local workflow:

1. start the simulator;
2. start the source-mounted ROS development stack;
3. optionally record a real run for offline scoring.

The normal controller is MPC. The development launch also supports Pure
Pursuit and FTG explicitly.

## 1. Start the simulator

The simulator is maintained outside this repository. The following starts the
pinned headless simulator container on the API port used by the bridge:

```bash
SIM_NAME=autodrive_sim
docker rm -f "$SIM_NAME" 2>/dev/null || true
docker run -d --name "$SIM_NAME" \
  --network host --ipc host --privileged --gpus all \
  --entrypoint /bin/bash \
  autodriveecosystem/autodrive_roboracer_sim:2026-icra-compete \
  -lc '
    set -e
    Xvfb :123 -screen 0 1920x1080x24 -ac >/tmp/autodrive_sim_xvfb.log 2>&1 &
    xvfb_pid=$!
    export DISPLAY=:123
    sleep 2
    kill -0 "$xvfb_pid"
    cd /home/autodrive_simulator
    exec ./AutoDRIVE\ Simulator.x86_64 -batchmode \
      -ip 127.0.0.1 -port 4567 \
      -logFile /tmp/autodrive_sim_unity.log
  '
```

Check that it is running:

```bash
docker ps --filter name=autodrive_sim
docker logs -f autodrive_sim
```

Leave the simulator running while starting ROS. The bridge should eventually
print `Connected!`.

### 1.1 Start the simulator with a visible GUI

The ROS launch and competition topic policy are unchanged. Replace the
headless simulator command above with one of these GUI options.

For a standalone simulator build on the host, run this from the simulator
checkout:

```bash
cd /home/akselmo/Documents/GitHub/AutoDRIVE/Builds
./AutoDRIVE-Simulator.x86_64 -ip 127.0.0.1 -port 4567
```

If using the pinned simulator container, forward the host X11 display and do
not use `-batchmode` or `Xvfb`:

```bash
DISPLAY_VALUE="${DISPLAY:-:0}"
export DISPLAY="$DISPLAY_VALUE"
xhost +local:root
docker rm -f autodrive_sim_gui 2>/dev/null || true
docker run --rm --name autodrive_sim_gui \
  --network host --ipc host --privileged --gpus all \
  -e DISPLAY="$DISPLAY_VALUE" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --entrypoint /bin/bash \
  autodriveecosystem/autodrive_roboracer_sim:2026-icra-compete \
  -lc '
    set -e
    cd /home/autodrive_simulator
    exec ./AutoDRIVE\ Simulator.x86_64 \
      -ip 127.0.0.1 -port 4567 \
      -logFile /tmp/autodrive_sim_unity_gui.log
  '
xhost -local:root
```

Keep that terminal open. Start the ROS stack from another terminal with
`./tools/start_dev.sh`; when the simulator window is open, use its `Connect`
control if it has not connected automatically. The bridge should then print
`Connected!`. The X11 permission is removed when the simulator command exits.

## 2. Start the development stack

From the repository root:

```bash
./tools/start_dev.sh
```

This builds `sdu-apex-autodrive:dev` when necessary, mounts the source tree,
runs the runtime topic-policy check, and starts one legal bridge, sensor odom,
EKF, AMCL, controller, and actuator chain. MPC is the default.

For a quick restart using the existing image:

```bash
AUTODRIVE_REBUILD=0 ./tools/start_dev.sh
```

If the development tag is not present locally and Docker cannot resolve the
registry, use the exact local image used by the validated high-speed
15-lap run instead of rebuilding:

```bash
SDU_APEX_IMAGE=sdu-apex-autodrive:cleanup-20260921-final3 \
AUTODRIVE_REBUILD=0 ./tools/start_dev.sh
```

This tag currently resolves to image ID
`sha256:0402847abdbd654b2418e981bb676ae7808091e70d25e855ac314aa61b5e3e89`,
which is the image recorded for that run. It starts the installed binaries
from that image with the current source tree mounted for configuration and
policy checks. Rebuild the image after production source changes once Docker
registry DNS is working again.

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
docker stop autodrive_sim
```

Do not start a second stack before stopping the first one. The development
launch rejects duplicate runtime nodes.

## 3. Exact competition-container rehearsal

Build the self-contained image without mounting the source tree:

```bash
docker build -t sdu-apex-autodrive:competition .
```

With the simulator already running, start the fixed MPC-only launch:

```bash
docker run --rm --name sdu_apex_autodrive_competition \
  --network host --ipc host --privileged --gpus all \
  sdu-apex-autodrive:competition
```

This path uses the installed production files and `competition.launch.py`. It
does not start RViz, a shadow controller, a simulator-truth input, or a
parameter overlay. Follow the container output with:

```bash
docker logs -f sdu_apex_autodrive_competition
```

The competition launch starts MPC only after the AMCL warm-up. The runtime
nodes consume simulator LiDAR, IMU, encoders, steering, and throttle feedback,
plus the team-derived `/odom`, `/ekf_odom`, and `/current_map_pose` topics.

## 4. Record a real run

Recording all topics is allowed for debugging and offline analysis, but the
active controller, odom, EKF, and AMCL must not subscribe to simulator truth,
IPS, simulator odom, lap, collision, reset, result, or absolute `/tf` topics.

Start the recorder in a second terminal. Create only the run parent directory;
`ros2 bag record` must create the final `run` directory itself.

```bash
RUN_ID=real_run_$(date +%Y%m%d_%H%M%S)
mkdir -p "live_runs/$RUN_ID"

docker run -d --name "rec_$RUN_ID" \
  --network host --ipc host --privileged --gpus all \
  -v "$PWD:/workspace/src:rw" \
  --entrypoint /bin/bash \
  sdu-apex-autodrive:dev \
  -lc '
    set -e
    source /opt/ros/humble/setup.bash
    source /home/autodrive_devkit/install/setup.bash
    exec ros2 bag record -a \
      -o /workspace/src/live_runs/'"$RUN_ID"'/run
  '
```

After the run:

```bash
docker stop "rec_$RUN_ID"
ros2 bag info "live_runs/$RUN_ID/run"
```

For a scored run, stop recording and the ROS/simulator containers at the first
collision. Never use post-collision samples for estimator or MPC conclusions.

## 5. Preflight and useful checks

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

## Mapping

Mapping remains separate from racing:

```bash
ros2 launch sdu_apex_autodrive mapping.launch.py
```

It uses the legal LiDAR/IMU/encoder-derived path and writes maps under
`f1tenth_planning/maps/`. Mapping outputs are not racing racelines.
