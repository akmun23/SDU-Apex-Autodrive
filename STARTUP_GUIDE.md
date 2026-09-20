# AutoDRIVE startup guide

There are separate local-development and competition startup contracts. The
GUI/source-built player workflow below is for local development only. The
competition contract uses the official compete image and its fixed entrypoint.

## Competition image

Build the submitted image from the pinned official 2026 IROS compete base:

```bash
cd /home/akselmo/Documents/GitHub/SDU-Apex-Autodrive
docker build -t sdu-apex-autodrive:iros-2026 .
```

Run it without Compose, source mounts, environment overrides, or a second
launch shell:

```bash
docker run --name autodrive_roboracer_api --rm -it \
  --network=host --ipc=host --privileged --gpus all \
  sdu-apex-autodrive:iros-2026
```

The image entrypoint starts `competition.launch.py` unconditionally. That
launch includes the installed official headless API bridge unchanged, starts
the legal sensor-derived odometry/EKF/AMCL/MPC/actuator chain, waits for AMCL
warm-up, and does not start development bridges, shadow MPC, recorders, RViz,
ground-truth tools, or parameter overrides. The simulator is the official
competition environment; do not mount this repository into the container.

The competition package base is pinned in `Dockerfile` to
`autodriveecosystem/autodrive_roboracer_api:2026-iros-compete`.

## Local GUI development

This workflow uses two terminals by design: one terminal owns Unity, and the
other owns the ROS 2 development/controller container.

```text
Unity compete player -> AutoDRIVE bridge -> odometry/EKF/AMCL -> MPC
                                                        -> actuator interface
```

The ROS 2 development workspace runs in Docker. The verified source-built
Unity player runs directly from the host so it uses the patched 40 Hz build
and opens its normal graphical window.
The controller terminal starts exactly one bridge and one controller.

## One-time setup

Make sure Docker and Unity are available:

```bash
cd /home/akselmo/Documents/GitHub/SDU-Apex-Autodrive
test -x /home/akselmo/Unity/Hub/Editor/2022.3.52f1/Editor/Unity
test -x /home/akselmo/Documents/GitHub/AutoDRIVE/Builds/AutoDRIVE-Simulator.x86_64
```

The verified player is:

```bash
/home/akselmo/Documents/GitHub/AutoDRIVE/Builds/AutoDRIVE-Simulator.x86_64
```

It was built from `Assets/Scenes/RoboRacer - Sim Racing.unity` with the local
camera-disabled/timing, parallel/reusable-LiDAR, and per-scan LiDAR transport
cache source changes. The Linux
Unity editor cannot import the source SketchUp track prefab, so the build uses
the exact official `SRL 2026 ICRA Track.obj` geometry recovered from the local
`2026-icra-compete` player, combined into both official mesh submeshes and
assigned the official Black Fabric and Shiny Aluminium materials. Its
MeshCollider uses the same combined geometry, so the visible walls and LiDAR
collision geometry match the competition player. The original missing prefab
reference is removed from the built player to avoid a duplicate collider. The
launcher refuses to fall back to the packaged Docker simulator because that
would invalidate the 40 Hz requirement.

If the player is ever missing, rebuild it with:

```bash
/home/akselmo/Unity/Hub/Editor/2022.3.52f1/Editor/Unity \
  -batchmode -quit \
  -projectPath /home/akselmo/Documents/GitHub/AutoDRIVE \
  -executeMethod BuildLinuxPlayer.BuildCompeteCurrentScene \
  -logFile /tmp/autodrive-compete-build.log
```

`BuildCompeteCurrentScene` builds a temporary copy of the checked-out scene and
does not run the older track-recovery helper that edits the source scene.

## Terminal 1 — simulator

Run this first and leave it running:

```bash
cd /home/akselmo/Documents/GitHub/SDU-Apex-Autodrive
./tools/start_simulator.sh
```

This starts only the verified source-built simulator. It deliberately uses
neither `-batchmode` nor `-no-graphics`, so the Unity GUI remains visible. The
The socket is enabled from the GUI: after Terminal 2 is running, press
`Connect` once in the simulator window. The `-ip` and `-port` values are filled
from the launcher arguments.

## Terminal 2 — development/controller

Open a second terminal, run this from the same repository, and leave it
running before pressing `Connect` in the Unity window:

```bash
cd /home/akselmo/Documents/GitHub/SDU-Apex-Autodrive
./tools/start_dev.sh
```

The development terminal starts:

1. one `autodrive_bridge_40hz`;
2. sensor odometry, EKF, map server, and CUDA AMCL;
3. MPC with the production model, weights, map, and mintime raceline;
4. the actuator interface.

The start script rebuilds the derived image by default; Docker reuses cached
layers when nothing changed. Set `AUTODRIVE_REBUILD=0` only when deliberately
using an already verified image. Once the socket connects, telemetry flows.
There is no second ROS launch and no second bridge. The GUI `Connect` button
enables the simulator socket; it does not create a second controller process.
AMCL starts with global particle
initialization and a scan-supported saved map-start prior. MPC waits for the
configured AMCL warm-up delay before it is loaded, and the first lap remains
ramped by the active controller policy. A collision is terminal: the stack
stops and does not reset or continue driving. Use the explicit FTG or Pure
Pursuit commands below only for diagnostics.

Do not also run `ros2 launch ... controller.launch.py`: `start_dev.sh` already
starts the unified launch.

The default controller is MPC. For an explicit diagnostic alternative:

```bash
SDU_APEX_CONTROLLER=ftg ./tools/start_dev.sh
# or
SDU_APEX_CONTROLLER=pure_pursuit ./tools/start_dev.sh
```

The default MPC uses the frozen
`f1tenth_mpc/config/mpc_iros_2026_competition.yaml` profile. Diagnostic recorders and per-cycle
MPC JSON diagnostics are also off by default so normal GUI operation does not
add file, serialization, or DDS load to the 40 Hz path. Enable MPC diagnostics
explicitly when collecting a controller trace:

```bash
SDU_APEX_MPC_PUBLISH_DIAGNOSTICS=true ./tools/start_dev.sh
```

To use another verified source-built player deliberately, override the path in
Terminal 1:

```bash
AUTODRIVE_PLAYER=/absolute/path/to/another/AutoDRIVE-Simulator.x86_64 \
  ./tools/start_simulator.sh
```

For a diagnostic-only headless run, opt in explicitly:

```bash
AUTODRIVE_BATCHMODE=1 ./tools/start_simulator.sh
```

This still does not add `-no-graphics`; the normal competition/debug workflow
is the visible GUI command above.

## Checks from a third terminal, if needed

```bash
docker compose exec workspace bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/autodrive_devkit/install/setup.bash
  source /workspace/install/setup.bash
  ros2 node list
'
```

## Stop everything

Stop both terminals with `Ctrl-C`, or run this from any terminal:

```bash
./tools/stop_autodrive.sh
```

This stops the development container but keeps its image and build cache. The
host Unity process is stopped with `Ctrl-C` in Terminal 1.

## What not to use

- Do not use `-no-graphics` with this HDRP simulator.
- Do not add `-batchmode` to the normal visible-GUI workflow.
- Do not run the old separate bridge, odometry, actuator, or controller
  commands alongside the unified launch.
- The packaged Docker simulator is not part of the supported local workflow;
  the source-built host player is used for the visible GUI and 40 Hz path.

The production ROS inputs are the native LiDAR/IMU/encoder streams. Simulator
ground truth remains diagnostics-only and is not used to drive the car.
