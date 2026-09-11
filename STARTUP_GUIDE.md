# AutoDRIVE startup guide

This is the single supported local startup path for the compete track. It uses
two terminals by design: one terminal owns Unity, and the other owns the ROS 2
development/controller container.

```text
Unity compete player -> AutoDRIVE bridge -> odometry/EKF/AMCL -> Pure Pursuit
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
camera-disabled/timing and parallel/reusable-LiDAR source changes. The Linux
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
  -executeMethod BuildLinuxPlayer.BuildCompete \
  -logFile /tmp/autodrive-compete-build.log
```

## Terminal 1 — simulator

Run this first and leave it running:

```bash
cd /home/akselmo/Documents/GitHub/SDU-Apex-Autodrive
./tools/start_simulator.sh
```

This starts only the verified source-built simulator. It deliberately uses
neither `-batchmode` nor `-no-graphics`, so the Unity GUI remains visible. The
the socket is enabled from the GUI: after Terminal 2 is running, press
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
3. Pure Pursuit with the production map and mintime raceline;
4. the actuator interface.

If the bridge source has changed since the container image was built, run the
first start with `AUTODRIVE_REBUILD=1 ./tools/start_dev.sh`; otherwise the
container may still contain the old bridge installation. Once the socket
connects, telemetry flows. There is no second ROS launch and
no second bridge. The GUI `Connect` button enables the simulator socket; it
does not create a second controller process. AMCL starts with global particle
initialization and a scan-supported saved map-start prior. Pure Pursuit may use
only the confirmed start-gated provisional pose, at the 1.5 m/s startup cap;
after 0.75 m of travel AMCL transitions to local tracking, and PP ramps its
speed cap across the first completed lap. This gives AMCL motion at low speed
without allowing a visually plausible closed-track alias to command a crash.
Use the FTG command below when you need an immediate moving sensor run without
raceline localization.

Do not also run `ros2 launch ... controller.launch.py`: `start_dev.sh` already
starts the unified launch.

The default controller is Pure Pursuit. For an immediate moving FTG run:

```bash
SDU_APEX_CONTROLLER=ftg ./tools/start_dev.sh
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
