# SDU Apex AutoDRIVE

ROS 2 Humble autonomy workspace for the AutoDRIVE RoboRacer simulator. The
workspace now builds in a reproducible container and uses AutoDRIVE-native
topics and frames.

The actuator adapter is intentionally **disarmed by default**. Its interface
uses the documented AutoDRIVE 2026 normalized command limits rather than
physical VESC calibration. Steering angles are converted exactly to normalized
input; target speed is controlled from AutoDRIVE odometry using bounded
forward throttle.

## Implemented runtime

```text
AutoDRIVE LiDAR ──> FTG ─────────────────────────────┐
                                                    │
AutoDRIVE LiDAR ──> SLAM / Nav2 AMCL                │
                         │                           │
                         └─> scan splitter           │
                             └─> lateral planner     │
                                 ├─> Pure Pursuit ───┤
                                 ├─> Stanley ────────┤
                                 └─> CPU MPC ────────┤
                                                    v
                      /cmd/* ──> Ackermann mux ──> adapter
                                                    │
                              Float32 steering/throttle
                                                    │
                                                    v
                                           AutoDRIVE bridge
```

Runtime frame contract:

```text
map -> world -> roboracer_1 -> lidar
```

AutoDRIVE owns `world -> roboracer_1` and sensor transforms. SLAM or AMCL owns
`map -> world`. Do not add a second publisher for those transforms.

## Build

From this repository:

```bash
docker compose build workspace
```

The image is based on the official AutoDRIVE RoboRacer API image and builds all
nine ROS packages under ROS 2 Humble. Rebuild after changing source or default
configuration.

Open a sourced workspace shell with:

```bash
docker compose run --rm workspace
```

The repository is mounted at `/workspace/src`; the built overlay is
`/workspace/install`.

## Start AutoDRIVE

The two stock containers may already exist. Check them on the host:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

Start the graphical simulator in its existing container:

```bash
docker exec -it autodrive_roboracer_sim bash
cd /home/autodrive_simulator
./AutoDRIVE\ Simulator.x86_64
```

Then start the ROS bridge in the API container:

```bash
docker exec -it autodrive_roboracer_api bash
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
ros2 launch autodrive_roboracer bringup_graphics.launch.py
```

In the simulator UI:

1. Set the RoboRacer vehicle to **Autonomous**.
2. Press **Connect**.
3. Keep the simulator UI accessible as the stop mechanism during early tests.

Those UI actions must be done by the user. A running container alone does not
activate the simulated sensors.

## Verify the bridge before running autonomy

In the workspace shell:

```bash
ros2 topic list | grep '^/autodrive/roboracer_1/'
ros2 topic type /autodrive/roboracer_1/lidar
ros2 topic type /autodrive/roboracer_1/odom
ros2 topic type /autodrive/roboracer_1/imu
ros2 topic hz /autodrive/roboracer_1/lidar
ros2 topic hz /autodrive/roboracer_1/odom
ros2 run tf2_ros tf2_echo world roboracer_1
ros2 run tf2_ros tf2_echo roboracer_1 lidar
```

Expected command types:

```bash
ros2 topic type /autodrive/roboracer_1/steering_command
ros2 topic type /autodrive/roboracer_1/throttle_command
```

Both must be `std_msgs/msg/Float32`.

## First smoke test: LiDAR to FTG

Run:

```bash
ros2 launch sdu_apex_autodrive control_ftg.launch.py
```

FTG publishes bounded Ackermann requests on `/cmd/ftg`. The mux selects them,
but the adapter remains disarmed and sends only neutral output. Inspect first:

```bash
ros2 topic echo /cmd/ftg
ros2 topic echo /cmd/selected
ros2 topic echo /autodrive/adapter/armed
```

Read the [AutoDRIVE actuator interface](docs/AUTODRIVE_INTERFACE.md) before the
first moving test. Keep the simulator reset control visible. While one
controller is running, arm once:

```bash
ros2 topic pub --once /autodrive/adapter/enable std_msgs/msg/Bool '{data: true}'
```

Disarm immediately with:

```bash
ros2 topic pub --once /autodrive/adapter/enable std_msgs/msg/Bool '{data: false}'
```

The adapter sends neutral on startup, disarm, invalid commands, stale commands,
stale odometry, and shutdown. The mux is the only Ackermann arbiter; the
adapter is the only node that should publish final AutoDRIVE actuator commands.

## Mapping

Run SLAM after the bridge and sensors are active:

```bash
ros2 launch f1tenth_stack mapping.launch.py
```

Move around the whole track slowly using a verified simulator control method.
Save the finished map into the host-mounted repository:

```bash
mkdir -p /workspace/src/autodrive_artifacts/maps
ros2 run nav2_map_server map_saver_cli \
  -f /workspace/src/autodrive_artifacts/maps/autodrive_track
```

Confirm `/map` looks correct in RViz before continuing. Existing maps in
`f1tenth_planning/maps` came from older environments and must not be assumed to
match the selected AutoDRIVE track.

## Localization

Launch the map server and Nav2 AMCL with the map made above:

```bash
ros2 launch f1tenth_stack localization_nav2.launch.py \
  map:=/workspace/src/autodrive_artifacts/maps/autodrive_track_multi_lap.yaml
```

Set the initial pose in RViz. Verify the complete TF chain:

```bash
ros2 run tf2_ros tf2_echo map world
ros2 run tf2_ros tf2_echo map roboracer_1
ros2 run tf2_ros tf2_echo map lidar
```

## Planning and path following

Create a `map`-frame raceline for the new AutoDRIVE map; do not reuse an old
track trajectory unless its map and coordinates are proven identical. Store
generated files under `autodrive_artifacts/trajectories`.

Start map-aware obstacle extraction and the lateral planner:

```bash
ros2 launch f1tenth_stack perception.launch.py \
  trajectory_file:=/workspace/src/autodrive_artifacts/trajectories/autodrive_raceline_safe.csv \
  avoidance_enabled:=false
```

Verify both outputs:

```bash
ros2 topic hz /scan_obstacles
ros2 topic echo --once /local_raceline
```

`/local_raceline.header.frame_id` must be `map`, with nonempty finite poses.

Start the mux/adapter in one shell:

```bash
ros2 launch sdu_apex_autodrive autodrive_base.launch.py
```

Then run exactly one path controller in another shell:

```bash
# Pure Pursuit
ros2 launch f1tenth_control pure_pursuit_autodrive.launch.py

# or Stanley
ros2 launch f1tenth_control stanley_autodrive.launch.py

# or CPU Riccati-ADMM MPC
ros2 launch mpc_riccati mpc_autodrive.launch.py
```

Inspect `/cmd/pure_pursuit`, `/cmd/stanley`, or `/cmd/mpc` before arming. The
checked-in AutoDRIVE safe raceline starts at `1.5 m/s` and keeps at least
`0.10 m` wall clearance beyond the `0.270 m` car body.

## Important limitations

- AutoDRIVE throttle dynamics still require controller tuning. The adapter
  therefore ships disarmed; steering conversion is source-backed.
- Exact geometry, mass, COM, steering limits, and top speed match the official
  2026 simulator specification. Yaw inertia and MPC tire-fit parameters remain
  unverified and require identification before high-speed MPC tuning.
- CPU Nav2 AMCL and CPU MPC are the supported baseline.
- CUDA localization is not enabled in this image. Add a CUDA 12.x workflow for
  the GTX 1080 Ti (`sm_61`) only after the CPU stack is validated.
- MPCC, FPGA/Kria, VESC, Hokuyo hardware, and the old gym simulator are not part
  of the installed AutoDRIVE runtime.

## Main configuration files

- `sdu_apex_autodrive/config/adapter.yaml`: final actuator safety boundary
- `sdu_apex_autodrive/config/mux.yaml`: controller priority/timeouts
- `f1tenth_control/config/ftg_autodrive.yaml`: first-drive controller limits
- `f1tenth_control/config/path_tracking_autodrive.yaml`: PP/Stanley limits
- `f1tenth_localization/config/nav2_amcl_params.yaml`: map localization
- `f1tenth_system/f1tenth_stack/config/slam_params.yaml`: mapping
- `f1tenth_system/f1tenth_lidar/config/scan_splitter.yaml`: obstacle scan
- `f1tenth_lateral_planner/config/lateral_planner.yaml`: local path generation
