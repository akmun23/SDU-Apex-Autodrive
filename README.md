# SDU Apex AutoDRIVE

ROS 2 Humble autonomy workspace for the AutoDRIVE RoboRacer simulator.
One launch starts the official bridge, exactly one controller, its required
localization/planning nodes, and the actuator interface.

```text
AutoDRIVE sensors -> controller -> /cmd/controller -> actuator interface
                                                        |
                                    normalized steering + throttle
                                                        |
                                                 AutoDRIVE bridge
```

There is no arming topic, command mux, or guarded-run wrapper. The actuator
interface becomes active when both controller commands and odometry are fresh.
It publishes neutral on startup, stale/missing input, invalid values, zero
target speed, and shutdown.

## Build and start

From this repository:

```bash
docker compose build workspace
docker compose up -d workspace
docker exec -it sdu_apex_autodrive bash
```

Start `AutoDRIVE Simulator.x86_64` in the simulator container. In its UI,
select **Autonomous** and press **Connect**. Do not start another ROS bridge;
the controller launch owns the single bridge instance.

Run FTG, which needs no map localization:

```bash
ros2 launch sdu_apex_autodrive controller.launch.py \
  controller:=ftg with_rviz:=true
```

Stop with `Ctrl-C`. Other supported controllers use the checked-in map and
raceline and automatically request AMCL global localization:

```bash
ros2 launch sdu_apex_autodrive controller.launch.py controller:=pure_pursuit
ros2 launch sdu_apex_autodrive controller.launch.py controller:=stanley
ros2 launch sdu_apex_autodrive controller.launch.py controller:=mpc
```

Defaults:

- map: `autodrive_artifacts/maps/autodrive_track_multi_lap.yaml`
- raceline: `autodrive_artifacts/trajectories/autodrive_raceline_safe.csv`
- command path: `/cmd/controller`
- native output: `/autodrive/roboracer_1/{steering,throttle}_command`

Pure Pursuit, Stanley, and MPC wait for five consecutive low-covariance AMCL
poses before driving. MPC also starts the scan splitter and lateral planner.
Override `map:=...` or `trajectory:=...` only with files from the same Unity
track.

## Actuation

AutoDRIVE accepts normalized steering and throttle. Steering is the exact
conversion

```text
steering_command = clamp(steering_angle_rad / 0.5236, -1, 1)
```

`AckermannDrive.speed` is a target in metres per second, while AutoDRIVE only
accepts motor throttle. The actuator interface therefore uses a feedforward
table plus signed PI feedback and asymmetric slew limiting. This removes the
old bang-bang throttle cut when actual speed crosses target. Gains and limits
can be changed at runtime; the feedforward arrays require restart.

The default feedforward table is provisional low-speed track data, with a
`0.10` normalized throttle ceiling. Replace it after wall-less raw
characterization before high-speed MPC tuning. See
[the actuator contract](docs/AUTODRIVE_INTERFACE.md).

FTG collision geometry uses the verified `0.270 m` car width plus an extra
`0.10 m` margin on each side. It does not suppress throttle as a steering
guard.

## Calibration and recording

The deterministic suite owns the bridge and required command publisher:

```bash
# Safe, non-driving capture
ros2 launch sdu_apex_autodrive test_suite.launch.py test:=sensor_record

# Raw native-input tests: use only in a wall-less Unity scene
ros2 launch sdu_apex_autodrive test_suite.launch.py test:=throttle_sweep
ros2 launch sdu_apex_autodrive test_suite.launch.py test:=throttle_steps
ros2 launch sdu_apex_autodrive test_suite.launch.py test:=zero_throttle_decel
ros2 launch sdu_apex_autodrive test_suite.launch.py test:=steering_steps

# Closed-loop target-speed tests
ros2 launch sdu_apex_autodrive test_suite.launch.py test:=speed_steps
ros2 launch sdu_apex_autodrive test_suite.launch.py test:=speed_ramp
```

Results go to `autodrive_artifacts/calibration/raw`. Driving tests abort on
collision, telemetry loss, excessive speed, or timeout and publish neutral on
exit.

Generate an empty ROS occupancy map with:

```bash
ros2 run sdu_apex_autodrive create_open_map \
  --output /workspace/src/autodrive_artifacts/maps/open_map
ros2 launch sdu_apex_autodrive open_map.launch.py \
  map:=/workspace/src/autodrive_artifacts/maps/open_map.yaml
```

This only removes occupancy-grid obstacles. It cannot remove collision walls
from the Unity simulator. Raw straight-line characterization needs a separate
wall-less Unity scene/build.

## Verified simulator model

The model follows the official 2026 RoboRacer specification: length `0.5000
m`, width `0.2700 m`, wheelbase `0.3240 m`, track width `0.2360 m`, mass
`3.906 kg`, rear-axle-to-COM `0.15532 m`, steering limit `0.5236 rad`, steering
rate `3.2 rad/s`, and stated top speed `22.88 m/s`. Yaw inertia and tire-model
fits remain unverified, so high-speed MPC still needs system identification.

Runtime frame contract:

```text
map -> world -> roboracer_1 -> lidar
```

AutoDRIVE owns `world -> roboracer_1` and sensor transforms. AMCL owns
`map -> world`; never add duplicate transform publishers.

## Main configuration

- `sdu_apex_autodrive/config/actuator_interface.yaml`
- `sdu_apex_autodrive/config/calibration.yaml`
- `sdu_apex_autodrive/config/localization_bootstrap.yaml`
- `f1tenth_control/config/ftg_autodrive.yaml`
- `f1tenth_control/config/path_tracking_autodrive.yaml`
- `f1tenth_localization/config/nav2_amcl_params.yaml`
- `f1tenth_lateral_planner/config/lateral_planner.yaml`
