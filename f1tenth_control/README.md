# F1Tenth control

ROS 2 Humble controller components for the AutoDRIVE RoboRacer stack.

The package contains three controller plugins:

- `f1tenth_control::FTGNode` — LiDAR-only Follow The Gap.
- `f1tenth_control::PurePursuitNode` — raceline tracking with AMCL pose and
  encoder/IMU odometry.
- `f1tenth_control::StanleyNode` — raceline tracking with AMCL pose and
  encoder/IMU odometry.

All controllers publish physical-unit commands to `/cmd/controller` as
`ackermann_msgs/msg/AckermannDriveStamped`. The single
`sdu_apex_autodrive/actuator_interface` node owns conversion to AutoDRIVE's
normalized steering and throttle topics.

## Runtime interfaces

FTG subscribes to:

- `/autodrive/roboracer_1/lidar` (`sensor_msgs/msg/LaserScan`)

Pure Pursuit and Stanley additionally subscribe to:

- `/odom` (`nav_msgs/msg/Odometry`) from encoder/IMU dead reckoning
- `/amcl_pose` (`geometry_msgs/msg/PoseWithCovarianceStamped`)
- `/local_raceline` when a lateral planner is enabled

The path frame is `map`; the command frame is `base_link`. Controller commands
are deliberately kept separate from native actuator outputs so that steering
conversion, speed control, timeout handling, and neutral-on-failure behavior
exist at one boundary.

## Build and run

Build with the workspace instructions in the root [README](../README.md). The
supported launch entry point is:

```bash
ros2 launch sdu_apex_autodrive controller.launch.py controller:=ftg
ros2 launch sdu_apex_autodrive controller.launch.py controller:=pure_pursuit
ros2 launch sdu_apex_autodrive controller.launch.py controller:=stanley
```

Use the MPC controller through the same launch file with
`controller:=mpc`; it starts the lateral planner and this package is not
required for that controller path.

The checked-in defaults follow the AutoDRIVE RoboRacer geometry: wheelbase
`0.324 m`, steering limit `±0.5236 rad`, and centre-steering rate `3.2 rad/s`.
The controller accepts the documented simulator envelope; the raceline
velocity column, curvature, clearance, acceleration, and actuator limits
regulate the actual corner speed.

## Configuration

- `config/ftg_autodrive.yaml`
- `config/path_tracking_autodrive.yaml`

The launch file loads the appropriate YAML and selects exactly one controller.
Do not start the deleted legacy hardware launches or an additional command
mux/bridge.
