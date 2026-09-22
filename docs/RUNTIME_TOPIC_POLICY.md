# Runtime topic policy

The normal `./tools/start_dev.sh` launch is restricted to the official
competition sensor inputs and team-derived state. It must not subscribe to
simulator ground truth, race-result, collision, reset, or simulator TF topics.

The official AutoDRIVE simulator guide defines these sensor inputs as allowed:

- `/autodrive/roboracer_1/front_camera`
- `/autodrive/roboracer_1/imu`
- `/autodrive/roboracer_1/left_encoder`
- `/autodrive/roboracer_1/lidar`
- `/autodrive/roboracer_1/right_encoder`
- `/autodrive/roboracer_1/steering`
- `/autodrive/roboracer_1/throttle`

The team may publish the actuator outputs
`/autodrive/roboracer_1/steering_command` and
`/autodrive/roboracer_1/throttle_command`. The controller may also consume
team-generated `/odom`, `/ekf_odom`, and `/current_map_pose`.

The following are restricted during autonomous runtime and are not consumed by
the racing launch:

- simulator pose/IPS and simulator `/odom`
- `/autodrive/roboracer_1/collision_count`
- `/autodrive/reset_command`
- lap/time/result topics, including `speed`
- absolute `/tf`

The competition launch uses the repository's rate-controlled wrapper around the
official headless API bridge. The wrapper only subscribes to the permitted
steering and throttle command topics and may still publish simulator telemetry
as part of the API protocol; no team node subscribes to those restricted
outputs. Team TF is remapped to `/sdu/tf` and `/sdu/tf_static`.

The unqualified `/odom` name in this repository is the team observer's own
derived topic; it is not connected to or remapped from
`/autodrive/roboracer_1/odom`. A live graph check must continue to show zero
subscribers on the official simulator odometry topic. Renaming team `/odom` is
therefore not required for legality, and would only create a large unnecessary
migration across MPC, PP, EKF, AMCL, recordings, and replay tools.

Collision aborts cannot be implemented from `collision_count` in a compliant
autonomous run. A collision must instead invalidate the run externally or be
identified from an offline rosbag/recording; no restricted collision signal is
fed back into the controller.

The old ground-truth/model-identification utilities are not part of the normal
controller launch. Any future offline analysis must use a recorded bag or
already captured artifact and must not be started alongside autonomous control.

Reference: [AutoDRIVE 2026 Technical Guide](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/).
