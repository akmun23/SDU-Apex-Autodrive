# Runtime topic policy

The fixed `competition.launch.py` is restricted to the official competition
sensor inputs and team-derived state. Its MPC and localization nodes must not
subscribe to simulator ground truth, race-result, collision, reset, or
simulator transform data. The development and mapping launches are separate
debug workflows and may use restricted streams for analysis.

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
the competition stack:

- simulator pose/IPS and simulator `/odom`
- `/autodrive/roboracer_1/collision_count`
- `/autodrive/reset_command`
- lap/time/result topics, including `speed`
- absolute `/tf`

The competition launch uses the repository's rate-controlled wrapper around the
official headless API bridge. The wrapper only subscribes to the permitted
steering and throttle command topics. The upstream API may publish restricted
telemetry, but the competition MPC, odometry, EKF, and AMCL do not consume it.
The competition launch has no transform-topic remaps and disables team
transform publication. The MPC consumes only team `/odom` and
`/current_map_pose`. Development mapping is intentionally separate: its SLAM
mapper uses simulator `/tf` and `lap_count` to build a map offline, and the
competition entrypoint never starts that launch or subscribes to those topics.

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
