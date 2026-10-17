# Runtime topic policy

The maintained autonomous stack consists of `competition.launch.py`, the
development wrapper around `controller.launch.py`, the official 40 Hz bridge,
team odometry/EKF/AMCL, the MPC, and the direct actuator interface. Every
controller and localization input must come from permitted sensors or team-
derived state. Development tools may record data; they do not feed restricted
simulator telemetry back into autonomous control.

Permitted sensor inputs used by the stack:

- `/autodrive/roboracer_1/imu`
- `/autodrive/roboracer_1/left_encoder`
- `/autodrive/roboracer_1/lidar`
- `/autodrive/roboracer_1/right_encoder`
- `/autodrive/roboracer_1/steering`
- `/autodrive/roboracer_1/throttle`

Team-derived inputs include `/odom`, `/odom/unity_plant_state`, `/ekf_odom`,
and `/current_map_pose`. The MPC may use the typed plant state published by
team odometry, but it must not consume Unity truth or simulator result topics.
Actuator outputs are steering command and normalized throttle command.

Restricted autonomous inputs include simulator ground-truth pose/odometry,
collision and lap/result topics, reset commands, and all transform topics or
aliases. The topic guard checks the competition launch, bridge, odometry,
localization, and MPC sources for those subscriptions. The official bridge
wrapper disables its restricted reset subscription.

Run the guard before launching:

```bash
python3 tools/verify_runtime_topic_policy.py
```
