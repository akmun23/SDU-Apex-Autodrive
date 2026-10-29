# Runtime topic policy

The maintained autonomous stack consists of `competition.launch.py`, the
development wrapper around `controller.launch.py`, our 40 Hz adapter around
the upstream AutoDRIVE bridge,
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

## Bridge timing guidance — 2026-10-29

The user supplied AutoDRIVE guidance explicitly permitting timer callbacks
and pointing to the upstream bridge's commented 40 Hz publisher. Therefore
timer-based ROS publication is not, by itself, a prohibited bridge change.
The earlier blanket uncertainty about whether a timer may be implemented
was too broad. This guidance is recorded from the user's message, not an
independently retrieved organizer ruling about this entire implementation.

Verified upstream source:
[autodrive_bridge.py, revision 894fe288](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/blob/894fe288ce94a50dadc1917c6349271c66b10475/autodrive_devkit/autodrive_roboracer/autodrive_bridge.py).
Its commented `timer_callback` publishes cached telemetry, and the commented
timer period is 0.025 seconds. That schedules ROS publication; it does not
prove that every publication contains a fresh simulator sample or provide a
Unity physics timestamp.

Our adapter imports the installed upstream decoder without editing its files.
It goes beyond that timer example: it paces outgoing `Bridge` requests,
associates responses by FIFO, shares the associated local request timestamp
across packet messages, omits camera processing, and receives permitted
actuator commands on a dedicated listener. Local request time is an
association convention, not a measured Unity sample time; FIFO cannot prove
the simulator's exact per-physics-step command application.

Keep these adapter details distinct from the timer permission when describing
the competition image. The supplied guidance supports timer customization;
it is not evidence that every additional adapter behavior has been reviewed.
All permitted-input restrictions still apply, and development packet timing
and reset inputs remain disabled in the competition launch. Topic-policy
checks verify that scope, not approval of the complete image.

Run the guard before launching:

```bash
python3 tools/verify_runtime_topic_policy.py
```
