# MPC

This package builds one production controller: a 30-stage, 40 Hz RTI MPC
whose plant rollout uses the source-derived Unity/PhysX vehicle state. Its
optimized inputs are steering rate and normalized throttle. Trajectory speed
and acceleration are tracking references; they are not actuator commands.

The ROS output is `sdu_apex_msgs/DriveCommand` (steering angle in radians,
normalized throttle in `[0, 1]`). The controller does not subscribe to an
Ackermann speed command or run a separate longitudinal speed controller.
The actuator bridge sends zero throttle as the simulator's brake branch.

Runtime settings and the single installed response surface are
`config/mpc_competition.yaml` and `config/yaw_response_surface_y1.csv`.
Production MPC reads team odometry, localized map pose, the typed plant state,
and permitted sensor/actuator feedback. It does not read simulator ground
truth.

Build from the repository root in the ROS 2 Humble environment:

```bash
source /opt/ros/humble/setup.bash
colcon build --merge-install --packages-up-to f1tenth_mpc
```
