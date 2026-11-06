# MPC

This package builds one production controller: a 30-stage, 40 Hz RTI MPC
whose plant rollout uses the source-derived Unity/PhysX vehicle state. Its
optimized inputs are steering target angle and normalized throttle. Trajectory
speed and acceleration are tracking references; they are not actuator commands.

The ROS output is `sdu_apex_msgs/DriveCommand` (steering angle in radians,
normalized throttle optimized in `[0.001, 1]`). The controller does not subscribe to an
Ackermann speed command or run a separate longitudinal speed controller.
Zero throttle is Unity's distinct hard-brake branch; it remains modeled for
observed inputs but is not an MPC decision.

Runtime settings are in `config/mpc_competition.yaml`. Production MPC reads
team odometry, localized map pose, and the typed plant state; it does not read
simulator ground truth.

QP corridor constraints linearize all four physical body corners in lateral
error and heading together. A small exact two-coordinate ADMM projection
handles those coupled half-planes and box bounds, including unequal penalties
and solver scaling. It adds no optimizer states or Riccati matrix dimensions.
Every published plan/prefix still undergoes the nonlinear source-plant and
rotated-footprint checks; linearization is not a collision certificate.

Frozen development QPs use schema **v4**, including the coupled corner planes.
Older schemas are rejected rather than replayed without their true constraints.

Build from the repository root in the ROS 2 Humble environment:

```bash
source /opt/ros/humble/setup.bash
colcon build --merge-install --packages-up-to f1tenth_mpc
```
