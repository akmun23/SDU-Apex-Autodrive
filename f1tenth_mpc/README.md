# BachelorProject MPC integration

This package wraps the tested pure-C MPC implementation from
`/home/akselmo/Documents/GitHub/BachelorProject/MPC`. It is the BachelorProject
MPC only: a 9-state augmented Frenet model with Riccati-ADMM, a 20-stage
horizon, 30 ms prediction steps, and steering-rate/longitudinal-acceleration
inputs.

The ROS node is intentionally shadow-only. It consumes the dev-side
sensor/localization state on `/mpc/control_state` and publishes diagnostics and
`/mpc/shadow_command`; Pure Pursuit remains the actuator-command owner. No
Unity file, simulator physics, ground-truth topic, or simulator odometry is
used by this controller.

The current control-state publisher marks lateral velocity unavailable, so the
wrapper supplies `v_y = 0` until a sensor-only estimate has been validated.
Steering feedback is used only when its source-time sample is marked valid;
otherwise the wrapper retains the last command it issued.

Build and test in the Humble workspace container:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-up-to f1tenth_mpc --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select f1tenth_mpc
```

Run the shadow path with the existing launch:

```bash
ros2 launch sdu_apex_autodrive controller.launch.py \
  controller:=mpc_shadow \
  with_model_id_recorder:=false
```

The diagnostic message reports the solver status, residuals, current and
predicted Frenet lateral error, predicted distance, steering, and physical
acceleration. The physical MPC output is `proposed_acceleration_mps2` plus
steering angle.

The offline `vehicle_model_replay` executable uses the same direct-input
dynamic-bicycle baseline as the BachelorProject model. Its input is
`dt_s,steering_rad,acceleration_mps2`; an optional initial-state CSV contains
`x_m,y_m,yaw_rad,u_mps,v_mps,r_radps[,steering_rad]`.

These tests prove core integration and finite solver output only. They do not
prove a clean simulator lap or real-car model fidelity. Ground truth remains an
offline comparison source, and the Unity simulator is outside this package's
write scope.
