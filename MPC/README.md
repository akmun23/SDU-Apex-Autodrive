# Riccati-ADMM MPC

CPU Riccati-ADMM model-predictive controller for the AutoDRIVE RoboRacer
runtime. The ROS 2 executable is `mpc_autodrive_node` in package
`mpc_riccati`; the supported integration is the `controller:=mpc` path in
`sdu_apex_autodrive/controller.launch.py`.

## Runtime contract

The node reads its runtime topics from environment variables supplied by the
integration launch:

- `/odom` — encoder/IMU odometry
- `/ekf_pose` — map-frame EKF pose fused from `/odom` and AMCL
- `/local_raceline` — map-frame path from the lateral planner

It publishes Ackermann acceleration commands to `/cmd/acceleration`. The integration's
actuator interface converts those commands to normalized AutoDRIVE steering
and throttle. This MPC node does not access simulator IPS, simulator odometry,
ground-truth TF, collision/lap telemetry, or physical-vehicle interfaces.

The model uses a fixed 20-stage horizon and a 100 ms prediction step (2.0 s
look-ahead). One solve and one `/cmd/acceleration` publication are triggered by
each native `/odom` update, so the controller runs at the available AutoDRIVE
cadence rather than fabricating a higher-rate sensor or command stream. The
actuator interface owns the command watchdog at that same boundary. The MPC
uses its previously issued Ackermann steering target as the effective-steering
state input and models the simulated longitudinal command response with a
tunable effective-acceleration state (`MPC_ACCEL_EFFECTIVE_TAU_S`, default
0.15 s); it does not subscribe to simulator steering feedback.

## Build

Inside the ROS 2 Humble container:

```bash
colcon build --packages-select mpc_riccati
source install/setup.bash
```

The standalone core can also be built without ROS:

```bash
cmake -S MPC -B /tmp/mpc-build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/mpc-build
ctest --test-dir /tmp/mpc-build --output-on-failure
```

## Configuration

Solver and model tuning values are read from environment variables in `src/mpc.c`
and `src/riccati_solver.c`. The most relevant groups are stage weights,
steering/acceleration effort and rate weights, ADMM tolerances, prediction
step, wall handling, and the tracking-error speed envelope. The latter scales
the horizon's target velocity only when the measured Frenet pose is already
off the local raceline (`MPC_TRACKING_ERROR_SPEED_GAIN`, default `0.75`;
minimum scale `0.45`). Review the defaults in `include/mpc_types.h` before
changing high-speed behavior.

The MPC path is intentionally not exposed through removed legacy simulator,
standalone gym, FPGA, or physical-actuator wrappers. Use the shared AutoDRIVE
launch and checked-in map/raceline artifacts.

## Offline weight testbench

The native benchmark exercises straight tracking, left and right curvature,
speed transition, and bounded corridor tracking. It runs without ROS and does
not consume simulator ground truth:

```bash
cmake -S MPC -B /tmp/mpc-build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/mpc-build --target test_closed_loop_benchmark
/tmp/mpc-build/test_closed_loop_benchmark
python3 MPC/find_passing_weights.py \
  --binary /tmp/mpc-build/test_closed_loop_benchmark \
  --candidates 32 --workers 2 \
  --csv /tmp/mpc-weight-sweep.csv
```

The sweep uses the current `MPC_W_*` environment names and never starts more
than four benchmark processes. A candidate is considered passing only when it
has no solver failures and no synthetic corridor violations. Offline results
must still be confirmed in the simulator before promoting a profile.
