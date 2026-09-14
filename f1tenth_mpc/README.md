# BachelorProject MPC

This package contains the BachelorProject Riccati-ADMM MPC core in its native
C layout. The only additional source is the source-time MPC observer
prototype, which is a legal-measurement API and is not connected to actuator
authority. There is no C++ adapter, shadow controller, track-model wrapper,
replay wrapper, or simulator-specific physics code here.

```text
f1tenth_mpc/
├── include/
│   ├── mpc.h
│   ├── mpc_types.h
│   ├── riccati_solver.h
│   ├── util_math.h
│   └── vehicle_model.h
└── src/
    ├── mpc.c
    ├── mpc_observer.c
    ├── riccati_solver.c
    ├── util_math.c
    └── vehicle_model.c
```

The native model remains the BachelorProject dynamic bicycle model. The only
timing adaptation is the simulator-compatible source interval: runtime calls
may provide the measured source `dt` (nominally 25 ms at 40 Hz), while the MPC
prediction horizon remains independently configured.

The observer state is `[u, v, r, delta, q_drive]`. It consumes odometry speed,
IMU gyro/lateral acceleration, source-associated steering/applied-command
feedback, applied throttle, and a source-epoch timestamp. Simulator truth is
never an observer input; it is used only by offline replay scoring. The current
lateral-state coefficients are a held-out-track prototype and are not
production MPC parameters.

Build from the Humble workspace container:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select f1tenth_mpc
```
