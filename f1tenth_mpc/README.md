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

The native model remains the BachelorProject dynamic bicycle model. The timing
contract is 40 Hz: runtime calls may provide the measured source `dt`
(nominally 25 ms), and the default MPC horizon is 30 stages, or 0.75 s.

The observer state is `[u, v, r, delta, q_drive]`. It consumes odometry speed,
IMU gyro/lateral acceleration, source-associated steering/applied-command
feedback, applied throttle, and a source-epoch timestamp. Simulator truth is
never an observer input; it is used only by offline replay scoring. The current
lateral-state coefficients are a held-out-track prototype and are not
production MPC parameters.

The active model contract is checked by
`tools/model_id/check_vehicle_model_manifest.py`. The production MPC baseline
and the offline identified plant are intentionally separate profiles: the
identified plant is not promoted until its architecture is migrated and
accepted. Canonical offline replay resolves its defaults from
`config/vehicle_model_manifest_v1.json` only when `--use-manifest` is passed;
report-driven replay is explicit for offline A/B comparisons. Python/C default
parity is part of the MPC package test suite.

The current project operating ceiling is 16 m/s. New identification and
controller validation must stay within that envelope; older 18/20 m/s runs
remain historical offline evidence only. This is a dev-side command and data
scope decision and does not modify simulator physics.

Build from the Humble workspace container:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select f1tenth_mpc
```
