# BachelorProject MPC

This package contains the BachelorProject Riccati-ADMM MPC core in its native
C layout. There is one controller and one vehicle model; no C++ adapter,
shadow controller, track-model wrapper, replay wrapper, or simulator-specific
physics code is included here.

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
    ├── riccati_solver.c
    ├── util_math.c
    └── vehicle_model.c
```

The native model remains the BachelorProject dynamic bicycle model. The only
timing adaptation is the simulator-compatible source interval: runtime calls
may provide the measured source `dt` (nominally 25 ms at 40 Hz), while the MPC
prediction horizon remains independently configured.

Build from the Humble workspace container:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select f1tenth_mpc
```
