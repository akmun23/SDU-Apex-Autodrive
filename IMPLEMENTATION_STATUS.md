# SDU Apex AutoDRIVE implementation status

Status date: 2026-09-05  
Repository: `/home/akselmo/Documents/GitHub/SDU-Apex-Autodrive`

## Current runtime

The active local odometry path is a deterministic observer in
`f1tenth_localization`. It consumes only exact source-timestamp packets made
from the left encoder, right encoder, and IMU. Incomplete packets are discarded
and counted; no sensor value is forward-filled, interpolated, or fabricated.

The observer implements the handoff plan:

- frozen 218-point wheel-speed map and 0.059 m wheel radius;
- scalar IMU propagation with braking affine correction and gated wheel update;
- 2-D body-frame RK2 turn mode with fixed lever-arm correction and hysteresis;
- timestamp regression, encoder reset, timing-gap, packet-drop, and coherence
  handling;
- version-2 compact diagnostics on `/odom/diagnostics`;
- no learned model, ground truth, camera, LiDAR, AMCL, or throttle input in
  runtime odometry.

The obsolete learned C++ runtime headers and their old learned-estimator
generation/scoring scripts were removed. Generic offline calibration tooling
remains separate from the runtime and can support future model development.

## Replay evidence

The fixed fit/test/blind recordings are:

1. `sdu_apex_autodrive/artifacts/calibration/raw/identification_grid_fit_40hz_20260905_final/identification_grid_20260905_144304.csv`
2. `sdu_apex_autodrive/artifacts/calibration/raw/identification_grid_test_40hz_20260905/identification_grid_20260905_151313.csv`
3. `sdu_apex_autodrive/artifacts/calibration/raw/identification_grid_validation_40hz_20260905/identification_grid_20260905_155143.csv`

Exact packet reconstruction and Python/C++ replay parity were verified on all
three. Results:

| recording | complete packets | moving samples | p95 relative error | <=2% |
|---|---:|---:|---:|---:|
| fit | 58,661 | 31,482 | 0.7208% | 99.1169% |
| test | 58,726 | 31,484 | 0.7218% | 99.0948% |
| blind validation | 58,747 | 31,476 | 0.7422% | 99.0342% |

The maximum Python/C++ replay differences were floating-point noise; all
observer flag outputs matched.

## Fresh live acceptance

The no-reset live motion run is:

`sdu_apex_autodrive/artifacts/calibration/raw/deterministic_observer_motion_40hz_20260905/zero_throttle_decel_20260905_172107.csv`

It contains 1,081 complete source-timestamp packets, zero incomplete packets,
zero duplicate event rows, and 390 moving replay samples. Python/C++ parity
again matched to floating-point noise; the moving samples were all within 2%
in the replay report.

The live timing gate and run-time monitor both passed. The verified source and
bridge streams were approximately 40 Hz with 25 ms-like intervals and no
reported burst, long-gap, or observer packet-coherence fault. The supplied
prebuilt player does not expose Unity simulation-time/frame metadata, so this
acceptance used `require_simulation_metadata:=false`; cadence, source stamps,
finite values, physical bounds, and packet sequencing remained enforced.

## Verification commands

Humble Docker build:

```bash
docker exec sdu_apex_autodrive bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/autodrive_devkit/install/setup.bash
  cd /workspace
  colcon build --merge-install --packages-select \
    f1tenth_localization sdu_apex_autodrive --symlink-install
'
```

Pure observer tests pass with CTest. The Python package tests include exact
packet reconstruction and reference-observer behavior. The fresh CSV can be
replayed with:

```bash
PYTHONPATH=sdu_apex_autodrive python3 -m \
  sdu_apex_autodrive.scripts.validate_odometry_observer \
  sdu_apex_autodrive/artifacts/calibration/raw/deterministic_observer_motion_40hz_20260905/zero_throttle_decel_20260905_172107.csv \
  --cpp-replay /tmp/odometry_observer_replay
```

The localization package also builds with `BUILD_GPU_AMCL=OFF`, so CPU odometry
and its tests do not require a CUDA toolchain. The shared 40 Hz policy is in
`sdu_apex_autodrive/config/timing_clean_40hz.yaml`; the race launcher defaults
RViz off.

## Remaining acceptance boundary

This is clean deterministic odometry and 40 Hz source-data acceptance. It is
not a claim that the complete track stack has been accepted. Track-map AMCL,
EKF, controller, raceline, and closed-loop racing still require their own live
acceptance run.
