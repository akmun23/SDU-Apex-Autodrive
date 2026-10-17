# Current runtime and rewrite status — 2026-10-17

## Maintained stack

The workspace now retains one ROS runtime path: the 40 Hz bridge, source-
derived PhysX odometry, EKF/AMCL, the RTI MPC, and a direct actuator interface.
The MPC output is `DriveCommand(steering_angle_rad, throttle_normalized)`;
raceline speed remains a tracking reference, not an actuator command. The
MPC's RTI state carries the typed plant channels and pending throttle/steering
history. The PhysX process source is in
`f1tenth_localization/src/physx_native_vehicle_step.cpp` and is built by the
localization package.

The active track assets are the competition map/raceline pair and the practice
map/raceline pair under `f1tenth_planning/`. The Y1 response surface is the
only installed MPC response table. No simulator was launched during this
cleanup.

## Purge performed

- Removed the ignored `live_runs` capture/optimizer corpus (about 25 GB) and
  generated Unity Library/Git metadata from the earlier cleanup. The Unity
  checkout is now reduced to the RoboRacer scene, F1TENTH vehicle prefab,
  vehicle/command/sensor scripts, and physics timing settings used as the
  simulator-source reference; it is intentionally not a second runnable
  project. Retained the PhysX 4.1 source and the exact libraries linked by
  odometry.
- Removed retired FTG/Pure Pursuit and mapping packages, the speed-controller
  implementation and its replay harness, one-off learning/open-plane/racing
  analysis scripts, stale candidate configurations/trajectories, duplicated
  optimizer implementations, obsolete progress archives, and the unused
  external track bundle.
- Retained only the launch/build/record/watch tools used to operate and verify
  the active stack. Historical measurements and experiment artifacts were
  removed with their source data; do not treat old report numbers as current
  validation.
- Removed the stale bag-replay utility: it imported a deleted analysis module
  and selected an obsolete image, so it could not operate on the maintained
  stack. The retained record/watch tools are the ones used for new live-run
  capture and collision/lap supervision.
- Removed bridge command-sequence injection, empty-LiDAR fabrication, and
  one-off packet/publication probes. The retained bridge path is the paced
  request/response FIFO, camera suppression, timing fault, and live actuator
  input. The simulator launcher now uses only official simulator images; it
  no longer mounts custom players or temporary dynamics/replay profiles.
- Reduced the PhysX checkout to the source/header trees, shared headers, and
  the seven static libraries linked by odometry. The development launcher now
  compiles the complete mounted runtime dependency chain when reusing an old
  image, so stale odometry or MPC binaries cannot silently remain active.
- Retention rule: every remaining ROS package is part of the launch dependency
  graph; both retained map/raceline pairs are referenced by the runtime or
  startup guide; retained tests exercise current production C++ components;
  Unity/PhysX files are limited to the simulator source and libraries used to
  implement the current plant/odometry. No old bags, model-training corpus,
  retired controller, standalone optimizer, or superseded candidate set is
  intentionally retained in the working tree.
- Removed obsolete package-install dependencies and generated PhysX compiler
  output that is not linked by the active odometry build. The ignored host
  `build*`, `install*`, `log*`, and Python-cache directories remain because
  their contents are root-owned and this session has no sudo rights; they are
  stale generated output, not runtime inputs. The full purge is not complete
  until those root-owned generated directories are removed.

## Verification state

Current package discovery returns exactly the five ROS packages in the active
launch graph; PhysX's vendored CMake package is excluded by `COLCON_IGNORE`.
Python/shell syntax checks, `git diff --check`, and the competition runtime
topic-policy check passed after the cleanup. No bag files or references to the
deleted replay script remain. A clean ROS build and real batch-mode simulator
run are still pending; no accuracy, 40 Hz timing, lap time, or collision-free
claim is made here. The workspace purge remains incomplete until the
root-owned generated build/install/log/cache trees are removed.
