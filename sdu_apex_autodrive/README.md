# AutoDRIVE runtime and live-data baseline

The active runtime is deliberately small:

```text
simulator -> bridge -> sensor odometry -> EKF -> CUDA AMCL -> Pure Pursuit
                                      \-> diagnostics-only CSV + ground truth score
```

Ground truth is never supplied to odometry, EKF, AMCL, or a controller. It is
recorded only by the diagnostics monitor so an actual simulator run can be
scored offline.

## Build

Build in the ROS 2 Humble workspace container:

```bash
source /opt/ros/humble/setup.bash
cd /workspace/src
colcon build --base-paths f1tenth_control f1tenth_localization \
  sdu_apex_autodrive f1tenth_planning \
  --merge-install --cmake-args -DBUILD_TESTING=ON
```

## Local startup

Use the repository-level [startup guide](../STARTUP_GUIDE.md) for the complete
simulator-plus-controller workflow. It starts the unified controller before
Unity connects, so the GUI `Connect` action does not require a second ROS
command.

## Active launches

The direct launch commands below are for controlled diagnostics only. For a
normal simulator run, use the root startup guide so the bridge and controller
are started exactly once before Unity connects. The mapping launch is FTG-only
and the racing launch is FTG or Pure Pursuit:

```bash
ros2 launch sdu_apex_autodrive mapping.launch.py
ros2 launch sdu_apex_autodrive controller.launch.py controller:=ftg
ros2 launch sdu_apex_autodrive controller.launch.py controller:=pure_pursuit
ros2 launch sdu_apex_autodrive localization.launch.py start_bridge:=false
```

The production defaults are:

```text
map:       f1tenth_planning/maps/autodrive_track_ftg_commit_20260909_025m.yaml
raceline:  f1tenth_planning/trajectories/autodrive_track_ftg_commit_20260909_025m_mintime_raceline.csv
```

This is the saved 2.5 cm FTG map of the ICRA compete track, paired with its
mintime raceline. The simulator asset and ROS map are selected independently:
the compose default now selects the compete scene, while these files remain
the runtime map/raceline inputs.

## Simulator cadence

`autodrive_bridge_40hz` is a bounded request-pacing bridge. Its name describes
the request clock; it does not fabricate sensor samples. The default simulator
asset is the ICRA compete scene. When using the rebuilt source player, the
source timestamps and measured native stream rate are the authority. Every
analysis must use source timestamps and report gaps, duplicates, and bursts.

For model-identification work, subscribe to
`/autodrive/roboracer_1/bridge_packet_timing`. Schema version 2 links each
packet to the bridge request sequence, sent normalized command, Unity
`simulation_physics_step`, telemetry sequence, and the command sequence and
normalized throttle/steering actually consumed by the vehicle controller.
`simulation_render_frame` is retained only as a render diagnostic; it is not a
physics-step identifier. This stream is diagnostics-only and must not be used
as a replacement sensor input.

The supported local player is the rebuilt source player described in the root
startup guide. Run it with its normal GUI; do not add `-batchmode` or
`-no-graphics` for the visible workflow:

```bash
./AutoDRIVE-Simulator.x86_64 \
  -ip 127.0.0.1 -port 4567 -logFile /tmp/autodrive.log
```

Start the ROS controller first, then click `Connect` in the player window.

## Actual-data recording

Start `controller.launch.py` with the diagnostics-only monitor and recorder,
writing each run to a new directory:

```bash
ros2 launch sdu_apex_autodrive controller.launch.py \
  controller:=pure_pursuit \
  with_ground_truth_monitor:=true \
  with_telemetry_recorder:=true \
  ground_truth_output_csv:=/workspace/src/f1tenth_planning/analysis/current/amcl_vs_ground_truth.csv \
  telemetry_output_dir:=/workspace/src/f1tenth_planning/analysis/current/telemetry
```

The recorder preserves source-event telemetry, commands, feedback, odometry,
EKF, AMCL, current-map pose, scan alignment, and collision count. Score the
recorded file with `score_localization_run`; use `scan_match_benchmark` only on
scans from the recorded run. Do not use a synthetic offline plant as evidence
for runtime acceptance. These outputs are intentionally ignored by Git.

## First model-identification run

The first MPC milestone is a causal timing artifact, not a controller swap.
The development-only launch uses the existing bounded calibration excitation
and records the bridge request/applied-command association alongside allowed
IMU, encoder, LiDAR, odometry, AMCL-health, and actuator-feedback events:

```bash
ros2 launch sdu_apex_autodrive model_identification.launch.py \
  mode:=identification_grid \
  output_dir:=/workspace/src/sdu_apex_autodrive/artifacts/model_id \
  run_name:=model_id_timing_test_current \
  duration_sec:=120
```

It writes the canonical `events.csv`, `manifest.json`, `timing_report.json`,
and topic partitions named `bridge_requests.csv`, `simulator_packets.csv`,
`imu.csv`, `encoders.csv`, `actuator_feedback.csv`, `gt_odom.csv`, and
`experiment_schedule.csv`. `gt_odom.csv` is an explicit not-recorded marker:
ground truth remains offline-only. A positive `duration_sec` shuts down the
whole experiment automatically; `duration_sec:=0` leaves it running until
interrupted. This launch does not start Pure Pursuit or use simulator ground
truth as a runtime input. The current actuator remains a speed-target cascade;
direct MPC actuation is a later, separately validated interface.
