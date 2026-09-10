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

## Active launches

Use one bridge and one controller launch. The mapping launch is FTG-only and
the racing launch is FTG or Pure Pursuit:

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

Run the graphical Unity player in batch mode. Do not use `-no-graphics` for the
HDRP simulator:

```bash
./AutoDRIVE\ Simulator.x86_64 -batchmode \
  -ip 127.0.0.1 -port 4567 -logFile /tmp/autodrive.log
```

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
