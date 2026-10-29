# IROS competition mapping and first raceline

## Current work — 2026-10-29

The IROS image is pinned in `tools/start_simulator.sh` under the `iros` profile.
The maintained practice map, trajectory and racing configuration remain available.
The competition racing aspiration is track-specific (initially about 7 s/lap,
later discussed as 6 s/lap), with the first lap treated as warm-up. Neither
value is a stopping target for the map-only optimizer. No sub-agent work is run concurrently: at
most one sub-agent may be active, including during later optimization.

FTG is restored from commit `9ce950439` (the completed-mapping revision).
It selects a connected LiDAR gap and checks Ackermann steering arcs against
the rear-axle vehicle footprint (0.510 × 0.273 m, LiDAR 0.2733 m ahead).
The ROS adapter publishes the current direct-throttle `DriveCommand` format.
Its mapping schedule scales the requested speed into a maximum 0.020 normalized
throttle; the speed range is 0.12–0.40 m/s. This schedule is deliberately a
slow mapping policy, not a calibrated speed controller. A stopped/no-path
request publishes zero throttle, Unity's brake branch.

The FTG revision restored first had large recovery thresholds and saturated
steering in normal corridors. It was replaced, including its recovery state
machine and configurations. The current revision passes all 27 algorithm tests.

### Measured mapping attempt

`live_runs/iros_mapping_20261029_123431` recorded one completed official lap,
zero collisions, 42.531 m of sensor-odometry travel, and 125.94 s wall time.
The raw recording contains permitted sensor/command topics, sensor odometry,
plant state, TF, FTG diagnostics and occupancy grids. Lap/collision counters
are consumed only by the development supervisor.

The first grid (481 × 819 cells at 0.025 m resolution) failed geometric
inspection: scan matching/loop closure duplicated and displaced the return
corridor. It was moved out of the production maps directory into that run's
`rejected_scan_match/`. Earlier partial grids are also confined to that run.
A completed lap alone does not certify a map.

The recording was replayed with only LiDAR, `odom→base_link`, and the
static sensor transforms. Recorded `map→odom` corrections and all actuator
commands are excluded from replay. This lets mapper changes be evaluated
without another live lap.

Disabling scan matching, scan barycenters and loop closure removed the
duplicated corridor. The accepted sensor-odometry-backed map is 238 × 903
cells at 0.025 m resolution; its image SHA-256 is
`e1538f05e9fb4d05cf3b6909151bb399915fefc629877bc1d057a1019b28f109`.
A second live mapping attempt, `iros_mapping_20261029_124417`, completed
another full lap with zero collisions using these settings, recording
42.533 m and 126.37 s. Its independently generated grid remains in that run
directory rather than replacing the inspected map. These checks establish
a usable initial map, not centimetric ground-truth map certification.

### Initial raceline and production driving

The exported minimum-curvature seed has 779 points and a 46.732 m lap length.
Its equivalent geometric steering requirement peaks at 0.444 rad. Dense
rear-axle footprint checks include the requested 5 cm wall margin. The
constant 1.2 m/s profile is deliberately an initial tracking baseline, not
the competition speed target or a dynamically optimal profile.

`iros_mpc_initial_20261029_01` exposed a launch-configuration regression:
loading the small fastline weight overlay without the base configuration
left MPC disabled. `controller.launch.py` now loads the full base first,
then the overlay. This preserves the established practice configuration.

With that correction, `iros_mpc_initial_20261029_02` completed two official
competition laps with zero collisions: 37.711 s warm-up, then 38.829 s.
The production controller, odometry and AMCL used permitted live inputs.
Ground truth was recorded only for development evaluation. At this low
speed, forward-velocity RMSE was 0.0031 m/s, lateral-velocity RMSE 0.0003 m/s,
and aligned odometry position drift peaked at 0.048 m over the run.
The map origin differs from Unity's world origin; the evaluator's raw
3.248 m AMCL/world offset is therefore not a localization-error score.

Of 3,076 published cycles, 3,051 ADMM solves converged; 26 commands used an
accepted line search and one used a revalidated prior plan. Publication-gate
elapsed time was 5.013 ms median, 7.652 ms p95 and 19.511 ms maximum, with
no deadline failures while driving. The logged stale-state/bridge-disconnect
fault occurred after the supervisor stopped Unity on the two-lap limit.
Later closed-loop predicted horizons are not fixed-command rollout tests.
The complete evaluation is in that run's `evaluation.txt`.

### Next work

1. Run the map-only pipeline documented in `docs/MAP_ONLY_OPTIMIZER.md`;
   it extracts its own centerline and builds/repairs command witnesses.
2. Continue the target-free, adaptive periodic-command campaign with exact
   production transitions, checkpoints and independent constraint/rollout
   checks. Do not mistake an initializer's duration for a performance target.
3. Verify the source model against competition-map recordings at increasing
   speeds, then validate the faster target with production MPC over multiple
   laps. A source-model certificate alone does not establish Unity parity.

AutoDRIVE guidance supplied by the user on 2026-10-29 explicitly permits
timer callbacks and references the upstream commented 25 ms publisher.
This resolves the objection to timer-based publication itself. Our adapter
also changes request pacing and timestamp association; the supplied guidance
does not establish approval of all those additional details. See
`docs/RUNTIME_TOPIC_POLICY.md`. Runtime topic-policy checks are not an
approval certificate for the complete image.

### Optimizer execution

The 779-node source-parameter/setup audit passed. The first source-plant
warm-start attempt (`live_runs/iros_ipopt_20261029_01`) did not reach IPOPT:
its simple feedforward path follower exceeded footprint clearance around
station 16–18 m at all three initialization speeds (2.4, 1.8, 1.3 m/s).
This is an initializer failure, not evidence that the raceline is infeasible;
the production MPC already followed it for two laps.

`sdu-apex-iros-ipopt.service` was then started with the explicitly analytical
initializer (`--seed-backend synthetic --seed-speed-from-centerline`) and
output root `live_runs/iros_ipopt_20261029_02`. This initializer is not
measured data or a certified trajectory. IPOPT must establish feasibility
before any result can be promoted. The physics constraints are unchanged.
The first attempt also caught an invalid CLI trust-factor spelling before
any solve; the accepted argument is comma-separated `2,8,64` (>1).

The run uses a 0.06 m mesh, 400 Phase-I iterations, 300 iterations per
minimum-time stage, and 600 CPU-second caps per solver stage. Intermediate
time-stage checkpoints are retained. `--skip-native` disables the separate
native replay, so the exported result remains unvalidated until the held-
command source checks and online MPC trials described above.

That standalone service was stopped during the map-only overhaul: its memory-
heavy setup had produced no IPOPT iterate. The old log remains available, but
it is no longer the live progress stream. The maintained entrypoint is now:

```bash
tools/run_raceline_campaign.sh f1tenth_planning/maps/autodrive_iros_2026_compete.pgm f1tenth_planning/maps/autodrive_iros_2026_compete.yaml
# Add --watch for current progress; --stop for a checkpoint-preserving stop.
```

This workflow requires no previous raceline or recording and has no 6 s or
7 s stopping cutoff. Its stage, exports and verification are documented in
`docs/MAP_ONLY_OPTIMIZER.md`.

Observe without interrupting:

```bash
tail -F live_runs/iros_ipopt_20261029_02/solver.log
systemctl --user status sdu-apex-iros-ipopt.service
```

## Live mapping

Start the simulator:

```bash
SDU_APEX_SIM_TRACK=iros SDU_APEX_SIM_MODE=batchmode tools/start_simulator.sh
```

In another terminal:

```bash
tools/run_iros_mapping.sh
```

The runner builds the current source, records the attempt under `live_runs/`,
and prints the run directory. The supervisor reports travel, speed, collisions,
lap count and map dimensions every five seconds. It stops on a collision,
stall, timeout, or completed lap; the host stops Unity when the result marker
appears. Failed maps stay inside the run directory. Successful lap grids are
saved under `f1tenth_planning/maps/autodrive_iros_2026_compete.*` and still
require geometric verification.

Mapping inputs are LiDAR and production sensor-derived odometry. Official
world-pose transforms are remapped to `/autodrive_ground_truth/tf*`, outside
the mapper's `odom→base_link→lidar` transform tree. FTG does not consume a map,
ground-truth pose, AMCL or MPC output.

## Replay and raceline tools

`tools/replay_mapping.py` runs inside the ROS mapping image. Mount the bag,
SLAM configuration and output directory; its `--help` describes the paths.
It never publishes vehicle commands.

Once the grid passes inspection, run:

```bash
SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv/bin/python tools/build_mapped_raceline.py \
  --map f1tenth_planning/maps/autodrive_iros_2026_compete.yaml \
  --driven-trace live_runs/iros_mapping_20261029_123431/sensor_odom_trace.csv \
  --output-dir f1tenth_planning/trajectories/autodrive_iros_2026_compete
```

The builder extracts a closed map corridor, validates the rear-axle footprint
plus a 5 cm wall margin densely, checks the geometric steering requirement,
and exports CSV, plot and a report. Its initial constant-speed profile is
explicitly a baseline; it is not a vehicle-dynamics minimum-time claim.
