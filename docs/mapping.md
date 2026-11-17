# Track mapping

Follow the Gap (FTG) drives a slow mapping lap while SLAM records LiDAR against sensor-derived odometry to build an occupancy grid.

## Mapping design

[tools/sim/start_simulator.sh](../tools/sim/start_simulator.sh) pins the IROS
image under the `iros` profile. Mapping uses LiDAR and production sensor odometry.
Official world-pose transforms are remapped outside the mapper's
`odom → base_link → lidar` tree, under `/autodrive_ground_truth/tf*`.
FTG consumes LiDAR without a map, ground-truth pose, AMCL or MPC output.

FTG selects a connected gap and checks Ackermann arcs against the rear-axle
footprint: 0.510 × 0.273 m, with LiDAR 0.2733 m ahead. Its ROS adapter publishes
direct-throttle `DriveCommand` messages. The mapping schedule scales a requested
0.12–0.40 m/s into at most 0.020 normalized throttle. This is a slow mapping
policy, not a calibrated speed controller. A stopped or no-path request
publishes zero throttle, which selects Unity's
brake branch if delivered. The shared actuator interface rejects throttle below
0.001, including this stop command; the mapping supervisor stops the run on
its collision, stall, timeout or lap-completion conditions.

[slam_mapping.yaml](../sdu_apex_autodrive/config/slam_mapping.yaml) disables
scan matching, scan barycenters and loop closure. The inspected sensor-odometry
grid is 238 × 903 cells at 0.025 m resolution, with image SHA-256
`e1538f05e9fb4d05cf3b6909151bb399915fefc629877bc1d057a1019b28f109`.
A completed lap alone does not verify geometry: inspect corridors and wall
alignment before using a grid for localization or optimization.

The two runs below each complete one official lap with zero collisions. The map
dimensions and image hash above can be checked directly.

| Run identifier | Sensor-odometry travel | Wall time |
|---|---:|---:|
| `iros_mapping_20261029_123431` | 42.531 m | 125.94 s |
| `iros_mapping_20261029_124417` | 42.533 m | 126.37 s |

The reported runs do not provide centimetric ground-truth certification of the
map. Bags include sensors/commands, odometry, plant state, TF, FTG
diagnostics and occupancy grids. Lap/collision counters belong to the
development supervisor only.

## Live mapping

From the repository root, start Unity and run mapping in another terminal:

```bash
SDU_APEX_SIM_TRACK=iros SDU_APEX_SIM_MODE=batchmode tools/sim/start_simulator.sh
```

```bash
tools/mapping/run_iros_mapping.sh
```

The runner builds the source, creates an ignored run directory and prints its
path. Every 5 s, the supervisor reports travel, speed, collisions, lap count and
map dimensions. It stops on collision, stall, timeout or lap completion; the
host stops Unity when the result marker appears. Failed maps stay in the run
directory. Successful grids are saved as the
[IROS map pair](../f1tenth_planning/maps/competition.yaml), subject
to geometric inspection. [Architecture](architecture.md#bridge-timing) describes
bridge timestamps and the permitted-input boundary.

## Replay and geometric reference

[tools/mapping/replay_mapping.py](../tools/mapping/replay_mapping.py) runs inside
the ROS mapping image. Mount the recorded bag, SLAM configuration and output
directory; consult its `--help` for arguments. Replay uses only LiDAR,
`odom → base_link` and static sensor transforms. It excludes recorded
`map → odom` corrections and actuator commands and never publishes commands.

After map inspection, set `trace` to the recording's generated
`sensor_odom_trace.csv` and run:

```bash
python3 tools/raceline/build_mapped_raceline.py \
  --map f1tenth_planning/maps/competition.yaml \
  --driven-trace "$trace" \
  --output-dir f1tenth_planning/trajectories/competition
```

Use the optimizer Python environment. The builder extracts a closed corridor,
densely checks the rear-axle footprint plus 5 cm wall margin, checks geometric
steering demand, and exports CSV, plot and report. Its constant-speed profile
is a tracking baseline. The checked-in [extraction report](../f1tenth_planning/trajectories/competition/map_report.json)
records 779 points, length 46.732 m, peak
equivalent steering 0.444 rad and requested speed 1.2 m/s. The reported
two-lap comparison at this speed gives forward-velocity RMSE of 0.0031 m/s,
lateral-velocity RMSE of 0.0003 m/s and maximum aligned odometry drift of
0.048 m.

Truth supplies offline scoring only.

## Dynamics optimization and driving

[Raceline optimization](raceline-optimization.md) takes the map pair, extracts
its own centerline and searches periodic source-model command trajectories.
It uses checkpoints and independent footprint, constraint and rollout checks,
without a requested lap-time cutoff. A source certificate does not establish
Unity model accuracy or controller performance.

The development launch loads the controller base configuration before overlays.
[Trials](trials.md) lists both runtime profiles and references and describes
Unity validation with one warm-up lap plus ten scored laps. Map and Unity
world origins differ; raw position offsets do not measure localization error.
