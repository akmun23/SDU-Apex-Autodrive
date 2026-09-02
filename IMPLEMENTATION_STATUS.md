# SDU Apex AutoDRIVE implementation handoff

Status date: 2026-09-01
Repository: `/home/akselmo/Documents/GitHub/SDU-Apex-Autodrive`
Branch: `main`
Recorded revision: `a074412 Refactor code structure for improved readability and maintainability`

## Purpose and authority

The user request is the governing requirement. The supplied files were treated
as technical handoff material and source-selection guidance:

- `/home/akselmo/Downloads/SDU Apex AutoDRIVE — Implementation and Cleanup Plan.md`
  describes the intended one-stack architecture, cleanup targets, allowed runtime
  interfaces, calibration checkpoints, and definition of done.
- `/home/akselmo/Downloads/SDU_Apex_Runtime_Rewrite_342eb3b.zip` contains the
  rewrite manifest, removal list, and reference implementation material. Its
  `REMOVE.txt` was not followed blindly where it conflicted with the working
  localization/configuration needed by the requested stack.

The implementation target is the AutoDRIVE RoboRacer competition interface. The
official guide is:
<https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2024/>.
Runtime controllers must use only known competition-compatible sensor/team
topics. Ground-truth topics are retained only for diagnostic/calibration programs
and are not inputs to AMCL, EKF, planning, or controllers.

## Shutdown state

At the start of this handoff, the two project containers were running:

```text
autodrive_roboracer_sim  Up 6 hours   sdu-apex-autodrive-simulator:2026-iros-practice
sdu_apex_autodrive       Up 22 hours  sdu-apex-autodrive:humble
```

The user then explicitly requested that everything be stopped. Both containers
were stopped after this document was written. No repository files were deleted or
reset as part of shutdown. The simulator executable and ROS processes therefore
are not running now; restart instructions are at the end of this document.

The last observed processes before shutdown were the official bridge, sensor
odometry, actuator interface, a ROS topic-rate monitor, the ROS 2 daemon, the
simulator executable, and the simulator container's idle shell. There was no
active AMCL, EKF, controller, planner, or MPC process at that moment.

## Current repository architecture

The workspace contains these main runtime packages:

| Package | Role |
| --- | --- |
| `f1tenth_localization` | Official-sensor odometry, CUDA AMCL, and odom-dominant AMCL-corrected EKF |
| `f1tenth_control` | FTG, Pure Pursuit, and Stanley controller components |
| `f1tenth_lidar` | Scan splitting and lateral-planning obstacle representation |
| `f1tenth_lateral_planner` | Optional local raceline/corridor planner for MPC |
| `f1tenth_planning` | Five-lap map inputs, raceline generation, map/trajectory utilities |
| `mpc_riccati` (`MPC`) | Riccati/ADMM MPC and AutoDRIVE ROS 2 wrapper |
| `sdu_apex_autodrive` | Unified launch, actuator boundary, speed controller, calibration, map saver, diagnostics |

The intended production graph is:

```text
AutoDRIVE API bridge
    ├── official LiDAR ───────────────► FTG / AMCL / optional planner
    ├── official rear encoders + IMU ─► sensor_odometry_node ─► /odom
    └── official actuator feedback ───► speed/steering diagnostics

/map + /odom + LiDAR ─► custom CUDA AMCL ─► /amcl_pose
/odom + /amcl_pose ────► custom EKF ──────► /ekf_pose + map→odom TF

/ekf_pose + /odom + raceline ─► Pure Pursuit / Stanley ─► /cmd/controller
/ekf_pose + /odom + local raceline ─► MPC ──────────────► /cmd/controller

/cmd/controller ─► one actuator_interface ─► official normalized commands
```

Only one controller and one actuator interface may be active in a competition
run. `controller.launch.py` is the single user-facing launch entry point.

## Implemented work

### Unified launch and runtime boundary

Implemented in `sdu_apex_autodrive/launch/controller.launch.py`:

- Selects exactly one of `ftg`, `pure_pursuit`, `stanley`, or `mpc`.
- Starts the official `autodrive_bridge` once.
- Starts team odometry, custom AMCL, EKF, optional lateral planner, one
  controller, and one actuator interface.
- Uses the verified current-competition map and full-range raceline by
  default: `f1tenth_planning/maps/autodrive_compete_2026.yaml` with the
  existing `icra_2025_raceline.csv` geometry.
- Defaults to no RViz, no ground-truth monitor, no telemetry recorder, and no
  collision-reset safety in a rules-compliant run.
- Keeps simulator-only collision reset available only as an explicit diagnostic
  option (`with_collision_safety:=true`).
- Keeps ground truth out of runtime control/localization subscriptions.

The compose file keeps both containers idle until an explicit simulator or ROS
launch command is issued. This prevents an automatically started duplicate
bridge from occupying port 4567.

### AutoDRIVE geometry and documented envelope

The implementation uses documented values and does not retune wheel size:

```text
wheel radius       0.0590 m
wheelbase          0.3240 m
vehicle width      0.2700 m
steering range     ±0.5236 rad
centre steer rate  3.2 rad/s
maximum speed      22.88 m/s
normalized command range [-1, 1]
```

The official guide documents longitudinal slip `Sx = (r*omega - vx)/vx`,
lateral slip `Sy = tan(alpha)`, and the two-piece cubic asymptotic force
response. The documented longitudinal knots are approximately
`(0.15, 0.72)` and `(0.25, 0.464)`; lateral knots are approximately
`(0.01, 1.0)` and `(0.10, 0.5)`. These values are model knowledge, not a
reason to alter the physical wheel radius.

### Sensor odometry and slip-aware confidence

Implemented in `f1tenth_localization/src/sensor_odometry_node.cpp` and
`f1tenth_localization/config/sensor_odometry.yaml`:

- Uses cumulative official rear left/right joint positions in radians.
- Uses the fixed official radius `0.0590 m`.
- Uses official IMU orientation, gyro, acceleration, and throttle feedback as
  available to the team stack.
- Publishes team `/odom` and isolated team TF.
- Uses IMU body speed for the high-slip observation gate.
- Avoids evaluating the slip ratio near standstill, where division by speed is
  ill-conditioned.
- Computes continuous wheel-observation confidence: full below the documented
  longitudinal extremum, smooth reduction through the extremum to the
  asymptotic knot, and zero confidence in the asymptotic region.
- Reduces odometry pose covariance as wheel confidence falls, allowing the EKF
  to trust AMCL more when encoder speed is no longer a reliable body-speed proxy.
- Treats encoder dropout while moving as low-confidence instead of presenting it
  as a good measurement.
- Publishes an offline-only `/odom/diagnostics` vector containing raw and
  corrected wheel speed, slip ratio, wheel confidence, the current bias
  placeholder, and an encoder-reset counter. The calibration recorder stores
  this vector without feeding it back into estimation or control.
- Encoder discontinuities re-baseline the angle counters while preserving the
  continuous odom position, yaw, and speed. A short covariance inflation marks
  the transition instead of resetting the odom origin.

### Custom AMCL and EKF

The custom CUDA AMCL is built from `f1tenth_localization/gpu_amcl_cpp`. It uses
the map, official LiDAR, team `/odom`, and the static raceline prior. It does
not use simulator pose, IPS, collision, lap, or debug topics.

The EKF in `gpu_amcl_cpp/src/core/ekf_node.cpp`:

- Predicts from every `/odom` sample.
- Applies delayed `/amcl_pose` corrections in map coordinates.
- Interpolates odometry history to the AMCL timestamp.
- Inflates only large AMCL innovations instead of allowing one scan to erase the
  motion estimate.
- Resets explicitly on a large, accepted AMCL relocalization jump.
- Publishes `/ekf_pose` and map→odom TF.
- Uses continuous odometry covariance, including slip confidence.

### Controllers and cadence

FTG is LiDAR-only and publishes a physical-unit
`ackermann_msgs/msg/AckermannDriveStamped` command.

Pure Pursuit and Stanley use the map-frame pose, `/odom`, and the official
LiDAR event as their control trigger. They do not run a synthetic 40 Hz command
timer. The current practice API was measured at approximately 10–12 Hz for
native sensor telemetry, so one controller command is produced per accepted
LiDAR event. This is the available runtime cadence even though the official
competition specification lists a 40 Hz LiDAR scan rate.

Pure Pursuit includes:

- Full `22.88 m/s` target domain instead of the old 5.5 m/s project ceiling.
- Raceline speed, curvature, lateral error, preview braking, and corridor
  clearance regulation.
- Map-frame pose and odometry freshness/covariance qualification.
- Allowed odometry-based short-term state extrapolation for scan delivery lag.
- Steering feedback in radians with bounded same-direction lead compensation.
- Steering-rate and speed-command shaping, with neutral output on invalid/stale
  state.

Stanley has the same map-pose/odometry freshness gating and LiDAR cadence. Its
local raceline subscription accepts map-frame `nav_msgs/Path` points with speed
stored in `position.z` and yaw stored in the planar quaternion.

### Speed controller and calibration

Implemented in `sdu_apex_autodrive/sdu_apex_autodrive/speed_controller.py` and
`actuator_interface.py`:

- Converts requested speed and acceleration into the official normalized
  throttle command.
- Uses measured full-range throttle feed-forward rather than a fixed throttle
  ceiling.
- Allows the full forward normalized range through `1.0`.
- Uses acceleration inverse gains and conditional integration for tracking.
- Coasts/brakes for overspeed and handles stop requests.
- Uses acceleration feedback gain `0.05` after isolated 10 Hz IMU spikes were
  observed cutting throttle at low speed.
- Does not tune the documented wheel radius.

The authoritative post-change closed-loop validation is:

`artifacts/calibration/raw/open_scene_speed_feedback05/speed_steps_20260831_141948.csv`

Measured results:

```text
native LiDAR/IMU/encoder/odom cadence       median 11.49 Hz
controller command cadence                  median 49.98 Hz (internal output path)
throttle/steering feedback cadence          median about 11.49 Hz
ground-truth speed                          0 to 22.572 m/s
ground-truth collisions                     0
settled high-speed tracking                 mostly within about 1.2 m/s by phase
settled tail absolute speed error, p95      1.611 m/s (all phases include transitions)
```

The direct full-range throttle run produced a measured throttle-speed fit and
wheel/body-speed map:

- `artifacts/calibration/raw/open_scene_throttle_full_range_settled/throttle_steps_20260831_113920.csv`
- `artifacts/calibration/derived/full_range_longitudinal_fit_settled_20260831.csv`
- `artifacts/calibration/derived/full_range_wheel_speed_body_map_settled_20260831.csv`

The high-speed coast/asymptote run reached `22.5996 m/s` without collision:

`artifacts/calibration/raw/open_scene_speed_coast_asymptote_final/speed_steps_20260831_134559.csv`

Diagnostic reset phases create large absolute ground-truth-versus-odom position
errors because simulator resets and team odom-origin continuity are not
perfectly phase-aligned. Speed-tail results are useful; reset-heavy absolute
position results must not be treated as final localization accuracy.

### Calibration analysis hygiene

The offline analyzer now uses `gt_odom_event_count` to collapse faster recorder
rows to unique simulator odometry events before truth-based fitting and error
metrics. It retains a raw-row count and reports how many duplicate rows were
removed. Truth-distance checks use source timestamps, the documented 22.88 m/s
speed envelope, a configurable jitter factor/margin, and an explicit legacy
fallback for old no-timestamp exports. Motion-regime summaries cover stationary,
straight, turning, slip, reset, and encoder-quality cases with speed errors
reported per regime.

Calibration provenance is recorded in
`sdu_apex_autodrive/artifacts/calibration/MANIFEST.yaml`. The header-only
`derived/full_envelope_acceleration_map.csv` artifact was removed; no runtime
calibration table was changed by this hygiene pass.

### MPC and vehicle model

The MPC wrapper in `MPC/src/mpc_hardware_node.c` is AutoDRIVE-native:

- Inputs: `/odom`, `/ekf_pose`, and `/local_raceline`.
- Trigger: one solve per newest native `/odom` sample.
- Output: `/cmd/controller`.
- Watchdogs publish neutral when required team inputs are stale or missing.
- The local raceline is projected by nearest path segment; it is not assumed to
  start at the vehicle.
- Solver timing/iteration telemetry is available on diagnostic topics.

`MPC/src/vehicle_model.c` now uses the documented lateral slip-knot Hermite
approximation with odd symmetry and the correct `sec^2(alpha)` derivative.
Exact simulator cubic coefficients are not published, so this is an explicit
documented-knot approximation rather than a claim of exact simulator parity.

### Mapping and planning

The mapping launch has a five-lap saver and writes:

- `f1tenth_planning/maps/autodrive_track_5laps.yaml`
- `f1tenth_planning/maps/autodrive_track_5laps.pgm`

The default planning output is:

`f1tenth_planning/trajectories/autodrive_track_5laps_raceline_full_range.csv`

The planning README documents map extraction, raceline optimization, direction
handling, speed limits, wall distances, and MPC-compatible output. The saved
trajectory uses map coordinates and is the default Pure Pursuit/MPC reference.

The current official ICRA competition simulator was verified against the
historical `icra_2025` map/raceline geometry before promotion to the runtime
default. Its map is retained under the explicit
`autodrive_compete_2026.{pgm,yaml}` name; the PGM hash is unchanged from the
historical source. Two new five-lap mapping attempts were also made on the
current simulator after increasing slam_toolbox's transform timeout to 1.0 s
and TF buffer to 60 s. The live map published, but each run collided before
completing a lap, so the partial maps are retained as diagnostics and are not
used for control.

### Collision diagnostic

The actuator interface contains an optional simulator collision latch/reset.
When enabled for diagnostic testing, a collision count change publishes neutral
and issues the simulator reset command. It is deliberately disabled by the
default competition launch because collision/reset is not a legal controller
input. Ground-truth collision monitoring remains diagnostic-only.

## Cleanup already represented

The rewrite removed the old runtime orchestration and physical-car baggage
described in the bundle, including old command mux/arming paths, Nav2 AMCL
integration, legacy VESC/Hokuyo/`ego_racecar` wrappers, obsolete localization
and benchmark nodes, and redundant calibration/test launch paths.

Intentionally retained:

- `f1tenth_planning` map/raceline generation and optimizer inputs.
- Current calibration CSVs and derived tables as evidence.
- The local `f1tenth_planning/.venv`, which is user data.
- Historical README/comments mentioning FPGA, MPCC, or physical tooling that do
  not create runtime dependencies. These can be rewritten in a separate
  documentation cleanup pass.

The checkout was clean at the recorded revision when shutdown began. The source
tree occupied approximately `1.9G`; the prior runaway Unity `Player.log`
issue was already resolved. The active simulator log was redirected/truncated
and the headless helper defaults to `/dev/null` logging. Do not perform a broad
recursive delete to address disk usage.

## Verification completed

All of the following were completed in the ROS 2 Humble AutoDRIVE workspace
container, not on host ROS Jazzy:

```text
Full colcon build, all 7 packages, Release, BUILD_TESTING=OFF: PASS
Focused Humble rebuild after calibration/odometry changes (`f1tenth_localization`,
`sdu_apex_autodrive`): PASS
f1tenth_localization CUDA/CPU targets: built successfully
MPC standalone vehicle_model test: PASS
MPC standalone closed_loop_benchmark: PASS
f1tenth_control + mpc_riccati colcon tests: 24 tests, 0 failures
f1tenth_control Pure Pursuit closed-loop test: PASS
f1tenth_control cppcheck/xmllint checks: PASS
sdu_apex_autodrive speed-controller pytest: 13 passed
calibration analyzer regression pytest: 17 passed (including speed-controller tests)
timestamp-aware analyzer run on the 20260831 speed-feedback dataset: PASS
git diff --check: PASS
full-range throttle diagnostic: completed, no collision
full-range speed feedback diagnostic: completed, no collision
steering response diagnostic: completed, no collision
high-speed coast/asymptote diagnostic: completed, no collision
live simulator full-suite recorder after explicit UI connection: completed,
  4,276 rows; all six odometry diagnostic fields populated
live odometry rebaseline check: 10 encoder-reset transitions preserved x/y and
  speed exactly at the sampled boundary; maximum sampled yaw change was
  0.000134 rad; the raw run recorded nine simulator collision/reset events and
  is diagnostic evidence, not a replacement for the clean calibration tables
```

The Python speed-controller tests are currently run directly with pytest; the
Python package's colcon test registration does not discover them automatically.

## Not yet proven / remaining work

These tasks are required before claiming a fully validated, competition-ready
stack:

1. **Runtime AMCL global recovery.** Start the unified stack with a deliberately
   wrong initial pose and verify custom AMCL convergence using only `/map`,
   official LiDAR, and `/odom`. Record convergence time, covariance, heading
   sign, and map→odom TF.
2. **Runtime EKF accuracy.** Run AMCL + EKF while driving and compare
   `/ekf_pose` to ground truth using the diagnostic monitor only. Separate
   simulator reset transients from ordinary motion and report steady-state
   position/yaw errors.
3. **Pure Pursuit real-lap acceptance.** Run the current raceline on the closed
   track, verify the straight does not produce the previous unexplained left
   turn, verify map/trajectory orientation, and complete at least five clean
   laps. Capture lap time, collision count, cross-track error, covariance, and
   command cadence.
4. **FTG mapping acceptance.** Complete the five-lap mapping procedure and
   inspect PGM/YAML origin, resolution, orientation, and closure. Regenerate the
   raceline only after the map is verified.
5. **Stanley/lateral-planner acceptance.** Test Stanley independently and test
   the optional lateral planner/MPC raceline on the same map. Do not let stale
   or empty `/local_raceline` silently become an inverted path.
6. **MPC real-simulator acceptance.** Run MPC with the new asymptotic lateral
   model, record solve time, iteration status, path projection, speed, steering,
   and at least one complete lap. Tune weights only after this data is captured.
7. **Steering actuator identification.** Confirm official steering feedback
   unit/sign and quantify command-to-angle delay at native cadence. The sweep is
   evidence, not yet a formal MPC steering model.
8. **Reset synchronization.** Encoder/IMU rebaselining and odom continuity are
   now implemented and live-checked, but `/odom`, AMCL, EKF, and ground truth
   still need a shared explicit epoch for a complete reset acceptance test.
9. **Complete topic audit.** With one controller active, use `ros2 topic info -v`
   to confirm no production node subscribes to IPS, truth pose, collision, lap,
   camera, or simulator debug topics. Keep truth monitors outside competition
   launch.
10. **Final redundancy pass.** Decide whether historical artifacts, old
    documentation references, and `.venv` should be archived. Do not delete
    them until ownership and reproducibility value are confirmed.

The implementation, calibration infrastructure, and offline tests are
substantially complete, but items 1–6 are integration evidence that was not
completed before shutdown. Pure Pursuit, AMCL/EKF, and MPC must not be declared
fully working solely from successful builds or offline tests.

### Resumed live-simulator evidence (2026-09-01)

The pinned simulator image was rebuilt because the prior container had an empty
`/home/autodrive_simulator`. The official binary then rendered successfully and
required its menu `Disconnected` control to be activated before the bridge
reported `Connected`. A connected full-suite run produced:

- `raw/runtime_diagnostics_20260901_connected/full_suite_20260901_064608.csv`:
  4,285 rows, with the pre-fix odometry observer; it is retained to document
  the terminal reset at 37.354 s.
- `raw/runtime_diagnostics_20260901_imu_rebaseline/full_suite_20260901_065233.csv`:
  4,276 rows after the IMU rebaseline change; ten reset transitions kept
  odom x/y/speed continuous and the largest sampled yaw delta was 0.000134 rad.

Both runs were made with simulator truth in the recorder only. They did not
change the active runtime calibration tables, and their repeated collision or
reset segments must not be treated as clean high-speed calibration evidence.

A connected unified Pure Pursuit localization smoke test was also resumed on
the same simulator image. The simulator rendered its open-ground scene, while
the launch used the saved five-lap map and raceline. AMCL repeatedly refined a
low-covariance candidate but rejected it at `0.703 m > 0.650 m`, so no
`/amcl_pose` or `/ekf_pose` was published and Pure Pursuit correctly remained
inhibited. This is unresolved racing acceptance evidence, not a justification
for widening the track gate or using simulator truth in production.

Follow-up live evidence used the current official competition track and the
historical map/raceline candidate. The map server loaded at 1054 x 580 cells at
0.025 m/cell, AMCL converged and accepted `(10.328, -9.594, 2.921)` with the
confidence floor reduced to 0.20, and Pure Pursuit selected raceline index 306
with 0.386 m cross-track error. The diagnostics-only ground-truth monitor
reported 0.000 m and 0.000 rad AMCL, EKF, and odometry error while stationary.
The candidate was tested with a zero speed cap, so this proves startup
localization/map alignment only; it is not a racing lap. The corrected Pure
Pursuit command path now leaves longitudinal acceleration at the actuator
boundary, which already closes the measured speed loop.

### Live closed-track tuning probes (2026-09-01)

After rebuilding in the Humble workspace, each probe restarted both containers,
started the unified launch before the simulator, activated the simulator menu
connection, and waited for `autodrive_bridge: Connected!`. Ground truth was used
only by the diagnostics-only monitor; it never entered AMCL, EKF, or the
controller.

The supported baseline was the promoted ICRA map/raceline, compact AMCL
(cluster floor 0.20, association radius 0.80 m, local correction gain 0.08,
cloud recentering enabled), sensor odom wheel-observer correction gain 0.75,
and Pure Pursuit minimum lookahead 0.65 m / curvature feed-forward 0.25.

- At controller cap 0.5 m/s, one collision occurred after about 57.8 s; before
  collision AMCL/EKF/odom maximum errors were 0.773/0.723/0.669 m.
- At 0.2 m/s, one collision occurred after about 73.6 s; maximum errors were
  0.939/0.951/1.099 m. This reached farther into the track but did not
  complete a lap.
- At 0.12 m/s, one collision occurred after about 77.1 s; maximum errors were
  1.816/1.804/1.930 m. Lowering the cap alone is therefore not an acceptance
  fix.
- A runtime wall-bias probe and stronger short-lookahead/feed-forward probe
  were rejected; both produced worse localization divergence before collision.
- A retained-cloud AMCL probe was rejected: it selected a corridor alias and
  reached 3.338 m AMCL error. Compact-cloud AMCL remains the selected baseline.

The CSV/log evidence is retained in
`sdu_apex_autodrive/artifacts/calibration/raw/runtime_diagnostics_20260901_track_tuning/`.
These are diagnostic/reset-containing probes, not clean-lap or competition
acceptance evidence. No speed increase or AMCL gate widening is promoted.

The additional 0.25 local scan-correction probe was rejected: it reached
14.522 m maximum AMCL error after selecting a wrong scan mode, so the selected
runtime gain remains 0.08. A fresh 0.20 m/s probe with the odometry stop guard
also collided at recorder time about 70.5 s and reached pre-reset maxima of
5.408/5.421/5.546 m for AMCL/EKF/odom. After the simulator reset, the guard
stopped publishing stale observer motion and the odometry speed settled at
0.0 m/s. This validates the stop behavior only; it is not clean accuracy
evidence.

### Map/raceline geometry cross-check (2026-09-01)

The default runtime uses
`f1tenth_planning/maps/autodrive_compete_2026.yaml` and
`f1tenth_planning/trajectories/icra_2025_raceline.csv`. The map is the
historical `icra_2025` PGM under an explicit competition-track name: SHA256
`66550e26d81641544b7a0b18bc960db87f6a123a8a9fb47d28acee99e8153352`, 1054 x
580 cells, 0.025 m resolution, origin `(-4.749971, -12.201061, 0)`. The
raceline has 4,469 points and SHA256
`ea441e7c8be90789d26d7e377c8d1f8f3ac6eb8b5a130ca6c4d0994882fd5720`.

All 4,469 raceline points are in bounds and on free map pixels. Moving
ground-truth samples from the connected track probes also landed on free map
cells. The simulator distribution does not expose its internal Unity track as
PGM/CSV, so this establishes matching live geometry and runtime pairing, not
byte identity with an internal mesh.

## Safe restart procedure

Use the repository directory explicitly; running `docker compose` from `~`
gives “no configuration file provided”. Start the unified ROS launch first so
the bridge is listening, then start the simulator application and activate its
UI connection control until the bridge reports `Connected!`.

```bash
cd /home/akselmo/Documents/GitHub/SDU-Apex-Autodrive
docker compose -f /home/akselmo/Documents/GitHub/SDU-Apex-Autodrive/docker-compose.yml up -d workspace simulator
```

Graphical simulator:

```bash
xhost +SI:localuser:root
docker exec -it autodrive_roboracer_sim bash
cd /home/autodrive_simulator
./AutoDRIVE\ Simulator.x86_64 -screen-width 1280 -screen-height 720 -screen-fullscreen 0 -ip 127.0.0.1 -port 4567
```

Headless/Xvfb simulator:

```bash
docker exec -it autodrive_roboracer_sim bash -lc '
  cd /home/autodrive_simulator
  exec xvfb-run --auto-servernum --server-args="-screen 0 1280x720x24" \
    ./AutoDRIVE\ Simulator.x86_64 -ip 127.0.0.1 -port 4567
'
```

Launch one stack only, before starting the simulator executable:

```bash
docker exec -it sdu_apex_autodrive bash -lc '
  source /opt/ros/humble/setup.bash
  source /workspace/install/setup.bash
  ros2 launch sdu_apex_autodrive controller.launch.py \
    controller:=pure_pursuit with_rviz:=false
'
```

Use `controller:=ftg`, `controller:=stanley`, or `controller:=mpc` for the
other paths. Do not separately start the bridge, odometry, actuator, or another
controller while the unified launch is active. For diagnostics only, add
`with_ground_truth_monitor:=true` or `with_collision_safety:=true`; remove
both for a rules-only run.

## Resume checklist

1. Confirm both containers are up and the simulator reports `Connected!`.
2. Source ROS 2 Humble and `/workspace/install/setup.bash`.
3. Confirm official LiDAR, IMU, left/right encoders, `/odom`, `/amcl_pose`,
   and `/ekf_pose` exist with expected frames/QoS.
4. Run the AMCL wrong-start test and EKF ground-truth diagnostic.
5. Verify map/raceline orientation numerically or visually.
6. Run Pure Pursuit conservatively for one clean lap, then progress to five laps.
7. Run MPC only after localization and raceline acceptance.
8. Keep all ground-truth recording outside the controller and remove diagnostic
   launch flags for competition operation.
