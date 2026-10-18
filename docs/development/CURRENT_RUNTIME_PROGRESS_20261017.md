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
map/raceline pair under `f1tenth_planning/`. No simulator was launched during
this cleanup.

## Purge performed

> **Data-retention correction (2026-10-18):** the statement below that no old
> bags remain was true immediately after the earlier purge, but is no longer
> true. `live_runs/` has since been repopulated with retained replay captures.
> In particular, keep `live_runs/historical_mpc_fullspeed_20261017/`: it
> contains the high-speed, three-lap odometry baselines described at the end
> of this log. Do not delete these captures as part of the retired-data purge.

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
  optimizer implementations, the unused yaw-response lookup table, obsolete
  progress archives, and the unused external track bundle.
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

## Production odometry/MPC rewrite — current work

### Implemented and build/unit-test checked; live validation pending

- `UnityPlantState` schema v4 publishes named odometry-owned motion, wheel,
  sprung-body, suspension, slip/load/force, actuator-feedback, and two-slot
  steering/throttle queue channels. The source-model origin is explicitly
  distinguished from simulator ground truth.
- MPC consumes synchronized `/odom` plus that complete plant-state message;
  its redundant direct IMU, encoder, actuator-command, and steering-feedback
  subscriptions were removed. It still uses map pose for path-relative
  projection, and its nonlinear horizon retains the vehicle's wheel, sprung-
  body, suspension, and actuator-delay states.
- The command-time predictor now has a single model-history path and rejects
  stale or retroactive source states instead of silently using the former
  alternate fallbacks.
- A source review found that MPC's steering FIFO did not match the odometry
  two-packet queue: it discarded the second pending slot. The plant transition
  now consumes slot 1, advances slot 2, then appends the new command, matching
  throttle and odometry. Inter-packet prediction now keeps the FIFO fixed
  during physics integration and advances it only at accepted command events.
- The FIFO shift/append Jacobian is filled analytically, avoiding unnecessary
  finite-difference plant evaluations for those exact bookkeeping columns.
  Focused regression cases were added for steering/throttle queue order and
  inter-packet queue timing.
- Condensed the RTI state from the old 28-coordinate layout to 25 coordinates.
  The 25 retain path error/heading, body `u/v/yaw-rate/longitudinal-accel`,
  steering delay slots plus physical steering, throttle delay slots, all four
  wheel rates, sprung-body heave/roll/pitch and rates, and all four suspension
  jounces. The separate latest-steering scalar and two previous-input memory
  coordinates were exact duplicates of queue slot 2 and are gone. Steering/
  throttle change costs now reference queue slot 2 through the QP state-input
  cross term; no physical plant state was removed.
- Reduced the solver's fixed maximum state dimension to 25 to match production
  and shrink its working arrays. Removed the obsolete augmented Riccati fast
  path, which assumed the deleted previous-input memory coordinates; the
  canonical 25-state model now uses the general Riccati recursion with its
  explicit queue-slot cross-costs.
- Corrected continuous inter-packet plant propagation so it holds both command
  FIFOs fixed during physics substeps; only a newly accepted command event
  shifts/appends a queue. Added a direct regression case for that invariant.
- The PhysX odometry correction now synchronizes the native actor's body
  velocity, yaw/roll/pitch, angular rates, and rear-wheel rates with the same
  sensor-corrected odometry epoch before its next source-plant step.
- Odom and `UnityPlantState` publication now use the synchronized sensor
  packet's source timestamp instead of callback/publication time. The observer
  still advances on its nominal 25 ms grid, so delivery jitter does not change
  its integration interval.
- Clarified the message contract for slip, load, suspension-rate, and tire-force
  outputs: they are the latest completed 1 ms PhysX solver outputs in the
  packet, before the observer's post-step state correction. They are not
  recalculated by causing an extra physics step; MPC rollout uses the
  corrected mechanical state, not those diagnostic force channels.
- Corrected the native tire-load channel: it now records the post-filter
  `tireLoad` actually passed to the PhysX tire-force callback, while
  `suspension_force_n` remains the wheel query's raw spring force. The two
  channels are now distinct by definition rather than aliases.
- Corrected an acceleration timebase mismatch in the native odometry plant.
  Unity's IMU differentiates world velocity every 1 ms `FixedUpdate`; the
  native process had differenced 25 ms-spaced outputs while dividing by the
  1 ms physics step. It now computes gravity-compensated, body-frame inertial
  acceleration and yaw acceleration on every 1 ms tick, publishing the final
  tick's sample. The ROS state message documents those acceleration channels
  as 1 ms samples, not 25 ms averages.
- The observer now corrects COM `u/v` using the proper rotating-body transport
  terms (`+r*v` and `-r*u`) when gyro yaw differs from the plant prediction,
  and trapezoidally integrates acceleration residuals across the 25 ms packet
  interval. Added focused observer tests for acceleration-residual integration
  and the yaw-rate-to-lateral-velocity coupling; both pass.
- Replaced empty-format `snprintf` calls in the 40 Hz PhysX request builder
  with bounded literal appends, removing the compile warning and avoiding
  formatter overhead for fixed commas/terminators.
- Removed stale speed-reference fields from predictor tests and updated their
  history entries to carry normalized throttle commands. Corrected an RTI test
  configuration that assigned steering-rate limit to the steering-angle
  bound, making the test's MPC configuration invalid. Updated packet-assembler
  expectations to match production's monotonic source-time policy: late and
  duplicate completed packets are dropped rather than fed into odometry.
- Corrected the prior torque audit after checking the active scene instance:
  the prefab asset default is `MotorTorque=85.6 N m`, but the active scene
  overrides it to `428 N m` (`DriveType=CAWD`, `BrakeType=CAWB`). For nonzero
  throttle, the controller sends `428 * throttle / 4` N m to each wheel. At
  exactly zero throttle, it sends zero motor torque and `428 N m` brake torque
  to each of the four wheels (not divided by four). Both odometry and MPC plant
  constants now use the scene-effective 428 N m scalar, with their existing
  per-wheel drive/brake branching. This corrects our model only; simulator
  physics was not changed.
- Confirmed the wheel-radius distinction in the active prefab: each
  `WheelCollider.m_Radius` is 0.059 m, while the controller's separate
  `WheelRadius=0.0325` field is only used by its skid-steer branch. The active
  front-wheel-steer/common-AWD physics, PhysX wheel geometry, and encoder
  conversion therefore correctly use 0.059 m; no radius change was made.
- Cross-checked the checked-in Unity timing and PhysX 4.1 implementation:
  Unity fixed timestep is 1 ms; the vehicle uses five substeps below 5 m/s and
  one at/above 5 m/s. The source plant and native PhysX setup use the same
  threshold/count. PhysX's `+0.1 m/s` longitudinal-slip denominator applies
  when wheel torque is active; the configured 4 m/s floor is the no-torque
  branch. The current production commands select drive or brake torque, so the
  source plant's active-torque branch matches that documented logic.

### Not yet established

- The source-only check found no references to the retired augmented Riccati
  path. A prior torque audit missed the scene-level override; that finding is
  corrected above. This is not a compile or behavioral test.
- The changed localization and MPC packages build in Release. Their focused
  test suites currently pass: 40 tests, 0 failures. This establishes compile and
  equation/invariant coverage only; no accuracy improvement, exact Unity
  parity, 40 Hz deadline, or driving success is claimed.
- The final serialization-helper change was rebuilt successfully in
  `f1tenth_localization`; it does not alter the plant equations.
- The 1 ms IMU-equivalent acceleration path and the new observer residual
  correction have been checked against the Unity IMU source and focused
  observer equations. Live packet alignment and behavior remain unverified.
- Any offline rollout comparisons made with the briefly substituted 85.6 N m
  constant are not validation of the scene-effective 428 N m production model.
  The old bags were purged, so new validation must use newly generated
  simulator data. The restored 428 N m scalar matches the checked-in scene
  override but still needs runtime validation.
- Slip/force/load outputs correspond to the final completed plant substep and
  are documented as pre-correction diagnostics. Their numerical agreement
  with the sensor-corrected mechanical state is not yet validated; MPC must
  not treat them as post-correction measured values.
- The native odometry plant uses public PhysX vehicle APIs and reconstructed
  Unity settings. It is not yet demonstrated to be a bit-exact copy of Unity's
  private WheelCollider/contact solve.
- Odom's persistent estimate is stepped by the native PhysX vehicle process,
  while MPC's differentiable rollout is a separate C++ source-derived plant.
  They consume/produce the same named state contract and intended scene
  parameters, but are not a shared numerical transition function; their
  step-to-step parity has not been measured on matched simulator inputs.

### State, cost, and fixed-dimension audit — 2026-10-17

The production optimizer has fixed dimensions `N=30`, `n_x=25`, `n_u=2`.
Its decisions are absolute steering target and normalized throttle; zero
throttle selects Unity's brake branch. Speed and longitudinal acceleration
remain references in the tracking objective, not actuator commands. No
weight values were changed in this audit because there is no new vehicle
validation data to support retuning them.

| RTI coordinate(s) | Meaning | Representation decision |
|---|---|---|
| `e_y`, `e_psi` | Path-relative geometry and heading | Keep optimized: determine corridor feasibility and path tracking. Global pose/progress is external; progress increments are propagated from the plant. |
| `u`, `v`, `r` | Body forward/lateral velocity and yaw rate | Keep: recursive vehicle motion states. |
| `a_x` | Latest longitudinal acceleration from the source-plant transition | Not an independent physical state, but keep as an RTI coordinate because it has a tracking cost and box constraint. Eliminating it exactly requires lifting its affine output cost into dense/low-rank state-input terms; the current solver cost representation is diagonal `Q/R` plus `N` cross terms. Do not drop its cost or bound. |
| `steer_queue[1:2]`, `throttle_queue[1:2]` | Two-packet command delays | Deterministic FIFO shifts, not free decisions. Keep in the Markov state because they determine which input is applied next and the prior command-change cost. Removing them requires an equivalent delayed-control coupling/condensed solver, not deletion from the plant. |
| physical steering angle | Unity steering actuator response | Keep: dynamic and affects future wheel forces. |
| four wheel angular speeds | Wheel spin/slip dynamics | Keep: dynamic and coupled to tire forces and body motion. |
| heave, roll, pitch and their rates | Sprung-body motion | Keep: dynamic suspension/load-transfer inputs to the source plant. |
| four suspension jounces | Per-wheel suspension history | Keep: recursive states for suspension and tire-load calculation. |

Four entries in Unity's 18-value source vector (`u`, `v`, yaw rate, and
physical steering) duplicated dedicated MPC coordinates. The MPC now carries
only the other 14 recursive wheel/body/suspension values and materializes the
Unity 18-vector at the source-plant boundary. This removes duplicate storage
and the possibility that those two representations disagree. Per-wheel
longitudinal/lateral slip and normal load are transition outputs, not
recursive states or QP decisions. They now live only in `MpcStageResult_t`,
not in every `MpcModelState_t`; optional horizon diagnostics reconstruct them
only at the five displayed prediction checkpoints. The ROS `UnityPlantState`
contract and odometry publication remain unchanged. Together, the 4 removed
duplicated doubles and 12 diagnostic doubles reduce each copied MPC/RTI state
record by 128 bytes without removing any nonlinear plant state.

The objective audit also found that the QP applied
`weight_longitudinal_acceleration`, but nonlinear candidate ranking omitted
the corresponding acceleration-tracking error. Candidate ranking now uses
the same acceleration term as the QP and terminal tracking objective. This
does not alter vehicle physics or the configured weight; it removes an
internal objective mismatch. Regression assertions cover the acceleration
quadratic and linear terms in stage and terminal QP costs, and confirm the
scalar tracking cost changes by exactly the weighted acceleration error.

The Riccati solver API is now specialized for `N=30`, `n_x=25`, `n_u=2`:
runtime dimension arguments and warm-start/factorization shape metadata were
removed, local loop bounds are compile-time constants, the input Schur
complement is always the production 2x2, and the production RTI solve rejects
any horizon other than 30. This removes the unused reduced-dimension/1x1 path
instead of carrying it as a second solver mode. The factorization is reused
across ADMM RHS updates when its curvature and penalties are unchanged.
Compiler-directed unrolling/SIMD remain deferred pending profiling; raceline
references are already preloaded and never become optimizer variables.

This continuation initially built `f1tenth_mpc` in the available ROS Jazzy
environment (ROS Humble is not installed on the host). Those build and unit
checks were followed by a Humble-container integration attempt below; neither
constitutes accuracy, Unity-parity, or worst-case timing evidence by itself.

### Riccati symmetry work — 2026-10-17 continuation

The backward recursion now forms only the upper triangle of each symmetric
value-Hessian update and mirrors it into the lower triangle, in both the
cached-factorization and unfactored paths. This preserves every state, control,
constraint, and cost term while avoiding the redundant lower-triangle
`A' P A + G' K` accumulation and separate averaging pass. The intermediate
`P A` product is still dense and unchanged. A focused full-size `25x2x30` test
checks exact stored symmetry and finite values, then compares factored and
unfactored primal trajectories within `1e-5`; it passes. This is an algebraic
operation-count reduction, not yet a measured wall-clock or 40 Hz improvement.
After this addition the three MPC test binaries report 40 passing cases, with
no failures at that checkpoint; the updated test total and live run are below.

### Supervised practice integration attempt — 2026-10-17

- The reused Humble image contained generated build/install output for deleted
  mapping files (`five_lap_mapping.yaml`, old SLAM config, and mapping launch).
  The first mounted-source build failed before launch because its stale Python
  package manifest still referenced those files. The development entrypoint
  now removes only that package's generated build/install artifacts inside the
  disposable container before rebuilding the mounted source. All five runtime
  packages then built successfully in Humble and the practice map/raceline
  loaded. No source, map, or reference trajectory was deleted or changed.
- The debug recorder initially lacked Humble type support for the current
  `UnityPlantState` message. It now builds `sdu_apex_msgs` from the mounted
  source before recording and subscribes successfully to `/odom/unity_plant_state`.
- One practice batch-mode screen was started with the watcher armed for one
  lap, 180 s maximum, and a 20 s no-progress stop. It ended with **0 laps,
  0 collisions, and 0 accepted MPC solve cycles**. The watcher stopped the
  stationary car on no odometry progress; it was not a driving success.
- The retained diagnostic bag is
  `live_runs/runtime_rewrite_practice_20261017/run` (11 MB, 50.45 s,
  25,814 messages). It includes simulator debug odometry/IPS, estimator odom,
  full odometry-owned plant state, AMCL/EKF, commands, MPC diagnostics, and
  bridge timing. The earlier recorder-only empty bag was removed.
- Measured in the bag: `/odom` and `/odom/unity_plant_state` each had 807
  samples, median source interval **24.9766 ms (40.04 Hz)**, p95 **25.687 ms**;
  the two streams' source timestamps matched exactly for all 807 samples.
  `/current_map_pose` had 803 samples and, after its first four startup
  messages, shared the same sampled timestamps. Thus the observed failure was
  not a 20 Hz sensor stream.
- MPC diagnostics show 5 initial `missing_map_pose` rejections, then 802
  `waiting for a synchronized complete odometry plant state` rejections.
  The exact-timestamp requirement on independently delivered odom and plant
  callbacks can reject a causal plant snapshot that arrives one packet behind
  the odometry-triggered control callback. The existing control-time predictor
  already propagates a past plant state to the current sensor epoch, so the
  handoff gate now accepts a plant snapshot only when it is not newer than
  either the odom or fused state and its age remains within the existing
  extrapolation bound. Three focused tests cover matching, one-packet-old,
  future, stale, missing, and invalid snapshots.
- After the gate change, the Jazzy test result is **44 tests, 0 failures** and
  the MPC package builds. This is not yet live validation of the correction.
- Ground-truth ROS topics available for this run cover pose, body odometry,
  velocity, and yaw. The simulator does not expose independent truth for all
  estimated suspension/load/slip internals in these ROS topics; those latent
  channels cannot be called truth-validated from this bag alone.

### Remaining work, in order

1. Implement the hybrid direct-throttle/brake formulation and delayed
   speed/wheel-spin candidate checks described below before another practice
   run; do not repeat the unchanged speed-profile controller.
2. On the next moving run, compare estimator odometry to simulator debug
   odom/IPS:
   forward/lateral speed, yaw rate, pose/yaw drift, plus the full published
   plant state against whatever independent Unity truth is actually exposed.
   Measure MPC horizon prediction errors at its recorded checkpoints, and
   report unavailable ground-truth channels explicitly rather than treating
   the estimator's own state as truth.
3. Measure bridge/odom/MPC cycle timing on that run, including solve-time
   p50/p95/p99/max and deadline misses against the 25 ms period. Fix8 already
   measured RTI timing below the configured budget, but did not establish
   timing for a corrected controller.
4. Finish the remaining static units/sign and source-vs-MPC transition parity
   audit with matched moving data, then profile the fixed `25x2x30` solver.
   Test unrolling/SIMD or targeted kernels only against a measured hot loop,
   with numerical equivalence checks; the symmetric-Hessian operation-count
   reduction is not yet a measured wall-clock improvement.
5. Resolve any demonstrated native-PhysX-vs-MPC transition mismatch before
   weight tuning or claims of accurate rollout. Preserve the practice
   reference trajectory and simulator physics throughout.

### Tracking and direct-throttle root-cause audit — 2026-10-17

This audit uses the existing supervised practice capture
`live_runs/runtime_rewrite_practice_20261017_fix8/run`; no simulator was
launched for this diagnosis.

- The MPC does publish normalized throttle directly, and its nonlinear plant
  applies the checked-in Unity torque/brake branch. However, the optimization
  is not yet a complete torque-centered design: it still tracks the raceline's
  speed and acceleration profile with `weight_u=200`,
  `weight_u_overspeed=200`, and `weight_longitudinal_acceleration=20`, while
  absolute throttle costs only `0.5` and throttle-change costs `5`. These
  weights were not empirically retuned for the new actuator semantics.
- The input is hybrid, not a conventional continuous acceleration input:
  exactly zero normalized throttle commands 428 N m brake torque at each
  wheel, while any positive value commands motor torque divided across four
  wheels. The RTI optimization still treats throttle as one continuous QP
  coordinate; it has no explicit drive-versus-brake mode schedule, and no
  wheel-slip objective or useful wheel-speed/slip constraint. Thus lowering
  positive throttle is not the same action as braking, and the QP's local
  smooth model cannot represent the zero-throttle branch switch as a normal
  continuous control change.
- The active practice line remains usable as a *spatial path/reference* for
  projection and corridor geometry. Its stored speed/acceleration profile is
  not demonstrated feasible for this delayed direct-throttle/brake plant and
  must not be treated as such. The capture shows the required distinction:
  at a moving interval the MPC state was about 4.4 m/s while its local path
  speed target was about 3.2 m/s, but a rejected-horizon best-effort action
  still published about 0.48 throttle. Later the state exceeded 5.3 m/s while
  the target remained about 3.4 m/s; zero throttle was not published until
  roughly one 25 ms command interval before contact. The two-packet actuation
  pipeline means that braking command could not affect the vehicle before the
  recorded collision. This is direct evidence that the candidate output and
  speed reference were not being reconciled safely when the horizon failed.
- Across 222 active MPC cycles, 101 were optimal and 6 degraded; 72 were
  residual-rejected, 41 failed exact nonlinear corridor rollout, and 2 had
  invalid input. Best-effort first actions were published in 106 cycles.
  Those actions are checked for one-step corridor feasibility and proximity
  to the nominal action, not for the full delayed speed/slip consequence.
  The 160-iteration change improved numerical acceptance relative to fix7 but
  did not solve tracking and did not avert contact. Do not raise the iteration
  cap again as a substitute for fixing the control formulation.
- Odom is also not currently accurate enough to justify treating it as the
  proven source of truth for this runtime. In this capture, around the
  pre-contact acceleration/turn, estimated forward speed exceeded Unity's
  reported body-forward speed by roughly 0.2–0.5 m/s; one-step delta errors
  from the prior matched-epoch analysis were 0.085 m/s RMSE in `u`, 0.022 m/s
  in rear-axle `v`, and 0.255 rad/s RMSE in yaw-rate change. The prior limited
  PhysX/model result is not proof of this current integrated estimator or of
  full-horizon path tracking. `UnityPlantState.state_origin=1` explicitly
  denotes a source-derived estimate, not Unity ground truth.
- Timing was not the observed root cause in fix8: measured RTI p95 was about
  17.7 ms and max about 18.3 ms against a 20 ms MPC budget / 25 ms packet
  period. The principal observed failures were control-horizon rejection,
  delayed braking, stale/best-effort positive throttle, and plant/reference
  mismatch—not a 20 Hz sensor feed.

#### Required correction before another practice run

1. Rework the longitudinal optimizer around Unity's actual hybrid input:
   make braking (`q=0`) an explicit scheduled mode and optimize positive motor
   torque only in drive mode, accounting for the two-packet command delay.
   Do not pretend a small positive throttle is a braking command.
2. Make the nonlinear acceptance/recovery logic score the delayed speed and
   wheel-spin consequences of a candidate; a one-step corridor-only check is
   insufficient grounds to publish a rejected-horizon throttle command.
3. Add a source-derived, per-wheel slip/wheel-spin penalty or feasible bound
   to the MPC formulation. The plant carries wheel angular speeds, but the
   present QP does not use them to discourage excessive spin.
4. Retain the existing raceline geometry for initial path tests, but regenerate
   its speed profile against the direct-throttle plant and actuator delay.
   Retune weights only after the mode-aware objective and speed profile are
   consistent; do not tune the current weights against the incompatible
   profile.
5. Revisit the odometry speed discrepancy using the captured source/Unity
   body velocity and plant-state channels before calling the estimate
   production-accurate. Then perform one supervised practice run of the
   corrected implementation, requiring command response, accepted MPC
   behavior, and full timing/trajectory comparisons; stop immediately on
   collision.

### Odometer PhysX wheel-spin correction — 2026-10-17

- Source inspection confirms the runtime advances the public PhysX vehicle
  plant for 25 fixed 1 ms steps per 40 Hz packet, with Unity-derived vehicle,
  WheelCollider, tire-curve, torque/brake, steering, and contact settings. The
  active Unity `WheelEncoder` reports accumulated wheel angle quantized by
  `PPR=16` and `GearRatio=120`, i.e. 1920 counts/revolution; the wheel radius
  used for encoder peripheral speed is the active WheelCollider radius,
  0.059 m.
- Found a stale feedback path: each packet previously overwrote PhysX rear
  wheel angular velocity with an encoder slope averaged over a 100 ms window.
  That adds roughly 50 ms of group delay precisely where torque and braking
  change wheel spin. The 100 ms window and its configuration were removed.
- Rear-wheel corrections now compare the 25 ms encoder-angle increment with
  PhysX's integrated wheel-rotation increment. The endpoint angular-speed
  correction is `2 * angle_residual / dt`, the endpoint consequence of a
  constant acceleration-model residual over the interval. One encoder count
  of residual is treated as unresolvable quantization, and PhysX's actual
  10π angle wrap is undone before comparison. Corrected rear spin is fed back
  into PhysX before its next 1 ms step; the plant remains the between-packet
  predictor rather than replacing it with a delayed encoder average.
- The odometry diagnostics now preserve raw pre-correction PhysX `u/v/r` and
  expose rear encoder average speed, model-angle residual, wheel-spin
  correction, and raw model/encoder wheel rates. Previously those raw fields
  were overwritten by the corrected estimate before diagnostics published,
  making model error invisible. This is a diagnostics-layout revision; the
  layout label is `sensor_corrected_physx_odom_v3_wheel_spin_residual` with 30
  values.
- Focused observer tests cover the integrated-angle correction, quantization
  deadband, and PhysX 10π wrap. The localization package builds with the
  optional CUDA AMCL target disabled; all workspace test results currently
  report 52 tests, 0 failures. A one-packet native-PhysX protocol smoke check
  also confirmed the new four-wheel-angle response is parsed correctly.
- This is a mechanically grounded wheel-spin observer correction, not yet a
  demonstrated improvement in body speed, yaw, or pose error. The 25 ms
  encoder average is still an interval measurement; the endpoint correction
  assumes the unmodeled wheel-acceleration residual is approximately constant
  over one packet. Validate that assumption on a new practice capture before
  treating the wheel state or odometry as improved in production.
- No simulator was launched for this change. The fix8 body-speed mismatch
  (about 0.2–0.5 m/s around the turn) remains unresolved, as do the explicit
  drive/brake MPC schedule, delayed candidate speed/wheel-spin feasibility
  checks, and a speed profile generated for this actuator plant. Do not launch
  another practice run until that MPC correction is implemented; then perform
  one supervised run and compare Unity truth, raw PhysX, corrected odometry,
  command/feedback delay, and collision state at matched 25 ms epochs.

### Hybrid-throttle MPC correction and PhysX-pose evaluation — 2026-10-17

- MPC now schedules an explicit Unity brake or motor branch at every horizon
  stage. Brake is exactly `q=0` (428 N m per wheel); drive is bounded to
  `q>=0.001`, the smallest positive value surviving the bridge's 3-decimal
  command serialization. The QP bounds cannot cross the zero/brake
  discontinuity within a stage.
- Mode selection looks two packet stages ahead, including queued commands,
  predicted speed versus the clamped speed reference, target acceleration,
  and the source plant's peak per-wheel longitudinal slip. The nominal speed
  reference is clamped to the lower/upper result of full-brake and full-drive
  source-plant rollouts under the queued commands and path steering. This is
  a local reachable bracket, not a proof that an entire lap speed profile is
  feasible for every intermediate throttle sequence.
- Exact candidate rollout retains the command-delay states and rejects the
  untransmittable positive-throttle deadband. The old rejected-horizon
  escape—which could publish a one-step corridor-safe throttle without
  validating the full delayed horizon—has been removed. The source tire-curve
  asymptote is a force plateau, not a physically impossible-state boundary;
  the later correction below replaces that incorrect hard rejection with a
  source-derived soft slip-loss objective.
  Only an accepted full-horizon result is published; otherwise the existing
  zero-throttle brake fallback is used. Weights remain unretuned pending
  evidence from the corrected formulation.
- The native PhysX plant's rear-axle pose was tested against the recorded
  practice truth rather than assumed superior because it uses 1 ms steps.
  On the fix8 pre-contact segment, raw PhysX pose had 0.194 m position RMSE,
  versus 0.118 m for recorded production odom. Reintegrating the updated
  sensor-corrected `u/v` at the established 25 ms epoch reduced RMSE to
  0.085 m. Production `/odom` therefore uses the corrected velocity
  integration; raw PhysX rear-axle pose remains separately available for
  diagnosing model drift in `UnityPlantState` schema v5 and the v4 odometry
  diagnostics layout.
- A separate fix7 replay checked transfer, truncated at its first collision:
  6.154 s / 247 samples, with the source run reaching 4.37 m/s. Position RMSE
  improved from 0.461 m in the recorded production odom to 0.145 m in the
  updated replay; forward-speed RMSE improved from 0.682 to 0.251 m/s, and
  lateral-speed RMSE from 0.164 to 0.130 m/s. On fix8 (5.43 s / 218
  pre-collision samples), position RMSE improved 0.118 to 0.085 m, forward
  speed 0.149 to 0.127 m/s, and lateral speed 0.177 to 0.169 m/s. The
  yaw-rate channel is gyro passthrough, so its zero residual is not an
  independent predictor result; yaw RMSE was unchanged at 0.0127 rad on fix8
  and 0.0055 rad on fix7.
- These are two short, collision-truncated replays, not live validation or
  full-lap evidence. Position and velocity error remain material; the replay
  metrics establish a gain over the previous estimator, not acceptable or
  near-zero odometry. Independent wheel-spin truth is not present in these
  ROS bags, so per-wheel model slip remains unvalidated against Unity truth.
- On this host, odometry replay packet-processing p99 was 10.6–11.4 ms and
  maximum 13.8–16.6 ms, below the 25 ms sensor period. This is not an MPC
  timing result and not a measurement inside the competition container under
  Unity load. The focused build for `sdu_apex_msgs`, `f1tenth_localization`,
  and `f1tenth_mpc` succeeds; all 54 package tests pass, and the runtime topic
  policy check passes.
- No new Unity run was started: the GPU was occupied by another application
  during this evaluation. Next: run one supervised practice screen through
  the established batch-mode/Xvfb procedure with the collision watcher armed;
  verify schema-v5 handoff acceptance, full-horizon MPC acceptance/fallback
  rate, actual throttle/feedback delay, odom against Unity debug pose/twist,
  and both odom and MPC timing. Abort on the first collision. Do not promote
  odometry accuracy or tune weights until that live comparison is available.
- A component-only MPC replay was attempted against the old map-pose stream
  and separately replayed odometry. The independent bag clocks/arrival times
  produced map/odom synchronization rejections and zero solve cycles; those
  generated bags were discarded. This is a failed harness, not an MPC result.
  MPC acceptance, throttle behavior, and the added reachability-rollout cost
  therefore remain unmeasured until a same-clock integrated replay or live
  practice run is available.

### Follow-up: same-clock MPC replay and odometry error profile — 2026-10-17

- A single bag containing timestamp-aligned map pose, `/odom`, and schema-v5
  plant state was replayed through the current MPC component. The output bag
  has 215 diagnostics/commands over 5.36 s: one initial missing-map-pose
  rejection, 212 `command_time_before_state` synchronization rejections, and
  two solver rejections (`iterations=0`). No candidate was accepted, so this
  replay says nothing about throttle tracking or the drive/brake schedule.
  The maximum observed solver call was 22.8 ms; because both calls rejected,
  this is not a valid 40 Hz performance or acceptance result.
- The controller calls `now()` inside the odometry callback and synchronizes
  that time against the newest fused header stamp. This replay uses the bag's
  simulated clock while the source bags contain no `/clock` topic; a playback
  clock that trails a just-delivered input can therefore explain the
  `command_time_before_state` burst. The output diagnostics do not currently
  include both clock and input timestamps, so this explanation is not yet
  directly proven. Do not change production synchronization based on this
  invalid controller replay; make the harness expose/align those timestamps
  or use a live clock before evaluating MPC behavior.
- The fix8 corrected-odometry comparison was further split into 40-sample
  windows, excluding the known collision sample and all later samples. Its
  position RMSE progresses from 2 mm in the first second to 10 mm, 37 mm,
  139 mm, then 120 mm and 99 mm in successive windows. Whole valid-segment
  results remain: position RMSE 85.3 mm (p95 181 mm, max 192 mm), forward
  velocity RMSE 0.127 m/s (bias +0.009 m/s), lateral velocity RMSE 0.169 m/s,
  relative yaw RMSE 0.0127 rad. The raw 1 ms PhysX pose is worse on this run
  (194 mm RMSE), which is why production pose integrates sensor-corrected
  PhysX-predicted `u/v` rather than blindly publishing native model pose.
  This confirms a measurable replay gain, not acceptable odometry accuracy or
  online tracking improvement.
- No simulator was started. The host GPU currently has an unrelated Payday 2
  process (PID 362159) using it; it was left untouched. The only active
  production odometry path already advances the source-derived PhysX vehicle
  25 times at 1 ms per packet and corrects its hidden state from permitted
  IMU, encoder, and actuator feedback. The raw PhysX pose is retained for
  comparison because available practice truth shows it drifts more than the
  fused pose. Further odometry changes need to target a measured residual,
  not replace that empirically better pose path with raw PhysX output.

### Source-derived tire-slip MPC objective correction — 2026-10-17

- Corrected the physical interpretation of the forward slip asymptote. Unity's
  source curve reaches its force extremum at |slip|=0.15, falls toward its
  asymptote value by |slip|=0.25, and then saturates. Therefore a state above
  0.25 can still be a valid plant state; it only means the source curve has
  reached its lower force plateau. The previous hard nonlinear-rollout reject
  at 0.25 was removed. The exact-zero brake branch and per-stage drive/brake
  mode bounds remain unchanged.
- Added one shared evaluator that uses the source Ackermann wheel angles,
  wheel locations, 0.059 m tire radius, and the exact Unity slip denominator
  `max(|wheel-forward-speed|, |wheel-surface-speed|)+0.1`. It returns signed
  longitudinal slip plus the wheel-surface-speed residual beyond the source
  curve's 0.15 extremum and its local wheel-angular-speed derivative. The
  source plant also records the maximum of that residual per wheel over every
  internal 1 ms step of a controller interval; positive drive slip and
  negative braking/lockup slip are both represented.
- Added a per-wheel Gauss-Newton local cost to the RTI QP's existing wheel
  angular-velocity coordinates and the exact source-slip loss to nonlinear
  candidate scoring. The residual is in m/s and activates only beyond the
  source curve extremum; it does not constrain the vehicle to remain below an
  arbitrary slip limit and does not alter the Unity-derived tire-force law.
  The configured weight is 50, one quarter of the nominal body-speed tracking
  weight to reflect four wheel contributions. This is an initial scale, not a
  tuned or live-validated value. The local QP Hessian retains its diagonal
  representation, so the slip derivative currently varies wheel omega while
  holding body speed and steering at the RTI linearization point; the exact
  candidate objective still scores the full coupled plant rollout.
- Tests verify the source residual and its derivative against finite
  differences for wheel overspeed and lockup, confirm the QP obtains a
  nonzero wheel-state cost and exact stage-cost increase, and confirm a
  physically valid rollout above the 0.25 asymptote is accepted rather than
  misclassified as impossible. Focused `f1tenth_mpc` build succeeds and all
  56 package tests pass.
- This changes MPC formulation but does not yet demonstrate improved driving,
  accepted full-horizon solve rate, 40 Hz timing, or lower wheel slip in the
  simulator. The latest same-clock replay still produced no accepted solve
  before this change, so it cannot validate the new cost. No Unity run was
  started: the GPU remains occupied by the user's unrelated Payday 2 process.
  Continue with the retained practice capture before seeking the next
  supervised live run; keep the collision monitor armed and abort immediately
  on contact.

### Native PhysX versus MPC-plant recursive replay — 2026-10-17

- Replayed two existing practice captures, fix7 and fix8, from the first
  synchronized `/odom/unity_plant_state` sample. Initial body/wheel/suspension
  state and the two-packet queue came from that sample. Each subsequent 25 ms
  stage consumed only recorded queue slot 1; no later sensors or truth were
  fed back. The replay stopped before the first positive collision-count
  sample: 246 transitions (6.15 s) in fix7 and 217 transitions (5.425 s) in
  fix8. Initial pose/heading were anchored to Unity truth.
- Ran both the fast MPC source plant and reconstructed public PhysX 4.1.2
  vehicle/scene on the same starting state and queued commands. On fix7,
  source-plant versus native-PhysX RMSE was respectively: forward velocity
  0.0780 vs 0.0773 m/s; lateral velocity 0.0322 vs 0.0291 m/s; yaw rate
  0.2046 vs 0.2120 rad/s; integrated rear-axle position 0.0502 vs 0.0366 m;
  integrated yaw 0.0194 vs 0.0178 rad. On fix8 the respective values were:
  forward velocity 0.0490 vs 0.0429 m/s; lateral velocity 0.0362 vs
  0.0148 m/s; yaw rate 0.1151 vs 0.1050 rad/s; rear-axle position 0.0527 vs
  0.0712 m; integrated yaw 0.0116 vs 0.0099 rad. Position p95/max absolute
  errors for source/native were 0.097/0.109 vs 0.080/0.090 m (fix7), and
  0.061/0.069 vs 0.110/0.112 m (fix8). Integrated-yaw p95/max absolute
  errors were 0.040/0.056 and 0.029/0.048 rad for the source plant, versus
  0.041/0.058 and 0.016/0.030 rad for native PhysX. Both models still have
  instantaneous yaw-rate outliers up to about 0.7–0.94 rad/s; low integrated
  yaw error partly reflects signed cancellation and must not hide that.
- Verified the pose point from the Unity prefab. The scene's `PositioningSystems`
  entry references the vehicle GPS component, whose `VehicleTransform` is the
  `Rear Axle Center` child at local `(0, 0.05, -0.16)` m. Thus simulator GPS
  truth is already at the rear axle despite its `child_frame_id=roboracer_1`
  label; compare it directly with production `base_link`. A later attempt to
  apply another rigid-body-to-rear-axle offset was wrong. Its position scores
  (0.085/0.102 m native/source on fix8 and 0.043/0.054 m on fix7) are discarded.
  The fresh direct-point scores above supersede the undocumented preliminary
  0.046 m source-plant score.
- Native PhysX improves lateral velocity and integrated yaw in both captures,
  with a small forward-speed gain; yaw-rate RMSE is mixed. Position also is
  mixed: native is better on fix7 but worse on fix8. It is not yet accurate
  enough to promote as the MPC rollout. These captures have already informed
  odometry work, so this is diagnostic replay, not untouched validation.
- A 30-stage, 750-internal-tick native PhysX horizon took 42.8 ms CPU on this
  host before any MPC optimization. Therefore replacing the complete RTI
  horizon directly with native PhysX cannot meet the 25 ms cycle budget in
  this form. The MPC remains on the fast differentiable source plant; do not
  claim its rollout is a native PhysX simulation. Use native PhysX as the
  mechanistic reference to identify the remaining fast-model mismatch, then
  implement and validate only a targeted low-cost correction.
- No Unity simulator was launched. Payday 2 still owns the GPU (PID 362159),
  and was left untouched. No live odometry, MPC acceptance, or 40 Hz claim
  follows from these offline replays.

### Causal actuator timing correction in production odometry — 2026-10-17

- The v4 fix7/fix8 bags include the pending two-packet command queue, actuator
  feedback at the packet boundary, and Unity IMU body-frame velocity. On each
  pre-collision packet, compared the next 25 ms ground-truth longitudinal
  velocity change (with the body-frame `r*v` transport term removed) when
  queue slot 1 and reported throttle feedback disagreed on drive versus brake.
  If queue slot 1 said brake while feedback still reported positive throttle,
  the next-interval mean longitudinal acceleration was -2.00 m/s² in fix7
  (12 cases; 67% decelerating) and -5.18 m/s² in fix8 (10 cases; 90%
  decelerating). In the opposite disagreement, queued drive / feedback brake,
  the means were +1.64 m/s² (13 cases) and +1.00 m/s² (11 cases). This
  confirms that the packet feedback describes the just-observed actuator
  epoch; it must not replace the queued command for the next interval.
- Found the matching implementation error in `PhysxOdomRuntime::advance`:
  it propagated the latest throttle feedback through the following 25 ms and,
  when steering feedback was valid, interpolated toward the same already
  measured steering angle instead of advancing Unity's steering servo toward
  the pending command. The command FIFO was maintained and published, but its
  head was bypassed exactly when feedback was available—the normal production
  case.
- Corrected the causal transition: current steering feedback now anchors the
  physical steering state at the packet boundary; the Unity 1 ms steering
  rate/target rule then advances it using queued steering slot 1 for all 25
  ticks. Queued throttle slot 1 selects the source drive/brake branch over
  those same ticks. The newly received controller command is appended only
  after the pending head has been consumed. Latest throttle feedback remains
  an observed diagnostic; it is not treated as a future actuator command.
- Extracted the 25-tick actuator profile into a small pure helper and added
  four equation tests for conflicting stale-feedback/queued-mode cases,
  measured-steering anchoring, and command-rate propagation. The Release
  localization package and `sensor_odometry_node` rebuilt successfully in
  16.7 s with GPU AMCL disabled. Focused odometry and actuator tests pass (2/2
  targets; 4 new actuator cases), `git diff --check` passes, and the runtime
  topic-policy check passes.
- Replayed existing fix7/fix8 sensor inputs, actuator feedback, and commands
  through the rebuilt production odometry node in an isolated ROS domain.
  Each run emitted 247/218 pre-contact outputs with exact source-stamp matches
  to Unity bridge odometry. The truth twist reference point was corrected
  before scoring: bridge `twist` exactly matches `V1 Linear Velocity` at the
  request stamp (zero residual over 244 fix7 and 217 fix8 packets), and Unity
  `IMU.cs` obtains that velocity from the Rigidbody at its center of mass.
  Production `/odom` twist is at the rear-axle `base_link`; compare lateral
  velocity against `v_rear = v_com - r*0.15532 m`, using the prefab COM-to-rear
  axle geometry. The earlier raw-twist lateral comparison and its steering
  bins in this section were at mismatched reference points and are superseded.
- Against that same rear-axle reference, queued-actuation odometry versus the
  recorded baseline was mixed by run. Fix7 forward/lateral-speed/position
  RMSE changed 0.6819/0.0824/0.4609 to 0.2672/0.0196/0.2538 (SI units); fix8
  changed 0.1486/0.0302/0.1185 to 0.1283/0.0473/0.1077. Thus forward speed
  and position improve in both captures, but lateral speed improves on fix7
  and regresses on fix8. In fix7, the measured-steering `>=0.30 rad` bin
  improves 0.2698 to 0.0370 m/s (19 samples); in fix8 it regresses 0.0332 to
  0.0771 m/s (44 samples). This is not a transferable lateral-accuracy win.
  Yaw rate is gyro passthrough, not an independently predicted quantity;
  relative-yaw RMSE is unchanged within each run (0.0055 rad fix7, 0.0127 rad
  fix8). These paired, previously seen captures are diagnostics, not
  independent validation.
- The unmodified queued-actuation replay's measured whole-packet odometry
  processing time was mean/p95/max 2.03/2.69/4.77 ms on fix7 and
  1.85/2.52/2.87 ms on fix8, below the 25 ms period in these bag replays.
  This is not an online 40 Hz guarantee under simulator/competition load.
- A source-derived acceleration-frame candidate was tested and rejected. It
  added nominal `r*v` and `-r*u` terms to the observer residual because the
  MPC source plant integrates body-velocity derivatives. That assumption was
  wrong for this path: `PhysxOdomRuntime` supplies accelerations produced by
  the native PhysX helper by differentiating world Rigidbody velocity and
  rotating it into the body frame, matching `IMU.cs`; both measured and
  predicted observer accelerations are already inertial body-frame values.
  Adding the nominal transport terms double-counted them and caused rapid
  divergence. The candidate's fix7 u/v/position RMSE was 0.300/0.261/0.245 m
  versus 0.267/0.020/0.254 for queued-actuation only; fix8 was
  0.694/0.360/0.472 versus 0.128/0.047/0.108. The candidate equations were
  reverted. The paired candidate outputs are retained at
  `live_runs/odom_kinematic_accel_replay_20261017_fix7` and `...fix8` as
  evidence for rejecting that change, not as production data.
- Detailed queued-actuation outputs, including full plant state and timing,
  are retained at
  `live_runs/odom_actuation_causal_replay_20261017_fix7_detailed` and
  `...fix8_detailed`; the first fix8 recorder artifact without plant state is
  also retained as a reproducible pose/twist comparison.
- No Unity simulator was launched. The unrelated Payday 2 GPU process and
  existing IPOPT optimizer were left untouched. The queued-input correction
  improves forward speed and position on both captures but has a real
  lateral-speed trade-off, especially on fix8; the attempted acceleration
  frame adjustment was not valid and was rolled back. Online behavior, MPC
  acceptance, and timing under live simulator load remain unverified. Next,
  isolate the fix7/fix8 lateral-velocity disagreement without refitting or
  adding transport terms, then run one supervised practice validation under
  the established procedure when available. Stop on first contact; compare
  same-point odometry `u/v/yaw`, pending/applied command timing, accepted MPC
  horizons, and per-cycle runtimes.

### Combined-slip lateral-innovation gate and held-out replay — 2026-10-17

- The fix7/fix8 residual decomposition showed that suppressing the lateral
  acceleration innovation everywhere is not transferable: it improved the
  high-steer part of fix8 but regressed fix7 lateral speed and fix8 path error.
  A steering-only taper was also rejected because it reduced fix8 high-steer
  lateral-speed RMSE (`0.0929 -> 0.0736 m/s`) while worsening full pre-contact
  position RMSE (`0.1077 -> 0.1224 m`). Neither broad rule is in the current
  implementation.
- Implemented a narrower correction in the production observer. The measured
  `a_y - a_y_model` residual is faded only if all of these are true: actuator
  feedback is valid, reported throttle is at least `0.35`, `|u| >= 0.5 m/s`,
  physical steering is at least `0.4 rad`, and encoder-derived rear-wheel
  overspeed relative to body speed exceeds `1.0`. The gain then tapers linearly
  to zero at Unity's configured `30 degree` steering limit. The gyro-versus-
  model yaw-rate residual is not gated. The default global residual gain remains
  `1.0`; ordinary drive/brake, lower-steer, and non-spinning states therefore
  retain the prior update.
- Added the per-packet effective gain as diagnostics schema v6, index 34, so a
  future run can directly verify when the condition is active. This is only a
  diagnostic-array revision; `UnityPlantState` and competition-topic inputs are
  unchanged.
- The gate selects `0/247` fix7 packets and `19/218` fix8 pre-contact packets.
  The paired fix8 replay gives forward-speed RMSE `0.1283 -> 0.1274 m/s`,
  rear-axle lateral-speed RMSE `0.0473 -> 0.0432 m/s` (8.7% lower), lateral
  p95 `0.0892 -> 0.0873 m/s`, and aligned position RMSE `0.1077 -> 0.1108 m`.
  Within the selected 19-packet subset, lateral-speed RMSE falls
  `0.1028 -> 0.0774 m/s`; outside it, the score is effectively unchanged.
  This is a localized lateral-speed gain with a 3.1 mm position regression,
  not an unqualified whole-run win.
- Replayed two additional existing captures through a fresh current production
  odometry process. On fix5, compare only the 54 pre-contact samples present in
  both recorded and replayed odometry (the capture reaches collision count 2
  later): `u` RMSE `0.1139 -> 0.0916 m/s`, rear-axle `v` `0.0202 ->
  0.0304 m/s`, aligned position `0.0399 -> 0.0329 m`, and yaw `0.0117 rad`
  unchanged. The gate activates on one shared packet. On fix6's 135 pre-contact
  shared samples, `u` RMSE is `0.1268 -> 0.0198 m/s`, `v` is `0.0133 ->
  0.0185 m/s`, aligned position is `0.0839 -> 0.0247 m`, and yaw remains
  `0.0024 rad`; the gate never activates there. These fix5/fix6 gains primarily
  reflect the already implemented queued-actuation correction, not the new
  lateral gate. Treat the modest `v` regressions as real trade-offs.
- All three comparisons use pre-contact intervals and same-stamp Unity bridge
  odometry. Rear-axle lateral reference is
  `v_com - r * 0.15532 m`; pose is rigidly aligned at the first common sample
  and compares against the corresponding rear-axle point. The yaw-rate output
  is gyro passthrough, so its near-zero residual is not a learned/predicted yaw
  improvement. These are replays of a small set of prior captures, not new
  independent simulator runs or evidence of collision-free driving.
- Release localization rebuild passed in `17.3 s`. Focused package tests pass:
  `test_sensor_packet_assembler` (3), `test_odometry_observer` (10), and
  `test_physx_odom_actuation` (4). `git diff --check` and the runtime topic
  policy check pass. No Unity process was launched and no live 40 Hz or MPC
  acceptance result was collected.
- Retained the clean fix5/fix6 final-gate replays and clean fix7/fix8 baseline
  plus final-gate replays under `live_runs/odom_*_replay_20261017_*`. Removed
  the first mixed-topic fix5 recording and the discarded global-zero,
  steering-only, and incomplete early recordings after recording their
  conclusions above. No source practice captures were deleted.

**Current judgment / next action:** keep the narrow gate as a production
candidate because it improves the targeted fix8 lateral-slip subset without
changing normal regimes, but do not call it fully validated. Before making a
broader odometry claim, run the established supervised practice-stack path with
this build and compare at source stamps; abort on first collision. Capture v6
gain, state errors, command delay, MPC acceptance, and process times together.
If the gate is rarely entered online or position regressions repeat, remove or
retune only this gate; do not compensate with MPC weights. The larger remaining
problem is overall model accuracy outside this narrow high-throttle/high-steer
slip condition, especially lateral velocity; that needs a separately diagnosed
correction.

### State necessity and calculation audit — supersedes the earlier 25-state table

The earlier state audit above described an intermediate `n_x=25` version and
said to keep longitudinal acceleration as a QP state because it had a cost and
bound. That conclusion was superseded after tracing the transition: previous
acceleration is not read by the plant step, and the acceleration tracking term
was absent from nonlinear candidate ranking. The current optimizer has
`N=30`, `n_x=24`, `n_u=2`; this section is the current state/cost decision.

The key distinction is that an MPC state is a quantity needed to make the
plant Markov and predict future motion. A state cost is a preference in the
objective. They are not the same list. Roll/pitch need not be driven toward
zero to be represented in the model.

| RTI state(s) | Objective use | Why it remains in the prediction state |
|---|---|---|
| `e_y`, `e_psi` | Direct corridor/path tracking weights | Define path-relative motion, cost, and hard corridor constraints. They are planner coordinates, not Unity rigid-body variables. |
| `u` | Direct speed tracking; overspeed uses its separate weight | Sets wheel-contact longitudinal velocity, slip, force, and distance advanced. |
| `v` | Direct lateral-velocity weight | Changes every wheel's lateral contact velocity/slip and Frenet motion. |
| `r` | Direct yaw-rate weight | Changes wheel contact-point velocities, tire slip, yaw moment, and path heading. |
| steering FIFO slots 1–2 | Slot 2 tracks the steering reference; command/slew terms also use the queue | Slot 1 is applied to the next plant interval; slot 2 is shifted into slot 1. Both are required to reproduce the two-packet delay. |
| physical steering angle | Direct actuator-angle tracking | Source steering slew and Ackermann wheel angles use the actual angle, not the requested target. |
| throttle FIFO slots 1–2 | Command and command-change costs are applied to the issued input/queue | Slot 1 selects the actual drive/brake branch; slot 2 is its deterministic successor. Removing them by deletion would apply commands too early. |
| four wheel angular speeds | No zero-speed target; the soft source-slip cost penalizes only force-loss slip excess | Each wheel's surface velocity determines longitudinal slip, tire force, and wheel-torque evolution. Four independent wheels are required by the source equations. |
| body height and heave rate | No direct tracking weight | Height changes all suspension jounces; heave rate participates in sprung-body damping and therefore affects subsequent loads. |
| roll / roll rate | No direct tracking weight | Roll changes left/right wheel geometry and spring loads; its rate propagates the transient roll state. These feed wheel normal loads and tire forces. |
| pitch / pitch rate | No direct tracking weight | Pitch changes front/rear jounce and suspension loads; its rate propagates transient load transfer under throttle/braking. |
| four suspension-jounce history values | No direct tracking weight | The source plant computes damper velocity from each wheel's jounce difference across substeps. The four history values feed four wheel-specific support forces. |

Direct state tracking weights in the active candidate are only for path
errors, `u/v/r`, and steering command/angle. Wheel omega affects the objective
through the source-curve slip-excess penalty, not by trying to drive wheel
speed to zero. Roll, pitch, heave, their rates, jounce, and delay bookkeeping
have no independent zero target. Adding a roll/pitch penalty would bias the
optimizer against the physically predicted load transfer; it is not justified
for this racing objective. If a future objective needs rollover protection,
that should be a validated physical bound or stability criterion, not a
generic zero-roll comfort cost.

#### Unity-source dependency trace

In `f1tenth_mpc/src/unity_source_plant.cpp`:

- `suspension_jounce()` uses body height, roll, pitch, and each wheel's fixed
  geometry to calculate four separate jounces.
- `suspension_force()` uses roll/pitch projection, sprung mass, spring
  stiffness, and the per-wheel jounce finite difference (damper rate).
- Those support forces generate four normal loads; the wheel-specific loads
  scale the tire curve forces.
- Wheel omega and body `u/v/r` jointly determine wheel-surface mismatch and
  lateral slip. Per-wheel tire forces then update body translation, yaw, roll,
  pitch, and wheel angular acceleration.
- Body height/rates and roll/pitch rates are integrated each plant step.
  Jounce history is carried so the next substep has the same damper history.
- Actual steering and delayed throttle/steering select the wheel angles and
  physical drive/brake input that enter those equations.

Thus roll/pitch are small-angle states in parts of this capture, but not
decorative states. Around zero angle, roll changes each side's wheel height by
approximately `wheel_y * roll`; the observed extreme `|roll|=0.0259 rad`
corresponds to about 3 mm per side (roughly 6 mm left-to-right). Pitch changes
front/rear wheel height by approximately `wheel_x * pitch`; the observed
maximum `|pitch|=0.0480 rad` corresponds to roughly 8 mm front-to-rear. These
are meaningful relative to the 25 mm suspension travel and change the spring
load entering the tire model. Rates matter because the simulation includes
transient suspension damping, not a static equilibrium-only model.

#### Captured evidence and limits

The one-lap state-24 practice capture at
`live_runs/mpc_state24_speed500_20261017/run` contains 2,547 Unity GT odometry
and IMU messages and 2,549 odometry plant-state messages. Decoded roll/pitch
from both GT odometry orientation and IMU orientation agree at the recorded
precision. Their observed ranges are roll `[-0.02588, 0.00738] rad` (p05
`-0.00631`, p95 `0.00147`), pitch `[0.00944, 0.04799] rad` (p05 `0.02224`,
p95 `0.03871`), roll rate `[-0.2235, 0.3413] rad/s`, and pitch rate
`[-0.1853, 0.4302] rad/s`. The `UnityPlantState` sensor-corrected mask marks
roll, roll rate, pitch, and pitch rate as corrected in this run. Therefore
these are not merely the plant's unconstrained guesses.

This was a low-speed screen, not a high-load validation: GT speed p95 was
`1.25 m/s`, maximum `2.11 m/s`. The bag's wheel omega, jounce, load, and slip
arrays are source-plant estimates/outputs, not independent per-wheel Unity
ground truth. They are useful for checking activity and consistency, but not
for claiming the tire/suspension model is accurate at racing load. The run
completed one lap without collision, but its lap time was about 106 s and it
does not validate race performance.

#### Redundancy removed and calculations retained

- Removed longitudinal acceleration from the RTI vector, array
  serialization/deserialization, finite-difference Jacobian coordinates,
  QP bounds, and state-tracking cost. The configuration parameter and test
  expectations for that QP weight were also removed. A regression test changes
  the cached acceleration by `-18` versus `+21 m/s^2` and confirms identical
  plant transition, state, and wheel outputs; another check confirms scalar
  candidate cost is independent of that cache.
- Acceleration remains a derived transition/ROS output and is used as the
  measured initial derivative when aligning the reachable speed-reference
  profile. The plant must still calculate longitudinal acceleration to update
  `u`; retaining that output does not make it a recursive state or a weighted
  state.
- Removed the duplicate `MpcStageResult_t.body_accel_mps2` copy because it had
  no consumers; the derived acceleration remains in the named plant output.
- Kept all tire-slip, per-wheel load/force, wheel-spin, suspension, and
  sprung-body calculations that feed the nonlinear transition. Keeping their
  state cost at zero does not make those calculations unnecessary.
- Kept the four command-queue coordinates for now. They are deterministically
  shifted, but exact removal from the QP needs a delayed-input formulation
  that fixes the two pending applied commands, shifts the published control
  index, and preserves command-slew costs/warm starts. Simply dropping them
  would be a two-packet timing bug. This is a separate possible QP-condensing
  change, not a physical-state deletion.
- The per-wheel maximum-slip aggregation and copies are diagnostic outputs,
  not forces. A nondiagnostic plant path could skip that aggregation, but it
  has not been changed here; first profile its cost and preserve the current
  diagnostics contract. The actual slip, tire curves, normal loads, forces,
  and state integration cannot be skipped while retaining this plant.

#### Verification and runtime implications

The targeted Humble release build of `f1tenth_mpc` passed. The focused
`test_vehicle_model_source_rollout` binary passes **27/27 tests**, including
the new acceleration-independence test. `git diff --check` is clean. The
existing one-lap run produced 2,469 MPC cycles: solve time p50/p95/p99/max was
`15.47/18.08/18.40/23.94 ms`, with three cycles over 20 ms and none over
25 ms. Control-cycle spacing was less healthy (p95 `26.77 ms`, maximum
`51.67 ms`), so this is not proof of a reliable 40 Hz end-to-end loop.

Going from 25 to 24 RTI coordinates removes one central-difference state
column: in the current 30-stage Jacobian path this avoids 60 perturbed
25 ms plant transitions per solve (about 1,500 internal 1 ms source steps).
It also reduces dense cubic state-matrix work by about 11.8% for the affected
matrix operations. These are operation-count estimates; no same-build,
same-weights 25-vs-24 timing A/B was run, so no wall-clock gain is claimed.

**Decision:** retain the 24-state plant, apply no direct roll/pitch/heave
weights, and do not remove any more physical coordinates based on size alone.
The next safe compute investigation is exact queue condensing or analytic
source-plant sensitivities, each benchmarked against unchanged rollout output
and the 40 Hz deadline. Current model accuracy and the very low speed in the
practice screen remain separate unresolved issues.

### Direct answer: exact queue condensation and measured rollout error

The four steering/throttle delay-queue coordinates are mathematically
condensible without changing the physical transition. For either command
channel, write the queue as `(q1, q2)` and let `a_k` be the command applied to
the plant at stage `k`. The existing shift is

```text
a_0 = q1_0                 a_1 = q2_0
a_k = U_(k-2), k >= 2
q1_(k+1) = q2_k             q2_(k+1) = U_k
```

Thus the two initial applied commands are fixed by the pending queue, while
the later applied commands are decisions shifted by two stages. In principle
the optimizer vector can shrink from 24 to 20 coordinates (two queues for
steering and two for throttle), with the two-packet delay preserved exactly.
This has **not** been implemented. It is not a safe four-coordinate deletion:
the published command must map to applied-input index 2; steering/throttle
command and slew penalties become shifted and adjacent-stage input costs;
warm starts and terminal stages must be shifted; and the last two decisions
need correct tail treatment. The current Riccati stage cost does not directly
represent an arbitrary neighboring-stage control-control Hessian. Either the
solver must support that coupling or some command-memory coordinates must
remain. Therefore 20 is the exact lower dimension for queue-free physical
state representation, not a measured or yet-safe production dimension.

The deployed one-lap capture's accepted MPC forecasts were aligned to future
simulator ground truth. The table below reports forecast-versus-realized
RMSE; speed is `m/s`, yaw rate `rad/s`, position `m`, and yaw `rad`.

| Forecast horizon | Forward speed `u` | Lateral speed `v` | Yaw rate `r` | XY position | Yaw |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 ms  | 0.095 | 0.050 | 0.115 | 0.065 | 0.055 |
| 125 ms | 0.298 | 0.051 | 0.327 | 0.065 | 0.057 |
| 250 ms | 0.697 | 0.052 | 0.543 | 0.086 | 0.082 |
| 500 ms | 1.521 | 0.054 | 0.749 | 0.325 | 0.183 |
| 750 ms | 2.188 | 0.055 | 0.750 | 0.760 | 0.295 |

At 750 ms the absolute-error p95 was `3.438 m/s` for forward speed,
`0.120 m/s` lateral speed, `1.557 rad/s` yaw rate, `1.203 m` XY position,
and `0.637 rad` yaw. At 25 ms the corresponding p95 values were `0.219 m/s`,
`0.109 m/s`, `0.232 rad/s`, `0.084 m`, and `0.098 rad`. The state estimate at
the forecast origin itself differed from GT by RMSE `0.084 m/s` in `u`,
`0.053 m/s` in `v`, and `0.126 rad/s` in `r`.

**Interpretation/caveat:** these are the live MPC's receding-horizon forecasts
compared to what the simulator later did, not a pure open-loop plant A/B with
one frozen command sequence. The controller replans and issues new commands
at subsequent cycles, and the forecast starts with estimated state, so the
error combines state estimate, reference/controller changes, timing, and
plant-model error. The run was also a slow one-lap screen (GT speed p95
`1.25 m/s`, maximum `2.11 m/s`), not a high-speed validation. It establishes
that the currently deployed long-horizon predictions are not accurate enough
to trust for fast racing; it does **not** isolate how much error belongs to
the source-derived plant itself. No “near-perfect” or bit-exact PhysX accuracy
claim is supported by this run.

The plant uses a 1 ms integration step and implements the source-derived
vehicle/actuator equations described above: delayed steering/throttle,
steering rate and Ackermann geometry, torque/brake and wheel spin, longitudinal
and lateral slip, combined-slip tire forces, normal-load/suspension effects,
and coupled body/yaw dynamics. This is a reduced source-grounded model, not a
copy of Unity's private WheelCollider/PhysX contact solver. The native wheel
ground-contact/raycast evolution, private contact/constraint impulse solve,
and all hidden per-wheel solver state are not replicated or exposed as true
runtime observations. Estimated wheel/load/slip state must not be described
as Unity ground truth. Therefore the model reflects a meaningful subset of
the documented/source-visible PhysX behavior, but not “every interaction” or
a 1:1 PhysX replay.

**Next measurement needed for plant-only rollout accuracy:** replay a captured
command/actuator-feedback sequence from a stated initial estimated plant
state, run the 1 ms model recursively without feeding it future sensors or
GT, and compare `u/v/r`, wheel speeds, and integrated pose against aligned GT
at 25 ms through 750 ms. Report both reset-per-window short-step error and
whole-window recursive error, separated by speed/steering/throttle regime.
That isolates model error more cleanly than the live receding-forecast table;
it must not be conflated with the live closed-loop metrics above.

### Source identity and rollout-error clarification — 2026-10-17

- Rechecked the public `Tinker-Twins/AutoDRIVE` `AutoDRIVE-Simulator` source
  branch (commit `75b7df5216524a3c6b3150d698139190c6c44f8f`) against the
  project files checked into this workspace. The active RoboRacer scene,
  F1TENTH prefab, `DynamicsManager`, `TimeManager`, and Unity version file are
  byte-identical. `VehicleController.cs` is identical apart from this
  workspace's development-only capture call. This confirms that the checked-in
  scene/controller parameters are not a guessed vehicle variant.
- This does **not** provide the WheelCollider solver implementation. Unity's
  public reference source declares WheelCollider properties and queries as
  native externs; its `WheelHit` API exposes contact force and forward/side
  slip after the native solve. Unity documents that WheelCollider friction is
  computed separately from the rest of the physics engine using a slip-based
  model. The Unity project does not contain the corresponding native contact,
  suspension, tire-force, or constraint-solver implementation. Therefore a
  source-faithful copy of the visible controller/settings is possible, but an
  exact software copy of the entire WheelCollider/PhysX transition is not
  available from these project files.
- Clarification of the conspicuous 750 ms error: `u=2.188 m/s` and
  `r=0.750 rad/s` are from live MPC receding-horizon forecasts compared with
  later realized simulator states while the controller continued replanning.
  They are strong evidence that the deployed long-horizon forecast is not
  trustworthy, but they are **not** a plant-only fixed-command rollout error.
- The existing command-only recursive replay of the available 63.7 s capture
  is the closer plant-only test: full-run RMSE was `u=0.0821 m/s`,
  `v=0.0161 m/s`, and `r=0.1018 rad/s`; for `|steering| >= 0.2 rad`,
  `u/v/r` RMSE was `0.1117 m/s / 0.0328 m/s / 0.2109 rad/s`. For samples
  with GT speed `>=1.5 m/s`, it was `0.0812 / 0.0459 / 0.2885` in the same
  units. That capture only reached `2.105 m/s`, so it is not racing-speed
  evidence and does not establish the requested near-zero error.
- The four optimizer queue coordinates remain present (`MPC_MODEL_NX=24`).
  They encode the two known pending command samples for steering and throttle;
  they are deterministic, but deleting them from the state vector without
  changing the QP would make the plant apply new commands two samples early.
  Exact condensation is still an implementation task: use the queued commands
  as fixed first-stage applied inputs, shift optimized decisions by two
  stages, map the published output to decision index 2, and preserve adjacent
  command-change and terminal costs/warm starts. No such refactor or timing
  benefit is claimed yet.
- No simulator was launched and no bag was modified in this source/error
  audit. The active IPOPT process (`run_ipopt.py`, PID 714501 at audit time)
  was left untouched.

**Current conclusion:** the 750 ms live forecast is unacceptable for racing,
and the recursive model has nontrivial high-steering yaw error even in the
slow available capture. The full Unity project branch confirms the source
vehicle configuration, but does not provide the native WheelCollider code
needed for a literal 1:1 port. The four delay states are candidates for exact
QP condensation, not redundant data that can simply be erased. Neither the
accuracy objective nor the queue-condensing objective is complete.

### High-speed odometry A/B and encoder/actuation source check — 2026-10-17

- Re-evaluated the two existing full-speed practice captures and the current
  PhysX-odometry replay; no new simulator run was needed to answer this A/B.
  The old production odometry in the 3-lap captures scored forward-speed RMSE
  `0.0771 m/s` and `0.1455 m/s`, lateral-speed RMSE `0.0119` and
  `0.0124 m/s`, with relative-pose endpoint errors `0.217` and `0.258 m`.
  Their best laps were `5.8608` and `5.8638 s`, with 3 laps and zero
  collisions in each capture; maximum GT speed was `8.093` and `8.078 m/s`.
- Replaying the second capture through the current v6 PhysX odometry produced
  forward-speed RMSE `0.4409 m/s`, lateral-speed RMSE `0.0222 m/s`, and
  relative-pose endpoint error `1.394 m`. At zero-throttle/full-brake samples
  (`n=117`), forward-speed RMSE was `1.001 m/s` with bias `-0.764 m/s`.
  This is a regression versus the old production odometry on that capture, not
  an improvement. Do not promote the current speed/pose estimate on this
  evidence.
- Direct source inspection of `WheelEncoder.cs` and the active F1TENTH prefab
  resolves the encoder semantics: `WheelEncoder.FixedUpdate()` reads
  `Wheel.rpm`, integrates revolutions, then quantizes total angle with
  `PPR=16`, `GearRatio=120`. The encoder is therefore a quantized wheel-rotation
  measurement generated by the simulated WheelCollider, not a direct copy of
  throttle percentage and not chassis speed. Torque/brake input affects
  the wheel, but tire contact also determines its RPM. At zero throttle the
  controller applies the source-configured `428 N m` brake torque per wheel;
  the wheel can stop while the body continues sliding, so encoder-derived
  peripheral speed must not replace body `u/v`.
- The two-packet throttle queue was checked against the recorded Unity
  `CurrentThrottle` feedback: feedback matched the command two 25 ms samples
  earlier with RMSE `0.00051` normalized throttle on the replay. This capture
  does not support changing the queue delay; the current failure is not
  explained by an arbitrary one-packet shift.
- The current practice experiment remained stationary: no MPC horizon was
  accepted, and zero-throttle fallback held the car. Its residual recovery
  threshold change had no benefit and has been reverted to `0.5`; the exact
  corridor feasibility gate was not relaxed.
- No production dynamics were changed in this check. The source-visible torque
  semantics and queue delay agree with the implementation; the large v6
  braking/sliding error remains unexplained by those inputs. The next valid
  correction must come from a demonstrated difference in the native
  WheelCollider/PhysX force or contact-state evolution, then improve the same
  held-out full-speed replay before any production promotion.
- Follow-up fixed-grid encoder/body comparison resolves the apparent signal
  ambiguity: across 804 adjacent 25 ms Unity truth/encoder epochs, the 115
  zero-throttle intervals had mean body-forward speed `4.898 m/s` but mean
  rear-wheel peripheral speed only `0.022 m/s`; median maximum-side wheel/body
  mismatch was `5.196 m/s` (p95 `7.457 m/s`). On 689 positive-throttle
  intervals, mean wheel peripheral speed was `4.633 m/s` versus body `u`
  `4.334 m/s`, with mean signed mismatch `+0.300 m/s`. Wheel speed is therefore
  strongly driven by the torque/brake mode, but it is not a valid substitute
  for body velocity: the capture directly shows locked/near-locked wheels while
  the chassis continues moving under full braking. This is a descriptive
  rear-wheel kinematic proxy (`encoder angle delta / 25 ms` versus
  `u +/- r*0.118 m`), not an independent measurement of Unity's tire-contact
  slip.

### Encoder sample-phase correction and ROS replay — 2026-10-17

- Confirmed the user's actuator interpretation against `VehicleController.cs`:
  positive throttle is a percentage of the configured 428 N m motor torque
  (CAWD divides it among four wheels), while exactly zero throttle removes
  drive torque and applies 428 N m brake torque to each wheel. The encoder
  reports the resulting `Wheel.rpm`; it is a wheel response to that torque/
  brake plus contact dynamics, not a throttle-valued signal.
- Found and corrected a unit error in the new encoder-phase predictor. Unity's
  `WheelEncoder.FixedUpdate()` does RPM -> revolutions/s -> accumulated
  revolutions -> integer ticks -> radians. The first native implementation
  stopped at revolutions and incorrectly labelled that as radians (missing
  `2*pi`). It was caught by the full ROS replay, fixed to mirror the source's
  `PPR=16`, `GearRatio=120` tick quantization, and the erroneous replay artifact
  was discarded. A focused regression test now checks the 84 rad/s, 1 ms
  sample gives the Unity-equivalent 25-tick angle (`0.08181 rad`), including
  reverse sign/reset.
- Rebuilt `f1tenth_localization` successfully and passed its focused observer
  and encoder tests (12 observer tests plus 2 new encoder tests). Replayed the
  full 805-packet/three-lap source capture through the production ROS node and
  recorded the corrected candidate in
  `live_runs/odom_replay_historical_20261017/physx_odom_v7_encoder_phase_fixed`.
  All 805 GT and odometry epochs matched; the playback completed with zero
  collisions. The node's full packet processing time was median `1.74 ms`,
  p95 `2.82 ms`, maximum `9.66 ms`; no packet exceeded 25 ms in this replay.
- The corrected sensor-phase prediction matched measured angle increments on
  drive-to-brake transitions with mean residual `-0.0022 rad` left and
  `-0.0010 rad` right (`n=10` each); full-run residual RMSE was `0.079–0.080
  rad`, so this is not an exact all-regime encoder replay. Critically, this
  implementation change did **not** materially improve body odometry: replay
  forward-speed RMSE was `0.4410 m/s`, lateral-speed RMSE `0.0222 m/s`, and
  aligned endpoint position error `1.394 m`, essentially unchanged from the
  prior v6 replay (`0.4409 m/s`, `0.0222 m/s`, `1.394 m`). Zero-throttle
  forward-speed RMSE remained `1.0015 m/s` with bias `-0.7642 m/s`.
- Thus the encoder's causal RPM/torque semantics and its pre-solve sampling
  phase are now represented correctly, but this narrow correction is not an
  odometry accuracy improvement and does not resolve the measured braking
  error. Do not claim production odometry is accurate or promote this result
  as a body-state gain. Continue tracing the remaining speed error from the
  1 ms physical acceleration/force sequence; do not tune away the source
  torque or brake law.

### PhysX hidden-state and MPC-rollout audit — 2026-10-18

- Rechecked the active scene override and `VehicleController.Drive()` rather
  than inferring torque from the prefab default. In the active scene,
  `MotorTorque=428 N m`, `driveType=CAWD`, and `brakeType=CAWB`: for nonzero
  normalized throttle `q`, each wheel receives `428*q/4 N m` drive torque;
  exactly `q=0` sends zero drive torque and `428 N m` brake torque to each
  wheel. The source plant and PhysX odometry implement those branches. This is
  not a reason to change the torque mapping.
- `WheelEncoder.FixedUpdate()` samples `Wheel.rpm` before the following Unity
  physics step, integrates it at the 1 ms fixed cadence, then quantizes the
  accumulated angle to 1,920 ticks/revolution. Thus wheel rotation responds to
  the commanded torque/brake **and** tire contact; it is not a direct encoding
  of throttle. Odom must continue using encoder counts to correct wheel-spin
  state, never substitute peripheral wheel speed for chassis `u/v`.
- Pinned PhysX 4.1 source identifies persistent per-wheel internals erased by
  `PxVehicleNoDrive::setToRestState()`: both wheel-speed arrays, wheel angle,
  low-forward/low-side sticky-friction timers, previous jounce, and cached
  suspension-query data. `PxVehicleWheelsDynData::copy()` does not copy jounce
  or the vehicle-constraint/scene solver state. These facts explain why a cold
  reset is not equivalent to continuing a mature vehicle simulation, but
  source thresholds also rule out the sticky timer as the cause of the tested
  fast-window errors: sticky forward behavior needs tire-point speed below
  0.2 m/s or a full second accumulated there; the audited windows start above
  3.1 m/s. The native process performs fresh suspension raycasts at every
  1 ms update, so invalidated cached hit planes are refreshed before use.
- Therefore, adding timer or cache coordinates to MPC is not justified by the
  current high-speed error evidence. The remaining material rollout gap is
  structural: `f1tenth_mpc/src/unity_source_plant.cpp` is a reduced force/state
  translation. It analytically reconstructs suspension jounce and integrates
  chassis velocity/angular state itself; it does **not** call
  `PxVehicleUpdates()` followed by `PxScene::simulate()/fetchResults()` as the
  odometry plant does, and it cannot reproduce Unity's private WheelCollider
  implementation. Wheel torque/brake and several vehicle equations are
  source-derived, but this is not a 1:1 rollout.
- Previously measured command-only native PhysX horizons, when the native
  process was first primed with the capture's past and then left persistent,
  were much closer to truth than the reduced MPC plant on the eight selected
  windows (750 ms macro RMSE: `u=0.00385 m/s`, `v=0.00029 m/s`,
  `r=0.00240 rad/s`, relative XY `0.00621 m`, yaw change `0.00185 rad`). A
  fresh current-state reset had 25 ms macro RMSE `u=0.1132 m/s` and
  `r=0.0510 rad/s`, falling by 750 ms to `u=0.0789 m/s`,
  `r=0.0207 rad/s`, XY `0.0147 m`. These are offline replay results, not live
  simulator validation; the error is concentrated at rollout initialization
  and is consistent with omitted persistent solver/contact context, not a
  wrong zero-throttle torque branch.
- Next correction target is the MPC transition implementation itself: compare
  the reduced model against the native `PxVehicleUpdates` + scene solve on the
  same held-out windows, isolate which force/integration term first diverges,
  then replace that demonstrated approximation with a shared native/source
  implementation. Preserve the existing torque branch and state dimensions
  unless measured error proves either wrong. Do not promote an internal-state
  field or a model change based only on source resemblance; require the same
  held-out state-by-state rollout metrics and an independent capture. No source
  dynamics were changed in this source audit, and no live simulator was run.
- Re-ran the focused existing actuation and encoder checks: both
  `test_physx_odom_actuation` and `test_unity_wheel_encoder` pass. Re-evaluated
  the retained 805-epoch raw bag as a provenance check; it is explicitly the
  old `sensor_corrected_physx_odom_v6_pose_innovation_raw_accel_lateral_gain`
  baseline, not the current candidate. Its zero-throttle subset remains
  `1.0014 m/s` forward-speed RMSE (`n=117`), so it must not be used to describe
  the newer v8 candidate or claim current production accuracy.

### High-speed bag selection for the in-process odometry work — 2026-10-18

- Inspected and re-scored the two retained runs in
  `live_runs/historical_mpc_fullspeed_20261017/` with
  `tools/evaluate_mpc_bag.py`. Both contain 25 ms-aligned Unity truth and
  production `/odom`, encoder, IMU, steering/throttle command and feedback,
  collision-count, and lap-count channels. They are usable racing-speed
  validation captures, not stationary or low-speed substitutes.
- `run/`: 811 matched epochs over 20.26 s, three completed laps, zero
  collisions, best lap `5.8608 s`; Unity truth speed p95 `7.052 m/s`, max
  `8.093 m/s`. Recorded legacy production odometry (`deterministic_odometry_v5`)
  has forward-velocity RMSE `0.0771 m/s`, lateral-velocity RMSE `0.0119 m/s`,
  and aligned relative-pose endpoint error `0.217 m`.
- `run_1/`: 805 matched epochs over 20.11 s, three completed laps, zero
  collisions, best lap `5.8638 s`; truth speed p95 `6.962 m/s`, max
  `8.078 m/s`. Its recorded v5 odometry has forward-velocity RMSE
  `0.1455 m/s`, lateral-velocity RMSE `0.0124 m/s`, and endpoint position
  error `0.258 m`.
- Both have essentially 40 Hz data: odometry mean interval `25.01 ms`, p95
  `25.89 ms` and `26.02 ms`. This is recorded source-time cadence; it does not
  prove the ROS node's current computation or bridge delivery is deadline-safe.
- The first run is the cleaner primary baseline; the second is an independent
  complete-run check with materially worse forward error and is retained as
  the transfer/validation run. Neither bag contains `UnityPlantState`, and
  there is no independent per-wheel simulator-truth angular speed channel.
  Rear encoder speed can be compared as a measured wheel-rotation signal, not
  mislabeled as wheel-speed ground truth. Simulator truth is to remain an
  evaluator-only input, never a production subscription.
- These captures already exercise the prior high-speed controller and are
  sufficient to test the odometry change at up to `8.1 m/s`; therefore no
  previous-commit MPC checkout or simulator run is required for this odometry
  phase. The old controller is only a data-generation fallback if later
  analysis identifies a specific unsupported regime. No MPC source was
  changed and no simulator was launched for this bag audit.
- Replayed each capture at 1x through the just-built production ROS odometry
  node in separate ROS domains, with only encoder, IMU, actuator feedback and
  actuator-command topics as estimator inputs. Unity truth odometry was
  present only in the isolated replay domain for recording/scoring; the
  competition topic-policy verifier passes and the node has no truth
  subscription. Result bags are retained at
  `live_runs/odom_prod_replay_highspeed_20261018/` and
  `live_runs/odom_prod_replay_highspeed_20261018_run1/`.
- The current production implementation achieved `u` RMSE `0.0041 m/s`,
  rear-axle `v` RMSE `0.0003 m/s`, and aligned position endpoint error `0.003 m`
  on `run/`; on `run_1/` it achieved `u` RMSE `0.0055 m/s`, `v` RMSE
  `0.0006 m/s`, and endpoint error `0.012 m`. Gyro yaw-rate output is sensor
  passthrough, so its zero residual is not a model-prediction result. The
  diagnostics label the active path `sensor_corrected_physx_odom_v7`.
- Against the recorded v5 odometry in the same captures, these are substantial
  replay improvements (v5 `u` RMSE `0.0771/0.1455 m/s`, `v` RMSE
  `0.0119/0.0124 m/s`, endpoint `0.217/0.258 m`). This verifies that the
  current estimator is accurate on these two held-out high-speed runs, not
  that it is an in-process source-only port: production still launches the
  separate `physx_native_vehicle_step` process linked to the public PhysX
  libraries. That process boundary remains explicitly unresolved. The next
  work must compare an in-process, source-derived candidate against these
  same captures and retain this result as the accuracy target; do not describe
  the existing external-process implementation as the requested port.
