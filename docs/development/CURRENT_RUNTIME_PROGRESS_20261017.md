# Current runtime and rewrite status — 2026-10-17

> **Latest MPC runtime update — 2026-10-21:** the supervised practice run,
> exact measurements, runtime input audit, bag-decoder correction, and current
> failure-policy behavior are documented in
> [MPC_RUNTIME_FAILURE_AND_BAG_DECODE_20261021.md](MPC_RUNTIME_FAILURE_AND_BAG_DECODE_20261021.md).
> Across the three capped practice runs in that report, the latest published
> exactly validated current/prior actions in 408/409 solve cycles, including
> 16 real prior-plan reuses; maximum end-to-end control time was 15.771 ms
> (<20 ms). It completed no lap and stopped after a genuine corridor rejection
> that the remaining prior plan also could not pass. This is not high-speed or
> racing validation.

> The dated progress entries below are historical snapshots. The 2026-10-21
> report linked above is the latest runtime status and supersedes older
> statements about schema versions, pending verification, and whether the MPC
> has been run.

## Maintained stack

> **Live status update — 2026-10-18:** this work is focused on production
> odometry and its source plant; no controller was run. The in-process plant
> remains a source-derived approximation, not a 1:1 port of Unity's private
> WheelCollider solver. Its low-speed constraint treatment now uses six
> sweeps, supported by two independent 1 kHz captures; the corrected one-step
> yaw-rate maximum fell from `0.0556/0.0669` to
> `0.0030/0.0040 rad/s`. This also removes the corresponding one-step forward
> speed error. The plant ABI has 18 states and `UnityPlantState` schema 9.
> Production odometry and targeted plant tests compile/pass. This is capture
> replay evidence, not yet a full-lap odometry claim.

The workspace now retains one ROS runtime path: the 40 Hz bridge, source-
derived PhysX odometry, EKF/AMCL, the RTI MPC, and a direct actuator interface.
The MPC output is `DriveCommand(steering_angle_rad, throttle_normalized)`;
raceline speed remains a tracking reference, not an actuator command. The
MPC's RTI state carries the typed plant channels and pending throttle/steering
history. Production odometry advances the in-process source-derived plant in
`f1tenth_localization/src/unity_source_odom_runtime.cpp` and
`f1tenth_mpc/src/unity_source_plant.cpp`; it does not launch a PhysX subprocess
or link the PhysX SDK. The old process implementation was not in the active
build and has been removed.

The active track assets are the competition map/raceline pair and the practice
map/raceline pair under `f1tenth_planning/`. Current odometry and rollout
evidence is restricted to the retained practice captures listed in the
practice-only evidence ledger.

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
  implementation and its replay harness, one-off learning and racing
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
  Unity fixed timestep is 1 ms. **Correction entered 2026-10-20:** the shipped
  player does not switch vehicle substeps at 5 m/s; exact-build disassembly
  shows `setSubStepCount(128.0f, 5, 1)`. The 5 m/s source constant is for
  low-speed wheel-pose-angle blending, not the vehicle substep schedule.
  PhysX's `+0.1 m/s`
  longitudinal-slip denominator applies
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

> **Implementation correction (2026-10-18):** the native-process and
> PhysX-SDK results in this historical audit describe an earlier prototype.
> That subprocess was never part of the active CMake runtime and its source
> files have now been removed. Current production odometry and MPC both use
> the in-process `UnitySourcePlant` translation; it does not call `PxVehicle`
> or Unity's private `WheelCollider` solver. Treat earlier native-process
> rollout numbers as historical experiments, not current production accuracy.

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
- Pinned public PhysX 4.1 source identifies persistent per-wheel internals erased by
  `PxVehicleNoDrive::setToRestState()`: both wheel-speed arrays, wheel angle,
  low-forward/low-side sticky-friction timers, previous jounce, and cached
  suspension-query data. `PxVehicleWheelsDynData::copy()` does not copy jounce
  or the vehicle-constraint/scene solver state. These facts explain why a cold
  reset is not equivalent to continuing a mature simulation of that public
  vehicle SDK. Those timers are not evidence that Unity WheelCollider uses
  the same behavior. In the public PhysX branch, sticky forward behavior needs
  tire-point speed below
  0.2 m/s or a full second accumulated there; the audited windows start above
  3.1 m/s. The native process performs fresh suspension raycasts at every
  1 ms update, so invalidated cached hit planes are refreshed before use.
- Therefore, public PhysX timer/cache coordinates are not justified in the
  optimizer without evidence that Unity uses those same states. The shared
  production odometry/MPC plant is a reduced force/state translation: it
  analytically reconstructs suspension jounce and integrates chassis
  velocity/angular state itself; it neither calls `PxVehicleUpdates()` nor
  reproduces Unity's private WheelCollider implementation. Wheel torque/brake
  and several vehicle equations are source-derived, but this is not a 1:1
  rollout.
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
  `0.0119/0.0124 m/s`, endpoint `0.217/0.258 m`). These are replay results on
  two held-out high-speed runs, not a general accuracy guarantee.

### Runtime-cost optimization and held-out replay — 2026-10-18

- Priority clarification: the objective is not to gain accuracy merely by
  porting equations. Minimize prediction loss while keeping computation as
  cheap as possible; retain only accuracy supported by measured captures.
- Profiled the in-process source-derived odometry using its per-packet
  diagnostics. Before optimization, plant advance p50/p95 was
  `107.6/427.4 us` on `run/` and `83.5/403.5 us` on `run_1/`; whole-packet
  processing p50/p95 was `278.7/638.8 us` and `199.6/595.9 us` respectively.
- The measured hot path recomputed roll/pitch trigonometry and the same
  unsteered ground-plane tire basis independently for each wheel. The source
  plant now computes those values once per PhysX vehicle substep and reuses
  them across all four wheel contacts and the gravity/body transforms. It does
  not change parameters, forces, torque branches, substep size/count, or the
  source equations. The shared slip-excess calculation uses the same cached
  basis.
- Built `f1tenth_localization` and `f1tenth_mpc` in Release. The focused source
  plant and in-process odometry tests pass: 15 plant tests plus 2 runtime
  tests. `tools/verify_runtime_topic_policy.py` passes. No simulator was
  launched; only recorded high-speed data was replayed in isolated ROS domains.
- The Release build exposed stale assertions and an out-of-bounds test copy in
  the fixed-dimension MPC rollout test. Corrected its map to include the 22
  dynamic internal optimizer coordinates while checking the four accumulated
  wheel-rotation outputs separately. The affected rollout test now passes all
  27 cases; production state dimensions and dynamics were not changed.
- Replayed the same two held-out captures after the optimization, excluding
  the recorded `/odom` and `/odom/diagnostics` inputs from playback so each bag
  contains only one newly produced odometry stream. Clean outputs are retained
  under `live_runs/odom_source_optimized_clean_replay_20261018/`. An initial
  replay that accidentally merged the capture's old and new output topics was
  invalid for accuracy scoring and has been removed; the original reference
  captures were not changed.
- The clean replay's accuracy summaries are unchanged from the pre-optimization
  source-derived candidate to reported precision: `run/` has `u` RMSE
  `0.0318 m/s`, `v` RMSE `0.0171 m/s`, relative pose endpoint error `0.072 m`;
  `run_1/` has `u` RMSE `0.0287 m/s`, `v` RMSE `0.0211 m/s`, endpoint error
  `0.096 m`. Both contain three laps with zero collisions and peak truth speed
  above `8.0 m/s`. The estimator's yaw-rate output is gyro passthrough; the
  uncorrected source model's yaw-rate RMSE remains `0.1605/0.1863 rad/s`, and
  its IMU acceleration residuals remain material (`ax 0.629/0.641`, `ay
  0.790/0.804 m/s^2`).
- Post-optimization plant-advance p50/p95 is `37.0/72.2 us` and `39.9/233.4
  us`; whole-packet p50/p95 is `75.9/129.0 us` and `85.9/355.3 us`. The
  slower tail in the second capture is host/runtime variability, but is still
  far below the 25 ms sensor period. One replayed command sample differed
  slightly between two independent ROS replays because callback delivery is
  not deterministic; aggregate held-out errors did not materially change.
- This is a computation-cost improvement without a measured accuracy gain.
  Against recorded v5 it improves forward-velocity and integrated-position
  errors but slightly worsens lateral-velocity RMSE. Against the current
  external-PhysX replay reference (`u` RMSE `0.0041/0.0055 m/s`, `v` RMSE
  `0.0003/0.0006 m/s`, endpoint `0.003/0.012 m`), the source-derived path is
  less accurate. It is therefore not described as a precision win or exact
  PhysX replication; it is a faster in-process tradeoff that still needs
  explicit acceptance against the project's required loss threshold and a
  supervised live-simulation check before any final production sign-off.

### Yaw-rate priority audit — 2026-10-18

- Re-scored the two retained three-lap high-speed captures and their clean
  source-plant replays in
  `live_runs/odom_source_optimized_clean_replay_20261018/{run,run_1}/bag`.
  No simulator was started. The capture has 811 and 805 matched 25 ms epochs,
  respectively, and each completed three laps without collision.
- Production `/odom` yaw rate is copied from the permitted IMU gyro, and its
  heading is copied from the IMU orientation. In both replays the published
  yaw-rate RMSE against the Unity bridge odometry is `0.0000 rad/s`; relative
  yaw RMSE is also `0.0000 rad`. This confirms correct same-epoch sensor
  routing on these captures, but it is not an independent physics-prediction
  result: Unity's bridge odometry and IMU angular velocity originate from the
  same Rigidbody state. Ground truth is not an input to the runtime.
- The uncorrected source-plant yaw-rate forecast is not fixed to that level:
  whole-run RMSE/bias is `0.1605/-0.0846 rad/s` and `0.1863/-0.0934 rad/s`.
  For measured `|steering|=0.15–0.30 rad`, RMSE rises to `0.2515` and
  `0.2798 rad/s`; the model overpredicts turn-rate magnitude. At
  `|steering|<0.10 rad`, RMSE is `0.0470` and `0.0606 rad/s`; at 6–8 m/s it
  is `0.0190` and `0.0213 rad/s`. Almost all high-steer samples are negative
  steering in these two runs, so the opposite-turn transfer is not established.
- This is predominantly sustained cornering error, not an onset-only effect:
  781/810 and 759/804 samples meet the near-steady criterion
  (`|steering rate| <= 0.5 rad/s`), and their yaw RMSE remains `0.163` and
  `0.188 rad/s`. In the
  high-steer subset, the model's lateral-acceleration residual has RMSE
  `0.993/0.939 m/s^2` and signed bias `-0.857/-0.810 m/s^2`. Across each
  complete run, yaw-rate and lateral-acceleration residuals correlate at
  `0.75/0.80`. This supports excessive modeled lateral tire response/yaw
  moment as the leading shared cause; it does not uniquely identify which
  wheel-force component is wrong.
- Source tracing confirms why a source-only exact correction is not yet
  justified. The active prefab supplies the sideways curve points
  (`0.01, 1.0, 0.1, 0.5`, stiffness `1`), but the production plant evaluates
  them with a locally chosen cubic response. Unity documents a two-piece
  spline and zero tangent at the extremum/asymptote, but does not publish the
  native WheelCollider spline coefficients or contact-force implementation
  [Unity `WheelFrictionCurve` API](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/WheelFrictionCurve.html).
  Thus the curve endpoints are source-known; its exact runtime force response
  remains Unity-specific and is not directly recoverable from these bags.
- No force-law coefficient was changed. Scaling yaw independently would break
  the force/moment relationship, while scaling all lateral forces could
  regress the currently acceptable lateral-speed and position estimates.
  The evidence is sufficient to target the sideways tire-force response next,
  but not to claim its exact cause or safely select a correction yet. Preserve
  the measured gyro/orientation path for odometry; evaluate any physically
  interpretable tire-response candidate on training captures and at least one
  whole held-out run, reporting yaw rate, lateral acceleration, `v`, and
  position together. The odometry yaw output is accurate on the existing
  replay evidence; the free-running yaw forecast is not.

### Retired non-practice development analyses — 2026-10-19

The prior development-player captures, generated players, and analyses based on them were removed from this workspace because they do not establish behavior for the active practice vehicle. None of those measurements are used as practice evidence. Use the practice-only evidence ledger below for current odometry and rollout scores.

### Steering-phase correction and 40 Hz hot-path optimization — 2026-10-19

- **Correction to the yaw-transition numbers immediately above:** the first
  evaluator pass initialized the plant from the post-`Steer()` rack angle,
  then advanced steering once more. In `VehicleController.FixedUpdate`, the
  logged `previous_feedback_steering` is pre-update feedback, while
  `rack_angle_deg` is post-`Steer()`. The evaluator now seeds from the former
  and lets the 1 ms plant step apply that update once. It reproduces the
  recorded post-update rack value at the transition. Do not use the earlier
  unaligned run1/run2 yaw scores; the corrected scores below supersede them.
- Corrected replay results on the same Unity captures: yaw-rate one-step
  RMSE/p95-absolute/max-absolute are `0.000300/0.000646/0.001697 rad/s`
  over 5,227 transitions (yaw-transition run1),
  `0.000290/0.000525/0.004846 rad/s` over 15,279 transitions (independent
  run2), and `0.000126/0.000183/0.003579 rad/s` over 5,999 transitions
  (locked-brake capture). Run2's max is a low-speed, zero-throttle sticky-row
  event; these scores describe these captures, not every possible regime.
- Optimized the shared odom/MPC plant without changing its equations or 1 ms
  evolution: body velocity and angular velocity are computed once per vehicle
  substep and reused for all eight tire/sticky contact calculations; the
  already-computed final attitude is reused by the constraint solve; active
  sticky rows are compacted in original PhysX order; and the solver now exits
  immediately when no rows are active. It still uses the same seven ordered
  PGS passes when rows exist.
- Added a minimal internal 1 ms propagation path for intervals where callers
  discard diagnostics. Odometry uses it for the first 24 steps in each 25 ms
  packet and computes complete diagnostics on the final step. The 25 ms MPC
  plant wrapper similarly computes complete output only on its final 1 ms
  step while accumulating the same per-interval slip maxima. A new parity test
  compares every dynamic state over one step and a batched 25-step rollout
  against explicit full-output 1 ms steps; states, maximum-slip summaries,
  final slip and acceleration diagnostics match exactly.
- Replayed the same 1,436-packet, 47.9-second `physx_sticky_online_20261018`
  bag at 1x through the rebuilt production odometry node in an isolated ROS
  domain, feeding only its encoder, IMU, actuator-feedback, and command topics.
  Before/after hot-path measurements in microseconds per packet were:

  | Measurement | Before p50 / p95 / p99 / max | After p50 / p95 / p99 / max |
  | --- | ---: | ---: |
  | Plant advance | 58.80 / 198.36 / 238.56 / 365.62 | 48.50 / 158.27 / 226.09 / 275.34 |
  | Whole odometry callback | 90.70 / 270.98 / 323.35 / 456.43 | 76.95 / 220.39 / 299.48 / 367.40 |

  The after replay had no callback above the 25 ms period. These are two
  paced replay samples, not a controlled hardware worst-case guarantee; the
  observed median and p95 reductions were about 15–18% for the full callback.
- Separately benchmarked the production 25 ms plant API at `-O3`: a 30-stage
  source-plant-only horizon is approximately `0.30 ms` at the 8 m/s turn
  point, `0.60 ms` at low-speed drive, and `0.85 ms` in the active sticky-brake
  case. This excludes MPC linearization, Riccati/ADMM, candidate checks,
  synchronization, and ROS overhead. It demonstrates that the plant advance
  itself is not the 40 Hz bottleneck; it does **not** establish full MPC
  deadline compliance. The latest recorded full RTI timing still had a
  `58.2 ms` maximum, so the current full-cycle worst case remains to be
  remeasured and addressed separately.
- Verification: Release `-O3` production MPC and odometry nodes rebuilt;
  `test_unity_source_plant`, `test_vehicle_model_source_rollout`, and
  `test_source_derived_odom_runtime` pass; corrected run2 yaw replay retains
  the metrics above; `git diff --check` passes. No simulator was launched and
  no physics, controller weights, topics, or command behavior changed.

### Fixed-input 30-step racing-lap replay after sticky correction — 2026-10-19

- Replayed both retained three-lap high-speed captures through the current
  `sensor_odometry_node` (ROS domains 76 and 77). The odometry node received
  encoder, IMU, actuator-feedback, and steering/throttle-command topics only;
  Unity truth odometry was separately recorded for offline scoring and was not
  an estimator subscription. The current runtime supplied the full starting
  plant state, wheel angles, suspension-history flag, sticky timers, and
  two-packet delayed commands.
- This is not an MPC test: from every valid 25 ms epoch, the evaluator advances
  the C++ source plant for the next 30 fixed logged commands (750 ms), without
  replanning or reading any future plant/sensor state. The commands are the
  recorded slot-1 delayed steering/throttle inputs, preserving the real
  two-packet actuation delay. Ground truth is used only to score endpoints and
  anchor each window's initial global pose. The 775 and 781 windows per capture
  overlap heavily; they are not independent statistical trials.
- Fixed an evaluator ABI defect before accepting these results. Its Python
  `ctypes` structs had an 18-value state where `unity_source_plant_c.h` defines
  26 values, an extra output field absent from the C ABI, and it omitted the
  two sticky-timer arrays when reconstructing a start state. The evaluator
  now uses the exact 26-value state and C struct layout, checks ABI sizes and
  offsets, and carries both sticky timers. Before this correction the 30-step
  evaluator skipped 774/775 windows; earlier output from that path is invalid.
  The separate `0.1605/0.1863 rad/s` raw-yaw diagnostic came from recorded
  odometry diagnostics and was not affected by the ctypes defect.
- Corrected-model 25 ms yaw-rate forecast across the full captures is
  `0.0201 rad/s` RMSE, `0.0865` max for `run/`, and `0.0225` RMSE, `0.1004`
  max for `run_1/`. This is materially below the stale pre-sticky-correction
  `0.18 rad/s` whole-run RMSE, though the second run's single-step maximum is
  just above `0.1 rad/s`.
- At the requested 30-step/750 ms endpoint:

  | Capture | `u/v/r` RMSE (m/s, m/s, rad/s) | yaw RMSE / p95 / max (rad) | displacement RMSE / p95 / max (m) |
  | --- | ---: | ---: | ---: |
  | `run/` | `0.00395 / 0.00043 / 0.00337` | `0.01391 / 0.02423 / 0.02860` | `0.02789 / 0.05763 / 0.06912` |
  | `run_1/` | `0.00517 / 0.00082 / 0.00671` | `0.01549 / 0.02605 / 0.06738` | `0.03197 / 0.05981 / 0.13869` |

  The yaw-rate endpoint maximum is `0.01884 rad/s` for `run/` and
  `0.09439 rad/s` for `run_1/`; the latter is a tail event despite its low
  RMSE/p95. The initially reported ~15 cm position RMSE was a scoring-frame
  error: it integrated COM lateral velocity against a rear-axle/base_link pose.
  After converting velocity to the pose reference point, the 750 ms position
  RMSE is 2.8–3.2 cm. Every window starts at its recorded plant state and Unity
  pose, so this is not free-running full-lap odometry drift.
- Full reports and regenerated state captures are retained under
  `live_runs/sticky_corrected_30step_replay_20261019/{run,run_1}/`. No MPC
  solve or Unity simulator was run. These results resolve the comparison for
  these two captures only; they do not establish accuracy at every speed,
  steering combination, or on a completely unseen lap.

### Endpoint position reference-point correction — 2026-10-19

- The 15 cm result came from a bug in the offline evaluator, not the production
  plant: the evaluator propagated COM lateral velocity directly into the
  rear-axle/base_link pose. Unity twist is at the rigid-body COM; the pose being
  scored is at the rear axle. For longitudinal offset `L=0.15532 m`, the
  matching lateral velocity is `v_rear = v_COM - r*L`.
- This diagnosis was directly checked against consecutive Unity pose samples
  on both independent captures. Integrating COM twist as if it were rear-axle
  twist had `7.19/7.47 mm` one-step displacement RMSE and `-3.79/-3.71 mm`
  lateral bias. Applying the source/production COM-to-rear-axle conversion
  reduced those to `3.54/4.29 mm` RMSE and `-0.18/-0.18 mm` lateral bias.
  Treating twist as world-frame was much worse (`~0.167 m` per step). The
  production code already performs this conversion in
  `integrate_rear_axle_pose()` and `vehicle_model_step_scalar()`; no runtime
  physics change was warranted.
- Corrected 750 ms decomposition:

  | Capture | Plant vs Unity pose RMSE / p95 | Unity twist-integral vs pose RMSE / p95 | Plant vs Unity-twist integral RMSE / p95 |
  | --- | ---: | ---: | ---: |
  | `run/` | `0.0279 / 0.0576 m` | `0.0052 / 0.0089 m` | `0.0281 / 0.0565 m` |
  | `run_1/` | `0.0320 / 0.0598 m` | `0.0057 / 0.0101 m` | `0.0328 / 0.0608 m` |

  The two captures therefore do not support the earlier claim of 15 cm
  rollout error. The remaining ~3 cm RMSE is a real plant-vs-reference motion
  residual; its tail is larger (max corrected endpoint error is `6.9 cm` in
  `run/` and `13.9 cm` in `run_1/`). A 1 ms pose-integration diagnostic using
  linearly interpolated 40 Hz endpoints produced essentially the same error,
  so simply shrinking the kinematic pose step does not explain or remove the
  residual.
- Applying the user's 5 cm endpoint requirement as a strict per-window gate,
  `700/781` windows in `run/` and `667/775` in `run_1/` pass; therefore the
  current rollout still fails the requirement. Windows overlap, so these are
  coverage counts, not independent confidence trials. The worst `run_1/`
  windows start near `3.5 m/s` and `0.258 rad` steering, end with `0.059–0.067
  rad` yaw error, but only about `0.007 m/s` forward-speed error and
  `0.002–0.003 rad/s` yaw-rate error. This points to accumulated yaw-angle
  trajectory error as the dominant tail mechanism there; it is not yet a
  proven causal attribution for every failing window.
- An offline oracle decomposition on those same windows confirms yaw is the
  limiting state: replacing only predicted heading with recorded future Unity
  yaw reduces 750 ms position RMSE to `5.9 mm` (`run/`) and `7.7 mm` (`run_1/`),
  with all windows below 5 cm. Replacing only predicted translation with
  recorded Unity `u/v/r` while keeping predicted yaw leaves RMSE at `2.79 cm`
  and `3.14 cm`, with 79/781 and 103/775 windows still over 5 cm. The GT yaw
  is used only for this offline attribution and is not a runtime input.
- In the worst `run_1/` windows, mean predicted-minus-GT yaw-rate error across
  750 ms is `+0.075–+0.094 rad/s`, while endpoint yaw-rate error is only
  `0.002–0.003 rad/s`. This identifies a transient yaw-trajectory mismatch
  that later reconverges, especially around `3.4–3.5 m/s` and `0.258 rad`
  steering. The 5 cm endpoint maximum criterion remains unmet; next inspect
  the source-derived tire lateral force/yaw-moment path and its known
  WheelCollider/PhysX approximations before changing the model. Runtime must
  predict this from current estimated state and queued controls, never future
  GT yaw.
- `tools/evaluate_mpc_bag.py` now scores pose at the rear-axle reference point,
  converts both predicted and GT COM lateral velocity before integration, and
  reports the pose/twist closure hypotheses. Verification: both bags rescored
  with the workspace messages sourced; script syntax and `git diff --check`
  pass. No simulator or controller ran.
- The 5 cm requirement is now checked both at the 750 ms terminal point and
  pathwise at every 25 ms prediction step within each 750 ms rollout. The
  pathwise gate passes `700/781` windows for `run/` and `661/775` for
  `run_1/`; `81` and `114` windows respectively exceed 5 cm at some point.
  Worst per-window error is `6.91 cm` and `13.87 cm`. This is stricter than
  checking the terminal point alone (`700/781`, `667/775`), because some
  rollouts briefly exceed the bound then recover. Neither capture meets a
  universal 5 cm guarantee.
- The current vehicle prefab's steering rate (`183.346 deg/s`), wheelbase
  (`324 mm`), track (`236 mm`), and sideways friction-curve points match the
  production plant constants. This rules out those direct configuration
  mismatches as the explanation for the high-steering yaw residual. Unity's
  public WheelCollider documentation describes separate forward and sideways
  slip-force curves; the plant's additional combined-slip cap and effective
  tire-load mapping remain acknowledged approximations. Do not remove the cap
  blindly: a previous aggregate ablation regressed another capture. Next,
  compare that approximation specifically on the failing `3.4–3.6 m/s`,
  `~0.26 rad` turning windows and across the rest of both held-out runs before
  any production change.
- A 5 cm endpoint error alone cannot guarantee obstacle clearance: safety also
  depends on the maximum error along the whole predicted horizon, vehicle
  footprint, map/corridor error, localization error, and tracking deviation.
  The new pathwise metric checks intermediate prediction positions, but the
  combined physical clearance budget has not yet been established.

### PhysX source-equation corrections — 2026-10-19

- Continued from source mechanics, not parameter fitting. The practice
  simulator's UnityPlayer symbols establish the native wheel-center target
  shift; the serialized prefab provides `suspensionDistance=0.05 m` and
  `targetPosition=0.5`. Production suspension compression, droop, and the
  separate native center shift are now all derived from those source values,
  rather than repeated as independent `0.025 m` constants.
- Corrected the wheel-speed transition to follow
  `PxVehicleUpdate.cpp::computeNoDriveBrakeTorques` and
  `integrateNoDriveWheelSpeeds`: signed brake torque is zero when wheel omega
  is zero, implicit damping is applied in the source denominator, and an
  active brake locks the wheel when old and new omega straddle or touch zero.
  The previous local zero-speed torque-complementarity condition was not the
  PhysX algorithm and could permit a stopped, braked wheel to spin if the
  opposing tire torque exceeded the brake torque.
- Removed another opaque wheel constant: the native UnityPlayer wheel-MOI
  construction in `WheelCollider::AddWheelToExistingVehicle` computes
  `0.5 * mass * radius^2`. The production plant now derives MOI from the
  prefab's 0.109 kg mass and 0.059 m radius, and a focused test checks the
  exact public PhysX implicit-update equation against a production step.
- Capture-derived comparisons from the prior development campaign were
  removed and are not accepted as practice evidence. The source equations and
  implementation remain; current accuracy claims are limited to the retained
  practice replays in the evidence ledger.

### Focused yaw-miss check after evaluator rebuild — 2026-10-19

The old candidate scores in this entry were removed because their input
provenance is not part of the retained practice-only evidence set. Use the
current production replay and outlier findings in
[`YAW_MISMATCH_EVIDENCE_LEDGER_20261019.md`](YAW_MISMATCH_EVIDENCE_LEDGER_20261019.md).

### Practice-only evidence ledger — 2026-10-19

The previous non-practice development captures and analyses have been removed.
The active evidence and precise remaining yaw/position limits are summarized
in [`YAW_MISMATCH_EVIDENCE_LEDGER_20261019.md`](YAW_MISMATCH_EVIDENCE_LEDGER_20261019.md).
The substep statement in this 2026-10-19 section was wrong. Exact-build native
disassembly on 2026-10-20 established `setSubStepCount(128.0f, 5, 1)`; the
current source and boundary test use five substeps below 128 m/s and one at or
above 128 m/s. This is established by the matching practice-player binary,
not inferred from bag timing. The unrelated public-PhysX 5 m/s constant is
only for low-speed wheel-pose-angle blending.

### Current production odom and plant-only rollout, practice bags only — 2026-10-19

- Replayed the two retained three-lap full-speed practice bags through the
  rebuilt production `sensor_odometry_node`, isolated from MPC and Unity. New
  input/output bags are retained at
  `live_runs/current_odom_practice_replay_20261019/{run,run_1}/bag/`.
- Current production odometry vs same-epoch Unity truth:
  - `run/` (811 epochs, truth max `8.093 m/s`): `u` RMSE/max
    `0.0044/0.0198 m/s`, `v` `0.0004/0.0023 m/s`, relative XY drift
    end/max `0.011/0.042 m`.
  - `run_1/` (805 epochs, truth max `8.078 m/s`): `u` RMSE/max
    `0.0058/0.0314 m/s`, `v` `0.0006/0.0114 m/s`, relative XY drift
    end/max `0.019/0.052 m`.
  - Published `r` and heading have zero error because they are direct gyro and
    IMU-orientation passthroughs; these are not independent yaw-model scores.
- Separately ran the source-plant evaluator on the captured production plant
  state, with 30 recorded delayed steering/throttle commands and no future
  state correction. At 750 ms, `run/` has `u/v/r` RMSE
  `0.00399 m/s / 0.00037 m/s / 0.00288 rad/s`, yaw RMSE `0.00155 rad`, and
  rear-axle XY RMSE `0.0055 m`; all 781 overlapping windows stay within 5 cm
  throughout, worst `4.80 cm`. `run_1/` has `0.00532 / 0.00068 / 0.00550`,
  yaw `0.00202 rad`, XY `0.0076 m`; all 775 endpoints pass 5 cm and 774/775
  paths pass, worst `5.13 cm`. The `run_1` rare yaw-rate endpoint maximum is
  `0.09023 rad/s` despite low RMSE. Overlapping windows are not independent
  trials.
- Also rescored the separate yaw-target practice bag. It starts at about
  `1.8 m/s` while the odometry replay starts from the settled reset state:
  whole-run `u` RMSE is `0.1680 m/s` (p95 `0.0143`, max `1.8143`) and relative
  XY reaches `0.454 m`, dominated by initialization mismatch. Its 750 ms
  plant-only rollout has `u/v/r` RMSE `0.00406 / 0.00033 / 0.00254`, yaw
  `0.00245 rad`, and XY `0.0319 m`, but a startup tail reaches `0.4207 m` and
  only 715/727 endpoints pass 5 cm. Keep this as a separate initialization
  failure, not a clean-spawn full-lap score. Its high-yaw raw/corrected forecast
  maxima are `0.1030/0.1117 rad/s`.
- Conclusion: on the two clean high-speed practice bags, odometry and the
  750 ms plant-only rollout score well, but heading/rate odometry is gyro
  passthrough, the second rollout contains a `0.09 rad/s` tail, and the
  yaw-target practice capture still has an initialization mismatch and
  `>0.1 rad/s` yaw-forecast tail. This is not a claim that yaw is solved.

### Practice-only cleanup and hot-path optimization — 2026-10-19

- Removed the Explore/open-plane capture code, player hooks, scene/build
  artifacts, data folders, standalone capture evaluator, generated evaluator
  binaries/reports, and the disconnected
  `SDU_Apex_IPOPT_PhysX_MinTime_v2` optimizer tree from the workspace. The
  final optimizer-tree remainder was moved to the desktop Trash, not
  irreversibly erased. Active runtime, tool, and Unity asset source scans
  contain no Explore/open-plane simulator references. Practice recordings
  and the separate historical practice MPC reference remain untouched.
- Reduced repeated per-wheel plant work by caching wheel-ray start vectors and
  shared attitude terms, reusing the fixed Ackermann basis, and avoiding
  repeated transforms. The localization node now reuses ROS output messages
  and frame strings; plant timing/copy/diagnostic work is opt-in and
  `publish_diagnostics` defaults to false.
- Replayed the two retained high-speed practice bags with the rebuilt
  production odometry. Plant advance median/p95/max fell from
  `699/1272/2017` to `41/71/84 us` on `run`, and
  `944/1761/3030` to `42/67/99 us` on `run_1`. Diagnostic-enabled whole-packet
  worst case was `0.326 ms`; this is a local replay measurement, not a
  machine-independent performance guarantee.
- The practice accuracy gate did not regress materially: `run` retained
  `0.0044 m/s` forward-speed RMSE, `0.0004 m/s` lateral-speed RMSE, and
  `4.80 cm` worst 750 ms path error. `run_1` retained `0.0060` and
  `0.0007 m/s` speed RMSE respectively, but has one 750 ms path at `5.13 cm`
  (one miss by `1.3 mm`); its full-run relative-position maximum is `5.1 cm`.
  The isolated `0.0902 rad/s` yaw-rate rollout residual remains unresolved.
- The full localization CTest suite passes (5/5), the plant, MPC control-time,
  and MPC state-synchronizer CTests pass, and the runtime topic-policy check
  passes. The MPC rollout binary passes 31/32 tests; its queued-command
  Jacobian reconstruction test misses strict yaw/state tolerances (maximum
  yaw reconstruction difference `9.77e-5 rad/s` against `5e-5`). No MPC code
  was changed in this cleanup.
- No simulator was launched. See the practice-only evidence ledger for the
  full metrics and caveats.

### Production odometry steering-feedback phase correction — 2026-10-19

- Unity source confirms `CurrentSteeringAngle` reports `currentS`, sampled in
  `FixedUpdate()` before `Steer()` advances the rate-limited rack for that
  physics tick. Production odometry had treated that pre-update sample as the
  current post-update rack angle.
- Corrected only the internal source-plant rack state: at each valid feedback
  boundary, advance it by one existing `unity_steering_step` using the oldest
  queued steering command under the existing two-packet delay assumption.
  Do not advance body motion again. Raw steering feedback publication, ROS
  topics, command behavior, and MPC inputs are unchanged. The command and
  feedback topics have no Unity step/sequence identifiers, so the exact
  command-to-feedback association remains unobservable in the retained bags.
- Added a focused unit test for the one reconstructed rack step plus the next
  25 plant ticks. Rebuilt `f1tenth_localization`; all 5 localization CTests
  pass. Runtime topic-policy verification also passes.
- Replayed both retained full-speed practice bags through the candidate
  production odometry. There is no material regression in the full-run
  metrics, but the correction does **not** solve the yaw tail: at run 1 sample
  715 forecast yaw changes only `1.35e-5 rad/s`, leaving the residual at
  `0.0903 rad/s`. The 750 ms path gates remain run `781/781` (worst 4.80 cm)
  and run 1 `774/775` (worst 5.13 cm). This removes a source-order mismatch;
  it is not evidence that the yaw mismatch is fixed. Full evidence and
  limitations are in
  [`YAW_MISMATCH_EVIDENCE_LEDGER_20261019.md`](YAW_MISMATCH_EVIDENCE_LEDGER_20261019.md).
- No simulator was launched and no tire/yaw parameter was fitted or changed.

### Diagnosis of the remaining 5 cm path-gate miss — 2026-10-19

- Replayed the retained run-1 bag with the production source-plant evaluator.
  The one failure starts near 7.44 m/s and near-zero steering, peaks at
  `5.13 cm` at 725 ms, then returns to `4.73 cm` at the 750 ms endpoint. At
  the peak, yaw error is `0.0016 rad`, forward-speed error `-0.0061 m/s`, and
  lateral-speed error about `+0.0002 m/s`; this is distinct from the isolated
  `0.0903 rad/s` yaw-rate outlier.
- Extended `tools/evaluate_mpc_bag.py` to split each 25 ms displacement error
  into plant-vs-integrated-GT-twist and integrated-GT-twist-vs-GT-pose terms.
  For the failing window's first step, the plant/twist term is about `0.3 mm`
  while GT twist/pose disagreement is `3.76 cm` longitudinal. At the 725 ms
  peak step, the plant/twist term remains about `0.3 mm`; the GT twist/pose
  term is about `4.7 mm` longitudinal and `1.5 mm` lateral. The retained
  evidence therefore points to reference sample phase/pose-point inconsistency,
  not an identified PhysX plant-equation error.
- The replay does not contain bridge packet timing or a Unity physics-step ID;
  current headers are stamped at the host request boundary. Consequently this
  is not enough to assert precisely why pose and twist disagree. Do not fit a
  pose offset, loosen the 5 cm limit, or tune vehicle dynamics to hide it. Keep
  the single window as a gate failure until same-physics-epoch pose/twist data
  are available. A development-only Unity sample timestamp/sequence plus the
  bridge timing topic can resolve alignment offline without becoming a
  production input. Details are in the evidence ledger.

### Disposition of the isolated 5 cm threshold exceedance — 2026-10-19

- The run-1 result is one exceedance among 775 overlapping 750 ms rollout
  windows: `5.13 cm`, only `1.3 mm` above the strict `5.00 cm` threshold. Its
  750 ms endpoint is `4.73 cm`; the other 774 windows pass. Because these
  windows overlap, this is a descriptive count, not 775 independent trials.
- Rechecked truth pose/twist epoch coherence on the two replay bags and an
  independent timed practice capture. A constant sample-phase shift does not
  reduce the residual meaningfully (best phase is 0 ms). Pose/twist closure
  anomalies also occur in other practice data; in the timed capture they are
  associated with packet-jitter magnitude, but that association is not causal
  proof. The failing run-1 bag itself has no bridge timing event or Unity
  physics-step ID, so its specific source epoch cannot be recovered.
- No production-code or plant-equation defect has been identified that
  explains this 1.3 mm overrun. The evidence instead shows a near-threshold,
  isolated path maximum alongside pose/twist reference inconsistency. No
  unsupported physics adjustment is justified. Treat this miss as negligible
  for practical replay assessment, while retaining the exact `5.13 cm` value
  and the strict 774/775 score in reports; do not relabel it as a strict-gate
  pass or claim the global 5 cm bound is proven.
- Added an offline-only `--truth-pose-twist-phase` diagnostic and the
  early-exit `--truth-pose-twist-phase-only` option to
  `tools/evaluate_mpc_bag.py`. They sweep a constant phase and, when bridge
  timing exists, compare pose/twist closure with request/receive jitter. They
  do not alter model inputs or apply a correction. Python syntax and diff
  whitespace checks pass. No simulator was launched and no vehicle-model
  parameter was changed.

### MPC iteration ceiling, state reduction, and timing investigation — 2026-10-19

- Production MPC is configured for at most 100 ADMM iterations per RTI pass.
  Adaptive refinement can execute R1 and then R2, so total ADMM iterations in
  one control cycle can reach 200 if both consume their full allowance. The
  node clamps
  its ROS parameter to `[1, MAX_ITERATIONS]`, and `MAX_ITERATIONS` is 100.
  `mpc_competition.yaml` sets 100 iterations and `rti_cycle_budget_ms: 0.0`,
  so there is no RTI wall-clock deadline in the competition profile. This is
  unbounded by elapsed time, not by iteration count: each solve still stops at
  convergence or after 100 iterations. The runtime's selected-solver maximum
  is not an aggregate R1+R2 counter; use the separate R1/R2 iteration fields
  when assessing total work. The open-loop horizon-test profile also
  requests 100 iterations and retains its separate 20 ms test budget.
- Kept the reusable Riccati solver's caller-supplied iteration capability. A
  proposed solver-wide clamp to 100 made the scaling-equivalence test fail at
  its `2e-3` solution tolerance (the test intentionally requests 500
  iterations); that over-broad change was reverted. Production remains capped
  in the ROS node, while numerical solver tests can request extra convergence
  iterations. The rebuilt MPC package passes all 4 CTests after the revert.
- Current RTI decision dimensions are 20 states, 2 controls, and 28 optimized
  stages for a 30-step physical prediction. The two already-queued command
  packets are applied as a deterministic prefix; the remaining planned control
  sequence is shifted through the two-packet delay. This preserves the delay
  in the nonlinear plant without four extra queue coordinates in the QP.
- State-role audit: `e_y/e_psi`, `u/v/r`, physical steering, four wheel angular
  speeds, body heave/roll/pitch and rates, and four suspension jounces remain
  in the 20-state QP. Tracking directly weights path errors, body velocity/yaw
  rate, and physical steering; the source-derived slip-loss term contributes
  wheel-speed costs. Heave, attitude, rates, and jounces are not direct tracking
  targets in the current cost, but they affect tire contact/load and future
  plant motion and remain dynamically coupled. Dropping them from
  the optimizer would require an equivalent reduced/condensed dynamics and
  constraint formulation; simply removing their weights or states is not an
  exact shortcut. No further physical state removal is justified by this
  audit. Per-wheel slip/load values are transition diagnostics, not separate
  optimizer states.
- Existing cost/solver work is already specialized in several relevant ways:
  compile-time fixed matrices, diagonal stage/terminal state costs and input
  costs, a 2x2 control-Hessian inverse, zero-weight cost-term elision, and
  cached Riccati factorization reused across ADMM RHS passes while `rho` and
  `rho_u` are unchanged. Adaptive-rho changes invalidate that factorization
  intentionally. Reimplementing these as a second cache or generic sparse
  layer would duplicate current behavior.
- A plant steering-trace cache was compared against ordinary 25 ms plant
  stepping in a focused optimized-build microbenchmark: ordinary stepping was
  `10.75 us`, cached stepping `11.30 us` (about 5% slower). The cache and its
  tests were removed. A generic branch-on-zero Riccati-loop experiment was
  also removed: the linearized plant has exact structural zeros
  `A[2:20, 0:2]`, but the branch-based implementation had no fair whole-solver
  speed evidence and risked blocking compiler vectorization. Neither experiment
  remains in production.
- The remaining substantial representation candidate is exact partial/full
  condensing: eliminate some or all 20 state decision vectors while preserving
  their plant propagation and corridor/state constraints. Full condensing
  would make those path and vehicle bounds dense in the 56 control decisions;
  it is not automatically cheaper than the current 20-state Riccati recursion.
  No state was removed on that assumption. A fair next comparison must feed the
  same captured linearized QP to both formulations and compare solve time,
  residuals, constraints, and recovered controls before considering a change.
- Retained three practice timing captures under `live_runs/`:
  `mpc_delay_condensed_timing_20261019/run` (earlier reference),
  `mpc_steering_trace_cache_20261019/run` (cache build, no completed lap), and
  `mpc_zero_skip_20261019/run` (experimental zero-skip build, collision; watcher
  stopped the run immediately). Do not count either experimental run as a
  driving success or as a clean A/B timing comparison.
- The earlier reference capture measured total RTI `24.62 ms` median,
  `33.32 ms` p95, and `38.32 ms` max. In the collision capture, total RTI was
  `31.32 ms` median, `47.86 ms` p95, `54.85 ms` p99, and `59.42 ms` max; its
  QP build was `6.41/8.59 ms` median/p95, R1 solve `12.86/20.38 ms`, and R2
  corridor repair p95 `6.07 ms`. The runs differ in driving regime and outcome,
  so this is not evidence that the reverted zero-skip code caused a timing
  regression. Available measurements do show that a sub-25 ms p95 and the
  preferred 10–15 ms p95 target have not been established.
- The cache-run watcher stopped after inadequate progress; it had no collision
  but completed no lap. The zero-skip run had one collision and no lap. These
  are failed drive validations. No new performance shortcut is promoted from
  these runs. `colcon build --packages-select f1tenth_mpc` and all 4 package
  CTests pass; `git diff --check` passes.

### Wheel-spin and suspension-state reduction audit — 2026-10-19

- Audited the 20-state QP against the source plant rather than removing
  coordinates based on whether they have direct cost weights. All four wheel
  angular speeds remain separate physical states: each has its own
  wheel-inertia update, contact-point velocity, tire slip/torque, normal load,
  and contribution to body force/yaw. Front Ackermann steering and left/right
  load/contact asymmetry break axle symmetry during turns. Combining them
  into axle averages would discard physically distinct slip and yaw channels.
  The wheel *rotation angles* are already excluded because they only
  accumulate encoder output and do not feed the source plant's forces.
- The four jounce entries are different: source `wheel_contact_kinematics`
  recomputes each from a common flat-road ray/plane geometry, while the saved
  value is retained only to form the next 1 ms damper rate
  `(jounce_now - jounce_previous)/dt`. Thus jounce history must be preserved,
  but its four values are not four independent physical degrees of freedom.
- Checked seven retained October 19 practice captures (5,460 plant-state
  messages; maximum logged forward speed 5.60 m/s). No jounce reached the
  source clamp limits of `-25/+25 mm`; observed values ranged from
  `-9.30` to `+9.28 mm`. With the source wheel geometry, the four unclamped
  jounces obey, to floating-point precision, the fixed affine relation
  `j_RR = j_RL + 1.0188125681639262 * (j_FR - j_FL)`; maximum relation
  residual across these captures was below `2e-10 m`. This establishes an
  exact 4-to-3 coordinate reduction on the observed grounded, unclamped
  practice branch while preserving the four physical jounces in the plant.
- A robust reduction across wheel unloading/clamping should carry the three
  underlying prior ray-plane geometry parameters, then regenerate and clamp
  all four jounces in the unchanged plant. Merely dropping RR and using the
  affine relation is not valid after independent per-wheel clipping. The
  current ROS plant-state message exports jounces, not those three raw
  geometry-history parameters, so the robust 4-to-3 optimizer transform would
  require a small plant/message/serialization change and transformed
  linearization (`A_red = P*A_full*S`).
- Replacing history with jounce calculated from the *current* body height,
  roll, and pitch is not an equivalent shortcut. On the dynamic
  `mpc_repair_reserve` capture, that projection differed from carried history
  by `0.20 mm` p95 and `4.14 mm` max. Because the damper divides that difference
  by a 1 ms step, a paired source-plant perturbation changed the one-step yaw
  rate by `0.0080 rad/s` p95 and `0.0209 rad/s` max over the 248 valid source
  transitions. This subset result is not a full-run accuracy claim, but it
  rejects the naive current-pose reconstruction.
- Jounce coordinates have no direct tracking cost in the active QP; they are
  retained for their effect on spring/damper load and subsequent tire/body
  motion. Wheel speeds receive the existing per-wheel slip-loss cost. A state
  with zero direct weight is therefore not automatically redundant.
- No production state was removed in this audit. The promising safe target is
  a 20-to-19 QP representation of the jounce-history geometry, not removal of
  jounce physics. Before promotion, implement the three-parameter history
  encoding and test exact source-transition parity including droop/compression
  branches, then compare QP controls, nonlinear candidate rollouts, and
  measured solver timing on the same practice inputs. The expected compute
  gain is modest and remains unmeasured. Keep all four wheel angular speeds.

### MPC computation hot-path audit — 2026-10-19

- Kept the QP at 20 physical states as requested. Re-read the active
  linearization, Riccati-ADMM, RTI, reference-sampling, and exact-rollout paths
  for work that can be removed without changing the vehicle model or MPC
  behavior.
- The latest retained completed practice timing capture,
  `live_runs/mpc_delay_condensed_timing_20261019/run`, has 488 valid timing
  diagnostics. Measured full RTI-cycle time was `24.62 ms` median,
  `33.32 ms` p95, `36.39 ms` p99, and `38.32 ms` max. Thus this capture does
  **not** establish reliable completion inside a 25 ms cycle, and it is not a
  fresh timing guarantee for any source edits made after that run.
- Phase breakdown (median / p95): R1 QP/Jacobian build `5.88/7.06 ms`, R1
  ADMM `3.25/6.35 ms`, R1 exact rollout `1.23/2.46 ms`; R2 build
  `5.72/6.92 ms`, R2 ADMM `3.89/6.49 ms`, R2 exact rollout `1.16/2.48 ms`.
  Nonlinear candidate scoring itself was only about `11 us` median. The dominant
  cost is rebuilding the stage Jacobians and solving a second RTI pass, not the
  exact source-plant horizon rollout or candidate score.
- R2 ran on `422/488` cycles (`86.5%`); exact corridor repair was used on
  `128/488` (`26.2%`). Therefore “skip R2” is not a behavior-neutral
  optimization. It would remove a frequently used refinement/recovery path and
  needs a separate acceptance/drive-quality study before any change.
- The single-solve control-time state prediction was `73 us` median,
  `125 us` p95, `637 us` max in this capture. It is not a hot-path target.
- No new low-risk matrix shortcut was found: dimensions and diagonal costs are
  fixed, the command-delay prefix is already condensed, the 2x2 control Schur
  complement is specialized, and the Riccati quadratic factorization is reused
  for RHS-only ADMM iterations until adaptive rho changes. Previous experiments
  with generic zero-skip loops and plant steering caches did not demonstrate a
  reliable end-to-end win and are not in production.
- The material remaining code-level opportunity is replacing repeated scalar
  perturb-and-rerun plant evaluations in the branch-aware finite-difference
  Jacobian with validated derivative propagation through the same plant
  equations. This is a substantial implementation, not a safe algebraic
  deletion: it must preserve branch decisions and match the current Jacobian
  and resulting QP/candidate on practice states before timing gains count.
  Smaller opportunities (duplicate buffer initialization and repeated fixed
  reference/cost setup) are likely minor and should only be retained if a
  same-input benchmark shows an end-to-end improvement.
- No production math or behavior was changed in this audit. Next performance
  work should instrument one completed practice run by phase and attempt the
  Jacobian sensitivity path first; report new current-code timings separately
  from the retained reference above.

### MPC Jacobian derivative-propagation investigation — 2026-10-19

- Corrected the finite-difference workload in `vehicle_model.c`: each active
  transition needs 20 expensive plant-state columns (`u`, `v`, `r`, applied
  steering/throttle, actual steering, four wheel rates, six body states, and
  four jounces). The two Frenet columns use the already computed displacement;
  the two queued-command shift columns and control append rows are analytic.
  For the 28 optimized transitions, that is 560 perturbed 25 ms source-plant
  rollouts per RTI pass (14,000 repeated 1 ms plant steps), before any
  opposite-side evaluations. R2 repeats the work when triggered.
- Re-read the original timing bag rather than relying on its first record,
  which is a startup rejection. It contains 488 valid solver records: R1 build
  median/p95 `5.885/7.061 ms`; R2 build `5.721/6.923 ms`. The logged
  `nonsmooth_columns` distribution is p50 `0`, p90 `5`, p95 `12`, maximum `30`;
  302/488 records have zero and 186 have one or more. The configured trigger
  threshold of 50 was never reached.
- Important limitation: the current nominal-branch signature only records
  steering clipping, throttle clipping, and the zero-throttle brake mode. It
  does not encode the internal tire-spline, load-filter, contact, wheel-lock,
  sticky-constraint, or wheel-substep branches. Therefore this logged count
  cannot be interpreted as a complete count of PhysX/source-plant active-set
  crossings, and an AD implementation must not claim branch parity from that
  counter alone.
- A targeted `-O3` microbenchmark of the current single-stage
  `mpc_vehicle_model_linearize_from_known_step` on the checked-in core library
  measured `199.6 us` median, `230.9 us` p95, `366.7 us` max over 1,000 timed
  calls after warm-up. This is a local 6 m/s, 0.35 rad turning-state benchmark,
  not a practice-run or competition timing claim; its scale is consistent with
  the recorded multi-stage QP-build time.
- Forward-mode differentiation remains a plausible candidate, but only as
  derivatives of the exact active branch of the existing plant. A compact
  four-lane (AVX2-capable) dual scalar could cover 20 columns in five plant
  passes; whether this beats 18 scalar finite-difference rollouts is an open
  measurement, not an assumed speedup. It requires templating/refactoring the
  shared plant math and careful handling of branch guards. A scalar dual
  implementation without vectorization may be slower because every arithmetic
  operation carries all tangent lanes.
- No production implementation was changed in this investigation. The next
  implementation gate is a side-by-side derivative prototype with unchanged
  nominal plant outputs, Jacobian parity against the current finite-difference
  path on smooth and branch-adjacent practice states, equivalent QP/candidate
  results, then same-input full-cycle timing. Do not replace the current path
  unless those checks show both acceptable numerical parity and an end-to-end
  latency gain.

### MPC Jacobian and parallel-build implementation — 2026-10-19

- Implemented forward-mode sensitivities in the production Unity-source plant
  equations. The primal state remains double precision; four tangent lanes are
  carried together, covering the 20 plant-dependent MPC state columns in five
  source-plant passes. Existing analytic FIFO/Frenet columns remain analytic.
  At the hybrid `throttle == 0` brake/drive boundary, the throttle column still
  uses the existing one-sided finite difference rather than pretending the
  branch is smooth.
- The active C MPC linearizer now uses the source sensitivities for those
  columns and falls back to its existing branch-aware finite difference when
  the sensitivity call fails or a column is deliberately left uncovered.
  No optimizer state was removed and no plant equations or control semantics
  were changed.
- Added direct tests comparing the sensitivity API with finite differences on
  smooth tire branches, attitude/body states, and all four jounce directions;
  all passed. The active QP-build result was also compared across 1/2/4/8/16
  Jacobian workers; stage A and affine terms matched the single-worker result
  within `1e-6` for all tested worker counts.
- A local synthetic 28-transition QP-build timing screen measured p50/p95:
  1 worker `7.27/7.31 ms`; 2 `3.73/3.75 ms`; 4 `1.89/2.04 ms`; 8
  `1.37/1.38 ms`; 16 `0.77/6.00 ms`. Eight is the best repeatable choice in
  this screen; 16 had a large tail. The competition config and node default
  are set to 8 workers accordingly. These are QP-build-only timings on this
  machine, not a complete RTI-cycle or simulator result.
- A single-stage micro-screen was less favorable: the full active linearizer
  measured about `270/276 us` p50/p95, while 20 bare one-sided plant calls were
  about `194/198 us`. That comparison excludes linearizer bookkeeping and is
  not fully apples-to-apples, but it prevents claiming a standalone stage
  speedup. The measured QP improvement comes from parallel stage processing;
  complete RTI timing must decide whether the derivative implementation is a
  net production win.
- Still required before calling this fast/effective: build the full node,
  run all focused MPC/plant regressions, and measure full RTI p50/p95/max on
  the established practice simulator path with eight workers. The current
  `15 ms` median / `25 ms` maximum objective has not yet been verified on that
  path; no live-drive, solve-quality, or timing guarantee is claimed here.

### Startup-bag ADMM convergence recovery — 2026-10-20

- The supervised practice capture is
  `live_runs/mpc_startup_convergence_20261020/run`; it ended after a collision
  near progress `48.5 m`. Its source-stamp-aligned solver input export is
  `live_runs/mpc_convergence_replay_20261020/startup_live_inputs.csv` with
  349 inputs at approximately 40 Hz. The separate follow-up live validation
  is described below.
- With the prior candidate, 7 inputs had no sub-50 converged R1/R2 pass; the
  hard tail cluster was rows 336–341. Repeating the same failed QP from a cold
  start did not help. Rho/relaxation and tracking-weight sweeps also failed to
  remove that cluster without making other rows worse.
- Implemented a bounded numerical recovery in `solve_rti_pass`: when ADMM
  reaches its iteration limit, rebuild the same linearized problem with the
  configured QP corridor padding, then try up to `0.25 m` and `0.50 m` padding.
  This padding affects only linearized QP bounds; exact Unity-source nonlinear
  rollout validation continues to use the original physical corridor. It does
  not alter vehicle equations, map, trajectory, or physical safety limits.
- Replayed all 349 recorded inputs sequentially with cold solver starts and a
  per-solve cap of 49. Every input now has at least one converged R1 or R2
  ADMM pass in 1–49 iterations (`349/349`). Per-pass maxima are 49; some
  numerical passes require up to three attempts and therefore up to 147 total
  ADMM iterations across attempts. R1 converged on `346/349` inputs; R2 on
  `343/349`. The bag-level cycle outcomes are 323 optimal accepted, 16
  degraded accepted, and 10 rejected by the exact nonlinear rollout/corridor
  gate. Solver convergence is therefore achieved for the requested inputs,
  but full-cycle acceptance is not.
- The remaining ten nonlinear rejections cluster near the collision-truncated
  tail and are not to be “fixed” by loosening the exact corridor. The captured
  state sequence is from the previous controller and is not causally replayed
  after the newly converging commands; it cannot tell whether the changed
  solver would avoid entering that tail in a live closed loop.
- The all-input convergence gate was also checked on a new, supervised live
  validation capture, `live_runs/mpc_startup_adaptive_padding_20261020/run`.
  It has 75 valid solver inputs (plus six startup diagnostics rejected before
  a state was available); replay found at least one sub-50 converged pass for
  all `75/75`. R1 converged on `71/75`, R2 on `72/75`, and the maximum final
  ADMM pass was 49 iterations. The runtime logged 74 optimal cycles, one
  degraded cycle, zero hard-fallback cycles, and a 49-iteration maximum.
- That practice validation nevertheless collided before a lap, around
  progress `26 m` / `1.65 s` of vehicle motion. The bag's final valid truth
  speed was about `5.02 m/s`, while the trajectory target had risen from
  `7.09` to `8.88 m/s`; this run does not establish that the requested speed
  is feasible or that the car can safely track this raceline. Its position
  stream also shows an AMCL correction problem: at about `0.8 s`, the logged
  odometry prediction was within roughly `0.03 m` of truth, but a local scan
  cluster of weight `0.706` applied a `0.132 m` correction in the wrong
  direction. The fused map pose then differed laterally from truth by about
  `0.20 m` before impact. This is a strong localization suspect, not proof it
  alone caused the collision. The compact recorder profile previously omitted
  raw AMCL/scan-alignment diagnostics; those topics are now included for the
  next supervised capture.
- Based on that truth-scored correction, raised
  `local_cluster_min_weight` from `0.70` to `0.75`; the separate strong,
  multi-scan recovery threshold remains `0.80`. This rejects the two observed
  low-confidence corrections while retaining the existing recovery route.
  The MPC/solver code and its convergence settings were not changed by this
  localization adjustment.
- The full current ROS workspace compiled successfully in the maintained
  development-entrypoint clean-build path (`sdu_apex_msgs`, planning,
  localization, MPC, and integration packages). A direct ad-hoc colcon attempt
  hit stale generated setuptools metadata; the maintained entrypoint cleaned
  only generated artifacts and then completed. At this checkpoint the next
  simulator run had not started; its result and the later convergence retry
  implementation are recorded in the following section.
- Do not claim that multi-attempt total work is under 50 iterations: the
  contract verified here is that each bag input obtains a converged ADMM pass
  with fewer than 50 iterations. Also do not equate that solver result with a
  safe or accepted command; the exact nonlinear gate remains authoritative.

### Bounded same-sample convergence retry and live recheck — 2026-10-20

- The follow-up `/mpc/diagnostics` bag replay identified one repeatable
  numerical corner in the 52-input capture: input row 6 exhausted all three
  QP corridor-padding attempts in both R1 and R2 at 49 iterations. This was a
  solver convergence miss, distinct from exact nonlinear feasibility.
- Raising the ADMM tolerance from `0.05` to `0.055` made that capture pass in a
  single cycle, but degraded the older 349-input replay from 10 to 15 exact
  nonlinear rejections. That looser tolerance was rejected; production remains
  at `0.05`.
- Implemented a single bounded extra RTI cycle in `mpc_rti_solve_cycle()` only
  when neither R1 nor R2 has a converged sub-50 pass and at least one completed
  ADMM pass missed that gate. The retry uses the same sensor sample/progress
  and the nominal/memory resulting from the first cycle, retains the original
  convergence tolerance and exact nonlinear corridor gate, and receives only
  the remainder of a configured cycle-time budget. Input/factorization/model
  errors do not trigger retries. If the retry fails, an already exact-feasible
  first candidate is preserved. Retry count/status/iterations/time are
  separately exposed in MPC diagnostics and the offline replay CSV.
- Rebuilt the full ROS workspace using the maintained dev-entrypoint build,
  compiled `mpc_bag_solver_replay` against that exact `mpc_core`, and replayed
  all current exports with external replay retries disabled. All `534/534`
  captured solver inputs across the `349`, `75`, `52`, and new `58`-input bags
  have at least one converged ADMM pass in `1–49` iterations at tolerance
  `0.05`. No bag input fails the gate. One input in the newest 58-row replay
  used the bounded same-sample retry (R1 `1`, R2 `25`); one input in the earlier
  52-row replay used it (R1 `37`, R2 `13`). Maximum iterations in any ADMM
  pass are `49`. This does **not** bound aggregate work across QP-padding
  attempts, R1/R2, or the extra RTI cycle to 49 iterations.
- Ran a supervised validation as
  `live_runs/mpc_startup_convergence_retry_20261020/run` with the production
  retry build, tolerance `0.05`, AMCL local-cluster threshold `0.75`, expanded
  compact recording, and the collision watcher. It was described at the time
  as using the practice trajectory/map; a later audit disproved that route
  description. The watcher stopped at the first collision: `66` Unity odometry
  samples over `1.63 s`, zero laps, one collision. See the route audit below.
- The live bag's `58/58` valid control inputs each had a converged R1/R2 pass
  under 50; the actual callback reported zero same-sample retries in this
  particular run. MPC cycle outcomes were 56 optimal, one degraded, and one
  exact nonlinear rejection (plus six startup diagnostics before a legal map
  pose existed). The independent production-core replay of those same 58
  state inputs triggered one retry and also passed `58/58`.
- Separating pre-impact samples from the respawn-contaminated tail gives
  AMCL scan-pose and control-map-pose XY RMSE `0.112 m`, p95 `0.229 m`, max
  `0.244–0.246 m` before collision. Across the whole collision-truncated bag,
  those errors rise to about `0.225 m` RMSE / `0.835 m` max. Only `15/60`
  scan-alignment updates were accepted; applied XY correction was p95 `0.054
  m`, max `0.158 m`. The `0.75` confidence threshold therefore did not make
  localization sufficiently accurate in this short run, and the bag does not
  record cluster confidence itself. Do not attribute the collision solely to
  AMCL from these correlations.
- The early control trace also remains poor: from rest at `t≈0.18 s`, with
  speed reference `7.1 m/s` and `e_y≈-0.18 m`, the first command was steering
  `+0.524 rad` and throttle `0.0097`; the next sample commanded throttle
  `0.001`. At `t≈1.18 s`, truth speed was `3.30 m/s`, AMCL/map pose error had
  reached `0.23 m`, and `e_y≈-0.53 m`; near `t≈1.38 s`, speed was `4.37 m/s`,
  `e_y≈-0.61 m`, and the selected throttle rose to `0.612`. The collision
  followed shortly after. This is evidence of unresolved tracking/pose and
  startup-command behavior, not evidence that the convergence retry failed.
- The collision remains a separate closed-loop/model/localization/trajectory
  problem. The bag evaluator reports truth speed max `4.51 m/s`, forward-speed
  RMSE `1.44 m/s`, relative XY endpoint error `0.90 m`, and no completed lap;
  those errors are collision-truncated and must not be read as a full-lap
  accuracy score. Solver convergence is now gated successfully, but no claim
  of a safe or fast raceline is made. The expanded AMCL diagnostics are in the
  bag for continued diagnosis.
- At completion, the watcher stopped both simulator and runtime; no containers
  are left running. Before another simulator run, replay any newly added bag
  inputs through this production-core version and require at least one ADMM
  pass per input below 50, as above.

### Practice raceline selection correction — 2026-10-20

- The collision-truncated retry bag was not following the intended practice
  raceline. Its MPC reference diagnostics match
  `autodrive_mintime_sim_5p0_dense/autodrive_mintime_raceline.csv` (for
  example, at progress about `23.4 m`, reference position near `(1.429, 1.098)`
  and speed near `7.94 m/s`). That reference lies beyond the practice map's
  occupied boundary in this section. The nearest practice-line point to the
  last pre-impact Unity pose `(1.088, 1.009)` is around progress `20.63 m`, at
  `(0.765, 1.033)`. The previous statement that this was a practice-line
  validation was incorrect. The bag does not encode the loaded Unity scene or
  contact manifold, so the route mismatch is a strong collision explanation,
  not proof of the exact collider/contact that caused it.
- Corrected the development launch default in
  `sdu_apex_autodrive/launch/controller.launch.py` to pair
  `maps/autodrive_practice_20260924_b.yaml` with
  `trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv`.
  Both AMCL's global-heading initializer and MPC receive that same trajectory.
  The competition launch and its configured route were not changed. Explicit
  map/trajectory overrides remain supported as a required pair.
- Extended `tools/export_mpc_bag_solver_inputs.py` with route-aware export:
  it reprojects each saved command-time map pose onto the selected trajectory
  using the MPC's full-track, forward-heading-gated segment projection, then
  writes the matching progress, lateral error, and heading error while
  retaining the synchronized sensor/plant state. This avoids replaying old
  Frenet coordinates as if they belonged to the newly selected line.
- Re-exported the three recent practice-route captures and replayed every
  available valid input against the corrected practice raceline using the
  production MPC core and the existing candidate solver settings. All
  `185/185` inputs passed the requested gate: at least one R1/R2/internal-retry
  ADMM pass converged in `1–49` iterations. Route projection max distances
  were `0.517 m` (75-row adaptive-padding capture), `0.402 m` (52-row AMCL
  capture), and `0.221 m` (58-row retry capture), all below the configured
  `0.80 m` startup projection limit. The older 349-row capture projects at
  median `0.923 m`, p95 `4.483 m`, max `4.562 m`; it is not evidence from this
  practice route and was excluded from this route-specific gate, not deleted.
- Ran the corrected pair in the established practice `-batchmode`/Xvfb path,
  with mounted-source rebuild, the convergence candidate overlay, full bag
  recording, and the collision/stall watcher. Startup logs independently
  confirm that `map_server` loaded the practice map and AMCL/MPC loaded the
  same practice CSV (1422 CSV rows; MPC's trajectory preparation retains 1421
  points after its duplicate-closure handling). The run is saved at
  `live_runs/mpc_correct_practice_raceline_20261020/run`.
- This was not a completed lap: the watcher observed no collision-count
  increment, but aborted after odometry stopped moving for its 20-second stall
  window. The bag has 1094 Unity odometry samples at mean `25.03 ms` spacing,
  zero laps, and collision count zero. The car moved along the route from
  projected progress `18.50 m` to about `35.96 m` (17.46 m), then remained at
  approximately `(-0.942, -1.461)` with zero speed for the remainder. Peak
  Unity speed was `6.06 m/s`. Do not label this as a successful or collision-
  free lap; it is a route-correct partial run that stalled without a recorded
  collision event.
- The per-cycle diagnostics show `737` valid solves: `159` optimal,
  `57` degraded, and `521` rejected by the exact nonlinear corridor check;
  six earlier diagnostics were rejected before a synchronized state existed.
  Every one of the `737/737` valid inputs had at least one converged R1/R2
  ADMM pass below 50 iterations (maximum `49`, no internal retry). Thus the
  saved-input convergence gate passed on the correct line, while nonlinear
  candidate acceptance/tracking remains the immediate blocker. The watcher
  stopped the simulator, runtime, and recorder; no active containers remain.
- At the terminal stopped state, progress was `35.9606 m`, `e_y=-0.274816 m`,
  and both candidates failed nonlinear corridor validation at stage 0. The
  tolerance-inclusive lower bound was `-0.274649 m` (about `0.17 mm` outside);
  the 5 cm footprint-clearance bound before tolerance was `-0.269649 m`, so
  the state was about `5.17 mm` beyond that stricter bound. Recovery was active
  but reported no re-entry stage, no candidate was selected, and the published
  fallback was zero steering/zero throttle (Unity's hard-brake command). This
  explains why the car remained stopped; do not “fix” it by relaxing the
  physical-clearance requirement without a separate measured safety analysis.

### RTI-versus-tracking diagnosis — 2026-10-20

- The machine was idle before this analysis; no simulator was launched. The
  route-correct partial capture
  `live_runs/mpc_practice_recovery_multilap_retry1_20261020/run` contains 748
  valid command-time states and did not complete a lap. It progressed into the
  sharp-turn region, then stalled around `s=16.7 m`, with no recorded
  collision. This is a failed tracking run, not an MPC success.
- The turn failure starts before the stall: at `s=14.16 m`, the measured state
  was approximately `u=3.38 m/s`, `r=-1.55 rad/s`, `e_y=0.26 m`,
  `e_psi=0.24 rad`; by `s=15.21 m`, `e_y=0.71 m`; by `s=15.97 m`,
  `e_y=1.06 m`. The practice trajectory requests roughly `7.5–8.1 m/s²`
  (`v_ref² * |kappa|`) in this section. Measured `|u*r|` was about `5.2`
  at 14.16 m, `3.2` at 14.57 m, and `1.3` at 15.21 m. This establishes a
  severe turn-response/trajectory-feasibility mismatch at the point where
  tracking is lost; it does not, by itself, distinguish tire-model error from
  actuator timing, state error, or an infeasible path/speed combination.
- An exact replay of the 748 saved states with the current production core,
  49-iteration cap, and R2 mode had at least one sub-50 converged pass for
  `746/748` rows. It still had `260/748` nonlinear corridor rejections. Over
  `s=16–19 m`, `259/439` rows produced no selected candidate and `156/439`
  selected the hard-brake recovery; mean selected throttle in that interval
  was `0.004`. The repeated braking is therefore a visible response to lost
  feasibility, not evidence that more ADMM iterations are needed.
- A replay that multiplied lateral- and heading-error weights by four changed
  steering/throttle commands, but left the sharp-turn stall/rejection count
  essentially unchanged. This does not prove weights are irrelevant; it does
  show that a broad weight increase alone did not repair this failure. The
  base profile already assigns `500` to under-speed error, `125` to lateral
  error, `31.25` to heading error, and only `0.1` to absolute throttle effort.
- A third SQP/RTI pass on the same capture did not lower the 261-rejection
  result in its prior replay and did not increase normal R1/R2 selections.
  Removed that third-pass line-search/relinearization branch rather than
  adding iterations without tracking benefit. The normal two-pass RTI remains.
- Runtime diagnostics previously mislabeled selected feedback-drive and
  feedback-brake recovery candidates as `"none"`, and health counters grouped
  them under generic degraded acceptance. They now expose the exact candidate
  name, whether it is an R1/R2 optimizer output, and whether a recovery action
  was published; periodic health logs count optimizer, corridor-repair, and
  feedback-recovery selections separately. This is an observability fix, not
  a driving improvement.
- Fixed the ADMM scalar-to-per-coordinate penalty migration so initialized
  scalar `rho`/`rho_u` values are used when the newer penalty arrays are empty,
  matching the documented migration and the focused solver regression. This
  did not change the fixed-capture outcome because that replay cold-started.
- Verification after these edits: `mpc_core`, `mpc_controller_node`, and the
  focused source-rollout/RTI test target compiled; all 31 tests in that target
  passed; `git diff --check` passed. The saved-input replay is diagnostic only;
  no simulator run has verified a behavior change. The unresolved work is to
  reduce the measured yaw/lateral response mismatch in the sharp turn and make
  ordinary R1/R2 candidates feasible there before another practice attempt.
- A previously skipped fixed-input report was rerun after sourcing the local
  ROS message overlay. Persistent reports are
  `live_runs/mpc_practice_recovery_multilap_retry1_20261020/fixed_input_rollout_report.txt`
  and `oracle_start_rollout_report.txt`. The first starts from logged plant
  state and replays recorded slot-1 commands without future state correction;
  it is the relevant plant rollout score, unlike the separately reported
  closed-loop horizon samples that are confounded by later replans.
- Fixed-input rollout over 1,483 windows: at 25 ms, `u` RMSE `0.0948 m/s`,
  `v` `0.0163 m/s`, yaw-rate `0.1170 rad/s`, yaw `0.00179 rad`, and XY
  `0.00264 m`; at 750 ms, `u` `0.0930 m/s`, `v` `0.0174 m/s`, yaw-rate
  `0.1301 rad/s`, yaw `0.0205 rad`, and XY `0.0676 m` (p95 `0.1133 m`, max
  `0.6192 m`). The complete 750 ms path stays within 5 cm for `1,346/1,483`
  windows; `137` windows exceed the 5 cm bound. This is useful predictive
  ability, but does not meet the requested worst-case 5 cm criterion.
- Errors are regime-concentrated, not uniform: fixed-input one-step yaw-rate
  RMSE is `0.249 rad/s` for the high-steer group and `0.318 rad/s` during
  high-steer turn-in; the alternate command-queue phase scores `0.459` and
  `0.514 rad/s`, respectively. One-step position remains small because it
  integrates over only 25 ms; that must not be mistaken for accurate yaw-rate
  dynamics. The run barely samples higher speed (truth max `6.15 m/s`, with
  only two 6–8 m/s samples in the controller horizon score).
- The diagnostic-only oracle-start replay replaces only initial `u/v/r` with
  same-epoch Unity truth; future inputs and hidden states remain as recorded.
  At 750 ms it improves XY RMSE from `0.0676` to `0.0435 m` and p95 from
  `0.1133` to `0.0716 m`; full-path >5 cm windows drop from `137` to `103`.
  But yaw-rate RMSE changes only from `0.1301` to `0.1282 rad/s` at 750 ms
  (`0.1170` to `0.1113` at 25 ms). Thus initial velocity-state error explains
  part, not most, of the yaw-response tail; a production yaw/steering/tire
  response discrepancy remains.
- The live run also missed 40 Hz control cadence: `/mpc/diagnostics` and
  `/mpc/drive_command` each recorded only `748` messages against `1,513`
  40 Hz odometry updates. Parsed `solve_us` is mean `46.0 ms`, p50 `43.3 ms`,
  p95 `76.1 ms`, max `97.9 ms`; `564/748` solves exceeded the 25 ms period.
  Command intervals were median `44.2 ms`, p95 `76.6 ms`. More RTI passes
  would worsen this. On the same saved states, an R1-only cold replay had
  `23.7 ms` p95 solve time but the same `260` nonlinear rejections and fewer
  normal optimizer selections (`245` versus `274` in the two-pass replay).
  R2 improves some candidate quality, not feasibility or cadence; a third pass
  was not retained.
- The correct conclusion is neither “weights alone are wrong” nor “RTI is
  unsolvable”: the present model has low typical translation error but a
  significant dynamic yaw tail, the sharp-turn path asks for about
  `7.5–8.1 m/s²` lateral acceleration, R1/R2 fail the exact corridor often,
  and the live solve is too slow to update at 40 Hz. Keep active braking only
  as a fail-safe, but count corridor repair, feedback recovery, and no-candidate
  cycles as failed tracking attempts—not successes. No further practice run is
  justified until the next candidate reduces these measured failure modes.

### Frozen 30-command MPC rollout audit — 2026-10-20

- Corrected the interpretation of the `fixed_input_rollout_report.txt` above:
  it restarts from many logged states and replays the commands that were
  actually emitted by the *closed-loop, replanning* controller. It is a useful
  source-plant response check, but it is not the accuracy of one MPC plan whose
  30 commands were latched and executed unchanged. It cannot substantiate a
  `<5 cm` maximum error for a planned MPC horizon.
- Re-evaluated the dedicated frozen-plan bags from
  `live_runs/mpc_horizon_20261018_16` through `_22`. The latest route-correct
  practice retry bag, `mpc_practice_recovery_multilap_retry1_20261020/run`,
  does not contain `/mpc/open_loop_test`; it has no frozen-plan score.
- Runs `_21` and `_22` are the complete, auditable plan tests. Each recorded a
  30-control plan, output stages `0..29` in order, and the published
  `DriveCommand` values matched the latched controls exactly. The corresponding
  normalized `/steering_command` and `/throttle_command` actuator-interface
  topics also matched all 30 controls (maximum absolute value discrepancy
  below `3.1e-8`; nearest publication within `9.3 ms`). Their output cadence was
  about `25.0 ms`. Neither had a new collision during the 750 ms frozen
  horizon; both later recorded a collision after that horizon and completed
  zero laps. These topic comparisons verify the MPC-to-actuator-interface
  sequence, not a per-packet Unity acknowledgement; old bridge telemetry does
  not provide a populated applied-command sequence for these runs.
- Run `_21`, latched at `2.521 m/s`, includes all 31 predicted states and
  truth-aligned states from `0` through `750 ms`: XY RMSE `0.1484 m`, p95
  `0.2586 m`, max `0.3106 m`; yaw RMSE `0.0778 rad`, max `0.1385 rad`; forward
  speed RMSE `0.1757 m/s`, max `0.2478 m/s`; yaw-rate RMSE `0.3865 rad/s`, max
  `0.9830 rad/s`.
- Run `_22`, latched at `2.500 m/s`, also has all 31 horizon states: XY RMSE
  `0.2892 m`, p95 `0.5711 m`, max `0.7381 m`; yaw RMSE `0.1724 rad`, max
  `0.3367 rad`; forward-speed RMSE `0.1697 m/s`, max `0.5772 m/s`; yaw-rate
  RMSE `0.7783 rad/s`, max `1.8570 rad/s`.
- Separating initial pose error from rollout drift does not rescue the result.
  The start XY mismatch is `3.78 cm` in `_21` and `1.84 cm` in `_22`, with
  negligible initial yaw error (`0.00065` and `-0.00049 rad`). After rigidly
  aligning only the initial pose for diagnostic decomposition, relative-path
  XY max remains `27.22 cm` / `72.67 cm`; relative-yaw max remains
  `0.1379` / `0.3372 rad`. The fit-to-start transform is for scoring only and
  is not applied to the controller.
- Stagewise comparison localizes the first large prediction error to yaw
  response: `|r_pred-r_GT|` first exceeds `0.1 rad/s` at 25 ms in `_21` and
  50 ms in `_22`; at 100 ms the errors are `0.711` and `1.146 rad/s`. Relative
  XY first exceeds 5 cm at 225 ms and 250 ms, respectively. The small initial
  pose offsets do not explain the accumulating path error.
- Both complete plans also expose steering chattering: all 30 throttle outputs
  are `1.0`; each plan reverses steering sign nine times, with maximum adjacent
  target changes of `0.756 rad` and `1.047 rad` per 25 ms. The test profile
  used absolute steering-target effort `0.1`. The QP cost penalizes absolute
  target/reference error and absolute target magnitude, but has no
  adjacent-target slew cost or constraint. This is a concrete objective-design
  weakness to address; smoothing alone cannot repair the early yaw-rate
  prediction error and is not a plant-model fix.
- Run `_18` emitted all 30 commands at `6.567 m/s`, but its plan event did not
  serialize the full predicted state array; only 17 sparse state observations
  align to truth. Its sparse position max is `0.636 m`, but it is not a
  complete 31-state horizon score. Run `_16` aborted at stage 16 with
  `state_too_old` and has no recorded output events, so it is not a completed
  plan test.
- Therefore no existing fully scored frozen-plan execution supports the
  earlier `<5 cm` statement. The two complete full-horizon tests fail that
  bound substantially. They also latch at only `2.5 m/s`; full-speed horizon
  accuracy remains unverified. These are genuine model/controller/command-path
  mismatches, but the bags do not by themselves separate plant-model error
  from low-level command application timing.
- Corrected the iteration-count interpretation for the route-correct live
  capture. `748/748` had at least one R1 or R2 ADMM pass converge below 50; that
  is not a total-cycle `<50` guarantee. Summing R1 and R2 ADMM iterations over
  their QP-padding attempts gives p50 `57`, p95 `146`, max `245`, with
  `398/748` cycles over 49 total iterations. There were no whole-cycle retries
  in this run; up to five QP attempts occurred across R1+R2. Thus both timing
  and total work fail the stated target despite the earlier per-pass gate.
- Code inspection confirms R1/R2 are sequential SQP/RTI linearizations, not
  alternative controllers: R2 normally relinearizes around R1's candidate;
  when both pass exact nonlinear validation, the lower nonlinear objective
  wins (corridor slack breaks ties). On the saved route inputs, R2 increased
  ordinary optimizer selections but left nonlinear corridor rejections at
  `260/748`; R1-only replay had the same rejection count and fewer ordinary
  candidates. Deleting R2 or increasing ADMM iterations is therefore not a
  demonstrated fix. The next design needs feasibility-aware refinement and one
  aggregate iteration/time budget. Non-optimizer safety actions must fault and
  stop the run rather than be reported as successful MPC output.
- Tightened `tools/evaluate_mpc_bag.py` so frozen-plan reports now filter
  completion/abort events by plan ID, compare every emitted stage and command
  against the latched plan, compare the actuator-interface command topics,
  flag collisions inside the frozen horizon, and report the number/time span
  of truth-aligned predictions, and separate initial-pose error from
  rigidly start-aligned relative-path error. This prevents a closed-loop replay
  or partial sparse record from being reported as a complete fixed-horizon
  validation.
  Python syntax validation and rescoring runs `_16`, `_18`, `_21`, and `_22`
  pass; only `_21` and `_22` qualify as complete full-state frozen-horizon
  scores, and neither meets 5 cm.
- No production MPC behavior was changed in this audit and no simulator was
  launched. The next controller change must target the measured horizon yaw
  and position mismatch and the observed nonlinear corridor rejection, then
  pass this frozen-plan score before another closed-loop race attempt. The
  earlier recommendation to retain an on-car hard-brake fallback is superseded
  by the user's stated preference: for supervised simulator experiments, stop
  the simulator on collision/fault rather than hold a hard brake and wait. Any
  recovery output must remain separately labeled and must not count as normal
  optimizer success.

## Exact-build Unity/PhysX reverse engineering — 2026-10-20 continuation

- Continued targeted decompilation of the matching practice `UnityPlayer.so`
  debug ELF (same Build ID), using Ghidra `-noanalysis`; no simulator was
  launched. Added native targets for automatic Rigidbody mass distribution,
  Unity COM/tensor setters, PhysX mass-property routines, wheel sprung-mass
  reset/set paths, `PxVehicleComputeSprungMasses`, and the vehicle wheel-ray
  prefilter. The target list is maintained in
  `tools/unity_reverse_engineering/ExportWheelCollider.java` and detailed
  findings are in
  `docs/development/UNITY_PRACTICE_WHEELCOLLIDER_REVERSE_ENGINEERING_20261020.md`.
- Reverse-engineered the exact mass path: Unity enumerates Rigidbody shapes
  and calls `PxRigidBodyExt::setMassAndUpdateInertia`; the checked-in PhysX 4.1
  implementation aggregates per-shape mass properties, scales to 3.47 kg,
  applies the explicit/automatic COM, diagonalizes the tensor, then writes the
  actor mass frame. The active prefab has automatic inertia enabled and the
  C# controller sets COM `(-0.00008, 0.06434, -0.00468)`.
- Independently solved the exact PhysX four-wheel minimum-variance sprung-mass
  constraints from the scene wheel coordinates. Results
  `[0.817194701, 0.816018680, 0.918992382, 0.917794237] kg` reproduce the
  production constants to float-rounding precision. Added source-provenance
  comments only; no vehicle equations or output behavior changed.
- Honest validation remains poor for open-loop rollout: current shared plant
  evaluation on the available Oct 19 practice captures has one-step `v` RMSE
  about `0.132 m/s` and yaw-rate RMSE `0.066–0.072 rad/s`; at 750 ms XY endpoint
  RMSE is about `1.38 m`, and only about `100/780` rolling windows stay within
  5 cm. Those bags predate the current source revision, and this is offline
  replay, not an online validation. The newly traced mass and sprung-mass
  equations are already represented by fixed practice-vehicle constants and
  do not explain or resolve the remaining error by themselves.
- Also traced the native suspension query: COM-pose ray origin, direction,
  `3*radius + maxCompression + maxDroop` length, hit-plane intersection, and
  Unity's ignore/layer prefilter. Our plant already implements the exact
  ray-to-plane intersection when the plane is known. It does not yet reproduce
  hit selection, per-triangle normals/materials, or dynamic scene actors from
  the practice scene. At the time of the initial audit, the replay had not
  causally identified the residual and no change was warranted. The later
  2026-10-20 continuation below found and corrected a separate exact callback
  equation mismatch; the contact-query gaps remain.

## 2026-10-20: corrected native combined-slip equation and replay

- Continued from the 750 ms XY endpoint error (~1.38 m) by inspecting the
  exact-build `VehicleComputeTireForce` decompilation and its native register
  dataflow. The prior implementation/report had inserted filtered tire load
  into the normalized lateral combined-slip demand. The Unity callback instead
  computes `sqrt(max(1, (rawFx/forwardExtremumValue)^2 +
  (rawFy/sidewaysExtremumValue)^2))`; filtered tire load multiplies each
  resulting force only. This was a concrete source/implementation mismatch,
  not a fitted correction.
- Removed the extra load factor in the production-shared plant and changed the
  callback-equation unit test to enforce the decompiled Unity expression.
  Focused `test_unity_source_plant` passes. Rebuilt `mpc_core`,
  `mpc_controller_node`, and `sensor_odometry_node`, so the equation is in
  both production paths and not evaluator-only.
- Replayed both available independent practice captures, using their recorded
  initial plant state and the recorded delayed slot-1 commands without future
  state correction. At 750 ms:
  - `run/`: `u RMSE=0.00521 m/s`, `v=0.00038 m/s`, `r=0.00300 rad/s`,
    heading `0.00135 rad`, XY endpoint RMSE `0.00570 m`.
  - `run_1/`: `u RMSE=0.00678 m/s`, `v=0.00069 m/s`, `r=0.00558 rad/s`,
    heading `0.00188 rad`, XY endpoint RMSE `0.00787 m`.
  The previous ~1.381 m endpoint RMSE fell by ~99.5% on both. Thus the stated
  `<5 cm XY RMSE` target is met in both offline practice replays.
- Keep the maximum separate from RMSE: all 781 endpoints in `run/` are under
  5 cm; `run_1/` has one endpoint at 50.31 mm and two intermediate path samples
  above 5 cm (maximum 53.7 mm). In those tail windows, the predicted plant
  increments agree with GT twist to sub-millimeter scale, while some recorded
  GT pose/twist increments differ by 2.9–3.8 cm in one 25 ms interval. Do not
  hide these tails or compensate with a force gain; they are pose/twist
  reference-consistency events and need a separate synchronization audit if
  the requirement is a strict per-sample maximum rather than RMSE.
- No live simulator was launched. The bags are practice-only and the result is
  offline replay using corrected current source, not online deployment proof.
  The exact native callback correction is supported by the matching Unity
  binary; arbitrary track-mesh contact selection and the full Unity scene
  constraint solve remain outside this reduced plant.

### MPC re-entry diagnosis after the plant correction — 2026-10-20

- Replayed the latest route-correct practice capture's 748 saved solver inputs
  through the rebuilt current `mpc_core` and the same startup-candidate
  settings. No simulator was started. The current source plant produced 259
  nonlinear corridor rejections, versus 261 in the saved earlier replay; this
  is not a material MPC improvement. R1/R2 still converge on many of these
  samples but neither supplies an exact-validated candidate through the
  failing section.
- The 259 rejections are localized to a short stalled/recovery region. Every
  failure is a corridor failure at the early horizon (R1 around stage 4; R2
  around stage 6), with approximately 8–12 mm excess beyond the scheduled
  normal-clearance bound. The recorded state is near stopped (about
  `0.024 m/s`) and already outside the nominal 5 cm-clearance corridor. The
  diagnostic reports `recovery_active=true`, no sustained re-entry found
  within the horizon, and an initial normal-corridor violation of about
  `7 mm` (the seed reaches about `18 mm`). This points to recovery/re-entry
  feasibility—not a failure to run another ADMM iteration—as the immediate
  reason these cycles cannot publish a normal candidate. The logged failure
  bounds are the normal-clearance bounds; the next check must distinguish the
  recovery allowance from the hard full-footprint track boundary before any
  schedule change is made.
- The relevant offline MPC replay utility no longer compiled because it still
  wrote two yaw-residual fields removed from the production state. Updated it
  to ignore those retained legacy CSV columns and rebuilt it against the
  current production core. This only restores the ability to replay saved
  solver inputs; it does not change runtime MPC behavior.
- Accuracy is therefore conditional, not globally accepted: the two Oct 19
  practice replays meet the 750 ms endpoint-RMSE target, but the newer Oct 20
  aggressive practice capture remains at `0.0676 m` endpoint XY RMSE,
  `0.1133 m` p95, and `0.6192 m` max; `130/1,483` endpoints exceed 5 cm and
  `137/1,483` windows exceed 5 cm somewhere in the 750 ms path. That capture
  also has `0.130 rad/s` yaw-rate RMSE at 750 ms. Do not call the plant
  accurate across the full observed practice range or use this as evidence
  that the MPC is ready for a speed-seeking run.
- Next: inspect the recovery schedule and exact hard-footprint bound at these
  failing samples, then make one recovery-feasibility correction only if the
  source-model rollout remains inside the hard track boundary. Replay the
  same saved inputs and require fewer exact-corridor rejections without
  increasing hard-bound violations; only then consider a supervised sim run.

### Rollout time-origin discrepancy — 2026-10-20

- The controller previously formed its optimization initial state at the
  callback's `control_ros_time`, then ran the solver and published the first
  command afterward. The vehicle plant already propagated its two command
  queue slots, but this omitted the pre-publication interval while the old
  queued command remained active.
- The two auditable frozen-plan bags show this interval is material: the first
  output was published about `18 ms` after the plan's control timestamp, while
  the next bridge requests were about `20 ms` (`_21`) and `23 ms` (`_22`) after
  that timestamp. In both recordings the first output missed that request and
  entered the recorded queue on the following request. An offline timing sweep
  moved the first planned-control effect later and improved both position
  scores, but did not reach the 5 cm worst-case target. This supports a timing
  contribution, not a complete explanation of the rollout error.
- The MPC now uses the previous cycle's measured ROS-stamp interval from its
  control snapshot to the command publication as the next cycle's prediction
  lead. The control-time plant predictor advances known queued inputs to that
  expected publication epoch before optimization; the MPC decision still
  enters through the existing two-packet queue. No physics constants, tire
  equations, or optimizer states were changed. The first command after startup
  has no measured previous-cycle lead and therefore starts with zero lead.
- The lead is emitted in optional MPC diagnostics as
  `candidate_command_stamp_ns` and `candidate_command_lead_s`. Open-loop plan
  metadata now uses the same expected command-publication epoch as its initial
  predicted state.
- Validation: `colcon build --packages-select f1tenth_mpc` succeeded; all
  eight `test_mpc_control_time_predictor` tests passed, including a new
  47 ms future-publication-epoch queue-propagation case. No simulator was
  started in this step.
- Still unresolved: command transport/packet phase is separate from solver
  time. In the two frozen-plan records, a command published just before a
  request was received just after it and therefore waited for the next packet.
  The production MPC does not subscribe to the diagnostic bridge-timing topic;
  it must infer the permitted command/packet timing from legal state and
  command history or preserve this as an uncertainty. A new frozen-plan
  practice run is required to measure whether the production change improves
  the full rollout and to isolate this remaining phase error. Do not call the
  5 cm full-horizon bound passed yet.

### Correct-format v300 trajectory attempt — 2026-10-20

- Replaced the earlier geometry-only reconstruction with the user's exact
  `v300_optimized_MPC_AMCL_raceline.csv`, copied byte-for-byte into the
  practice trajectory directory. The runtime loaded all 1,024 points and the
  supplied velocity, acceleration, and corridor fields. SHA-256:
  `2874816456f7209b52ea359458c8fdaf6fb51bd4fa10781671d3e3483b06bb3d`.
- Ran the exact candidate in the established practice batch-mode simulator
  procedure with runtime topic policy passing and a collision watcher. It
  collided at about 5.11 s before completing the first lap. The exact-run bag
  is `live_runs/mpc_v300_exact_r2reject_practice_1lap_20261020/run`. The prior
  geometry-only attempt is not counted as validation of this supplied CSV.
- In the exact bag, the first non-optimal MPC cycle occurred around 1.50 s;
  an R1/R2 residual rejection took ~100 ms and produced ~90 ms command lead.
  At ~3.87 s cross-track error was ~0.51 m, both nonlinear checks rejected,
  the cycle took ~201 ms with ~191 ms command lead, and collision followed at
  ~5.11 s. Diagnostic cycle timing was median/p95/max ~12.7/199/234 ms, with
  30/104 cycles over 25 ms. AMCL scan XY error was 10.5 cm RMSE (17.7 cm p95).
- Exact-file offline solver replay on the 163 pre-collision GT-pose/production-
  plant inputs found that `r2_on_rejection` reduced nonlinear-rejected samples
  from 58 (R1-only) to 33 and increased optimal/degraded outcomes, but cycles
  over 25 ms rose from 30 to 46 and the maximum solver time rose from 45.5 to
  68.3 ms. This is not closed-loop proof or a reason to promote R2; neither
  mode is validated to drive the complete candidate lap.
- No trajectory physics or production defaults were changed based on this
  single collision. The next step is code-level diagnosis of the repeated
  solver work/rejected-cycle command behavior and the separate AMCL pose
  error. Keep both diagnostic bags. Do not start another simulator run until
  one specific correction has offline evidence; retain immediate collision
  abort behavior.

### v300 native-controls path-follower experiment — 2026-10-20

- Imported the exact 11-column user file, preserving the comment header and
  all supplied values. It matches the nine geometry/speed/bounds columns of
  the correct v300 trajectory. Added direct parsing for this format to the
  development follower, diagnostics for the native steering reference, and a
  repeat-count launch setting. The follower uses native throttle indexed by GT
  path progress; GT does not enter production odometry or MPC.
- First practice batch-mode attempt used existing GT pure-pursuit steering
  and the CSV throttle. It reached 7.24 m/s and advanced 25.37 m along the
  path, but collided before a complete first lap. Cross-track distance first
  exceeded 0.25 m at ~5.56 s and 0.50 m at ~5.76 s on the tight turn. At that
  point pure-pursuit steering was ~-0.454 rad versus the file's native-applied
  steering reference ~-0.229 rad, and pure pursuit reached its -0.524 rad
  limit. Bag: `live_runs/v300_native_controls_pp_5lap_20261020/run`.
- A second attempt replayed the provided native-applied steering and throttle
  by GT path progress. It collided early (~1.5 s bag span, 5.34 m/s max);
  cross-track error exceeded 0.50 m after ~1.25 s of motion. Bag:
  `live_runs/v300_native_controls_replay_5lap_20261020/run`. This column is
  not established as a directly commandable steering profile; direct replay
  is not retained as the follower's default.
- The initial follower version used GT pure-pursuit steering plus CSV throttle
  and logged native steering only as a reference. Raw native-steering replay
  was also attempted, but is not retained as a policy; both bags remain
  diagnostic.

### v251 CSV continuous-follow capture — 2026-10-20

- Ran two repeated five-lap practice captures with the development-only
  ground-truth pure-pursuit/CSV-throttle follower, MPC disabled. Each reached
  lap count 5 with zero collisions. Across ten simulator-reported laps, mean
  was `9.721 s`, best `9.649 s`; this is data collection, not MPC evidence.
- Both full bags and detailed tracking/plant-replay metrics are recorded in
  [CSV_V251_PRACTICE_FOLLOW_20261020.md](CSV_V251_PRACTICE_FOLLOW_20261020.md).
  The source-plant 750 ms endpoint XY RMSE was `2.20–2.53 cm`, but 64–67 of
  roughly 1,850 endpoints per run exceeded 5 cm; worst anywhere along the
  750 ms path was `10.1 cm` and `34.4 cm`. Thus the strict 5 cm path-error
  target remains unmet.
- Current full-data bags are `live_runs/csv_v251_gt_5lap_run1_20261020/run`
  and `live_runs/csv_v251_gt_5lap_run2_20261020/run`. Removed the 451 previous
  ROS bag payload/metadata files (2.50 GiB across 225 bag directories) from
  `live_runs`; verified that only these two bag captures remain. Non-bag
  analysis files were not part of the requested bag cleanup.

### v300 follower lookahead follow-up — 2026-10-20

- Added a development-only `SDU_APEX_CSV_LOOKAHEAD_SPEED_GAIN_S` launch
  setting for the CSV follower; the default remains `0.12 s`. Offline
  recomputation on the first capture suggested `0.60 s` would moderate the
  high steering saturation, so a watched practice run tested it. That bag
  reached `7.24 m/s` but collided at the first closing turn near CSV progress
  `26.8 m` (cross-track about `0.35 m`), before lap completion:
  `live_runs/v300_native_controls_pp_l060_5lap_20261020/run`.
- A `0.30 s` compromise attempt was not scoreable: its first bagged truth was
  already in motion at a different pose and its collision count was already
  1; the 125 ms bag contains no CSV follower diagnostics:
  `live_runs/v300_native_controls_pp_l030_5lap_20261020/run`. Do not interpret
  it as a collision caused by that gain. Later captures enabled the built-in
  development reset and began at the expected saved spawn.
- No successful lap or lap time was produced by these v300 attempts. The
  original `0.12 s` run advanced farther than the valid `0.60 s` run; no gain
  is promoted. Full details and per-capture interpretation are in
  [CSV_V251_PRACTICE_FOLLOW_20261020.md](CSV_V251_PRACTICE_FOLLOW_20261020.md).

### v300 spatial control handoff — 2026-10-20

- The user clarified that pure pursuit should first bring the vehicle onto
  the path, then the CSV's interpolated native steering at the closest
  GT-projected station should be used. The development follower now follows
  that policy: native steering activates at cross-track `<=0.05 m` and
  heading error `<=0.05 rad`; it returns to PP when either exceeds `0.10`.
  CSV throttle is always interpolated at current projected progress; the
  lookahead station is used only by PP. Diagnostics include native-control
  mode. This is dev-only; GT never enters production odometry or MPC.
- With the built-in reset held during practice simulator startup, the vehicle
  began at the expected pose and reached `7.26 m/s`. It briefly entered native
  control near progress `21 m`, then left as tracking error exceeded the
  release thresholds, returned to PP, saturated steering, and collided at
  about `3.42 s` near progress `29.9 m` with about `0.84 m` path error. No lap
  completed. Bag: `live_runs/v300_native_controls_handoff_5lap_20261020/run`.
- A separate valid test of continuous native steering feed-forward plus PP
  deviation correction (not the acquire/release handoff) reached `7.26 m/s`
  and collided at about `3.76 s`, near progress `32.3 m`. No lap completed.
  Bag: `live_runs/v300_native_controls_ff_pp_5lap_20261020/run`. Neither
  policy has demonstrated a complete lap or lap time; no faster result is
  validated.

### Faster v300 bag and MPC/rollout diagnosis — 2026-10-20

- A later speed-governed GT path-following run completed one practice lap with
  zero collisions and peaked at `7.9565 m/s`:
  `live_runs/v300_native_controls_speed_governor_20261020/run`. This is a
  successful data capture, not an MPC-controlled lap or sub-5-second result.
- Path lookup is continuous GT projection onto a locally searched path
  segment, monotonic in station, followed by interpolation. It is not a
  discrete nearest-row lookup. PP handles tracking correction, while native
  controls are blended according to cross-track/heading agreement and sampled
  at current projected progress plus the configured `50 ms` command-delay
  distance. In this capture, full native-control weight occurred on 19% of
  ticks; the median weight was 0.666. The prior hard handoff attempt collided,
  so it is not being silently substituted for the current policy.
- On 850 one-step windows, the queue-phase A/B found yaw-rate RMSE `0.0404
  rad/s` for steering slot 2 plus throttle slot 1, versus `0.1126 rad/s` for
  steering slot 1 plus throttle slot 2. This does not prove the runtime phase.
  Replay of bridge request-stamped commands with carried source state had
  `0.0758 rad/s` yaw-rate RMSE. The 750 ms source-plant XY endpoint RMSE is
  `6.47 cm`, p95 `13.41 cm`, maximum `26.30 cm`; only `444/764` endpoints
  stayed within 5 cm. Error accumulates primarily through yaw/heading
  prediction, with the largest growth in the first 25–75 ms at roughly
  `4.5–4.7 m/s` and `0.14–0.16 rad` steering. This is still outside the
  agreed accuracy bound; no accurate-rollout claim is made. A future-steering
  feedback oracle reduces 750 ms XY RMSE to `4.34 cm` (p95 `8.49 cm`, path
  max `13.34 cm`), still outside the gate. Steering-actuator prediction is a
  contributor, but not the sole remaining mismatch.
- GT-seeded offline MPC replay on 854 bag points yielded 25 nonlinear rejects
  with R1 alone, 2 with R2 every cycle, and 1 with R2 only after R1 rejection.
  The remaining conditional-R2 rejection violates the nonlinear corridor by
  about 1.1 cm at stage 21; R2's p95/max cycle time is `25.97/62.62 ms`.
  This replay is an offline solver diagnostic—not online MPC validation.
- Detailed capture, projection policy, rollout partitions, replay results,
  and reproduction artifacts are recorded in
  [CSV_V251_PRACTICE_FOLLOW_20261020.md](CSV_V251_PRACTICE_FOLLOW_20261020.md).
  No production plant/MPC default or trajectory was changed. Continue with
  the demonstrated yaw/steering-transition mismatch and the single nonlinear
  feasibility rejection before a supervised MPC practice run.

### Update: repeated faster-profile bags and odometry torque feedback — 2026-10-20

- Two independent five-lap practice bags now exist for the 85%-scaled v300
  speed/throttle profile: `live_runs/v300_native_controls_speed85_5lap_run1_20261020/run`
  and `...run2...`. Both completed five laps with zero collisions; valid-lap
  means were 7.663 s and 7.674 s. These used the development GT path follower,
  not MPC, and are not a sub-5-second result.
- Production source-derived odometry now uses fresh existing throttle
  feedback to represent torque during the elapsed packet interval while
  retaining the requested-command delay queue for future prediction. Unity
  source confirms the feedback is normalized `DriveTorque/MotorTorque`; zero
  enters the configured full-brake branch. No topic was added.
- Identical offline replay on both bags shows the feedback-conditioned
  750 ms endpoint XY RMSE decreases from 8.33 to 5.94 cm and 7.97 to 5.43 cm;
  the 5 cm threshold is still missed by many endpoints. This is a useful
  odometry update, not evidence that the open-loop MPC rollout is accurate.
- The localization package builds, and all 8 focused source-plant runtime
  tests pass. A post-change practice run was then captured at
  `live_runs/v300_native_controls_speed85_5lap_post_odom_20261020/run`; it
  reached 3 laps with zero collisions before the GT-stale watchdog stopped it.
  It is partial, not a completed five-lap validation. The bag's bridge latency
  was abnormally high (median/p95/max 19.84/66.56/261.94 ms), and the 750 ms
  feedback replay did not confirm a clean gain; worst path error regressed.
  The transport outlier makes that run inconclusive for plant physics.
- One retry with identical settings completed 5 laps with zero collisions:
  `live_runs/v300_native_controls_speed85_5lap_post_odom_retry_20261020/run`.
  Its lap times were 8.203, 7.507, 7.732, 7.526, and 7.541 s (mean 7.702,
  best 7.507); max speed was 7.149 m/s. Bridge latency returned to
  4.76/29.63/36.27 ms median/p95/max. Production odom u/v RMSE was
  0.0432/0.0089 m/s; 0.179 m integrated pose drift. These GT-follower data
  are not MPC closed-loop validation.
- On this clean bag, 750 ms source-plant XY RMSE changed from 5.96 cm with
  planned-command-only propagation to 4.69 cm when each elapsed interval used
  its recorded throttle feedback; endpoint coverage under 5 cm rose from
  1112/1536 to 1182/1536. Full-path max-error RMSE remained 5.29 cm with the
  feedback replay, and 391/1536 paths still exceeded 5 cm. This supports a
  retrospective odometry correction, but does not make future MPC rollout
  accurate: its future feedback is unknown. The bridge timing outlier bag is
  retained and documented separately, not treated as a vehicle failure.
- Full metrics and caveats are in
  [CSV_V251_PRACTICE_FOLLOW_20261020.md](CSV_V251_PRACTICE_FOLLOW_20261020.md).

### Follow-up: newest-request command phase and repeated practice bags — 2026-10-20

- The current runtime source plant and MPC now use the newest request-associated
  command for the next modeled interval (queue slot 2), avoiding an extra
  unsupported 25 ms command delay. The MPC delay prefix is one step; command
  history remains outside the QP state vector. The source Jacobian preserves
  the exact queue shift/append derivatives. Focused source-plant, MPC rollout,
  predictor, and state-synchronizer tests and the package build passed before
  the repeated capture.
- Three independent five-lap practice bags are retained under
  `live_runs/v300_native_controls_speed85_phasefix_5lap_run{1,2,3}_20261020/run`.
  All used the same 85%-scaled v300 path/profile, GT-based development follower,
  MPC disabled, established Xvfb batch-mode practice simulator, and collision
  watcher. All completed five laps with zero collisions. Mean lap times were
  7.752 s (run 1), 7.698 s over run 2's non-startup laps, and 7.770 s (run 3).
  These are data-gathering runs, not MPC validation or a track-record claim.
- On run 3, source-plant A/B with the same initial states/commands showed
  current-request phase vs preceding-request phase: one-step yaw-rate RMSE
  `0.0902 vs 0.1325 rad/s`, forward-speed RMSE `0.0651 vs 0.0792 m/s`; 750 ms
  XY RMSE `9.08 vs 11.19 cm`. The phase change is retained. It does not meet
  the 5 cm rollout bound: `816/1,551` full 750 ms paths exceeded 5 cm and worst
  error was `45.2 cm`.
- Offline oracle diagnostics on this unseen run attribute a substantial part
  of the residual to command/actuator association: future throttle feedback
  reduced 750 ms forward-speed RMSE `0.191 -> 0.051 m/s`; future steering
  feedback reduced yaw-rate RMSE `0.105 -> 0.060 rad/s`; both together yielded
  `2.76 cm` XY RMSE, but still `158/1,551` paths above 5 cm and `33.2 cm`
  maximum error. These future-feedback replays are diagnostic only and are not
  production MPC inputs. Do not claim high-accuracy rollout or a complete
  PhysX match. The remaining work is a causal permitted-input representation
  of actuator/command application phase; the bridge capture has no Unity
  FixedUpdate/applied-command sequence with which to make the oracle deployable.
- The detailed bag inventory, lap metrics, 25 ms cadence, phase A/B results,
  oracle partitions, and limits are in
  [CSV_V251_PRACTICE_FOLLOW_20261020.md](CSV_V251_PRACTICE_FOLLOW_20261020.md).

### Follow-up: is the remaining rollout error command timing? — 2026-10-20

- The evidence says **command/actuator epoch association explains a substantial
  part, but not all, of the remaining error**. On practice run 3, substituting
  recorded future actuator feedback (an offline oracle, unavailable to a
  forward rollout) reduced 750 ms forward-speed RMSE from `0.191` to
  `0.051 m/s` with throttle feedback, and yaw-rate RMSE from `0.105` to
  `0.060 rad/s` with steering feedback. Supplying both reduced XY RMSE from
  `9.08` to `2.76 cm`, but still left `158/1,551` full paths above 5 cm and a
  `33.2 cm` worst path. The oracle proves sensitivity to actuator values/epoch;
  it does not by itself prove network receive delay is the cause.
- The current MPC transition is not batching 25 ms of dynamics under one final
  steering angle: `unity_source_plant_step()` calls `step_interval()`, which
  executes 25 sequential 1 ms plant steps. The Unity configuration uses a 1 ms
  outer fixed timestep. Thus the obvious coarse-step scheduling mismatch is
  ruled out in this path.
- Unity source explains a remaining sample-phase ambiguity: each
  `VehicleController.FixedUpdate()` copies `DriveTorque` and `SteeringAngle`
  into `currentT/currentS` **before** it calls `Steer()` and `Drive()`. The
  feedback getters therefore expose the prior tick's actuator state, while
  `Socket.OnBridge()` updates autonomous command targets asynchronously with
  respect to those fixed updates. Bridge ROS headers are deliberately stamped
  at the host request boundary, although the packet arrives later (run 3
  request/response latency median `4.79 ms`, p95 `30.16 ms`, max `63.62 ms`).
  The bags have no Unity fixed-step index or applied-command sequence, so they
  cannot establish the exact tick at which each target affected the vehicle.
- Disposition: do not add another fixed 25 ms shift or fit a timing offset.
  Existing phase A/B already supports the current request-associated command
  phase over the preceding-request phase. The residual is consistent with
  variable command-to-fixed-step / feedback sampling association, plus other
  unresolved plant or truth-sampling error; the current bags cannot separate
  those causes. A causal conclusion requires development-only recording of
  command sequence at Unity receipt, command sequence used by each
  `FixedUpdate`, physics-step index/time, and the feedback/pose/twist snapshot
  index. These fields must remain diagnostics and must not become production
  controller inputs.

### MPC performance / multi-SQP handoff progress — 2026-10-21

Progress and detailed replay data are tracked in
[`MPC_PERFORMANCE_MULTI_SQP_PROGRESS_20261021.md`](MPC_PERFORMANCE_MULTI_SQP_PROGRESS_20261021.md).
The latest supervised capture,
`live_runs/mpc_performance_aligned_dual20_live_20261021/run`, used the current
mounted workspace build on the **practice** simulator and passed the runtime
topic-policy guard. Its CDR decoder recovered 396 schema-11 plant states, 303
drive commands, and all actuator-timing events. This verifies that the live bag
is analyzable with the current decoder.

This was **not a successful run**: one collision occurred, no lap completed,
and the peak speed was 2.207 m/s. The exact publication gate recorded one
overrun at 32.771 ms (`p50/p95/max = 9.991/13.825/32.771 ms`), so the requested
hard `<20 ms` maximum is not met. The aligned `Dual20` tangent layout preserved
409-row replay outputs exactly and showed a promising p50 reduction, but did
not establish a reliable p95/tail win. Do not add another nonlinear SQP pass
until every cycle has measured timing reserve. The detailed log documents the
current single-solve candidate/previous-plan acceptance path and its remaining
timing-fault limitation. A follow-up source change now records, per cycle,
whether prior-plan validation was attempted, exactly feasible, and then too
late for the gate, plus whether any command was actually published. Those
fields are verified in the subsequent absolute-deadline bag.

An offline 4-versus-8 Jacobian-worker sweep was then run four times per setting
on identical 409-row practice replay inputs. Eight workers preserved every
status and first steering/throttle output and reduced the median of per-run
cycle p50/p95 from 7.467/9.909 ms to 4.269/6.100 ms on the 8-core Ryzen 9
5900HS. A supervised practice run then launched the actual MPC with eight
workers, diagnostics, the focused recorder, and collision/fault watcher. It
published 529 commands over 13.2 s with no collision; odometry averaged 25.03
ms and forwarded actuator sequences had no gaps. The publication gate was
6.392/10.158/19.730 ms p50/p95/max, with no over-budget published command.
However, after 530 solve cycles the watcher stopped the run on exact corridor
rejection (candidate stage 16 and prior plan stage 27); no lap completed. The
competition YAML is set to eight on this 8-core host. Treat the timing result
as a successful short capped run, not robust high-speed or full-lap proof: the
maximum left only 0.270 ms of margin.

The next deadline-capture bag,
`live_runs/mpc_absolute_deadline_practice_20261021/run`, had exactly one MPC
rejection: an exactly feasible `accepted_feasible` result missed publication at
20.448 ms; 220 other cycle outputs were published. Replay showed that raising
the ADMM cap from 100 to 150 or 200 worsened p95 time and reduced exact-valid
actions, so keep 100. Bounded nonlinear backtracking recovers only a small
number of offline corridor cases and remains disabled in production until the
Unity command-effect epoch can be certified. Based on the latest observed
3.797 ms candidate-rollout max plus 0.484 ms callback tail, the post-solve
reserve has been changed from 3.5 to 5.0 ms; this code change is pending a
supervised live verification with the watcher armed before simulator motion.
The latest capture reached 9 collisions before the late-armed watcher stopped
it; it completed no lap and peaked at 1.999 m/s, so it proves neither a safe
full run nor racing behavior. Detailed evidence is in
[MPC_PERFORMANCE_MULTI_SQP_PROGRESS_20261021.md](MPC_PERFORMANCE_MULTI_SQP_PROGRESS_20261021.md).

### MPC solver/recovery update — 2026-10-21

The follow-up 5 ms post-solve reserve passed a short supervised practice
capture: 536 published commands, publication-gate p50/p95/max
`8.765/12.077/17.998 ms` (all below 20 ms). It did **not** complete a lap:
the current candidate and revalidated prior-plan prefix both failed the exact
corridor check at stage 2 after the car reached `e_y=0.860 m`,
`e_psi=0.606 rad`, `u=2.610 m/s`. A collision was counted at shutdown, so this
is not a collision-free run. See the linked performance log for the bag,
decoded inputs, and replay evidence.

More ADMM iterations do not currently solve that case. On the same 537 saved
states, cap 100 exactly accepted 515; cap 500 accepted 513 and still missed
the stage-2 upper corridor by 3.2 cm, while its worst offline cycle reached
22.4 ms. The cap-500 setting was only a temporary diagnostic and has been
restored; live cap remains 100. Keep bounded line search disabled live: it
did not recover this state, and no command-effect epoch signal has been
validated for safely scheduling optional trials. Timing is improved on this
short capped run, but lap completion and reliable recovery remain unsolved.

Follow-up analysis supports but does not prove a weight-priority problem:
`|e_y|` stayed below `0.15 m` through sample 401, then rose from `0.264 m` to
`0.860 m` over the final 22 samples in a tight bend. In fixed-state replay,
the terminal one-stage speed-undershoot term is about `536`, compared with
about `92` for lateral error and `12` for heading error. An extreme offline
path-first replay (path weights ×8, speed weights ×0.01) raises exact-valid
states from `515/537` to `524/537`; it accepts 10/22 late states above
`|e_y|=0.25 m`, including 1/11 above `0.5 m`, but still rejects the terminal
`0.860 m` state and every state above `0.75 m`. Offline replay timing reached
`16.5 ms` p95 / `26.9 ms` max. This confirms objective priority contributes,
but fixed-state replay is not closed-loop evidence and the extreme setting is
not a production fix. No live weights have been changed. Details and replay
artifacts are in the performance log's path-tracking diagnosis section.
