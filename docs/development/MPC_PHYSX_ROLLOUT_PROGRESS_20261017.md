# MPC PhysX Rollout Progress — 2026-10-17

## Current status

The production MPC is now running the compact 16-state RTI plant, including
the source-derived 1 ms vehicle step in
`f1tenth_mpc/src/unity_source_plant.cpp`, its carried wheel-spin states, and
the supported Y1 yaw correction. The RTI state remains compact because the
26-state experiment missed the control deadline badly. This is an integrated
candidate, **not** a 1:1 native PhysX contact-solver implementation and not yet
a high-precision full-horizon predictor.

The current practice map and raceline were left unchanged. No simulator
physics or runtime topic subscriptions were changed. The runtime topic-policy
check passed before the live run.

## Rejected-action continuity guard and full practice run

The prior compact run had a rejected RTI first action that remained inside the
immediate corridor but jumped far from the shifted nominal, including
alternating steering-rate saturation. The rejection-only action repair now
backtracks toward the shifted nominal until the action both passes the exact
one-step vehicle-footprint corridor and stays within the existing RTI2 action
correction limits (0.50 rad/s steering-rate change and 1.0 m/s² target-rate
change). Accepted optimal/degraded candidates are unchanged. This does not
repair model accuracy; it prevents an unconverged first action from making a
large discontinuous jump.

The real batch-mode practice run
`live_runs/mpc_rejected_trust_guard_practice_20261017/` used the existing
practice trajectory with the collision watcher armed. It completed one
warmup, all ten scored laps, and one extra lap with **zero collisions**:

| Measure | Result |
|---|---:|
| Scored laps | 10/10 |
| Best / median / mean | 5.984 / 6.026 / 6.046 s |
| Extra lap | 5.971 s |
| Truth cross-track absolute p95 | 0.192 m |
| One-step speed-error p95 | 0.069 m/s |
| One-step yaw-rate-error p95 / max | 0.244 / 1.216 rad/s |
| Collision delta | 0 |

The action repair path was exercised: 49 rejected best-effort actions were
reported, 43 were backtracked. This is an uncontrolled comparison with the
prior one-lap collision run, so it does not prove the guard alone caused the
full-run success.

### Control-loop timing

The bag recorded 3,063 LiDAR messages at 39.981 Hz and 3,060 bridge packets
with no sequence gaps. The MPC produced 3,009 diagnostics over the run. Solver
latency was p50/p95/p99/max **16.36/24.84/30.21/46.24 ms**; 143 of 3,006
measured solves exceeded the 25 ms budget. Full pipeline p95/p99/max was
25.08/30.46/46.36 ms. Thus the sensor stream is at 40 Hz, but the MPC still
misses some strict 40 Hz compute deadlines and is not yet timing-robust.

Most long cycles used a second RTI pass after a failed nonlinear corridor or
residual check. The diagnostic trigger was commonly the R1 corridor-repair
path; several R2 attempts ran to the 100-iteration cap. This is the measured
source of tail latency, not the 1 ms plant integration itself.

### First-divergence evidence

The largest yaw errors occurred during low-speed, high-steering transitions,
not in forward-speed prediction. At one example, the current state was
`u=2.235 m/s`, `r=-1.010 rad/s`, physical steering `-0.321 rad`; the 25 ms
prediction was `r=-1.118 rad/s` while truth reached `-2.334 rad/s`. In the
same corner shortly afterward, at `u=3.69 m/s` and steering near `-0.265 rad`,
the prediction overshot truth by about `0.70 rad/s`. The error changes sign
between steering entry and the held-turn phase, ruling out a single steady
gain correction.

Across the run, absolute yaw error was largest below 4 m/s and at steering
magnitudes above 0.2 rad. The Y1 surface correction was materially active on
only 78 of 2,946 scored one-step predictions; the worst turn-entry errors
occurred outside its steady-response support. This identifies a missing
steering-transition/actuator-and-slip response model. It does not yet identify
which hidden PhysX contact state is responsible, so no broad Y1 gate change or
new yaw gain is promoted.

The odometry comparison in the same bag is also not near-perfect: after one
fixed initial SE(2) alignment, integrated `/odom` position p95 reached
0.271 m over the run (0.293 m at the final lap segment), while forward-speed
RMSE was 0.0288 m/s, lateral-speed RMSE 0.0083 m/s, and gyro yaw-rate matched
truth to the reported precision. AMCL's raw map-position p95 was 0.181 m. MPC
one-step path progress/lateral errors stayed much smaller than the integrated
odom drift, but this does not establish an accurate long-horizon plant.

The detailed live reports are in:

- `live_runs/mpc_rejected_trust_guard_practice_20261017/analysis/race_metrics/summary.md`
- `live_runs/mpc_rejected_trust_guard_practice_20261017/analysis/control_timing/control_state_latency_summary.json`
- `live_runs/mpc_rejected_trust_guard_practice_20261017/run/run_0.db3`

## Native reusable-server reset verification

The persistent PhysX server now accepts a complete initial-state reset before
each candidate rollout. The reset restores rigid-body pose/velocity, wheel
shape transforms, wheel angular speed/angle, and the prior suspension jounce;
it also clears PhysX filtering state and wakes the actor. On the practice r09
40.000--41.000 s command-only window, a fresh server and a reset server were
run with the same initial condition and all 1,000 1 ms commands. Their
40 Hz-stage outputs were bit-identical for body position/orientation/velocity,
wheel speeds/angles/jounce, and longitudinal slip. Repeated resets therefore
do not introduce candidate-to-candidate state contamination in this test.

An earlier comparison in the working notes labeled reset-versus-continuous
output as a native-build discrepancy. That comparison used a continuing
vehicle state as its baseline, not an equivalent fresh initial state; it was
not a valid reset parity test. Direct replay of the identical input through
the old temporary executable, the workspace binary, and the reset-capable
binary produced byte-identical JSON outputs (SHA-256
`7cd82d8f76563db73eee49b5f8cb3ae6fbf75ff291b3497d6b1eb11e0ca79878`).
The earlier apparent old/new binary mismatch is resolved.

This verifies repeatability of the native research server, not a production
MPC integration. The server is still a JSON-line executable; its PhysX plant
has not yet been embedded in the ROS controller or timed as part of a complete
40 Hz solve.

## Evidence from existing captures

The fast candidate was replayed from truth-initialized windows on independent
r11 and r12g captures, with the recorded steering/throttle command sequence
as its only forward input. The capture truth is used only for initial state
and scoring.

| Capture | 25 ms position RMSE | 25 ms yaw-rate RMSE | 1 s recursive position RMSE | 1 s recursive yaw-rate RMSE |
| --- | ---: | ---: | ---: | ---: |
| r11 exact replay | 0.81 mm | 0.0578 rad/s | 0.220 m | 0.104 rad/s |
| r12g dynamic validation | 0.79 mm | 0.0686 rad/s | 0.110 m | 0.0299 rad/s |

The r11 worst recursive windows remain around 4.9 m/s and 0.16 rad steering;
the largest 1 s position error is 1.44 m. The candidate therefore fails the
required full-state rollout accuracy despite low short-step position error.

The native PhysX 4.1.2 scene harness already in
`tools/vehicle_dynamics_learning/physx_native_vehicle_step.cpp` was run on the
same difficult r11 interval, starting from the captured body/wheel state and
using only the recorded per-wheel steering, motor-torque, and brake-torque
sequence thereafter. Its 1 s result was:

- planar position RMSE: **0.20 mm**;
- yaw RMSE: **0.000113 rad**;
- forward/lateral speed RMSE: **0.000026 / 0.000037 m/s**;
- yaw-rate RMSE: **0.000267 rad/s**;
- wheel-speed pooled RMSE: **0.000175 rad/s**.

The persistent native server completed that 1 s segment in **0.134 s wall
time** (7.45x real time). This is strong evidence that the public PhysX
vehicle-plus-scene solver reproduces this interval; it is not yet a production
MPC result. The native input used the captured actuator-level commands. On
this interval, source `VehicleController` equations reconstruct captured
per-wheel steering and torque to numerical precision, but the logged command
changes are not uniformly aligned to fixed 25-row blocks. That command timing
must be represented correctly in the production input contract.

## Source-driven corrections tried in the fast candidate

- Corrected tire combined-slip order: normalize using raw friction-curve
  response, then apply curve stiffness. Focused tests pass. This did not
  materially reduce recursive error.
- Replaced the candidate's full Euler/gyroscopic angular update with the
  mass-space diagonal, no-gyroscopic-term `PxVehicleUpdate.cpp::integrateBody`
  behavior. Added a regression test proving pure roll rate does not invent
  yaw torque in that source integrator. It modestly improved one-step yaw on
  r11 but worsened r11 recursive yaw error; it is not a validated MPC fix.
- Initialized each evaluation window's suspension damper history from the
  captured wheel pose instead of zero history. This is the correct available
  state initialization, but it did not materially reduce the trajectory
  error.

The fast model still omits the native rigid-body scene contact solve and has
not demonstrated accurate recursive tire/body coupling. Its measured errors
are not to be hidden by tuning MPC weights.

## Next implementation work

1. Use the saved 10-lap bag plus existing open-plane captures to isolate the
   low-speed/high-steering steering-entry and held-turn response separately.
   Keep whole runs separated for validation. Explicitly condition the analysis
   on physical steering rate, throttle/brake state, wheel/body-speed mismatch,
   and packet age; do not tune a common yaw gain from pooled samples.
2. Compare that regime against a native PhysX 25 ms replay initialized from
   the best available body, wheel, and suspension state. This determines
   whether the current yaw error comes from the simplified tire/contact
   calculation or from its runtime inputs/timing. Preserve the native plant's
   measured runtime cost: the existing 1 s replay takes 0.134 s and cannot be
   inserted naively as a 40 Hz, full-horizon optimization model.
3. Make one transition-specific model correction only after that comparison;
   replay the same practice and independent open-plane captures and report
   yaw, speed, lateral velocity, wheels, slip, and integrated pose separately.
   Keep the source plant and Y1 model as comparators.
4. Reduce the measured RTI tail by profiling the second-pass/recovery path.
   Preserve the 25 ms cadence and do not suppress recovery passes without
   showing that the accepted actions and collision-free lap transfer remain
   intact.
5. Only after the prediction and deadline gains transfer, tune MPC weights for
   the fastest speed profile. A safe 6.046 s mean is not sub-5 s progress.

## Artifacts

- Fast-plant implementation: `f1tenth_mpc/src/unity_source_plant.cpp`
- Focused plant tests: `f1tenth_mpc/test/test_unity_source_plant.cpp`
- Capture evaluator: `tools/vehicle_dynamics_learning/evaluate_mpc_source_plant_capture.py`
- Native PhysX implementation: `tools/vehicle_dynamics_learning/physx_native_vehicle_step.cpp`
- Detailed source audit: `docs/development/PHYSX_41_WHEELCOLLIDER_SOURCE_PARITY_20261016.md`
- Paired reports: `live_runs/mpc_source_plant_eval_20261017/analysis/`

## Continuation: actuator-predictor parity and repeat practice validation

The second clean 12-crossing practice capture is
`live_runs/mpc_nominal_reject_recovery_v2_practice_20261017/`. It completed
1 warmup + 10 scored + 1 extra with zero collisions: mean 6.0252 s, best
5.9148 s, `|CTE|` p95 0.1884 m, 25 ms speed-error p95 0.0539 m/s, and yaw-rate
error p95 0.2535 rad/s. This is only 0.0208 s faster in mean than the earlier
6.0460 s run; the runs were not randomized paired trials, so the small timing
difference is not attributable to an MPC change. The added shifted-nominal
candidate did not trigger and has since been removed as unexercised code.

## Continuation: correctly configured recovery experiment and isolation

Two earlier attempts in this continuation used `start_dev.sh` defaults, which
loaded the competition map/trajectory rather than the practice pair. They are
not practice-MPC evidence and are not included in any comparison. The later
run below explicitly logged the practice map and trajectory at startup.

`live_runs/mpc_recovery_seed_practice_corrected_20261017/` used the correct
practice map and `...speed_headroom_codex.csv`, diagnostics, and a live
collision watcher. It **failed** after five scored laps (lap counter 6) with
one collision; this is not a usable controller result:

| Measure | Result |
|---|---:|
| Scored laps | 5/10 |
| Best / median / mean completed scored lap | 5.970 / 6.063 / 6.095 s |
| Truth `|CTE|` p95 | 0.205 m |
| One-step speed / yaw-rate error p95 | 0.091 m/s / 0.286 rad/s |
| Sensor / bridge rate and sequence gaps | 39.96 / 39.95 Hz; 0 gaps |
| MPC solve p50 / p95 / p99 / max | 17.0 / 25.6 / 34.8 / 48.5 ms |
| Solve cycles over 25 ms | 93 / 1,537 |
| MPC status counts | 1,403 optimal; 64 degraded; 25 nonlinear rejects; 40 residual rejects; 5 input rejects |

The collision followed a growing path/heading error in the opening corner of
the next lap. In the last ~0.2 s before collision, truth lateral offset from
the target grew to about -0.36 m, the nearest body-wall clearance fell to
0.156 m, the commanded/feedback steering stayed near -0.20 rad, and the
controller repeatedly rejected its RTI candidate. The recovery seed was
reported as not found near the end. This identifies the failure sequence but
does not by itself prove whether the cause was the recovery law, widened Y1
blend, plant yaw error, or their interaction.

In that test, two experimental config changes were made together: the Y1
steering blend was widened from the previously used 0.29/0.30 rad gate to
0.23/0.25 rad, and the recovery seed changed from nominal to
`brake_heading_feedback`. Because the run regressed and those changes are
confounded, both have been restored to the previous collision-free settings
before testing the separate source-derived positive-acceleration prediction
change. No claim of a causal fix is made from this failure.

The control-timing report and race report are preserved under the run's
`analysis/` directory. The candidate fails both the collision-free 12-crossing
requirement and the strict 25 ms compute deadline; the next controlled live
test must use the reverted Y1/recovery settings and the explicit practice map
and raceline. The known 12-crossing run remains the reference, not this
candidate.

The bag analyzer now reports every published prediction node, but its new
long-horizon numbers are deliberately labelled as planned-horizon versus
future closed-loop comparisons. MPC replans every 25 ms, so later planned
actions are not the actions actually executed. On the second run, the 25 ms
`|Δyaw rate|` p95 was 0.2535 rad/s and the 750 ms value was 0.8001 rad/s;
the corresponding offset-subtracted values were 0.2683 and 0.8021 rad/s.
These remain useful system-level discrepancy indicators, not pure open-loop
plant validation. The lap-time rows in the Markdown report are also now kept
under their own table after the report generator ordering fix.

### Production speed-loop equation brought into the plant predictor

Source inspection found a specific mismatch in `source_actuator_throttle()`.
The live `SpeedController._predicted_positive_acceleration()` predicts an
approaching speed-hold transition from the maximum of (a) positive measured
acceleration and (b) acceleration implied by current throttle above the
feed-forward throttle at measured speed, capped by the configured maximum
acceleration. The MPC predictor previously used only the measured term. It
could therefore delay its feed-forward hold handoff while the live controller
had already recognized that current throttle would cross the handoff band.
The C predictor now uses the matching equation and the same throttle-per-
acceleration table as its maximum-acceleration calculation. A focused
regression case was added for this specific state transition.

This is an actuator/control-state fidelity correction, not a change to Unity
physics or a claim that the tire/body plant is accurate. It was built in the
existing `practice-physx-odom-20261016` dev image (no image rebuild), passed
the runtime topic-policy gate, and the focused
`test_vehicle_model_source_rollout` suite passed. The regression specifically
checks the predicted throttle-lead transition into speed hold.

The real-practice validation is now complete as
`live_runs/mpc_source_actuator_handoff_practice_20261017/`. It completed the
warmup and ten scored laps, then collided during the extra lap. The watcher
aborted immediately on the collision signal and stopped the simulator,
controller, and recorder containers. The run is therefore not a full
collision-free validation.

| Measure | Result |
|---|---:|
| Scored laps | 10/10 |
| Best / median / mean | 5.964 / 6.017 / 6.041 s |
| Extra lap | collided before completion |
| `|CTE|` p95 / max | 0.193 / 0.350 m |
| MPC one-step speed / yaw-rate error p95 | 0.085 m/s / 0.222 rad/s |
| MPC one-step lateral / progress error p95 | 0.045 / 0.150 m |
| Collision count increment | 1 |

Compared with the previous clean v2 run (mean 6.025 s, best 5.915 s, zero
collisions), this run was 0.016 s slower in mean and 0.049 s slower at best.
Those are unpaired runs, so neither difference is attributable to the actuator
change. The measured speed-error p95 also increased (0.054 to 0.085 m/s),
while yaw-rate p95 changed from 0.253 to 0.222 rad/s. The actuator parity
change is retained as a source-consistency correction, not promoted as a
demonstrated performance or safety gain.

### Collision and large-error diagnosis

The bag shows the failure was not preceded by dropped bridge packets: packet
sequences remained consecutive through the impact and the measured sequence
gap count is zero. A bridge `Socket.IO connection lost` event occurred about
150 ms after the collision, so it is downstream of the impact, not evidence
that transport delay caused it.

In the extra lap, the car was initially close to its prior-lap behavior, but
its tracking then diverged through a low-speed, high-steering turn. Around
2.1 m/s and physical steering `-0.46 rad`, the one-step plant predicted yaw
rate `-0.70 rad/s` while simulator truth was `-1.43 rad/s` (0.73 rad/s
underprediction). Cross-track error subsequently grew to about 0.34 m and the
estimated right-body clearance fell from roughly 0.40 m to 0.073 m. At the
last pre-impact sample, speed was 3.03 m/s, throttle feedback was zero, and
steering feedback was about `+0.083 rad`; collision count then incremented,
speed dropped to 0.03 m/s, and pose jumped to a stationary checkpoint-like
location. The Unity source confirms `AutomobileController.OnCollisionEnter`
increments the exported collision count for any collision; `LapTimer` separately
respawns on racetrack collision. The count alone does not identify the collider,
but the pose/speed discontinuity is consistent with an impact and respawn.

MPC diagnostics in the same interval show repeated `rejected_residual`
solutions at the 100-iteration cap. When no exact-feasible first action could
be verified, the existing production rejection path held steering and reduced
the target speed by 0.2 m/s per cycle (8 m/s² configured deceleration). This
is a concrete recovery weakness, but the bag does not prove that the actuator
parity patch caused the yaw error or collision. The yaw mismatch and repeated
solver rejection are the next demonstrated issues to fix; changing weights
alone is not justified by this evidence.

### 40 Hz status

The completed bag's LiDAR header rate was 39.979 Hz and bridge packet sequence
gaps were zero. This confirms the input stream cadence, not deadline compliance.
MPC source-age p95 was 37.11 ms; solver latency p50/p95/p99/max was
16.52/25.70/33.80/48.77 ms, with 187 of 2,913 measured solves over 25 ms.
Full callback-to-diagnostic p95/p99/max was 25.89/34.02/48.90 ms. Therefore
the production controller still does not meet a hard 25 ms deadline on every
cycle.

### Deadline status remains open

The strict 40 Hz compute deadline remains open. Continue from the witnessed
low-speed/high-steering yaw underprediction and residual-rejection episode;
preserve the current practice map and trajectory, and use collision-triggered
abort for the next real run.
