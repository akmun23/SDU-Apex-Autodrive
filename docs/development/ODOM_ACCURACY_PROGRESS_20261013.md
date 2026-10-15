# Odom and offline-plant progress — 2026-10-13

## Exact-packet Unity acceleration and encoder-window follow-up

This follow-up joins each replay diagnostic to its source packet stamp, rather
than selecting an IMU sample by nearest callback/receipt time. The source stamp
is encoded as `Float64` Unix seconds, so integer recovery differs from the raw
stamp by at most 384 ns; every sample joined within 2 microseconds, far below
the 25 ms packet interval. The replay diagnostics confirm the Odom-consumed
`ax`, `ay`, and gyro-z are from that packet. The fixed-step replay uses exactly
25 ms between consecutive packet sequence IDs.

The Unity-derived acceleration conversion was checked against packet-aligned
body-velocity differences on R01–R04. For R02/R04, the current packet's IMU
sample, transformed from Rigidbody COM to rear axle using
`ax,rear=ax,COM+d*r^2`, `ay,rear=ay,COM-d*rdot`, gives acceleration RMSE of
`0.352/0.241` and `0.362/0.240 m/s^2` (longitudinal/lateral). The previous
packet sample is much worse (`0.930/0.520`, `0.994/0.528`); averaging adjacent
packets is also worse (`0.491/0.360`, `0.532/0.363`). This supports the
existing endpoint-sample, fixed-25-ms Odom integration; it does not indicate a
wrong COM-offset equation or justify acceleration filtering. The one-step
velocity error remains about 0.009 m/s, but this does not make IMU-only
recursive integration accurate.

I tested two targeted alternatives against the same held-out R02/R04 practice
captures, each using the same residual model and scored on identical full-run
windows:

| Odom variant | R02 forward-speed RMSE / position RMSE / endpoint | R04 forward-speed RMSE / position RMSE / endpoint |
| --- | --- | --- |
| Current 100 ms wheel window + existing deceleration correction | `0.04396 m/s / 0.1513 m / 0.2090 m` | `0.04344 m/s / 0.1908 m / 0.2988 m` |
| 150 ms wheel window | `0.06735 / 0.3006 / 0.4965` | `0.06057 / 0.2372 / 0.3674` |
| Unity-equation deceleration (`scale=1`, `offset=0`) | `0.04390 / 0.1338 / 0.1793` | `0.04347 / 0.2107 / 0.3326` |

The longer wheel window smooths its isolated signal slightly, but degrades
recursive speed and position on both runs; it is rejected. Removing the
empirical deceleration scale/offset gives a small R02 pose gain but worsens
R04, so that change is also not promoted. Neither result establishes a
transferable Odom improvement. No simulator physics, reference path, or
runtime sensor subscription was changed by these experiments.

The current source-derived diagnosis is therefore narrower: Unity's IMU
sampling convention and rigid-body frame conversion agree with the present
Odom equations at 40 Hz. The remaining position error is mainly longitudinal
state error accumulated across encoder updates and wheel slip/burst intervals,
not an incorrect IMU sample-time shift. The exact per-wheel force/slip solver
is not exposed in the available native WheelCollider implementation; the
development-only r07/r11 WheelHit captures remain useful mechanics evidence,
but are not proof of parity with the pinned player. Next work stays on the
longitudinal observer's wheel/IMU handoff and run-to-run residual transfer,
using the existing bags first. No new simulator run was started.

## Latest evidence and implementation (later on 2026-10-13)

### Source-timed observer and validation status

`external/autodrive-unity-source/Assets/Scripts/IMU.cs` computes linear
acceleration by finite-differencing Rigidbody velocity in `FixedUpdate`; the
project timestep is 1 ms. `MotorEncoder.cs` likewise accumulates WheelCollider
RPM at each physics update and emits cumulative quantized ticks. The bridge
delivers these streams at about 40 Hz, and their ROS headers are receive-time
stamps rather than physics sample times. The observer now advances its
internal integration clock by exactly 25 ms per complete paired packet while
preserving received stamps on published messages and timing diagnostics. No
sensor subscription or simulator physics was changed. Captured streams in the
two clean validation runs averaged 25.009/25.011 ms, with no packet loss.

The 25 ms clock is source-grounded but **not yet live-validated successfully**.
The official practice run using this node and the retained prior residual
collided during its first lap; the collision watcher stopped it immediately,
so it produced zero scored laps. The prefix shows MPC nonlinear-rollout
rejections and oscillating steering before impact; the collision cannot be
attributed to the Odom clock alone. The raw bag and prefix analysis are
preserved under
[`odom_fixed25ms_runtime_practice_20261013`](../../live_runs/odom_fixed25ms_runtime_practice_20261013/analysis/collision_prefix/summary.md).

Offline fixed-clock replay, scored against simulator truth with one initial
alignment over the entire continuous run, gives mixed comparison to the
recorded receive-time Odom:

| Run and mode | Forward-speed RMSE | Position RMSE | Endpoint error |
| --- | ---: | ---: | ---: |
| R02 recorded receive-time Odom | 0.06626 m/s | 0.25665 m | 0.49511 m |
| R02 fixed 25 ms + prior residual | 0.04420 m/s | 0.21257 m | 0.32293 m |
| R04 recorded receive-time Odom | 0.04158 m/s | 0.09172 m | 0.13719 m |
| R04 fixed 25 ms + prior residual | 0.04349 m/s | 0.14806 m | 0.19765 m |

Thus the fixed-clock replay materially helps R02 but modestly worsens R04; it
is not a universal accuracy win. Disabling the residual under the fixed clock
was substantially worse on both complete runs (R02: `0.06497 m/s`, `0.33482
m`; R04: `0.06524 m/s`, `0.57419 m`), so the currently deployed residual is
retained rather than discarded.

Because the residual includes `dt` as an input, I refit it using no-residual,
fixed-clock replay baselines on training-only R01/R03, with whole-run R02/R04
still held out. Recursive production-node replay of that refit yielded:

| Held-out run | Speed RMSE | One-alignment position RMSE | Endpoint error |
| --- | ---: | ---: | ---: |
| R02 | 0.04357 m/s | 0.18505 m | 0.25683 m |
| R04 | 0.04325 m/s | 0.16445 m | 0.24839 m |

This refit improves R02 versus the prior residual under the fixed clock, but
R04 position/endpoint error worsens. In the independent collision prefix, its
speed RMSE is `0.28568 m/s`, close to the prior residual's `0.28921 m/s` and
worse than no residual's `0.19324 m/s`. The refit is therefore **not
promoted**; the previous deployed coefficients remain in YAML. The complete
fit, recursive replays, and scores are preserved under
[`odom_fixed25ms_residual_refit_20261013`](../../live_runs/odom_fixed25ms_residual_refit_20261013/fit/fit_and_validation_report.json).

The next required Odom work is to explain the run-to-run transfer difference
without selecting on the collision or changing MPC. Until that is resolved,
the fixed-step runtime edit has only mixed replay evidence and a failed live
drive; the offline plant remains unsuitable for trusted whole-lap simulation
or raceline ranking.

Runtime Odom remains restricted to the paired rear encoders and IMU; the
competition topic-policy check passed. Simulator truth is used only for
offline fit/scoring. The reference trajectory and simulation physics remain
unchanged.

### Prior snapshot (superseded by the results above)

The Unity source check confirms the IMU acceleration is a finite difference of
`Rigidbody.velocity` in `Assets/Scripts/IMU.cs`; the measured point is the
Rigidbody COM, not the IMU Transform. The active F1TENTH prefab sets COM z to
`-0.00468 m`, while the rear-axle reference is `-0.16000 m`, giving the
forward offset `d=0.15532 m`. In ROS rear-axle axes, the rigid-body conversion
is `a_x,rear = a_x,COM + d*r^2` and
`a_y,rear = a_y,COM - d*dr/dt`. Both terms, including the latter derived
causally from consecutive gyro samples, were already present with the
correct sign in `odometry_observer.cpp`; no duplicate physics correction was
added. The public project does not expose PhysX's internal tire-force solver.

A path-aware forward-speed residual was fit on practice R01 and R03 only. It
uses the existing 22-feature observer/encoder/IMU diagnostic vector, and its
training objective jointly penalizes relative forward-speed error and
one-alignment full-run XY error. It was replayed recursively from only
encoders and IMU on the two untouched 12-lap runs R02 and R04. Simulator pose
was used only for offline scoring. With a single initial pose alignment for
the entire scored run, results were:

| Held-out run | Odom | Forward-speed RMSE | Full-run position RMSE | Endpoint drift |
| --- | --- | ---: | ---: | ---: |
| R02 | recorded model | 0.07537 m/s | 0.40239 m | 0.79351 m |
| R02 | candidate recursive replay | 0.04569 m/s | 0.16713 m | 0.26419 m |
| R04 | recorded model | 0.04178 m/s | 0.10466 m | 0.18377 m |
| R04 | candidate recursive replay | 0.04145 m/s | 0.06928 m | 0.07631 m |

The candidate therefore improved full-run position RMSE on both held-out
captures (58% and 34% reduction) and endpoint drift (67% and 59% reduction),
without changing the lateral-speed estimate or gyro yaw-rate output. This is
the first recursive, whole-run transfer result for this pooled path-aware
residual. It is a bounded 2–8 m/s correction, not a complete 0–12 m/s vehicle
model, and it has not yet been confirmed in a new live drive.

The deployment YAML now contains the evaluated candidate coefficients. The
runtime subscriptions remain encoder-left, encoder-right and IMU only; no GT,
simulator pose, steering command, or actuator topic was added. The correction
acts only on published forward velocity and pose integration; it does not
rewrite the observer's internal wheel or lateral states. Next: verify the YAML
loaded candidate reproduces the isolated candidate replay, then run one
established batch-mode practice capture with collision monitoring before
claiming production transfer. The reference trajectory and simulator physics
remain unchanged.

## Current status

The learned longitudinal residual is now enabled in the standard Odom YAML,
restricted to its measured 2–8 m/s band. It was fit using practice R01, held
out on R02, and then evaluated in two fresh 12-lap practice runs. Same-capture
baseline replays show repeatable reductions in forward-speed error and
continuous-run position drift. This is a confirmed Odom improvement, not a
finished high-accuracy estimator: the new runs still end 0.14–0.16 m from
truth after one initial alignment, no evidence covers speeds at or above
8 m/s, and MPC still uses AMCL pose rather than Odom pose. The offline 3-D
plant remains too inaccurate for trusted whole-lap raceline ranking.

No simulator physics, reference map, or reference trajectory was changed.
Ground truth was used only offline for fitting/scoring. Runtime Odom continues
to consume the mapped rear encoders and IMU; the runtime topic-policy check
passed.

## Earlier production Odom change: turn speed-bias fade

The deployed turn-slip speed-bias fade changed from 4.5–5.5 m/s to 4.0–4.5
m/s in `sensor_odometry.yaml` and the matching C++ defaults. The rationale was
fixed-sensor, recursive replay on two separate practice captures: forward
speed RMSE improved 11% on each, while lateral-speed RMSE changed by less than
0.0001 m/s. The setting was not kept offline: the current-workspace stack ran
the pinned practice simulator for one warmup, ten scored laps, and one extra
lap, with the collision watcher armed before simulator connection.

Artifacts:

- Run provenance: [`run_manifest.json`](../../live_runs/odom_turn_fade_current_mpc_practice_20261013_r02/run_manifest.json)
- Raw bag: [`run_0.db3`](../../live_runs/odom_turn_fade_current_mpc_practice_20261013_r02/run/run_0.db3)
- Race report: [`summary.md`](../../live_runs/odom_turn_fade_current_mpc_practice_20261013_r02/analysis/race_report/summary.md)
- Current-setting Odom score: [`production_odom_score.json`](../../live_runs/odom_turn_fade_current_mpc_practice_20261013_r02/analysis/production_odom_score.json)
- Old-setting, same-input replay score: [`production_odom_score.json`](../../live_runs/odom_turn_fade_current_mpc_practice_20261013_r02/analysis/old_fade_baseline_replay/production_odom_score.json)

Run result:

- 10/10 scored laps; mean 6.0575 s, best 5.9997 s; zero collisions.
- AMCL map-normal error p95 4.31 cm; tangential error p95 15.33 cm; absolute path tracking error p95 6.36 cm.
- `/imu`, both encoders, simulator odometry, and `/odom` each had 3,011 samples over the run, 39.97 Hz by receipt timestamps, with median intervals about 25.17 ms. Bridge timing diagnostics reported zero timing faults. Command streams were 40 Hz. Receipt-time jitter included one roughly 50 ms gap; no bridge sequence/timing fault was recorded.
- MPC diagnostics: 3,005 accepted-optimal cycles and six rejected startup samples in the first 125 ms, before any progress; the launch log shows the initial `missing_map_pose` wait. There were no in-lap rejections. One-step model-error p95 remained 0.114 m/s in speed and 0.236 rad/s in yaw rate.

### Same-run paired Odom comparison

The old setting was replayed offline from this exact bag's encoder and IMU
streams. It consumed no future truth. The candidate score is the recorded
current-workspace `/odom`; its data-matching fraction was 100%. The only A/B
parameter difference was the speed-fade interval.

| Metric | Old fade 4.5–5.5 m/s | Candidate fade 4.0–4.5 m/s | Change |
|---|---:|---:|---:|
| Forward speed RMSE | 0.08691 m/s | 0.06626 m/s | −23.7% |
| Forward speed absolute p95 | 0.15986 m/s | 0.13407 m/s | −16.1% |
| Forward speed bias | −0.02314 m/s | −0.01262 m/s | 45% smaller magnitude |
| Lateral speed RMSE | 0.010187 m/s | 0.010204 m/s | +0.000017 m/s |
| Relative path RMSE, aligned per lap | 0.06032 m | 0.04892 m | −18.9% |
| Position RMSE, one initial alignment for whole run | 0.43079 m | 0.25665 m | −40.4% |
| Position endpoint error after 12 laps | 0.85791 m | 0.49511 m | −42.3% |

The per-lap-aligned score is useful for within-lap shape, but it hides
accumulated drift; the one-initial-alignment score is the long-run measure.
Using the run's lap-count labels, the ten scored laps are sequence indices
2–11 (index 1 is warmup; index 12 is the extra lap). Their mean absolute
endpoint error fell from 0.463 m to 0.266 m (−42.6%). An earlier summary
mistakenly averaged indices 1–10, including warmup and omitting the tenth
scored lap. This is not zero drift: the candidate still reached about 0.50 m
endpoint error by the end of the run.

The gain is concentrated in the 4–6 m/s band, where forward-speed RMSE fell
from 0.11163 to 0.07013 m/s (−37.2%). It improved 2–4 m/s by 7.1%; 0–2 m/s
(only 48 samples) and 6–8 m/s did not materially change. By steering, the
0.1–0.2 rad group improved 0.1365 to 0.0807 m/s; 0.2–0.3 rad remained about
0.0522 m/s. Thus the change is bounded and evidence-backed, not a full-band
solution. This run peaked at 7.82 m/s and does not validate 8–12 m/s practice
behavior.

Yaw-rate RMSE is zero in the logged Odom comparison because Odom passes
through the measured IMU yaw rate. That is sensor tracking, not evidence that
the future yaw model is perfect. The MPC's 0.236 rad/s one-step yaw-error p95
remains a separate model limitation.

## Offline simulator state

Unity source and the public vehicle configuration have already been used to
build a command-only 1 kHz 3-D mechanics plant: rigid-body/quaternion motion,
four wheel-spin states, Ackermann geometry, suspension-load approximation,
source steering/torque rules, documented tire curves, and fitted causal
residuals. The public Unity project still does not expose the native
WheelCollider/PhysX force solver; this plant is therefore a source-informed
surrogate, not an exact engine clone.

On the independent Unity-source r11 capture, a source-mechanics 3-D baseline
had 5 s horizontal position RMSE 1.874 m. A causal yaw DAgger residual
reduced it to about 0.794 m, with 5 s position p95 about 1.39 m; the
state-coupled residual comparator was about 1.00 m at 5 s. These results are
meaningful gains over the earlier planar replay, but are not accurate enough
for trusted whole-lap raceline ranking. They were not integrated into runtime
Odom or MPC.

## Next work in progress

1. Use the newly captured practice bag as an untouched racing-domain evaluation
   of the offline plant. `bridge_packet_timing` contains the same-packet
   simulator state and sent commands; the plant must initialize once, then
   consume only the held 40 Hz command sequence with 1 ms internal steps. GT
   after initialization is strictly for scoring.
2. Reconstruct the plant/GPS transform from `GPS.cs`, `IMU.cs`, and the F1TENTH
   prefab rather than adding an ad-hoc offset. The public GPS receiver is the
   `Rear Axle Center` child at local `(0, 0.05, -0.16)` m; the 3-D plant state
   is the Rigidbody transform origin. The body velocity is the Rigidbody COM
   velocity in the Unity root frame.
3. Report first divergence separately for position, body velocity, yaw rate,
   and observable rear-wheel speed over 25 ms, 100 ms, 500 ms, one lap, and
   whole-run horizons. Use the older complete practice capture for any fit and
   the new run for untouched evaluation until the model-selection decision is
   frozen.
4. Continue Odom work from residuals that transfer across the existing
   independent captures. The current candidate has confirmed its 4–6 m/s
   gain; the next unsupported target is the unchanged 6–8 m/s residual and
   eventually 8–12 m/s. Do not tune an extra correction from this single run
   alone.
5. Only after the plant passes a whole-lap practice replay with small
   position/speed/yaw/wheel errors should it score aggressive racelines. Until
   then, do not report its lap-time ranking as a reliable prediction.

No open-plane or additional track test is running. The simulator, controller,
and recorder containers are shut down; all run data and reports are stored
under `live_runs/odom_turn_fade_current_mpc_practice_20261013_r02/`.

## Follow-up: source tire/slip and wheel-spin equations — 2026-10-13

The offline greybox equation audit caught a semantic mismatch: capture-derived
Unity `WheelHit` slip telemetry had been substituted for the AutoDRIVE guide's
tire-force slip coordinates. Those are different quantities. The guide force
inputs are `Sx=(R*omega-vx)/vx` and `Sy=vy/abs(vx)`; the highly accurate
regularized reconstruction of `WheelHit.sidewaysSlip` is a telemetry fit,
not proof of the force-curve input. The greybox now uses guide coordinates
with a finite low-speed denominator floor; WheelHit comparisons are labeled
as diagnostic output only. Source wheel-center geometry is also used for all
greybox variants.

The first WheelHit-coordinate revision was tested and rejected: with the same
old checkpoint, R03/R04 macro position radial RMSE worsened from `6.84/6.71`
to `7.36/7.52 m`, speed RMSE from `3.70/3.60` to `3.99/4.02 m/s`. A fresh
WheelHit-coordinate refit was worse still. A clean guide-coordinate refit
halved 250 ms yaw-rate error (`0.0948/0.0879` to `0.0456/0.0460 rad/s`), but
did not jointly improve the per-lap free runs: position became `6.72/6.80 m`,
speed `3.56/3.58 m/s`, and whole-sequence heading RMSE regressed to
`1.292/1.263 rad`. It is not promoted.

Unity source also confirms the active wheel torque law: CAWD divides the
428-throttle motor torque equally (107 N m per wheel at full throttle), while
exactly zero throttle commands 428 N m brake torque to each wheel. The
1 kHz source capture's recorded per-wheel torque matches this source mapping
to roughly `7e-7 N m` RMSE. The default greybox's fitted `motor_torque_scale`
near `0.48 N m` is therefore only a compensating effective parameter, not a
physical motor torque. A source-actuator option was added and screened with
25 substeps per 25 ms, using the guide tire equations. Despite open-plane
validation score `0.399`, it fails unseen practice badly (position radial
RMSE `10.58/10.23 m`, speed RMSE `1.48/1.58 m/s`, wheel-surface RMSE about
`453 m/s`). Correct source torque by itself does not repair the greybox tire
force law; no runtime integration is justified.

Artifacts:

- Clean train/validation export: [`greybox_source_slip_clean_train_validation_dataset_20261013`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/greybox_source_slip_clean_train_validation_dataset_20261013/manifest.json), 15 trainer-eligible train runs, 7 validation runs, 655,234 exported rows, no extraction errors; no test/final-test data were read.
- Guide-slip refit: [`training_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/greybox_guide_slip_clean_refit_20261013/training_report.json)
- Source-torque 1 kHz screen: [`training_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/greybox_guide_slip_exact_actuators_1khz_screen_20261013/training_report.json)
- Practice transfer report: [`fresh_practice_r03_r04_exact_actuators_1khz_20261013.json`](../../live_runs/derived_dynamics_learning_20260928/command_only_750ms_rollout_20261013/fresh_practice_r03_r04_exact_actuators_1khz_20261013.json)

Code changes are limited to the offline greybox, its tests, and the contact
slip analysis. Eight model tests pass, including documented slip equations,
the low-speed regularization, the 428/4 N m torque mapping, and zero-throttle
wheel stop. No production Odom/MPC behavior changed. The full progress and
interpretation are in the [Unity source audit](UNITY_WHEELCOLLIDER_PHYSX_SOURCE_AUDIT_20261012.md).

## Follow-up: exact autonomous-steering actuator semantics — 2026-10-13

The 1 kHz capture exposed a source/model mismatch in the 3-D rollout: its
default steering path rate-limited the physical steering angle symmetrically,
but `VehicleController.Steer()` does not implement that law. In autonomous
mode it compares normalized `AutonomousSteering` against the sign-inverted
internal `lastSteeringAngle / SteeringLimit`, then increments/decrements and
clamps using direction-specific comparisons. In particular, some same-side
unwinds snap directly to the target rather than respecting the nominal
183.346 deg/s rate. This is a source behavior, not a fitted steering gain.

I replayed that exact C# branch against the new 300,000-sample r12g source
capture, initializing from the captured wheel angles and using each recorded
normalized command. Inverting both Ackermann front-wheel angles to recover the
center angle gave `3.9e-6 deg` overall RMSE and `6.9e-7 deg` RMSE on command
changes. A symmetric rate-limited actuator had `1.43 deg` overall RMSE and
`1.42 deg` RMSE on command-change samples. This is direct source-to-capture
verification, not a fitted model result.

The source plant now defaults to the exact autonomous-steering update. Its
capture timing is respected: the recorded wheel angle is the angle used by
the current physics step; the current command computes the next boundary's
steering state. Center steering is initialized by inverting both source
Ackermann wheel-angle equations rather than averaging the two angles; at full
lock, the old average was biased by up to `0.84 deg`. This inverse is covered
by tests at both steering signs and several magnitudes. No Unity physics,
competition runtime Odom/MPC, or reference trajectory has been changed. A new
r07-trained/r12g-held-out coupled rollout using both exact source details is
reported below.

The corrected model was trained only on the reset-free r07 capture (8,491
25 ms transitions) and evaluated on r12g. The 300 s r12g source capture
contains seven reset teleports; all 5 s windows crossing a teleport were
rejected, leaving 44 contiguous windows from one independent run. At 5 s:

| Candidate | Position RMSE | Heading RMSE | Forward-axis velocity RMSE | Yaw-rate RMSE | Four wheel-speed RMSE |
| --- | ---: | ---: | ---: | ---: | --- |
| Source-equation baseline | `0.816 m` | `0.0382 rad` | `0.0544 m/s` | `0.0393 rad/s` | `0.0503, 0.0557, 0.0580, 0.0533 rad/s` |
| r07 yaw DAgger residual | `0.303 m` | `0.0203 rad` | `0.0547 m/s` | `0.0241 rad/s` | `0.0586, 0.0674, 0.0631, 0.0581 rad/s` |
| r07 coupled u/v/yaw residual | `0.383 m` | `0.0180 rad` | `0.0135 m/s` | `0.0146 rad/s` | `0.0552, 0.0647, 0.0607, 0.0551 rad/s` |

The exact actuator correction roughly halves the source-baseline 5 s position
error compared with the previous symmetric-steering implementation (`1.612 m`
on this same capture). The yaw residual then reduces the corrected baseline
position error by 63%; the coupled residual reduces forward velocity error by
75% and yaw-rate error by 63%. Neither candidate improves all wheel-speed
errors, and position still drifts by 0.30–0.38 m in five seconds. These are
44 windows from one capture, not 44 independent-run estimates. The residuals
remain offline research candidates and are not promoted to Odom/MPC.

The full-command-profile model was also fit on r07 and scored on the same r12g
holdout. At 5 s, profile-DAgger gives `0.291 m` position, `0.0189 rad`
heading, `0.0546 m/s` forward-axis velocity, `0.0197 rad/s` yaw rate, and
wheel-speed RMSE `[0.0557, 0.0655, 0.0623, 0.0553] rad/s`. The simpler
first-pass yaw residual gives `0.303 m`, `0.0203 rad`, `0.0547 m/s`,
`0.0241 rad/s`, and `[0.0586, 0.0674, 0.0631, 0.0581] rad/s`. Thus the
full command profile buys only `0.012 m` in position at 5 s versus the
non-profile first pass; forward-speed error is unchanged, and its yaw-rate
error is higher than the non-profile DAgger result (`0.0197` vs
`0.0173 rad/s`). This is a small tradeoff, not a successful full-plant model.
The later r11 and r12g recursive comparisons are recorded in the follow-up
section below.

Also tested whether a shared combined-slip ellipse explains the remaining tire
force mismatch. Using the existing r07-only force-curve fit with oracle
captured WheelHit slips and loads, force-vector RMSE was `2.353 N` on r07 and
`2.436 N` on r12g; held-out yaw-moment RMSE was `0.3732 N m`. Fitting a shared
ellipse scale on r07 returned `1.449` (above the unmodified cap) and changed
r12g force RMSE only from `2.436` to `2.434 N`, with effectively unchanged
yaw-moment error. The shared ellipse hypothesis is therefore not a useful
correction and has not been integrated. Since this analysis uses oracle slip
and loads, the remaining error cannot be attributed only to predicting those
states; the force-curve/force-balance closure itself remains imperfect.

## Follow-up: documented slip equations and recursive transfer — 2026-10-13

The guide's longitudinal/lateral slip equations were implemented as an
isolated offline option and compared against the existing contact-speed slip
coordinates. On captured WheelHit diagnostics, the guide's lateral ratio has
larger error than the native-contact reconstruction, especially at 3–6 m/s
(front-wheel RMSE around `0.03–0.064`, versus `0.0003–0.00056` for the
regularized contact reconstruction). A force-balance refit using the guide
coordinates reduced held-out r11 force residual (`6.052` to `1.876 N`) but
did not transfer to recursive motion: at 5 s it increased position RMSE from
`1.865` to `4.241 m`; fitting the guide tire parameters made it `4.841 m`.
On r12g the same alternatives increased 5 s position error from `0.816` to
`1.715–2.028 m`. The documented slip law therefore remains a useful
diagnostic, not a replacement for the current plant's slip coordinates.

The first recursive report for r12g accidentally included four rollout
windows crossing explicit simulator reset teleports (position jumps over
`0.05 m` between adjacent 1 ms samples). It is superseded by
`documented_tire_slip_recursive_comparison_clean_20261013.json`, which rejects
any such window; the clean counts are 39 r11 windows and 44 r12g windows.
No tire/slip candidate was promoted.

The exact Unity wheel-axis audit found the source Ackermann directions agree
with captured WheelHit frames within `0.10–0.24 deg`; contact-point residuals
are about `8–10 mm`, but a constant contact offset had already worsened force
and yaw-moment prediction. Thus neither a gross steering-frame error nor a
simple contact-point shift explains the remaining recursive drift.

A second sequential DAgger round was fit on r07 only (52 five-second windows
per round; 10,400 residual labels each), with r11 and r12g excluded from
training. It was scored on both held-out driving captures. At 5 s, round 2
changes r11 position/heading/yaw-rate RMSE from the round-1
`0.806 m / 0.0576 rad / 0.0130 rad/s` to
`0.782 m / 0.0561 rad / 0.0243 rad/s`; on r12g it changes
`0.291 m / 0.0189 rad / 0.0197 rad/s` to
`0.238 m / 0.0117 rad / 0.0183 rad/s`. Thus it improves pose and yaw-rate
relative to the source-physics baseline on both captures, but versus round 1
it trades a material r11 yaw-rate regression for modest r11 position gain and
better r12g pose. No single round dominates all states and captures. Do not
promote round 2 as the general model.

The attempted extra evaluation on r12b–r12f failed eligibility for a concrete
reason: all five captures are stationary, with maximum recorded body speed
`0.027 m/s`; they have no fully grounded 5 s driving windows. Their bridge
logs show no connection for r12b/c/f and a lost Socket.IO connection for
r12d/e, followed by neutral-command latching. These files are retained as
connection diagnostics, not vehicle-training or validation data; their
machine-readable exclusion is in
[`stationary_capture_exclusion_audit_r12b_f_20261013.json`](../../live_runs/unity_exact_binary_audit_20261012/stationary_capture_exclusion_audit_r12b_f_20261013.json).
The older
2026-09-27 source-player wheel CSV has high-speed/high-steer content but is
already documented as unpaired and failing pinned-player behavior equivalence,
and its body inertia differs materially from r07; it is not admitted to fitting
or promotion.

A reusable score-only tool,
[`score_unity_3d_yaw_residual_bundle.py`](../../tools/vehicle_dynamics_learning/score_unity_3d_yaw_residual_bundle.py),
has been added to compare saved bundles on additional whole-run captures
without refitting. The intended next validation uses the already-captured
r12b–r12f data only if future runs actually contain driving; they remain
excluded at present. The prior r07-trained profile candidate remains the
better balanced demonstrated result: at 5 s it reduces
r11 position RMSE `1.865→0.806 m` and r12g `0.816→0.291 m`, with yaw-rate RMSE
`0.0851→0.0130 rad/s` and `0.0393→0.0198 rad/s`, respectively. Forward-speed
error is essentially unchanged, and wheel-speed errors increase slightly.
This is a meaningful yaw/pose gain, but not yet accurate enough to claim a
reliable full-lap offline simulator or production odometry improvement.

## Unity source equations and deterministic runtime parity — 2026-10-13

The offline-plant direction is source-first, not RL-first. Reproduce the
public Unity C# controller, encoder, IMU, serialized setup, and explicit
rigid-body equations directly. The `WheelCollider` force/contact implementation
is native Unity engine code and is not present in the C# project; for that
boundary, use the matching Unity runtime as the exact teacher/oracle rather
than replacing it with a learned model and calling it source-faithful.

The first source-player-to-pinned-Explore command replay (`r02`) exposed a
test-harness flaw, not a vehicle-model result: 10,406 commands were replayed,
but ROS callback scheduling reached `130.283 ms` lateness and packet-matched
sent commands had `0.04157` steering-normalized and `0.02050` throttle-normalized
RMSE. Although the run had zero collisions, zero timing faults, and median
packet spacing `24.962 ms`, its `8.92 m` relative-position RMSE is ineligible
for physics conclusions because the applied commands did not match exactly.

To remove that ambiguity, the bridge now has an opt-in development profile
mode that injects the captured sent-command pair synchronously at each existing
40 Hz request boundary. The source-vs-pinned Explore replay (`r03`) completed
with exact packet-aligned commands, zero collisions, and zero timing faults;
it established the bridge injection path but still showed nonzero long-run
cross-player state divergence.

The more literal offline test now builds the Unity 2022.3.52f1 Explore player
from the public source and runs the same 10,487 sent-command pairs directly at
25 x 1 ms fixed steps per pair. It completed all 262,175 steps with no command
error and no overwritten 1 kHz samples. Against source run r07, same-phase
body-velocity RMSE is `0.00128 m/s`, yaw RMSE `0.00748 rad`, yaw-rate RMSE
`0.00468 rad/s`; relative position RMSE is `0.443 m` and endpoint error
`0.259 m` over 262.2 s. The position reference was corrected using the
serialized rear-axle GPS transform, not a fitted lever arm. Error grows in the
last high-speed portion, so this proves a practical source-runtime offline
plant—not negligible full-run/lap error yet. Full run notes, captures, and
reproduction details are in
[`UNITY_WHEELCOLLIDER_PHYSX_SOURCE_AUDIT_20261012.md`](UNITY_WHEELCOLLIDER_PHYSX_SOURCE_AUDIT_20261012.md)
and [`unity_source_exact_offline_plant_20261013_r01`](../../live_runs/unity_source_exact_offline_plant_20261013_r01/).

No learned residual, odometry/MPC runtime, or simulator physics was changed.
This is a development-only offline plant/replay path. The practice command
replay and an initial timing-phase check are now complete; full-run pose
divergence remains unresolved, so no production odom or raceline-optimizer
gain is claimed.

## Source-built practice-scene replay — 2026-10-13

The source-first approach has now been exercised in the practice race scene,
not just the Explore scene. A Linux player built from the Unity project ran
the captured practice commands with the exact Unity 2022.3.52f1 runtime,
serialized race-scene vehicle setup, 1 ms fixed step, and the active practice
track collision mesh exported from the official simulator build. The vehicle
and physics settings were not changed. The replay applies each captured
command for 25 physics steps; all 2,994 command pairs matched exactly and all
74,850 fixed steps completed with zero course collisions.

Against the 12-lap practice capture, the replay's body-velocity vector RMSE is
`0.00235 m/s`, yaw-rate RMSE is `0.00076 rad/s`, and yaw orientation RMSE is
`0.0211 rad`. Relative-position vector RMSE is `0.0939 m` (p95 `0.1858 m`),
with `0.1049 m` endpoint error. Per-lap local path-increment RMSE ranges from
`0.0040` to `0.0221 m`; these small local errors accumulate into the larger
global pose discrepancy. One missing 25 ms state sample is explicitly
accounted for at packet indices 996–997. The capture phase (`-26` fixed
steps) comes from a separate timing diagnostic; it is not a fitted vehicle
parameter.

This is a real implementation of the simulator's mechanics, not a learned
surrogate: the source `VehicleController` and Unity's native WheelCollider
solver execute during the replay. It demonstrates a useful source-backed
offline plant, but does not yet establish negligible full-lap pose error,
improved production odometry, an MPC improvement, or faster raceline. No
production odom/MPC or runtime physics setting was changed. Reproduction,
capture, and detailed metrics are in
[`UNITY_WHEELCOLLIDER_PHYSX_SOURCE_AUDIT_20261012.md`](UNITY_WHEELCOLLIDER_PHYSX_SOURCE_AUDIT_20261012.md)
and [`unity_source_practice_scene_command_replay_20261013_r07`](../../live_runs/unity_source_practice_scene_command_replay_20261013_r07/).

## Source settings audit and command-phase check — 2026-10-13

I extracted the official practice image's `globalgamemanagers` settings
read-only and compared them with the Unity source project. Both select
`Assets/Scenes/RoboRacer - Sim Racing.unity` and Unity `2022.3.52f1`. The
1 ms fixed step, gravity, contact offset, bounce/depenetration settings,
6/1 solver iterations, friction/solver type, max angular speed, and other
serialized global physics values match. The official scene's `VehicleController`
is the `Assembly-CSharp.VehicleController` script; its 3.47 kg rigid body,
four 59 mm/109 g wheel colliders, 500 N/m springs, 100 Ns/m dampers, and
forward/sideways friction curves also match the source-built replay runtime.
The extracted settings file and its hash are preserved in r08's
`official_assets/` directory.

To test the remaining command/physics boundary without changing vehicle
physics, I added a replay-only `command_phase_fixed_steps` setting restricted
to 0 or 1 ms. The 1 ms candidate replayed the same 2,994 packet commands for
74,851 fixed steps, exact command match, and zero course collisions. It did
not improve parity: relative-position RMSE changed `0.09386→0.09561 m`,
yaw RMSE `0.02109→0.02201 rad`, body-velocity RMSE
`0.002354→0.004254 m/s`, and yaw-rate RMSE `0.000758→0.001193 rad/s`.
The original zero-phase replay remains the reference. This rules out a blanket
one-fixed-step command delay as the correction; no learned correction or
production change was made.

The concrete remaining task is source-level chronology: reconcile the
recorded packet/state snapshot with Unity's `Socket.OnBridge` callback,
queued telemetry, and `VehicleController.FixedUpdate`, then diagnose the
per-lap pose/yaw accumulation on the same source-built plant. Do not replace
the Unity dynamics with RL or fit a residual merely to reduce the current
12-lap score.

## Exact source-plant repeatability — 2026-10-13

I reran the original zero-phase r07 profile as r09 from a fresh batch-mode
player process, using the same source-built executable, official practice
image/container, course collider, and 2,994-command input profile. Both runs
completed 74,850 fixed steps with zero course collisions. Every numeric
1 kHz capture field compared exactly across the two runs: simulation time,
pose/quaternion, body linear/angular velocity, commands, and all 28 captured
wheel fields (RPM, steering, motor torque, contact/slip/force). Maximum
absolute difference was `0` for every field.

This is strong evidence that the source-built Unity plant is deterministic and
reproducible for this fixed initial scene and command stream. It is therefore
usable as a genuine offline plant/evaluator without an RL or learned-dynamics
replacement. It does **not** erase the separate comparison against the
practice capture: that cross-run comparison remains at `0.09386 m` relative
position RMSE and `0.02109 rad` yaw RMSE over 12 laps. The remaining work is
to identify the cross-run initial-state, command-to-fixed-step, or public
source-vs-official-binary difference; no single one is yet proven. The
repeatability capture is preserved at
[`unity_source_practice_scene_command_replay_20261013_r09_repeat_zero_phase`](../../live_runs/unity_source_practice_scene_command_replay_20261013_r09_repeat_zero_phase/).

## Unity actuator-branch Odom: corrected recursive A/B — 2026-10-13

I re-scored the two saved production-node replay outputs against the same
12-lap practice bag, keyed by exact sensor source stamp. This resolved a
misleading first interpretation of the output-directory names: the replay
with `use_unity_actuator_u_experts=true` reproduces the run's recorded
`/odom` output, while the replay with the parameter explicitly false is the
ablation. The false replay is not the production baseline.

On the 2,464 scored samples (72.09 s; truth used offline), enabling the
source-conditioned experts changes forward-speed RMSE from `0.04428` to
`0.01459 m/s` (67.1% lower), with lateral-speed and gyro-yaw-rate outputs
unchanged. One-alignment continuous-run position RMSE changes from `0.22389`
to `0.12145 m` (45.8% lower); endpoint error changes from `0.45113` to
`0.21130 m` (53.2% lower). The enabled replay matches the recorded production
Odom's scored state errors exactly. The yaw error is about `8.1e-5 rad` RMSE;
the remaining pose drift is predominantly longitudinal, not yaw.

This was not only a bag-only candidate: the matching recorded run completed
one warmup, ten scored practice laps (mean `5.9615 s`, best `5.9128 s`), and
one extra lap, with zero collision delta. It verifies that the enabled Odom
configuration operated through a full 12-lap run. The current speed and pose
errors are still nonzero, and this evidence covers one practice run rather
than multiple independent configurations; it does not yet meet the broader
near-zero-error goal.

Accordingly, `sensor_odometry.yaml` keeps the experts enabled and retains the
1 ms feedback-settle interval used by the successful run. The observer uses
rear encoders, IMU, and the permitted steering/throttle feedback only; no
simulator truth or restricted pose topic is a runtime input. The topic-policy
check passes. The model is confined to its measured 2–8 m/s support band.
The model-off comparator and corrected A/B scores are preserved under
[`odom_unity_actuator_u_experts_practice_20261013_r01`](../../live_runs/odom_unity_actuator_u_experts_practice_20261013_r01/analysis/).

Next Odom work: validate this same enabled configuration on independent
existing practice captures and diagnose the residual longitudinal drift by
first divergence. Do not retune MPC or the raceline on the basis of this
Odom-only result.

## Lateral-acceleration integration A/B — 2026-10-13

I replayed the same captured 12-lap practice run through the production Odom
node with only `integrate_lateral_acceleration_in_turn` overridden to `true`.
The reference is the same run's recorded production `/odom`, which exactly
matches the current enabled configuration replay. Scoring used the same 2,464
scored source-stamped samples and one initial SE(2) alignment; no simulator
truth was supplied to the observer.

This candidate fails decisively. Forward-velocity RMSE rises from
`0.01459` to `0.87379 m/s`, lateral-velocity RMSE from `0.01430` to
`0.44010 m/s`, while yaw-rate error is unchanged. Continuous one-alignment
position RMSE rises from `0.12145` to `4.46080 m`; endpoint drift rises from
`0.21130` to `9.67008 m`. Its first lap endpoint error is already `0.178 m`
versus `0.036 m` for the production replay, and error grows through the run.
The experiment is preserved under
[`lateral_acceleration_turn_candidate`](../../live_runs/odom_unity_actuator_u_experts_practice_20261013_r01/analysis/lateral_acceleration_turn_candidate/).

The production parameter remains `false`. This rejects the full candidate
configuration; it does not yet distinguish whether the main cause is direct
integration of the 40 Hz lateral-acceleration signal, its accumulation across
long turns, or its interaction with downstream speed residuals that were fit
using the existing causal lateral-speed estimate. A clean follow-up must
isolate that interaction before any revised lateral model is considered.

## Ground-truth source audit and current Odom work — 2026-10-14

The repository explicitly defines `/autodrive/roboracer_1/odom` as the
simulator development truth oracle; it is not the production observer output.
Therefore earlier fits and scores that use `MotionSample.state` from that
topic are not self-comparisons and are not invalidated. The production node
continues to consume only encoders, IMU, and permitted actuator feedback.

For clearer Unity-physics semantics, the current fit and replay scorer now
also support labels derived directly from the simulator's bridge rigid-state
packet: body-frame COM velocity and angular rate are converted to rear-axle
`[u,v,r]` with `v_rear = v_com - 0.15532*r`; the simulator pose packet is the
pose reference. Source-parity audits over 386,045 valid r04 samples and
369,937 valid r05 samples found exact equality between this rigid-state
conversion and the documented simulator `/autodrive/.../odom` twist on all
three axes. The two pose sources differed by `0 m` direct XY RMSE and yaw
RMSE of `0.000431 rad` (r04) / `0.000443 rad` (r05). Re-scoring held-out
reversal capture r03 with the
packet reference reproduced its previous twist RMSE exactly:
`[0.070870, 0.004534, 0]` for `[u,v,r]`; relative-pose metrics changed only at
the sub-millimetre level. These checks confirm the prior GT-based scores.

The current-source Odom baseline batch is complete for all ten short
training/holdout captures and the two throttle sweeps (r04 training, r05
held-out transfer). Each replay rebuilt the current localization source in an
isolated container and consumed only encoder, IMU, steering-feedback, and
throttle-feedback inputs. All ten short runs have the same Odom source hash.
The R03 replay used the clean `R03_retry` output; an earlier output was not
scored because its recorder had not closed.

The current-source fit uses a 34-feature affine correction derived from Unity
actuator/rigid-body quantities and production Odom diagnostics. Simulator
rigid-state truth is the offline target only. Training-only leave-one-run-out
selection chose ridge alpha `100`, correction gain `0.8`, throttle split
`0.15`, and Odom-speed limits of `[3.2,3.4] m/s` below 15% throttle and
`[3.2,4.6] m/s` at or above 15% throttle. Steering support is `(0.30,0.50]`
rad and throttle support `[0.11,0.18]`. The gate was selected with a 5%
maximum regression allowance in populated joint speed/throttle bins; the
chosen policy improved every one of the seven training folds and had no
joint-bin regressions. A first scalar-gate fit exposed a bad low-throttle
3.5–3.54 m/s correction; the throttle-conditioned support removes that
unsupported overcorrection while retaining the high-throttle branch.

Untouched pointwise validation improved in the supported target region:
L03 `0.0608 -> 0.0579 m/s` RMSE (`p95 0.0681 -> 0.0525`), R03
`0.0415 -> 0.0246` (`p95 0.0707 -> 0.0481`), and throttle repeat T05
`0.0580 -> 0.0467` (`p95 0.0690 -> 0.0203`). H01 has no samples in the gate.
The model changes only published forward speed; lateral velocity and yaw-rate
predictions are unchanged.

The actual C++ Odom node has now been recursively replayed with the candidate
on L03 and R03, with a paired current-source baseline/candidate replay running
on T05. Both completed recursive replays improved full-capture errors without
changing lateral velocity or yaw rate:

| Holdout | Forward-speed RMSE (m/s) | Relative-position sequence-macro RMSE (m) | Lateral-speed / yaw-rate RMSE |
|---|---:|---:|---:|
| L03 | `0.15518 -> 0.15493` | `0.08729 -> 0.08694` | unchanged: `0.02355 / 0` |
| R03 | `0.07088 -> 0.06765` | `0.04323 -> 0.04024` | unchanged: `0.00453 / 0` |

These full-run improvements are smaller than the selected-regime gains because
the gate correctly leaves unsupported samples on the existing observer. The
paired T05 recursive result is still pending. The candidate has not yet been
enabled in production YAML; a real practice batch-mode check and one
sub-agent acceptance are still required before completing Odom Stage 1.

The existing 12-lap practice capture has 2,976 source samples, zero collisions,
and peaks at `0.3068 rad` absolute steering; none of its samples satisfy the
candidate's joint speed/steering/throttle gate. It can verify no regression on
the current conservative line, but cannot demonstrate that the high-steering
expert improves practice tracking. Integrated-pose improvement must first be
shown on independent high-steering simulator captures; any later track claim
must identify whether the expert actually activated.

No production Odom change has been made in this stage yet. Stage 1 is not
complete or subagent-reviewed; raceline and MPC work remains deliberately
untouched until the Odom acceptance evidence exists.

### High-yaw wheel-slip observer recheck — 2026-10-14

The current-source replay of training capture W03 with
`use_high_yaw_wheel_slip_observer:=true` is now complete. On its 17,285
matched 25 ms samples, forward-speed RMSE falls from `1.988` to `1.052 m/s`
overall; within the 3,548 samples at `|yaw rate| >= 2.5 rad/s`, it falls from
`3.471` to `1.077 m/s`. In the 644-sample negative-body-speed subset, it falls
from `6.579` to `0.435 m/s`. Lateral-speed and yaw-rate scores are unchanged;
the per-sequence relative-position macro mean falls from `0.819` to `0.611 m`
while its maximum changes from `2.305` to `2.319 m`.

This is a strong result on the W03 training capture, not independent evidence
for deployment. The already-completed held-out wheel-spin validation replay
had identical baseline/candidate outputs (zero changed state samples), so the
observer did not activate there. The feature remains disabled in production;
its threshold/support must be diagnosed and independently exercised before
promotion. Scoring artifacts are `baseline_W03_fixed25ms.json` and
`W03_highyaw_score_fixed25ms.json` in the 2026-10-14 analysis directory.

### High-steer expert live validation — 2026-10-14

The high-steer expert's runtime implementation has been exercised in the
production Odom node as a sensor-only sidecar. Ground truth was captured for
offline scoring only. Runtime node inspection confirmed subscriptions only to
the two encoders, IMU, steering feedback, and throttle feedback; no simulator
truth topic was consumed by Odom.

The first live probe used 3.5 m/s target speed and reached about 3.43 m/s.
It was safe (24/24 scheduled phases, zero collisions, zero timing/quality
failures, 39.98 Hz sensor streams), but the expert's low-throttle upper speed
gate was 3.4 m/s: only one baseline sample and zero candidate samples were
inside the gate. This is non-regression evidence only, not a test of the
correction. The probe profile was made to use its requested speed parameter,
and a second short run was targeted below the gate boundary.

The retargeted live run requested 3.4 m/s and measured 3.314–3.318 m/s. It
completed 24/24 phases with zero collisions, zero timing/quality failures,
and 39.60 Hz command output. On the same 714 valid samples, exact production-
node baseline and candidate replays show:

| Metric | Baseline | Candidate | Result |
|---|---:|---:|---:|
| Forward-speed RMSE, all samples | 0.05203 m/s | 0.04972 m/s | 4.4% lower |
| Forward-speed RMSE, 391 gated samples | 0.02123 m/s | 0.00465 m/s | 78.1% lower |
| Gated forward-speed absolute p95 | 0.04404 m/s | 0.00915 m/s | 79.2% lower |
| Relative-pose RMSE, phase-macro mean | 0.04211 m | 0.04472 m | 6.2% worse |
| Worst phase relative-pose RMSE | 0.05836 m | 0.06409 m | 9.8% worse |

The expert therefore improves its intended forward-speed estimate in the
measured regime, but its recursive pose integration regresses slightly and
consistently across the eight reset-isolated sequences. It is **not promoted**
to the production YAML. The first run's boundary mismatch also showed why a
successful run alone is not enough: the correction must be confirmed active.
This pose tradeoff must be diagnosed against training-only data and the long
held-out T05 replay before making a production decision. Both live run bags,
paired replay outputs, and fixed-25-ms score JSONs are retained under
`live_runs/odom_highsteer_live_v6_20261014_r01` and `_r02`.

The training-derived model support is absolute steering `(0.31, 0.50]` rad,
throttle feedback `[0.11, 0.18]`, and published Odom speed `[3.2, 3.4]` m/s
below 15% throttle or `[3.2, 4.6]` m/s otherwise. The earlier report's prose
gate threshold was stale at `0.30`; its model payload and fitter use `0.31`,
and the report text is corrected. The production configuration still keeps
the expert disabled. Raceline and MPC changes remain untouched during this
Odom gate.

### Odom stage implementation and fresh held-out validation — 2026-10-14

The 0.20-gain high-steering forward-speed expert has now been enabled in the
production `sensor_odometry.yaml` and exercised by the production C++ sensor
Odom node on the Explore simulator. The node received encoder, IMU, steering,
and throttle-feedback inputs only; simulator truth was recorded for offline
scoring. `tools/verify_runtime_topic_policy.py` passes. The active production
configuration is the main localization YAML, not the separate A/B overlay.

The first launcher attempt (`r01`) produced no driving: the experiment
launcher defaulted this short profile to 3600 s although the driver accepts at
most 1200 s. Its small empty bag was discarded; startup logs remain in the
run directory. The follow-up 4.5 m/s run (`r02`) exposed a second test setup
mistake: a generic speed default overrode the profile's intended point. It
completed safely and is retained as an extrapolation check, not as evidence
for the in-support fit. The launcher now defaults this profile to 3.3 m/s and
1200 s; explicit speed overrides remain available.

At 3.5 m/s (`r03`), measured throttle was about 0.14, so the expert's
low-throttle published-speed ceiling of 3.4 m/s left it active in only one of
eight scored sequences. This was an edge/gate diagnostic, not the decision
run. The corrected 3.3 m/s run (`r04`) reached 3.20–3.31 m/s and completed all
24 scheduled phases with eight valid reset-isolated sequences, no collision,
no timing fault, no quality failure, 39.60 Hz commands, and approximately
40 Hz sensor streams. The expert changed 49–50 samples in each sequence.

The fixed-25-ms paired baseline and candidate replays used the same `r04`
sensor packets, production C++ implementation, and all other existing Odom
experts; only the high-steer expert gain was set to 0 versus 0.20. The 0.20
candidate was selected from training captures before this fresh run. Results:

| Odom output/error | Baseline | 0.20 expert | Change |
|---|---:|---:|---:|
| Forward-speed RMSE, all 716 samples | 0.05264 m/s | 0.05207 m/s | 1.08% lower |
| Forward-speed RMSE, `|steering| >= 0.40 rad` (370 samples) | 0.01816 m/s | 0.01503 m/s | 17.2% lower |
| High-steer gain, mean over eight independent sequences | — | — | 17.2%; sequence-bootstrap 95% interval 16.0–18.5% |
| Forward-speed gain by whole sequence | — | — | improved in all eight; 0.55–1.44% |
| Lateral-speed RMSE | 0.003394 m/s | 0.003394 m/s | unchanged |
| Yaw-rate RMSE | 0 rad/s | 0 rad/s | unchanged (gyro passthrough) |
| Relative-position sequence-macro RMSE | 0.041566 m | 0.041852 m | +0.286 mm / +0.69% |
| Relative-position p95 / maximum | 0.056933 / 0.074215 m | 0.056905 / 0.074214 m | effectively unchanged; both slightly lower |

The position mean is therefore not claimed as an improvement: its 0.286 mm
increase is small relative to the 4.16 cm baseline RMSE, while p95 and maximum
are unchanged to sub-0.03 mm. The supported high-steering speed estimate is a
repeatable improvement across all eight fresh sequences, and the other
reported motion states do not materially regress. The 4.5 m/s extrapolation
run also reduced speed error slightly, but it is not used to assert support
beyond the training range.

Artifacts are retained under:

- `live_runs/odom_highsteer_gain02_production_validation_20261014_r02/`
  (4.5 m/s extrapolation check)
- `live_runs/odom_highsteer_gain02_production_validation_20261014_r03/`
  (3.5 m/s gate-edge diagnostic and gain replays)
- `live_runs/odom_highsteer_gain02_production_validation_20261014_r04/`
  (in-support held-out production capture, exact paired replays, and scores)

The high-steer expert is an actual Odom improvement in its supported regime,
but it is not a broad full-spectrum or full-lap odometry solution: the
existing practice line peaks below its 0.31 rad steering gate, so this change
does not activate there. Stage 1's evidence is prepared for the requested
independent review; the reviewer must decide whether the bounded speed gain
with effectively unchanged position error satisfies the Odom-stage criterion.
No raceline or MPC changes have been made while that decision is pending.

### Stage 1 completion gate — 2026-10-14

Independent subagent review: **ACCEPT**. Stage 1 is complete for this task's
bounded requirement: the production Odom path has a sensor-only, supported
high-steer speed correction with a repeatable real-simulator accuracy gain
and no material regression in integrated position. Position is considered
effectively neutral, not improved. Do not repeat this validation. Proceed to
Stage 2 (raceline optimization); full-spectrum Odom and practice-line transfer
remain known limitations, not claims of completion.
