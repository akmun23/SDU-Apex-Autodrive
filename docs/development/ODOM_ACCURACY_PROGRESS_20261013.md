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

### Source-plant Odom investigation — 2026-10-16

This is a new, source-derived Odom investigation; it does not undo the prior
bounded Stage 1 review and is not a claim that deployed Odom is now complete.
It tests the proposed architecture directly: hold each known 40 Hz actuator
command for 25 Unity physics ticks, propagate the source-grounded plant at
1 kHz, and use permitted current sensor state to anchor the estimate.
Simulator truth is used only offline for initialization and scoring.

Two separate official Unity batch-mode captures were replayed with the native
PhysX plant, source vehicle parameters/shapes, and reconstructed
`VehicleController` equations. Future capture states were not included in the
rollout inputs:

| Unity capture | Coverage | Recursive rollout position RMSE / p95 | Yaw RMSE | Forward-speed RMSE |
|---|---|---:|---:|---:|
| `unity_source_time_indexed_baseline_20261014_r01`, 13.875 s | two lap crossings; mean 4.08 m/s, peak 7.79 m/s, peak physical steer 17.63° | 0.267 / 0.610 mm | 0.0000395 rad | 0.00000462 m/s |
| `unity_r13_empirical_full_lap_focus_k3_20261015_r01`, 14.700 s | two lap crossings; mean 3.94 m/s, peak 7.90 m/s, peak physical steer 31.99°; source run reports zero collisions | 0.436 / 0.854 mm | 0.000101 rad | 0.0000195 m/s |

These are strong whole-run plant-parity results for the tested motion, not
production Odom results: both runs initialize from the captured complete
vehicle state and the Unity batch captures provide the exact 40 Hz command
holds. They establish that this source-derived plant can recursively predict
body motion and wheel spin in those runs without a learned force fit.

The command clock is critical. On the aggressive capture, an offline replay
that deliberately delayed each command by one 25 ms packet changed position
RMSE from 0.436 mm to 105 mm, yaw RMSE from 0.000101 to 0.0367 rad, and
forward-speed RMSE from 0.0000195 to 0.0893 m/s. This diagnostic perturbation
is not a proposed runtime delay. It confirms that the simulator command
active for each physics interval must be paired causally; a generic
latest-value snapshot or late actuator feedback is not a safe substitute.

On the existing 42.75 s reset-centered `r12g` capture, command-aligned native
rollout errors are 0.183 m position RMSE (0.293 m p95), 0.00351 rad yaw RMSE,
0.00362 m/s forward-speed RMSE, and 0.000832 m/s lateral-speed RMSE. The
largest instantaneous yaw residual is a near-stop, four-wheel-braked steering
transient; it is sparse and does not explain the full position drift. For a
bounded fusion test, native predicted body velocities were integrated at the
40 Hz packet boundaries using the allowed IMU orientation as the yaw anchor.
This gives 18.8 mm position RMSE, 31.2 mm p95, and 34.5 mm maximum over
42.75 s, versus 239.1 mm RMSE for the current production sensor Odom replay
on the same window. A denser 1 kHz integration variant was worse (32.6 mm
RMSE), so it is not selected. This is an offline candidate result, not a
production change or a live confirmation.

The same yaw-anchored integration was then repeated with the ordinary ROS
`steering_command` and `throttle_command` streams only—without using the
packet-timing debug topic to choose the command for each simulator request.
That candidate scores 76.1 mm position RMSE, 100.0 mm p95, and 103.6 mm
maximum; its one-step plant forward-speed RMSE is 0.00940 m/s and lateral
speed RMSE is 0.00272 m/s. Against the production replay's 239.1 mm position,
0.1408 m/s forward-speed, and 0.00795 m/s lateral-speed RMSE, this is a
promising offline reduction of 68%, 93%, and 66%, respectively. It still does
not meet the requested near-zero/full-run accuracy, and its 7.6 cm pose error
is not a deployable result by itself. The stronger 18.8 mm result requires
offline request-aligned command reconstruction; it is a diagnostic upper
bound, not a permissible runtime input path.

Artifacts are in
`live_runs/odom_physx_observer_stage1_20261016/`; native rollout inputs and
metrics retain each exact command sequence and scoring result. The
`packet_aligned` experiment uses `/bridge_packet_timing` strictly as an
offline-only diagnostic to reconstruct the legal command sent on each
simulator request. That topic is not subscribed to by Odom, MPC, AMCL, or EKF
at runtime.

**Current Odom conclusion:** the Unity-source plant is demonstrably useful
for predicting full-run motion when its initial state and command phase are
correct. The remaining task is to make that information available causally
to production Odom using only permitted inputs, initialize/correct hidden
wheel and suspension state from permitted sensors, and verify the result in a
real ROS-driven simulator run. Until that is implemented and validated on an
independent practice run, no new production Odom improvement is claimed and
no work has moved to MPC or raceline optimization.

### Streamed-plant and permitted-IMU follow-up — 2026-10-16

The command/feedback/response streams were separated rather than treating
feedback as the command that drove the same physics tick. On the existing
`r12g` bag, each ordinary ROS command topic was sampled causally at the bridge
request time and compared offline with the bridge's exact sent-command
diagnostic. Within `0.001` normalized units, the ROS steering and throttle
topics match the sent command on 99.97% and 99.98% of requests, respectively.
The bridge packet topic was used only for this offline audit; it is not an
input to the candidate evaluator or production Odom. Reported actuator
feedback is a distinct, later response measurement and must not replace the
command timeline in a forward plant rollout.

The native public-PhysX replay was extended with a persistent line-streaming
mode in `tools/vehicle_dynamics_learning/physx_native_vehicle_step.cpp`. A
250-step, 1 ms stream was compared state-by-state against the existing batch
run from the same reset and command inputs: body position, orientation,
linear/angular velocity, and all wheel angular speeds matched exactly
(maximum absolute delta `0`). The process completed 250 ms of plant time in
29.4 ms wall time (`8.51x` real time), including process startup and scene
construction. The parity record is
`live_runs/odom_physx_observer_stage1_20261016/physx_streaming_batch_parity_250ms.json`.
This demonstrates a feasible persistent calculation core; it is not yet
linked into the ROS Odom node or the competition image.

The 42.75 s command-only rollout was then scored with the actual ROS IMU
orientation stream instead of substituting the source-capture quaternion.
`tools/vehicle_dynamics_learning/analyze_physx_odom_imu_anchor.py` samples the
latest IMU message causally, consumes only the two legal ROS command topics
and `/imu`, and uses the source capture only for offline scoring. On this run:

| State/result | Recursive source-plant candidate | Existing production Odom replay |
|---|---:|---:|
| Forward-speed RMSE | 0.00905 m/s | 0.14078 m/s |
| Lateral-speed RMSE | 0.00388 m/s | 0.00795 m/s |
| Position RMSE using actual ROS IMU yaw | 0.1603 m | 0.2391 m |
| Position p95 using actual ROS IMU yaw | 0.3054 m | 0.3011 m |

The actual-IMU-yaw position RMSE is 33% lower, but its p95 is not improved;
this is a mixed result, not a production-ready odometry win. The source plant's
own yaw has `0.00358 rad` RMSE against source truth, but integrating its yaw
alone gives `0.1971 m` position RMSE. The measured IMU yaw has `0.0636 rad`
RMSE and a `0.383 rad` maximum discrepancy; the bag contains a reset-period
gap with 35 of 1,710 scored samples older than 50 ms (maximum age 0.919 s).
The sensor is stationary during that reset gap, but the initial heading
transient must be explicitly handled and independently checked.

A convex blend of model and IMU yaw with gain `0.55`, selected on only the
first half by integrated position error, scores `0.1371 m` full-run position
RMSE and `0.1767 m` on the second-half temporal holdout; p95 full-run error is
`0.2425 m`. This is a promising **offline candidate**, not a promoted
production change: it is selected and evaluated on only one run, the holdout
is from that same run, and startup state is still the explicit reset seed
rather than a fully sensor-initialized PhysX state. A different gain chosen
only to minimize first-half yaw error is zero, so position and yaw metrics
select different fusion behavior. That conflict is why the gain is not being
silently installed in Odom.

The detailed reproducible output is
`live_runs/odom_physx_observer_stage1_20261016/r12g_ros_imu_anchored_odometry_metrics.json`.
The next Odom implementation step is to host the persistent plant behind a
bounded 40 Hz interface, initialize it from reset plus permitted live sensors,
and replay it against an independent held-out practice bag. Only after that
passes should its body velocity/yaw feed the production pose integrator. No
MPC or optimizer work has started.

### Practice-stack Odom score and handoff to MPC — 2026-10-16

The user's acceptance criterion is now explicit: AMCL may continue correcting
global pose; Odom is useful if its speed and yaw-rate outputs are accurate
enough for control. Odom is not required to remain globally drift-free by
itself. This supersedes the earlier AMCL-free pose requirement for this
assessment. I rescored the already-recorded production-stack run on the
current working practice raceline; no simulator was launched.

Run provenance and score:

- Bag: [`run_0.db3`](../../live_runs/practice_y1_reference_current_runtime_20261014/run/run_0.db3)
- Raceline: `y1_matched_9g_20261012_r04_025/autodrive_mintime_raceline_wallchecked.csv`
- Runtime image: `sdu-apex-autodrive:practice-y1-full-model-20261014`
- The run completed eight scored laps (mean `5.7583 s`, best `5.6118 s`) and
  then recorded one collision. Only pre-collision scored-lap samples are used
  for the speed/yaw assessment below.

Across the 1,840 scored-lap samples, production `/odom` has forward-speed
RMSE `0.0307 m/s` and rear-axle lateral-speed RMSE `0.0157 m/s`; per-lap
RMSE ranges are `0.0205–0.0383 m/s` and `0.0089–0.0191 m/s`, respectively.
Its yaw-rate residual is numerically zero in this
bag because `/odom` reports the IMU gyro yaw rate; that is a direct measured
signal, not a learned yaw predictor. It is useful evidence for the present
state estimate in this practice regime, but not for the accuracy of a
future-yaw rollout.

AMCL remains active and materially improves the pose supplied to MPC. Over
the same bag, raw `/amcl_pose` has `0.182 m` position-error p95 against
bridge truth, while `/current_map_pose` (the controller's pose input) has
`0.159 m` p95. Integrated `/odom`, after one initial frame alignment held
fixed, has `0.267 m` p95 over the run; its per-lap p95 rises to about
`0.24–0.27 m` in the later completed laps. Thus AMCL correction is important
and the observed run supports the user's criterion: velocity/yaw-rate
estimation is useful even though dead-reckoned position is not precise enough
to replace AMCL. The eight-lap run is not collision-free and does not validate
the full 10-lap objective.

Independent review accepted moving to MPC work under this clarified criterion.
The acceptance is deliberately limited to useful Odom motion-state output on
this recorded practice line; it does not certify full-envelope Odom accuracy,
sensor-only long-horizon plant prediction, or a collision-free racing run.

This is a measurement of the production stack from the October 14 image, not a
live validation of the current unintegrated PhysX candidate. The manifest's
Odom node source hash matches the current node file, but its `odometry_observer`
hash does not match the current worktree, so any current-source claim still
needs an actual rebuilt-stack run. The source-derived plant's practice replay
is a separate result: on two captured closed-course windows it predicts the
recorded motion to sub-millimetre position error, but it starts from captured
body/wheel truth and uses the recorded command sequence. It has not been
installed as production Odom and this is not a fair sensor-only A/B against
the production run.

The exact-window, teacher-forced 1 ms diagnostic on reset trials 4 and 5
separates local equation error from recursive drift. It resets to the captured
full state every 1 ms and uses captured per-wheel actuation and wheel pose;
therefore it is a physics diagnostic only, not an Odom result. At or above
`5 m/s`, the norm of the one-step angular-velocity residual has p95
`0.00073/0.00080 rad/s` for trials 4/5; trial 4 has one sparse
`0.0528 rad/s` spike, while trial 5's maximum is `0.00233 rad/s`. Yet the
corresponding recursive source-plant rollouts still reach yaw-rate RMSE
`0.0178/0.0125 rad/s`, p95 `0.00636/0.00551 rad/s`, with maxima
`0.295/0.169 rad/s`. This points away from a broad per-step mismatch and
toward rare transients plus state/history accumulation; the precise cause is
not yet isolated. Exact-window summaries are in
[`trial_04`](../../live_runs/odom_physx_observer_stage1_20261016/r12g_reset_trials/trial_04/one_step_metrics_40hz.json)
and
[`trial_05`](../../live_runs/odom_physx_observer_stage1_20261016/r12g_reset_trials/trial_05/one_step_metrics_40hz.json).

The next blocker is now MPC rollout/actuation, not another broad Odom fit.
The deployed MPC has an 8-state plant (`e_y`, `e_psi`, `u`, `v`, `r`,
`target_speed`, steering command and delayed/physical steering states),
augmented to 12 solver states; its controls are steering-rate and
target-speed-rate. It publishes Ackermann speed targets with acceleration
zero, and a separate actuator interface converts the target speed into the
normalized throttle command. The model therefore does not optimize the actual
throttle channel or its wheel-spin response. On this same practice run, the
logged one-step MPC/truth error has p95 `0.151 m/s` speed and
`0.277 rad/s` yaw rate; the largest yaw-rate error is `1.592 rad/s`. This is
direct evidence to replace the speed-slew abstraction with a throttle-aware
rollout and relevant vehicle states, not to tune weights first.

Next work, using the existing bags before requesting any new run:

1. Treat current Odom speed and IMU yaw-rate as the accepted practical state
   baseline; preserve AMCL fusion for pose. Do not claim the separate PhysX
   plant is production-ready.
2. Build and score a causal MPC candidate whose longitudinal decision is
   normalized throttle (with its actuator state/history) rather than target
   speed rate, and whose rollout carries the vehicle states needed to explain
   speed, wheel spin, lateral velocity and yaw. Keep permitted runtime inputs
   separate from simulator-truth labels used only offline.
3. Compare current and candidate rollout errors on the same whole practice
   captures, by horizon and by run, before live promotion. Retain the current
   controller as a reference; do not accept a model solely on a one-step score.
4. Use the validated rollout in the source-based raceline optimizer only
   after its multi-step practice transfer is demonstrated; then test a separate
   faster candidate on the practice simulator with the established batchmode
   procedure and collision watcher.

No runtime Odom, MPC, AMCL, EKF, physics, or raceline was changed in this
assessment. No new simulator run was started.

The saved MPC-error trace was then paired with the controller/throttle trace
by nearest timestamp (within 40 ms) to localize the failure region. Of 2,090
prediction samples, 559 have measured steering-feedback magnitude in
`0.20–0.30 rad`; that band has yaw-rate absolute-error p95 `0.330 rad/s`
and includes the run's maximum. At `t=53.3266 s`, the vehicle is at
`u=4.321 m/s`, feedback steering `-0.256 rad`, and throttle command/feedback
`0.189/0.179`; MPC predicts `-2.869 rad/s` while truth is `-1.278 rad/s`.
This is a specific, measured yaw-rollout overprediction in the ordinary
practice steering range, not a generic high-speed failure. The active Y1
response surface begins blending at `0.29 rad`, so it does not cover this
sample; simply increasing its iterations cannot correct it.

The command/feedback distinction remains important: the MPC publishes a
desired speed, the actuator interface computes the normalized throttle
command, the simulator later reports throttle feedback, and the body/gyro
signals report the resulting vehicle response. A same-time command-feedback
difference is therefore a timing/actuator diagnostic, not a direct physical
slip label. The next candidate must optimize the actual throttle command and
propagate a causal throttle/wheel-spin state, while scoring against later
measured motion.

#### Reset IMU-age diagnosis and evaluation guard — 2026-10-16

The IMU-yaw outlier list now includes the source row and causal sample age.
All large yaw residuals in this capture are the same held ROS IMU orientation
(`-0.38293 rad`) while the source-truth heading is approximately zero; its age
increases from 43.7 ms to 893.6 ms across the dropout. This identifies stale
orientation reuse across the reset-period sensor gap, not a repeatable vehicle
yaw-response regime. It does not establish that the production node has the
same failure: its packet assembler and gyro propagation are different, and no
runtime code was changed.

The offline evaluator now records the input-state provenance and rejects an
input that does not explicitly declare the reset-state initialization contract.
The scored plant state is therefore clearly labeled as reset-initialized, not
initialized from live Odom sensors. A 50 ms freshness-gated blend was also
scored, selecting its blend gain on the first half only. It selected `0.55`
and produced `0.13715 m` full-run position RMSE / `0.24255 m` p95, effectively
the same as the ungated `0.13712 m` / `0.24253 m`; freshness gating alone did
not materially improve this run. The fused yaw is less accurate than plant yaw
on the second half (`0.0149 rad` vs `0.00266 rad` RMSE), so this remains a
position-only offline tradeoff, not a generally better yaw observer.

The next required Odom step remains sensor-state initialization (permitted IMU,
left/right encoders, and current commands only), followed by an independent
held-out bag comparison using common timestamps and the same reset/pose frame.
Until that exists, the low plant velocity errors and position blend are not
production Odom improvements. No simulator was launched in this continuation.

#### Source-built live Odom fusion comparison — 2026-10-16

This section supersedes the earlier statement that the PhysX observer had only
offline/replay evidence. The implementation was built from the current
workspace and run in the practice simulator, then compared against a second
live run using the same source build with the PhysX fusion explicitly disabled.
Both used the practice map, the same working raceline, MPC setup, and simulator
physics. Each run completed three laps without collision; the candidate's
recorded lap times were 7.5876 s (warmup), 5.6768 s, and 5.6448 s. The
source-only baseline is a separate run, not a deterministic same-seed replay.

The production `sensor_odometry_node` now supports a bounded fusion weight
`physx_velocity_blend`. With the PhysX predictor enabled, it advances 25 x 1 ms
steps between nominal 40 Hz observations and blends the resulting forward and
lateral velocities with the existing encoder/IMU observer at 0.10. The
predictor's lateral velocity is at the Rigidbody COM; before fusion it is
shifted to the rear-axle `base_link` reference using
`v_rear = v_com - r * 0.15532 m`. The encoder/IMU latent observer remains
independent and is not recursively overwritten by the PhysX prediction. The
runtime subscriptions remain the permitted encoders, IMU, and steering/throttle
commands; simulator truth/IPS is used only after recording for evaluation.

| Live workspace-source run | Forward-speed RMSE vs truth | Lateral-speed RMSE | Integrated position p95 |
| --- | ---: | ---: | ---: |
| Sensor-only (`physx_velocity_blend=0`) | 0.0997 m/s | 0.0171 m/s | 0.1482 m |
| PhysX fusion (`physx_velocity_blend=0.10`) | 0.0447 m/s | 0.0159 m/s | 0.1075 m |

On these runs, forward-speed RMSE was lower by 55%, lateral-speed RMSE by 7%,
and integrated-position p95 by 27%. Candidate integrated-position p50/p95/p99
was 0.0640/0.1075/0.1252 m, after one initial SE(2) alignment held fixed for
the run. Position error p95 was 0.1047 m and 0.0711 m on the first two lap
segments and 0.1055 m on the next; the final recorded segment was partial.
Yaw rate is passed through from the IMU in both configurations, so this change
does not claim a yaw-rate improvement. The position result is a drift/trajectory
comparison, not absolute global localization accuracy; AMCL remains active.

The comparison bags are:

- `live_runs/odom_physx_fused_online_practice_20261016/run/run_0.db3`
- `live_runs/odom_sensor_only_source_online_practice_20261016/run/run_0.db3`

The candidate bag also replays through the source-built sensor-only observer:
that counterfactual gave 0.0955 m/s forward-speed RMSE and 0.1274 m position
p95, versus the recorded fused run's 0.0447 m/s and 0.1075 m. This is useful
paired evidence on identical sensor/truth samples, but it is a replay, not a
second live A/B. The candidate's 25 ms displacement RMSE was 2.993 mm versus
3.060 mm in that replay, only a 2% improvement; the stronger demonstrated
benefit is velocity estimation, not a dramatic per-step position-increment
change.

Important limitation: an earlier run using an older installed binary (without
the new workspace parameters) scored better than the source-only comparison
here: 0.0382 m/s forward-speed RMSE and 0.0903 m position p95. That binary was
not a source-matched control and its exact implementation/configuration was
not established. Therefore this is evidence that the new workspace
implementation improves on the current workspace's sensor-only configuration
in the tested practice run; it is not evidence that it beats every historical
prebuilt runtime. Rebuild/deployment provenance must be checked before
replacing any competition image.

The two source-built live bags were received at approximately 40 Hz, with no
collision and no MPC failure. The candidate report is reproducible with
`python3 tools/analyze_practice_state_accuracy.py <bag/run_0.db3>`; IPS and
simulator bridge odometry are used only by this offline scorer. Runtime topic
policy passed before launch. No MPC, AMCL, EKF, physics, or raceline behavior
was changed. This is a real, bounded Odom implementation improvement, not a
claim of a perfect observer or proof of multi-map/general-regime performance;
another independent source-built practice run is still needed to establish
repeatability before calling the blend a generally reliable production default.

The persistent server was then exercised over the full existing `r12g` reset
segment rather than only the earlier 250 ms parity slice. The new
`tools/vehicle_dynamics_learning/stream_physx_recursive_rollout.py` harness
sent 42,750 1 ms command steps to one retained PhysX process and emitted 1,710
40 Hz states. All emitted x/z positions, quaternion components, and world
velocities matched the batch rollout exactly (maximum delta `0`); the 42.75 s
plant rollout took 4.29 s wall time (`9.96x` real time). Results are in
`live_runs/odom_physx_observer_stage1_20261016/r12g_ros_command_streaming_parity_42p75s.json`,
with the compact prediction stream in
`live_runs/odom_physx_observer_stage1_20261016/r12g_ros_command_streaming_40hz_predictions.csv`.
This removes the concern that only a short streaming slice was tested. It
still validates the research executable against its batch implementation,
not integration with the ROS Odom process or Unity's private PhysX plugin.

The existing production-observer bag replay was also run from before the
sensor gap so its encoder-epoch detector could observe both sides. In the 70 s
window it emitted two `reset_epoch` events; at the second event, the first
complete post-gap sample had zero mapped wheel speed and zero forward/lateral
observer speed. This event is at bag-relative `67.732 s`, about `0.906 s`
after the last pre-gap sensor messages. Under the current source/bag clock
pairing, the corresponding Unity reset capture has zero wheel RPM and
near-zero horizontal speed. The replay confirms the existing observer
recognizes the large encoder-count reset when messages resume. The 70 s
aggregate position score is invalid across these teleports because the
observer has no new absolute pose from encoder/IMU inputs alone; it is not
treated as a continuous-run accuracy score. Event details are in
`live_runs/odom_physx_observer_stage1_20261016/current_odom_r12g_reset_transition_70s/report.json`.

Around that reset, the ordinary throttle command changes from `0.27` to `0.0`
at bag-relative `66.824 s` while reported throttle feedback remains `0.27`;
IMU and encoder messages then stop for about `0.906 s`. Those are stale
pre-reset measurements, not proof that zero throttle physically maintained the
old wheel speed. This supports using command topics as plant inputs and
treating actuator feedback/sensors only as measurements from their actual
sample epoch. It does not yet establish a general runtime reset detector
without using the restricted reset/debug topics.

#### Low-speed PhysX fusion schedule: production implementation and repeat check — 2026-10-16

The production observer now uses a speed-scheduled PhysX velocity weight. The
existing sensor-only estimate was notably weak during launch, while the
reset-initialized plant is most useful near zero speed. The observer therefore
fades the PhysX weight linearly from `0.45` at sensor-estimated `|u|=0` to the
previously validated `0.10` by `1.5 m/s`, then holds `0.10`. This is a bounded
output/pose correction; the sensor observer's latent state remains independent.
It consumes only legal encoder/IMU inputs and steering/throttle commands. No
simulator truth, IPS, AMCL, or other debug topic is read at runtime.

The parameters are `physx_low_speed_velocity_blend` and
`physx_low_speed_blend_transition_mps` in
[`sensor_odometry.yaml`](../../f1tenth_localization/config/sensor_odometry.yaml).
The change was built into the development image and the node log confirmed
`25 x 1 ms` integration, `0.100` normal-speed weight, and `0.450` low-speed
weight fading by `1.50 m/s`. Runtime topic policy passed before launch.

The first source-built practice run produced 912 aligned samples over
`22.787 s` (about 40 Hz), three scored laps, and zero collisions. Versus the
source-only replay of that exact bag, the deployed schedule improved forward
speed RMSE from `0.04352` to `0.02901 m/s` (33%) and integrated position p95
from `0.09460` to `0.05187 m` (45%). A paired reconstruction against fixed
`0.10` on the same packet stream gave forward-speed RMSE `0.03241` and
integrated-position p95 `0.07412 m`; the scheduled values were `0.02901` and
`0.05020 m`. Lateral-speed RMSE remained approximately `0.0119 m/s`, so this
change is not a lateral-model gain. These comparisons hold the initial SE(2)
alignment fixed and use truth only offline.

A fresh source-built repeat used the same practice map and working trajectory,
completed two further laps with zero collisions, and recorded 413 Odom samples
over `10.29 s` at approximately 40 Hz. Its forward/lateral velocity RMSE was
`0.0309/0.0119 m/s`; yaw rate is the IMU signal and is not an independent
PhysX-model improvement. The run began after the simulator had already
reached lap 1, so it is a repeat check of the moving regime, not an independent
launch-weight validation. Its initial integrated-position p95 score of
`0.1612 m` was invalid: the evaluator paired headerless IPS to Odom by bag
receive time, and this bag contains a roughly `300 ms` delivery burst in which
about 13 source-stamped Odom samples arrived within `2 ms`. That compared
different simulator instants. The evaluator now compares integrated Odom
against bridge truth at matching source/header stamps, with the initial SE(2)
alignment held fixed. The corrected repeat position p95 is `0.1005 m`
(`0.0672/0.1005/0.1083/0.1294 m` p50/p95/p99/max), versus `0.0519 m` p95 in
the initial run. Per-lap repeat p95 is `0.0700 m` for lap 1, `0.0900 m` for
lap 2, and `0.1099 m` on partial lap 3. The run-to-run position difference
remains after correcting the timing artifact; aligned Odom yaw matches bridge
truth because yaw is IMU-fed, not because PhysX independently predicts yaw.
AMCL remains active and its corrections were not disabled.

The two practice bags also contain development-only `bridge_packet_timing`
records, used here only for offline diagnosis. Across both independent runs,
throttle feedback at request sequence `k` equals sent throttle command at
`k-2` exactly (`906/906` and `412/412` matched pairs). Steering feedback at
`k` equals command at `k-2` converted to physical radians (`command * 30 deg`)
within `0.0001 rad`; same-request steering mismatch is `0.0125 rad` RMSE in
the first capture. This is a two-request phase relation in the development
captures, not a runtime subscription or a change to competition topic policy.
Production PhysX had been advancing with the newest command captured for the
current sensor packet, although the simulator-reported actuator state was two
requests behind it. That is a source-supported phase mismatch, especially
during throttle and steering transitions.

Production PhysX integration now uses the already-subscribed fresh actuator
feedback when available: throttle feedback drives the Unity CAWD/CAWB torque
branch, and measured steering feedback is converted back to Unity's internal
center-angle sign and interpolated across the 25 internal 1 ms steps. This
uses the measured actuator trajectory between 40 Hz observations. Normalized
commands remain the startup fallback until both feedback channels are fresh.

#### Online validation of actuator-feedback-driven PhysX — 2026-10-16

Two independent source-built practice runs used the same map and reference
trajectory. The installed localization package had to be rebuilt inside each
ephemeral development container; a preflight caught and excluded a launch
that did not expose the new PhysX parameters. For both scored runs, the node
log confirmed the predictor enabled and the runtime parameter readback
confirmed `use_physx_odom_velocity=True`. The standard runtime topic-policy
check passed. The simulator was launched via the established Xvfb batch-mode
path; the recorder and collision/lap watcher were armed before connecting it.
Each run completed three laps with zero collisions.

The runs produced 825 and 823 matching `/odom` samples. Sensor source-time
interval medians were `25.063 ms` and `24.961 ms` (p95 `25.697 ms` and
`25.676 ms`); the actuator feedback streams also had 825/823 samples. The
first run's Odom receive latency relative to source time was 4.91 ms median,
6.01 ms p95, with no accumulating cadence loss. Thus the native 1 kHz PhysX
process completed its 25 internal steps without reducing the live sensor rate.

| Run | Forward velocity RMSE | Lateral velocity RMSE | Integrated Odom position p95 | Position max | Collisions |
|---|---:|---:|---:|---:|---:|
| Feedback PhysX r01 | `0.02814 m/s` | `0.00918 m/s` | `0.0753 m` | `0.0816 m` | 0 |
| Feedback PhysX r02 | `0.03358 m/s` | `0.00946 m/s` | `0.0854 m` | `0.0900 m` | 0 |
| Earlier low-speed-blend run | `0.02901 m/s` | `0.01185 m/s` | `0.0519 m` | — | 0 |
| Earlier low-speed-blend repeat | `0.03091 m/s` | `0.01191 m/s` | `0.1005 m` | `0.1294 m` | 0 |

Across the two new runs, lateral-speed RMSE is consistently about 20–23%
lower than in the earlier pair. Forward-speed RMSE is mixed (one better, one
worse), and integrated position p95 falls between the earlier run-to-run
results. Therefore this is a demonstrated lateral-velocity improvement and a
safe online integration, but **not yet a consistent overall Odom-accuracy
gain** and not near-perfect accuracy. Yaw rate remains the measured IMU value;
its zero error against simulator truth is not a PhysX prediction result.

The largest repeatable forward errors are now localized to launch/low speed
(`0–3 m/s`, low steering: about `0.051–0.052 m/s` RMSE in both new runs) and
moderate steering at `3–5 m/s` (about `0.041–0.050 m/s` for `0.1–0.2 rad`).
At `3–5 m/s` and `0.2–0.3 rad`, lateral-speed RMSE remains `0.015–0.016 m/s`.
These are the next model-diagnosis targets; changing the fusion weights without
separating the predictor, encoder, and actuator-input residuals would be
premature.

Artifacts:

- [`r01 bag`](../../live_runs/odom_physx_actuator_feedback_practice_20261016_r01/run/run_0.db3)
- [`r01 state score`](../../live_runs/odom_physx_actuator_feedback_practice_20261016_r01/analysis/)
- [`r02 bag`](../../live_runs/odom_physx_actuator_feedback_practice_20261016_r02/run/run_0.db3)
- [`combined regime comparison`](../../live_runs/odom_physx_actuator_feedback_practice_20261016_r01/analysis/combined_regimes.json)

Next: use these closed bags to separate launch and moderate-turn velocity
residuals into sensor-only observer, PhysX state, and actuator-phase
contributions. Keep feedback-driven PhysX enabled as the source-consistent
production path while retaining the old values as comparators; do not claim
forward-speed or full-pose improvement until a paired comparison shows it.

The largest remaining forward-speed residuals in the first capture are short
encoder-burst rejection episodes: 106/912 samples, with `0.0611 m/s` RMSE
versus `0.0215 m/s` on non-rejected samples. The two largest residuals are
about `-0.218` and `-0.212 m/s` at `4.6–4.9 m/s`; they clear within the next
two packets. A separate hard-braking burst has about `0.15 m/s` error for
roughly six samples. The burst gate is driven by an implausibly abrupt rolling
wheel-rate change and packet/window inconsistency; accepting such a reading
without testing risks turning a delayed cumulative-angle update into false
vehicle acceleration. Existing paired analysis also shows the native PhysX
predictor is substantially less accurate in these bursts than the observer;
the burst-only optimal fusion weights are uncertain (`about 0.055–0.134`
across captures). I did not raise the PhysX weight or relax the encoder
rejection rule on this evidence.

The repeated-run artifact is
[`run_0.db3`](../../live_runs/odom_physx_low_speed_blend_practice_20261016_repeat/run/run_0.db3);
the initial scheduled run is
[`run_0.db3`](../../live_runs/odom_physx_low_speed_blend_practice_20261016/run/run_0.db3).
The repeat was rescored with `python3 tools/analyze_practice_state_accuracy.py`
after fixing its timestamp pairing. The sim, stack, recorder, and watcher were
stopped by the established lap watcher. The next validation is a source-built
practice run of the actuator-feedback-driven PhysX predictor, with immediate
collision abort and the same source-time state scoring.

#### Full forward-PhysX production fusion and first live prefix — 2026-10-16

The production observer now takes forward body speed directly from the 1 kHz
PhysX shadow (`physx_forward_velocity_blend: 1.0`) while keeping the existing
sensor-based lateral estimate (`physx_velocity_blend: 0.10`). This replaces the
previous small forward blend and reactive encoder-burst-only correction. The
native plant returns wheel angular speed and WheelHit longitudinal slip for
diagnosis; its predicted chassis forward speed is what Odom integrates. No
Unity physics, map, reference trajectory, or runtime topic policy changed.

The basis is the known VehicleController branch, not a fitted throttle gain:
`DriveTorque = ThrottleLimit * AutonomousThrottle * MotorTorque`; at exactly
zero drive torque it sets motor torque to zero and applies `MotorTorque` as
brake torque to the configured wheels. Otherwise the torque is apportioned to
the active drive wheels. PhysX then determines tire force from wheel/contact
slip and load; commanded torque is not itself chassis acceleration. The native
shadow advances this branch and its wheel/contact dynamics in 25 consecutive
1 ms steps per Odom packet. The active sim-racing scene overrides the public
F1TENTH prefab default and uses `MotorTorque=428 N·m`, CAWD/CAWB: at 50%
throttle this is `53.5 N·m` drive torque per wheel, at 30% it is `32.1 N·m`
per wheel, and at exactly zero it is `428 N·m` brake torque per wheel. Thus a
50%→30% transition reduces drive torque but does not apply brakes; only the
exact-zero branch switches to the brake torque.

Two existing independent practice captures confirm a consistent two-sample
command-to-feedback phase: reported throttle at sample `k` matches the
recorded command at `k-2` with RMSE `0.00029` and `0.00027`, while same-index
command/feedback RMSE is `0.03622` and `0.03597`. This is a 50 ms relationship
on the 25 ms recorded grid. It is not evidence of a 50 ms tire-force or
WheelCollider response constant: `VehicleController.CurrentThrottle` sets the
autonomous input, and `FixedUpdate` consumes it in the torque branch. The
production PhysX shadow uses fresh reported actuator feedback, which reflects
the input actually active in Unity, rather than pretending the latest local
command has already reached the vehicle. This avoids advancing the model with
a command that is ahead of the observed actuator state.

An online run used the rebuilt production localization package and the
unchanged practice controller/trajectory. Topic-policy preflight passed; the
established batch simulator, recorder, and immediate collision watcher were
used. A collision occurred about `1.099 s` after the first moving truth sample,
so the watcher stopped the run and there are zero scored laps. The bag has no
`/mpc/diagnostics` topic, so it cannot establish the controller's internal
reasoning. On the 44 pre-impact matched samples, the newly published Odom had
forward-speed RMSE `0.0033 m/s` and position RMSE `0.0041 m` after one initial
pose alignment. This is direct online-prefix evidence of accurate forward
tracking, but not full-run validation.

The bag also records a large accepted AMCL scan correction immediately before
the collision: after 22 rejected scans and no previous accepted correction,
the first accepted correction at `1.053 s` applied `0.384 m` of translation
and `0.052 rad` of yaw correction; the map pose consequently jumped about
`0.39 m` while simulator truth and Odom remained continuous. The collision
followed at `1.099 s`. This chronology makes localization/controller
interaction a concrete confound, not a proven collision cause. The available
prefix is too short to classify the collision or establish that the Odom
change caused or prevented it. The bag and extracted collision-prefix table
are preserved under
[`odom_physx_forward100_online_practice_20261016_r01`](../../live_runs/odom_physx_forward100_online_practice_20261016_r01/).

The command/feedback timing supports using the known torque input to predict
wheel spin versus body motion before encoder deltas arrive. In production this
is realized by the feedback-driven 1 ms PhysX state propagation, now fused
fully for forward speed. On the two independent captures, the predicted rear
wheel/body speed gap also correlates with encoder-versus-truth error two
packets later (`r=0.952` and `0.950` overall; `0.950` and `0.928` on rejected
wheel bursts). During throttle drops greater than 5%, predicted rear slip is
near `0.98`; the median measured encoder/truth gap grows from `1.16/1.21 m/s`
at the event to about `3.25 m/s` two packets later. This is direct evidence
that the model identifies when wheel speed is no longer a valid proxy for
body speed, before the delayed encoder discrepancy reaches its maximum.

The same two captures give PhysX forward-speed RMSE `0.0044/0.0050 m/s`,
versus sensor-only replay `0.0452/0.0482 m/s`; using PhysX forward speed with
sensor lateral speed and yaw gives integrated-position RMSE `0.0140/0.0144 m`
versus `0.0720/0.0718 m` for sensor-only replay. These are offline replay
comparisons on independent captures, not a full-lap production claim.

The remaining distinction is temporal: current Odom must estimate the vehicle
state that has actually occurred, so it should use the actuator input actually
active during that interval; future-command preview is a separate forward
prediction and must honor the measured two-packet command-to-actuator phase.
Do not substitute the newest command for fresh feedback or call the 50 ms
phase a tire dynamic.

Current conclusion: the earlier modest PhysX fusion did leave a large
available Odom gain unused. The full forward-PhysX implementation is now in
production code and matches truth closely over the live prefix as well as the
two independent offline captures (`u` RMSE `0.0044` and `0.0050 m/s`). It is
not yet proven over a complete collision-free live run. Before another lap
test, resolve the evidence gap around the accepted large first AMCL correction
and capture MPC diagnostics; do not weaken the collision-abort rule. Keep
lateral fusion and yaw unchanged while this forward-speed candidate is
evaluated.
