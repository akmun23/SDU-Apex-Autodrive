# Practice-only odometry and rollout evidence — 2026-10-19

This is the current authoritative summary for the retained practice-track
captures. Non-practice recordings and their development-player artifacts were
removed from the workspace and are not accepted as evidence for this vehicle.

## Runtime contract

`sensor_odometry_node` consumes only the two encoder streams, IMU, steering and
throttle feedback, and steering and throttle commands. Ground truth is used by
offline evaluation only. `tools/verify_runtime_topic_policy.py` passes. The
node publishes `/odom`, `/odom/unity_plant_state`, and TF; diagnostics are now
disabled by default and can be explicitly enabled for profiling.

## Retained practice evidence

The two high-speed three-lap source captures are replayed through the rebuilt
production node, without MPC or a simulator. The shared source plant is scored
separately by propagating logged commands without future sensor correction.

| Capture | Odom `u` RMSE / max | Odom `v` RMSE / max | Full-run relative XY end / max | 750 ms path gate |
|---|---:|---:|---:|---:|
| `current_odom_practice_replay_20261019/run/bag` (811 epochs, max truth speed 8.093 m/s) | 0.0044 / 0.0198 m/s | 0.0004 / 0.0023 m/s | 0.011 / 0.042 m | 781/781 pass; worst 4.80 cm |
| `current_odom_practice_replay_20261019/run_1/bag` (805 epochs, max truth speed 8.078 m/s) | 0.0058 / 0.0314 m/s | 0.0007 / 0.0114 m/s | 0.019 / 0.052 m | 774/775 pass; worst 5.13 cm |

The 5 cm limit is therefore **not a complete pass**: one overlapping 750 ms
path exceeds it by 1.3 mm, and run 1's full-run pose maximum is 5.1 cm. The
second run's 750 ms endpoint gate itself passes 775/775. Published yaw rate and
heading are direct permitted gyro/IMU inputs; zero error in those output
channels is routing agreement, not a yaw-model accuracy result.

The largest 25 ms predicted yaw-rate residual in run 1 is 0.0903 rad/s at
sample 715 (6.31 m/s, zero throttle, approximately 0.055 rad measured steering).
The adjacent sample returns to ordinary error. A second isolated point is
about 0.0893 rad/s. Encoder angle is flat and longitudinal acceleration
prediction is close at the first point; steering feedback changes during the
interval.

### Source-order steering-feedback correction — 2026-10-19

`VehicleController.CurrentSteeringAngle` returns `currentS`. In
`FixedUpdate()`, Unity assigns `currentS` from the rack angle before calling
`Steer()`, which then rate-limits and updates the rack for that physics tick.
The production odometry plant had been treating the reported pre-`Steer()`
feedback as the post-update rack state. The runtime now advances only its
internal rack state by one `unity_steering_step` at the packet boundary, using
the oldest queued steering command under its existing two-packet delay model.
It does not advance body dynamics an extra time, alter the published raw
steering feedback, or change topics, command behavior, or MPC inputs.

The command and feedback topics are headerless `Float32` messages. They carry
no Unity fixed-step ID or applied-command sequence, so associating the oldest
queued command with the feedback sample remains an assumption of the existing
delay model; exact per-tick command timing cannot be recovered from these bags.

On practice replay, the correction does **not** explain or remove the large
yaw residual. At sample 715 the raw plant yaw forecast changes from
`-1.1429153` to `-1.1429018 rad/s` (only `1.35e-5 rad/s`); the residual remains
`0.0903 rad/s`. Full-run motion metrics remain within replay variation: run 1
forward-speed RMSE is `0.0058 m/s`, and its 750 ms path gate is still 774/775
with a 5.13 cm worst path. Run remains 781/781 with a 4.80 cm worst path.
Thus the source-order discrepancy is corrected internally without a material
practice replay regression, but it is **not** a yaw-tail fix and does not
establish exact actuator-phase alignment. No tire/yaw parameters were fitted
or changed. The focused localization suite passes 5/5; no simulator was
launched.

### Single 5 cm path-gate miss decomposition — 2026-10-19

The only run-1 path-gate miss is one window starting near 7.44 m/s and nearly
zero steering. Its maximum error is 5.13 cm at 725 ms; at 750 ms it has fallen
to 4.73 cm, so every 750 ms endpoint passes. At the 725 ms peak, yaw error is
0.0016 rad, forward-speed error is -0.0061 m/s, and lateral-speed error is
about +0.0002 m/s. This is not explained by the earlier 0.0903 rad/s isolated
yaw-rate tail.

The evaluator now reports per-25 ms displacement error split into plant vs.
integrated Unity truth twist and truth twist vs. truth pose. In this failing
window's first step, the plant differs from integrated truth twist by only
about 0.3 mm, while integrated truth twist differs from the recorded truth
pose by 3.76 cm, almost entirely longitudinal. At the 725 ms peak step, the
per-step plant-vs-twist difference is still about 0.3 mm, while truth
twist-vs-pose differs by about 4.7 mm longitudinal and 1.5 mm lateral. Thus
the measured path miss is dominated by inconsistency between recorded pose
increments and same-packet twist integration, not evidence that changing the
plant equations will remove it.

The replay bag has no `/autodrive/roboracer_1/bridge_packet_timing` topic or
Unity physics-step/sample timestamp. The bridge stamps simulator odometry at
the host request boundary, so the bag cannot tell whether this pose/twist
disagreement is caused by sampling phase, a delayed position source, or a
different pose reference point. Do not tune physics, shift pose by a fitted
offset, relax the 5 cm gate, or call the window a pass. Keep it classified as
a gate failure pending same-physics-epoch pose/twist alignment. A future
practice capture should include the bridge timing diagnostics and a
development-only Unity fixed-step timestamp/sequence attached to the sampled
pose and twist; use these only for offline scoring, never as production
odometry inputs. If aligned truth still shows >5 cm plant error, then diagnose
and change the physical model.

### Follow-up disposition — 2026-10-19

The `5.13 cm` value is a single maximum over 775 overlapping rollout windows,
only `1.3 mm` above the nominal `5.00 cm` criterion. The final 750 ms endpoint
is `4.73 cm`, and the other 774 windows remain below threshold. This is
practically negligible for this replay, so it does not justify a speculative
change to production odometry or the plant. The strict metric is still recorded
as 774/775—not rounded into a pass—and this does not prove a universal 5 cm
bound.

The exact cause is not recoverable from this run's bag: it has neither bridge
packet timing nor a Unity physics-step/sample ID. A phase sweep on the two
replay captures and an independent timed practice capture finds no useful
constant phase correction (best phase 0 ms). Pose/twist closure anomalies recur
in other practice captures, and in the timed capture their magnitude is
associated with request/receive jitter (correlation `+0.660`; lowest- versus
highest-jitter quartile closure RMSE `0.301 cm` versus `0.914 cm`). This is
support for a timing/reference-coherence contribution, not causal proof and
not evidence that timing explains the exact 5.13 cm path maximum. Therefore the
single exceedance is accepted as negligible for practical assessment, while
the data-quality limitation and unrounded strict-gate result remain visible.

The offline diagnostic is available via
`tools/evaluate_mpc_bag.py --truth-pose-twist-phase-only`; it does not shift
samples or change any model input. No simulator was launched and no vehicle
parameters were changed for this follow-up.

The two retained 1 ms practice contact captures are:

- `live_runs/practice_yaw_contact_20261019/unity_wheel_dynamics_20261019_115929_029.csv`
- `live_runs/practice_yaw_contact_flush_20261019/unity_wheel_dynamics_20261019_120839_097.csv`

They cover low-speed/near-straight contact behavior, not the high-yaw racing
transient. Do not generalize them to that regime.

## Runtime performance after hot-path work

The same practice bags were replayed through the rebuilt node with diagnostics
temporarily enabled. These timings are local wall-clock measurements; the
diagnostic publisher itself is disabled in normal configuration.

| Run | Plant advance median / p95 / max, before → after | Whole packet median / p95 / max, before → after |
|---|---:|---:|
| `run` | 699 / 1272 / 2017 → 41 / 71 / 84 µs | 781 / 1392 / 2149 → 128 / 179 / 326 µs |
| `run_1` | 944 / 1761 / 3030 → 42 / 67 / 99 µs | 1030 / 1910 / 3214 → 130 / 177 / 249 µs |

The comparison is between recorded pre-change diagnostic timings and a fresh
same-input replay of the rebuilt binary. It is strong evidence of reduced
runtime cost, but not a hardware-independent benchmark. Even the diagnostic-
enabled worst measured callback was 0.33 ms, far below the 25 ms packet
period; disabling diagnostics removes its instrumentation/publish work.

## Changes in this cleanup

- Removed non-practice run folders, generated players, capture hooks, scene,
  logs, custom build scripts, and the capture evaluator that explicitly
  consumed static open-plane WheelCollider captures. The live `live_runs/`
  evidence set contains practice captures only.
- Kept production inputs and outputs unchanged. The plant hot path now caches
  wheel-ray start vectors and shared attitude terms, avoids repeated per-wheel
  transforms, and directly computes the Ackermann steering basis. ROS output
  messages and frame strings are reused rather than reconstructed each packet.
- Diagnostics publisher and its clocks/data-copy work are opt-in; the shipped
  configuration sets `publish_diagnostics: false`.
- No MPC solver, weights, raceline, physics constants, or controller behavior
  were changed in this task.
- Separately, the odometry runtime now reconstructs the post-`Steer()` rack
  angle from Unity's pre-`Steer()` `CurrentSteeringAngle` feedback, advancing
  only the internal steering state by one source-rate-limited tick. Practice
  replay confirms no material aggregate regression but shows the `0.0903
  rad/s` yaw tail is unaffected; detailed caveats are recorded above.

## Verification and remaining gate

- `f1tenth_localization` and `f1tenth_mpc` build successfully.
- Runtime topic-policy check passes.
- Focused plant and localization tests pass (plant test; observer, packet
  assembly, wheel encoder, and source-derived runtime tests).
- Two rebuilt-node practice replays preserve odometry and rollout accuracy;
  their 5 cm results are shown above.
- The broader MPC rollout test `MpcRtiStageParallelism.CondensesQueuedCommandsWithoutChangingPlantJacobian`
  still fails its yaw reconstruction tolerance (`9.77e-5` vs `5.0e-5 rad/s`)
  and has not been changed here. MPC work should wait until this is resolved.
- No simulator was launched. This task used retained practice bags only.

Relevant implementation: `f1tenth_localization/src/sensor_odometry_node.cpp`,
`f1tenth_localization/src/unity_source_odom_runtime.cpp`, and
`f1tenth_mpc/src/unity_source_plant.cpp`. Offline scoring is performed by
`tools/evaluate_mpc_bag.py`; it is not a runtime input