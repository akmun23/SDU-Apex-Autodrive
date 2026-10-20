# v251 Practice CSV Follow — 2026-10-20

## Purpose and method

Collected repeatable, full-motion practice-track captures while following
`SDU_Apex_IPOPT_PhysX_MinTime_v2/v252_reference/v251_best_commands.csv`.
The development-only path follower uses simulator ground-truth pose/speed for
pure-pursuit path tracking and the CSV throttle profile. It continues along
the repeated CSV path; it is not a fixed 30-command replay. MPC was disabled,
so these runs measure the CSV/GT follower and plant response, not MPC tracking.
Ground truth is only used by this development experiment, never by the
competition controller.

The practice simulator was run in the established `batchmode` procedure, with
the collision/lap watcher armed. Both five-lap runs ended at lap count 5 with
collision count 0. The watcher stopped the run at the five-lap target.

## Preserved full-run bags

| Run | Bag | Size | Laps | Collisions | Simulator-reported lap times (s) | Mean / best |
|---|---|---:|---:|---:|---|---:|
| 1 | `live_runs/csv_v251_gt_5lap_run1_20261020/run` | 26 MiB | 5 | 0 | 9.872, 9.653, 9.718, 9.649, 9.720 | 9.723 / 9.649 s |
| 2 | `live_runs/csv_v251_gt_5lap_run2_20261020/run` | 27 MiB | 5 | 0 | 9.845, 9.659, 9.717, 9.654, 9.719 | 9.719 / 9.654 s |

Across the ten reported laps, mean time was 9.721 s; best was 9.649 s. These
are data-collection laps, not a sub-5-second result. Five consecutive lap
count increments were observed in each run.

## Signal quality and tracking

Both bags contain simulator truth, production odometry and plant state,
commands, actuator feedback, encoders, IMU, lap/collision counters, bridge
timing diagnostics, and `/csv_replay/diagnostics`.

- GT/odometry cadence: mean 25.04 ms, p95 25.70 ms.
- CSV follower diagnostics: 24.99–25.00 ms mean interval, p95 about 25.2 ms.
- Drive-command cadence: 25.51 ms in run 1; 25.22 ms in run 2.
- GT speed: median 3.045/3.051 m/s, p95 3.344 m/s, maximum 3.362/3.360 m/s.
- CSV path cross-track distance: RMSE 2.1 cm, p95 4.2–4.3 cm, max 5.6 cm.
- Path heading error: p95 0.052–0.055 rad, max 0.154–0.159 rad.
- Speed-profile error: RMSE 0.399–0.402 m/s; the follower does not precisely
  realize the CSV target speed profile. Its observed throttle was 0.095–0.136.
- Production odometry versus GT: forward-speed RMSE 0.0067–0.0069 m/s,
  lateral-speed RMSE 0.0038 m/s. Relative XY pose drift over each ~48 s capture
  ended at 0.170–0.183 m. Reported yaw-rate error is zero because this odometry
  output passes the gyro yaw rate through; that is not an independent yaw-model
  accuracy result.

The controller's `complete=False` diagnostic means it did not reach the end of
the entire three-copy CSV extension before the five-lap watcher stopped it. It
followed 139.4–139.5 m, or 94.3–94.4% of the remaining extended path, and did
complete all five requested laps.

## Open-loop source-plant replay

The production C++ source-derived plant was replayed offline from each logged
plant state using recorded commands, without future-state corrections. This is
a test of the reduced source plant, not execution of Unity/PhysX itself.

| Horizon | Run 1 XY RMSE / p95 | Run 2 XY RMSE / p95 |
|---|---:|---:|
| 25 ms | 1.06 / 0.55 cm | 0.66 / 0.36 cm |
| 100 ms | 1.41 / 0.61 cm | 0.56 / 0.60 cm |
| 250 ms | 1.51 / 0.83 cm | 0.84 / 0.87 cm |
| 500 ms | 1.82 / 2.24 cm | 1.33 / 2.29 cm |
| 750 ms | 2.53 / 4.70 cm | 2.20 / 4.68 cm |

At 750 ms, 1765/1829 (96.5%) and 1809/1876 (96.4%) endpoints were within
5 cm. The maximum error anywhere along the 750 ms path was 34.4 cm in run 1
and 10.1 cm in run 2; therefore the strict 5 cm full-path condition is not
met. These metrics are specific to the speed/steering regime in these runs and
do not establish full-envelope or MPC rollout accuracy.

## Data cleanup

Before cleanup, `live_runs` contained 451 older ROS bag payload/metadata files
(`metadata.yaml`, `.db3`, `.mcap`) in 225 bag directories, totaling
2,685,538,225 bytes (2.50 GiB). Those old bag files and their empty
directories were removed. Verified afterward: the only ROS bag files left in
the repository are the two five-lap bags listed above. The earlier short pilot
and invalid fixed-start replay were not retained. Non-bag analysis files were
left untouched; `live_runs` is now 183 MiB including the two new bags.

## Reproduction and next use

Use the `csv-follow-5` watcher and the practice `batchmode` simulator procedure
with the v251 CSV profile to repeat the capture. Analyze either bag with
`tools/evaluate_mpc_bag.py --csv-follow-reference <v251 CSV> --source-plant-rollout`.
Do not interpret these GT-follower laps as proof that the current MPC converges
or can drive this path. The next useful comparison is to replay the same
commands/plant states and inspect the reported 750 ms tail windows, then later
test MPC separately once requested.

## Correct-format v300 MPC candidate — 2026-10-20

The user supplied the correctly formatted candidate
`v300_optimized_MPC_AMCL_raceline.csv`. It is preserved in the established
track trajectory directory as
`f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_v300_optimized_mpc_amcl.csv`.
The workspace copy is byte-identical to the supplied file (SHA-256
`2874816456f7209b52ea359458c8fdaf6fb51bd4fa10781671d3e3483b06bb3d`). It has
1,024 rows and the expected columns for station, XY, heading, curvature,
velocity, acceleration, and left/right corridor bounds. Runtime logs confirmed
that both AMCL and MPC loaded those 1,024 points (loaded path length about
28.29 m).

The earlier v300 attempt used a geometry-derived trajectory that copied the
old speed profile and reconstructed corridor widths. Although its XY geometry
matches the supplied file, it was not a test of the supplied speed,
acceleration, or corridor values and must not be cited as exact-file
validation. The first-collision diagnostic bag is retained at
`live_runs/mpc_v300_r2reject_practice_1lap_20261020/run`.

The exact supplied file was then run with the established practice simulator
batch-mode procedure and current workspace source. Runtime topic policy passed;
the simulator loaded the practice map and exact v300 trajectory. The one-lap
watcher observed collision count rise from 0 to 1 at about 5.11 s and stopped
the run immediately. Lap count remained zero, so this candidate has not yet
demonstrated a completed lap. Its bag is retained at
`live_runs/mpc_v300_exact_r2reject_practice_1lap_20261020/run`.

The exact-run evidence identifies a controller/runtime failure as well as a
path-tracking failure; it does not establish that the trajectory is
physically infeasible:

- The first logged non-optimal cycle was at about 1.50 s: both R1 and R2 were
  rejected on residual checks, with three QP attempts each. The cycle took
  about 100 ms and the command lead reached about 90 ms.
- At about 3.87 s, cross-track error was already about 0.51 m. Both R1 and R2
  failed exact nonlinear corridor validation. That cycle took about 201 ms;
  the candidate-command lead was about 191 ms. Collision followed at about
  5.11 s.
- Across the 104 recorded MPC diagnostic cycles, median/p95/max solver-cycle
  time was about 12.7/199/234 ms; 30 cycles exceeded 25 ms. The diagnostic
  publication interval itself averaged 39.0 ms with a 120 ms p95. This is not
  a 40 Hz-ready closed-loop result.
- AMCL scan-pose XY error against truth was 10.5 cm RMSE (17.7 cm p95); the
  controller map-pose channel was 10.9 cm RMSE. This is significant relative
  to narrow corridor clearances and must be kept separate from optimizer
  failure.

An offline, fixed-input solver replay used the exact CSV and the 163 exported
pre-collision samples with GT pose plus logged production plant state. This is
an oracle diagnostic—not a closed-loop driving test. R1-only produced 98
optimal, 6 degraded, and 58 nonlinear-rejected samples (one input rejection).
`r2_on_rejection` produced 109 optimal, 20 degraded, and 33 nonlinear-rejected
samples (one input rejection). Thus R2 improved offline candidate acceptance
counts, but cycles exceeding 25 ms rose from 30 to 46; p95 time was about
39.7 ms for R1-only and 38.1 ms for R2-on-rejection, with maxima of 45.5 and
68.3 ms. This does not justify putting R2 on the production path: it still
fails many samples and adds computation. The live collision run also does not
show that R2 made the car track successfully.

Do not modify the supplied trajectory's geometry, speed, or corridor values
on the basis of this single aborted run. Next isolate the measured causes in
order: bound/measure solver work per 25 ms update, determine how rejected
cycles map to published commands, then separate AMCL pose error from MPC
cross-track growth using the retained exact-run bag. A further simulator run
should wait until that offline diagnosis yields a specific correction; the
collision watcher remains mandatory.

## v300 native-control path-follower captures — 2026-10-20

The user supplied
`v300_ros_path_with_observed_native_controls_11col.csv`. The exact file is
preserved as
`f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_v300_native_controls_11col.csv`
(SHA-256 `3a45ca84bd4134a5b5e9b4ed062d9859701dcc276eb4b586b18a112cb328eea1`).
It contains 1,024 rows; target speed is 2.72–8.07 m/s (median 5.16), throttle
is 0–0.446 (median 0.224), and trapezoidal `ds/v` gives a nominal 5.92 s
around the 28.26 m path. The nine geometry/speed/bounds columns match the
correct-format v300 trajectory above to less than `1e-8` per value.

Extended the development CSV follower to read the comment-prefixed 11-column
schema directly and interpolate supplied native controls by GT-projected path
progress. It does not invent lateral velocity or yaw-rate labels.
`SDU_APEX_CSV_PATH_REPEATS` selects path copies; the tests used eight copies
so a five-lap watcher would not exhaust the CSV first. MPC was disabled, GT
was used only by this development follower, and tests used the established
practice batch-mode simulator, full bag recording, and immediate collision
watcher.

Two supervised attempts were recorded; neither completed a lap:

| Bag | Steering policy | Span / max GT speed | Outcome |
|---|---|---:|---|
| `live_runs/v300_native_controls_pp_5lap_20261020/run` | GT pure pursuit; supplied native throttle | 6.63 s / 7.24 m/s | Collision on the first lap, at ~6.35 s; path progress advanced 25.37 m from the initial projection, short of one complete lap. |
| `live_runs/v300_native_controls_replay_5lap_20261020/run` | Supplied native-applied steering and throttle replayed by GT path progress | 1.50 s / 5.34 m/s | Collision early in the first lap. |

In the pure-pursuit attempt, path-distance error first exceeded 0.25 m at
5.56 s and 0.50 m at 5.76 s, on the tight turn near CSV station 13.76 m. There,
pure pursuit requested about `-0.454 rad` while the CSV's observed native
steering reference was about `-0.229 rad`; pure-pursuit steering then saturated
at `-0.524 rad`. This is a measured controller/reference discrepancy, not
proof that either value alone caused the collision.

The second attempt showed that directly replaying the `observed_native_applied`
steering column from startup is not a safe substitute: error exceeded 0.50 m
about 1.25 s after motion began. The column should be indexed spatially after
the path follower has aligned the car, rather than treated as an open-loop
startup command sequence. The direct-replay bag is retained as a diagnostic
and will not be repeated as-is.

These captures are substantially faster than the v251 data collection runs
(previous max speed about 3.36 m/s), and are useful high-speed partial-run
data, but they are not successful lap bags and provide no lap-time result.
Do not claim that the faster raceline has been validated. The subsequent
alignment and lookahead experiments are recorded below; keep the collision
watcher and full recording enabled for any further test.

### Lookahead follow-up — 2026-10-20

The first valid bag showed PP steering saturating at `-0.524 rad` where the
CSV's observed steering reference was about `-0.24 rad`. Recomputing PP from
the captured poses (diagnostic only; not a closed-loop simulation) showed
that `lookahead_speed_gain_s=0.60` reduced the command near that turn to
approximately `-0.27 rad`, while `0.30` gave approximately `-0.35 rad`.
Added `SDU_APEX_CSV_LOOKAHEAD_SPEED_GAIN_S` as a development-only setting; its
default remains `0.12 s` and normal non-CSV launches are unchanged.

- Gain `0.60`: valid practice capture at
  `live_runs/v300_native_controls_pp_l060_5lap_20261020/run`. It reached
  `7.24 m/s`, then collided at about `2.06 s` after the first truth sample,
  near CSV progress `26.8 m`; cross-track distance was about `0.35 m` and the
  command again reached roughly `-0.47 rad` at the final recorded sample.
  It did not complete a lap. This is evidence that the longer preview delayed
  turn-in too much on the first closing turn, despite moderating the offline
  PP command at earlier samples.
- Gain `0.30`: not valid for controller scoring. The first bagged truth was
  already at `(-1.804, -3.363) m`, moving at `0.84 m/s`, rather than at the
  expected start near `(0.8, 3.16) m`; the first recorded collision count was
  already `1`, and the 125 ms bag contains no CSV follower diagnostics.
  Retain it as a simulator/startup diagnostic, not as evidence against this
  gain.

No lookahead value tested here produced a complete lap: the original `0.12 s`
run got farther along the path than `0.60 s`; the `0.30 s` attempt had an
invalid initial state. No lap time can be reported. The gain parameter is
exposed for controlled development tests, but no new default is promoted.
The built-in development reset was later enabled and held through simulator
startup; the valid follow-up captures began at the expected `(0.8, 3.16) m`
spawn with zero collision count. The earlier invalid-start bag remains
excluded from controller comparisons.

### Spatial native-control handoff — 2026-10-20

The user clarified that pure pursuit should first bring the vehicle onto the
path, then the interpolated native control at the closest GT-projected path
station should be used. Updated the development follower accordingly: native
steering becomes active at cross-track `<=0.05 m` and heading error
`<=0.05 rad`; it returns to pure pursuit if either exceeds `0.10` (hysteresis).
CSV throttle remains interpolated at the current projected station. The
current GT station selects the control; the lookahead point is used only by
pure pursuit. Added diagnostics for whether native-control mode is active.

The watched test began at the expected spawn after the built-in reset, reached
`7.26 m/s`, and briefly entered native-control mode near progress `21 m` once
the path error fell below the acquire thresholds. It left that mode shortly
after as cross-track/heading error exceeded the release thresholds, returned
to pure pursuit, saturated at the steering limit, and collided at about
`3.42 s` near progress `29.9 m` with cross-track around `0.84 m`. It did not
complete a lap. Bag:
`live_runs/v300_native_controls_handoff_5lap_20261020/run`.

A separate valid test of continuous native steering feed-forward plus the PP
deviation correction (not the later acquire/release handoff) reached
`7.26 m/s`, then collision count increased at about `3.76 s`; the nearest
tracking diagnostic was around progress `32.3 m`. No lap completed. Bag:
`live_runs/v300_native_controls_ff_pp_5lap_20261020/run`. Neither policy has
completed this path or earned a lap-time result. No policy is promoted to MPC
or production.

### Speed-governed v300 practice capture and offline MPC/rollout analysis — 2026-10-20

The follower's reference lookup is spatial, not a nearest-row lookup: each GT
pose is projected onto nearby path segments within a forward-only progress
window, then the path and control columns are linearly interpolated at that
continuous station. Pure pursuit supplies alignment/path correction; native
steering is blended according to measured cross-track and heading error. The
native profile lookup is shifted by the configured `50 ms` to compensate the
two-packet command delay. The successful bag used this blended policy, not a
latched hard switch: full native steering weight occurred on `98/514` ticks
(`19.1%`), and the median blend weight was `0.666`. The earlier explicit
acquire/release handoff is separately documented above and collided; it must
not be confused with this successful blended capture.

The practice batch-mode capture at
`live_runs/v300_native_controls_speed_governor_20261020/run` reached one
simulator lap increment, zero collisions, and `7.9565 m/s` peak speed. The GT
path projection covered about `1.9` path lengths. One full trajectory-stride
transit took `7.024 s` from GT-projected progress, but this is not an official
lap time: only one stride crossing was observed and the simulator's
`last_lap_time` included startup/idle. GT, odometry, and controller data were
captured at roughly 40 Hz. This is a useful faster practice bag, but it is a
CSV/GT path-follower run with MPC disabled—not a closed-loop MPC success.

With the freshly rebuilt source-plant evaluator, `850` one-step windows were
scored. The independent queue-phase A/B found yaw-rate RMSE about
`0.0404 rad/s` for steering slot 2 plus throttle slot 1, versus
`0.1126 rad/s` for the tested steering-slot-1/throttle-slot-2 phase. That
does not establish the true runtime queue phase. Replaying bridge request-
stamped commands through the carried source state yielded yaw-rate RMSE
`0.0758 rad/s`. The
750 ms open-loop rollout still fails the `5 cm` target: endpoint XY RMSE
`6.47 cm`, p95 `13.41 cm`, max `26.30 cm`; only `444/764` endpoints and
`427/764` complete paths stayed within 5 cm. The worst windows start around
`4.5–4.7 m/s` and `0.14–0.16 rad` steering; their yaw error grows sharply in
the first `25–75 ms` (one sampled window: yaw-rate error `0.145`, `0.366`,
`0.609 rad/s` at those steps) and reaches about `0.094 rad` by 750 ms. GT
heading re-anchoring is diagnostic only, but reduces 750 ms XY RMSE to
`0.0079 m`; this localizes most accumulated position error to yaw/heading
prediction, not pose integration. No future-truth correction is a deployable
fix. A separate future-steering-feedback oracle reduced 750 ms XY RMSE to
`4.34 cm` (p95 `8.49 cm`, full-path max `13.34 cm`), still failing the 5 cm
gate. Thus steering-rack prediction contributes, but cannot alone explain or
repair the remaining yaw/endpoint error.

Offline GT-seeded MPC replay on `854` synchronized bag points found:

| Replay mode | Optimal / degraded / rejected | Solver-cycle median / p95 / max |
|---|---:|---:|
| R1 only | 759 / 70 / 25 | 8.43 / 19.27 / 38.24 ms |
| R2 every cycle | 787 / 65 / 2 | 16.00 / 27.76 / 51.90 ms |
| R2 only after R1 rejection | 798 / 55 / 1 | 8.67 / 25.97 / 62.62 ms |

R2 materially improves offline candidate acceptance, but its conditional
replay still has one nonlinear corridor rejection and a long-tail overrun;
this does not justify promotion or establish online tracking. The remaining
reject is nonlinear feasibility, not ADMM nonconvergence: R1 exceeded its
upper corridor by about `1.26 cm` at stage 12, and R2 by about `1.09 cm` at
stage 21. Keep this GT-seeded result distinct from actual MPC execution.

Reproduction artifacts are in the same `live_runs/v300_native_controls_speed_governor_20261020/`
directory: `rollout_bridge_request_phase1.txt`,
`rollout_state_reanchor_diagnostic.txt`, `oracle_solver_inputs.csv`, and the
three `mpc_*_gt_replay.csv` files. No production plant, MPC default, raceline,
or runtime physics was changed from this analysis. Next, isolate and correct
the early yaw/steering transition residual in the reduced plant and examine
whether the remaining nonlinear rejection is caused by the reference corridor
or by model/prediction error before any supervised MPC run.

### Repeated five-lap capture on the faster v300 profile — 2026-10-20

Two independent practice batch-mode bags were captured using the 85%-scaled
v300 speed/throttle profile with its steering/path data unchanged. These are
development GT path-follower runs with MPC disabled; they are not closed-loop
MPC validation or a track-record claim. The table below contains the two
pre-odometry-change captures; the post-change five-lap retry is recorded
separately below.

| Bag | Laps / collisions | Subsequent lap times | Mean / best | Peak speed |
|---|---:|---|---:|---:|
| [`run 1`](../../live_runs/v300_native_controls_speed85_5lap_run1_20261020/run) | 5 / 0 | 7.704, 7.779, 7.525, 7.645 s | 7.663 / 7.525 s | 7.098 m/s |
| [`run 2`](../../live_runs/v300_native_controls_speed85_5lap_run2_20261020/run) | 5 / 0 | 7.749, 7.519, 7.712, 7.717 s | 7.674 / 7.519 s | 7.138 m/s |

The first reported lap in each bag includes startup/idle (`100.335 s` and
`135.562 s`) and is excluded from the four-lap means. Each SQLite bag is about
23 MiB and contains approximately 47k messages at the expected 40 Hz sensor
cadence. Follow-path diagnostics report about `4.8 cm` distance RMSE and
`8.8 cm` p95. These captures provide the requested repeated multi-lap data;
no further run was started for this update.

### Source-grounded throttle feedback use in production odometry — 2026-10-20

`VehicleController.FixedUpdate()` exposes `CurrentThrottle = currentT`, with
`currentT = DriveTorque / MotorTorque` sampled before `Drive()` updates the
next command. This is a normalized torque observation from the recently
applied actuator state, not wheel/body speed. The source `Drive()` branch
confirms that normalized zero means the configured full-brake branch. The
existing throttle-feedback subscription is now passed to the production
source-derived odometry plant as the completed packet interval's throttle; if
feedback is stale/unavailable, the prior delayed-command value remains the
input. The two-command request queue is still shifted and published unchanged
for future MPC stages. No new topic was added and runtime does not consume
ground truth.

The same two bags were rescored with identical bridge-reconstructed command
phase (`epoch offset +1`) with and without interval throttle feedback:

| Bag | Command-only 750 ms XY RMSE / p95 | Feedback 750 ms XY RMSE / p95 | Endpoints within 5 cm, command → feedback |
|---|---:|---:|---:|
| run 1 | 8.33 / 16.87 cm | 5.94 / 12.90 cm | 744/1500 → 1104/1500 |
| run 2 | 7.97 / 15.86 cm | 5.43 / 10.36 cm | 796/1546 → 1103/1546 |

At 750 ms, the feedback-conditioned full-path max-error RMSE/p95/max were
`6.41/13.72/36.53 cm` and `5.70/10.41/26.24 cm`, respectively. Feedback
helps materially, but the remaining error still fails the `5 cm` requirement.
The replay supplies feedback at each completed packet interval, supporting
retrospective odometry propagation; it is not a valid future-feedback source
for MPC's open-loop horizon. These bags predate this runtime-code change, so
the comparison is offline reconstruction, not online verification of the
changed binary.

The focused C++ source-plant test now verifies that feedback changes the
elapsed interval's wheel-spin propagation without rewriting the future
command queue. `colcon build --packages-select f1tenth_localization
--cmake-args -DBUILD_TESTING=ON` succeeded and the focused
`test_source_derived_odom_runtime` passed all 8 tests.

A post-change live check was then run in the same practice `batchmode`/Xvfb
procedure with the mounted source rebuilt, MPC disabled, and the same follower
profile. Its retained bag is
[`post-change partial run`](../../live_runs/v300_native_controls_speed85_5lap_post_odom_20261020/run).
It reached 3 laps at zero collisions (7.971, 7.577, 7.694 s; peak 7.193
m/s), then the follower's 150 ms stale-GT guard aborted at 163.6 ms. This is a
partial live validation, not a completed five-lap run.

The bag explains why its large errors cannot be read as a clean plant test:
there were 1,110 bridge timing events, with request-to-response latency
median/p95/max `19.84/66.56/261.94 ms`, versus `4.84/30.79/69.17 ms` and
`4.87/30.01/36.24 ms` in the two completed bags. Timing jitter correlated
strongly with truth-pose/twist closure error in the partial run (`r=0.828`;
lowest-jitter-quartile RMSE `0.493 cm`, highest `10.902 cm`). Unity physics-step
IDs were absent in all 1,110 events. The watcher initiated shutdown after the
stale-GT abort; no collision or independent Unity crash was observed.

On the partial bag, source-plant 750 ms endpoint XY RMSE was `8.73 cm` with
command-only replay and `8.54 cm` with interval feedback; the full-path
max-error RMSE changed `9.69 -> 9.45 cm`, while the worst path error worsened
`56.06 -> 70.64 cm`. Given the bridge latency anomaly, this does not confirm
the feedback change online and exposes a tail regression that needs a clean
capture. The production odometry itself logged forward-speed RMSE `0.144
m/s`, lateral-speed RMSE `0.015 m/s`, and `2.283 m` relative pose drift; the
reported yaw-rate equality is direct gyro pass-through, not a plant-model
result. At the time of this partial capture, the two earlier completed
five-lap bags still satisfied the repeated-capture request. A clean
post-change retry below subsequently completed the full five laps.

### Post-change five-lap retry — 2026-10-20

The retry used the same practice map, eight-copy v300 follower path,
85%-scaled speed/throttle profile, rebuilt source-mounted localization code,
MPC-disabled GT follower, Xvfb batch-mode simulator, and five-lap collision
watcher. The full bag is
[`post-change retry`](../../live_runs/v300_native_controls_speed85_5lap_post_odom_retry_20261020/run).
It completed 5 laps with 0 collisions. Simulator lap times were `8.203,
7.507, 7.732, 7.526, 7.541 s` (mean `7.702 s`, best `7.507 s`); peak speed
was `7.149 m/s`. The follower maintained `25.04 ms` mean odom cadence, with
`25.71 ms` p95, and path-distance RMSE/p95 were `4.8/8.7 cm`. This verifies
that the rebuilt odometry stack started and ran through five laps under the
existing development follower; it does not verify MPC control.

The production odometry output on this bag had forward-speed RMSE `0.0432
m/s` and lateral-speed RMSE `0.0089 m/s`. Integrated odometry position drift
ended at `0.179 m` over the capture. Its zero yaw-rate error is not evidence
for the plant model: the published yaw rate is gyro pass-through.

For a matched open-loop plant replay, command-only versus interval-throttle-
feedback results were:

| Horizon | Command-only XY RMSE / p95 | Interval-feedback XY RMSE / p95 |
|---|---:|---:|
| 25 ms | 0.73 / 0.88 cm | 0.73 / 0.87 cm |
| 100 ms | 1.14 / 1.88 cm | 1.04 / 1.37 cm |
| 250 ms | 2.09 / 4.44 cm | 1.61 / 3.07 cm |
| 500 ms | 3.84 / 8.40 cm | 2.89 / 6.60 cm |
| 750 ms | 5.96 / 13.36 cm | 4.69 / 10.59 cm |

At 750 ms, endpoints under 5 cm increased `1112/1536 -> 1182/1536`; full-path
max-error RMSE/p95/max changed `6.45/13.36/30.84 -> 5.29/11.66/17.53 cm`.
This supports using measured torque for retrospective odometry state
propagation, but still misses the 5 cm all-window/full-path target. The
interval-feedback replay uses the feedback available as each packet closes;
that signal cannot be supplied for future MPC stages. The MPC rollout must
therefore still be evaluated with planned commands only, and the model's
future prediction remains insufficiently accurate.

This retry's bridge timing was normal relative to the interrupted attempt:
1,563 timing events, request-to-response latency median/p95/max
`4.76/29.63/36.27 ms`, versus `19.84/66.56/261.94 ms` in the partial run.
The earlier timeout is documented as a one-run transport outlier, not a
vehicle collision or repeatable follower failure. Both complete pre-change
bags and this complete post-change bag are retained as the three comparable
multi-lap captures.

### Command-phase correction and three repeated five-lap captures — 2026-10-20

The production source-plant and MPC transition were corrected to consume the
newest request-associated command for the next modeled interval (queue slot 2),
instead of imposing one additional unsupported 25 ms delay through slot 1. The
production odometry still uses fresh measured throttle feedback for the
completed interval; future MPC stages still use planned commands. The matching
MPC delay-prefix length is one step. The RTI source Jacobian also preserves the
exact shift/append derivatives for both command-history coordinates rather
than overwriting them with plant finite-difference results.

Three same-procedure practice captures are now retained. Each used the
85%-scaled v300 command profile, the eight-copy continuous path, GT-based
development follower, MPC disabled, the established Xvfb batch-mode practice
simulator, and an immediate collision watcher. All reached five laps with zero
collisions; none is an MPC-controlled lap:

| Bag | Lap times (s) | Mean / best (s) | Peak speed |
|---|---|---:|---:|
| [`phase-fix run 1`](../../live_runs/v300_native_controls_speed85_phasefix_5lap_run1_20261020/run) | 8.100, 7.739, 7.772, 7.525, 7.623 | 7.752 / 7.525 | 7.13 m/s |
| [`phase-fix run 2`](../../live_runs/v300_native_controls_speed85_phasefix_5lap_run2_20261020/run) | startup lap 41.94; then 7.713, 7.768, 7.786, 7.525 | 7.698 over laps 2–5 / 7.525 | 7.14 m/s |
| [`phase-fix run 3`](../../live_runs/v300_native_controls_speed85_phasefix_5lap_run3_20261020/run) | 8.112, 7.785, 7.509, 7.750, 7.693 | 7.770 / 7.509 | 7.17 m/s |

Run 2's first lap is a startup/idle anomaly and is excluded from its four
steady-lap mean. Run 3 lasted 50.49 s as a rosbag (48,186 messages, 22.6 MiB);
sensor/command topics ran at approximately 40 Hz (odom mean/p95
`25.03/25.71 ms`, command mean `24.98 ms`). Its path-distance error was
`4.9 cm` RMSE, `9.0 cm` p95, and `15.0 cm` max. The bags are repeatable data
captures, not evidence of a faster official lap.

On run 3, an identical-input phase A/B replay supports the production phase
change: using the command associated with the current Bridge request rather
than the preceding request reduced one-step source-plant RMSE from
`0.1325` to `0.0902 rad/s` for yaw rate and from `0.0792` to `0.0651 m/s` for
forward speed. At 750 ms, XY endpoint RMSE changed from `11.19` to `9.08 cm`,
and complete-path maximum-error RMSE from `11.56` to `9.46 cm`. This is a
phase improvement, not an accuracy pass: with the corrected phase,
`759/1,551` 750 ms endpoints and `816/1,551` full paths still exceeded 5 cm;
worst path error was `45.2 cm`. At 750 ms, forward-speed RMSE was
`0.191 m/s`, yaw-rate RMSE `0.105 rad/s`, and lateral-speed RMSE `0.0134 m/s`.

Three oracle replays isolate remaining mismatch without proposing future
measurements as production inputs. With future interval throttle feedback
substituted, 750 ms forward-speed RMSE fell from `0.191` to `0.051 m/s` and XY
RMSE from `9.08` to `4.39 cm`. With future steering feedback substituted,
yaw-rate RMSE fell from `0.105` to `0.060 rad/s` and XY RMSE to `8.33 cm`.
Supplying both future feedback signals reduced 750 ms RMSE to `0.051 m/s` in
forward speed, `0.0036 m/s` lateral speed, `0.0288 rad/s` yaw rate, and
`2.76 cm` XY. Even this non-deployable oracle had `158/1,551` full paths over
5 cm and a `33.2 cm` maximum. This points to two distinguishable residuals:
command-to-applied-torque association dominates forward-speed drift, while
steering state/command association contributes yaw drift. It does not prove
the internal tire/contact equations are exact, and the oracle cannot be used
by an MPC predicting future steps.

The code-level phase correction is retained because same-capture A/B and the
new repeated run support it. No further delay, gain, or tire parameter is
changed on these results: the remaining mismatch needs a causal model of
command application timing based only on permitted runtime signals. The
current bridge diagnostics do not expose Unity's applied-command sequence or
FixedUpdate index, so future feedback replay is diagnostic only. The
production rollout remains outside the requested 5 cm full-path bound.
