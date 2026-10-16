# Odom, raceline optimizer, and MPC stage record — 2026-10-14

This document tracks the user's required sequence. Do not reopen a stage after
independent acceptance; record its acceptance once and proceed to the next.
Simulator physics and the reference trajectory remain unchanged. Real simulator
validation uses the established batch-mode launch, with collision monitoring
armed before motion and an immediate stop on any collision.

## Stage 1 — production odometry

**Status: complete; independent subagent accepted on 2026-10-14.**

The bounded, sensor-only high-steering speed correction was verified on
held-out production captures, with repeatable speed-estimate accuracy gain and
integrated position effectively neutral (not improved). It is not a
full-spectrum or full-lap odometry solution. Details and the one acceptance
record are in [ODOM_ACCURACY_PROGRESS_20261013.md](ODOM_ACCURACY_PROGRESS_20261013.md).
This stage is closed; do not repeat its validation.

## Stage 2 — offline-model raceline optimization

**Status: in progress; no independent acceptance yet.**

### Stage-2 objective and evidence boundary (clarified 2026-10-14)

Stage 2 is to produce a *theoretical* fastest raceline from a plant that
predicts the car near its measured slip boundary. The unchanged runtime MPC is
not the oracle for whether that raceline is physically feasible. A collision,
fallback, or tracking failure in a run with that MPC diagnoses the current
raceline/controller pairing only; it must not be fed back as proof that the
offline plant or raceline is physically infeasible. Such a run remains useful
as a separate integration/trackability observation.

The MPC was **not modified as part of this Stage-2 optimizer work**. The practice
screens used the existing pinned controller image/configuration. Stage 3,
including any MPC model or weight change, remains unopened.

The current Frenet minimum-time optimizer is not yet a 1:1 plant: it uses an
identified yaw-response model, an affine longitudinal response, and a
speed/steering lateral-acceleration envelope. It does not recursively propagate
the Unity-derived per-wheel spin, tire slip, tire relaxation, and body dynamics
as its optimization state. Its global lateral-envelope multiplier is therefore
an optimizer sensitivity knob, not a calibrated slip-boundary parameter.
Likewise, the current one-dimensional Bayesian search is not yet a valid
search for the theoretical slip edge: its earlier feasibility labels included
runtime collision/trackability and must not be interpreted as physical
feasibility. Keep those observations for runtime diagnosis, but do not use a
collision penalty to reject a theoretical line.

An existing empirical-curvature-envelope fit is not a substitute: its own
whole-run held-out GGV validation failed in every exercised moving speed band,
so it is not enabled as a capability law.

Important model distinction: the workspace contains two different offline
plants, and their results must not be conflated. The **source-built Unity
simulator** executes the actual project vehicle code and native WheelCollider
solver at 1 kHz. On a replay of the captured 12-lap practice command stream, it
completed all 74,850 fixed steps and achieved 0.09386 m global
relative-position RMSE (p95 0.18581 m, endpoint 0.10486 m), with per-lap local
path-increment RMSE of 0.00396–0.02206 m. This is the strong 9.39 cm result
and is the relevant exact-source offline evaluator. It is deterministic across
two independent replays. The remaining difference is against the recorded
practice run, not between offline replays; see
[the source audit](UNITY_WHEELCOLLIDER_PHYSX_SOURCE_AUDIT_20261012.md).

Separately, the **standalone research 3-D mechanics port** reconstructs the
Unity mechanics without executing Unity's native WheelCollider solver. Its
independent open-plane command-only holdout has about 0.17 m position RMSE at
1 s and 1.88 m at 5 s, with heading RMSE growing from 0.068 rad at 1 s to
0.239 rad at 5 s. Those poor figures apply to that approximate research port,
not to the source-built Unity simulator above. It must not be described as the
12-lap offline simulator or substituted for the exact-source result.

The 9.39 cm replay is meaningful evidence for offline raceline evaluation on
the tested practice command sequence; it is not a claim of 9 cm production
odometry, exact prediction for arbitrary optimized commands, or validated
slip-limit behavior. Prospective raceline rankings still need to be generated
and checked with the exact-source evaluator, with its cross-run residual kept
visible rather than assumed away.

For clarity, the following per-window analysis refers only to the standalone
research 3-D mechanics port, not the exact-source Unity evaluator. I reran that
same saved 1 kHz research-model replay only to expose per-window errors (no new
simulator data). Across its 39 independent 5 s windows, median
final horizontal position error is 1.17 m and p95 is 3.64 m. The median is
only 4 mm at 250 ms, then 6.3 cm at 1 s, so this is predominantly recursive
drift rather than a bad initial pose. The 0.10–0.20 rad initial-steering group
(6 windows) has 0.092 rad p95 heading error at 500 ms and 0.270 rad at 1 s;
the lower-steering group (27 windows) has 0.043 rad p95 at 1 s. In this small
holdout, final position error has only weak correlation with mean steering
(r=0.29), and essentially none with initial speed (r=-0.03) or mean throttle
(r=-0.02); these are diagnostics, not causal claims. The one-step 1 ms learned
yaw-residual candidate already in the archive worsens recursive held-out
5 s heading RMSE from 0.239 to about 0.670 rad and position RMSE from 1.88 to
about 7.03 m, so it is explicitly not the correction to promote.

The current same-runtime 2-lap screen is a useful example of why the evidence
must stay separate: scale 1.01875 predicted 5.698 s in the spatial optimizer
and completed two scored laps at 5.7378 s mean, versus 5.739 s predicted and
5.8073 s measured for the scale-1.0 screen. This is an encouraging screen
signal, not a validated theoretical-model gain or race result. Its measured
wheel/body-speed mismatch p95 rose from about 0.235 to 0.277 m/s and CTE p95
from 0.101 to 0.131 m. The mismatch is only a slip-related proxy, not direct
tire slip or force. The scale-1.0375 warmup collision is likewise a
current-stack failure, not evidence that its line is physically impossible.

Inner IPOPT iteration count is not the bottleneck in these cases: the candidate
solves converged in 34–38 iterations (and the 1.0 baseline in 79), with
successful solver status. Raising the per-solve iteration cap cannot repair
the missing physics. Bayesian optimization can help only after its objective
is grounded in a sufficiently accurate offline plant; more samples of the
current one-variable envelope scale would merely search that surrogate.

### Updated Stage-2 work order

1. Keep runtime crash/fallback/CTE and lap timing in a separate trackability
   ledger; do not use them as the theoretical plant's feasibility label.
2. Improve and independently score the command-only offline plant on recursive
   motion, including wheel spin, body forward/lateral speed, yaw, and position.
   Use the existing Unity source and captures; no new simulator run is justified
   until a specific unsupported input/state is identified.
3. Require whole-practice-lap agreement before using the plant to rank racelines.
   The optimizer's current 5.6–5.7 s values are only its own surrogate scores.
4. Then run Bayesian search over meaningful raceline/speed/throttle-policy
   parameters against the validated offline plant, with predicted slip and
   lap time reported separately. Preserve wall geometry and the reference line.
5. Only after the theoretical optimizer has passed those checks, use the
   established batch simulator to assess the separate existing-MPC pairing;
   stop a real run on collision, but do not retroactively call that a plant
   rejection. Stage 3 remains separate.

### Confirmed evidence

- A previous Y1/9-g candidate completed the full 1 warm-up + 10 scored + 1
  extra-lap protocol: 10/10 scored laps, mean 5.7622 s, best 5.7038 s, zero
  collisions. It ran with an earlier controller image, so it is a useful
  reference but not yet a same-image baseline for the current stack:
  `live_runs/y1_matched_9g_r04_mpc025_20261012_r01/analysis/race_report/summary.md`.
- A current-runtime-matched optimizer candidate with heading-error
  regularization weight 0.5 predicted 5.7446 s. It lowered optimizer heading
  error p95 from 0.3074 to 0.2777 rad and reduced nodes above 0.25 rad from 19
  to 14, but predicted lap time was 0.006 s slower than its parent.
- Its one-warm-up/one-scored-lap screen was collision-free at 5.8198 s, with
  |CTE| p95 5.7 cm and one-step yaw-rate error p95 0.101 rad/s. This was a
  screening result only.
- The subsequent full attempt was stopped on the first collision after seven
  complete scored laps. Those laps averaged 5.7897 s (best 5.6958 s); this is
  incomplete and not a successful candidate. The better best lap does not
  establish an improvement, and the mean does not beat the 5.7622 s historical
  reference.
- Immediately before the failure, simulator-truth yaw rate was about
  -1.4 rad/s and odometry yaw rate matched it. The synchronized MPC state was
  only about -0.7 rad/s, but the first-step rollout then swung to roughly
  -3.2 rad/s while the next simulator-truth sample was about -1.4 rad/s. Thus
  the current MPC state under-represented yaw, while its *next-step response*
  over-predicted the turn. Localization normal error was about 1 cm. The MPC
  then rejected its nonlinear rollout corridor; this is a measured failure
  signature, not yet a proven causal explanation. MPC remains closed until
  this stage is accepted.
- The single bridge timing fault was `Socket.IO connection lost` at bag time
  64.229 s, after the collision and after the vehicle was stopped. It did not
  precede or cause the collision.
- The hard optimizer heading cap of 0.35 rad failed to converge after 1,800
  iterations and is rejected. No trajectory was exported from that failed
  iterate.

### Same-runtime reference replay and yaw-response diagnosis

- The unchanged `y1_matched_9g` reference was replayed with the current image,
  production MPC configuration (no overlay), and the established Xvfb
  batch-mode protocol. It completed 8 of 10 scored laps (mean 5.7583 s, best
  5.6118 s) and then collided. This is not a valid complete race result and
  does not establish the old 10-lap result on the current image.
- The previous same-image `practice_y1_runtime_matched` trajectory also has a
  preserved incomplete run (4/10 scored, mean 5.7380 s, collision). Across
  these and the regularized candidate, three different lines fail with MPC
  nonlinear/corridor rejection; the optimizer output alone is not yet
  demonstrated to be the sole cause.
- The aligned one-step prediction log resolves the transient error direction:
  at the reference failure, predicted next yaw was about -2.6 to -2.9 rad/s
  versus next truth about -1.3 to -1.7; at the regularized-candidate failure,
  predictions reached about -3.2 to -3.4 versus truth about -1.4. The MPC's
  synchronized current yaw state was simultaneously lower in magnitude than
  current odometry/truth. This is a current-state/control-time plus fast
  turn-in-response mismatch, not an AMCL pose jump.
- I re-used only `r01` from
  `derived_yaw_full_spectrum_grid_dataset_20261009` to form a local steady-yaw
  map and scored its independent `r02` phases. At the two observed
  failure neighborhoods, held-out steady-yaw RMSE was 0.0082 rad/s
  (4.2 m/s, |steering| 0.20–0.32) and 0.0023 rad/s (2.9–3.9 m/s,
  |steering| 0.30–0.42); the kinematic baseline was 2.24 and 2.67 rad/s RMSE
  there. This is strong evidence for a bounded local model, not a full
  continuous-speed surface.
- A blocked leave-speed-out check is the limitation: errors are often
  0.2–0.4 rad/s RMSE (with several speed bands worse), despite near-zero error
  when validation repeats a sampled speed. The archive therefore does not
  justify blindly interpolating this table over arbitrary speeds. It must be
  evaluated on the continuous practice transitions before optimizer use.
- As a diagnostic only, a one-pole rollout using the train-only local surface
  and ground-truth/odom-equivalent *current* yaw reduced one-step yaw RMSE in
  the two saved pre-collision windows to 0.072 and 0.100 rad/s, versus the
  current MPC's 0.603 and 0.752 rad/s. It still had p95 errors 0.155 and
  0.192 rad/s, and it does not include the deployed MPC's control-time state
  initialization. This is evidence for the next model correction, not a
  production improvement or a reason to accept Stage 2.

### Preserved artifacts

- Candidate and solve reports:
  `live_runs/raceline_candidates/practice_y1_runtime_heading_regularized_050_20261014/`
- One-lap screen and bag:
  `live_runs/practice_y1_runtime_heading_regularized_050_screen_20261014/`
- Incomplete full attempt, bag, scoring, failure timeline, latency analysis,
  and vehicle contract:
  `live_runs/practice_y1_runtime_heading_regularized_050_10lap_20261014/`
- Failed hard-cap attempt:
  `live_runs/raceline_candidates/practice_y1_heading_cap035_20261014/`
- New per-window 1 kHz plant replay diagnostics (same existing train/holdout
  captures, no new simulator run):
  `live_runs/unity_exact_binary_audit_20261012/stage2_plant_window_diagnostics_20261014.json`

The full-attempt manifest originally recorded `build_localization=0` because
the manifest command did not inherit the launch environment. It has been
corrected to `1` from the captured launch/build record; the MPC build was also
enabled. No bag contents or source code were changed by this metadata repair.

### Superseded Stage-2 acceptance note

The earlier proposal to require the unchanged MPC to track a candidate before
Stage-2 acceptance is superseded by the clarified objective above. Keep
`y1_matched_9g` unchanged as the runtime comparison reference, but a runtime
collision is not a theoretical plant/raceline rejection. Stage-2 acceptance
requires a credible offline plant, a slip-aware theoretical optimizer, and
held-out whole-lap model agreement. Runtime pairing remains a separate report.

### Exact-source generated-lap steering screens (2026-10-14)

These are development-harness experiments in the source-built Unity player,
not production odometry/MPC results and not new trajectory geometry. The same
28.410 m practice reference path was followed in closed loop using the Unity
rigid-body pose and a spatial pure-pursuit controller; Unity physics remained
at 1 kHz and commands at 40 Hz. The exported practice collider was installed,
and the player aborted on first collider contact. The map-clearance figures
below are a separate 0.20 m vehicle-center margin screen, not collision truth.

- `r02`, initial closed-loop baseline: 6.676 s; no Unity collision; path RMS
  0.312 m, maximum 0.547 m; peak speed 7.87 m/s. It failed the map-margin
  screen (minimum 0.05 m; 91.1% of samples at least 0.20 m).
- `r07`, global target-speed scale 1.12: rejected. It completed without a
  collision but left the route badly (8.876 s, RMS 5.003 m, maximum 13.288 m).
  This showed that a global speed increase with the then-current steering
  schedule was not viable.
- `r08`, sector speed scales
  `[1.12,1.12,1.12,1.08,1.04,1,1,1,1,1,1,1]`: 6.501 s, no collision,
  RMS 0.300 m, maximum 0.548 m, peak 8.237 m/s. It improved 0.175 s over
  `r02`, but only 92.46% of samples met the 0.20 m map-margin screen; the
  weakest areas were sectors 6–7 and 9 (one-based numbering).
- `r09` adds per-sector steering-gain multipliers
  `[1,1,1,1.10,1.20,1.20,1.10,1.20,1.10,1,1,1]`, retaining the `r08`
  speed schedule and other controller settings. It completed the 28.410 m
  generated lap in 6.301 s, with zero Unity collisions, 0.205 m RMS and
  0.400 m maximum tracking error, and the same 8.237 m/s peak. Against `r08`,
  time improved by 0.200 s and tracking RMS fell 32%. Every captured 1 kHz
  sample passed the map-derived 0.20 m margin screen; minimum clearance was
  0.350 m (p05 0.400 m). The median signed path error in the former weak
  sectors 6–9 fell from `+0.467,+0.459,-0.219,-0.444 m` to
  `+0.256,+0.226,-0.117,-0.178 m`, respectively. The largest remaining
  median offset is about 0.38 m in sector 12, where the map margin remains
  about 0.40 m.

The r09 result is a meaningful, simulator-confirmed improvement to local
path-following behavior and speed over the earlier generated-lap screens. It
does not validate an MPC, a production controller, or an offline theoretical
slip-limit optimizer, and its 6.301 s is not close enough to claim the 4.9 s
goal. The dev-only profile builder accepts `--sector-steering-gains`, and the
source runner applies each point's `steering_gain_scale`; no Unity vehicle
physics, production odometry/MPC, or reference trajectory was changed. The
candidate profile, 1 kHz capture, and result are preserved in
`live_runs/unity_source_raceline_feedback_mintime_20261014_r09/`; its source
player build is in
`live_runs/unity_source_raceline_feedback_player_20261014_r02/`.

The follow-up `r10` increased only the target-speed scales in sectors 1–3
(from `1.12,1.12,1.12` to `1.16,1.20,1.20`) while preserving the r09 steering
schedule. It did **not** improve motion: lap time stayed 6.301 s, peak body
speed stayed 8.237 m/s, and path RMS was effectively unchanged at 0.205 m.
The measured throttle command in sectors 2–3 rose from 0.484/0.459 to
0.616/0.614, while body speeds remained 6.839/7.877 m/s (r09) versus
6.839/7.879 m/s (r10). At the same time, all-wheel mean `WheelHit.forwardSlip`
rose from 0.43/0.21 to 0.55/0.40, and mean wheel RPM rose from about
1965/1883 to 2504/2513. The practice prefab's forward friction curve peaks at
slip 0.15 and transitions to its asymptote at 0.25. This is direct evidence
that the extra target-speed feedback generated more wheel rotation/slip but
essentially no extra body speed in those sectors. The r10 clearance screen
remained identical to r09 (minimum 0.35 m, all samples above 0.20 m), with no
Unity collision. `r10` therefore does not count as a speed gain.

This is a source-simulation diagnostic: `WheelHit.forwardSlip` is Unity
internal telemetry, not an input used by the candidate controller and not
available as a competition sensor. The F1TENTH source also exposes a wheel
geometry distinction requiring separate odometry review: each WheelCollider
has radius 0.059 m, while `VehicleController.WheelRadius` is 0.0325 m and
`MotorEncoder` averages raw WheelCollider RPM. No radius or physics values
were changed here, and this observation alone does not establish which value
is the correct encoder conversion.

The `r11` targeted feedback-limit run kept r10's speed targets and r09's
steering schedule, but scaled speed-error feedback to 40% only in sectors 2–3.
It completed with zero collisions in the same 6.301 s, with essentially
unchanged body speeds (6.839/7.877 m/s) and path RMS (0.204 m). The throttle
commands fell from r10's 0.616/0.614 to 0.425/0.404; mean forward slip fell
from 0.553/0.396 to 0.355/0.155, and mean wheel RPM fell from 2503/2513 to
1727/1655. Peak body speed remained 8.238 m/s. Minimum map clearance was
0.335 m, and every sample remained above the 0.20 m screen. This is a strong
wheel-spin reduction at no measured lap-time or body-speed cost on this lap,
but it has not yet produced a faster lap or established repeatability.

The control reads only each command's current target-speed error; it does not
read Unity slip or future truth. The speed-feedback multiplier is a
route-specific development experiment, not a promoted production policy. The
`r12` follow-up halved that multiplier again to 0.20 in sectors 2–3. It
completed in 6.201 s (0.100 s faster than r11), with zero collisions, 0.212 m
path RMS, 0.399 m maximum error, 8.748 m/s peak speed, and 0.350 m minimum
map-derived clearance; every sample still passed the 0.20 m margin. In sectors
2–3, mean speed rose to 7.074/8.495 m/s while mean throttle fell to
0.351/0.308; mean forward slip was +0.188/-0.084. Thus the limited-feedback
candidate is currently the best generated-lap screen, and the data support
reducing futile positive feedback under those conditions. This is a single
lap, not a repeated-lap or production validation.

The first high-steering speed push (`r13`) increased sectors 6–9 target scales
by 12%, retaining the r12 limits in sectors 2–3 and the r09 steering schedule.
It completed with no Unity collision but was rejected: time regressed to
6.851 s, path RMS rose to 1.194 m (max 2.971 m), and tracking first became
materially poor in sector 8 before leaving the map-derived margin in sectors
9–12. Sector 6–7 speeds rose to 3.71/5.21 m/s with moderate CTE; sector 9
reached 6.77 m/s while median CTE grew to -0.91 m. This localizes the excess
speed demand to the later high-curvature portion rather than proving all four
sectors unsupported.

Follow-up `r14` increased only sectors 6–7 by 12%; `r15` kept those targets
and raised sector-9 steering gain from 1.10 to 1.35. Neither recovered the r12
result: r14 was 6.551 s (RMS 0.553 m, max 1.097 m), and r15 was 6.501 s (RMS
0.530 m, max 1.049 m), both with zero Unity collisions. Drift was already
visible in sector 8 and persisted through sector 9; raising steering gain
changed the command but left sector-9 median CTE near -0.69 m. This rejects a
local sector-9 steering gain as the fix. These runs reinforce that speed and
steering schedules must be optimized jointly across a turn sequence.

The `r16` straight-line push raised only sectors 2–3 target scales from 1.20
to 1.30 while retaining their 0.20 speed-feedback multipliers. It regressed
to 6.301 s and reduced peak speed from r12's 8.748 to 8.263 m/s. Across
sectors 2–3, mean body speed fell from 7.074/8.495 to 6.839/7.883 m/s, while
throttle rose from 0.351/0.308 to 0.394/0.373 and mean forward slip rose from
`+0.188/-0.084` to `+0.303/+0.107`. Raising the speed demand above the r12
point therefore causes more wheelspin and less body speed even with feedback
capped.

The `r17` feedback bracket retained r12's target and steering schedules and
reduced only the speed-feedback multipliers in sectors 2–3 from 0.20 to 0.10.
The generated lap completed without Unity collision in 6.251 s (0.050 s slower
than r12); path RMS improved slightly from 0.212 m to 0.204 m, maximum error
was effectively unchanged (0.399 to 0.397 m), target-speed RMSE improved from
0.532 to 0.478 m/s, and peak body speed fell from 8.748 to 8.437 m/s. In
sectors 2–3, mean throttle fell 0.351/0.308 to 0.325/0.285, all-wheel mean
forward slip fell +0.188/-0.084 to +0.125/-0.120, and mean wheel RPM fell
1423/1260 to 1320/1169. Sector-3 mean body speed also fell 8.495 to 8.207 m/s.
This supports the expected tradeoff: lower feedback reduces wheel activity
and slightly improves path/target tracking, but gives back speed and time on
this one lap. The midpoint r18 result follows below. Keep these screens
separate from the still-required recursive command-only plant validation and
the unopened MPC stage.

The `r18` midpoint test used 0.15 speed-feedback multipliers in sectors 2–3,
with all other parameters held to r12/r17. It completed without collision in
6.201 s, matching r12; path RMS was 0.207 m (vs 0.212 m in r12), maximum error
0.398 m, target-speed RMSE 0.503 m/s, and peak speed 8.613 m/s. Sector-2 mean
speed/throttle/slip/RPM were 7.138 m/s / 0.336 / +0.142 / 1362, intermediate
between r12 and r17. Sector 3 was 8.382 m/s / 0.303 / -0.086 / 1240, close to
r12. This is the most balanced setting of the three single-lap screens, not a
lap-time improvement or repeatability result; retain it as the current
development-profile candidate, with r12 retained as the fastest reference.

The `r19` anticipatory-steering test kept the r14 +12% target speeds in sectors
6–7, but increased sector-8 steering gain from 1.20 to 1.50; sector 9 retained
r15's 1.35 gain. This directly tested whether the drift that began in sector 8
could be corrected earlier. It did not: the lap slowed to 6.701 s, path RMS
rose to 0.829 m (maximum 1.792 m), and only 70.2% of samples met the 0.20 m
map-margin screen (minimum clearance 0 m). The median signed path error
worsened from r14's -0.171/-0.685/-0.553 m in sectors 8–10 to
-0.212/-0.982/-1.155 m. There was no Unity collision, but this candidate is
rejected on route fidelity, clearance, and time. More steering gain is not the
fix for the high-speed-entry drift.

The `r20` bounded entry-speed test reduced the r14 increase to +6% in sectors
6–7, restored the r09 steering schedule, and retained r18's 0.15 speed-feedback
scales in sectors 2–3. It completed without Unity collision in 6.151 s, 0.050 s
faster than r12/r18, with 0.213 m path RMS, 0.395 m maximum error, 0.507 m/s
target-speed RMSE, and 8.613 m/s peak speed. The map-margin screen passed at
all samples: minimum clearance was 0.316 m and p05 was 0.400 m. Sector speeds
rose from r18's 3.378/4.733 m/s to 3.547/4.961 m/s in sectors 6–7; downstream
sector-8/9 median offsets moved slightly to -0.135/-0.214 m but stayed inside
the margin. The captured-telemetry gate scorer reports a 6.1348 s gate-to-gate
time, 48.4 ms faster than r12's 6.1832 s; Unity's `offline_result.json` is
quantized to 25 ms because lap completion is checked at command boundaries.
The independent fresh-reset repeat `r21` reproduced the exact 6.1348 s gate
time, 6.151 s command-boundary duration, path RMS (0.213 m), max error
(0.395 m), target-speed RMSE (0.507 m/s), peak speed (8.613 m/s), and
clearance statistics (minimum 0.316 m, p05 0.400 m), with zero collisions.
The +6% gain is repeatable in this source-Unity generated-lap setup.

The `r22` +9% target-speed probe in sectors 6–7 completed without Unity
collision. Its command-boundary duration was 6.151 s, while the 1 kHz gate
crossing time was 6.1263 s (8.5 ms faster than r20). Path RMS rose to 0.227 m,
maximum error was 0.387 m, and target-speed RMSE was 0.510 m/s. The map-margin
screen was just below its threshold: minimum clearance 0.180 m and 99.84% of
samples at least 0.20 m (p05 0.304 m). Keep it as a faster-but-rejected
boundary observation, not a candidate.

The `r23` +8% point completed without collision and passed the map-margin
screen at every sample (minimum 0.292 m; p05 0.381 m). Its command-boundary
duration was 6.126 s, and the 1 kHz gate crossing was 6.1218 s—13.0 ms faster
than r20 and 61.4 ms faster than r12. Path RMS was 0.215 m (r20 0.213 m),
maximum error 0.393 m, target-speed RMSE 0.510 m/s, and peak body speed
8.613 m/s. This is a new provisional best that stays inside the map margin;
the fresh-reset repeat `r24` reproduced the exact 6.1218 s gate time, all
tracking metrics, and the same 0.292 m minimum / 0.381 m p05 clearance with
zero collisions. Retain r23/r24's +8% entry speed as the current
repeatable development candidate. Its sector-6/7 target demand is supported
better than +9% under the configured margin; the +12% profile remains
rejected.

The `r25` geometry candidate applied half of the smoothed sector offsets
measured from r24 to a separate profile; the source trajectory remains
unchanged. The generated line shortened from 28.410 m to 27.880 m. Unity
completed it without collision in 6.026 s at command boundaries; the 1 kHz
gate scorer measured 6.0129 s, 108.9 ms faster than r24. Path RMS rose to
0.234 m and maximum error to 0.429 m, while target-speed RMSE was 0.515 m/s.
The map-margin screen still passed at all samples (minimum 0.292 m, p05
0.381 m). Fresh-reset repeat `r26` reproduced the same profile geometry,
6.0129 s gate time, tracking metrics, and clearance exactly, with zero
collisions. This is repeatable progress from a measured path correction, but
it remains a development pure-pursuit profile—not MPC or the production
optimizer—and its 6.013 s is still slower than the prior 5.7622 s full-practice
result.

The second residual correction `r27` moved the geometry back toward a longer
path (28.033 m) and slowed the gate time to 6.0450 s; RMS improved slightly to
0.233 m, max error 0.417 m, and minimum map clearance remained 0.269 m. It is
rejected for the speed objective. A direct 0.75 correction from the original
reference using the same r24 measured offsets (`r28`) shortened the route to
27.617 m and completed at 5.9627 s gate-to-gate, 50.2 ms faster than r25/r26
and 159.1 ms faster than r23/r24. It had zero Unity collisions and all samples
passed the map margin (minimum 0.292 m, p05 0.381 m), but tracking errors rose
to 0.246 m RMS / 0.456 m max and target-speed RMSE to 0.517 m/s. This is a
promising new provisional best. The fresh-reset repeat `r29` reproduced the
same 5.9627 s gate time, errors, and clearance exactly, with zero collisions.
The original reference trajectory remains unchanged. The full correction
candidate `r30` shortened the profile geometry further to 27.356 m, but is
rejected: although the development runner's route-index loop reported complete,
the independent physical start/finish gate had no second crossing. Path error
rose to 0.360 m RMS / 0.638 m max, and map margin failed badly (minimum 0 m;
only 93.3% of samples at least 0.20 m). There was no Unity collision, but this
was not a valid lap and must not be scored as one. This demonstrates why the
gate-crossing scorer, not the runner's route-loop status alone, is authoritative.
The intermediate 0.80 correction `r31` did complete the physical gate-to-gate
lap in 5.9541 s, 8.6 ms faster than r28. Its separate line is 27.565 m; path
RMS was 0.249 m and maximum error 0.452 m. It had zero Unity collisions and
passed every map-margin sample (minimum 0.292 m, p05 0.316 m). Fresh-reset
repeat `r32` reproduced its 5.9541 s gate time and all reported metrics exactly.
Retain the 0.80 geometry as the current best generated-profile candidate. The
full 1.0 correction remains rejected. Raising only sectors 6–7 to +9% on this
geometry (`r33`) also failed: the independent gate saw no finish crossing and
only 93.2% of samples met the map margin (minimum 0 m). Sector-6/7 body speeds
rose to 3.625/5.088 m/s, with mean forward slip 0.089/0.099. Reject r33 despite
the runner's route-loop completion flag. The +8.5% probe `r34` did produce a
physical gate lap, but was slower at 5.9709 s, with minimum map clearance
0.141 m and only 97.9% of samples meeting the 0.20 m margin. Reject it. The
repeatable +8% r31/r32 point is the best tested speed setting in these sectors;
do not raise it further without changing another part of the plan.

## Stage 3 — MPC model and horizon rollout

**Status: not started.**

Do not change MPC until Stage 2 has an independently accepted completion.
When opened, compare rollout against simulator truth at correctly aligned
timestamps and report yaw, forward/lateral speed, and position separately;
fix a demonstrated model/state mismatch before changing weights.

## Overall target and current result

The requested final goal is a repeatable 10-scored-lap average below 4.9 s.
This has **not** been achieved. The best located full practice result is still
5.7622 s mean with zero collisions. Stage 1 is complete; Stages 2 and 3 are not.

### Continued exact-source sector search (r35–r56, 2026-10-14)

All of these are separate, one-generated-lap development-harness runs in the
same practice simulator and source player described above. Unity physics
remained at 1 kHz, commands at 40 Hz, the original mintime trajectory file was
not edited, and the extracted Unity practice collider remained active with
first-collision abort. Lap times below are from interpolated 1 kHz physical
start/finish gate crossings; `offline_result.json`'s route-loop completion is
not accepted as lap timing. The 0.20 m map-clearance figure is a separate
vehicle-center geometry screen, not a substitute for collision truth.

- `r35` halved speed-error feedback only in sector 1: mean forward-slip
  magnitude fell about 0.51→0.34, but body speed was effectively unchanged and
  the physical lap was 1.8 ms slower. `r36` reduced that multiplier to 0.10;
  the single lap was about 6 ms slower than its parent. Neither is a promoted
  change. `r38` raised only sectors 2–3 target speeds by 10%; wheel slip rose
  and body speed fell (sector 2: 7.12→6.84 m/s; sector 3: 8.38→7.90 m/s), while
  map margin also failed. Reject it. `r39`'s very-low-feedback variant did
  not physically finish and departed the line; its runner loop status is not
  a valid lap. These reinforce the existing finding that extra demand can
  increase wheel rotation without increasing body speed.
- `r37` tested signed throttle as an alternative way to modulate braking. It
  did not finish the physical gate and had severe tracking error, so it is not
  promoted. Source inspection explains why: the practice scene uses the
  common-actuator AWD configuration with all-wheel braking. In
  `VehicleController.Drive`, exactly-zero throttle applies the full configured
  brake torque on all four wheels; positive throttle clears brake torque and
  drives the wheels. With this scene's AWD mode, negative throttle takes the
  negative-motor-torque path rather than the independent-brake path, so it is
  much weaker than the zero-command brake. The signed mode remains opt-in in
  the development harness only; production behavior and Unity physics were
  not changed.
- `r40/r41` repeated the `r32` geometry with sector-6/7 target scale +8.5% and
  steering gains 1.3/1.2. The exact physical lap was 5.9116 s on the repeat,
  42.5 ms faster than `r32`, collision-free, with minimum map clearance
  0.3808 m and p05 0.4472 m. `r42` raised only those target scales to +9%; it
  was 1.2 ms slower than `r40`. The +8.5% setting is the better of these
  tested local points, not a global optimum.
- `r43/r44` retained `r40` but raised only sector-12 steering gain from 1.0
  to 1.2. The reset repeat reproduced 5.8957 s exactly, 15.9 ms faster than
  `r40`, with zero collisions and the same 0.3808 m minimum map clearance.
  Sector-12 median absolute path offset fell about 0.439→0.368 m. A larger
  1.4 gain (`r45`) oversteered, slowed to 5.9740 s, and failed the clearance
  screen (minimum 0 m); reject it. Sector-1 gain 1.2 (`r46`) was effectively
  neutral, 0.4 ms slower than `r43`, and is not promoted.
- Sector-12 target-speed probes then gave repeatable, collision-free results:
  `r47/r48` at +5% scored 5.8727 s; `r50/r51` at +7.5% scored 5.8614 s;
  `r52/r53` at +8.75% scored 5.8569 s; and `r54/r55` at +9.375% scored
  5.8542 s. The repeats reproduced their physical gate times and clearance
  exactly. For `r54/r55`, path RMS/max were 0.2178/0.4133 m; minimum map
  clearance was 0.3808 m, p05 0.4472 m, and every sample passed the 0.20 m
  margin. Relative to +8.75%, +9.375% buys only 2.7 ms; sector-12 mean body
  speed changes by about 0.02 m/s and peak speed is unchanged. Treat that as a
  small repeatable screen improvement, not a material optimization result.
  The +10% `r49` profile had no second physical gate crossing and is invalid
  as a lap despite the runner's route-loop flag. Thus +9.375% is the highest
  tested passing sector-12 point; the failure at +10% is a boundary observation,
  not evidence that the entire sector's physical limit is known.
- The 1 kHz telemetry identifies a more important brake-zone behavior on the
  repeatable `r52–r55` profile. In sector 4, mean body speed is 6.92 m/s
  against a 6.24 m/s target, throttle is zero for 99.7% of samples, and mean
  absolute four-wheel `forwardSlip` is about 0.985. In sector 5, mean speed is
  4.04 m/s against a 3.43 m/s target, with throttle zero for 51.8% and mean
  absolute forward-slip proxy about 0.57. This is consistent with full-brake
  wheel lock under the source controller mapping; `forwardSlip` is internal
  Unity telemetry and is not available to a competition controller.
- `r56` was a deliberate two-sector test that raised only the speed targets
  for the connected S4–S5 braking zone to scales 1.20/1.22, chosen to bring
  their requested speeds near the measured mean speeds. It reduced mean
  absolute forward slip modestly (S4 ~0.99→0.89; S5 ~0.57→0.53), but the run
  did not complete a physical lap: it reached the 20 s profile limit with no
  collision, 30.49 m path RMS, 61.52 m maximum path error, and only 26.80 m of
  27.56 m route progress. The first >1 m route error appeared around 10.221 s
  in sector 6, after the S5 speed increase. Reject this coupled S4/S5 setting;
  it is not evidence that either isolated sector is independently infeasible.

The current fastest *generated-profile screen* is `r88/r89` at 5.731677 s;
the lower-error `r79/r80` profile remains the tracking alternative.
This is a one-lap development pure-pursuit screen, not a competition-MPC or
10-lap result; its numerical comparison to the historical 5.7622 s full-run
mean is not apples-to-apples, and the 4.9 s ten-lap objective remains unmet.
These screens are not MPC, production odometry, or a 10-lap result, and must
not be presented as beating that reference. The S4/S5 interaction has now
been isolated: S4 target increases reduce the wheel-lock proxy without
increasing body speed; S5 target increases lose practice-line tracking as the
car enters S6. Do not carry those S5 settings forward. The next raceline
experiment returns to the known-good sector schedule and tests a small increase
in the measured geometry-correction fraction. No production odometry, MPC,
reference trajectory, or simulator physics was changed in these runs.

The isolated S4 checks are `r57` (target scale 1.20) and `r58` (1.25), each
compared with the known-good `r55` profile. Both completed with zero collisions
and exactly the same map-clearance statistics (minimum 0.3808 m; all samples
above 0.20 m). Their gate times were 5.854213 and 5.854229 s versus 5.854219 s
for `r55`; body speed in S4 remained 6.921 m/s. Mean throttle in S4 rose from
0 to 0.027 and then 0.084, while mean absolute forward-slip proxy fell from
0.985 to 0.892 and 0.692. Thus the actuator mapping can reduce wheel lock
without measurable body-speed or lap-time benefit at these target levels.

The S5-only `r59` setting (1.22; S4 restored to 1.08) reproduced the `r56`
failure signature: no collision, no second physical gate crossing, 20 s
profile-limit timeout, 30.49 m path RMS, and 26.80 m route progress. S5 body
speed rose to 4.277 m/s from 4.039 m/s and its absolute-slip proxy fell only
0.573→0.531; the first >1 m route deviation was in S6 at 10.221 s. The smaller
S5-only midpoint `r60` (1.15) eventually crossed the physical gate, but only
after 7.652868 s; path RMS/max were 3.27/8.78 m and only 49.9% of samples
passed the 0.20 m clearance screen (minimum clearance 0 m). It is rejected.
These results show that S5 target increases of the tested size are not
transferable through the present S6 transition, even when isolated from S4.

Artifacts for this continuation are in
`live_runs/unity_source_raceline_feedback_mintime_20261014_r57/` through `r60/`.

The geometry check `r61/r62` increased only the offset-correction fraction
from 0.80 to 0.85, retaining the `r55` sector speed, steering, and feedback
settings. The generated line shortened from 27.565 m to 27.512 m. Both
independent resets completed the physical gate in exactly 5.847898 s, with
zero collisions and identical results: path RMS/max 0.221/0.423 m, minimum
map clearance 0.3808 m, p05 0.4301 m, and all samples above the 0.20 m margin.
This is a repeatable 6.3 ms gain over the 0.80 geometry at a small tracking
error cost; keep 0.85 as the current development candidate. The next single
parameter probe is correction fraction 0.90, between this result and the
previously failing full correction. Artifacts are under `..._r61/` and
`..._r62/` in the same run directory prefix.

The 0.90 geometry check `r63/r64` changed no other profile values. Its
generated line is 27.460 m long. Both reset runs completed the physical gate
in exactly 5.840524 s with zero collisions; path RMS/max were 0.224/0.430 m,
minimum map clearance 0.3808 m, p05 0.4301 m, and all samples passed the
0.20 m screen. This is a repeatable 7.4 ms gain over 0.85 and 14.2 ms over
0.80.

The 0.95 geometry check `r65/r66` changed no other profile values. Its
generated line is 27.408 m long. Both resets completed the physical gate in
exactly 5.831217 s with zero collisions; path RMS/max were 0.228/0.436 m,
minimum map clearance 0.3808 m, p05 0.4301 m, and all samples passed the
0.20 m screen. This is a repeatable 9.3 ms gain over 0.90 and 23.0 ms over
0.80.

The 0.975 geometry check `r67/r68` changed no other profile values. Its
generated line is 27.382 m long. Both reset runs completed the physical gate
in exactly 5.828305 s with zero collisions; path RMS/max were 0.230/0.444 m,
minimum map clearance 0.3808 m, p05 0.4272 m, and all samples passed the
0.20 m screen. This is a repeatable 2.9 ms gain over 0.95 and 26.9 ms over
0.80.

The full-correction test `r69` with the current schedule completed safely in
5.829933 s (zero collisions, 0.3808 m minimum map clearance, all samples above
0.20 m). It was 1.6 ms slower than 0.975, with path RMS/max 0.233/0.446 m.
This resolves the old failure: full correction is viable under the current
speed/steering schedule, but is not faster than 0.975. The current tested
geometry optimum remains 0.975. The follow-up `r70` raised only sector-12
target scale from 1.09375 to 1.10 on the 0.975 geometry. Unity's route-loop
flag said complete, but the independent physical gate scorer found no second
crossing, so it is not a valid lap and is rejected. The repeatable +9.375%
S12 setting remains the last valid point. Artifacts are under `..._r63/`
through `..._r70/` in the same run directory prefix.

The high-curvature S6 steering-gain bracket retained the 0.975 geometry and
all other `r67` settings. `r71` raised only the S6 gain from 1.30 to 1.40; it
was poor: 7.1164 s gate time, 2.326 m path RMS, 6.567 m maximum error, and
only 46.0% of samples passed the map-margin screen (minimum clearance 0 m).
S6 median residual increased 0.296→0.461 m. Reject 1.40. `r72` lowered only
the gain to 1.20; it completed at 5.870424 s, 42.1 ms slower than 1.30, with
path RMS/max 0.259/0.433 m, minimum clearance 0.206 m, p05 0.316 m, and all
samples passing the 0.20 m screen. S6 median residual was 0.337 m. These
`r73` tested the 1.35 midpoint. It regressed to 6.220952 s, path RMS/max
0.718/1.518 m, and only 69.9% of samples passed the margin (minimum 0 m).
The S6 median residual was 0.333 m. The tested bracket therefore supports
keeping S6 gain at 1.30; lower 1.20 and higher 1.35/1.40 were worse than the
5.828305 s `r67/r68` baseline. Do not keep tuning S6 gain.

The S11 residual-led test raised only S11 steering gain from 1.0. At 1.20,
`r74/r75` reproduced 5.754352 s exactly, zero collisions, path RMS/max
0.197/0.408 m, and unchanged clearance (minimum 0.3808 m, every sample above
0.20 m). S11 median residual fell 0.217→0.176 m and absolute p95
0.321→0.201 m. Raising it to 1.30 in `r76/r77` gave another exact repeated
improvement: 5.740019 s, path RMS/max 0.192/0.408 m, same clearance, S11
median/p95abs 0.169/0.181 m. This is 88.286 ms faster than `r67/r68` in this
harness. It also reduced S12 median/p95abs residual to 0.215/0.230 m, showing
the sector transition is coupled. On `r77`, the largest remaining measured
median residuals are S1 +0.341 m, S6 +0.296 m, and S7 +0.225 m. The next
single-variable check is S1 steering gain 1.20 (baseline 1.0), with other
parameters fixed. Artifacts are under `..._r71/` through `..._r77/`.

The S7 follow-up `r78` raised only its steering-gain multiplier from
1.20 to 1.30. It completed the physical gate in 5.740033 s with zero Unity
collisions; compared with `r76/r77` at 5.740019 s, this is a 14.6 microsecond
slowdown and therefore no meaningful time gain. Path RMS improved slightly
from 0.191814 to 0.189748 m, max error stayed 0.407891 m, and the sampled
minimum map clearance increased from 0.380789 to 0.400000 m (p05 0.430116 m;
all samples above 0.20 m). Treat S7=1.30 as a small clearance improvement, not
a faster result. The S1 residual-led probe `r79` raised only its S1 steering
gain from 1.0 to 1.20 on top of the r78 profile. It finished in 5.739809 s,
0.224 ms faster than r78, with zero collisions; path RMS improved to 0.185026
m, maximum remained 0.407891 m, and minimum map clearance was 0.403113 m
(p05 0.447214 m; all samples above 0.20 m). The fresh-reset repeat `r80`
reproduced the exact gate time and all metrics. This is a small but
repeatable generated-screen improvement, not a material lap-time advance.
Artifacts are under `..._r78/` through `..._r80/`.

The S1 steering-gain bracket at 1.30 was tested as `r81` and rejected. It
took 5.746402 s, 6.6 ms slower than the exact-repeat 1.20 runs `r79/r80`;
collisions and map clearance were unchanged. Against its own target, S1
median/p95abs error changed only 0.324/0.407→0.319/0.407 m, while S2/S3 also
shifted. The steering-gain sweep is closed at S1=1.20.

To test whether the remaining S6 residual is better addressed by geometry
instead of more steering gain, the profile builder now accepts an optional
per-sector signed path-offset adjustment in metres. The default is all zero,
so existing profiles are unchanged. `r82` added +0.075 m in S6 only, about a
quarter of that sector's measured +0.292 m median residual. It improved S6
median/p95abs error to +0.259/0.281 m and overall path RMS from 0.1850 to
0.1819 m, but increased path length 27.3823→27.5005 m and slowed the lap
5.739809→5.760825 s. It remained collision-free and passed the 0.20 m map
margin at every sample (minimum 0.381 m). Do not promote it for speed.

`r83` tested +0.050 m in S1 only. The line length rose to 27.3990 m and lap
time was 5.740754 s, 0.945 ms slower than `r79/r80`; path RMS improved to
0.1823 m and max error to 0.3829 m. S1 median/p95abs error improved to
+0.284/0.380 m from +0.324/0.407 m; collision count remained zero and map
clearance was unchanged (minimum 0.403 m, all samples above 0.20 m). This is a
small tracking-accuracy trade-off, not a racing-time improvement. The fastest
screen remains `r79/r80` at 5.739809 s. Artifacts are under `..._r81/` through
`..._r83/`.

The next geometry direction check `r84` applied -0.050 m in S1 only. It
shortened the generated path to 27.3676 m, but the Unity gate lap slowed to
5.745584 s; path RMS/max worsened to 0.1904/0.4339 m. There were no collisions
and every sample retained at least 0.20 m map margin (minimum 0.403 m). This
rejects the opposite S1 offset for both time and tracking error; keep the
zero-extra-S1-offset `r79/r80` candidate. Artifacts are under `..._r84/`.

`r85/r86` applied -0.050 m in S6 only, relative to the exact-repeat `r79/r80`
geometry. This shortened path by 0.0759 m and reproduced a 1.151 ms faster
physical lap (5.738659 s on both resets), with zero collisions and every
actual-position sample above the 0.20 m map margin (minimum 0.403 m, p05
0.430 m). The trade-off is worse overall path RMS, 0.1941 m versus 0.1850 m,
and S6 signed median/p95abs offset rises from +0.292/0.305 to +0.319/0.341 m.
Maximum path error stays 0.4079 m. Keep this as the fastest screen candidate,
but preserve `r79/r80` as the lower-error path; neither has production-MPC
validation. The next isolated probe is S6 target-speed scale 1.10 (from
1.085), with path geometry and other sector speeds unchanged. Artifacts are
under `..._r85/` and `..._r86/`.

Artifacts are preserved by run ID under
`live_runs/unity_source_raceline_feedback_mintime_20261014_r35/` through
`r86/`. The S6-only target-scale probe `r87` increased its multiplier from
1.085 to 1.10. It completed with zero collisions at 5.739859 s, effectively
neutral (+0.05 ms versus `r79/r80`) but with path RMS 0.1899 m; it is not
retained.

The S7-only target-scale probe `r88` raised its multiplier from 1.085 to
1.12, leaving geometry and S6 speed unchanged. It completed with zero
collisions in 5.731677 s, 8.133 ms quicker than `r79/r80`; minimum actual
map clearance was 0.361 m, p05 0.424 m, and every sample passed the 0.20 m
screen. S7 mean body speed rose 5.090→5.194 m/s while mean target rose
5.216→5.381 m/s, mean throttle 0.227→0.242, and mean absolute forward-slip
proxy 0.099→0.140. Path RMS rose 0.1850→0.1916 m; max remained within the
same 0.408 m. Fresh-reset `r89` reproduced all of these values exactly, so
retain 1.12 as the best observed S7 speed point, while noting the increasing
slip and tracking trade-off. Artifacts are under `..._r87/` through
`..._r89/`.

The next S7 point `r90` raised only target scale from 1.12 to 1.15. It
completed without collision but regressed to 5.740474 s, 8.798 ms slower than
the repeated 1.12 result. S7 mean target rose to 5.533 m/s, while body speed
fell 5.194→5.084 m/s; mean throttle rose 0.242→0.305 and mean absolute
forward-slip proxy more than doubled 0.140→0.326. This is direct evidence in
this tested region that the extra demand produces wheel spin rather than
forward motion. Reject 1.15 at full speed-feedback gain. The midpoint `r91`
at 1.135 produced 5.735329 s (3.653 ms slower than 1.12), S7 body speed
5.099 m/s, and slip proxy 0.274, confirming the useful edge is close to 1.12
and not a broad high-speed plateau. Next test the same 1.15 target with only
the S7 speed-feedback multiplier reduced to 0.50; this isolates whether
feedback throttle contributes to the observed spin-up. Do not extrapolate the
1.12 result beyond this band. Artifacts are under `..._r90/` and `..._r91/`.

At the same 1.15 target, `r92` changed only S7's speed-feedback multiplier
from 1.0 to 0.50 relative to `r90`. This cut S7 mean throttle 0.305→0.237 and
absolute forward-slip proxy 0.326→0.119, while body speed recovered
5.084→5.189 m/s. The physical lap was 5.732408 s: 0.731 ms slower than the
1.12/full-feedback best, with zero collisions and the same 0.361 m minimum
map clearance. This supports feedback-amplified wheel spin as the mechanism
behind the 1.15 regression; S7 can sustain the high target only when throttle
feedback is limited. `r93` raised only this S7 multiplier from 0.50 to 0.70
at the same 1.15 target. Body speed increased slightly to 5.223 m/s, but lap
time worsened to 5.735751 s, S7 slip rose to 0.153, path RMS to 0.1981 m,
and minimum map clearance fell to 0.316 m (still all samples above 0.20 m,
no collisions). The 0.50/0.70 interval therefore offers no improvement over
the faster, repeated 1.12/full-feedback point; retain 1.12 for racing. The
next useful work is to apply the diagnosed wheel-speed/throttle mismatch in
the development plant/controller rather than continue scalar target sweeps.
Artifacts are under `..._r92/` and `..._r93/`.

Artifacts are preserved by run ID under
`live_runs/unity_source_raceline_feedback_mintime_20261014_r35/` through
`r93/`. The old signed-throttle development path is reproducible from
`external/autodrive-unity-source/Assets/Scripts/OfflineCommandProfileRunner.cs`
and `tools/racing/build_unity_raceline_feedback_profile.py`; it remains
disabled by default.

### Sector-speed discontinuity and exact-source follow-up (r94–r116)

The early r95–r97 comparison is invalid as a racing result: those players were
built with the Explore scene. The scene has fewer colliders and does not
reproduce the practice racing course. The corrected r08 player was built from
the external Unity source using `BuildLinuxRacingDevelopment`. Its limiter-off
run r101 exactly reproduced r90's 5.740474 s physical gate lap, zero
collisions, 0.1849 m runner RMS, and 0.4031 m minimum map clearance. Use r08
for subsequent practice A/B runs; do not interpret the Explore-scene runs as
vehicle or controller evidence.

The failed high-demand screen r108 used S7 target scale 1.20, S7 speed-feedback
scale 0.50, and the positive-forward-slip throttle trim. It crossed the
physical gate in 6.138014 s, with zero collider contacts, but failed the map
margin screen: only 91.3% of samples retained 0.20 m clearance and path RMS/max
were 0.511/1.394 m. Tracking error first grew materially in S8 and became
largest in S12. Steering stayed below 0.56 normalized, so actuator saturation
does not explain the loss. The whole-S7 lower target probe r110 (1.17) was
worse at 6.387899 s and only 84.5% of samples met the map margin. A positive
drive-slip trim added only in S8 (`r109`) was a no-op: its reported trim and
run metrics matched r108. At the S8 command instants this positive-only gate
did not see above-threshold drive slip; the 1 kHz samples' mean signed forward
slip there was negative. Do not extend that limiter to the whole track on the
basis of absolute-slip magnitude alone.

The useful diagnosis was in the generated command profiles. Applying a
piecewise-constant speed multiplier per sector introduced an artificial target
step of 1.130 m/s at the S7/S8 boundary (about 6.822 to 5.692 m/s); another
boundary dropped about 0.88 m/s. The unscaled trajectory speed is smooth at
those same points—the discontinuities came from the sector multipliers, not
from the measured path geometry. `build_unity_raceline_feedback_profile.py`
now interpolates sector speed multipliers linearly between sector centers,
wrapping across the lap. This is the default profile-generation behavior and
only changes per-point target speed; geometry, steering gains, physics, and
production odom/MPC are untouched. The legacy step profiles remain preserved
as run artifacts. Math checks confirmed constant-scale invariance and exact
values at sector centers; profile comparison showed the maximum adjacent
target step fell from 1.130 to 0.0437 m/s.

The smoothed S7=1.20 candidate `r111/r112` completed at exactly 5.722500 s on
both fresh resets, with zero collisions, 0.18797 m path RMS, 0.40789 m maximum
error, 0.38079 m minimum map clearance, and every sample above the 0.20 m
margin. The 13,501-row numeric captures were identical. Relative to r108,
S8 path-error p95 fell 0.323 to 0.212 m, mean absolute forward-slip p95 fell
0.868 to 0.137, and mean absolute side slip fell 0.026 to 0.014. S7 signed
forward slip fell +0.183 to +0.110. This identifies the discontinuous target
schedule as a plausible cause of the previous transition loss and gives a
repeatable 9.2 ms improvement over the previous best one-lap screen r88/r89
(5.731677 s). It remains a development pure-pursuit result, not an MPC result.

The same smooth profile's S1 data showed mean signed forward slip +0.521,
while body speed averaged 5.277 m/s against a 6.176 m/s target. Enabling the
existing development-only positive-slip trim in S1 produced `r113/r114`:
both completed at exactly 5.721751 s, with identical 13,501-row telemetry,
zero collisions, and unchanged map margin. S1 throttle mean fell 0.445 to
0.375 and signed forward slip fell +0.521 to +0.430, while mean body speed
remained 5.277 m/s. The time change versus r111 is only 0.75 ms; retain this
as a repeatable wheel-spin reduction at essentially unchanged lap time, not a
meaningful time gain. This internal WheelCollider slip is available only to
the development runner; it is not a deployable competition input.

The all-sector r111 replay then identified S4 as a braking mismatch: body speed
averaged 7.097 m/s versus a 6.659 m/s target, throttle was zero, and mean
forward slip was -0.983 (strong wheel-speed deficit under full braking). The
single S4 target-scale change 1.08 to 1.15, retaining smoothing and the S1/S7
trim settings, gave `r115/r116`. Both fresh resets completed at exactly
5.712204 s, with zero collisions, 0.18749 m path RMS, 0.40789 m maximum
error, 0.38079 m minimum map clearance, and all samples above 0.20 m. The
gate-time gain is repeatable: 9.55 ms over r113/r114 and 19.47 ms over the old
r88/r89 screen. S4 mean body speed increased to 7.125 m/s, target to 6.973
m/s, throttle rose from zero to 0.008, and mean signed forward slip improved
to -0.959; braking slip remains severe. The common numeric telemetry prefix
for r115/r116 is bit-identical; captures differ by one final shutdown sample,
but gate and path metrics match exactly. Keep r115/r116 as the current fastest
repeatable one-lap development profile, not as a production raceline.

**Current Stage-2 status:** the source-backed profile generator now avoids
sector-induced speed-command steps, and the exact-source practice harness has
repeatably improved from 5.731677 s to 5.712204 s while preserving collision
and map-margin screens. This is real but modest progress; it does not close the
gap to the requested sub-5 s goal. The test changes a development command
profile and runs the source-built Unity racing scene; it does not update the
reference trajectory, production odometry, or MPC, and it is not the required
full 10-scored-lap result. Stage 2 remains open and Stage 3 remains unopened.
Next, investigate the remaining longitudinal/braking mismatch (especially
S4/S5) from these existing captures, then make one evidence-led speed-profile
change at a time and retain the same physical-gate, collision, and map-margin
checks. Do not infer a deployable slip controller from Unity-only telemetry.

Artifacts for the corrected-scene baseline, rejected probes, smoothed profile,
and reset repeats are under
`live_runs/unity_source_raceline_feedback_mintime_20261014_r99/` through
`r116/`; the shared r08 racing-scene player remains under
`live_runs/unity_source_raceline_feedback_player_20261014_r08/`.

### Braking and S8 speed-boundary probes (r117–r120)

The S5 scale increase from 1.04 to 1.15 (`r117`) did not produce a valid
trackable lap: the first >0.5 m path error occurred 2.297 s after the gate in
S6, S6 mean absolute side slip rose to 0.047, only 40.0% of samples met the
0.20 m map margin, and the eventual gate time was 10.5708 s. The 1.095 midpoint
(`r118`) also failed to physically finish the gate; its first >0.5 m error was
in S6 at 2.411 s, with 0.048 mean absolute side slip and 1.52 m runner path
RMS. Neither is accepted because Unity's route-loop completion flag alone is
not a valid lap.

`r119` kept S5 at 1.095 but reduced S6 scale from 1.085 to 1.0. It recovered
the physical lap and map margin (5.745391 s, zero collisions, RMS 0.1863 m,
all samples above the 0.20 m screen), but was 33.2 ms slower than r115/r116.
The extra S5 target therefore requires sacrificing S6 speed and is not a
faster setting in this paired test. Keep S5=1.04/S6=1.085 in the current
candidate.

The S8 scale increase from 1.0 to 1.05 (`r120`) remained collision-free and
inside the map margin, but slowed the gate lap from 5.712204 to 5.716282 s,
raised path RMS from 0.1875 to 0.2007 m, and reduced minimum map clearance
from 0.3808 to 0.2915 m. S8 body speed rose by about 0.17 m/s with still-low
forward slip, but the extra demand propagated into S9 and worsened path error.
Reject it for the speed objective; retain S8=1.0.

The current best remains the repeatable `r115/r116` profile at 5.712204 s.
These probes do not change the <4.9 s ten-lap goal status. Artifacts are under
`live_runs/unity_source_raceline_feedback_mintime_20261014_r117/` through
`r120/`.

### Sub-5 gap diagnosis from the current best (2026-10-14)

The r115 raw 1 kHz capture was independently re-scored by physical start/finish
gate crossings and projected onto its route sectors. It gives 5.712084 s
(consistent with the campaign's 5.712204 s within 0.12 ms). This remains
0.712 s over the 5 s benchmark; it is not a near-sub-5 result.

Measured sector times are S1–S12: 0.463, 0.333, 0.284, 0.345, 0.580, 0.704,
0.462, 0.454, 0.391, 0.453, 0.667, and 0.576 s. S6 and S11 alone consume
1.371 s; S5 and S12 add another 1.156 s. In S6, mean speed is 3.64 m/s,
mean absolute path curvature is about 0.71 1/m, and `u*|yaw_rate|` is about
9.4 m/s². S11 is about 3.60 m/s, 0.61 1/m, and 7.8 m/s². These are turn/grip
limited sections, not obvious throttle-starvation cases: S6 and S11 body speed
are already close to the command target on this run. Across the corresponding
path, the largest curvature is about 0.91 1/m (S6) and 0.87 1/m (S11).

The current r115 target-speed schedule integrates to approximately 5.253 s
under the optimistic pointwise `ds / target_speed` calculation, before any
tracking or actuator lag. This is not a hard physical lower bound, but it
shows that the current speed schedule itself is not a sub-5 plan. The recent
sector-scale micro-adjustments cannot close the remaining gap. The next
optimizer work must jointly change turn geometry and its speed plan, using
measured Unity behavior to constrain both; raising throttle or speed targets
alone is not supported by the S5/S6 failure evidence. Keep r115/r116 as the
reference candidate until a separate candidate beats it in the same exact
source-scene gate-time and clearance evaluation. Production trajectory,
runtime physics, odometry, and MPC remain unchanged.

### Post-gap follow-up experiments (r16/r18)

These two single-variable source-Unity tests were run from the repeatable
5.67138 s development candidate `r02`; neither is an accepted line.

- `r16` changed only the S3 speed-feedback multiplier from 0.15 to 0.45.
  S3 body speed did not increase (8.34 to 8.31 m/s), while mean rear forward
  slip rose from +0.015 to +0.155. The car first exceeded 0.50 m path error
  2.058 s into the lap in S6. Physical gate time was 9.009 s and only 56.1%
  of scored samples met the 0.20 m map-clearance screen. Reject: added S3
  throttle caused wheelspin without a speed gain and did not preserve the lap.
- `r18` changed only pure-pursuit lookahead from `0.35 + 0.12*v` m to
  `0.25 + 0.08*v` m. It completed a physical gate crossing but left the
  mapped course: gate time 8.010 s, path RMS 3.99 m, max error 10.96 m, and
  only 48.3% of samples met the 0.20 m clearance screen. Reject the shorter
  lookahead; its higher reactivity destabilized the generated-line tracking.

Both used the exact source-built practice player in the established Xvfb
batch-mode launcher. The source runner aborted on actual track collision;
these two misses were detected by physical gate timing and map/trajectory
metrics before any promotion. They did not change physics, the reference
trajectory, production odometry, or production MPC. Captures and profiles are
retained under `live_runs/unity_source_joint_geometry_curvature_20261014_r16/`
and `..._r18/`.

This closes two tempting shortcuts: raising throttle feedback in S3 and
shortening pure-pursuit lookahead do not close the 0.67 s gap. Existing
open-plane high-steering validation further shows why “8 m/s at 0.4 rad” is
not itself a racing-speed capability: in that joint regime the measured
lateral acceleration median/p95 is only about 6.66/6.96 m/s², whereas a
track curve requires `u² * |curvature|` to sustain its path. The next
geometry search must maximize turn radius within measured map clearance and
derive its speed schedule from the measured speed/steering response, rather
than treating commanded steering as achieved curvature or adding a generic
lateral-acceleration number. The best remains `r02` at 5.67138 s; no
sub-5-second candidate has been produced.

### Optimizer objective clarified: theoretical plant limit, not controller success

The user's objective is to use the known vehicle physics to find the fastest
physically feasible path/speed plan. Pure Pursuit or MPC convergence is a
separate runtime-integration question and must not be the feasibility oracle
for this theoretical search. A candidate is physically disqualified by the
Unity plant only for actual physical/course constraints (for example collision
or leaving the mapped course), not because a particular controller rejects
its rollout. Keep controller tracking metrics in a separate ledger.

The exact-source Unity player is itself a useful offline plant: it runs the
project's vehicle and native WheelCollider physics at 1 kHz and can evaluate
candidate steering/throttle sequences while recording body motion and all four
wheels' forward/sideways slip. This is materially different from the separate
Python 3-D mechanics port, which remains too inaccurate for trusted lap-time
ranking. Consequently, the theoretical optimizer need not wait for a learned
model to become perfect before using the exact-source player as its black-box
physics oracle.

There is already a bounded example of that use. The source-Unity sector search
`live_runs/unity_source_sector_raceline_20261014_r01/campaign.json` evaluated
63 candidates using six throttle and six steering-scale parameters. Its
collision-free, map-clear candidate improved exact Unity gate time from
6.140702 s to 5.946760 s (0.193942 s); best minimum map margin was 0.335 m.
This is a real source-plant optimization gain, but it is not yet a free
geometry/speed optimum: the decision basis only scales a captured command
profile, and it did not optimize a new path shape or use wheel-slip/lateral
motion as explicit constraints. It therefore demonstrates *some* value from
the exact simulator, not completion of the user's intended optimizer.

Do not use the later 5.712204 s Pure-Pursuit feedback screen as evidence that
the theoretical optimizer improved: it is a controller-dependent result and
belongs only in the runtime-pairing ledger. Conversely, do not reject a
theoretical candidate just because that screen's controller cannot track it.

The required next optimizer implementation is a direct-source, low-dimensional
trajectory search: decision variables must change path geometry and speed/
longitudinal command together; each candidate is propagated by the exact
Unity plant; the objective is physical gate-to-gate time; and the evaluation
reports the resulting body lateral velocity/sideslip and per-wheel slip
against the configured Unity tire curves, plus course clearance and collisions.
Do not substitute command-following RMS or MPC convergence for this objective.
Use Bayesian optimization over a smooth, finite-dimensional path/control basis
when treating Unity as a black box; IPOPT cannot obtain valid gradients through
Unity's native WheelCollider solver. A differentiable vehicle surrogate may
still seed/refine a candidate, but exact-source Unity remains the decisive
physics evaluation. No runtime MPC, production trajectory, or simulation
physics changes are authorized by this clarification.

### Controlled-slip feasibility diagnosis (2026-10-14)

The source Unity plant can represent and predict a slide because it runs the
same Rigidbody/WheelCollider solver and records per-wheel slip at 1 kHz. This
does not mean the current production odometry or MPC can observe or predict
that state accurately. In particular, `WheelHit.forwardSlip` and
`WheelHit.sidewaysSlip` are development-only teacher signals, not competition
inputs.

I re-analyzed the existing exact-source `r115` scored lap rather than launching
another parameter sweep. It is the repeated 5.712204 s generated-profile
result documented above; the physical lap has zero collisions and at least
0.3808 m map-derived vehicle-center clearance. Its recorded body sideslip is
not uniformly small: whole-lap p95 is 0.111 rad and peak is about 0.117 rad.
However, those large sideslip samples are confined to slow, tight turns. For
samples with body speed at least 6 m/s, sideslip p95 is 0.0234 rad and
sideways tire-slip p95 is 0.00646; at least 7 m/s those fall to 0.0128 rad
and 0.00489. The sideways-slip tire curve peaks at 0.01 and reaches its
asymptote at 0.10, so the fast portions are below the curve peak and nowhere
near its asymptote. Samples with body sideslip at least 0.10 rad occur around
3.7 m/s, not in the fast regime. Therefore this lap demonstrates that the
vehicle can complete the course with modest low-speed sideslip; it does not
demonstrate controlled high-speed drifting.

The already-preserved higher-speed-entry stress attempts `r117` and `r118`
provide a useful partial answer, even though neither is a valid lap. In `r117`,
the maximum per-wheel sideways slip first crossed the tire-curve asymptote
(0.10) 2.070 s after the gate, at 3.42 m/s, body sideslip 0.088 rad, and
0.237 m distance to the candidate path. Path error passed 0.50 m 227 ms later;
at that point tire sideways slip was 0.159 and body sideslip had fallen to
0.055 rad. In `r118`, the same slip crossing occurred at 2.251 s, 3.56 m/s,
0.079 rad body sideslip, and 0.332 m path error; path error passed 0.50 m
160 ms later, with tire slip 0.143 and body sideslip 0.053 rad. The successful
`r115` reference never crossed 0.10 sideways tire slip (its maximum was
0.046). Thus wheel lateral slip exposed a saturation signature roughly
6–9 control periods before substantial path divergence, while body sideslip
alone was already decreasing. This is promising as a teacher-side early
warning, but it is not causal proof, does not show that the present controller
could recover, and occurs at only 3.4–3.8 m/s—not in a high-speed drift regime.
An additional sensor-relevant precursor appears in the same windows: using
measured front-wheel steering and the source prefab's 0.324 m wheelbase, the
simple kinematic yaw reference is about 0.2–0.4 rad/s above measured yaw in
the successful `r115` turn (p95 deficit 0.41 rad/s). In the `r117` turn, the
deficit first exceeded 1 rad/s on the 40 Hz sample grid about 250 ms before
path error passed 0.50 m; in `r118` it did so about 225 ms before. In the
2.00–2.25 s window, yaw-deficit p95 was 2.03 rad/s in `r117`, 1.20 rad/s in
`r118`, versus 0.41 rad/s in `r115`. This is a promising observable under-
response signature from yaw rate, speed, and steering, but the calculation
uses Unity's measured wheel angles and body speed. A deployable detector must
recompute it from allowed inputs and account for steering actuator lag; these
failed Pure-Pursuit stress laps do not prove the candidate trajectory itself
was physically impossible.

As an offline diagnostic, the simple rule `kinematic yaw deficit > 1 rad/s`
was checked at 40 Hz against front tire slip `> 0.10`, using only scored
samples before the path-error threshold. It flagged all 13/13 saturated
samples in `r117` (6 false positives; precision 68.4%) and all 22/22 in
`r118` (7 false positives; precision 75.9%); the successful `r115` window had
no saturated samples and no flags in 230 samples. The first flag preceded
front slip crossing 0.10 by 25 ms in `r117` and 75 ms in `r118`, and preceded
0.50 m path error by 250/225 ms. This is a promising candidate early-warning
feature, not a validated production detector: its inputs still include
Unity-truth forward speed and actual wheel angle, and there are only two
stress captures.

The axle split identifies which tire group lost authority. In the same
2.00–2.25 s window, front-wheel sideways-slip p95 was 0.153 in `r117` and
0.090 in `r118`, while rear-wheel p95 stayed at 0.005 and 0.006. The stable
`r115` turn had front/rear p95 0.032/0.007. At the first asymptote crossing in
`r117`, the front-right tire was at 0.100, front-left at 0.073, and rear tires
at 0.003–0.004. This is front-axle saturation/understeer, not
power-on rear drift. Extra throttle or rear-wheel-spin control cannot restore
front steering authority; the relevant variables are corner-entry speed,
path curvature, and steering demand.

Longitudinal wheelspin is a separate, real effect. The profile's development
throttle trim uses Unity's measured positive forward slip, targeting 0.24
(near the forward curve's 0.25 asymptote) with gain 0.4 in selected sectors.
The earlier matched `r111` → `r113/r114` comparison reduced S1 mean signed
forward slip from +0.521 to +0.430 and throttle from 0.445 to 0.375, while S1
body speed stayed 5.277 m/s and lap time changed by only 0.75 ms. That is a
repeatable spin reduction, not a meaningful speed improvement, and the direct
Unity slip signal is not deployable. It should not be presented as proof that
slip feedback has closed the lap-time gap.

To prevent future conflation of fast wheelspin with fast lateral sliding,
`optimize_unity_source_raceline.py` now reports separate sideslip, sideways
tire-slip by front/rear axle, driven-wheel slip, and kinematic
yaw-under-response metrics; speed-conditioned fields at or above 6 and 7 m/s
are also carried into the sector campaign summary. The calculations were
checked against preserved `r115`, `r117`, and `r118` captures and reproduce
the values above. Python syntax and capture-based assertions pass. This is an
analysis instrumentation improvement only; it does not alter a candidate,
plant, production odometry, or MPC.

The meaningful next effort is not another global throttle/slip threshold
adjustment. The `r117/r118` captures give low-speed lateral-saturation evidence,
but no current capture establishes controlled high-speed lateral sliding or a
sensor-only estimate of the 160–227 ms warning. For the optimizer, use the
exact Unity plant to search path geometry and speed/steering commands jointly;
keep front-axle lateral slip and yaw under-response visible rather than hiding
them in one four-wheel slip score. This points the search at reducing turn
curvature/entry demand, not lowering a global throttle number. If a candidate
needs a high-speed drift regime not represented by those runs, run one
targeted practice-track turn-in/hold/unwind capture for that unsupported
state, with collision abort and telemetry retained. Do not substitute the
Unity-only `WheelHit` feedback as a production controller input. Odom/MPC
changes remain outside this step.

## Exact-Unity open-loop control search (2026-10-14)

To test whether the plant can exploit the existing vehicle's slip behavior
without relying on the prior raceline-feedback controller, I extracted the
successful `r115` lap's 40 Hz steering/throttle history from the 1 kHz Unity
capture. A new 8-parameter search smoothly scales steering and throttle at
four periodic route-progress knots; Unity WheelCollider physics scores the
result. The route geometry and physics are unchanged. Position is used only
to advance through the command schedule. The search records wheel/body slip
and requires a collision-free lap plus at least 0.20 m map-derived clearance.
This tests control timing/magnitude on the current path; it does not yet search
new path geometry or constitute a full offline plant model.

The spatial-profile player was verified before the search: the unit-scale seed
replayed in 5.712084 s, with zero collisions, full route progress, 8.6052 m/s
peak speed, 0.3808 m minimum map clearance, and exactly 16,775/16,775 requested
fixed steps. This exactly reproduces the previous open-loop seed lap time.
Every completed candidate is now checked against its requested Unity fixed-step
horizon; a mismatch terminates the search. Seven focused unit tests pass.

The first search attempt used an older player that stopped after 13,500 steps
despite a 16,775-step request. Its candidate lap results are invalid and the
campaign is explicitly marked `invalid_harness_horizon`; its files are retained
for diagnosis, not treated as vehicle evidence. A subsequent one-off launcher
also generated malformed JSON and Unity rejected it before any physics step;
that setup attempt had zero commands, zero steps, and zero collisions and is
not a vehicle result.

The corrected 24-evaluation campaign completed with the dedicated
spatial-optimizer player. Five candidates passed the map-margin check; none
beat the verified 5.7121 s seed. The search found no safe, faster point in its
four-knot gain-only parameterization. The apparent 5.01–5.16 s crossings all
left the map and are not valid lap times. A feasibility-aware acquisition fit
to these results did not identify a promising safe/faster candidate (maximum
predicted feasibility probability 0.214), so that campaign was closed rather
than extending the same parameter sweep.

- Uniform 1.10 throttle gain completed a 5.0713 s gate-to-gate lap with no
  Unity collision, but left the mapped track; minimum map clearance was 0 m
  and p95 absolute sideways tire slip rose to 0.106. It is rejected.
- Uniform 1.20 throttle gain also left the map (0 m clearance), with a 5.1138 s
  crossing and p95 sideways tire slip 0.112; rejected.
- Uniform 0.90 throttle gain was slower (6.5834 s) and also failed map
  clearance; it did not improve the baseline.
- 0.95 steering scale completed a valid 5.8338 s lap with 0.3606 m minimum
  clearance and no collision, but is slower than the 5.7121 s seed.

The apparent 5.07 s result is therefore not a valid faster lap: telemetry
places the vehicle at the map's extreme edge for 2,149 of 5,072 scored samples.
This is evidence that simply increasing throttle can create an off-track
shortcut / loss of path control; it is not evidence of a usable controlled
drift. No production odometry, MPC, reference trajectory, or Unity physics has
been changed. The best valid lap remains 5.7121 s and the sub-5 s goal is
unmet.

### Follow-up: steering phase and localized exit throttle

The failed 1.10-throttle capture was compared with the verified seed on the
same route-progress grid. Around route progress 0.40–0.46, the candidate is
about 0.6 m/s faster (median 4.36 versus 3.80 m/s), front-wheel sideways-slip
p95 rises to 0.10–0.15, rear-wheel sideways-slip stays near 0.003–0.004, and
yaw rate falls from about 2.56 to 1.15 rad/s. Thus this specific failure is
front-axle lateral authority loss/understeer, not rear power-on wheelspin.
The profile is already unwinding steering there. A local steering hold
(1.10 throttle globally, steering sampled 0.012 lap-progress earlier over
0.30–0.50) still left the mapped course: 5.1691 s gate crossing, zero
minimum map clearance, only 54.4% of samples above the 0.20 m margin, and
front/rear sideways-slip p95 0.124/0.007. Holding the turn command longer is
not sufficient to recover the high-throttle case.

A second exact-source probe applied 15% more throttle only over route progress
0.57–0.70, after the first tight turn. It kept the earlier high-curvature
sector within the map, and whole-run front/rear sideways-slip p95 was lower
(0.054/0.009), but the added speed carried into the following turns: the car
reached only 98.1% ordered route progress, map clearance fell to zero in
sectors 8–12, and it did not complete a valid lap. This is not an accepted
speed gain. Both probes completed all 16,775 requested 1 ms Unity steps with
zero collisions; each is retained as useful failure telemetry, not as a lap
result.

These targeted outcomes reject two simple fixes: delaying steering unwind
does not restore front-axle authority at the tested speed, and adding drive
after one turn without jointly planning the next braking/curvature demand only
moves the departure to the next turn. The actionable signal remains the
pre-divergence yaw under-response already seen in the independent r117/r118
stress captures. The next implementation should make steering geometry,
turn-entry speed, and longitudinal planning a coupled exact-Unity search;
the speed/sideslip state and front/rear tire slip must be reported separately.
Do not promote a slip rule or a candidate based on a gate crossing alone.
Production MPC, odometry, reference trajectory, and Unity physics remain
unchanged.

### Slip-control experiment: yaw under-response and torque unloading

The invalid uniform-1.10-throttle capture was used as a stress profile to test
whether slide onset can be detected and corrected at the actual 40 Hz command
rate. The measured trigger was the positive yaw deficit
`max(0, |u*tan(delta_front)/0.324| - sign(expected_yaw)*r)`. In the unguarded
capture, this deficit rose above 0.6 rad/s roughly 50–75 ms before front
WheelCollider sideways slip crossed 0.10 in the first high-demand turn. This is
a useful early-warning correlation in this profile, not a universal slip
model.

Three source-player tests separate the policy effects:

- The first guard build recomputed state feedback every 1 ms. It is not an
  available controller rate and is not an eligible 40 Hz test. It made 1,169
  brake updates, completed the requested physics horizon without collision,
  but only reached 0.614 ordered route progress; its overall front/rear
  sideways-slip p95 was 0.051/0.259. Keep it only as a harness diagnostic.
- Correcting the guard to evaluate once per 25 ms yielded 17 brake updates.
  In the first understeer event, median yaw rate rose 1.44→2.64 rad/s and front
  lateral-slip p95 fell 0.143→0.068. Later, the brake action created severe
  combined/rear slip and the car stalled near 0.63 route progress. Reject
  braking as the response to this yaw-deficit signal.
- Replacing brake with bounded reduction of positive drive torque (threshold
  0.60 rad/s, gain 0.50 command fraction per rad/s, maximum 75% reduction) gave
  49 torque-unload updates. It reduced first-turn front-slip p95 to 0.098 and
  delayed the first <0.20 m map-margin sample by 177 ms (route progress
  0.426→0.441), but did not complete a valid lap: ordered progress reached
  0.983, map clearance still fell to zero, and less than half of attempted-lap
  samples retained the required margin. This is a localized improvement, not
  a usable slip controller.

The same 40 Hz torque-unload policy on the already validated feedback path,
with target speed raised 10%, completed a path-index wrap without collision
but was not a valid lap: gate time 6.866 s, path-error RMS 1.322 m, and only
49.7% of samples had the 0.20 m map margin. Therefore a yaw trigger plus
throttle cut does not by itself make an over-demanding speed profile feasible.

The next test changed one deeper assumption: under sideslip, body-forward speed
and body heading are not the car's ground-course speed/direction. An opt-in
source-runner mode used planar ground-velocity magnitude for speed feedback and
lookahead, and computed path curvature in the ground-course frame. On the same
+10% target and torque-unload profile, this reduced gate time 6.866→6.772 s,
path-error RMS 1.322→1.260 m, and raised the map-margin fraction 49.7→50.4%.
Front sideways-slip p95 remained 0.152, the minimum clearance stayed zero, and
sectors 6–9 remained off-map. This is a small directionally positive change,
not a valid lap or a sufficient fix.

The next oracle experiment uses front WheelHit lateral slip normalized by each
front WheelCollider's live asymptote to unload positive drive torque at the
40 Hz command boundary. It tests whether direct slip feedback can recover the
same fast path; Unity-only slip remains strictly development-side and will not
be integrated into the competition controller. If this oracle cannot recover
the line, the evidence points to jointly replanning path geometry/speed rather
than further tuning the yaw trigger. Production odometry/MPC and the reference
trajectory remain untouched. Reproducible captures and builds are in
`live_runs/unity_source_slip_guard_test_20261014/`,
`live_runs/unity_source_slip_guard_test_20261014_r02/`,
`live_runs/unity_source_slip_guard_test_20261014_r03/`,
`live_runs/unity_source_yaw_unload_feedback_20261014_r01/`, and
`live_runs/unity_source_yaw_unload_course_feedback_20261014_r01/`.

### Free-geometry IPOPT prototype: status and root cause (2026-10-14)

The previous sector-speed/multiplier experiments changed commands on an
existing path; they did **not** generate a new raceline. A separate prototype
was then written with one free lateral path offset per practice-centerline node
and jointly optimized body motion, steering, and drive torque. Its geometry
comes from the map centerline, not from the production/reference raceline.
However, it has produced **no feasible candidate** and must not be described as
an achieved raceline or as progress in lap time. Three solves ended at
`Maximum_Iterations_Exceeded`; their reported objective values are infeasible
iterates and are not lap-time estimates.

The prototype also has a real bottom-up defect: exact-zero CAWB braking nodes
were copied from the old lap and held fixed. Thus its geometry is free, but its
hybrid brake schedule is not. Unity `VehicleController.Drive()` confirms the
discontinuity: exactly zero requests full CAWB brake torque, while nonzero
throttle selects motor torque. The next optimizer formulation must make this
mode a decision (or explicitly optimize brake-event locations); it must not
inherit the old schedule as a constraint. Wheel-spin was changed to a
command-hold algebraic balance because the fitted 1.75 ms time constant cannot
be resolved on the prototype's 120 mm, roughly 20–30 ms spatial cells. That is
an approximation, not a 1:1 wheel-spin rollout.

Existing Unity 1 kHz captures were used to test the force-to-yaw hypothesis,
without launching another simulator. The frozen shared tire fit trained on
r07 gives a practice-r02 contact-yaw-torque RMSE of 0.532 Nm (p95 absolute
1.119 Nm). An axle-specific force/moment fit reduced yaw-torque RMSE on r11
from 0.136 to 0.087 Nm, but worsened the unseen practice-r02 result from 0.272
to 0.333 Nm. It therefore fails practice transfer and is **not promoted**.
This confirms the optimizer's force allocation is not yet adequate for a
credible free-path solve; simply increasing IPOPT iterations is not justified.

One suspected source mismatch was ruled out empirically. Although an inactive
Unity scene object serializes different Ackermann dimensions, the active
captured front-wheel angles invert to a common center angle using the
optimizer's 0.324 m wheelbase and 0.236 m track width (less than 3e-6 degrees
left/right disagreement on r07, r11, and practice r02). Do not change the
Ackermann geometry based on that inactive object.

Current decision: keep production MPC, odometry, reference trajectory, and
Unity physics unchanged. Do not send this failed IPOPT candidate to the track.
The next optimizer work must (1) remove the reference-derived brake mask,
(2) use a validated practice-transferring yaw/force model or the exact Unity
plant as its evaluator, and (3) emit a genuinely new path before track
validation. Available disk at this checkpoint is only about 14 GB; no Unity
simulation was started during this analysis.

### Free-geometry IPOPT follow-up (2026-10-15)

The criticism that multiplier adjustments are not new path generation is
correct. The current optimizer is a separate free-geometry OCP: every spatial
node has its own lateral offset, and path, speed, body state, steering, and
throttle are optimized jointly. It neither loads nor scales the reference
raceline geometry. It is not yet a successful optimizer result: no feasible
trajectory has been emitted, no candidate has been replayed in Unity, and
there is no new lap time to report.

The first two IPOPT formulations failed and exposed a specific modeling error.
The r08 attempt imposed instantaneous per-wheel torque equilibrium despite
loading a fitted wheel inertia; it ended at 8,000 iterations with 6.539 maximum
constraint violation. That equation suppresses the torque-driven wheel-spin
transient, so it cannot represent throttle-induced slip. It has been removed.
The r10 formulation now integrates each drive-mode wheel angular speed using
the source CAWD motor-torque split, tire reaction torque, fitted wheel inertia
(0.00043659 kg m²), and wheel damping. In exact-zero CAWB mode it retains the
measured locked-wheel state. The number and order of brake/drive phases are
still seeded from one captured lap, but IPOPT optimizes their lengths and
therefore their switch positions; this remaining hybrid-mode limitation is
explicit and is not equivalent to a fully free mode sequence.

The r10 solve reached 8,000 iterations with objective 5.27766 (an infeasible
NLP iterate, **not** a predicted or measured lap time), maximum constraint
violation 0.37453, and dual infeasibility 398.38. Largest residuals were
lateral-force balance at node 191 (0.375 m/s²) and lateral-velocity spatial
collocation (0.318 m/s), followed by much smaller yaw/force residuals. Thus the
wheel-spin state correction is implemented, but the free-path solve still
fails its feasibility gate. The lower residual versus r08 is not evidence of
a usable trajectory or faster driving. Diagnostic outputs are in
`live_runs/raceline_candidates/unity_free_path_20261015_r08/` and
`live_runs/raceline_candidates/unity_free_path_20261015_r10/`; the initial
wheel-dynamics residual report is in `..._r09_initial_diag/`.

No simulator was launched for these optimizer attempts. The active/reference
raceline and runtime vehicle physics remain unchanged. Next work is to isolate
the r10 residual around the identified spatial nodes and correct the cause in
the collocation/model formulation, then solve again. Only a constraint-feasible
free path proceeds to the established batch-mode Unity replay; only that replay
can establish trackability or lap-time gain. Although the Unity C# controller,
serialized vehicle parameters, configured friction curves, and capture-derived
rigid-body equations are used, the native WheelCollider/PhysX tire solver is
closed and the tire force response remains an identified approximation. This
optimizer is therefore not yet an exact 1:1 Unity plant.

#### Follow-up solver diagnosis (2026-10-15)

r13 reproducibly found a constraint-feasible free-geometry initializer
(maximum residual 2.29e-12). Its predicted 5.642 s is only 0.034 s below the
5.676 s source-capture lap and comes from minimizing distance to the captured
motion, not from minimum-time optimization. The minimum-time stage began from
that verified point (warm-start residual 2.29e-12), then the exact-Hessian
solve ended at 8,000 iterations with 22.50 constraint violation and objective
5.692 s. It was rejected.

A unit-scaling experiment (r15) was also rejected: feasibility ended at 0.135
instead of 2.29e-12. The r16 minimum-time stage used IPOPT's limited-memory
Hessian for the piecewise tire-force model. It reduced final violation to
0.0166 after 8,000 iterations, but still missed the 5e-4 feasibility gate;
its infeasible objective was 7.019 s. The fallback written to r16 is the
feasible initializer and is explicitly labeled
`feasible_candidate_not_time_optimized`. It is not evidence of a faster
raceline. No Unity replay was run because neither minimum-time stage emitted a
feasible optimized path.

The runner now preserves the stage-two objective, primal/dual residuals, and
worst constraint labels for the next targeted diagnosis. The duplicate output
directory creation that crashed after reaching candidate emission is fixed.
The current answer to “new path or multipliers?” is: the feasible fallback is
free geometry at every station, not a multiplier-adjusted reference, but it is
not time optimized and shows no meaningful predicted gain. The number/order
of brake/drive phases is still seeded from the captured lap while phase lengths
are free; the native WheelCollider solver is still closed. Production MPC,
odometry, reference trajectory, and simulator physics remain unchanged.

#### Stiff wheel-spin and solver continuation follow-up (2026-10-15)

The fitted wheel damping and effective inertia give a wheel-spin time constant
of 1.75 ms, while the current 0.12 m spatial mesh spans roughly 13–30 ms at
racing speed. The old trapezoidal wheel-spin update is not L-stable at that
step-to-time-constant ratio. A backward-Euler replacement was tested and
rejected: its feasibility solve timed out after 2,621 iterations with maximum
constraint violation 118.66. This is retained as a failed numerical variant,
not promoted.

The next discretization analytically integrates the linear wheel torque /
damping response over each spatial interval, holding the tire reaction at the
interval endpoint. It is stable for the stiff mode but still is not a 1 kHz
PhysX rollout. The initializer also now sets wheel speed to zero at both sides
of each seeded CAWB/CAWD switch, matching the optimizer's own transition
constraints. That reduced the initial maximum constraint residual from 17.58
to 2.61; after the feasibility solve, r18 reached 2.68e-6.

r18's pure-time stage still failed: after 3,119 iterations and its 600 s CPU
limit, maximum violation was 0.6655. The dominant residuals were lateral force
balance (0.6655 m/s² at node 97) and heading-error spatial collocation
(0.5206 rad), followed by lateral-force residuals at nodes 134, 213, and 191.
Its 7.144 s objective was an infeasible iterate and is not a lap-time estimate.
The retained feasible point was the captured-motion initializer at 5.6458 s;
that is not a time-optimized raceline and is not a validated improvement over
the 5.676 s source lap. No simulator was run.

r19 tested a three-level objective continuation (0.20 s, 0.05 s, then pure
time) from the r18 feasible initializer. The first 0.20 s stage reached an
infeasible 5.528 s iterate, with maximum violation 2.345 m/s² at lateral-force
balance node 159; it also violated lateral-force balance at nodes 213 and
160. The later continuation stages were correctly not attempted. The selected
result remained the 5.646 s captured-motion feasibility point, not a faster
raceline. This rejects a soft objective penalty as a sufficient safeguard.

Next, use an explicit trust region on the squared deviation from the feasible
captured-motion solution while minimizing lap time. Increase the trust radius
only after a stage returns a fully feasible iterate, retaining the best
feasible time at every step. This is a solver globalization measure, not a
vehicle limit or a claim that the measured path is optimal. Unity runtime and
reference trajectory remain unchanged. The WheelFrictionCurve's documented
two-piece spline structure matches the implemented curve shape, but its
slopes/scales remain fitted and the native WheelCollider/PhysX contact solver
is unavailable to the optimizer
(https://docs.unity3d.com/6000.0/Documentation/ScriptReference/WheelFrictionCurve.html).

#### Tire-force interpretation and trust-region attempt r20 (2026-10-15)

The lateral-force-balance constraint is Newton's second law for the rigid body,
not a Unity-side lateral-force command or a constraint that each tire must sit
at its asymptote. Unity documents `WheelFrictionCurve` as a slip-to-force
relationship with a peak at the extremum and a lower sliding force approaching
the asymptote; forward and sideways curves are evaluated separately. In this
OCP, each wheel's fitted slip curve produces `Fy_i`, and the equality
`sum(Fy_i) = mass * (a_y + drag*v)` makes the predicted body acceleration
consistent with those modeled tire forces. The asymptote limits the per-wheel
curve output once slip exceeds its asymptote point; it is not itself the
vehicle's lateral-acceleration equation.

That equality is only as trustworthy as the force surrogate feeding it. The
native WheelCollider/PhysX contact calculation is closed. This implementation
uses a fitted curve scale, estimated quasi-static normal loads, simplified
slip coordinates and independent longitudinal/lateral curves, so a force
balance residual is evidence of model inconsistency, not proof Unity could not
make that motion. In particular, force balance must not be presented as a
validated hard vehicle limit until force reconstruction transfers on held-out
whole runs.

r20 applied the explicit trust-region continuation, but the required initial
feasibility solve did not pass: after 2,631 IPOPT iterations and 231.5 s wall
time, maximum constraint violation was 0.4758 (feasibility gate: 0.0005), led
by lateral-force balance at node 127 (0.4758 m/s^2), lateral-velocity spatial
collocation (0.3627 m/s at node 356), and longitudinal force balance at node
122 (0.2489 m/s^2). The initial guess had 2.6095 rad/s wheel-spin defects at
nodes 59/64. IPOPT reported `Maximum_CpuTime_Exceeded`; no feasible incumbent
was produced, so the trust-region time-minimization stages were never entered
and no raceline candidate was generated. This is especially notable because
r18/r19 reached 2.68e-6 feasibility from the same seed, with final deviation
objective 19.4153. A plausible solver regression is the new phase-I cap of
1e9, which does not constrain the useful search region. The next targeted
attempt caps phase-I deviation at 25 (above the recorded feasible value), then
uses the existing increasing trust-region schedule only after feasibility.
This is numerical globalization, not a tire/vehicle limit. The 5.6458 s
captured-motion fallback remains unchanged and is not a minimum-time result.
No Unity replay was run.

r21 tested that phase-I cap of 25. It reached `Infeasible_Problem_Detected`
after 197 iterations with maximum residual 0.1509; the cap was active and the
largest defects were Frenet offset/heading collocation. The cap did not provide
a reliable phase-I route from the raw capture seed and is not retained as a
general fix.

r22 instead loaded the already-feasible r18 point, reconstructed each wheel
speed from its recorded model slip and state, and initialized the OCP at r18's
optimized drive/brake phase lengths. A separate initial-constraint diagnostic
confirmed maximum residual 2.77e-6. The full solve then found a feasible
captured-motion point in 23 iterations (5.06e-9 maximum residual), but the
first time stage with a 1.01 trust-budget factor diverged: its nominal 4.867 s
objective was infeasible by 84.99, dominated by the trust-region constraint
and lateral-force residuals. It was rejected. The saved fallback predicts
5.6464 s, effectively the same as r18 and not a time-optimized improvement;
minimum side-clearance slack is only 4.7 mm beyond the configured 0.15 m
buffer. No Unity replay was run.

The runner and solver now support reusing a prior model-consistent OCP state
and phase mesh, and can skip phase-I optimization when that supplied point is
already feasible. The first trust-region expansion was widened from 1.01 to
2.0 because r22's allowed normalized displacement was only about 0.026 and the
solver diverged far beyond it.

r23 confirmed that this alone does not solve the globalization problem. Its
feasible captured-motion seed had a 2.77e-6 maximum constraint residual. The
first time stage ran 1,126 iterations and stopped at the CPU limit with a
14.00 maximum violation (the trust-region bound and lateral-force balance were
dominant). The reported 5.589 s objective was infeasible and rejected. The
saved fallback is 5.6458 s, not a time-optimized result and not evidence of a
faster trajectory. No Unity replay was run.

The free-geometry formulation does replan the continuous path: `ey` is a
decision at every optimizer station, and curvature is recomputed from the
resulting XY path. It jointly optimizes that geometry with vehicle states,
steering, throttle, and the lengths/locations of the seeded drive/brake phases.
The discrete count/order of drive/brake phases is still inherited from the
capture, so the control-mode topology is not globally searched. r23 did not
find a new feasible minimum-time geometry; its output is the prior feasible
seed.

#### Wall-clearance audit and r24 (2026-10-15)

The vehicle envelope is currently approximated as a 0.510 m by 0.273 m
rectangle, projected onto the track normal using heading error. Its aligned
half-width is 0.1365 m; the code floors that at 0.150 m, adding only 13.5 mm
to the physical half-width when aligned. The larger planning restriction is a
separate 0.150 m wall buffer, inherited from the controller-oriented
trajectory pipeline. Thus the aligned center-to-wall requirement is 0.300 m;
at nonzero heading it is 0.150 m plus the projected rectangle half-width.
This extra 15 cm is not vehicle size and is not needed to represent ideal
tracking. The practice map resolution is 0.050 m, so the research candidate
runner now exposes the buffer and defaults to one map cell (0.050 m), while
the shared production configuration and old/reference trajectory remain
unchanged. The physical-footprint estimate is retained pending collision
validation against the Unity vehicle collider; reducing the extra buffer does
not authorize touching the wall.

r24 tested the free-geometry solver from r23's feasible seed with the
candidate-only 0.050 m buffer. The minimum-time stage again failed its
feasibility gate: after 1,051 iterations, maximum constraint violation was
10.883, dominated by the captured-motion trust region (10.883) and yaw-rate
spatial collocation (1.493 rad/s). Its 5.614 s iterate was rejected. The saved
fallback is exactly the same feasible seed geometry and 5.6458 s estimate as
r23; its reported slack rises by exactly 0.100 m, as expected from reducing
the buffer by 0.100 m. Therefore this clearance change alone did not create a
faster path. It only makes more corridor available for the next optimizer
revision. No simulator was run and no trajectory was promoted.

The capture/source audit also confirms a material model boundary: the optimizer
uses fitted tire forces with good held-out one-step force reconstruction
(r11 vector RMSE about 1.88 N against 27.06 N target RMS), but still uses
quasi-static load transfer. The existing 1 kHz suspension analysis predicts
per-wheel `WheelHit.force` from body-frame wheel-center travel and rate using
the serialized 500 N/m spring and 100 Ns/m damper with about 0.40 N RMSE per
wheel on independent r11. That state-level result is not yet a recursive
vehicle-model validation, but it identifies a source-grounded load model to
replace or test against the OCP's current quasi-static approximation. Native
WheelCollider force/substep behavior remains unavailable in source and must be
validated against the Unity player. The 5 cm margin is retained as a
candidate-only one-map-cell guard; production settings and runtime physics
remain unchanged.

#### Quasi-static load-transfer error audit (2026-10-15)

Added `tools/vehicle_dynamics_learning/compare_optimizer_load_transfer.py` and
ran it on the existing 1 kHz r07 training capture and independent r11 capture.
It differentiates the captured body velocity after rotating it into world
coordinates, uses simulator truth only as a diagnostic acceleration input,
then compares the optimizer's exact current load equation against each
grounded wheel's `WheelHit.force`. Nothing was fit or promoted.

The current per-wheel load model is materially incomplete. At the 21 ms
comparison filter (results are stable across 11, 21, and 41 ms):

| Capture | Optimizer quasi-static RMSE | Static-only RMSE | Model RMS transfer from static | Unity RMS load change from static |
| --- | ---: | ---: | ---: | ---: |
| r07 training | 2.255 N | 3.937 N | 1.784 N | 3.937 N |
| r11 held-out | 2.115 N | 3.653 N | 1.640 N | 3.653 N |

So Newtonian transfer helps by about 43% against static load, but predicts
under half the observed per-wheel load variation. On held-out r11, the
per-wheel RMSE is 2.02–2.15 N; at `|ay| >= 8 m/s²` it rises to 3.375 N
(321,120 wheel-samples). The front-wheel residual bias is also positive
(+0.68/+0.94 N), while rear biases are smaller and opposite-signed. In
contrast, the existing source spring/damper calculation using measured 3D body
pose and wheel-center travel gives 0.375–0.403 N per-wheel RMSE on r11, with
absolute 95th-percentile error below 0.55 N. That is an oracle-state result,
not yet a causal rollout, but it identifies missing vertical body/suspension
state—not more tire-force fitting—as the next optimizer model issue.

The serialized prefab confirms `spring=500 N/m`, `damper=100 Ns/m`,
`targetPosition=0.5`, `suspensionDistance=0.05 m`, and
`forceAppPointDistance=0`. Unity describes the spring as applying linear spring
and damping forces and defines the force application point from the resting
wheel base; this matters for the roll/pitch torque arm ([WheelCollider API](https://docs.unity3d.com/ScriptReference/WheelCollider-forceAppPointDistance.html),
[suspensionSpring API](https://docs.unity3d.com/ScriptReference/WheelCollider-suspensionSpring.html)).
The next model revision is a causal reduced-order heave/roll/pitch plus
per-wheel spring/damper state driven by predicted vehicle forces. The existing
offline capture evaluation first showed this model predicts held-out r11
WheelHit loads at 0.652 N RMSE (p95 0.942 N), versus 2.115 N for the former
quasi-static OCP loads. That recursive state/load evaluator is diagnostic and
does not consume future attitude or WheelHit values.

#### Dynamic suspension integrated into research OCP (2026-10-15)

Implemented the source spring/damper load model in the separate free-geometry
research optimizer. It now optimizes body heave, heave rate, roll, roll rate,
pitch, and pitch rate alongside the planar states and four wheel speeds.
Suspension ray travel is constrained to the serialized 0.05 m WheelCollider
range; wheel loads are derived from the calibrated r07 spring target and
serialized per-wheel 500 N/m, 100 Ns/m parameters. Heave/roll/pitch integrate
the current predicted tire forces and wheel-normal moments using the Rigidbody
mass, inertia, drag, and zero force-application-point offset. No production
physics, production trajectory, or runtime odometry/controller code changed.

The candidate-only wall buffer remains 0.050 m. The continuous path geometry
is still optimized through a lateral offset at every spatial node and the
curvature is recomputed from those XY coordinates; the drive/brake phase
count/order remains seeded while all phase lengths and switch locations are
free. Thus this is a true continuous path replan, but not a global search over
hybrid phase topology.

The first OCP attempt (`r25_suspension`) did not produce a feasible incumbent:
IPOPT reached its configured CPU limit at 234 iterations. The initial defect
was dominated by roll-rate dynamics (3.04 in the unscaled residual), and the
late iterate remained far outside the captured-motion trust region. This is a
failed run, not a candidate. A 1 kHz periodic RK4 initializer was then added
for heave/roll/pitch using the warm-start trajectory's own predicted planar
accelerations, with scaled collocation defects. The same seed's largest
initial normalized defect dropped to 0.152 and shifted to yaw-rate dynamics.
The updated `r26_dynamic_suspension` solve reached a feasible periodic
incumbent after 117 feasibility iterations, with maximum constraint violation
`5.0e-9`. Its free path is 28.699 m versus 30.792 m centerline progress;
offset RMS/peak are 0.320/0.724 m. The model predicts 5.6618 s, peak speed
8.743 m/s, peak lateral acceleration 10.493 m/s², peak steering 0.298 rad,
and minimum footprint-to-wall clearance 0.104 m in addition to the requested
0.050 m buffer. Peak roll/pitch are 5.04/3.34 degrees. This is a feasible
candidate under the revised equations, but it is 16.0 ms slower than the
old-model r24 estimate of 5.6458 s; it is not a time improvement and is not
sub-5 s.

The time stage at trust factor 2 exhausted its 240 s CPU allowance; its best
iterate was infeasible, with the trust-region bound as the dominant violation
(6.57) and smaller pitch-rate collocation defects. The feasible phase-I
incumbent was retained. The current follow-up is to continue through larger
trust regions from this same feasible r26 point rather than reject the whole
time search after its first too-tight radius. All track, tire, wheel-spin,
suspension-travel, load, and motion equations remain hard constraints. No
Unity replay has been performed on r26 or any subsequent candidate.

#### Uncapped continuation policy (2026-10-15)

The user clarified that optimization may use the PC without CPU-time or
wall-time limits. Removed the optimizer's `max_cpu_time` settings and the
`--max-solver-seconds` CLI option. IPOPT now has a very high integer iteration
ceiling (`2,147,483,647`) in feasibility and time stages; normal convergence,
infeasibility/numerical failure, or interruption—not a configured time budget—
ends a solve. The already-running r27 process was launched before this change
with a 600-second phase-I budget and 240-second per time-stage budgets. Those
options were parsed when it started and could not be changed in place; it was
allowed to finish, and its report is interpreted with the old cap. Any
continuation launched afterward uses the uncapped settings.

r27 completed after 490.2 s and retained the feasible warm start at 5.66181 s.
Its only time stage stopped after 1,089 iterations with
`Maximum_CpuTime_Exceeded`; maximum constraint violation was 4.183, led by the
captured-motion trust boundary (4.183), lateral-force balance (1.054), and
yaw-rate collocation (0.474). It did not establish a faster or optimized
raceline. The report's legacy `optimality_proven=true` field was misleading:
the selected point was the initial feasible incumbent, not an IPOPT-converged
solution. New reports now record `ipopt_converged` only for IPOPT's explicit
`Solve_Succeeded` or `Solved_To_Acceptable_Level` statuses.

Uncapped independent solves are running from (1) the r27 feasible point,
(2) the older r24 path/state, (3) the older r13 path/state, and (4) no
candidate warm start but a captured-motion/path initializer. The fourth run
is not a pure centerline-path seed, as clarified below. These are research
optimizer runs only; no simulator has been launched. Results will be compared
by feasibility, IPOPT termination, modeled lap time, path clearance, speed,
and tire/load constraints before any candidate is selected for Unity batch
validation.

#### Source fidelity and path initialization clarification (2026-10-15)

The optimizer is materially more source-grounded than the prior generic
kinematic/lateral-limit formulation, but it is not a 1:1 implementation of
Unity/PhysX. It uses active serialized vehicle values and source code for
Rigidbody mass/inertia/drag, wheel radius and steering geometry, motor torque,
WheelCollider friction-curve landmarks, and the configured per-wheel
spring/damper geometry. It includes predicted heave/roll/pitch and suspension
loads. Effective wheel inertia, tire-force scales/tangents, and suspension
target travel remain capture-calibrated. The tire response and wheel-spin
integration are reduced-order approximations; the native WheelCollider solver
and its 1 kHz contact/substep behavior are not reproduced exactly. Thus the
current answer to “does it resemble Unity more accurately?” is “yes, by
explicitly using source mechanics and serialized parameters, but not yet
apples-to-apples; Unity replay remains the truth check.”

The active controller was checked against the downloaded source rather than
assuming every vehicle script applies. The F1TENTH prefab uses
`Assets/Scripts/VehicleController.cs`, whose FixedUpdate runs the source
Ackermann steering and CAWD/CAWB drive branches. The speed/drift-dependent
friction mutation, drift torque, traction control, and steering-helper code
found during the initial scan belongs to the separate
`AutomobileController.cs`; it is not attached as the F1TENTH controller and
must not be copied into this model. The exact runtime capture/config audit,
not the generic F1TENTH prefab's default `MotorTorque` field, supplies the
active 428 N·m motor/brake value used by the optimizer.

The remaining direct source-vs-OCP discrepancies are concrete: the source
steering actuator updates at the simulator's 1 ms physics step under held
25 ms commands, with a sign-sensitive increment/clamp branch; the OCP uses a
continuous symmetric steering-rate bound. The source's zero-throttle branch
applies finite CAWB brake torque (428 N·m per wheel); the OCP's brake phases
instead constrain each wheel angular speed to exactly zero. This is not a
literal 1 ms brake transient, but it is supported as a 40 Hz equilibrium
approximation: the independent r11 capture has all wheel RPM exactly zero in
99.99% of zero-throttle samples. Tire forces remain the largest native-PhysX
gap: the OCP evaluates fitted source-curve approximations rather than Unity's
WheelCollider solver. The source steering and tire-force approximations are
the meaningful fidelity targets after the current multi-start comparison;
there is no evidence-based reason to re-fit the already confirmed source
torque or steady braking behavior.

The path is a genuine continuous free-geometry decision, not a set of
multipliers on a supplied raceline: lateral offset at each spatial node
determines XY and curvature, while speed, steering, throttle, vehicle states,
and all brake/drive phase lengths are optimized. One gate position/heading is
fixed. However, the current phase-I feasibility objective is centered on its
state/path seed, and time minimization begins from that feasible point through
increasing trust regions. This can still affect the basin reached by local
IPOPT. In particular, the run previously named `r30_centerline_start` had no
candidate warm start, but its measured-state initializer included captured
lateral and heading offsets; it was **not** a true centerline path
initialization. The run name overstated what it tested.

Added `--path-seed centerline` to initialize lateral and heading path offsets
to zero while retaining measured motion-state values (speed, lateral speed,
yaw, wheels, steering, acceleration, and controls) and the measured drive/brake
phase topology. The path remains free to optimize at all non-gate nodes. A
matched true-centerline-seed solve (`r32_true_centerline_seed`) is now running
against the same map, model, wall buffer, state capture, and brake schedule as
the captured-offset `r30` solve. This tests the user's proposed centerline
initialization directly rather than inferring its benefit from a misleading
run label. Its `ey` and `epsi` initialization is zero over the map centerline;
measured speed/velocity/yaw/wheel/steering/acceleration/control initialization
is retained. The non-gate geometry is free in IPOPT. The explicit path seed
choice is recorded in the candidate manifest; a syntax check passed before
the solve was launched.

One existing throttle-sweep sequence was also checked against the optimizer's
predicted drive-wheel slip, with no new simulation: the straight 0-to-0.45
throttle sequence (`sequence_index=1141`) has a 25 ms fixed sample grid and
reaches 5.84–6.76 m/s. In 11 adjacent valid intervals at 0.45 feedback, the
rear encoder surface-speed/body-speed ratio implies median longitudinal slip
of about 0.79. The matching r26 candidate nodes (5.81–6.33 m/s, near-zero
steering, throttle 0.402–0.45) predict median rear slip about 0.76 and 95th
percentile about 0.80. This is a useful narrow-regime check on wheel spin and
shows the large slip is plausible in the existing data. It is not broad
validation of tire forces, coupled turning slip, or a full Unity lap. Encoder
angles were differenced directly on the 25 ms grid; phase unwrapping was not
used because per-sample rotation can exceed pi.

### Centerline-started direct-control search (2026-10-15)

The user clarified that the existing successful line is a fallback/seed, not a
geometry constraint, and asked for a source-faithful optimizer rather than a
reduced tire-force approximation. The hard limitation is now explicit: the
Unity source configures the active `VehicleController`, but native
WheelCollider/PhysX contact forces are not available as differentiable source
equations to Python/IPOPT. The exact source-built Unity player is therefore
the plant, not a model being fitted to imitate that plant. Bayesian search is
appropriate for the non-differentiable Unity objective; it must not be
described as a 1:1 Python simulator.

Added `tools/racing/optimize_unity_centerline_control_bayesian.py`. It uses the
map centerline only to index open-loop commands by physical route progress.
The centerline is explicitly **not** a tracking target and no previous path
geometry is imposed: each candidate's raceline is the actual body trajectory
generated by Unity's WheelCollider solver. The optimizer varies eight smooth
steering and eight throttle corrections over the lap. Corrections are added
to the source seed's measured 40 Hz controls, so zero correction preserves all
control detail rather than replacing it with a coarse 8-knot absolute
approximation. The seed controls are mapped from their captured spatial
references to uniform centerline arc progress before resampling. Physics,
production odometry/MPC, and the stored reference raceline are unchanged.

The runner uses the established practice `tools/start_simulator.sh` batch-mode
procedure (Xvfb-backed, no GUI and no `-no-graphics`), the pinned practice
container, the source-built player, 1 ms Unity physics, and commands held for
25 fixed steps. First collision abort remains enforced by the Unity runner.
Candidates are scored on actual forward gate-to-gate Unity time, zero
collision during the scored lap, and the established 0.20 m map-derived
vehicle-center clearance screen. The clearance screen is not a tire/physics
surrogate; Unity collision is the physical collision check. Captures are
retained only for the baseline, current best, and furthest-progress failed
candidate; per-trial dynamics summaries and profiles remain for the full
campaign to keep storage bounded.

Initial exact-player experiments isolated a command-indexing mistake before
the search was expanded:

- `..._r01` used a coarse 8-knot absolute steering/throttle reconstruction.
  Unity completed all 16,775 requested steps, with zero collision, but only
  34.2% ordered route progress and no lap; full-run p95 absolute forward wheel
  slip was 0.947. This rejected basis discarded important 40 Hz command
  structure and is not a valid candidate.
- `..._r02` preserved the seed controls in command-index order but replaced
  their nonuniform spatial references with uniform centerline points. It
  completed 16,775 steps with zero collision, reached 80.3% route progress,
  and did not finish. This showed that preserving command samples alone is
  insufficient: each control must remain aligned to its measured route
  progress.
- `..._r03/trials/trial_000` fixes that transfer. The verified seed controls
  are interpolated against the centerline projection of their original
  spatial references, then evaluated on 360 centerline-index points. At zero
  correction the exact Unity result is **5.712084 s**, zero collisions,
  0.380789 m minimum map clearance, 8.6052 m/s peak body speed, and exactly
  16,775/16,775 fixed steps. This matches the previously verified open-loop
  seed's 5.7121 s result and validates gate alignment, spatial control
  transfer, and fixed-step accounting.

The 64-evaluation `..._r03` campaign is running: the first 37 points comprise
the exact baseline, one-knot-at-a-time steering/throttle perturbations, and
four coupled low-frequency probes; Bayesian expected-improvement proposals
follow. Each candidate is a fresh physical path produced by its commands, not
a path offset constrained to the old raceline. Until this campaign has
completed and a faster candidate has passed the same Unity collision and map
screens, no new raceline is promoted. Current valid record remains 5.712084 s;
sub-5 s remains unmet.

At evaluation 20/64, three additional single-knot controls have produced
valid but slower laps: steering `+0.10` at knot 6 gives 6.204525 s / 0.300 m
minimum clearance; throttle `-0.09` at knot 0 gives 5.816932 s / 0.320 m;
throttle `+0.09` at knot 0 gives 5.805836 s / 0.335 m. These are feasible
search points, not promoted results. Other single-knot steering changes
produced gate crossings between 4.90 and 7.90 s but only 17–76% of scored
samples met the map margin, with some sectors at zero clearance; all are
rejected as shortcuts/off-course paths. This directly confirms that the Unity
gate time must be paired with the full path-clearance constraint. At this
historical checkpoint the search had completed 29/64 runs; the current
campaign status is recorded below.

The five reduced-order IPOPT jobs r28–r32 were stopped on 2026-10-15 before
they produced candidate artifacts, to free compute for the exact-Unity
campaign. Earlier completed candidates and source data were preserved. Their
outputs cannot be represented as exact Unity results. The prior note that
r32 was running is historical and superseded by this status.

### Acceleration and deceleration path audit (2026-10-15)

The active exact-Unity path was checked in both source and the baseline's
1 kHz capture (`trial_000`, scored gate interval 7.7750–13.4871 s). This
distinguishes correct simulator actuation from what the optimizer currently
commands:

- Unity's `VehicleController.FixedUpdate()` reads the autonomous command and
  calls `Drive()` each fixed step. The player reports a 1.000 ms fixed step;
  the offline profile runner holds each command for 25 fixed steps. The
  baseline capture contains 16,775 samples with exact 1 ms spacing and the
  result reports 16,775/16,775 steps applied. There is no missed or slower
  acceleration-update cadence in this replay.
- Positive throttle maps directly to motor torque (`ThrottleLimit * input *`
  `MotorTorque`); the RoboRacer sim-racing scene overrides the F1TENTH
  prefab's 85.6 N·m default with 428 N·m, and the capture confirms the
  resulting torque distribution across four CAWD wheels. On the scored baseline,
  the largest requested throttle was only 0.453, corresponding to 48.5 N·m
  per wheel, not full throttle. The capture confirms commanded torque reached
  the wheel colliders; there is no evidence here of a software throttle
  multiplier silently capping the request below the profile.
- The deceleration behavior is a consequential source-level discontinuity:
  exactly zero throttle sets motor torque to zero and applies 428 N·m brake
  torque to all four CAWB wheels. It does **not** coast. Negative throttle is
  not used by this optimizer; with the active CAWB branch it would enter the
  motor-torque branch rather than either independent-brake branch. Thus the
  current optimizer must treat zero as a brake mode, not as the low end of a
  smooth throttle continuum.
- In the scored baseline, the wheel telemetry classified 88.6% of samples as
  driven and 11.4% as braked; commands were never negative. The zero-throttle
  portions decelerated at approximately 6.8–7.6 m/s² in the captured
  body-forward direction. During positive-throttle samples, the median
  measured forward acceleration was about 2.05 m/s², with substantial
  turn-dependent spread; that distribution is not a straight-line motor
  acceleration curve. Peak speed was 8.605 m/s, but median speed was only
  4.873 m/s and the car spent 0.29 s of the scored lap above 8 m/s.
- Therefore 8.61 m/s is not an optimizer-imposed cap: baseline peak throttle
  is 45.3%, and exact-Unity candidates have reached 9.64, 11.09, and 13.58
  m/s. Those higher-speed trials were off-track/incomplete and slower or had
  no valid lap, so they are not evidence of a useful speed improvement.

Conclusion: the acceleration and braking *actuation* are being applied
faithfully by the source Unity simulation; the evidence does not show a
simulation-rate or command-to-torque bug. It does show that the seed policy is
conservative on throttle and that the optimizer faces an on/off full-brake
transition at zero. Its smooth additive throttle search can cross that mode
boundary, so the remaining campaign results should be interpreted with this
hybrid behavior in mind. Do not change Unity physics or infer that globally
raising throttle is safe; exact-Unity trials show higher peak speed without a
faster valid lap.

The 64-evaluation campaign is complete. Eight candidates passed both the
full-lap and map-margin checks: trial 000 at 5.712084 s, trial 014 at
6.204525 s, trial 017 at 5.816932 s, trial 018 at 5.805836 s, trial 020 at
5.784847 s, trial 023 at 6.472213 s, trial 029 at 6.353605 s, and trial 031
at 6.451761 s. No candidate improved the baseline. Trial 062 reached a
17.271 m/s peak but did not complete a valid lap; trial 034 crossed in
7.676680 s at 9.635 m/s but had zero map clearance and only 30.0% of samples
inside the vehicle-margin threshold. These runs confirm that peak speed or a
fast gate crossing alone is not a useful optimization result. The campaign
ended above the 5.0 s target and produced no trajectory to promote.

### Positive-throttle deceleration and optimizer candidate (2026-10-15)

The user's observation that speed reduction should not be equated with the
simulator's exact-zero full-brake command was checked against existing paired
captures before starting any new simulator run:

- Signed throttle is not a supported production command. The production
  `LongitudinalCommand` contract rejects values outside `[0, 1]`, the speed
  controller saturates requested throttle to `[0, throttle_max_forward]`,
  and the optional raw-throttle diagnostic override clamps to the same
  nonnegative interval. Unity's generic vehicle script can parse signed
  values, but that source capability is not evidence that the competition
  controller/actuator path can issue or use them. No negative-throttle mode is
  included in this optimizer candidate.
- Unity's active `Drive()` branch applies positive motor torque and clears
  wheel brake torque for every nonzero command; exact zero selects the
  separate full-brake branch. Thus reducing a positive command is not itself
  braking: it lowers propulsion, and body speed decreases only when the
  resulting net tire/drag/resistance forces point backward. The optimizer
  must demonstrate that net deceleration from its model and replay rather
  than treating a lower throttle target as a braking force.

- `openplane_swerve_throttle_slew_train_r05_20261006` and the independent
  `openplane_swerve_throttle_slew_validation_r05_20261006` each contain 20
  valid matched step-vs-0.30 s ramp pairs for throttle reductions. Start/end
  commands were positive in every pair: approximately `0.18 -> 0.10` at
  4.5 m/s, `0.27 -> 0.19` at 6.5 m/s, `0.31 -> 0.23` at 7.5 m/s,
  `0.393 -> 0.313` at 9.5 m/s, and `0.462 -> 0.424` at 11.1 m/s. Command
  tracking and fixed 25 ms timing integrity gates passed. Thus these are
  positive motor-torque changes, not the exact-zero brake branch.
- For every one of the 20 pairs in both captures, the step produced a more
  negative forward-speed change than the ramp during the first 0.10–0.35 s.
  The median paired step-minus-ramp difference was −0.658 m/s (training) and
  −0.661 m/s (validation). In these coupled turning maneuvers, the positive-
  throttle down-transition profiles do decelerate, and the paired difference
  shows that the transient matters, not just the final throttle value.
- Rechecking the per-phase validation records makes the evidence boundary
  clearer: all 40 valid down-transition profiles (20 matched conditions ×
  step/ramp) had a lower mean `u` over 0.10–1.00 s than their own pre-stimulus
  window; the independent training capture also had 40/40. This is direct
  observed deceleration while commands remained positive, but steering was
  changing substantially in these swerve profiles, so it does **not** isolate
  straight-line throttle-only deceleration. The model/replay must retain the
  coupled tire-force explanation and cannot treat the throttle drop as an
  active brake torque.
- The same step increased mean absolute rear-wheel/body-speed mismatch in
  19/20 pairs in each capture. Median mismatch increase was +0.227 m/s in
  training and +0.276 m/s in validation. The effect is strongest at 4.5 m/s
  (validation median +0.626 m/s) and smaller at 11.1 m/s (+0.048 m/s).
  Median step-vs-ramp mismatch differences had returned close to zero by
  0.65–1.00 s. The ramp therefore reduces the transient wheel-speed mismatch;
  these results do not justify claiming zero slip.
- Under the positive-throttle ramp, measured median forward speed at
  0.65–1.00 s fell from 4.41 to 2.39 m/s for `0.18 -> 0.10`, 6.60 to 4.66
  for `0.27 -> 0.19`, 7.56 to 5.64 for `0.31 -> 0.23`, 9.44 to 7.57 for
  `0.393 -> 0.313`, and 11.04 to 10.17 for `0.462 -> 0.424`. The same
  medians appear in both captures. Steering was also changing during this
  swerve experiment, so these are observed coupled responses, not a pure
  throttle-only equilibrium curve.
- These paired runs contain one capture per split, so the 20 pairs describe
  condition-level repeatability within each capture, not a large run-level
  population. They establish a useful control/model effect, not a universal
  braking law or a proof that every positive-throttle reduction is stable.
- The older interrupted straight-only sweep was salvaged offline to check
  whether it filled this specific evidence gap:
  `live_runs/openplane_throttle_transitions_straight_20260930_r01/`.
  Its phase metadata confirms an intended 5,050-condition increasing-step
  design. The closed bag contains 245 completed phases, one started but
  incomplete phase, 4,804 not-started conditions, no duplicate keys, no
  collision/timing-fault messages, and an overall active-stream rate of
  39.98 Hz (p95 gap about 25.7 ms; maximum below 50 ms). The 245 completed
  conditions all increase throttle; there are **zero positive-to-lower-positive
  steps**, so this bag cannot validate controlled throttle reduction. Its
  legacy phase-end records also omit the newer per-condition response and
  steering/odom-quality fields, so its speed changes are not being promoted
  into the deceleration fit. A separate partial-capture report was written
  without modifying the source bag; its overall integrity status correctly
  remains failed. `tools/analyze_open_plane_throttle_transitions.py` now has an
  explicit `--allow-partial` audit mode for such salvage, while the default
  behavior for incomplete captures remains unchanged.

The source-grounded optimizer already maps each positive throttle value to
four-wheel motor torque and predicts wheel spin/tire force. Its spatial
control parameterization did not constrain falling throttle rate, however,
and its default mode seed came from a measured trajectory containing exact
zero-brake sections. An optional, isolated candidate path now supports (a)
all-positive-throttle mode seeding and (b) a one-sided falling-throttle
constraint in time. The candidate rate `0.267 normalized throttle/s` is the
measured `0.08 / 0.30 s` down-ramp rate; it is a control constraint, not a
change to Unity physics or a claim about native actuator lag. Defaults,
production odometry/MPC, and the reference trajectory remain unchanged.

Important distinction: the Unity command-to-torque branch is source-exact in
this optimizer (428 N m total, split equally across four driven wheels for a
positive command; exact zero instead applies the CAWB brake branch). The
torque-to-body-acceleration mapping is **not** a 1:1 WheelCollider simulation:
it integrates wheel spin with an identified effective inertia/damping and
uses a reconstructed two-piece tire curve whose native spline tangents and
full PhysX contact solve are not available in the optimizer. In addition,
Unity holds each generated command for 25 physics ticks (25 ms), while the
OCP integrates a spatial collocation mesh and its output is later resampled
to those held commands. Therefore exact torque scaling alone does not prove
accurate acceleration or deceleration prediction; the source-player replay
must compare predicted versus actual speed, wheel speed, and acceleration.

Two all-positive-throttle, centerline-initialized OCP attempts have failed at
`live_runs/raceline_candidates/unity_positive_throttle_decel_20261015_r01`
and `_r02`. Both stalled at essentially the same large initial/collocation
defect (about 0.275); removing the falling-throttle constraint in `_r02` did
not reduce it. The first attempt also violated the proposed throttle-rate
constraint, but this does not establish that the rate cap caused the solve
failure. These centerline-seeded failures therefore say nothing conclusive
about whether positive-throttle-only motion is feasible. A third attempt is
now using the captured path geometry rather than replacing its lateral and
heading offsets with zero:
`live_runs/raceline_candidates/unity_positive_throttle_decel_20261015_r03`.
No Unity replay or candidate lap-time gain has been established yet. If this
returns a feasible profile, it must still pass the established batch-mode
Unity replay with first-collision abort and full map-margin checks.
At 2026-10-15 13:52 CEST, r03 was still actively solving after 33 minutes
(about one CPU core and 2.39 GB RSS); no simulator or other test process was
running. This is solver progress only, not a feasible candidate or a result.

### Straight-line positive-throttle downshift capture (2026-10-15)

To fill the specific gap in the earlier steering-coupled downshift captures, I
ran `race_domain_straight_throttle_downshift_train` in the established pinned
Explore batch-mode simulator. This was a separate data capture, not a physics
or production-controller change. The schedule randomized six reset-matched
step/ramp pairs: two each at 4.5, 6.5, and 8.5 m/s, all at zero steering.
Each pair compared a one-control-tick throttle step with a 0.30 s ramp to the
same lower positive throttle. There were 12/12 successful reset recoveries,
6/6 valid matched pairs, no collisions or timing faults, and no speed-governor
intervention. Commands and throttle feedback passed the profile gates in all
12 phases; packet-aligned Unity rigid-body forward velocity and odometry
agreed to the shown precision. Active streams were approximately 39.97 Hz,
with p95 gaps below 26 ms.

| Initial speed | Positive throttle step/ramp | Unity truth speed, pre → 0.10–0.35 s (ramp / step) | Unity truth speed, 0.65–1.00 s (ramp / step) | Step-minus-ramp difference in early speed change |
|---:|---:|---:|---:|---:|
| 4.5 m/s | 0.18 → 0.10 | 4.44 → 3.58–3.74 / 3.39 m/s | 2.484 / 2.485 m/s | −0.194, −0.348 m/s |
| 6.5 m/s | 0.27 → 0.19 | 6.60 → 5.90–6.04 / 5.25–5.43 m/s | 4.680 / 4.679 m/s | −0.471, −0.787 m/s |
| 8.5 m/s | 0.35 → 0.27 | 8.49 → 7.89–8.02 / 7.14 m/s | 6.607 / 6.601 m/s | −0.750, −0.874 m/s |

All six step/ramp pairs show a larger *early* speed reduction for the abrupt
step; by 0.65–1.00 s, their measured speeds are nearly the same under the
common final throttle. That later window is not a steady-state measurement.
This confirms, in the tested
straight-line conditions, that dropping to a lower **positive** throttle can
decelerate the vehicle without invoking Unity's exact-zero brake branch, and
that throttle slew changes the transient. It is not an all-speed deceleration
law: this is one capture with two within-capture repeats at each of three
speeds, so it does not provide independent run-level uncertainty or broad
throttle coverage. The encoder/body-speed residual also rises more in the
step phases (step-minus-ramp change about +0.17 to +0.71 m/s in the early
window); this remains a wheel-speed mismatch proxy, not direct tire-slip
measurement.

Interpretation for implementation: Unity's command semantics are represented
correctly in the positive-throttle candidate—positive values produce motor
torque; exact zero selects the configured CAWB braking branch; negative
throttle is neither needed nor used. The candidate also allows positive-only
throttle nodes and includes wheel-spin/tire-force terms. However, the
torque-to-wheel/body response is still an identified approximation rather
than the native WheelCollider solve, and this new straight-line capture has
not yet been replayed through that model to score predicted versus measured
`u`, acceleration, and wheel-speed response. Thus the *command branch* is
source-grounded, but the optimizer's throttle/acceleration prediction is not
yet validated as correct. Do not promote the optimizer or claim an improved
raceline from this experiment alone.

The analyzer's zero-steering label was also corrected: these observations now
appear under `turn_direction="straight"`, not a spurious turn direction. The
saved JSON and compact CSV were regenerated from the closed bag after that
fix. Schedule test, analyzer/experiment Python compilation, and
`git diff --check` pass. The simulator was shut down after capture; no
simulator container remains running.

At the latest process check, the captured-path positive-throttle OCP attempt
r03 remained active after about 63 minutes at one CPU core and approximately
2.39 GB RSS, without a completed candidate/report artifact. This is not
evidence of convergence. The status at 13:52 above is historical; this newer
capture and process check supersede its statement that no simulator test had
yet been run.

### Wheel-speed/body-speed mechanism: source and signed data check

The relevant chain is not an unknown throttle-to-acceleration lookup. The
active Explore scene sets `MotorTorque=428 N·m`; at throttle `t`, the CAWD
controller applies `428*t/4 = 107*t N·m` at each wheel. Thus 50% means
214 N·m total, 53.5 N·m per wheel. These are torque units, not N·m/s. At
exactly zero, CAWB instead applies 428 N·m braking torque to each wheel. The
test above stayed above zero, so its wheel-speed response is not that brake
branch. See `VehicleController.Drive()` and the configuration audit linked
above.

The source encoder code integrates `WheelCollider.rpm` at the 1 ms physics
tick, accumulates/truncates encoder ticks, and publishes angle. The configured
left and right encoders each observe one rear wheel. On this straight test I
computed signed rear wheel surface speed from consecutive 25 ms encoder-angle
samples using the serialized 0.059 m collider radius, then compared it with
packet-level Unity body-forward speed. Let `e = mean(R*omega_rear) - u`; this
is a rear longitudinal-slip *kinematic estimate* in the straight case, not a
WheelHit force measurement.

| Speed | Step: `e` before → early response | Ramp: `e` before → early response | Step-minus-ramp change in `e` |
|---:|---:|---:|---:|
| 4.5 m/s | about +0.09 → −0.81 m/s | about +0.09 → −0.21 m/s | −0.59, −0.62 m/s |
| 6.5 m/s | about +0.19 → −0.59/−0.40 m/s | about +0.19 → −0.27/−0.21 m/s | −0.32, −0.19 m/s |
| 8.5 m/s | +0.315 → −0.32/−0.39 m/s | +0.315 → −0.24/−0.23 m/s | −0.08, −0.15 m/s |

The response window is 0.10–0.35 s; the two entries per speed are the two
within-capture repeats. All six abrupt steps make `e` more negative than the
matched ramp: the rear wheels are rotating slower than body rolling speed
after the reduction. With the guide convention
`Sx=(R*omega-vx)/vx`, that is consistent with negative longitudinal slip and
a tire force opposing the body's forward motion. This directly supports the
user's wheel-spin/slip explanation. The body speed also falls (the step
reduces it more quickly), so the useful model must propagate wheel angular
speed and body speed together through slip-dependent tire force; encoder
speed cannot be substituted for body speed.

Correction to the earlier evidence wording: for this straight, zero-steering
capture, the signed wheel/body quantity is more informative than an unsigned
“mismatch proxy” and supports a rear-slip estimate. It still does not reveal
the native per-wheel `WheelHit` force, front-wheel slip, or exact force
transient. Those are available in the separate 1 kHz development capture, not
in the competition-compatible 40 Hz bag. The Unity project gives the exact
actuator branch, wheel geometry, friction-curve parameters, and rigid-body
equations; `WheelCollider`'s actual contact/friction solve is a native engine
implementation rather than C# project source. This is an implementation
fidelity boundary, not a reason to treat the vehicle physics as arbitrary.
The standalone OCP already has the right coupled state structure, but its
effective wheel inertia/damping, tire-force mapping, normal loads, and spatial
discretization remain approximations that must be checked against the signed
step/ramp trace and the 1 kHz WheelHit capture before it can be called
source-faithful.

### 1 kHz wheel/contact-slip confirmation and current-odom replay (2026-10-15)

The separate 1 kHz source-player trace was aligned to the 40 Hz experiment
events using the logged throttle profiles (one constant epoch offset, 4.715 s;
all 12 event speeds also match their source-trace plateaus). The compiled
diagnostic player had a Unix-epoch base 10x too small, while its elapsed
1 ms sample intervals were intact. Therefore the trace is suitable for
within-run dynamics analysis after this alignment, but its raw absolute
timestamps must not be used directly.

The trace records per-wheel RPM, motor/brake torque, `WheelHit.forwardSlip`,
and Unity rigid-body velocity. For the positive-throttle downshift, the
step-minus-ramp differences below compare each treatment's change from its
own pre-stimulus state; they are not absolute acceleration values. Means are
over the two paired repetitions at each speed. Rear wheel speed is RPM times
the serialized 0.059 m radius; the final column is Unity's measured rear
`WheelHit.forwardSlip`.

| Start speed | At 50 ms: body-speed change | Rear wheel rolling-speed change | Wheel-minus-body change | `WheelHit.forwardSlip` change |
|---:|---:|---:|---:|---:|
| 4.5 m/s | −0.006 m/s | −1.594 m/s | −1.588 m/s | −0.356 |
| 6.5 m/s | −0.141 m/s | −1.398 m/s | −1.257 m/s | −0.189 |
| 8.5 m/s | −0.287 m/s | −1.699 m/s | −1.412 m/s | −0.170 |

Thus the abrupt lower-torque command can reduce wheel rotation much faster
than body speed in the first 50 ms, with measured longitudinal slip shifting
negative. By 200 ms the wheel/body difference is nearly closed at 6.5 m/s and
has crossed sign at 8.5 m/s. The transient is speed-dependent; a fixed
wheel-to-body scale or a direct throttle-to-acceleration lookup would miss
its time evolution. This directly supports the proposed wheel-spin/slip
mechanism. It does not mean that the encoder is erroneous: the encoder
reports wheel rotation, which is not identical to body speed during slip.

The actuator mapping itself is exact and simpler than the tire response. The
active Explore scene sets `MotorTorque=428 N·m`; the CAWD branch applies
`107*t N·m` to each wheel for normalized positive throttle `t`, and clears
brake torque. At 50%, that is 214 N·m total / 53.5 N·m per wheel. Exactly zero
selects the separate CAWB branch, applying 428 N·m brake torque at each
wheel. The tested downshifts remained positive, so that brake branch was not
involved. Torque is in N·m, not N·m/s. Unity source: `VehicleController.Drive()`
and the Explore scene override; the compiled native `WheelCollider` solver is
not C# source in the project, but its RPM and contact-slip outputs are exposed
by this development-only capture.

The existing command-only 1 kHz plant was evaluated on this new capture as
held-out data, using the existing r07 training fit and no future truth as an
input. It produced 34 reset-safe 1 s windows (11 reset-crossing windows
excluded). The baseline fitted-slip variant gave horizontal-position RMSE
0.0536 m / p95 0.1349 m, forward-speed RMSE 0.0777 m/s, and rear-wheel angular
speed RMSE about 0.0854 rad/s. The Unity-guide slip formula on the same
capture gave 0.0514 m / p95 0.1314 m, 0.0690 m/s, and about 0.0533 rad/s,
respectively. That is a promising local result, not a global promotion: on
the earlier independent r11 5 s replay, the guide-slip variant had 4.24 m
position RMSE versus 1.87 m for the existing formulation. Keep the current
general fit until a regime-conditioned alternative passes both domains.

The current production observer was also replayed from only the bag's
40 Hz encoders, IMU, steering, and throttle feedback (1,137 matched samples,
12 reset-separated phases, 39.97 Hz, zero timing faults/collisions). Current
configuration (`wheel_update_beta=0.85`) scored 0.1175 m/s longitudinal-speed
RMSE / 0.1962 m/s p95 absolute error, and 0.0706 m phase-macro relative-pose
RMSE. An offline global beta override of 0.5 scored 0.1168 m/s / 0.1808 m/s,
but phase-macro pose RMSE slightly worsened to 0.0706 m. This is too small and
not consistently beneficial to promote. It rejects the simple global-weight
change and points back to a causal wheel/body-slip state, not arbitrary
retuning of the encoder weight.

No production odometry, MPC, or optimizer physics was changed from these
results. The replay-only beta override is diagnostic; the current offline
plant remains a separate research model and is not yet in production odom.
Next for this issue: use the torque/slip balance as the observer's latent
wheel/body state, with only legal 40 Hz inputs at runtime; keep Unity
`WheelHit`/truth strictly as offline labels, and validate the candidate on
held-out practice data before promoting it. No new simulator run is justified
yet by this capture.

#### Actuator-expert ablation on the downshift capture

The same 40 Hz bag was replayed through production odometry with only
`use_unity_actuator_u_experts` disabled. Steering remained zero, so the
high-steering expert was inactive in both replays; all other observer
configuration and input topics were unchanged. With the source-conditioned
experts enabled, forward-speed RMSE was 0.1175 m/s (bias +0.0075, p95 absolute
error 0.1962 m/s), and reset-separated relative-position macro RMSE was
0.0706 m. Disabling them raised speed RMSE to 0.1808 m/s (bias −0.1235, p95
0.3620 m/s) and position macro RMSE to 0.1827 m (p95 across phase sequences
0.2639 m). The source-conditioned correction is materially useful on this
downshift dataset; it stays enabled. The global wheel blend change to 0.5
remains unpromoted because its small speed-score change did not improve the
phase-position score consistently.

This confirms the proposed causal chain in the instrumented capture: the
command-to-torque mapping is directly known, while wheel rotational response,
tire contact force, and body deceleration are coupled through the
slip-dependent contact solve. One precision distinction: the source field is
`MotorTorque=428 N·m` (not a torque rate); normalized 1.0→0.5 changes each
CAWD wheel request from 107 to 53.5 N·m. The `85.6` values in the production
residual feature vector are a legacy feature scaling used during fitting, not
the physical torque configured by the active simulator. Because these are
regression features, changing their scale alone would require a matching
coefficient refit and is not evidence by itself of a physical-model error.
The offline four-wheel plant uses the 428 N·m source value; production
odometry still applies a fitted source-mode residual rather than integrating
wheel angular momentum and tire force as explicit latent states. That is the
remaining implementation gap to address and validate, not uncertainty about
the throttle-to-commanded-torque mapping.

## Raceline optimization failure analysis (2026-10-15)

The sub-5 s objective has not progressed. The best verified Unity lap remains
5.712084 s; the gap to 5.0 s is 0.712084 s (12.5% of the current lap time).
No new sub-5 s or faster-valid-lap result is established here.

The strongest evidence is the 64-evaluation campaign
`live_runs/raceline_candidates/unity_centerline_direct_control_20261015_r03/`:

- 8/64 trials passed full-lap and map-clearance validation; 56/64 did not.
- The fastest valid trial is trial 000, the unchanged seed, at 5.712084 s.
  The other seven valid trials are slower (5.784847–6.472213 s).
- The nominal 3.012969 s trial 050 is invalid with zero map-derived clearance.
  It is an off-track shortcut, not an optimization result.
- This campaign's 16 decision variables are eight broad steering and eight
  broad throttle corrections to a fixed seed command stream. Its route is used
  to index commands; the declared setup does not constrain vehicle motion to an
  optimized geometric path. Thus the campaign did not perform a bottom-up
  raceline geometry optimization.

A separate nine-trial sector-focused campaign,
`live_runs/raceline_candidates/unity_r13_final_sector_focus_search_20261015_r01/`,
used `unity_r13_clearance_seed_local_control_search_20261015_r02/trials/trial_030`
as its seed. That seed's trial record is invalid (7.213051 s, zero clearance).
All nine trials in the follow-on campaign are invalid, including trial 000
(zero parameter adjustment). Those trials cannot support conclusions about
local optimization around a valid racing line; the campaign should not have
been started from that seed.

For the wheel-level IPOPT run
`live_runs/raceline_candidates/unity_bottomup_gate_anchored_20261015_r01/`,
IPOPT stopped after 283 iterations with `Infeasible_Problem_Detected` and no
feasible incumbent. Its final worst residual is the exact gate lateral-offset
constraint: 5.63 mm. The initial guess is materially inconsistent with the
collocation equations (maximum residual 0.794 m), so this is not simply a
successful optimizer held back by a harmless gate tolerance. A retry with a
1 cm gate tolerance was interrupted before completion and produced no result;
it must not be described as a solver success. Even a converged OCP candidate
would remain a model prediction until replayed in Unity: the plant uses fitted
tire response and is not the native WheelCollider/PhysX solver.

The saved hybrid-profile artifact
`live_runs/raceline_candidates/unity_sector5_throttle_only_hybrid_20261015_r02/profile_build.json`
reports a 5.607584 s **predicted** time but also records
`measured_lap_time_s: null` and `candidate_profile_built_not_sim_validated`.
That value is not evidence of a faster lap. Likewise, the existing baseline
replay is the only valid timing comparison to use until a candidate has a
persisted complete Unity capture, gate-to-gate time, collision result, and map
clearance analysis.

### Diagnosis and next decision

The lack of progress is not a lack of BO iterations. The search space is
poorly matched to the goal: campaign r03 perturbed a fixed command stream with
eight broad steering and eight broad throttle knots (16 variables) over only
64 Unity evaluations. It did not search a free geometric line or an
independently planned speed profile. Its large invalid fraction left it with
very few feasible comparisons. The current script has since been reduced to
four knots per control with tighter bounds, but r03's saved manifest is the
source of truth for that historical run. A later campaign additionally
started from a known-invalid seed. In parallel, the bottom-up OCP has not
produced a feasible incumbent, and its Unity-derived reduced plant is not an
exact replacement for the source simulator.

Before claiming further optimization progress, the next campaign must:

1. Reject a seed unless its own completed Unity trial is valid and meets the
   map-clearance criterion; do not inherit candidate seeds merely because
   their simulated trajectory can be serialized as a profile.
2. Keep the verified 5.712084 s profile as the immutable incumbent and compare
   every proposed line against it using persisted full Unity evidence.
3. Optimize actual path geometry and speed planning as separate decision
   variables, with hard track-boundary feasibility in the search rather than
   accepting attractive off-track times as useful candidates.
4. Do not run a large BO campaign or wait hours on IPOPT until the optimizer
   can reproduce the incumbent as a feasible starting point. The OCP initial
   state/collocation mismatch and gate anchoring need resolution first; then
   its candidate must be checked against unchanged Unity physics.

No new simulation was started for this diagnosis. There are no active
optimizer or Unity processes from this work at the time of writing.

## Follow-up: why the faster OCP paths failed clearance (2026-10-15)

I compared the saved OCP states to the two completed Unity candidate captures,
using 40 Hz samples projected onto continuous OCP path segments. This is a
spatially matched path/state comparison, **not** an equal-time open-loop
rollout score. The analysis is now available in
`tools/racing/validate_unity_wheel_raceline.py` and is attached to future
candidate validation reports; it records path, forward/lateral speed, yaw
rate, and measured road-wheel steering errors, plus first threshold crossing
and signed path bias by route sector. A focused test covers the calculations.

Evidence from the existing Unity captures:

| Candidate | Unity time | Clearance | Path error RMSE / p95 | u RMSE | v RMSE | yaw-rate RMSE | wheel-steering RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| r09, 5 cm OCP margin | 5.6511 s | 0.100 m (invalid) | 0.270 / 0.436 m | 0.118 m/s | 0.019 m/s | 0.096 rad/s | 0.004 rad |
| r10, 20 cm OCP margin | 5.6966 s | 0.112 m (invalid) | 0.263 / 0.422 m | 0.114 m/s | 0.022 m/s | 0.121 rad/s | 0.005 rad |

Both laps are faster than the 5.7121 s valid reference, but both fail the
0.20 m clearance criterion; neither is a valid time improvement. The path
error first exceeds 0.10 m about 0.70–0.73 s after the gate. Forward-speed
error never exceeds 0.50 m/s on either lap, and lateral-speed error never
exceeds 0.20 m/s. Yaw-rate error first exceeds 0.20 rad/s near 2.0 s. Measured
road-wheel angle closely follows the OCP steering state (about 0.005 rad
RMSE), so a steering command normalization correction is not supported.

The signed cross-track error has a repeatable spatial pattern on these two
profiles: the actual car is roughly 0.22–0.35 m to the right of the planned
line in route sectors 2–4, then roughly 0.24–0.44 m to its left in sectors
6–9; the leftward bias is smaller in sector 10. This pattern persists when
the OCP wall margin changes from 0.05 to 0.20 m. A uniform margin increase
is therefore not the right correction.
The two runs also show no useful uniform yaw-rate multiplier: a scalar fit
changes yaw-rate RMSE only from about 0.100 to 0.100 rad/s on r09 and from
0.121 to 0.120 rad/s on r10. The residual is state/sector dependent, not a
single global gain error.

I implemented that correction as an **optional OCP boundary model**, not as a
runtime or vehicle-dynamics change. `--tracking-bias-training-run-dir` fits
the 12-sector signed path residual from one completed practice candidate;
`--tracking-bias-validation-run-dir` requires an independent completed lap on
the same centerline. Each sector needs at least five samples, and the largest
sector-median transfer error must be no greater than the selected wall margin.
The solver then constrains `actual lateral offset = OCP lateral offset +
fitted path bias`. It changes only track-boundary constraints. Defaults remain
unchanged, and the candidate manifest records both source runs and transfer
metrics.

On r09→r10, transfer RMSE is 0.033 m, p95 absolute sector error is 0.065 m,
and maximum sector error is 0.070 m. Thus it passes the explicit 0.20 m
candidate margin used in this experiment. The first bias-aware OCP attempt
used r09 for fitting and r10 for transfer validation, with no Unity run. IPOPT
reduced maximum constraint violation from 0.224 m to 0.100 m in its bounded
600 feasibility iterations, but stopped at `Maximum_Iterations_Exceeded`;
there is no feasible incumbent or new raceline. The largest remaining
violations are track-boundary constraints, so this is **not yet a usable
candidate**. The run report is
`live_runs/raceline_candidates/unity_tracking_bias_first_transfer_20261015_r02_solve/solver_failure.json`.

Repeating from the more conservative r10 OCP seed reduced the initial maximum
violation from 0.222 m to 0.05914 m at 600 iterations. Extending the same
bounded solve to 1,200 iterations plateaued at 0.05928 m (dual infeasibility
13.47); more iterations did not help. Constraint labels now identify the
remaining regions: the right boundary is active around centerline station
10.9–12.2 m (about 3.2–3.6 m/s, fitted bias near -0.32 m), while the left
boundary is active around 18.9–19.9 m (about 6.1–6.3 m/s, bias near +0.44 m).
The result is a candidate-model feasibility/initialization limitation, not an
iteration-budget issue. Reports are in `r06_r10_seed_solve/` and
`r07_r10_seed_1200/` under the same candidate directory.

I also checked whether subtracting the learned bias from the warm-start
lateral state would help. It increased the initial Frenet collocation defect
to 0.267 m; deriving a new heading seed from the altered path increased the
worst defect to 0.978 rad. Both seed-only edits were removed. The optional
model therefore changes only the boundary constraints; it does not pretend
that a geometric offset alone is a dynamically consistent vehicle rollout.

A minimal projection of the r10 seed into the bias-corrected corridor removed
the boundary residual initially but raised the initial Frenet defect to
0.164 m. Its 600-iteration solve ended at 0.744 m maximum violation, led by
lateral force balance at the first nodes. That projected-seed code was also
removed. The optimizer needs a dynamically consistent reinitialization for a
shifted path; simply clipping lateral position is not acceptable.

This is a bounded candidate-generation experiment for this course, not a
general vehicle model. r09/r10 are related variants on one practice course;
test/final-test data remain sealed. No Unity physics, production odom, MPC,
reference trajectory, or runtime controller changed. Focused verification:
24 tests passed. No Unity simulator was launched for this follow-up. Do not
claim a faster valid lap from r09/r10, and do not loosen the global wall margin
to hide the remaining OCP boundary infeasibility.

### Minimum-time IPOPT search focus (2026-10-15)

The current objective is the theoretical minimum-time path under the OCP's
map, vehicle, steering, throttle/brake, and 12 m/s constraints. Unity tracking
is not the optimization target for this step. A Unity lap is not required to
rank nominal OCP solutions; it is only a later check that the approximate
plant's predicted time is physically credible.

Review of the last time-search reports found a continuation bug: IPOPT stopped
at the first expanded trust radius whenever dynamics residuals were also high,
so it could fail to reach the final near-unconstrained time stage. The solver
now retries each wider radius from the best feasible incumbent after any
finite failed stage, while still refusing to promote an infeasible iterate.
The per-stage time-search budget is now configurable and defaults to 2,000
iterations (previously 450). No Unity/runtime physics or production
controller/trajectory was changed.

The latest feasible r26 start has a 5.66181 s *feasibility-fallback* estimate,
not an optimized result. Its active slow zones include approximately 3.0–3.3
m/s around stations 12.8–13.6 m and 26.2–26.5 m, where predicted lateral
acceleration is roughly 6.5–9.1 m/s². Its peak speed is 8.74 m/s despite the
12 m/s cap, so straights are not the limiting constraint. This supports
searching for a different turn geometry / periodic speed solution rather than
increasing the straight-line speed cap. The route's minimum clearance remains
constrained by the 5 cm modeled footprint buffer.

New IPOPT solves, using the same physical limits and no tracking-bias model:

- `unity_free_geometry_time_optimization_20261015_r01`: feasible dynamic-
  suspension r26 path/state start; 2,000 iterations per time stage.
- `..._r02`: independent older r24 path/state start; its 600-iteration
  feasibility stage stopped at 0.00109 maximum violation, so it is only a
  near-feasible iterate and not a result.
- `..._r03_centerline_start`: centerline geometry initialization failed to
  reach feasibility in 600 iterations (0.484 maximum violation). This exposed
  that simply zeroing path offset/heading is not a model-consistent centerline
  initialization; it is not a valid time result.
- `..._r04_periodic_gate`: r26 start with periodic lap-entry speed, yaw, wheel
  speeds, and steering free, rather than fixed to the warm-up gate state. The
  full state remains periodic. This matches a repeatable racing lap and tests
  whether the warm-up anchor had suppressed the time optimum.
- `..._r05_r24_feasibility`: r24 start with a 2,000-iteration feasibility
  stage, targeted at r02's near-feasible residual. It ended at 0.000945
  maximum violation after 2,000 feasibility iterations, still outside the
  0.0005 feasibility threshold; it is not a time-optimized candidate.
- `..._r06_centerline_feasibility`: the original centerline reset seed did not
  improve beyond 0.484 maximum violation even after 3,000 feasibility
  iterations (the 600-iteration attempt had reached the same residual).
- `..._r07_centerline_kinematic_diag`: a better centerline state initializer
  now uses zero lateral velocity/offset, yaw from centerline curvature,
  wheelbase steering, and wheel speeds preserving the prior seed's slip. Its
  initial maximum residual is 0.814, dominated by yaw-rate collocation. This
  diagnostic did not launch a solve; the complete feasibility attempt is r09.
- `..._r08_fully_periodic_gate`: r26 start with *all* initial Frenet and motion
  states free; periodic collocation closes every state around the lap.
- `..._r09_centerline_kinematic`: the revised centerline initializer with a
  free periodic gate and 3,000 feasibility iterations. It reduced the maximum
  residual only from 0.814 to 0.587, still dominated by yaw-rate collocation;
  it did not reach time optimization and is rejected as a candidate.

At this update, r01, r04, r08, and r10 are still running. No new minimum-time
incumbent or sub-5-second OCP solution is established yet. r01 has reached a
time-stage solve and is continuing through the larger trust radii; the
intermediate stage output is not itself a candidate. Only constraint-feasible,
selected time-stage solutions count as optimization results. The r24/r26
fallback estimates do not.

The r05 residual was inspected by constraint label and units: the maximum is
0.000945 m/s² in lateral force balance; the next largest is 0.000403 m/s².
The previous 0.0005 blanket acceptance threshold therefore blocked a
numerically adequate start without violating a map or actuator bound. The NLP
acceptance tolerance is now 0.001 (reported in each optimizer report); this
does not loosen the equations or any physical/map bound. A new r10 time solve
starts from r05 under that tolerance and frees the periodic gate state. It is
running. A model tolerance of 0.001 is not being treated as proof of lap-time
accuracy; the result remains an OCP prediction under the fitted plant model.

The `--free-gate-state` option now removes all gate-state equalities, including
the old accidental `ey[0]` and `epsi[0]` pins. Periodic collocation and map
constraints remain. Candidate status strings now distinguish converged local
time solutions from feasible but nonconverged iterates. These two corrections
were made after the independent review; the reported status continues to
include the remaining local-optimum and mesh-discretization caveats.

The independent review also identified that wall bounds were imposed only at
the spatial nodes. The OCP now imposes the footprint corridor at each segment
midpoint as well, and its clearance report checks both nodes and midpoints.
This adds no margin or physics change; it closes half of each previous 0.12 m
constraint gap. Runs r01/r04/r08/r09/r10 were launched before this edit and
therefore use node-only boundaries. They remain useful time-search/basin
diagnostics, but a returned path must be re-solved with midpoint bounds before
it is called corridor-feasible under the updated formulation. A direct
interpolation check on the r26 fallback gives 0.10369 m node and 0.10382 m
midpoint clearance, so the warm-start itself satisfies the denser check; this
does not certify any of the newly optimized paths.

### Continued minimum-time search (2026-10-15, update)

The requested target is a minimum-time raceline, not controller tracking.
The final OCP stage minimizes the integrated lap-time objective with free
path geometry; no MPC tracking cost is present. The current 5.66181 s value
is only the model-consistent feasibility fallback, not an IPOPT time optimum.
The practical blocker is finding a feasible, faster local basin: time-stage
iterations have so far moved away from the feasible start and violated the
vehicle collocation constraints.

Added `--path-seed centerline-offset-only` as a new IPOPT initialization: it
places the seed path on the map centerline while preserving captured heading,
velocity, yaw, steering, wheel, and suspension states. This differs from the
earlier failed centerline seed, which replaced those dynamic states with
kinematic approximations and remained infeasible. The optimization still has
free path geometry and the same vehicle/map limits; this flag changes only
the initial point.

Added a second, geometry-distinct seed, `--path-seed curvature-outside`. It
initializes the path outward of signed track bends (clipped to a conservative
0.255 m footprint plus the requested 5 cm wall clearance), while preserving
the captured dynamic state. This is only a starting basin: the same Unity
vehicle equations and full free-geometry time objective determine the final
path. It is running as
`unity_free_geometry_time_optimization_20261015_r13_outside_bend_seed`.

The new seed was syntax-checked and launched as
`unity_free_geometry_time_optimization_20261015_r12_centerline_offset_seed`
with the full periodic gate free, 5 cm extra wall clearance, 12 m/s speed
cap, and 2,000 iterations per time stage. It is an active experiment, not a
claimed improvement. The earlier r01/r04/r08/r10 basin and gate-state solves
were started before midpoint track-boundary constraints. r01/r04 have now
completed and failed to produce feasible time-stage improvements; r08/r10
remain running. Any promising result from these older node-only runs would
need a corrected midpoint-bound re-solve before promotion. No simulator or
controller-tracking test was run for this minimum-time search.

Completed results from the first two time-search starts:

- r01 (measured gate anchored) retained a 5.67816 s feasible fallback. All
  four time stages ended at 2,000 iterations without feasibility; the final
  free-time iterate was 6.66524 s with 0.2383 maximum constraint violation.
- r04 (periodic gate experiment, started before the final gate-pin correction)
  retained the 5.66181 s feasible fallback. All four time stages were
  infeasible; the final iterate was 7.60081 s with 0.7144 maximum violation.
- r10 (r24 near-feasible warm start) retained its 5.67367 s feasibility
  fallback (initial residual 0.0009446). All four time stages were
  infeasible; its final iterate was 8.53120 s with 0.13887 maximum violation.
- r12 (centerline offset from raw capture) failed feasibility: its 0.2805
  initial residual ended at 0.2564 after 600 iterations, dominated by lateral
  position collocation and phase-length closure. It is not a candidate.

These results show why increasing IPOPT iterations alone is not a solution:
limited-memory time stages leave the feasible manifold, and large path changes
from either raw or feasible sampled states do not give IPOPT a feasible
collocation start. r13/r14/r16 are now complete failures, not candidates.
r08/r10 started from the old geometry with older node-only bounds; both have
completed and are retained only as solver diagnostics.

Because the limited-memory time stages were not consistently reducing
constraint defects, a direct exact-Hessian comparison was added. r15 used
CasADi/IPOPT exact automatic-differentiation Hessians for minimum-time stages;
no vehicle model, physics, speed, steering, or wall constraints changed. It
retained the 5.66181 s feasible fallback. Its factor-16 stage reached 5.22145
s with 0.001316 maximum residual, while the final unconstrained-radius stage
fell to 4.60516 s but diverged to 1.9819 residual. Exact Hessians did not
produce a promotable time solution; r15 is a diagnostic, not a candidate.

r13, the outside-bend seed from raw capture, has now failed feasibility with
1.726 maximum constraint violation after 600 iterations; it is rejected as a
candidate. r14 also failed at 0.2083 residual and r16 at 0.9520. The result
reinforces that a geometric seed alone is not enough unless the dynamic state
is already consistent.

The centerline-offset-only experiment was removed after both raw-capture and
model-consistent-state attempts failed feasibility (r12: 0.2564; r14: 0.2083
maximum residual). The direct curvature-outside seed also failed from both
raw capture (r13: 1.7263) and the feasible r26 state (r16: 0.9520). Together
with earlier full centerline initializations r03/r06/r09 (0.484/0.484/0.587),
these show that replacing the path offset while leaving sampled dynamic states
in place creates an infeasible collocation start. Those failed seed options
were removed from the CLI; their evidence remains recorded here.

However, completed r08 exposed a promising near-sub-5 result in its final
full-periodic minimum-time iterate: 4.8903959 s with maximum constraint
residual 0.0007348, solely lateral-force balance at one mesh node. The run
selected its 5.6618076 s feasibility fallback because its loaded acceptance
threshold was 0.0005; the time iterate itself was not written as a candidate,
and IPOPT stopped at its 2,000-iteration budget. Therefore 4.8904 s is not yet
a valid candidate or a claimed improvement. r18 is rerunning this basin with
the current 0.001 numerical tolerance, midpoint corridor constraints, and
latest-feasible continuation. Its output will determine whether this can be
recovered as a genuinely feasible time solution.

The brake-mode audit found three measured zero-throttle events on the lap; the
feasible r26 optimization had already compressed the shortest event to its
minimum 0.03 m phase length. Added `--brake-event-count` to select the longest
measured brake events and build a different phase mesh. r17 keeps the two
longer events, reinitializes the removed node's wheel speeds kinematically,
and lets the same full free-geometry time OCP optimize path, speed, and brake
phase lengths. The model/physical bounds are unchanged. This is running as
`unity_free_geometry_time_optimization_20261015_r17_two_brake_events`.

Another continuation defect was found in the time-stage loop: it restarted
each wider trust region from the lowest-time feasible point, even when an
earlier stage had reached a different feasible (but temporarily slower)
basin. The loop now preserves the latest feasible continuation point
separately from the best-time incumbent. This does not change the objective or
constraints; it prevents discarding a feasible path transition. r18 is the
same feasible r26 full-periodic start under the updated midpoint bounds, run
with this corrected continuation logic.

The r26 feasibility path was also inspected at both slow corners, not just by
its aggregate lap time. At the first right-hand bend (roughly s=11.4–14.1 m),
it runs at 3.03–3.77 m/s with path curvature about -0.61 to -0.87 1/m and
stays close to the inside/right corridor; after footprint and the 5 cm buffer,
right-side residual clearance is about 0.16–0.35 m while left-side room is
about 0.45–1.68 m. The second slow bend around s=25.4–27.0 m shows the same
inside-biased pattern (3.23–3.84 m/s, roughly -0.55 to -0.73 1/m, with much
more room on the outside/left). Thus the 12 m/s cap is not the bottleneck;
the promising geometric opportunity is to widen these turns. That evidence
motivated the curvature-outside seed. Direct geometry replacement did not
produce a feasible solution, so the next useful route-search step must preserve
dynamic feasibility while moving geometry (for example, a constrained
geometry-guidance continuation), not another raw lateral-offset seed. No
speed gain is claimed.

### Minimum-time IPOPT search — superseding status (2026-10-15)

The goal is a time-minimized raceline, not tracking. The OCP objective is the
integrated lap time with free path geometry and jointly optimized states,
steering, throttle, and brake-phase lengths. The 5.66181 s figure remains only
the feasible model-consistent fallback. IPOPT is local; a faster iterate counts
only after the dynamics/collocation residual and dense footprint corridor pass.

The best time-search evidence remains r08: 4.8903959 s with 0.0007348 maximum
residual, at a lateral-force-balance node. It used node-only wall constraints
and was not saved under the then stricter 0.0005 threshold. It is therefore
not a valid candidate, and the path between nodes was not corridor-certified.
r15 reached 5.22145 s at trust factor 16 with 0.001316 residual (mostly
heading collocation); its 4.60516 s final iterate had 1.9819 residual. Neither
is accepted. Thus the optimizer has demonstrated a faster basin, but not yet a
feasible sub-5 solution.

Current code keeps the Unity-derived vehicle model and physical limits intact.
It adds node-and-midpoint footprint corridor constraints at the requested
0.05 m clearance, exact-Hessian as an IPOPT option, and continuation that
retains the latest feasible point separately from the best-time incumbent.
Near-feasible iterates (residual <= 0.005) may seed wider stages only; output
selection still requires residual <= 0.001. Failed direct path-offset seeds
were removed after they broke collocation feasibility. Syntax and focused diff
checks pass.

Three OCP solves were started; none uses a controller or simulator:

- r17: completed with two longest measured brake events. It retained the
  5.58083 s feasible initialization, but its time stages failed: final 9.54546
  s with 4.3491 maximum constraint violation; intermediate lateral-force
  defects reached 4.82. The alternate brake phase topology is rejected.
- r18: completed from the feasible r26 periodic start with midpoint corridor,
  limited-memory Hessian, and corrected continuation. It retained the 5.66181
  s fallback. The final stage reached 5.48816 s but had 0.02974 maximum
  constraint violation, led by lateral-force balance at nodes 70 and 80. Its
  earlier stages had large lateral-force and yaw/roll collocation defects.
  The 5.48816 s iterate is not a candidate.
- r19: completed with exact Hessians. Its factor-16 stage reached 5.22145 s
  with 0.001316 maximum heading-collocation residual; the unrestricted stage
  reached 4.50660 s but had 0.01380 lateral-force-balance residual. Neither
  passes the 0.001 acceptance threshold, so r19 retained the 5.66181 s
  fallback. Completed stages took about 96–154 s, versus 480–510 s in r17/r18.
  The 0.01380 residual is in m/s² (not tire slip); it is about 0.13% of the
  feasible fallback's 10.49 m/s² peak lateral acceleration. In a 25 ms interval
  it corresponds to 0.00035 m/s lateral-velocity change if held constant.
  This is plausibly negligible but is not accepted on magnitude alone.
- r20: same feasible r26 start and exact Hessian with an 8,000-iteration
  budget per time stage. This is the targeted response to r18 stopping at
  2,000 iterations with a 0.02974 lateral-force residual; physical bounds and
  acceptance tolerances are unchanged. It is active; its first exact-Hessian
  stage took about 210 s. It also tests whether the r19 4.50660 s iterate's
  lateral-force defect can be reduced without losing the fast basin.
- r21: completed from r26 with exact Hessians, 2,000 iterations per time stage,
  and acceleration force-balance residuals divided by 10. It saved a 4.511844 s
  free-geometry trajectory with midpoint corridor constraints and
  `feasible_incumbent=true` under the scaled NLP tolerance; IPOPT stopped at
  the iteration budget, not KKT convergence. Its maximum scaled constraint
  residual is 0.000682, corresponding to 0.006820 m/s² unscaled force-balance
  residual (about 0.024 N at 3.47 kg). All other reported NLP residuals are
  smaller. Minimum extra footprint clearance beyond the requested 5 cm is
  7.2 micrometres, and minimum wheel load is 0.0506 N against a 0.05 N floor;
  this is an edge-of-envelope mathematical solution, not a robust-margin
  claim. Saved candidate:
  `live_runs/raceline_candidates/unity_free_geometry_time_optimization_20261015_r21_scaled_force_balance`.
  Because the scale factor makes the generic 0.001 tolerance equivalent to
  0.01 m/s² physically, this candidate is promising but not the final strict
  force-balance result.
- r22: warm-started from the saved r21 trajectory, now with force-balance
  equations unscaled so the 0.001 feasibility threshold remains 0.001 m/s².
  Exact Hessian, 4,000 iterations per time stage, 5 cm wall-clearance request.
  It is active.

At the latest resource check, r20 (8,000-iteration strict solve from r26) and
r22 (strict unscaled warm-start from r21) were actively using CPU, with
approximately 17 GiB memory available. Elapsed time alone is
not treated as progress. The latest feasible r26 path is inside-biased in two slow right bends (about
3.0–3.8 m/s, curvature magnitude 0.55–0.87 1/m) with more corridor remaining
outside. The speed cap is not active there; IPOPT already has free geometry,
and the unresolved task is to reach that wider-turn basin while satisfying
the complete OCP. No improved raceline is yet certified or claimed.

### Production MPC screen and corridor-aligned re-optimization (2026-10-15)

The r21 4.51184 s OCP trajectory was converted into the nine-column runtime
MPC CSV and loaded successfully by the production controller in the official
practice batch simulator. This was a real controller screen, not an offline
controller substitute. It did not complete a lap: the vehicle stopped after
about 3.56 m of physical progress, and the collision watcher stopped the run
after its 20 s no-progress guard. Collision count stayed at zero; lap count
stayed at zero, so no lap time was recorded.

The saved bag contains 893 MPC diagnostics and simulator-truth samples. The
40 Hz data path remained healthy (lidar 39.981 Hz; p95 receipt interval
26.046 ms; no gaps over 30 ms). The first rejected nonlinear rollout was
reported around candidate progress 1.96 m, at horizon stage 5; after the car
stopped, 818 subsequent nonlinear rejections were corridor failures at stage
0. The final reported state was stationary near progress 4.215 m. Localization
remained close to truth (map normal-error p95 about 0.032 m; yaw-error p95
about 0.00022 rad). The offline race report found only 65 one-step MPC/truth
matches (speed-error p95 0.180 m/s, yaw-rate-error p95 0.082 rad/s); these do
not validate a lap or the 4.51 s prediction. The failure is not a collision
and this run does not establish that the optimizer's path is driveable.

The first corridor audit exposed a mismatch: the OCP used only 0.05 m
additional wall clearance, while production MPC reserves 0.15 m plus its
heading-dependent footprint. A second source audit found that the OCP
footprint also used a 0.255 m front extent, while MPC uses 0.430 m to the
front bumper. On r21's states, the difference between the projected MPC and
OCP footprint reaches 0.1033 m at |heading error|=0.6314 rad. Therefore simply
raising the old OCP wall clearance to 0.15 m would still not make the path
consistent with production MPC. The strict force-balance refinement of r21
and the first 0.15 m-only continuation were stopped; neither would resolve
the complete corridor mismatch.

The geometry has now been made a shared contract in
`tools/racing/mpc_vehicle_geometry.py`, using the dimensions in
`f1tenth_mpc/include/mpc_types.h` (0.273 m width, 0.430 m forward extent,
0.080 m rear extent). The Unity-wheel OCP's node/midpoint footprint constraints
now use the same heading projection as MPC. The exporter preserves raw wall
distances (without subtracting a second footprint) and checks that nominal
OCP states fit the production MPC corridor before creating a runtime CSV.
Applying the corrected static check to r21 rejects it with a worst nominal
corridor slack of -0.1057 m. A focused geometry-equivalence test passes (6
cases), along with Python syntax checks for the modified modules.

The source r26 path also had a 4.6 mm OCP node interval at a discrete
brake-mode transition; directly exporting those uneven nodes yielded a
spurious 9.07 1/m curvature spike. The production export now resamples the
same periodic spline uniformly at 0.10 m, interpolating speed, acceleration,
and wall distances by route arc. On r26 this produces 287 points, reduces the
exported peak curvature to 0.868 1/m, and retains +0.01414 m nominal MPC
corridor slack (maximum body/path heading mismatch 0.12452 rad). Focused
geometry, exporter, and OCP unit tests pass (15 total). The static corridor
audit checks both MPC waypoints and segment midpoints, matching the
controller's linear reference interpolation.

The first exact-geometry fast-seed continuation (r25) exhausted the default
600-iteration feasibility phase. Its worst violation fell from 0.19653 m at
initialization to 0.13745 m, led by periodic phase-length and heading/offset
collocation constraints; no candidate was produced. This was not a completed
time-optimization stage. By contrast, the distinct r26 / 5.66 s seed passes
the corrected static MPC-corridor check with +0.01419 m minimum nominal slack
and 0.12447 rad maximum body/path heading mismatch; it is the stronger
feasible starting path. Its exact-geometry minimum-time solve is active. In
parallel, r27 restarted the fast r21 seed with the feasibility budget raised
to 4,000 iterations. It exhausted that budget without a feasible incumbent:
maximum residual was 0.13673 m, led by the periodic phase-length sum (0.13673 m)
and periodic heading collocation (0.12762 rad). The minimum-time stage did not
start, so r27 produced no candidate. The r26 continuation remains active. Both
solves use 0.15 m wall clearance, the exact MPC footprint, and exact Hessians;
the time-optimization stage budget is 4,000 iterations. These change only the
optimizer's geometric feasibility envelope, not vehicle dynamics or Unity
physics. Each trajectory must pass the static MPC-corridor check before its
collision-guarded official simulator run. Success still requires a
complete measured lap below 4.9 s without collision; no such candidate is
confirmed yet. This status supersedes the earlier notes above that list
r20/r22 as active jobs.

The first live screen of the statically corridor-compatible r26 export was
run with the production MPC on the pinned practice simulator in Xvfb
batchmode. It made substantial progress but did not complete a lap: collision
count remained zero, while the no-odometry-progress guard stopped the run at
about 25.18 m of a 28.70 m route. The bag is
`live_runs/r26_mpc_corridor_aligned_20261015/run/run_0.db3`; the read-only
report is in `live_runs/r26_mpc_corridor_aligned_20261015/analysis/`.
There were 1,089 truth/lidar samples at 39.968 Hz (receipt p95 26.94 ms),
245 optimal and 8 degraded MPC acceptances before the first sustained failure,
and zero collisions. The first nonlinear rollout rejection occurred at
candidate progress 23.43 m, failed the corridor at horizon stage 29, with
controller lateral/path error about 0.37 m. Subsequent rejected rollouts were
stage-0 corridor failures and the published command became zero throttle.
Truth-derived physical wall clearance remained at least 0.225 m during the
captured run, so this was a controller/reference-trackability failure, not a
wall collision. This candidate is not accepted as a usable raceline.

At the failure approach, the commanded reference was moving to the opposite
side of the centerline while actual lateral offset lagged it: around 5.81 s,
truth speed was 6.07 m/s and path error 0.14 m; by 6.21 s, truth speed was
5.34 m/s and path error 0.36 m; it reached roughly 0.43 m immediately before
the first rollout rejection. The MPC state and one-step prediction also
diverged from truth in this transition (full-run one-step p95 speed error
0.453 m/s and yaw-rate error 0.829 rad/s; these include post-stop samples and
are not lap metrics). This points to a dynamic/path-transition mismatch near
the final bend; it does not justify simply relaxing the MPC corridor. A
separate review of matched encoder, odometry, actuator-feedback, and bridge
timestamps is required before attributing the transition to odometry or
actuator delay.

A bounded Bayesian search is now running against the source-built Unity
practice player. It starts from an existing complete 5.71208 s lap with zero
collisions and 0.38079 m minimum map clearance; its valid-lap gate requires
zero collisions and at least 0.20 m clearance. The search varies four smooth
route sectors of steering/throttle commands and targets a measured lap below
4.9 s. This gives real Unity trajectory evidence but is open-loop: its result
must still be converted to a path and driven by the production MPC before it
can count as a usable raceline. At the latest check it had completed two of
64 permitted evaluations; the baseline remained best-valid at 5.71208 s.

### Follow-up: active odometry correction ablation and fast-seed geometry

The full r26 run was replayed through the currently built production odometry
node using only encoders, IMU, steering, and throttle. Default replay passed
recorded-`/odom` equivalence exactly for all 1,089 source-stamped outputs
(zero twist or aligned-pose difference), confirming the replayed parameters
match the active controller. The bag has one terminal bridge timing fault,
`simulator Socket.IO connection lost`, after the screen was stopped. For the
comparisons below, that terminal sample was excluded and the 1,082 contiguous
pre-fault packet samples were scored on the fixed 25 ms grid; this is a
partial live screen, not a full-lap validation.

On that same sensor history, the active production configuration scored
forward-speed RMSE 0.429 m/s and relative-position RMSE 0.302 m (macro mean
across three reset/lap-count segments). Disabling only the general
Unity-actuator residual experts, while retaining the learned residual and
the independently gated high-steer expert, changed those to 0.086 m/s and
0.066 m. Disabling the learned residual as well gave 0.066 m/s and 0.075 m.
The `learned-off / general-expert-on` ablation remained poor at 0.414 m/s and
0.294 m. Lateral-velocity RMSE (about 0.0074 m/s) and yaw rate (direct IMU,
zero difference) were unchanged. This isolates the general zero-throttle /
nonzero-throttle expert as the dominant speed/pose error on this screen; it
does not yet prove transfer beyond this run. The worst mismatch was a rejected
wheel packet at 6.109 s: truth was 5.64 m/s and the causal pre-residual
prediction was 5.56 m/s, while the zero-throttle branch published 0.87 m/s.
The general Unity-actuator expert is now gated on `wheel_update_used`, so it
cannot override the causal estimate when the current encoder packet was
rejected; the residual remains enabled on accepted wheel updates. Replay of
the same full bag with this guard scored 0.075 m/s forward-speed RMSE and
0.069 m relative-position RMSE, versus 0.429 m/s and 0.302 m before. This is
still one partial live screen, so production promotion awaits the next
collision-guarded MPC run.

The 4.512 s optimizer result from the older r21 scaled-force candidate was
re-exported with the current shared production footprint and corridor check.
That check rejects it at -0.1039 m minimum nominal MPC-corridor slack. Its
previous CSV predates the current exact geometry contract and must not be
loaded by MPC. A new exact-geometry continuation was started from that fast
seed with 6,000 feasibility and 6,000 time-optimization iterations, 0.15 m
wall clearance, exact Hessians, and a 12 m/s speed ceiling. The independent
r26 exact-geometry solve is still active; neither has produced a checked
candidate yet. The fast seed remains a search initializer only.

The bounded Unity Bayesian campaign has since completed 27 of 64 evaluations;
the best valid result is still the 5.71208 s baseline. It has not yet shown a
usable improvement. This will be reported as a neutral/negative search result
unless a later measured candidate improves on the baseline and passes the
collision and clearance gates.

### Live validation of the Odom guard and corridor margin (2026-10-15)

The bounded source-Unity BO campaign completed all 64/64 permitted
evaluations. Its best valid lap remains the unchanged 5.712084 s seed; no
proposal reached 4.9 s. This campaign optimized four coarse steering/throttle
corrections to an open-loop command profile, not MPC raceline geometry, so it
is closed as a negative result and is not a candidate trajectory.

The current source-level Odom guard was then built into the mounted
localization package and tested with the production MPC on the r26 path. Both
tests used the established pinned practice Unity batch-mode launch, started
the collision/lap watcher before Unity, and retained the closed bag and
analysis. The first one-lap screen completed without collision. A two-lap
screen then separated the startup lap from the scored steady lap:

| Run | MPC corridor margin | Warm-up lap | Scored lap | Collision delta | LiDAR rate | MPC diagnostics |
|---|---:|---:|---:|---:|---:|---|
| `r26_odom_guard_mpc_two_laps_20261015` | 0.15 m | 7.6616 s | 5.8098 s | 0 | 39.980 Hz | 558 accepted-optimal, 6 startup `missing_map_pose`, 2 nonlinear corridor rejections |
| `r26_mpc_corridor005_20261015` | 0.05 m (temporary overlay) | 7.6866 s | 5.9558 s | 0 | 39.972 Hz | 556 accepted-optimal, 6 startup `missing_map_pose`, 10 nonlinear corridor rejections |

The 0.15 m run's minimum measured body-to-edge clearance was 0.221 m.
Its two in-motion MPC rejects were both nonlinear corridor failures in the
same fast final-turn region (horizon stages 9 and 12); the best-effort speed
guard handled them and the health summary reported zero hard fallbacks. At a
0.05 m configured margin, the MPC exploited the additional corridor and then
hit the nonlinear corridor check more often (10 in-motion rejects). It was
also 0.146 s slower on the single scored lap. This paired evidence does not
support reducing the production margin; the temporary overlay remains an
experiment only and the production 0.15 m setting is unchanged. The Odom
guarded production build completed both laps with 40 Hz data and no collision,
but these tests prove safe execution, not sub-4.9 performance.

Artifacts:

- `live_runs/r26_odom_guard_mpc_20261015/run/run_0.db3` — one-lap startup screen.
- `live_runs/r26_odom_guard_mpc_two_laps_20261015/run/run_0.db3` — scored
  5.8098 s lap; detailed race metrics in its `analysis/race_metrics/` folder.
- `live_runs/r26_mpc_corridor005_20261015/run/run_0.db3` — temporary-margin
  comparison and its `analysis/race_metrics/` report.

The 4.511844 s r21 warm-start is still not an accepted path. Its IPOPT run
used a 0.05 m optimizer clearance and stopped at 2,000 iterations with a
0.000682 m/s^2 maximum lateral force-balance defect. Re-export using the
current path geometry and a 0.05 m MPC margin still found -0.0039 m nominal
corridor slack; with the production 0.15 m margin it was -0.1039 m. A
continuation using the production 0.15 m wall margin ran 6,000 iterations
without a feasible incumbent (maximum defect 0.1375, periodic states/phase
length), so no CSV from it is eligible for MPC. A separate fast-seed
continuation with the requested 0.05 m optimizer margin is still running.

The exact-geometry r26 continuation using 0.15 m wall clearance is also still
running and has not emitted a final report or eligible trajectory. Its warm
candidate's prior time-optimization stage had terminated at a CPU-time limit
with a 6.57 constraint defect; its 5.6618 s number is therefore an initial
candidate objective, not a converged optimizer result. Current live r26
evidence is a real 5.8098 s scored lap with zero collisions, not a sub-4.9
result. Next: collect the two active IPOPT outputs, reject any infeasible
iterate, export only candidates that pass the exact production corridor,
then run the production MPC under the collision watcher and score at least
one complete scored lap before considering a candidate usable. The requested
sub-4.9 raceline has not yet been achieved.

### Trackability diagnosis and bounded safer-line solve (2026-10-16)

The exact-corridor-exported 5.1531 s r26 OCP candidate was screened with the
production MPC and collided before completing its first lap. Its failure
timeline is in
`live_runs/r26_ipopt_feasible_mpc_20261015/analysis/failure_abs/`. The
failure was not caused by delivery timing or an AMCL jump: LiDAR remained at
40.004 Hz with 25.76 ms p95 receipt spacing, source-order checks found no
reversals, and localization normal error remained small during the approach.
Instead, body/path error grew from about 0.1 m to 0.66 m in roughly 0.4 s;
the production steering request reached about -0.91 normalized and feedback
about -0.45 rad while measured yaw rate stayed near -1.3 rad/s. MPC then
repeatedly rejected corridor rollouts and the car contacted the wall.

Inspection of the OCP states found the critical model/controller mismatch:
the optimized motion deliberately relied on substantial drift through the
turn. Peak body-to-optimized-path heading error was 0.146 rad (p99 0.138)
and planned lateral speed reached 0.460 m/s (p99 0.452). In the successful
5.8098 s production-MPC lap on the existing r26 path, the corresponding
measured scored-lap p99 values were only 0.055 rad and 0.0606 m/s, with
maximum path error 0.0593 m. The free-geometry OCP had no production-MPC
trackability limit on these variables; its static wall-corridor pass alone
therefore did not establish that the MPC could execute the planned drift.

The optimizer now has optional, explicitly reported constraints for absolute
body-to-path heading error and lateral body speed. The heading constraint is
computed from the Frenet path tangent, not from a proxy on centerline heading.
No Unity physics, tire parameters, production MPC configuration, or reference
trajectory was changed. A continuation from the feasible 5.1531 s candidate
is running with the exact production 0.15 m wall margin, 0.12 rad maximum
body/path heading error, 0.12 m/s maximum lateral speed, and exact IPOPT
Hessians. These limits are grounded in the successful lap's measured
trackability with approximately 2x allowance; they are a candidate-screening
constraint, not a claim that the OCP now exactly reproduces MPC behavior.
The new candidate must first remain NLP-feasible and pass the shared exact
MPC-corridor export, then it will be tested with the production controller
and immediate collision watcher.

The 0.05 m-wall-margin fast-seed continuation
`...r21_scaled_mpc_geometry_aligned_w05` is also still consuming one CPU core
(PID 936561 at the last check, about 2.6 GB RSS). It is an experimental
comparison only; it cannot replace the production-margin candidate. The
trackability-constrained continuation is a separate process. Current safe
reference remains 5.8098 s with zero collision; the requested sub-4.9 s
usable line is still unachieved.

Follow-up on feasibility: the first 0.12/0.12 constrained continuation ended
without a feasible incumbent after 4,000 phase-I iterations (maximum defect
0.3525); the feasibility objective reached its old 25.0 deviation budget.
A diagnostic-only solve from measured-capture initialization showed lateral
speed of 0.4067 m/s in the constrained bend. The default measured gate anchor
would preserve lateral speed above the new 0.12 m/s limit, so it cannot be
used for this constrained periodic problem. The optimizer now exposes its
phase-I deviation budget, and the constrained rerun leaves the periodic gate
state free. Midpoint heading/lateral-speed checks are included in this rerun.
Active output directory:
`live_runs/raceline_candidates/unity_free_geometry_time_optimization_20261016_trackable_freestate_012/`.
It starts from the finite 5.1531 s r26 candidate, uses the production 0.15 m
corridor, 0.12 rad body/path heading limit, 0.12 m/s lateral-speed limit, a
1,000 phase-I deviation budget, and 8,000 phase-I / 4,000 per-stage IPOPT
iteration budgets. No report/candidate has been emitted yet. The separate
0.05 m-margin comparison remains active and is not production-eligible unless
it passes the 0.15 m corridor exporter.

The free-gate strict solve has now exhausted 8,000 phase-I iterations without
a feasible incumbent. Its maximum NLP defect fell from 0.3396 to 0.1778, but
the feasibility objective again reached the configured trust bound (1,000).
The largest residuals are lateral-velocity collocation (0.1778) and a
remaining local lateral-speed violation of 0.0249 m/s; the car state/force
model is not being modified to hide these defects. No trajectory was written
or exported from this failed solve. A paired 0.20 rad / 0.20 m/s continuation
is running from the same seed, with the same production corridor and free
periodic gate, to test whether relaxing only the empirical trackability
envelope yields a feasible path. If it does, it still must pass static MPC
corridor validation and an immediate-abort production-MPC Unity screen.

### 2026-10-16 continuation update

The 0.20/0.20 r26-seed solve completed and failed feasibility after all 5,000
phase-I iterations. Final maximum constraint violation was 1.3232; the phase-I
trust bound was exceeded by 1.3232 and periodic forward-speed collocation
defects reached 0.1849. The initial path exceeded its 0.20 m/s lateral-speed
cap by 0.2596 m/s. This candidate is rejected; no trajectory was exported.

To preserve the fast basin rather than repeatedly restarting from r26, a
separate continuation now starts from the prior 4.5118 s r21 OCP result, while
using the corrected production MPC footprint, 0.15 m wall margin, and the
0.20 rad / 0.20 m/s trackability constraints. A second continuation starts
from the feasible r26 seed with relaxed 0.30 rad / 0.30 m/s trackability
limits. Both use a free periodic gate and exact IPOPT Hessians. They remain
optimizer experiments only until they pass feasibility, exact corridor
export, and a watched production-MPC Unity lap. The separate 0.05 m-wall-margin
experiment is not production-eligible without passing the 0.15 m exporter.

At launch of these continuations, about 19 GiB RAM was available and disk
space was 8.9 GiB; no simulator was running concurrently. The known safe live
reference remains 5.8098 s with zero collisions. No sub-4.9 s usable raceline
has yet been verified.

The 0.30/0.30 r26-seed continuation has now passed phase-I feasibility: IPOPT
reported an optimal solution with maximum NLP constraint residual below
1e-7. It has advanced to its exact-Hessian lap-time minimization stage. The
0.20/0.20 continuation from the 4.5118 s r21 seed and the 0.40/0.40
continuation from that same fast seed are still solving. Phase-I success is
not yet a candidate result; the final time-optimized report, production
corridor export, and real MPC lap are still required.

The first 0.20/0.20 fast-seed continuation (from r21, trust budget 1,000)
also exhausted 5,000 phase-I iterations. Its final defect was 0.1701 m/s in
periodic forward-speed collocation; it exceeded the lateral-speed cap by
0.0112 m/s and the trust budget by 0.0159. This is not a candidate. Because
the deviation budget was the next active limit and the local trackability
violation was already small, a follow-up from the same seed is now running
with a 5,000 trust budget and 10,000 phase-I iterations. No physics or
production controller parameter was changed.

That higher-trust 0.20/0.20 retry has also completed without a feasible
incumbent. Doubling phase-I iterations and increasing the trust budget did
not help: maximum violation rose to 0.3333, dominated by longitudinal force
balance at node 54 (0.3333 m/s^2), periodic forward-speed collocation
(0.1204 m/s), and a local lateral-speed cap miss of 0.0182 m/s. This confirms
the failure is not just an undersized trust budget; this fast-seed/trackability
combination is not being solved by the current phase-I initialization.

The initial 0.40/0.40 fast-seed continuation also failed its 5,000-iteration
feasibility stage and is rejected. The seed initially violated the corrected
right-track boundary by 0.197 m; after optimization, the largest defect was
0.0783 m/s^2 longitudinal force balance at node 221. Thus merely relaxing the
trackability envelope does not make the old r21 path feasible under the
production footprint/corridor. The 0.30/0.30 solve remains the better
continuation because it started from the production-corridor-feasible r26
candidate and already passed phase-I feasibility.

The long-running 0.05 m-margin comparison has since exited after completing
its solver stages, but could not save artifacts: its output directory was
populated concurrently with an MPC-margin overlay while the solve was running,
and the final exclusive directory creation raised `FileExistsError`. Since
that run also used a non-production 0.05 m wall margin, it is not being
repeated or treated as a usable result. Its final candidate/status was not
persisted and cannot be verified from the saved files.

Because the existing time continuation jumps directly through trust factors
4, 8, 16, and 1e6, while the first two `.30/.30` time passes left feasibility
by 0.160 and then 1.732, the optimizer now accepts a validated custom
`--time-trust-region-factors` schedule. The existing default is unchanged;
this is a solver-path option only and does not alter the plant or constraints.
The focused optimizer test suite passes 9/9, including schedule validation.
The next solve will use small increasing trust-radius steps so each accepted
time improvement can remain feasible before expanding further.

### Near-feasible fast seed and safer refinement (2026-10-16)

The corridor-only continuation from the old fast r21 seed produced a
near-feasible warm-start (not a promoted candidate): predicted lap time
4.74636 s, maximum NLP residual 0.00181784 m/s^2 at one longitudinal-force
balance node. All other listed NLP defects were below 0.00017. Export using
the exact production footprint and 0.15 m MPC corridor reported minimum
nominal slack -0.000764 m, so it falls short of the corridor by 0.76 mm and
is not a static pass. Its model also predicts peak speed 9.72 m/s, max MPC
heading error 0.1724 rad, max lateral speed 0.4977 m/s, and minimum wheel
normal load 0.05035 N (the old numerical floor). This is a valuable fast
initialization but not a safe/usable raceline.

The optimizer now allows a configurable per-wheel normal-load floor while
preserving the old 0.05 N default. The safer refinement uses 0.20 N, above the
feasible r26 comparator's 0.1375 N minimum, plus 0.17 m optimizer wall
clearance, 0.25 rad body/path heading, and 0.30 m/s lateral-speed limits. It
starts from the near-feasible 4.746 s states and uses the new gradual trust
schedule (1.1, 1.25, 1.5, 1.9, 2.5, 3.2, 4.2, 6, 9, 13, 1e6), with 10,000
phase-I and 1,500 per-time-stage iteration budgets. This is currently running.
The separate r26 `.30/.30` fine-schedule solve remains active; the original
default-schedule solve has completed as documented below. No simulator is
running.

Focused optimizer tests pass 10/10 after adding trust-schedule and wheel-load
validation. The candidate remains unverified until it reaches NLP feasibility,
exports with positive production-corridor slack, and completes a watched
production-MPC Unity run under 4.9 s without collision.

The r26 `.30/.30` fine-schedule run has passed phase-I feasibility with maximum
constraint residual 9.59e-8 and entered the first time stage. The safer
0.17 m-clearance / 0.25 rad / 0.30 m/s / 0.20 N refinement from the 4.746 s
near-feasible seed is still in its phase-I solve.

The r26 `.30/.30` run with the original large trust jumps completed. Its
phase-I point was fully feasible (maximum residual 9.59e-8), but all four
time stages failed to preserve feasibility; the stage objectives were 5.985,
6.132, 6.695, and 6.502 s. The output is therefore the feasible,
not-time-optimized fallback at 5.56039 s, with modeled peak speed 8.60 m/s,
peak lateral speed exactly 0.30 m/s, body/path heading error 0.1025 rad, and
minimum normal load 0.0500 N. This is not under 4.9 s and the near-zero wheel
load prevents calling it robust. It remains an optimizer seed only, not a
real-simulator validation.

### Watched production-MPC screen of the 4.746 s fast seed (2026-10-16)

The exported near-feasible fast seed was run once on the pinned practice
simulator in the established Xvfb `-batchmode` procedure, with the local
production MPC/Y1 controller image, practice map, and candidate trajectory.
This was a diagnostic attempt, not an acceptance run. The controller was
started before the recorder and watcher in this attempt, so the first ~12 s
are not in the bag; subsequent screens will start controller, recorder, and
watcher before launching Unity.

The watcher stopped the simulation for a 20 s odometry stall, not a collision.
The partial bag is
`live_runs/r21_corridor_only_seed_mpc_screen_20261016/run/run_0.db3`.
It contains 22.91 s / 906 LiDAR messages, no collision, no lap crossing, and
39.938 Hz receipt rate (header-stamp rate 39.984 Hz). MPC diagnostics show
907 `rejected_residual` statuses; the live node repeatedly reported that it
had no exact-feasible horizon candidate and held a bounded command. Before
the bag started, the launch log showed transient motion reaching 2.44 m/s,
then MPC rejection; no scored lap was completed.

Offline projection of the recorded portion found the car stopped at
`(-2.9099, -6.9774)` while the candidate's nearest path point was
`(-2.4235, -6.4845)`: 0.6925 m path error, with the car at +0.427 m from the
centerline and the candidate path at -0.263 m. Thus a dynamically feasible,
static-corridor-valid free path can still put the production controller on
the opposite side of the road from its planned line and cause repeated MPC
rejection. There was no collision and the fast seed is not usable.

The two active fine-trust IPOPT continuations were stopped after this direct
closed-loop evidence: the r26-seed first time stage had reached 1,500
iterations at 5.779 s objective with 0.209 maximum constraint defect; the
fast-seed safer refinement had not completed phase-I. Neither yielded a
candidate. Their prior feasibility-stage result (r26 maximum residual
9.59e-8) and the near-feasible 4.746 s iterate remain saved.

Next optimizer revision: constrain free-geometry candidates to stay within a
measured tube around the already successful production-MPC r26 path, while
still optimizing speed and allowing a bounded geometry improvement. This
addresses the demonstrated path/MPC mismatch rather than relaxing the wall
corridor or changing vehicle physics. The tube width will be an explicit
candidate parameter and included in node and segment-midpoint constraints.
Then re-export using the exact production footprint and run with recorder and
collision watcher already ready before Unity. Acceptance still requires a
complete collision-free scored lap below 4.9 s; no such candidate exists yet.

### Route-tube optimizer continuation (2026-10-16)

The route-tube geometry was mapped and checked before starting the NLP. On the
gate-rotated practice centerline, the proven r26 trajectory is 0.315 m RMS and
0.724 m maximum from centerline (it uses real available track width). The r26
OCP geometry maps back to that trajectory within 0.047 m maximum / 0.009 m RMS.
By contrast, the screened 4.746 s r21 free-path seed differs by 0.394 m
maximum / 0.228 m RMS. This supports using the known MPC route as a geometric
reference rather than treating the failed free path as a minor speed-profile
adjustment.

The optimizer now accepts a closed reference trajectory and bounded lateral
departure at both collocation nodes and segment midpoints. The export includes
per-node reference offset/deviation for audit, and the report records route
correspondence statistics. A synthetic signed-offset test and the focused
optimizer tests pass (11/11); Python compilation passes. The diagnostic-only
path exposed and fixed a missing-report-key bug.

Candidate `unity_reference_tube_time_opt_20261016_r01` ended
`Infeasible_Problem_Detected` after 4,346 phase-I iterations with 0.1502
maximum constraint defect; it produced no candidate. The cause is now
identified in the seed contract: the saved r26 OCP had `gate_state_anchor:
null` (free periodic gate), but this invocation used the default measured-gate
equality. At the first node that overwrote the warm-start state by −0.218 m/s
forward speed, +0.450 rad/s yaw rate, and +0.0469 rad steering, creating a
2.04 m/s^2 initial force-balance defect. This was an inconsistent
initialization choice, not evidence against the route-tube constraints. The
follow-up preserves `--free-gate-state`, matching the proven seed's original
OCP contract; its added constraints will first be checked against that seed
before time optimization. No Unity simulator has been launched for this solve.

With that correction, the free-gate initial-guess diagnostic has maximum
violation 0.01119 m, exclusively from the 0.15 m right-wall clearance
constraint around nodes 99–106 and 169–171. The existing r26 OCP path/export
passes the production MPC corridor with only 0.01414 m minimum nominal slack,
so the 1.12 cm miss is consistent with requiring a slightly safer inward
shift. Candidate `unity_reference_tube_time_opt_20261016_r02` is now solving
from this free-gate seed with the same 0.15 m route tube and exact optimizer
settings. It has not yet produced a feasible/time-optimized result.

### Fast-basin refinement and watched Unity result (2026-10-16)

The r02 solve completed at 5.2706925 s predicted time, with IPOPT converged
and maximum unscaled constraint defect 8.66e-9. It enforced 0.15 m additional
wall clearance, 0.20 N minimum wheel load, and a 0.15 m tube around the proven
r26 path. The production exporter accepted its footprint/corridor with
0.02906 m minimum nominal slack. This was a valid optimizer result, not a
validated driving result.

The controller, recorder, and collision watcher were all active before the
official practice simulator was started in its established Xvfb batch mode.
The run aborted at the first impact. The saved bag is
`live_runs/unity_reference_tube_mpc_screen_20261016_r02/run/run_0.db3`; it
contains 176 LiDAR messages at 39.994 Hz receipt rate, lap count remained 0,
and collision count rose 0 -> 2 at 4.225 s after first LiDAR. MPC reported
148 accepted-optimal, 3 nonlinear-rollout rejected, and 12 residual-rejected
horizons before impact.

The first clear divergence was near route progress 10.4 m. The short-horizon
speed prediction was already low by 1.47 m/s; yaw-rate error then reached
about 1.56 rad/s around 11.1 m. Truth-to-candidate lateral error grew from
centimetres to 0.69 m by 12.2 m while the requested steering saturated at
-0.524 rad. Collision followed. The 40 Hz sensor stream was healthy, so this
is a vehicle/controller operating-envelope mismatch, not a bridge cadence
failure. r02 is rejected as non-driveable and is not promoted.

Candidate `unity_fast_basin_safety_refinement_20261016_r03` is now being
re-optimized from the saved 4.512 s r21 fast basin, retaining the r26 measured
route as a 0.40 m geometry tube. It raises the added wall margin to 0.15 m,
the minimum per-wheel load to 0.20 N, and limits lateral speed/body-to-path
heading to 0.45 m/s / 0.25 rad. The unconstrained final trust stage is omitted
so a fast but constraint-violating iterate cannot replace a feasible basin.
The initial fast seed exceeded the added wall margin by 0.197 m; feasibility
restoration is therefore an explicit solver phase, not assumed. r03 remains
unvalidated until it exports with positive production-corridor slack and
completes a watched Unity lap without collision. Neither r02 nor r03 changes
the reference trajectory, Unity physics, or runtime controller code.

