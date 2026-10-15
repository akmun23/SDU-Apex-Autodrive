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

