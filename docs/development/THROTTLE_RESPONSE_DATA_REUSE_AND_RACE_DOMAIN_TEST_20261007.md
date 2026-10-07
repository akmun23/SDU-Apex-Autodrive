# Throttle-response data reuse and race-domain follow-up — 2026-10-07

## Why this follow-up was chosen

The 2026-10-06 paired-swerve study and the 2026-10-07 high-speed throttle-up
study were reprocessed together before collecting more data. This showed that
the prior empirical response model had used only the two newest training runs,
even though the workspace already contained five clean training captures and
eight clean whole-run validation captures from the earlier study. The model was
therefore rebuilt on the complete approved cohort, with all validation captures
kept out of fitting.

The practice trajectory supplies the relevant speed domain: its throttle-slew
candidate run had p50 speed 4.612 m/s, p95 7.499 m/s, and max 8.019 m/s. The
older training cohort at 4.5–7.5 m/s was concentrated at 0.30–0.42 rad
steering; low-steering data in that speed band were validation-only. This
follow-up fills that gap with positive throttle changes at 4.5/6.5/7.5 m/s
and 0.08/0.14/0.20 rad steering.

No physics, production odometry, MPC, or reference trajectory was changed.
No candidate observer or throttle policy was integrated.

## Reprocessed evidence and model results

The approved legacy bags were reanalyzed with the current analyzer. Five
training captures contributed 160 valid pairs; eight validation captures
contributed 253 governor-clean pairs. The two new 9/10 m/s training captures
and their two independent validation captures add 31 and 32 pairs,
respectively. Combined, the current model uses 191 training pairs from seven
whole captures and scores 285 validation pairs from ten separate captures.
The known partial validation bags remain excluded.

`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/empirical_response_model_all_approved_v3.json`
scores exact condition-cell lookup against a coarser speed/delta baseline:

| Held-out response | Coarse RMSE | Exact-cell RMSE | Result |
|---|---:|---:|---|
| Rear wheel/body mismatch proxy | 0.03399 m/s | 0.03684 m/s | worse |
| Forward acceleration | 0.11520 m/s² | 0.12298 m/s² | worse |
| Absolute lateral acceleration | 0.06140 m/s² | 0.03382 m/s² | better on all 10 runs; run-cluster gain CI +0.02089 to +0.03535 m/s² |
| Absolute roll | 0.001063 rad | 0.001090 rad | no improvement |
| Absolute roll rate | 0.004739 rad/s | 0.004317 rad/s | lower point estimate, but run-cluster gain CI includes zero |

The state-feature evaluation is in
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/state_model_all_approved_v3.json`.
Adding pre-stimulus lateral velocity, yaw rate, wheel mismatch, and roll did not
improve held-out wheel-mismatch or forward-acceleration prediction. It slightly
improved roll-rate prediction, while the condition-only response model already
captured the lateral-acceleration improvement. The current models therefore do
not explain the key wheel/forward-acceleration variability well enough for a
runtime throttle rule.

The response lookup now scores the recorded rear-wheel slip-ratio proxy as a
separate target, and a throttle action must agree in wheel/body mismatch, slip
ratio, and longitudinal acceleration. A separate mirror-symmetric fit pools
left/right cells within each capture, as supported by the vehicle symmetry and
checked by the paired data. With the existing cohorts plus new 4.5 and 6.5 m/s
training captures (285 valid training pairs from 11 captures), it scores 12
whole-run validation captures (333 pairs). The
4.5 m/s validation runs have since been used for mirror-symmetry model
selection, so these pooled scores are exploratory, not final confirmatory
performance. The 6.5 and 7.5 m/s validation captures in this experiment remain
sealed for the confirmatory score. Compared
with the coarse speed/delta baseline, its validation wheel mismatch RMSE is
0.02315 vs 0.02967 m/s (run-cluster gain CI +0.00145 to +0.01277), and slip-ratio
RMSE is 0.00308 vs 0.00445 (CI +0.00042 to +0.00256). Acceleration RMSE is
0.08543 vs 0.08246 m/s²; the difference is small and its CI includes zero
(−0.00658 to +0.00028). The direction-specific fit did not have a positive
wheel/slip confidence bound. Pooling also raises the supported action-sign
agreement from 28/46 (60.9%) to 29/36 (80.6%) validation cell/run observations.
This is a measurable improvement in predicting the paired slip response, not
yet an absolute movement simulator or practice-validated control change.
In leave-one-training-capture-out checks across the expanded 11-capture set,
the symmetric lookup lowers wheel mismatch and slip-ratio RMSE in 9/11 folds
(mean coarse-minus-exact gains +0.01112 m/s and +0.00224); it lowers
acceleration error in only 2/11 folds. This independently reinforces the
specific response-model gain and the unresolved acceleration dynamics.

Training-pair time histories show why one integrated response value is
insufficient. For a +0.04 throttle change with the 0.30 s ramp at 4.5 and
6.5 m/s, the 0.10–0.35 s step-minus-ramp wheel residual is about +0.14 to
+0.18 m/s and slip ratio about +0.026 to +0.027: ramping suppresses the early
wheelspin proxy. In
0.35–0.65 s these effects shrink and, for the 0.30 s ramp, reverse to about
−0.036 m/s and −0.005 to −0.007 slip ratio as the step catches up. Thus rate
and time since command both matter. The fit now retains separate early,
middle, and late outputs. In leave-one-training-capture-out checks, early
wheel-residual and slip-ratio error improves in 11/11 folds; middle improves
in 8/11 and 9/11; late worsens in 10/11. On the model-selection validation
cohort, early wheel residual RMSE is 0.0486 vs 0.0947 m/s and early slip-ratio
RMSE is 0.00692 vs 0.01444, while late wheel/slip prediction is worse. This
supports only a short transient response submodel, not unrestricted recursive
rollout. The current time-resolved artifact is
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/time_resolved_mirror_response_interim_v4p5_v6p5train2.json`.

In the earlier direction-specific v3 lookup, only three exact training action
suggestions passed the independent per-cell sign check; across its 26 supported
validation cell/run observations, 15 signs matched and 11 did not. That model
did not support a broad ramp/step rule.

Practice validation also provides no lap-time improvement so far. Existing
10-lap throttle-slew candidates were collision-free but scored means between
5.7678 and 5.7780 s, versus the established reference mean 5.7633 s; none
improved the best reference lap of 5.7018 s. Those outcomes are retained as
negative transfer evidence, not presented as progress in lap time.

## Focused race-domain experiment

The new profiles
`race_domain_swerve_throttle_rate_race_domain_train` and
`race_domain_swerve_throttle_rate_race_domain_validation` each accept one of
the practice-domain speed anchors (4.5, 6.5, 7.5 m/s). Each capture contains 24
randomized matched pairs:

- steering magnitudes 0.08, 0.14, 0.20 rad, each in both swerve directions;
- positive throttle changes of 0.04 and 0.08 normalized command;
- a 25 ms step versus a 0.15 s or 0.30 s ramp to the same final throttle;
- reset and rebuild the target speed before each paired member; and
- record odometry, simulator truth for offline scoring, IMU, encoders,
  actuator feedback/commands, collision count, packet timing, reset and phase
  events. No LiDAR/camera is recorded.

There are two independent training captures and two independent validation
captures at each speed: 12 captures, 288 matched pairs total. No validation
capture is used for fitting. The 4.5 m/s validation runs were used for
model-architecture selection and are excluded from the final confirmatory
score. The four 6.5/7.5 m/s validation captures were kept sealed until all four
completed; they are now opened only for the final score below. Each capture uses the established
pinned Explore Docker in Xvfb `-batchmode`, with Unity logs sent to `/dev/null`;
there is no GUI and no `-no-graphics` flag. The existing collision/tilt/speed,
40 Hz stream, encoder-join, phase-profile and reset-recovery checks remain
active. A capture failing any whole-run hard gate is excluded, but its raw bag
is preserved for diagnosis.

The schedule is implemented in `tools/open_plane_excitation.py`, wired through
`tools/run_open_plane_experiment.sh`, and analyzed by
`tools/analyze_throttle_slew_pairs.py`. Focused schedule checks cover all cells,
pair matching, randomization, split labels, reset budgets and the 1,200 s
capture timeout.

## Capture status

| Run | Speed | Split | Result |
|---|---:|---|---|
| `openplane_swerve_throttle_rate_racedomain_train_v4p5_r01_20261007` | 4.5 m/s | train | Complete: 24/24 valid pairs; 48/48 resets; 40 Hz gate passed; zero collisions/timing faults; both encoder joins 100%; 39.81 Hz command rate. Analysis: `race_domain_train_v4p5_r01_analysis.json`. |
| `openplane_swerve_throttle_rate_racedomain_train_v4p5_r02_20261007` | 4.5 m/s | train | Complete: 24/24 valid pairs; 48/48 resets; 40 Hz gate passed; zero collisions/timing faults; both encoder joins 100%; 39.81 Hz command rate. Analysis: `race_domain_train_v4p5_r02_analysis.json`. |
| `openplane_swerve_throttle_rate_racedomain_validation_v4p5_r01_20261007` | 4.5 m/s | validation | Complete: 24/24 valid pairs; 48/48 resets; 40 Hz gate passed; zero collisions/timing faults; both encoder joins 100%; all 48/48 actuator feedback profiles passed (mean normalized RMSE 0.00759, max 0.01622); 39.80 Hz command rate. Analysis: `race_domain_validation_v4p5_r01_analysis.json`. |
| `openplane_swerve_throttle_rate_racedomain_validation_v4p5_r02_20261007` | 4.5 m/s | validation | Complete: 24/24 valid pairs; 48/48 resets; 40 Hz gate passed; zero collisions/timing faults; encoder joins 100.000%/99.988%; all 48/48 actuator feedback profiles passed (mean normalized RMSE 0.00740, max 0.01325); 39.81 Hz command rate. Analysis: `race_domain_validation_v4p5_r02_analysis.json`. |
| `openplane_swerve_throttle_rate_racedomain_train_v6p5_r01_20261007` | 6.5 m/s | train | Complete: 23/24 valid matched pairs; 48/48 resets; 40 Hz gate passed; zero collisions/timing faults; both encoder joins 100%; 48/48 feedback profiles passed. One pair excluded for a 0.77–0.79 m/s initial rear-wheel/body residual mismatch despite matched body state/commands. Analysis: `race_domain_train_v6p5_r01_analysis.json`. |
| `openplane_swerve_throttle_rate_racedomain_train_v6p5_r02_20261007` | 6.5 m/s | train | Complete: 24/24 valid pairs; 48/48 resets; 40 Hz gate passed; zero collisions/timing faults; encoder joins 99.989%/99.989%; 48/48 feedback profiles passed; command rate 39.82 Hz. Analysis: `race_domain_train_v6p5_r02_analysis.json`. |
| `openplane_swerve_throttle_rate_racedomain_validation_v6p5_r01_20261007` | 6.5 m/s | validation | Complete: 24/24 valid pairs; 48/48 resets; 40 Hz gate passed; zero collisions/timing faults; both encoder joins 100%; 48/48 feedback profiles passed; command rate 39.82 Hz. Analysis: `race_domain_validation_v6p5_r01_analysis.json`. |
| `openplane_swerve_throttle_rate_racedomain_validation_v6p5_r02_20261007` | 6.5 m/s | validation | Complete: 24/24 valid pairs; 48/48 resets; 40 Hz gate passed; zero collisions/timing faults; both encoder joins 100%; 48/48 feedback profiles passed (max normalized RMSE 0.01623); command rate 39.82 Hz. Analysis: `race_domain_validation_v6p5_r02_analysis.json`. |
| `openplane_swerve_throttle_rate_racedomain_train_v7p5_r01_20261007` | 7.5 m/s | train | Complete: 24/24 valid pairs; 48/48 resets; 40 Hz gate passed; zero collisions/timing faults; both encoder joins 100%; all 48/48 actuator feedback profiles passed (mean normalized RMSE 0.00762, max 0.01622); 39.999 Hz throttle-command stream. Analysis: `race_domain_train_v7p5_r01_analysis.json`. |
| `openplane_swerve_throttle_rate_racedomain_train_v7p5_r02_20261007` | 7.5 m/s | train | Complete: 24/24 valid pairs; 48/48 resets; 40 Hz gate passed; zero collisions/timing faults; encoder joins 99.990%/99.990%; all 48/48 actuator feedback profiles passed (mean normalized RMSE 0.00777, max 0.01622); 39.83 Hz command rate. Analysis: `race_domain_train_v7p5_r02_analysis.json`. |
| `openplane_swerve_throttle_rate_racedomain_validation_v7p5_r01_20261007` | 7.5 m/s | validation | Complete: 24/24 matched pairs; 48/48 resets; 40 Hz gate passed; zero collisions/timing faults; encoder joins 99.990%/99.990%; all 48/48 actuator feedback profiles passed (mean normalized RMSE 0.00743, max 0.01622); 39.998 Hz command stream. Analysis: `race_domain_validation_v7p5_r01_analysis.json`. |
| `openplane_swerve_throttle_rate_racedomain_validation_v7p5_r02_20261007` | 7.5 m/s | validation | Complete: 24/24 matched pairs; 48/48 resets; 40 Hz gate passed; zero collisions/timing faults; both encoder joins 100%; all 48/48 actuator feedback profiles passed (mean normalized RMSE 0.00758, max 0.01622); 39.998 Hz command stream. Analysis: `race_domain_validation_v7p5_r02_analysis.json`. |

Across the two 4.5 m/s training captures, grouped step-minus-ramp wheel/body
mismatch effects are positive for all four throttle size/rate combinations
(about +0.015 to +0.246 m/s). The acceleration contrast is not yet stable:
for the +0.08/0.15 s profile it changes from −0.007 to +0.067 m/s² between
the two captures. These are preliminary training-only summaries, not held-out
claims or a controller recommendation.

The first two 6.5 m/s training captures strengthen the small-step result. For
the +0.04 throttle increment, ramping reduced both wheel mismatch and the
slip-ratio proxy while increasing body acceleration in 23 of 24 valid
condition/capture observations across 0.08/0.14/0.20 rad and both ramp
durations. For +0.08, the 0.30 s ramp was Pareto-favorable in 8/11 valid
observations, while the 0.15 s ramp was favorable in only 4/12. These remain
training-only signs; final 6.5 m/s validation results are reported below.

The first 7.5 m/s training replicate also shows a strong but transient
step-versus-ramp contrast for the +0.08 throttle increase: across 0.08/0.14/
0.20 rad steering, early-window wheel-mismatch effects are +0.74 to +0.79 m/s
and slip-proxy effects +0.090 to +0.099. By 0.35–0.65 s, these effects shrink
or change sign depending on steering and ramp rate. This was one capture only;
the second training capture and held-out results below determine which effects
replicate.

The two 7.5 m/s training replicates agree on the early-window sign for
step-minus-ramp wheel mismatch in all 12 mirror-pooled steering × throttle-
increment × rate cells (the step produces more mismatch) and for the slip
proxy in 12/12; the middle-window sign agrees in 11/12 for both. The early
acceleration sign agrees in 11/12 cells but is split between conditions
favoring the step and the ramp. Thus the repeated signal is regime- and
time-dependent, not a universal ramp recommendation. This is training
evidence only; the confirmatory 6.5/7.5 m/s validation results follow.

A reproducible training-only model-family comparison now uses leave-one-whole-
capture-out folds across 333 valid pairs, 168 mirror-pooled capture/cell
samples, and 13 training captures. It is stored in
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/training_model_family_loco_20261007.json`
and generated by `tools/racing/compare_throttle_response_model_families.py`.
For early wheel mismatch, the exact regime-cell lookup scored 0.04460 m/s
macro-capture RMSE versus 0.14247 for the speed/delta-only baseline and 0.04762
for RBF. For early slip ratio, it scored 0.00620 versus 0.01952 and 0.00736;
the run-bootstrap RBF-minus-exact interval was +0.00051 to +0.00182, favoring
the exact lookup. The RBF model did better for early longitudinal acceleration
(0.26515 versus 0.28486 m/s²; paired interval −0.03486 to −0.00469), while
middle-window exact and RBF scores were effectively tied. No model family wins
every output. The evidence supports using the discrete, supported lookup for
short-window wheel/slip effects, not replacing the full vehicle model with one
smooth fit. This screen is training-only; held-out results follow.

An earlier model-selection interim fitted only to the existing training cohorts plus the two
4.5 m/s race-domain training captures predicted validation r01's 24-cell
wheel-mismatch response with RMSE 0.01258 m/s, versus 0.04090 m/s for the
coarse speed/delta baseline. On independent validation r02 it scored 0.01352
versus 0.03426 m/s. This response improvement reproduces on both new runs.
Forward-acceleration RMSE did not improve: r01 was 0.10858 versus 0.10665
m/s², and r02 was 0.12603 versus 0.09868 m/s². Across all 12 held-out runs,
the wheel-response run-cluster gain CI includes zero (−0.00262 to +0.00973
m/s), while forward-acceleration prediction is significantly worse for the
exact-cell lookup (gain CI −0.01641 to −0.00164 m/s²). The lookup also
improves absolute lateral-acceleration response (gain CI +0.02326 to +0.04174
m/s²) and roll-rate response (+0.00029 to +0.00122 rad/s); these are paired
response prediction metrics, not a full plant or policy result.

Directly comparing the two independent 4.5 m/s validation runs with turn
direction kept separate, 11 of 24 tested cells show the same ramp-Pareto
direction (lower wheel/body mismatch proxy and higher body acceleration), 9
change direction between runs, and 4 are tradeoffs; none consistently favors
the step. Since steering sign should be symmetric, pooling left/right provides
an additional symmetry check: across both runs and all three tested steering
magnitudes, all six +0.04 throttle-increment × ramp-rate combinations show
lower mismatch and higher body acceleration under the ramp in each run. The
corresponding +0.08 increment combinations do not show this consistency. The
training-only direction-specific lookup marks five exact 4.5 m/s cells
validated across both validation runs; this remains narrow evidence, not a
general ramp policy. It does not justify changing the default actuator.

## Frozen confirmatory response model

The final fit uses 333 valid training pairs from 13 independent captures. The
confirmatory score uses only 96 held-out pairs from the four new 6.5/7.5 m/s
validation captures; the previously opened 4.5 m/s validation runs are not in
this score. No test/final-test rows were loaded. The artifact is
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/empirical_throttle_response_confirmatory_20261007.json`.

The exact mirror-pooled regime lookup predicts the *step-minus-ramp response*
within the supported experimental cells. It is not a full plant model, a tire
force model, or a free-running lap simulator.

| Held-out response | Exact-cell RMSE | Coarse speed/delta RMSE | Run-cluster 95% CI for coarse minus exact |
|---|---:|---:|---:|
| Integrated rear wheel/body mismatch, 0.10–1.00 s | 0.01375 m/s | 0.13327 m/s | [+0.02574, +0.21331] m/s |
| Early mismatch, 0.10–0.35 s | 0.04174 m/s | 0.20101 m/s | [+0.14681, +0.17110] m/s |
| Early wheel-slip ratio proxy, 0.10–0.35 s | 0.00603 | 0.02542 | [+0.01790, +0.02204] |
| Middle mismatch, 0.35–0.65 s | 0.03531 m/s | 0.12527 m/s | [+0.03770, +0.14222] m/s |
| Middle wheel-slip ratio proxy, 0.35–0.65 s | 0.00435 | 0.01603 | [+0.00490, +0.01847] |
| Integrated longitudinal acceleration | 0.06394 m/s² | 0.44399 m/s² | [−0.00156, +0.76166] m/s² |

Positive gains favor the exact lookup. Wheel mismatch and slip-proxy gains
replicate across all four held-out captures. The integrated acceleration
interval includes zero, so there is no overall acceleration-accuracy claim.
The larger late-window errors also show why an integrated score alone is not
enough.

The training lookup proposed dominance in 30 held-out capture/cell instances;
26 directions matched and 4 did not. Twelve exact cells passed the stricter
rule of matching in both held-out captures at that speed. One distinct,
validated condition is 6.5 m/s, 0.08 rad, +0.08 throttle, and a 0.15 s ramp
(0.533 normalized/s). In all two training and two validation captures, ramp
minus-step was better on both measured wheel/body mismatch and the slip proxy,
and also on integrated body acceleration. The held-out step-minus-ramp means
were +0.117/+0.120 m/s mismatch, +0.0173/+0.0180 slip proxy, and
−0.081/−0.033 m/s² acceleration.

An apparently attractive high-speed/high-steering rule did not transfer:
7.5 m/s, 0.14 rad, +0.08 throttle at 0.533/s was ramp-dominant in training,
but failed the direction check in both validation captures because the
acceleration contrast changed sign. It is explicitly not recommended.

## Bounded development transfer candidate

The new overlay
`config/racing/throttle_slew_6p5_lowsteer_fast_ramp_candidate.yaml` uses the
validated +0.08/0.15 s ramp rate only from 6.3–6.7 m/s and steering magnitude
0.07–0.09 rad. An optional maximum steering bound was added to the controller
configuration so this gate cannot leak into the 0.14–0.20 rad cells that show
different acceleration tradeoffs. Its default is π, preserving prior
behavior; the overlay is development-only and does not change production
defaults, physics, MPC, odometry, or the reference trajectory. Five actuator
replay tests and nine response-model tests pass.

The remaining check is one established practice-track batch transfer on the
frozen map/reference, with 12 laps (warmup + 10 scored + extra), immediate
collision abort, and lap/sector plus targeted wheel-mismatch scoring. The
overlay will be rejected if it does not improve the measured behavior; no
additional open-plane grid is planned unless this transfer identifies a
specific unsupported state/action cell.

Raw bags remain under `live_runs/openplane_*`; derived reports and models stay
under `live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007`.
Nothing is written to `/tmp` as a data destination.

## Data status and next use

All 12 planned captures are complete, converted to pair reports/surfaces, and
included according to their split. The four new validation captures remain
excluded from fitting. The fitted artifact and single narrow controller
candidate above are the usable outputs; broader ramp policies remain
unsupported. The practice transfer determines whether this candidate is
retained or rejected.
