# Swerve, throttle-slip, and regime-model progress — 2026-10-06

## 2026-10-07 high-speed frontier follow-up (latest status)

This dated addendum extends the original study; it does not replace its
training/validation split or change any physics, production odometry, MPC, or
reference trajectory. The new `race_domain_swerve_throttle_slew_frontier_validation`
profile filled the missing paired swerve/throttle-reduction region at
9.5/10.5/11.1 m/s, using steering magnitudes 0.14/0.18/0.20 rad at 9.5 and
10.5 m/s and 0.12/0.16/0.20 rad at 11.1 m/s, both turn directions, and
randomized ramp/step order. It runs one bidirectional S-curve per probe and
resets before each matched pair. At these near-frontier speeds the perturbation
is a bounded throttle reduction; an upward step would hit the existing speed
governor and would not be a clean comparison. Existing fixed-steering data
remain the evidence for upward throttle changes.

Three independent batch-mode captures are complete:

| Capture | Pairs | Quality | Note |
|---|---:|---|---|
| `frontier_validation_ext_r01_20261007` | 18/18 | clean | 54 phases, 40 Hz receive gate passed |
| `frontier_validation_ext_r02_20261007` | 17/18 usable | partial | one simulator packet gap invalidated only `v11.1_a0.160_turn+1_throttle-1`; other 17 pairs are retained |
| `frontier_validation_ext_r03_20261007` | 18/18 | clean | 54 phases, 40 Hz receive gate passed |

All three had zero collisions, zero bridge timing faults, 18/18 reset
recoveries, and 40 Hz stream rates. All 108 measured throttle-feedback
profiles passed; 107/108 command-profile checks passed, with the single
non-pass coinciding with the r02 packet-gap pair that is excluded. Aggregate
analysis uses the 53 valid matched pairs, not the incomplete pair.

The primary effect remains the paired step-minus-ramp change in mean absolute
rear wheel/body-speed residual over 0.10–1.00 s, each relative to its own
pre-stimulus median. Positive means the rapid throttle step produces more of
this longitudinal mismatch proxy than the gradual ramp. The high-speed
throttle-down results are:

| Target speed | Mean effect | Run-cluster 95% interval | Runs | Reading |
|---:|---:|---:|---:|---|
| 9.5 m/s | +0.0139 m/s | +0.0070 to +0.0191 | 3 | small positive effect |
| 10.5 m/s | +0.0400 m/s | +0.0252 to +0.0619 | 3 | repeatable positive effect |
| 11.1 m/s | +0.0164 m/s | −0.0202 to +0.0458 | 3 | inconclusive; run/turn/steering dependence is material |
| All tested high-speed cells | +0.0240 m/s | +0.0097 to +0.0382 | 3 | modest positive aggregate effect |

These are three independent run clusters, not thousands of independent
40 Hz samples. The 11.1 m/s interval spans zero, so the aggregate cannot be
turned into a universal throttle rule. Individual 11.1 m/s steering/turn cells
vary in sign, and one cell has only two valid runs because of the r02 packet
gap. A focused six-pair profile,
`race_domain_swerve_throttle_slew_11mps_replication`, repeated only the
11.1 m/s steering/direction cells in three additional independent captures.
All three completed 6/6 pairs, passed the 40 Hz gate, recovered all six resets,
and had zero collisions or bridge timing faults. Run effects were +0.0256,
+0.0420, and +0.0630 m/s; the three-run mean was +0.0435 m/s with run-cluster
95% interval [+0.0256,+0.0630]. Thus the overall 11.1 m/s throttle-down
contrast is now repeatable across these captures. At the individual
steering/turn level, five of six mean contrasts are positive; the 0.20 rad
left-turn cell is −0.0143 m/s and its interval crosses zero. This does not
support a direction-independent per-cell throttle correction.

The 11.1 m/s steering × turn estimates are preserved separately rather than
hidden in the pooled mean:

| Steering magnitude | Left-turn swerve | Right-turn swerve |
|---:|---:|---:|
| 0.12 rad | +0.0310 m/s [0.0216, 0.0394] | +0.0613 m/s [0.0267, 0.1269] |
| 0.16 rad | +0.0446 m/s [0.0071, 0.0891] | +0.0220 m/s [0.0199, 0.0231] |
| 0.20 rad | −0.0143 m/s [−0.0769, 0.0394] | +0.1165 m/s [0.0358, 0.1602] |

Values are rapid-step minus gradual-ramp residual changes; brackets are
run-cluster 95% intervals from three independent captures. The wide cells
remain uncertain and the left/right split is a reason to validate on practice,
not a justification for hard-coding a steering-direction table.

Secondary paired outcomes make the tradeoff clearer. The rapid step minus
gradual ramp change in the normalized rear-wheel longitudinal mismatch proxy
is +0.00415 (95% interval [+0.00253,+0.00594]); the residual effect is
+0.0435 m/s as above. In the first second, the rapid step also produces
0.123 m/s more forward-speed reduction than the ramp (step-minus-ramp
−0.123 m/s; 95% interval [−0.126,−0.121]). In this tested maneuver, gradual
throttle reduction therefore retains slightly more speed while reducing the
wheel/body mismatch proxy. This is a useful development-side signal for a
future smooth-throttle candidate, not a tire-force measurement or a proven
lap-time gain. It has not been inserted into the MPC or actuator path.

The bags were converted into separate, reproducible analysis products rather
than left as raw captures only:

- `frontier_validation_ext_all3_pair_analysis.json`: command/feedback profile
  checks, per-pair validity, run-cluster effects by speed and condition, and
  the packet-gap exclusion.
- `frontier_validation_ext_all3_odom_regimes.json`: sensor-sidecar u/v errors
  by measured speed and steering regime for all three runs. Overall sidecar
  u RMSE is 0.247/0.258/0.254 m/s and v RMSE is 0.0150/0.0150/0.0150 m/s.
  The 11.1 m/s steering cells are retained despite r02's one local packet gap
  because their pointwise source-stamped state/ground-truth joins remain valid;
  no temporal integration crosses that gap.
- `frontier_validation_ext_r01_r03_dataset/`: 12,599 samples in 36 reset-
  separated sequences from the two clean captures. The packet clock is fixed
  at 25 ms; r02 is excluded from sequence-based observer scoring because of
  its gap.
- `frontier_validation_ext_r01_r03_frozen_swerve_v1/report.json` and
  `..._frozen_equal_run_v3/report.json`: frozen (not refitted) observer
  comparisons on both clean whole-run captures.
- `frontier_validation_11mps_all3_pair_analysis.json`: separate run-clustered
  analysis of the three additional, all-clean 11.1 m/s captures (18/18 pairs).
- `frontier_validation_11mps_all3_odom_regimes.json` and
  `frontier_validation_11mps_all3_frozen_swerve_v1/report.json`: sidecar and
  frozen-observer scores for these same independent 11.1 m/s runs.

On these two unseen captures, frozen swerve candidate v1 predicts rear-axle
velocity with aggregate u/v RMSE 0.146/0.00224 m/s. The current Explore sensor
sidecar reports 0.247/0.0150 m/s on the same two captures. By comparison, the
equal-run v3 candidate scores 0.191/0.00836 m/s. This is a repeatable
sensor-only velocity-estimation improvement on these open-plane swerve runs,
not proof of practice-track transfer or a full-lap plant simulator. When each
reset-separated sequence is initialized at simulator truth for offline
scoring, v1's path RMSE is 0.196 m and endpoint error 0.307 m on r01, and
0.204 m / 0.317 m on r03. These short sequence metrics are not full-lap pose
drift. The candidate remains offline-only; do not integrate it into odometry
or MPC, especially given its previously poor practice-transfer result.

Across the three 11.1 m/s captures, the frozen swerve v1 observer scores
u/v RMSE 0.124/0.00209 m/s (6,160 valid prediction rows across 18
reset-separated sequences). The current Explore sidecar scores about
0.268/0.0176 m/s on those captures; equal-run v3 scores 0.227/0.0103 m/s.
The frozen v1 sequence-initialized path RMSE is 0.257/0.210/0.258 m and
endpoint error 0.383/0.298/0.399 m across the three runs. Those short
open-plane results are promising but do not establish a full-lap offline
simulator, competition-stack odometry improvement, or practice transfer. The
candidate is unchanged and remains diagnostic-only. No more broad open-plane
grid is warranted by this study: the high-speed throttle-down grid has
three-run replication, and further data should be triggered only by a concrete
practice/MPC transfer test that identifies a missing regime.

The focused replication schedule is covered by the schedule unit tests (25
tests pass), shell syntax check, Python compilation, and `git diff --check`.
All simulators and experiment containers are stopped. No experiment output is
in `/tmp`; bags and derived reports remain under `live_runs/`.

## Purpose and current status

Determine whether throttle slew changes rear-wheel/body-speed mismatch during
realistic steering reversals, measure lateral response over the already
observed speed/steering envelope, and produce evidence that can support an
offline, regime-aware plant model. Nothing from this work is integrated into
production odometry, MPC, or the raceline optimizer. Reference trajectory and
simulator physics are untouched. Test/final-test datasets remain sealed.

The targeted experiment and its independent whole-run validation cohort are
complete. Training runs r01–r05 and clean validation runs r01–r03/r06–r07 used
the pinned Explore simulator with the repository's Xvfb `-batchmode` procedure.
Every accepted run passed 96 phases, 32/32 matched pairs, 64/64 throttle
command/feedback profiles, the 40 Hz receive gate, and 32/32 reset recoveries.
Collision count stayed 0→0; there were zero bridge timing faults and 100%
encoder packet matches. Across clean captures the slowest stream rate was
39.950 Hz, fastest 39.999 Hz, and worst p95 receipt gap 25.94 ms. Each bag is
about 43 MiB; the ten clean runs total about 440 MiB. Validation attempts r04/r05
each contained 31/32 individually usable pairs but failed whole-run waveform
quality and are excluded from the clean validation cohort and its confidence
interval. They remain separately summarized as sensitivity data. No raw bag was
changed or removed.

The validation cohort confirms the training effect: step-minus-ramp mean
absolute rear-wheel/body-speed mismatch proxy is +0.1330 m/s with run-cluster
95% CI [+0.1246,+0.1449] over five independent runs. This is evidence for a
condition-specific throttle-slew effect, not proof of tire force or a production
controller improvement. The learned observer results below did not beat the
current practice odometry on the only held-out practice bag and are not
integrated.

The first offline audit exposed analysis-boundary issues, not invalid data:
the simulator reset command could be recorded on the final packet after the
phase-end event, and ramp/step steering commands could straddle their phase
event on the 40 Hz packet stream. The analyzer excludes that terminal reset
packet from the prior throttle-profile score, converts normalized steering
commands to radians, and allows only the bounded two-packet event-bracketing
offset implied by the 25 ms stream and the commanded steering-rate limit. This
was checked against r02: the sole initial pair rejection differed by 0.043–0.050
rad only during the moving swerve transitions, while both members carried the
same recorded waypoint profile and matched initial states exactly. A focused
test still rejects a materially different steering trace. Re-scoring r01–r05
passes all 32 pairs per run; no raw bag was altered.

The Explore `bridge_packet_timing` record also carries simulator rigid-body
velocity and angular-rate telemetry. For r02, its x/y linear velocity and z
angular rate exactly matched `/odom` at all 10,182 packet identities (30,546
scalar comparisons; zero difference). This establishes that the wheel/body
residual here is referenced to simulator rigid-body motion, not a wheel-derived
or learned odometry estimate. The residual remains a kinematic slip proxy—not a
tire-force or direct contact-patch measurement—and the finding is specific to
the Explore simulator capture path, not a claim about competition localization.

The encoder/body comparison initially used endpoint body speed against wheel
speed averaged over four 25 ms intervals, which could distort transient
residuals. The current analyzer trapezoidally averages each rear contact-point
body speed over the exact same four intervals before computing residual and
normalized proxy. All five bags were reprocessed with this alignment; raw
captures remain unchanged.

## Existing evidence used to choose this experiment

| Existing data | What it establishes | What it does not establish |
|---|---|---|
| Fine throttle surface: 1,508 valid reset-isolated throttle transitions, 13 fixed steering commands, two replicates per command/steering condition; 40 Hz; up to 20.8 m/s | Broad response to throttle level/slew at fixed steering, with reliable command/feedback and reset behavior. | Not a swerve, no randomized speed/steering trajectory, and no recursive movement model. Endpoint model failed to transfer to an unseen steering command (2.829 m/s RMSE, R² −0.023). |
| Fixed-steering throttle ramp/step study: 10 clean whole-run captures; 24 matched pairs per run (240 pairs, 480 probe phases); five runs at each of 4.5 and 6.5 m/s | A global step-vs-ramp change in the wheel residual was not established: +0.0043 m/s, run-bootstrap 95% CI [−0.0127, +0.0192]. There is a localized throttle-down/high-steering interaction: at 6.5 m/s and 0.42 rad, lateral-acceleration contrast was turn-signed (+0.311 left, −0.307 right m/s²); wheel-residual changes were about 0.10 m/s. | Fixed steering cannot show what happens during turn-in, unwind, or reversal. Encoder-minus-kinematic wheel speed is only a slip proxy, not tire force/slip measurement. |
| Dynamic-coupled captures: four whole-run training and two whole-run validation runs, randomized turns/reversals and throttle changes across supported speed bands | Broad dynamic steering transitions and independent whole-run validation are already available; another general lateral grid would duplicate coverage. | Existing captures do not pair a rapid throttle step against a gradual ramp during the same swerve waveform. |
| High-steering transient captures: three training and three validation whole runs; steering to 0.30/0.42 rad around a 7.5 m/s target | A repeatable high-steer operating region and causal turn-in/unwind response are present in held-out data. | No matched throttle-slew contrast during a swerve. |
| Racing baseline: two clean 10-lap runs (20 scored laps), best 5.7378 s, mean 5.7904 s | Current system is still about 0.74 s off the stated sub-5 s target. | Neither run covers the proposed high-speed/high-steering combinations; it cannot certify a model for them. |

The response-surface and plant evidence is intentionally not being collapsed
into one lateral-limit number. The learned/empirical feasible envelope varies
with speed, steering and history, and the previously fitted envelope failed
practice transfer. The separate high-steering SUBNET candidate also showed a
strong domain trade-off: it improved short high-steer rollouts but was worse
on whole practice runs; a hard switch did not improve either. Those results
argue against inserting a speed-only expert switch or an unverified limit.

One causal one-step comparison on the four dynamic-coupled training captures
and two whole-run validation captures found that adding short causal state and
actuator history changed validation RMSE as follows:

| Predicted response | Current-state features | + causal history | Change |
|---|---:|---:|---:|
| Longitudinal acceleration | 0.497 m/s² | 0.477 m/s² | 4% better |
| Lateral acceleration | 0.936 m/s² | 0.854 m/s² | 9% better |
| Yaw acceleration | 2.077 rad/s² | 1.887 rad/s² | 9% better |
| Rear wheel/body mismatch | 0.337 m/s | 0.349 m/s | 4% worse |

Simple speed-specialist and speed×steering experts mostly regressed overall;
their small high-steering improvements were not enough to offset losses in
other regions. These are one-step estimates on only two independent validation
runs, not recursive rollouts or evidence of an accurate full-lap offline
simulator. They support keeping regime conditioning as a hypothesis to test,
not promoting the current experts.

## New paired swerve protocol

Each matched pair runs the same bidirectional S-curve once with a 25 ms throttle
step and once with a 300 ms throttle ramp to the *same* final throttle. Pair
order (step/ramp) and condition order are seeded and randomized. The simulator
resets to spawn before every pair; a state-recovery gate requires matched speed,
lateral velocity, yaw rate, steering feedback, and throttle feedback before
each member. At each phase, steering turns in, unwinds through zero, reverses,
then unwinds again. A collision aborts the experiment immediately; the
existing 8-degree tilt stop, speed governor/hard stop, stream freshness and
40 Hz bridge checks remain enabled.

| Speed target | Steering magnitudes | Throttle perturbation | Pairs per run |
|---:|---:|---|---:|
| 4.5, 6.5, 7.5 m/s | 0.30, 0.42 rad | ±0.08 normalized command | 24 total |
| 9.5, 11.1 m/s | 0.08/0.14 and 0.06/0.12 rad respectively | Downward 0.08 only | 8 total |

Every row is tested in both turn directions. At the 9.5/11.1 m/s frontier an
upward step would encounter the existing 11.2 m/s governor and contaminate the
intended command profile; those high-speed conditions therefore test bounded
throttle reductions only. The existing fine throttle surface already covers
upward transitions, albeit at fixed steering. The experiment stays within the
measured steering support at each speed; it does not extrapolate 0.42 rad to
9–11 m/s.

One run has 32 pairs (64 probe phases and 32 spawn-reset speed approaches).
The schedule's conservative bound is 839.2 s before the 5 s completion margin;
the configured experiment timeout is 1,200 s. Five training and five clean
whole-capture validation runs yielded 320 accepted pairs in total, 160 per
split. Validation captures were not used for model fitting. Every accepted run
used a fresh Explore simulator start and a distinct seed.

Recorded channels are the established minimal 40 Hz set: odometry, simulator
pose/IPS for offline scoring only, IMU, encoders, steering/throttle feedback,
commands, packet timing/faults, collision count, phase events, and reset
commands. The controller is direct open-plane excitation, not the competition
controller. No LiDAR/camera is recorded. Unity log output is directed to
`/dev/null`; bags/logs remain under `live_runs/` in the workspace.

Run procedure (one capture per fresh simulator start):

```bash
SDU_APEX_SIM_TRACK=explore SDU_APEX_SIM_MODE=batchmode \
SDU_APEX_SIM_LOG_FILE=/dev/null ./tools/start_simulator.sh

SDU_APEX_EXPERIMENT_PROFILE=race_domain_swerve_throttle_slew_train \
SDU_APEX_EXPERIMENT_RUN_ID=openplane_swerve_throttle_slew_train_r01_20261006 \
SDU_APEX_EXPERIMENT_SEED=202610061 ./tools/run_open_plane_experiment.sh
```

Use the corresponding `..._validation` profile for validation runs. Stop and
restart Explore between independent captures, especially after any abort; do
not retry in a stale post-abort simulator world.

## Analysis contract

`tools/analyze_throttle_slew_pairs.py` now accepts both the legacy fixed-steer
pair bags and the new swerve captures. It verifies the recorded step/ramp
throttle commands and feedback, the same commanded steering waypoint profile
within the measured two-packet event-bracketing tolerance, matched measured
state at the throttle transition, phase validity, packet sequence continuity,
bridge timing and collision count. It reports separate results for
legacy/fixed, training, and validation profiles; it does not pool those
different maneuvers into one effect estimate.

The frozen primary outcome is the paired step-minus-ramp difference in the
time-mean absolute rear wheel/body-speed residual over 0.10–1.00 s, each
relative to its own pre-stimulus median. This covers the complete measured
transient rather than selecting a late window that could hide early wheelspin.
Positive values mean the rapid step increased the residual, so the gradual ramp
reduced this proxy. Three shorter windows remain diagnostics for response
reversals. Secondary outcomes include signed/common and left-right asymmetric wheel
residual, forward/lateral body speed, yaw rate, and IMU lateral acceleration.
Uncertainty is bootstrapped by whole run, not 40 Hz samples or repeated pairs.
The wheel residual is a proxy: do not label it tire slip or force without
independent tire-force/normal-load truth. The actuator command is explicitly
checked against its measured feedback profile; commanded steering and
measured steering feedback are kept separate.

After the captures pass run-level quality gates, fit only on the training
captures and existing approved training data. Compare at least the frozen
plain model against a causal-history/nonlinear-residual or smooth
regime-conditioned candidate. Features must separate speed, steering angle
and rate, throttle and throttle slew, body yaw/lateral motion, and wheel/body
mismatch. Score one-step acceleration and wheel responses by regime first,
then free recursive rollout on whole held-out runs. No model may read future
sensor feedback, simulator truth, or validation captures during rollout or
selection. Report longitudinal, lateral, yaw, wheel mismatch and integrated
position independently. A lower one-step error alone is not a usable offline
simulator.

Only after that holdout test shows repeatable gains may the offline plant be
used to tune a throttle policy or rank a separate candidate raceline. Any
candidate must be rechecked for causal feasibility, exact optimizer/MPC model
parity, and practice transfer. No runtime/physics change or raceline change is
authorized by this data study itself.

## Code and validation status

- New train/validation schedules and shell profile wiring are in
  `tools/open_plane_excitation.py` and `tools/run_open_plane_experiment.sh`.
- Matched-swerve analysis is in `tools/analyze_throttle_slew_pairs.py`.
- Focused schedule and pair-matching tests are in
  `tools/test_open_plane_excitation_schedule.py`.
- `python3 -m unittest tools.test_open_plane_excitation_schedule -q`: 25 tests pass,
  including two-packet event bracketing, contact-speed temporal alignment, and
  the frozen whole-response mean window.
- `tools/vehicle_dynamics_learning/evaluate_swerve_sensor_observer.py` fits a
  causal sensor-only rear-axle velocity observer diagnostic and evaluates
  whole-run, speed/steering-regime, and lap-segment errors. Its extra-dataset
  option fits only runs marked `train`; test/final-test labels are not used for
  fitting or scoring. `equal-run` weighting gives each independent training
  capture equal total mass. A fixed model seed makes reruns deterministic.
  Both modes serialize diagnostic-only checkpoints; neither is wired into
  odometry or MPC.
- `tools/vehicle_dynamics_learning/test_evaluate_swerve_sensor_observer.py`:
  7 tests pass for causal history boundaries, regime grouping, packet-clock
  integration, simulator COM→rear-axle conversion, lap segmentation, and
  equal-run weighting.
- `tools/run_explore_sensor_odometry.sh` starts only the current workspace's
  sensor odometry node, with TF disabled and output isolated on
  `/explore_sensor_odom`; it subscribes to the permitted encoder and IMU inputs
  only. `run_open_plane_experiment.sh` can now require and record that output
  plus its diagnostics with `SDU_APEX_EXPERIMENT_CAPTURE_SENSOR_ODOM=1`.
- `tools/vehicle_dynamics_learning/evaluate_recorded_odom_regimes.py` joins the
  sensor odometry output to truth by packet source stamp and scores measured
  speed × steering bins. Headerless steering feedback is joined to the matched
  truth packet by bag receipt time; receipt time is never used as a dynamics
  integration interval. Its math tests pass. Running it on the existing
  practice bag with `--odom-topic /odom` exactly reproduced the independent
  report's 3,092-sample production twist errors (u 0.11156, v 0.00984,
  yaw-rate 0.00000) and emitted per-regime metrics at
  `live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261006/production_odom_regime_reference_v2.json`.
  This one practice capture's measured production-odom cells include:

  | Speed (m/s) | Absolute steering (rad) | Samples | u RMSE (m/s) | v RMSE (m/s) |
  |---:|---:|---:|---:|---:|
  | 0–3 | 0–0.1 | 189 | 0.0687 | 0.0030 |
  | 0–3 | 0.1–0.2 | 66 | 0.1597 | 0.0071 |
  | 3–5 | 0–0.1 | 818 | 0.1344 | 0.0058 |
  | 3–5 | 0.1–0.2 | 786 | 0.0722 | 0.0129 |
  | 5–7 | 0–0.1 | 1,031 | 0.1167 | 0.0110 |
  | 7–9 | 0–0.1 | 202 | 0.1218 | 0.0075 |

  These are correlated samples from one run, not independent replicates; there
  is no practice support above 0.2 rad in this capture.
- The training-only whole-response effect is +0.1247 m/s (step minus ramp),
  with run-bootstrap 95% CI [+0.1219,+0.1275]. Clean held-out validation is
  +0.1330 m/s, CI [+0.1246,+0.1449]. By validation target speed, effects are
  +0.2063 m/s at 4.5 m/s [+0.2019,+0.2114], +0.1400 at 6.5
  [+0.1365,+0.1440], +0.1568 at 7.5 [+0.1167,+0.2194], +0.0219 at 9.5
  [+0.0125,+0.0314], and +0.0360 at 11.1 [+0.0193,+0.0521]. The global
  mean is driven primarily by the 4.5–7.5 m/s conditions.
- Per-condition validation is direction-consistent for positive throttle
  changes during high-steering swerves: step-minus-ramp proxy effects are
  about +0.25–0.27 m/s at 4.5 m/s and +0.21–0.22 at 6.5 m/s across 0.30/0.42
  rad and both turn directions. At 7.5 m/s most cells are about +0.20–0.21;
  the 0.30-rad left-turn cell is larger (+0.446 m/s) and variable (run-cluster
  CI [+0.183,+0.936]). For throttle decreases the effects are smaller, about
  +0.05–0.16 m/s at 4.5–7.5 m/s and +0.02–0.05 at 9.5–11.1 m/s. This
  supports a future measured-condition throttle-rise policy experiment, not a
  global slew limit: no controller parameter or behavior has been changed.
- The combined train/validation/practice-transfer dataset has 102,993 source
  frames and 322 packet-contiguous sequences. The approved archive contributes
  15 training runs / 511,383 training rows. Those existing data were converted
  into compact learning views and repeatable model reports instead of another
  general sweep. The training-only view is
  `live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261006/train_capture_dataset/openplane_dynamics.npz`;
  the combined comparison view is
  `live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261006/paired_teacher_dataset_with_practice_transfer/openplane_dynamics.npz`.
- Observer comparison: fitting only the five new swerve training runs gives
  about 0.075 m/s forward-speed and 0.0013 m/s lateral-speed RMSE on the five
  held-out swerve runs, but practice transfer is poor (1.320/0.0116 m/s). Adding
  the archive's 15 training runs improves practice forward speed to 0.487 m/s,
  but lateral RMSE worsens to 0.074 m/s. Equal total weight per training run
  reduces the practice error to 0.252/0.0386 m/s, while swerve held-out error
  is 0.166/0.0121 m/s. These candidates beat a deliberately simple raw
  wheel-mean/zero-lateral baseline on aggregate, but do not beat the actual
  production `/odom` in the available practice bag: production twist RMSE is
  0.1116 m/s forward and 0.0098 m/s lateral. The practice capture was used to
  compare training weights, so it is not a final holdout. The equal-run
  observer is not an odometry improvement and remains diagnostic-only. The
  deterministic reports are under `.../sensor_observer_candidate_practice_v4/`,
  `.../sensor_observer_candidate_multidomain_sample_v3/`, and
  `.../sensor_observer_candidate_equal_run_v3/` within the study directory.
- On that practice bag, production `/odom` pose after one fixed initial SE(2)
  alignment has p50/p95/max error 1.408/2.083/2.179 m over the recorded
  warmup/race/extra interval; AMCL pose p95 is 0.170 m and MPC map-pose p95 is
  0.168 m. The equal-run observer's independently integrated candidate path
  has 1.74 m RMSE and about 2.95 m endpoint error; its lap-counter segments,
  separately anchored at truth for diagnostics, average 0.39 m path RMSE and
  0.28 m endpoint error. It does not demonstrate low multi-lap drift or beat
  current production odometry.
- The largest equal-run observer errors on the practice capture occur at
  5–7 m/s near 0–0.1 rad (u RMSE 0.298 m/s), and lateral velocity at 3–5 m/s
  over 0.1–0.3 rad (v RMSE 0.049–0.056 m/s). The practice capture is one
  whole-run validation trajectory, so these cells identify where to test next;
  they do not estimate uncertainty across unseen laps.
- Historical note: the original ten throttle/swerve bags did not contain the
  sensor-only `/explore_sensor_odom` sidecar. The six 2026-10-07 frontier
  follow-up bags now do; their pointwise odometry and frozen-observer results
  are reported in the addendum above. This closes the open-plane sidecar
  capture gap, but not the practice-transfer gap: the open-plane bags do not
  contain the competition controller's production `/odom` stream, and the
  frozen observer remains diagnostic-only. Keep independent whole-run practice
  data untouched for any future transfer assessment.
- The sidecar was started alongside each fresh, pinned batch-mode Explore
  simulator for the 2026-10-07 captures and stopped after each run. Do not
  repeat the completed throttle grid; the replicated 11.1 m/s profile now has
  six total independent captures across its original and focused cohorts.
- No MPC, production odometry, AMCL, physics, or reference trajectory change
  has been made. No race lap was run in this analysis phase. The simulator and
  experiment processes are currently stopped.
