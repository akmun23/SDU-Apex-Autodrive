# Yaw-regime modelling progress — 2026-10-07

Living status and handoff for the work to model yaw response across speed and
steering regimes. Append dated updates as work proceeds; retain failures and
distinguish findings from hypotheses so an external reviewer can resume
without chat history.

## Objective and boundaries

Develop a yaw-response model from simulator-truth labels that captures
speed/steering/phase dependence, then test it on complete unseen runs and
off-grid open-plane conditions. Future truth must never be an input to a
recursive prediction. The eventual goal is a useful full-band offline plant
description, not merely a good one-step score.

The current atlas is research-only: **not integrated into MPC or odometry**,
not a sensor-only observer, and not yet a validated recursive plant or lap
simulator. Do not attribute other unrelated MPC/Y1 worktree changes to this
effort.

## At-a-glance handoff

- **Done:** two independent targeted Explore captures; audited packet timing,
  collisions, and command/feedback tracking; built a clean combined training
  dataset; refit the frozen phase-conditioned atlas as v6; ran whole-run
  validation and the focused relevant tests.
- **Done:** v6/v7 residual diagnosis, two clean paired throttle-step/ramp
  training captures, and frozen whole-run comparisons of v6–v9. Data and
  reports are retained below.
- **Finding:** v8/v9 barely change aggregate one-step yaw error; the largest
  unwind errors remain. Auditing the exact held-out sequences refined the gap:
  the difficult 8.75 m/s samples occur while a 9.5 m/s, 0.14–0.20 rad
  throttle-cut swerve decelerates through that speed—not after a steady 8.75
  m/s, 0.075 rad setup.
- **Done:** the finite training-only profile matching the measured 9.5 m/s
  frontier condition is implemented and its schedule passes the focused test
  suite (67 tests across schedule/model/data gates).
- **Done:** two clean frontier throttle-slew captures and a combined
  1,784-sample training dataset; r01/r02 stream and packet gates passed.
- **Finding:** the v10 fit made a tiny full-validation improvement but did
  not change the failing unwind cells. The new runs added no samples to those
  exact speed/steering/phase cells because their steering transient was
  mistimed relative to the rapid speed drop.
- **Done:** the retimed repeated profile passed a pilot run; measured feedback
  now populates both target unwind cells above the model's per-run sample
  minimum.
- **Done:** independent r02 also passed; combined measured-data support is 59
  and 61 samples in the two target unwind cells, across two runs.
- **Done:** v11 exact-target-cell assessment and a full-atlas nonlinear
  ExtraTrees comparison on the same whole-run validation set.
- **Finding:** the tree model materially improves one-step yaw prediction on
  supported validation samples, but has a large worst-case error and the
  validation set has now been used for model-family comparison.
- **In progress:** make the candidate reproducible/loadable, then run one new
  frozen off-grid Explore capture and score it without refitting. Keep
  recursive plant accuracy and runtime integration as separate gates.
- **Next:** if the new off-grid result transfers, test free recursive yaw
  rollout with predicted state only; otherwise diagnose the failing regimes
  from that capture. Do not integrate into MPC/odometry until these gates pass.

## Current machine/run state (2026-10-07, latest check)

- The pinned Explore simulator is running in its established batch-mode
  container. r01/r02 captures and combined dataset export are complete; the
  v11 fit and full-atlas tree sweep are complete; no analysis or recorder is
  active. The host has 16 logical CPUs, about 8.7 GiB memory available, and
  125 GiB disk free. The Explore batch container remains up; no second
  simulator or legacy experiment process is running.
- Both `openplane_yaw_fullband_gapfill_train_20261007_r01_fixed` and
  `openplane_yaw_fullband_gapfill_train_20261007_r02_fixed` completed 108/108
  phases (36 reset-isolated maneuvers each), with zero collisions, aborted
  phases, or bridge timing faults. During the maneuvers, state streams were
  about 39.96 Hz; command streams were 40 Hz. Steering feedback tracked the
  held command to 0.0001 rad median error, and held-speed error was about
  0.038 m/s median.
- r02 has one unmatched **startup-only** odometry packet: it is the first
  sample in the bag, 3.60 s before the first phase marker and before any
  reset. All samples in its 36 scored maneuvers join to packet IDs exactly;
  whole-bag matching is 9,304/9,305 = 99.989%, above the unchanged 99.9%
  threshold. No sample with a missing packet ID enters the exported sequences.
- A first combined-dataset attempt rejected r02 because its largest receipt
  gap was 128 ms, despite contiguous packet IDs and a 25.9 ms p95 gap. The
  fixed-timebase audit incorrectly counted intentional reset-separated
  sequences as packet loss and was restricted to whole-run captures. That
  audit is now corrected to check continuity *within* each retained sequence;
  rate, p95, command-gap, collision, timing-fault, and packet-join gates
  remain in force. The corrected dataset uses the exact simulator 25 ms step
  and admits r01/r02 without treating inter-reset packet jumps as losses.
- Combined training dataset:
  [`yaw_fullband_gapfill_train_r01_r02_fixedpacket_dataset`](../../live_runs/racing_model_diagnostics_20261007/yaw_fullband_gapfill_train_r01_r02_fixedpacket_dataset/manifest.json)
  — 7,865 samples, 74 packet-contiguous sequences, both captures assigned to
  `train`; both pass the clean stream/collision gate. The manifest records
  r02's single unmatched startup sample.
- Frozen-threshold v6 refit:
  [`fullband_yaw_regime_atlas_v6_gapfill`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v6_gapfill/threshold_0p0025/fullband_yaw_regime_atlas_report.json)
  — used 26 train runs and 18 whole-run validation captures; no test or
  final-test arrays were opened. Versus v5, direct-phase coverage grew from
  91.22% to 91.51% and its RMSE changed from 0.06149 to 0.06179 rad/s
  (a small aggregate regression); direct-cell coverage grew from 96.79% to
  97.18% and RMSE changed from 0.06942 to 0.06937 rad/s. Direct-phase beats
  matched persistence on all 18 validation runs; run-cluster 95% CI for mean
  RMSE difference is [−0.02784, −0.01207] rad/s. Phase-supported cells grew
  from 425 to 454, but some target cells still have zero or only 1–2
  validation transitions. This is limited support/one-step improvement, not a
  full-band accuracy claim.
- The prior off-grid final capture r03 remains spent for v5 and is not evidence
  for v6. Do not score/select v6 on it. No yaw-atlas change has been integrated
  into MPC or odometry; the current fit predicts one-step simulator-truth yaw
  rate, not recursive rollout or a sensor-only observer.
- A separate v7 diagnostic adds current steering/throttle command-minus-
  feedback features to v6. It uses the same 26 train / 18 validation runs and
  has 91.51% direct-phase coverage, 0.06137 rad/s RMSE, versus v6's 0.06179.
  Paired per-run v7−v6 RMSE change is −0.000230 rad/s (95% bootstrap CI
  [−0.000385, −0.000106], 17/18 runs); this is statistically consistent but
  practically small. Worst-cell RMSE barely changes, so v7 is a comparator,
  not a solved response.
- New finite training profile:
  `yaw_unwind_throttle_slew_train` uses 8.75 m/s, ±0.075-rad turn-in and
  unwind through ±0.025 rad, a matched 0.12 throttle cut (one-tick step versus
  0.30-s ramp), both steering directions, and two randomized repetitions.
  It has 8 reset-isolated probe conditions / 24 phases per capture; two
  separately seeded captures are planned. The schedule tests check profile
  pairing, the actual command waveform, speed/phase setup, and randomization.

## Completed work and evidence

### Fullband, phase-conditioned atlas fit

Implemented in
[`fit_fullband_yaw_regime_atlas.py`](../../tools/racing/specialists/fit_fullband_yaw_regime_atlas.py),
with focused tests in
[`test_fit_fullband_yaw_regime_atlas.py`](../../tools/racing/specialists/test_fit_fullband_yaw_regime_atlas.py).

The fit uses 0.5 m/s speed cells and 0.025 rad signed-steering cells. It fits
run-balanced robust Huber ridge models for separate turn-in, unwind, and
near-steady/low-rate phases. Current-step features predict the next 25 ms
simulator-truth yaw-rate increment; no future truth is an input. Features
include yaw state/increment, within-cell speed/steering offsets, measured
steering/rate, throttle/rate, wheel/body speed mismatch, rear lateral speed,
and body speed rate.

Baseline artifact:
[`fullband_yaw_regime_atlas_v3/fullband_yaw_regime_atlas_report.json`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v3/fullband_yaw_regime_atlas_report.json),
SHA-256 `65d5d1df1164df0097f487a0df5998fbb09ee2f618f5772140b382ab7b930d4f`.
It has 236 near-steady, 72 turn-in, and 32 unwind trained cells. The combined
source data reach 11.37 m/s and |steering|=0.5236 rad, but the report explicitly
says the full Cartesian 0–12 m/s × steering domain is **not** covered and
unsupported cells are not filled.

### New off-grid Explore check, r02

Added
[`score_yaw_atlas_openplane_holdout.py`](../../tools/racing/specialists/score_yaw_atlas_openplane_holdout.py)
to join reset-isolated maneuver labels to exact packet sequence IDs and score
only those windows. Capture `openplane_yaw_atlas_interp_holdout_20261007_r02`
contains 6 off-grid speed/steering points × 2 turn directions × 2 repetitions:
24 valid conditions, 5,658 aligned samples, zero collisions, timing faults, or
whole-bag quality failures. Sensor rate was ~39.95 Hz. Dataset manifest:
[`yaw_atlas_interpolation_holdout_dataset_r02/manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_atlas_interpolation_holdout_dataset_r02/manifest.json).

Frozen v3 score:
[`openplane_interpolation_holdout_r02.json`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v3/openplane_interpolation_holdout_r02.json).

| Predictor | Scored samples | Yaw-rate RMSE | |error| < 0.1 rad/s |
|---|---:|---:|---:|
| Persistence | 3,084 | 0.1011 rad/s | 82.5% |
| Direct phase-conditioned cells | 2,497 | 0.0662 rad/s | 94.0% |
| Bilinear phase-conditioned cells | 1,443 | 0.0687 rad/s | 93.9% |
| Direct cells, no phase | 2,960 | 0.1078 rad/s | 86.1% |
| Bilinear cells, no phase | 1,668 | 0.0661 rad/s | 94.0% |

Matched-support persistence RMSE was 0.0956 for direct-phase and 0.1023 for
bilinear-phase scoring. Phase-conditioned models improve one-step yaw on the
covered subset; non-phase direct lookup is worse than persistence. Coverage is
partial, with worst absolute error ~0.58 rad/s. This does **not** establish
recursive rollout, sensor-only operation, or lap accuracy.

r02 was opened and scored as a final-test diagnostic for v3. It is now spent:
do not use its values to fit, tune, select thresholds/checkpoints, or claim a
future candidate's blind final-test result. A new untouched final-test capture
is required after selection is frozen using training/validation only.

### Low-angle steering-rate captures

Added `yaw_low_angle_rate_surface` to
[`open_plane_excitation.py`](../../tools/open_plane_excitation.py), its runner,
and schedule tests. At each target speed it compares a one-tick steering step
with a 0.30 s ramp to the same target: magnitudes 0.05/0.075/0.10/0.125 rad,
both signs, two reset-isolated repetitions, shuffled. It augments transient
coverage at low steering; it does not cover every steering regime.

| Target | Capture | Phases | Quality/collision | Prepared train data |
|---:|---|---:|---|---|
| 4.25 m/s | `yaw_low_angle_rate_4p25_train_r01_20261007` | 96/96 | 0 / 0 | [`manifest`](../../live_runs/racing_model_diagnostics_20261007/yaw_low_angle_rate_train_4p25_r01/manifest.json) |
| 6.25 m/s | `yaw_low_angle_rate_6p25_train_r01_20261007` | 96/96 | 0 / 0 | [`manifest`](../../live_runs/racing_model_diagnostics_20261007/yaw_low_angle_rate_train_6p25_r01/manifest.json) |
| 8.25 m/s | `yaw_low_angle_rate_8p25_train_r01_20261007` | 96/96 | 0 / 0 | [`manifest`](../../live_runs/racing_model_diagnostics_20261007/yaw_low_angle_rate_train_8p25_r01/manifest.json) |

Command feedback confirmed the ramp/step contrast. At 4.25 m/s, median
requested steering slew was 3.57 vs 0.308 rad/s, feedback t90 ~0.078 vs
0.313 s, and plateau command/feedback median absolute error ~0.0001 rad. At
6.25 m/s feedback t90 was ~0.079 vs 0.317 s. These verify actuator inputs,
not a yaw-model gain by themselves.

After adding only the 4.25 m/s run, a diagnostic v4 fit reduced validation
RMSE in the supported 4.0–4.5 m/s, 0.10 rad cell from 0.136 to 0.080 rad/s
(9 validation samples; matched persistence 0.245). The neighboring 0.075 rad
cell improved only 0.236 to 0.199 (19 samples), still poor. This is narrow and
small-sample evidence, not a general accuracy claim. Artifacts are under
[`fullband_yaw_regime_atlas_v4_lowangle/`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v4_lowangle/).
The 6.25 and 8.25 m/s captures are not yet in a frozen candidate comparison.

The previously missing 10.25 m/s anchor was repeated as
`yaw_low_angle_rate_10p25_train_r02_20261007` using the same profile and
seed-shuffled reset procedure. All 96 phases completed (32 valid probes), with
zero collisions, quality failures, or bridge timing faults; the schedule ran
at 39.74 Hz. The closed bag passed `analyze_open_plane_dynamics.py`. Dataset:
[`yaw_low_angle_rate_train_10p25_r02/manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_low_angle_rate_train_10p25_r02/manifest.json)
— 8,542 exported samples, all 32 valid phases, zero simulator packet-sequence
gaps, 99.989% exact odom-to-packet joins. Command/feedback checks over the
16 step and 16 ramp probes gave median command t90 0.010 vs 0.262 s, measured
steering-feedback t90 0.074 vs 0.316 s, and matched median held speed 10.220
vs 10.217 m/s. Thus the intended actuator-rate contrast was actually applied
and measured. This run is admitted as training data.

### Four-speed refit and phase-threshold choice

After admitting the clean 4.25, 6.25, 8.25, and 10.25 m/s captures, the
fullband fitter selected 24 train runs (197,819 samples) and 18 independent
validation runs (124,377 samples). It explicitly excluded test/final-test
arrays. The two report files are under
[`fullband_yaw_regime_atlas_v5_lowangle_4speeds/`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v5_lowangle_4speeds/).

On the same whole-run validation set, phase threshold 0.0025 (units are
rad²/s, because it thresholds steering × steering-rate) is better than 0.05
for the phase-conditioned one-step model:

| Threshold | Direct-phase RMSE | Coverage | Matched-support persistence RMSE | Run wins |
|---:|---:|---:|---:|---:|
| 0.05 | 0.06561 rad/s | 92.70% | 0.08507 rad/s | 17/18 |
| 0.0025 | 0.06149 rad/s | 91.22% | 0.08700 rad/s | 18/18 |

For 0.0025, the run-cluster bootstrap 95% CI of mean per-run RMSE difference
versus persistence is [−0.02736, −0.01180] rad/s. The non-phase direct-cell
model is unchanged across thresholds: 0.06942 rad/s RMSE at 96.79% coverage,
versus matched-support persistence 0.08934 rad/s. The 0.0025 choice is frozen
as a **research one-step candidate**, based on validation only; it does not
make the map full-domain or recursively validated.

Focused checks: fitter math tests **3 passed**; open-plane schedule tests
**45 passed**. These support the analysis tooling but do not substitute for
held-out vehicle data.

### New off-grid final test, r03 — frozen v5 score

The final capture used six previously unseen speed/steering points, both turn
directions, and two reset-isolated repetitions (24 labeled conditions). Its
bag has 5,755 exact odometry/bridge-packet joins. The original 5,542-row
archive was preserved; its one incomplete phase join was traced to the generic
49-sample sequence cutoff. A separate scoring view retained short fragments
(5,576 samples, 25 contiguous sequences, all still `final_test`) and mapped
them by recorded reset epoch. All 24 phases passed the unchanged 98% integrity
gate; minimum join was 127/128 = 99.2188%. No model refit or threshold choice
used these test values.

Frozen report and score:
[`fullband_yaw_regime_atlas_report.json`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v5_lowangle_4speeds/threshold_0p0025/fullband_yaw_regime_atlas_report.json),
[`offgrid_final_r03_score.json`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v5_lowangle_4speeds/threshold_0p0025/offgrid_final_r03_score.json).
The scored model report SHA-256 is
`f7063839bb066ba8e86b7dbab9478edd881f2f3dc22366970a250b05fac0b5b9`.

| Predictor | Scored transitions | Coverage of persistence support | RMSE (rad/s) | Same-support persistence RMSE |
|---|---:|---:|---:|---:|
| Persistence | 3,077 | 100% | 0.09099 | — |
| Direct cell, phase-conditioned | 2,055 | 66.8% | 0.05464 | 0.08802 |
| Bilinear cell, phase-conditioned | 521 | 16.9% | 0.04431 | 0.10146 |
| Direct cell, no phase | 2,632 | 85.5% | 0.06983 | 0.09705 |
| Bilinear cell, no phase | 1,951 | 63.4% | 0.06775 | 0.10387 |

The candidate improves one-step yaw-rate RMSE on every predictor's matched
support. The bilinear phase result is the most accurate but only covers 16.9%
of available transitions; do not treat that selective score as full-band
accuracy. This is one capture with two within-run repeats, not an independent
run-level uncertainty estimate. It tests current-state one-step prediction
only, not recursive rollout, sensor-only observation, or MPC tracking.

At the requested operating-point windows (both signs and repetitions pooled),
the support gaps are explicit:

| Requested speed / steering magnitude | Direct-cell coverage / RMSE | Direct-phase coverage / RMSE | Bilinear-phase coverage / RMSE |
|---|---:|---:|---:|
| 4.75 m/s / 0.0625 rad | 96/96 · 0.0296 | 96/96 · 0.0323 | 0/96 · — |
| 4.75 m/s / 0.1125 rad | 91/91 · 0.0914 | 11/91 · 0.0713 | 4/91 · 0.0454 |
| 6.75 m/s / 0.0875 rad | 91/91 · 0.0311 | 91/91 · 0.0164 | 84/91 · 0.0160 |
| 8.75 m/s / 0.0625 rad | 96/96 · 0.0645 | 8/96 · 0.1052 | 0/96 · — |
| 8.75 m/s / 0.1125 rad | 0/92 · — | 0/92 · — | 0/92 · — |
| 10.75 m/s / 0.0875 rad | 0/92 · — | 0/92 · — | 0/92 · — |

Small-support RMSEs in this table are descriptive only. The frozen atlas has
no prediction at all for the last two operating points. Across the complete
24×43 possible cell grid, the current training set supports 307 unphased cells
and 425 phase-specific cell models (127 turn-in, 208 steady/low-rate, 90
unwind). Many Cartesian speed/steering combinations are physically
unreachable; this is not a command to fill those with invented values. It does
show that the measured map is not complete, and that the next useful test must
target observed high-speed gaps rather than repeat a generic broad sweep.

## Current unresolved capture

`yaw_low_angle_rate_10p25_train_r01_20261007` stopped after two complete
maneuver probes; overall schedule progress was 8/96 phases. It had no collision.
The experiment recorded a source-odometry timeout. It has **not** been exported
or admitted as training data. The clean r02 repetition means this interruption
was not reproduced; its precise trigger remains unknown and is retained as a
failed diagnostic capture, not evidence that 10.25 m/s is intrinsically
unrecordable.

Offline analysis of its bag found 865 synchronized odom/steering/encoder/IMU
samples over 24.7 s, normal p95 gaps ~26 ms, but a maximum receipt gap of
~949 ms across the whole capture. Inspection shows the three ~0.93–0.95 s
gaps occur at reset transitions, not inside the scored maneuver windows. Within
marked phases, all sensor streams are ~39.95 Hz with maximum inter-message
gaps of ~45 ms, and packet IDs are consecutive. The actual timeout was a
**trailing loss of source samples**: final packet 879 arrived at capture-relative
24.359 s; no later sensor packet was recorded, and the aborted phase closed at
24.619 s, about 260 ms later. The experiment's 250 ms source timeout therefore
fired as designed. The bridge timing-fault topic remained false, and bridge
logs contain no explicit disconnect. Cause of the bridge/output cessation is
still unknown. Preserve the bag and quarantine it. Do not salvage the two
probes for training until each reset boundary, phase label, and continuous
packet window is independently verified.

The 0.0025 s phase-threshold variant appears better on some aggregate
training/validation summaries than the v3 default, but low-angle support is
uneven. Choose thresholds only from whole-run training/validation evidence;
never use r02 to tune them.

## Next steps, in order

1. **Now:** run two separately seeded `yaw_unwind_throttle_slew_train`
   captures. Each has 8 reset-isolated step/ramp × turn-direction × repeat
   conditions (24 phases), targeted to the confirmed 8.75 m/s / ±0.025 rad
   unwind command-slew gap. Verify actual throttle feedback follows both
   profiles, phase coverage, 40 Hz packet continuity, collisions, and speed.
2. Admit only clean captures to training, add them to the combined dataset,
   then fit a candidate with current command-tracking errors and one-step
   command slew rates. Compare against v6 and v7 on the same 18 whole-run
   validation captures, including the high-error throttle-slew runs and
   run-cluster uncertainty. Do not change phase threshold based on final-test
   data.
3. If the command-slew candidate fails to improve the supported braking /
   unwind cells, analyze actuator/sensor packet alignment and candidate
   features before deciding whether another specifically targeted experiment
   is needed; do not repeat broad tests.
4. Once a one-step candidate is selected without final-test data, freeze it and
   use a small independent open-plane capture at new speed/steering values as
   `final_test`. Keep prior r03 sealed for the v5 result.
5. Evaluate recursive yaw predictions on whole held-out runs at 25/100/250/
   400/750 ms with no future truth, separated by turn-in, unwind, and steady
   regimes. The current report is a one-step evaluator; this missing evaluation
   is required before calling the atlas a useful plant teacher.
6. Continue adding data only for demonstrated reachable gaps/errors, one
   finite capture batch at a time. The present atlas still covers only a
   fraction of the 24×43 grid; unsupported cells must abstain rather than be
   invented or silently interpolated. No runtime MPC/odom integration until
   recursive and sensor-causal evidence supports it.

## Data/provenance rules

- Keep original captures and derived datasets in `live_runs/`, not `/tmp`;
  preserve failed captures for diagnosis.
- Split at whole-run level, never random rows.
- Simulator truth is an offline label only. Recursive plant prediction must
  use its own state and current command; a deployable observer must use only
  permitted sensor history.
- Keep one-step GT fit, recursive plant rollout, sensor-only observer, and
  runtime MPC evidence clearly separate.
- No production integration is justified by the evidence above.

## Append-only work log

### 2026-10-07 — initial status recorded

Recorded the fullband atlas, r02 one-step held-out result and its spent status,
three clean low-angle-rate training captures, the 10.25 m/s timeout, current
machine/run state, and ordered next steps. No production integration or new
simulator run was made while writing this document.

### 2026-10-07 — 10.25 m/s timeout localized

Read the failed bag through the existing ROS bag decoder. The apparent
~0.95 s worst gaps are between reset-separated conditions; inside every
completed phase, synchronized sensor streams remained near 40 Hz. The
experiment received consecutive bridge packet IDs through 879, then recorded
no further source packet for ~260 ms; its 250 ms timeout closed phase 8.
No collision or bridge timing-fault flag occurred. Thus this is not evidence
of a general 20 Hz feed problem or an internal packet-ID hole; it is a short
terminal cessation of the source stream. Its cause remains undetermined, and
the capture remains quarantined. No simulator was launched for this diagnosis.

### 2026-10-07 — retry, validation-only refit, and new blind-point profile

Repeated the single missing 10.25 m/s anchor in the same batch-mode Explore
procedure. All 32 reset-isolated conditions completed at 39.74 Hz; the clean
bag was exported to train and command/feedback tracking was verified. Refit
with all four new speed anchors; the 0.0025 phase threshold won on the same 18
whole validation runs (18/18 per-run wins against persistence). No final-test
array was used for this choice.

Added a separate `yaw_atlas_offgrid_final` profile with six speed/steering
points distinct from the already opened r02 points, both turn directions and
two reset-isolated repeats. Its schedule test and profile parsing pass (46
schedule tests total); Bash syntax and Python compilation checks pass. The
model is frozen for this single final-test score.

Completed the final capture in the established Explore batch-mode procedure:
72/72 phases, 24/24 valid conditions, 0 collisions, 0 quality failures, 0
timing faults, and 39.71 Hz command-loop rate. The simulator was stopped after
capture. The dataset was explicitly assigned `final_test` before scoring.
The frozen v5 score now completes using the split-identical scoring view; all
24 phase joins pass the unchanged 98% gate (minimum 99.2188%). This is a
current-state one-step score only, and r03 is now spent for v5.

Diagnosis: the existing final-test archive contains 5,542 samples in 24 reset
sequences, while its raw bag has 5,755 exact odometry/packet joins. For phase
`atlas_r02_v10.75_a0.0875_turn+1`, the archive ends at packet 1030 although
the raw phase continues through packet 1068. Packet 1031 lacks the encoder
window needed by the generic feature row; that breaks the row sequence, and
non-coalesced preparation drops the remaining 37-packet fragment because it
is shorter than the generic 49-sample minimum for history plus rollout. This
is an export eligibility/truncation issue, not missing simulator packets or
an unclean run. The original archive remains unchanged; the separate score
view retains the valid fragments and joins them by reset epoch.

The final off-grid score exposed precise training gaps: direct-phase support
was absent at 8.75 m/s / 0.1125 rad and 10.75 m/s / 0.0875 rad; the
4.75 m/s / 0.1125 rad window had only 11/91 direct-phase predictions. A
focused `yaw_fullband_gapfill_train` profile samples exact local steering-cell
centers at 4.75, 8.75, and 10.75 m/s, both signs, two randomized within-run
repetitions, and reset before every condition. It covers 36 conditions / 108
phases per capture. The new schedule adds two checks (48 schedule tests pass);
Python compilation, shell syntax, and profile registration pass.

The first launch failed before any phase because the new profile was missing
from the Python CLI choices. Its partial 299 kB bag and logs were moved to
[`live_runs/failed_experiments/openplane_yaw_fullband_gapfill_train_20261007_r01_cli_registration_failure/`](../../live_runs/failed_experiments/openplane_yaw_fullband_gapfill_train_20261007_r01_cli_registration_failure/)
so default training-bag discovery cannot see them. After adding the CLI choice,
the successful r01 capture completed 108/108 phases (36 valid probes, zero
invalid), schedule complete, zero collision-count changes, zero bridge timing
faults, and 39.73 Hz command output. Exact odom-to-packet alignment was
9,292/9,292. Within-phase odom, steering, both encoders, IMU, and packet-timing
streams all measured 39.967 Hz, p95 receipt gap 25.82–25.90 ms, max
46.03–46.13 ms. The whole-bag analyzer reports 34.919 Hz receipt rate / ~952
ms max gap because it includes intentional reset boundaries; the phase-scoped
fixed-step audit confirms uninterrupted data intervals remain near 40 Hz. On
probe plateaus, steering-feedback error was 0.0001 rad median / 0.0003 p95,
and GT speed error 0.038 m/s median / 0.058 p95. Peak measured steering was
0.1251 rad. This is a clean training capture, not a validation/test result.

The independent r02 capture, seed `202610075`, has started. Next: verify its
quality and actual actuator/speed tracking; prepare one combined training
dataset from only clean r01/r02 bags; refit with the frozen model family and
threshold; assess whole-run validation; then create a new unopened off-grid
final test for any selected successor. r03 must not be reused to select or
validate that successor.

Resource cleanup finding: the idle Explore Unity log had grown to 2,958,347,585
bytes from repeated connection-refused messages while no experiment bridge was
running. The simulator process was stopped; its disposable `--rm` container
and that generated log were removed. The relevant evidence (log purpose/size,
no bridge, and the captured sensor-stop diagnosis) is retained here. The
simulator was restarted for the 10.25 m/s capture with the same pinned image,
batch mode and Xvfb procedure, but `SDU_APEX_SIM_LOG_FILE=/dev/null`; after
the capture it was stopped again. No `/tmp` dataset or model artifact was
created.

### 2026-10-07 — r02 audit, fixed-step export, and v6 refit

The independently seeded r02 bag completed all 108 phases / 36 valid
reset-isolated maneuvers. Capture summary: `schedule complete`, not aborted,
zero invalid phases, zero collision-count change, zero bridge timing faults.
Phase-window delivery was 39.955–39.956 Hz for odometry, steering, throttle,
encoders, IMU, and packet-timing; p95 receipt gaps were 25.83–25.92 ms. The
largest per-phase receipt gap was 125–129 ms, but packet identity localized
the only failed odometry join to bag sample 0: 3.605 s before the first phase
marker and before the first reset. There were no unmatched samples in any
returned maneuver sequence; the 9,304/9,305 whole-bag join remains above the
existing 99.9% threshold. This is an extra startup sample, not an internal
missing transition.

The phase-scoped actuator audit covered 36 maneuver sequences and 72 plateau
windows per run, 1,366 samples in r01 and 1,363 in r02. Both runs had median
window steering-command error 0.0001 rad (95th percentile across windows
0.0003 rad); median window speed error was 0.0380 and 0.0381 m/s respectively.
The second run therefore independently reproduced the intended steering and
speed conditions.

The first combined export attempt used the ordinary receipt-gap ceiling and
correctly refused r02: it reported 128 ms maximum receipt gap even though p95
was 25.9 ms and packet IDs were contiguous inside each scored maneuver. The
fixed-timebase path was also incorrectly limited to a single continuous
whole-run capture and counted packet jumps across intentional reset-separated
sequences. Corrected `prepare_dataset.py` to count discontinuities within each
retained sequence and allow the explicit fixed 25 ms timebase for these
reset-separated captures. It still requires contiguous IDs within every
exported sequence, the existing minimum rate and p95-gap gates, command-gap
limits, collision/timing checks, and the existing ≥99.9% join rule. The first,
rejected export is not used; the successful archive is
[`yaw_fullband_gapfill_train_r01_r02_fixedpacket_dataset`](../../live_runs/racing_model_diagnostics_20261007/yaw_fullband_gapfill_train_r01_r02_fixedpacket_dataset/manifest.json),
with 7,865 samples in 74 packet-contiguous sequences and both clean runs
assigned to training. Manifest quality failures are empty for both runs.

Refit the frozen Huber-ridge atlas family at the already-selected 0.0025
phase threshold. The fitter selected 26 train runs and 18 distinct whole-run
validation captures. It excludes test/final-test arrays before reading their
values. v6 artifact SHA-256:
`4eafe5dcb660b66a9cc42330c85645a15d9ce8e1e956b1bde91d3e57c595f9d4`.
Validation comparison against v5:

| Model / predictor | Coverage | RMSE (rad/s) | Matched persistence RMSE | Run wins |
|---|---:|---:|---:|---:|
| v5 direct phase | 91.22% | 0.06149 | 0.08700 | 18/18 |
| v6 direct phase | 91.51% | 0.06179 | 0.08758 | 18/18 |
| v5 direct cell | 96.79% | 0.06942 | 0.08934 | 17/18 |
| v6 direct cell | 97.18% | 0.06937 | 0.08934 | 17/18 |

The v6 direct-phase aggregate RMSE is very slightly worse, despite +0.29
percentage points of coverage; its run-cluster 95% CI versus matched
persistence remains below zero at [−0.02784, −0.01207] rad/s. Direct-cell
RMSE changes by only −0.00005 rad/s. Local support improved: unphased cells
307→318 and phase-conditioned cells 425→454 (turn-in 127→135, unwind 90→95,
steady 208→224). This is a modest support gain, not evidence that the whole
0–12 m/s range is solved. Several speed/steering/phase cells remain unsupported
or have only 1–2 validation transitions, and the 4.75 m/s high-angle cells
still show roughly 0.12 rad/s direct-phase RMSE.

`test_prepare_dataset.py` (9 tests), `test_fit_fullband_yaw_regime_atlas.py`
(3 tests), and `test_open_plane_excitation_schedule.py` (48 tests) all pass.
The simulator was stopped after capture; latest process check found no
simulator or model-fit process. No final-test values were used for the v6 fit.

The first cell-level v6 review found 232 speed/steering cells with at least 15
phase-conditioned validation transitions. Direct phase models beat matched
persistence in 199/232 cells and lose in 33; the largest per-cell RMSE loss is
about 0.008 rad/s. This does not mean the absolute error is already small:
notable high-error cells include 6.25 m/s / −0.075 rad (0.277 rad/s RMSE, 15
transitions), 8.75 m/s / −0.025 rad (0.194, 40), 8.75 m/s / +0.025 rad
(0.178, 46), and 7.25 m/s / −0.075 rad (0.164, 167). The latter cells have
substantial training-run support, so simply adding more samples is not yet a
justified fix; the next diagnosis should inspect residual structure and phase
classification there. Separately, 5,616/123,585 validation transitions have
no direct phase prediction; high-speed regions include validation-observed
cells with no matching phase-trained model. No new simulator capture has been
started for this audit because existing validation already measures the
largest-error cells. A targeted capture is justified only for confirmed
reachable cells lacking training support or to independently confirm a
specific residual hypothesis.

Follow-up residual decomposition for the largest cells found the error is not
uniform across all independent runs. At 8.75 m/s / −0.025 rad during steering
unwind, the 22 validation transitions score 0.257 rad/s RMSE; the coupled
dynamics run contributes 0.083 on five transitions, while three throttle-slew
validation captures contribute 0.228–0.345 on their 5–7 transitions each.
At +0.025 rad unwind, the 35 transitions score 0.203 overall; two coupled
dynamics runs contribute 0.017 and 0.032 on 17 transitions, while the
throttle-slew runs contribute 0.190–0.353 on 18 transitions. Phase-specific
training support is only 2 runs / 95 samples at the negative cell and 5 runs /
122 samples at the positive cell, despite the broader unphased cells having
more runs. The validation wheel/body speed-mismatch ranges overlap the training
ranges, so a simple “outside the observed slip range” explanation is not
supported. At 7.25 m/s / −0.075 rad in phase 0, both coupled-dynamics runs
have large error (0.245–0.302 RMSE; 48 transitions total), and this phase model
has only 2 training runs / 45 samples. Current evidence therefore points to
phase-specific support and/or missing response history as hypotheses; it does
not prove throttle slew itself causes the yaw error. Follow-up offline checks
compared command tracking, actual actuator slew, wheel mismatch, speed rate,
and phase-specific train/validation support before deciding what experiment,
if any, was justified.

### 2026-10-07 — causal command feature trial and targeted paired profile

Confirmed the atlas input omission: base features include measured steering,
throttle feedback/rates, wheel mismatch, and speed rate, but not the current
steering/throttle commands. The prepared frame already stores both commands,
so an offline-only diagnostic v7 adds command-minus-feedback for steering and
throttle. It does not alter MPC, odometry, or simulator behavior.

The hypothesis was that pending actuator response helps predict yaw during
rapid command changes. Validation confirms these features correlate with
residuals, but the first candidate does not materially fix the high-error
cells. v7 keeps the same 26 train / 18 validation runs and 91.51% direct-phase
coverage; RMSE is 0.06137 rad/s versus 0.06179 for v6. On paired whole-run
RMSE, v7−v6 is −0.000230 rad/s (95% bootstrap CI [−0.000385, −0.000106],
17/18 run wins). The high-error 8.75 m/s / ±0.025 rad cells change only from
0.19448 to 0.19418 and 0.17821 to 0.17765 rad/s. Treat this as a small
diagnostic gain, not the missing response model. v7 report SHA-256:
`e0d7ab37fb823d50292782500fd2c7fe03ac32e19d7d98b71c6ff56d40ff34f5`.

The training/validation comparison identifies a specific missing combination:
in the 8.75 m/s / −0.025 rad unwind specialist, training has 2 run clusters /
95 samples and throttle-command rate is usually near zero (90th percentile
0); the throttle-slew validation runs contain one-tick command changes near
5.0 normalized units/s, with measured feedback lagging. At the +0.025 rad
unwind specialist, training has 5 run clusters / 122 samples, still with far
fewer such steps than validation. Thus the current data establish an input
distribution gap for command slew in the relevant phase; they do not establish
that command slew alone explains the yaw error.

Added `yaw_unwind_throttle_slew_train` as a finite training-only profile: at
8.75 m/s, steer ±0.075 rad and unwind through ±0.025 rad; pair the same 0.12
throttle reduction as a one-tick step versus 0.30 s ramp; randomize both turn
directions and two repetitions. Each capture contains 8 reset-isolated probes
(24 total phases including approach/settle); two distinct seeds/run IDs provide
two independent run clusters. The initial schedule used 0.350→0.230 throttle;
the subsequent failed matched-start attempt and evidence-based correction are
recorded below. This capture is justified by the demonstrated phase-specific
training gap, not by a generic desire for more data.

Focused verification before launch: schedule tests **51 passed**, yaw-atlas
math tests **4 passed**, dataset quality tests **9 passed**, Python compilation
passed, shell syntax passed. Pre-run check at 15:29 showed no running sim or
experiment, load 0.40 / 0.97 / 2.34, and 125 GB available. Next update must
record each capture's real throttle feedback, packet/40 Hz quality, collisions,
whether the model fit admits it, and the v8 whole-run validation comparison.

### 2026-10-07 — paired-profile launch diagnosis and correction

The first launch failed before any maneuver because the new profile name was
missing from the Python CLI `choices`. Its 328 KB preflight bag is preserved at
[`failed_experiments/openplane_yaw_unwind_throttle_slew_train_20261007_r01_cli_registration_failure`](../../live_runs/failed_experiments/openplane_yaw_unwind_throttle_slew_train_20261007_r01_cli_registration_failure/experiment.log).
The CLI choice was added, help output confirmed the profile, and the schedule,
atlas, and dataset checks passed (64 combined tests).

The next launch reached its initial reset and speed approach but stopped
before the first scored probe: 2/24 setup phases, 39.87 Hz commands, zero
quality failures, and no collision. Its bag showed speed settling at
8.493 m/s while the probe required 8.75 ±0.20 m/s, so the strict start gate
correctly rejected it. This was a setup-control mismatch, not yaw-model
evidence; none of those samples enter training. The failed 1.7 MB capture is
preserved at
[`failed_experiments/openplane_yaw_unwind_throttle_slew_train_20261007_r01_matched_start_gate_failure`](../../live_runs/failed_experiments/openplane_yaw_unwind_throttle_slew_train_20261007_r01_matched_start_gate_failure/experiment.log).

The cause is confirmed in the controller path: while waiting for a matched
start, `slew_probe` used generic speed feedforward rather than the high-speed
race-domain feedforward. The preceding clean captures show the relevant
8.75 m/s bin at 8.725–8.766 m/s with throttle feedback 0.360–0.363 (median
0.361); the failed attempt held 0.350 and fell below the bin. The schedule now
starts at measured 0.361, ends at 0.241 (same 0.120 decrement), and explicitly
uses race-domain feedforward during matched-start settling. The original
speed/yaw/steering/feedback gates are unchanged. Schedule tests assert this
profile setting; after the correction, all 51 schedule tests pass, Python
compilation passes, and shell syntax passes.

The fresh r01 capture completed all 24/24 phases (8 probes plus their
approach/settle phases), with zero collisions, zero timing faults, and zero
harness quality failures. The dataset quality exporter admitted all 8 probes
and exported 1,280 samples. Exact packet alignment was 2,661/2,661; packet
sequence gap count was zero; active-stream rates were 39.956 Hz for odometry,
39.955 Hz for throttle feedback, 39.953 Hz for steering feedback, and 39.999 Hz
for both command streams. Odom p95 inter-message gap was 25.855 ms. The
bag-wide maximum receipt gap is higher because of reset/transition boundaries;
the fixed packet timebase exporter segmented those boundaries and found no
within-sequence packet loss.

Measured throttle feedback followed the intended pair: all probes began at
0.361; step commands changed to 0.241 in one command tick, while ramp commands
traversed the same 0.120 reduction over 0.30 s; feedback reached 0.241 in each
case. The observed speed then falls during the throttle cut, as expected; the
experiment deliberately disables speed-error rejection for these response
probes. This capture supplies training data only, not validation evidence.

Current status after r01: the pinned Explore simulator remains in the
established batch-mode container and no experiment is active. The corrected
profile is now empirically verified at the command/feedback and packet-rate
level. Next: run independent r02 with a different seed, then combine and fit
only if it passes the same gates. No yaw model or runtime MPC/odometry code has
been changed or promoted.

### 2026-10-07 — r02, v8/v9 analysis, and narrowed support gap

The independent `openplane_yaw_unwind_throttle_slew_train_20261007_r02_matched`
capture also completed all 24 phases and all 8 probes. It had zero collisions,
zero timing faults, zero invalid phases, and no harness quality failures. The
combined clean train archive is
[`yaw_unwind_throttle_slew_train_r01_r02_matched_dataset`](../../live_runs/racing_model_diagnostics_20261007/yaw_unwind_throttle_slew_train_r01_r02_matched_dataset/manifest.json):
2,563 samples in 16 reset-separated probe sequences, both runs explicitly
assigned `train`. Exact packet alignment is 2,661/2,661 and 2,682/2,682;
within-sequence packet-gap count is zero. Odom rates are 39.956 and 39.950 Hz;
command rates are 39.999 and 39.997 Hz; odom p95 receipt gaps are 25.855 and
25.875 ms. The manifest's whole-bag clean collision/timing gate passes both.

The paired intervention was executed as intended. Across 8 step/ramp pairs,
the median starting-speed difference was about 0.00001 m/s. During 0.75–1.0 s
after the stimulus, signed yaw rate in the step condition was lower by a mean
0.01979 rad/s and speed lower by 0.3994 m/s; during 1.0–1.5 s those differences
were −0.00689 rad/s and −0.0825 m/s. This is a repeatable descriptive response,
but it does not isolate a yaw-only causal term because the step also changes
speed; only two independent capture runs exist, so do not claim broad
statistical generalization from these eight within-run pairs.

Expanded training v7 (command-error features, no command-rate feature) and v8
(same plus command rates) each use 28 train / 18 whole-run validation captures;
the validation captures and split are unchanged. v8 direct-phase one-step
metrics: 91.593% coverage, RMSE 0.061387 rad/s, MAE 0.012283 rad/s, p95
absolute error 0.051232 rad/s, and 97.275% of supported transitions below
0.1 rad/s. Compared with expanded-train v7, run-macro RMSE changes by
−0.000128 rad/s (95% paired run-cluster bootstrap CI [−0.000272, −0.000016],
13 wins / 5 losses): statistically detectable but very small. Compared with
the earlier 26-run v7, v8 is effectively unchanged (mean run difference
+0.000091 rad/s; CI [−0.000088, +0.000297]). It is not a high-accuracy solution.

Cell-level audit confirms why aggregate gains are not enough. At 8.75 m/s,
−0.025 rad, unwind, expanded v7 RMSE is 0.2650 rad/s and v8 is worse at
0.2702 (22 validation transitions); at +0.025 rad it is 0.2087 versus 0.2083
(35 transitions). The 7.25 m/s / −0.075 rad steady cell stays near 0.2806
(48 transitions). A v9 diagnostic adds signed left-minus-right rear wheel
surface speed because v8 only used their mean; global RMSE is 0.06138665, and
the examined high-error cells change by less than 0.00001 rad/s. The wheel
split is therefore not a useful correction in this local linear family; it is
not promoted. The optional wheel-split feature and its sealed-holdout scorer
support remain diagnostic only. Reports:
[`v7 expanded training`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v7_gapfill_command_tracking/threshold_0p0025/fullband_yaw_regime_atlas_report.json),
[`v8 command-slew`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v8_command_slew/threshold_0p0025/fullband_yaw_regime_atlas_report.json),
[`v9 wheel-split`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v9_wheel_split/threshold_0p0025/fullband_yaw_regime_atlas_report.json).
v8 SHA-256:
`69bdb6fd4275201b894043e796cd97f507ab2d352627258ba0c21f1ebe5b0b17`.

The targeted 8.75 m/s capture did not reproduce the held-out maneuver history.
Raw validation sequences show the relevant frontier probes start at 9.5 m/s,
steer 0.14/0.18/0.20 rad, and reduce throttle approximately 0.393→0.313; speed
then traverses 9.5→7.56 m/s during the same 1.85 s swerve. The 8.75 m/s error
samples are inside that transient, not a steady 8.75 m/s hold. The current
8.75 m/s training profile starts at only 0.075 rad after a spawn reset, so it
does not teach the deceleration/history combination. This is the evidenced
data gap for the next finite test.

Implemented `yaw_frontier_throttle_slew_train` to fill exactly that gap. It
reuses the measured frontier swerve schedule at 9.5 m/s, with 0.14/0.18/0.20
rad, both turn signs, the same 0.08 throttle cut, 0.60 s stimulus delay and
0.30 s ramp versus one-tick step. Condition and treatment order are seeded and
randomized. Each condition pair gets the existing built-in reset, speed
approach and strict matched-state gate; step/ramp members are settled and
paired as in the validation profile. Two independent training run IDs/seeds
are planned. The profile is included in the approved Explore-only reset path;
runtime physics and competition code are untouched.

Focused verification after this profile addition: **67 passed** across the
open-plane schedule, yaw model math, and dataset-quality tests; Python
compilation and shell syntax pass; both experiment help paths expose the new
profile. Two captures were planned:
`openplane_yaw_frontier_throttle_slew_train_20261007_r01` and
`openplane_yaw_frontier_throttle_slew_train_20261007_r02`, with seeds
202610071 and 202610072. Each uses the 9.5 m/s, 0.14/0.18/0.20 rad frontier
matrix, both turn directions, paired 0.08 throttle-cut step/ramp responses,
and reset-separated condition pairs.

r01 and r02 completed and passed the full-bag exporter gates. The combined
train dataset is
[`yaw_frontier_throttle_slew_train_r01_r02_dataset`](../../live_runs/racing_model_diagnostics_20261007/yaw_frontier_throttle_slew_train_r01_r02_dataset/manifest.json):
1,784 samples across 24 probe sequences, with 12/12 scored probes exported
from each capture. The six speed-approach phases per run are explicitly
unscored. Both runs have zero collisions, timing faults, invalid phases, and
quality failures; packet IDs are contiguous within every retained sequence.
r01 packet joins are 2,196/2,196; r02 joins are 2,194/2,195 (99.954%, above
the existing 99.9% gate). Active state/actuator-feedback streams are
39.945–39.951 Hz, command streams 39.998–39.999 Hz, and p95 receipt gaps
25.2–25.9 ms. The fixed 25 ms packet timebase exporter retained only
packet-contiguous sequences and did not treat receipt jitter as physics dt.
Measured r01 feedback confirms the requested step reaches 0.313 from 0.393 in
one command tick and the paired ramp spans about 0.30 s. Each response starts
at 9.50 m/s and ends around 7.56 m/s; all six steering-magnitude/sign
conditions are present, with treatment order randomized. These are admitted
training data, not validation evidence. A redundant r01-only intermediate
export was removed after the combined archive passed; both source bags and the
combined train dataset remain. Next: fit the expanded atlas and compare
against the same 18 whole-run validation captures. Do not use validation or
prior final-test captures for fitting. If these matched training data still
fail to lower the problematic held-out cells, the next diagnosis must change
the model's history/response representation rather than collect more points
from the same condition. No candidate is in production.

### 2026-10-07 — v10 result and correction to the frontier capture

The v10 command-error + command-slew atlas was refit after adding the two
frontier captures. It used 30 train runs and the same 18 whole-run validation
runs; the report confirms test/final-test arrays were not read. Compared with
v8, direct-phase coverage rose from 91.593% to 91.954%, sample RMSE changed
from 0.0613866 to 0.0612455 rad/s, MAE from 0.0122827 to 0.0122038 rad/s, and
p95 absolute error from 0.051233 to 0.050608 rad/s. Run-macro paired RMSE
change was −0.000113 rad/s (18 runs; paired bootstrap 95% CI
[−0.000262, −0.000001]); 11 runs improved and 7 worsened. This is a real but
very small aggregate change, not a high-accuracy result.

The target cells did not improve at all. Re-scoring the identical validation
samples with the v8 and v10 coefficients gives exactly the same direct-phase
results: 8.75 m/s / −0.025 rad / unwind RMSE 0.270202 rad/s (22 samples in 4
runs); 8.75 m/s / +0.025 rad / unwind RMSE 0.208328 rad/s (35 samples in 5
runs); and 7.25 m/s / −0.075 rad / steady RMSE 0.280555 rad/s (48 samples in
2 runs). Inspection of v10 training coefficients shows the new frontier runs
are absent from those target models. Direct counting confirms they contributed
**zero** samples to 8.5–9.0 m/s at ±0.025 rad during unwind.

The cause is now empirical, not speculative. In the captured step response,
speed falls from 8.97 m/s at 0.70 s to 8.22 m/s at 0.80 s after stimulus. The
old steering waveform is still at about 0.10 rad in that interval; it crosses
near 0.025 rad only after the vehicle has left the 8.5–9.0 m/s band. The data
matched initial speed, steering magnitude, and throttle cut but not the joint
speed/steering trajectory responsible for the error. This is why another fit
to the same captures could not fix that cell.

Implemented a corrected training-only profile,
`yaw_frontier_lowangle_unwind_train`, without changing simulator physics or
runtime control. It retains the 9.5 m/s start, 0.14/0.18/0.20 rad magnitudes,
both turn signs, and paired 0.08 throttle-cut step/ramp. It repeats each of the
six angle/sign conditions four times, randomizes pair/treatment order, and
retimes steering so the feedback should traverse the ±0.025 rad bin around
0.725–0.775 s, while the step response crosses 8.5–9.0 m/s. This gives several
25 ms opportunities per pass; whether the measured actuator feedback actually
hits the exact cells must be checked before training. The schedule has 72
phases per capture (24 approaches and 48 probes), with a 1,200 s hard timeout.

Focused schedule/model/dataset verification after this change: **69 passed**;
Python compilation, shell syntax, and the experiment help entry also pass. The
previous frontier data are retained as valid general training captures, but
not credited as support for the failing low-angle unwind cells. Next action is
one new training pilot. It completed all 72 phases and exported 3,570 samples
across 48 probes; 24 speed approaches are unscored. It passed the bag gate:
8,484/8,484 packet joins, zero collisions/timing faults/invalid phases, no
quality failures, active state and actuator streams at 39.917–39.918 Hz, and
commands at 39.999 Hz. Receipt p95 is 25.84–25.96 ms; the ~110 ms maximum gaps
occur at reset boundaries, while every retained sequence has contiguous packet
IDs. Using measured steering feedback and the fitter's exact phase/cell
classification, r01 contributes 27 samples to (8.5–9.0 m/s, −0.025 rad,
unwind) and 29 to the corresponding +0.025 rad cell, distributed across 24
distinct probe sequences. Both exceed the per-run minimum of eight. This
justifies running one independent replicate; it does not yet satisfy the
two-run/40-sample fit threshold. The pilot dataset is
[`yaw_frontier_lowangle_unwind_train_r01_pilot_dataset`](../../live_runs/racing_model_diagnostics_20261007/yaw_frontier_lowangle_unwind_train_r01_pilot_dataset/manifest.json).

The independent r02 capture also passed the full-bag gate: 8,489/8,489 packet
joins, zero collisions/timing faults/invalid phases/quality failures, and
39.958 Hz state streams / 39.998 Hz command streams. It contributed 32 samples
to each target unwind cell. The combined train dataset is
[`yaw_frontier_lowangle_unwind_train_r01_r02_dataset`](../../live_runs/racing_model_diagnostics_20261007/yaw_frontier_lowangle_unwind_train_r01_r02_dataset/manifest.json),
with 7,142 exported samples across 96 probe sequences. The fitter's exact
25 ms row extraction yields 59 samples in the 8.5–9.0 m/s / −0.025 rad /
unwind cell and 61 in the +0.025 rad cell; r01/r02 contribute 27/32 and 29/32
respectively. Thus each cell meets the current >=40 samples and >=2 runs fit
gate. A redundant r01-only pilot export will be removed after the combined
archive is confirmed; both source bags remain. The current next action is the
v11 fit and frozen-validation comparison. No yaw model has been promoted to
MPC or odometry.

Report: [`v10 frontier-slew fit`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v10_frontier_slew/fullband_yaw_regime_atlas_report.json).

### 2026-10-07 — full-atlas nonlinear fit compared with v11

The v11 linear atlas was not discarded. As a separate research comparator, I
fit one run-balanced `ExtraTreesRegressor` per exact supported
speed/steering/phase key, using the same 14 current-time features as v11:
120 trees, maximum depth 5, minimum leaf size 4, and 0.8 feature sampling.
The fitter re-read only clean `train` and `validation` archives; its audit
reports 32 training runs and 18 whole-run validation captures. Test and
final-test arrays were not opened. Fit keys exactly reproduce the 478
phase-local keys in v11. No neighboring model or unsupported-cell fallback is
used, and this candidate is not integrated into MPC or odometry.

On identical direct-phase support (113,908 of 123,585 validation transitions,
92.17% coverage), the tree fit improves one-step yaw-rate prediction over
v11:

| Metric | v11 Huber-ridge | Exact-cell ExtraTrees |
|---|---:|---:|
| RMSE | 0.06133 rad/s | 0.04396 rad/s |
| MAE | 0.01222 rad/s | 0.00918 rad/s |
| 95th-percentile absolute error | 0.05071 rad/s | 0.03789 rad/s |
| Samples with absolute error <0.1 rad/s | 97.31% | 97.97% |
| Worst absolute error | 1.279 rad/s | 1.221 rad/s |

Across the 18 independent validation runs, ExtraTrees wins on 17 and loses on
one. Mean run-macro RMSE change is −0.01547 rad/s, with paired run bootstrap
95% CI [−0.01926, −0.01165]. Matched-support persistence RMSE is 0.08794
rad/s. This is meaningful broad one-step progress, not near-zero accuracy:
the worst error remains above 1 rad/s, 2.03% of supported predictions still
exceed 0.1 rad/s, and 7.83% of all validation transitions have no direct-phase
model. The observed data reach 11.37 m/s, not 12 m/s, and do not cover the
full speed-by-steering Cartesian product.

The validation set has now been used to compare/select the nonlinear family,
so these figures are development evidence, not a blind final-test claim. The
frozen report is
[`v11 linear atlas`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/fullband_yaw_regime_atlas_report.json);
the complete-atlas comparison is
[`ExtraTrees vs v11`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/extratrees_full_atlas_comparison.json).
The candidate has a reproducible saved model/scorer and is undergoing one new
unopened off-grid Explore holdout. After that, the key gate is a free recursive
rollout using predicted state only; the one-step score does not establish an
offline lap simulator. Simulator physics, runtime code, and the reference
trajectory remain unchanged.

The frozen v1 ExtraTrees checkpoint is
[`yaw_extratrees_fullband_v1.joblib`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/yaw_extratrees_fullband_v1.joblib)
with manifest
[`yaw_extratrees_fullband_v1_manifest.json`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/yaw_extratrees_fullband_v1_manifest.json), SHA-256
`8b5f8e7622968c797115879012c2274e6ed72104e80559171c9bc13ab7386c53`. Replay
from the saved artifact reproduced the 113,908-row comparison exactly; all
six final-test points have direct model support for both signs and both
turn-in/unwind phases.

The independent capture
`openplane_yaw_atlas_extratrees_final_20261007_r01` is now running against the
already-running pinned Explore batch simulator. It uses the established 40 Hz
bridge and minimal dynamics bag, with no MPC or runtime-code changes. The API
container was selected by its local image ID, which has the exact repo digest
used by the runner; this avoids a local Docker digest-inspect parser error
without changing image contents. The randomized schedule has 24 reset-isolated
scored maneuvers / 72 phases, six off-grid points `(4.62,0.108)`,
`(6.62,0.083)`, `(7.62,0.133)`, `(8.12,0.058)`, `(8.62,0.058)`,
`(10.62,0.033)` in m/s and rad, both turn directions and two repetitions. At
the latest live check, the bridge and recorder were connected and two
speed-approach phases had completed; no score or data-quality result is
claimed until the bag closes and passes its gates. The capture is reserved as
a new `final_test`; it will not be used for fitting or selection.
