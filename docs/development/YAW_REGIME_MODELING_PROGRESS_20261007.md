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

## Current machine/run state (2026-10-07, ~14:35 Europe/Copenhagen)

- No fitting, scoring, recorder, or experiment process is active; no old test
  process was found.
- Explore simulator container `sdu_apex_sim_explore` is running idle. No
  experiment bridge is active.
- Host load average at check: 6.38 / 5.61 / 4.76.
- The 10.25 m/s profile stopped on source-odometry timeout, not collision.

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

Focused checks: fitter math tests **3 passed**; open-plane schedule tests
**45 passed**. These support the analysis tooling but do not substitute for
held-out vehicle data.

## Current unresolved capture

`yaw_low_angle_rate_10p25_train_r01_20261007` stopped after two complete
maneuver probes; overall schedule progress was 8/96 phases. It had no collision.
The experiment recorded a source-odometry timeout. It has **not** been exported
or admitted as training data.

Offline analysis of its bag found 865 synchronized odom/steering/encoder/IMU
samples over 24.7 s, normal p95 gaps ~26 ms, but a maximum receipt gap of
~949 ms across the source streams. Commands ran at 39.78 Hz; one quality failure
was recorded. The bridge timing-fault topic had zero true samples, so the
cause of the synchronized source gap remains unknown. Preserve the bag and
quarantine it until packet IDs and source/receipt timestamps are inspected.
Do not salvage the two probes without proving their reset boundaries and
continuity independently.

The 0.0025 s phase-threshold variant appears better on some aggregate
training/validation summaries than the v3 default, but low-angle support is
uneven. Choose thresholds only from whole-run training/validation evidence;
never use r02 to tune them.

## Next steps, in order

1. Diagnose the 10.25 m/s synchronized gap from the bag: identify missing or
delayed packet IDs/timestamps across bridge timing, odom, IMU, and encoders;
decide whether it is simulator/bridge stall or recording artifact. Keep the
run quarantined. Salvage only independently verified clean probes; otherwise
repeat only this missing speed anchor after understanding the failure.
2. Refit candidate(s) with admitted train runs, including the clean 4.25,
6.25, and 8.25 m/s captures. Compare on the same whole-run validation set.
Report per-run/per-cell RMSE, MAE, p95, coverage, both turn signs, and
matched-support persistence; expose regressions and unsupported cells.
3. Evaluate recursive yaw predictions on held-out runs at 25/100/250/400/750
ms without future truth. Separate turn-in, unwind, and steady regimes. A good
one-step score is not enough to call this a plant model.
4. Freeze a candidate using train/validation only. Then run one new
reset-isolated Explore final-test capture at off-grid speeds/angles, both
turns, and mark it final-test before scoring. r02 cannot serve as this blind
check.
5. If recursive/off-grid validation still fails, document exact weak regimes
and missing evidence. Do not integrate a failed surface into MPC/odom or call
the model complete over 0–12 m/s while cells are unsupported.

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
