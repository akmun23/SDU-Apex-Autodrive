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

The current full-band atlas/teachers are research-only: they are **not
integrated into MPC or odometry**, are not a sensor-only observer, and are not
yet a validated recursive plant or lap simulator. Separately, production MPC
already has the measured Y1 high-steering response surface enabled; odometry
does not use a learned yaw predictor. Do not attribute unrelated MPC/Y1
changes to the full-band teacher experiments.

## Latest error-map and targeted-test update — 2026-10-07

This section is the current status and supersedes older machine/run-state and
“not yet captured” statements later in this historical log.

### What the 0.1 threshold means

The audited quantity is absolute future **yaw-rate prediction error**, in
rad/s, at 25, 100, 250, 500, 750, and 1000 ms. It is not yaw-angle error in
radians. The current sensor/known-command ExtraTrees teacher uses whole-run
training/validation captures; simulator truth is only an offline label. Its
worst-error requirement is not met:

| Horizon | RMSE (rad/s) | p95 absolute error (rad/s) | Samples within 0.1 rad/s | Maximum (rad/s) |
|---:|---:|---:|---:|---:|
| 25 ms | 0.0575 | 0.1133 | 94.0% | 1.259 |
| 100 ms | 0.0713 | 0.1575 | 89.5% | 2.083 |
| 250 ms | 0.0775 | 0.1631 | 88.5% | 1.938 |
| 500 ms | 0.0777 | 0.1703 | 88.7% | 1.931 |
| 750 ms | 0.0757 | 0.1677 | 88.5% | 1.974 |
| 1000 ms | 0.0762 | 0.1686 | 88.5% | 1.929 |

The expanded audit is
[`expanded_error_support_audit.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_multihorizon_teacher_v2_mismatch/expanded_error_support_audit.json).
It reports joint validation error and independent train-run/reset-sequence
support, not only marginal steering or speed bins. For example, at 1000 ms:

| Joint region | Validation support and error | Training support | Read |
|---|---|---|---|
| 6–8 m/s, |steer|<0.1 rad, steering gap<0.025 rad | 27,217 samples / 23 runs / 365 sequences; p95 0.181 rad/s; 16.7% exceed 0.1 | 34,157 / 25 runs / 381 sequences | Ample common-state support; repeated residual tail is not explained by an empty speed/steering bin. |
| 6–8 m/s, |steer|<0.1 rad, gap 0.025–0.05 rad | 2,108 / 15 runs / 213 sequences; p95 0.232 rad/s | 3,389 / 14 runs / 178 sequences | Mismatch-transition coverage is thinner and directly testable. |
| 4–6 m/s, |steer| 0.1–0.2 rad, gap<0.025 rad | 1,550 / 8 runs / 12 sequences; p95 0.235 rad/s | 3,020 / 16 runs / 52 sequences | Both fewer independent validation sequences and high residuals: transition diversity may be missing. |
| 2–4 m/s, |steer| 0.35–0.525 rad, gap<0.025 rad | 2,590 / 12 runs / 28 sequences; p95 0.727 rad/s; max 1.929 | 4,083 / 14 runs / 53 sequences | Severe full-steer transient/reversal tail; steady high-steer support alone is insufficient. |

The added speed-by-steering map, now in
[`expanded_error_support_audit_v2.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_multihorizon_teacher_v2_mismatch/expanded_error_support_audit_v2.json),
shows the broad low-steer tails are not isolated small cells: at 1000 ms,
4–6 m/s and |steer|<0.1 has 14,422 validation samples across 22 runs / 397
sequences (p95 0.231 rad/s), 6–8 m/s has 30,718 across 23 / 397 (p95 0.189),
8–10 m/s has 17,476 across 19 / 237 (p95 0.168), and 10–12 m/s has 17,855
across 16 / 112 (p95 0.150). The matching training support is 23,567 across
30 runs for 4–6, 39,608 across 25 runs for 6–8, 28,055 across 22 runs for
8–10, and 16,645 across 14 runs for 10–12. These persistent tails require a
better response/hidden-state description; simply adding more steady samples
in those bands is not justified.

At 25 ms, p95 error is 0.057 rad/s when steering command/feedback differs
by <0.025 rad, but rises to 0.219–0.578 rad/s for gaps above 0.025 rad.
Throttle-gap and rear-wheel/body-speed mismatch are also associated with
large tails. That association is diagnostic, not proof of causality. Worst
examples include (a) a 3.47 m/s, 0.42-rad high-steer response with a wrong-sign
1 s prediction and (b) 8–9.5 m/s low-steer transients with high lateral IMU
acceleration and substantial wheel/body-speed separation even when steering
command and feedback agree. Thus there are two different problems: sparse
combinations of transition phase, and model residuals inside heavily sampled
common regimes. More steady-state repetition alone cannot solve the latter.

The full speed-by-steering audit confirms this is not one isolated operating
point. For cells with at least 100 held-out rows, the count with p95 above
0.1 rad/s is 10/14/20/16/20/11 at 25/100/250/500/750/1000 ms. Across those
cells, 9,811/16,721/19,314/18,056/18,503/17,247 validation rows respectively
exceed 0.1 rad/s. At 250 ms every 2–12 m/s band has at least one failing
speed/steering cell. This table includes every cell with p95 >0.1 at 250 ms
and at least 20 validation rows. It gives the corresponding 1-second result
and its smaller row count, plus 250 ms training support. Each p95 is absolute
error in rad/s; the adjacent percentage is the fraction of rows over 0.1.

| Speed | |steer| | Validation n / runs / sequences (250 ms) | p95 / >0.1 (250 ms) | 1 s n; p95 / >0.1 | Training n / runs / sequences (250 ms) |
|---|---:|---:|---:|---:|---:|
| 0–2 m/s | 0.1–0.2 | 46 / 4 / 6 | 0.143 / 50.0% | 46; 0.140 / 13.0% | 88 / 5 / 9 |
| 0–2 m/s | 0.2–0.35 | 208 / 5 / 8 | 0.200 / 13.5% | 28; 0.884 / 17.9% | 129 / 3 / 5 |
| 0–2 m/s | 0.35–0.525 | 202 / 4 / 7 | 0.582 / 15.8% | 22; 1.154 / 40.9% | 141 / 3 / 5 |
| 2–4 m/s | 0.1–0.2 | 1,132 / 11 / 36 | 0.211 / 20.1% | 990; 0.173 / 13.9% | 1,361 / 15 / 88 |
| 2–4 m/s | 0.2–0.35 | 2,414 / 12 / 40 | 0.202 / 30.3% | 2,077; 0.147 / 7.1% | 4,856 / 14 / 90 |
| 2–4 m/s | 0.35–0.525 | 2,947 / 12 / 28 | 0.817 / 15.4% | 2,700; 0.727 / 10.1% | 4,992 / 14 / 53 |
| 4–6 m/s | 0–0.1 | 15,109 / 22 / 397 | 0.192 / 10.1% | 14,422; 0.231 / 17.6% | 25,817 / 30 / 482 |
| 4–6 m/s | 0.1–0.2 | 3,280 / 12 / 113 | 0.208 / 32.4% | 2,980; 0.237 / 35.0% | 5,444 / 19 / 158 |
| 4–6 m/s | 0.2–0.35 | 5,480 / 15 / 121 | 0.231 / 16.9% | 4,682; 0.196 / 18.5% | 7,902 / 19 / 135 |
| 4–6 m/s | 0.35–0.525 | 2,153 / 10 / 61 | 0.236 / 22.7% | 1,632; 0.082 / 1.8% | 3,044 / 10 / 51 |
| 6–8 m/s | 0–0.1 | 31,565 / 23 / 397 | 0.179 / 16.5% | 30,718; 0.189 / 18.6% | 40,971 / 25 / 397 |
| 6–8 m/s | 0.1–0.2 | 6,932 / 19 / 248 | 0.205 / 26.6% | 6,425; 0.223 / 25.8% | 9,493 / 18 / 204 |
| 6–8 m/s | 0.2–0.35 | 13,680 / 19 / 227 | 0.124 / 7.2% | 12,838; 0.067 / 2.9% | 12,484 / 18 / 175 |
| 6–8 m/s | 0.35–0.525 | 10,796 / 15 / 110 | 0.108 / 5.8% | 10,122; 0.029 / 1.2% | 9,532 / 16 / 86 |
| 8–10 m/s | 0–0.1 | 18,599 / 19 / 237 | 0.144 / 12.6% | 17,476; 0.168 / 16.8% | 30,297 / 22 / 276 |
| 8–10 m/s | 0.1–0.2 | 3,479 / 13 / 108 | 0.155 / 12.2% | 2,814; 0.104 / 5.5% | 4,953 / 16 / 154 |
| 8–10 m/s | 0.2–0.35 | 1,973 / 16 / 70 | 0.129 / 7.1% | 1,180; 0.049 / 2.2% | 3,139 / 15 / 103 |
| 8–10 m/s | 0.35–0.525 | 957 / 9 / 30 | 0.158 / 7.8% | 448; 0.017 / 0.4% | 3,091 / 9 / 61 |
| 10–12 m/s | 0–0.1 | 19,326 / 16 / 112 | 0.146 / 8.5% | 17,855; 0.150 / 10.0% | 18,542 / 14 / 93 |
| 10–12 m/s | 0.1–0.2 | 5,960 / 16 / 82 | 0.135 / 8.0% | 4,799; 0.084 / 2.5% | 5,910 / 14 / 51 |
| 10–12 m/s | 0.2–0.35 | 901 / 8 / 18 | 0.112 / 8.4% | 712; 0.071 / 0.6% | 336 / 5 / 5 |

This map deliberately separates “not enough data” from “data exists but
prediction is wrong.” The largest persistent tail is the 2–4 m/s high-steer
cell: p95 0.817 rad/s at 250 ms and 0.727 at 1 s. At 250 ms, 0–2 m/s above 0.2 rad
has only 129–141 training rows across three runs/five sequences, and 10–12 m/s
at 0.2–0.35 rad has 336 training rows across five runs/five sequences. Most
other failing cells have thousands of samples across 10–30 independent
training runs and dozens to hundreds of sequences; simply repeating their
steady state is not justified. They call for transient-conditioned fitting
or state/model changes. High-speed tests stay within the existing measured
steering frontier; sparse cells above it are not a license to test beyond
that envelope.

The 250 ms three-way cuts refine the likely causes. The 2–4 m/s, 0.35–0.525
rad cell remains at p95 0.817 rad/s with steering command/feedback gap below
0.025 rad (2,801 validation rows across 12 runs), so actuator lag alone cannot
explain it. For 4–6 m/s, a 0.025–0.05 rad steering gap at low steering has
p95 0.292 (1,068 rows / 9 runs); at 10–12 m/s and 0.1–0.2 rad it has p95
0.260 (770 / 10). The wheel/body-speed mismatch cut shows p95 0.257 for
4–6 m/s, |steer|<0.1, and 0.1–0.25 m/s mismatch (3,216 rows / 12 runs),
versus 0.250 for the 0–0.1 m/s mismatch slice (3,025 / 12). So slip is a
useful condition to model, but it does not explain the full residual by
itself. The highest error also appears in the low-throttle-gap slice; this
rules out throttle command/feedback mismatch as the sole cause. These are
stratified correlations, not causal proofs; the paired step/ramp and throttle
cut tests are intended to distinguish the effects.

### Targeted Explore capture suite

The existing 0–12 m/s, signed-steering, throttle, and dynamic captures remain
the broad support set. New captures are limited to the error-conditioned
gaps, reset-isolated, randomized in order, paired across step/ramp and turn
sign, and split by whole run. No competition/runtime model or simulator
physics is changed.

| Family | Designed probes per run | Conditions targeted |
|---|---:|---|
| `yaw_error_steering_event_gapfill` | 32 | 5.0 m/s turn-in at 0.20/0.28 rad; 3.5 m/s reversal at 0.35/0.42 rad; both signs, step/0.30 s ramp, 0.25/0.75 s event age. |
| `yaw_error_command_gap_gapfill` | 48 | 6.5–9.5 m/s at 0.075/0.104 rad, both signs, step/ramp, two event ages. |
| `yaw_error_command_gap_lowspeed_gapfill` | 32 | 4.5/5.5 m/s at 0.075/0.104 rad, same paired command/feedback-mismatch waveform; added because 4–6 m/s held-out tails were independently present. |
| `yaw_error_lowspeed_steering_gapfill` | 64 | 1.5/2.5 m/s at 0.35/0.50 rad; turn-in and reversal, both signs, step/ramp, two event ages. Addresses the clearly under-supported 0–2 m/s high-steer bins. |
| `yaw_error_midspeed_steering_gapfill` | 64 | 4.5/6.5 m/s at 0.15/0.20 rad; onset and reversal, both signs, step/ramp, 0.25/0.75 s event age. Targets transient diversity in populated 4–8 m/s cells. |
| `yaw_error_highspeed_steering_gapfill` | 64 | 10.5 m/s at 0.14/0.20 rad and 11.1 m/s at 0.12/0.18 rad; onset/unwind, both signs, step/ramp, two event ages, within the existing measured steering frontier. |
| `yaw_error_wheelspin_gapfill` | 48 | 8.5/9.5/11 m/s at 0.06/0.10 rad; both signs, small throttle increase and throttle cut, step/ramp; log wheel, actuator, IMU, and yaw response. |

The plan is two independent training runs and one independent whole-run
validation capture per family. All 21 captures in the original seven-family
matrix are now complete and admitted. They total 1,056 reset-isolated probes
(352 conditions repeated in each of the three independent runs). The final
wheel-spin validation capture passed with 11,784 samples / 56 sequences,
48/48 scored phases valid, zero collisions and timing faults, exact packet
alignment, 39.973 Hz odometry, and a 26.12 ms p95 receipt gap. Its export is
[`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_wheelspin_validation_r03_dataset/manifest.json).

The original schedule missed several residual cells, so a finite supplemental
profile is now implemented and schedule-checked. It adds 160 conditions per
capture, repeated as two training runs plus one independent validation run
(480 more probes; 1,536 total across 24 captures). Estimated schedule time is
about 47 minutes per supplemental capture, with a 3,600 s timeout. This is
error-conditioned rather than a Cartesian sweep. The 40 Hz Explore bridge,
exact 25 ms packet timebase, collision/timing-fault gate, and command-feedback
checks remain mandatory. Final-test data stays sealed. If errors persist in
densely supported cells after this finite collection, the next action is a
model-structure diagnosis, not another round of repeated steady holds.

#### Remaining-cell coverage audit and suite extension

Comparing every failing 250 ms cell in the table above against the concrete
probe coordinates revealed several holes in the first seven-family schedule.
This is important: a broad speed/steering bin can contain many samples while
the exact transition phase at a particular speed-angle point is still absent.
The first schedule does not directly probe 0–2 m/s at 0.1–0.35 rad, 2–4 m/s
at 0.1–0.35 rad, 4–6 m/s above 0.35 rad, or the 8–10 m/s 0.1–0.525 rad
transition cells. Its 6.5 m/s mid-angle point also does not include the
0.2–0.35 rad bin. These gaps are added rather than claiming the original
matrix covers every failing cell.

The supplemental profile `yaw_error_residual_steering_grid_gapfill` is
implemented as a finite, targeted 160-probe schedule per run. It uses the same measured-speed
approach and reset-isolated step/ramp protocol, with both turn signs, onset
and reversal, and 0.25/0.75 s transition ages at these measured-envelope
points: (1.5 m/s, 0.15/0.25 rad), (3.5 m/s, 0.15/0.25 rad), (4.5 m/s,
0.42 rad), (6.5 m/s, 0.25/0.42 rad), and (8.5 m/s, 0.15/0.25/0.42 rad).
Each point contributes 16 paired conditions. It targets omitted failing
cells rather than a new uniform grid. The focused schedule test verifies all
ten points, the pairing and event ages, reset approaches, and a 3,600 s
timeout. Every requested angle remains inside the measured steering
frontier; do not extrapolate steering authority to fill a table cell. Run it
twice as training and once as whole-run validation. The original seven
profiles remain unchanged and their results are retained.

### Capture facts and artifacts so far

- Steering r01: 96/96 phases, 32/32 probe phases valid, zero collisions and
  timing faults; 39.80 Hz active sensor rate, 26.6 ms p95 receipt gap, exact
  contiguous packet IDs within each reset-delimited sequence. Prepared export:
  5,163 samples / 34 sequences at
  [`yaw_steering_train_r01_retry_dataset`](../../live_runs/racing_model_diagnostics_20261007/yaw_steering_train_r01_retry_dataset/manifest.json).
- Steering r02: same 96/96 and 32/32 clean completion; 39.95 Hz active sensors,
  26.1 ms p95 gap, 100% packet joins and no within-sequence gaps. Prepared
  export: 5,335 samples / 35 sequences at
  [`yaw_steering_train_r02_dataset`](../../live_runs/racing_model_diagnostics_20261007/yaw_steering_train_r02_dataset/manifest.json).
- Steering r03 validation: 96/96 and 32/32 clean completion; 39.68 Hz active
  sensors and 26.8 ms p95 gap. There is one unmatched startup-only odometry
  sample (5,556/5,557 joined = 99.982%, above the 99.9% gate); no packet IDs
  are missing inside retained sequences. Prepared export: 4,830 samples / 35
  sequences at
  [`yaw_steering_validation_r03_dataset`](../../live_runs/racing_model_diagnostics_20261007/yaw_steering_validation_r03_dataset/manifest.json).
- Raw-bag whole-duration rate is lower because it includes reset gaps; active
  sequence rate, packet continuity, and p95 timing are the relevant rates.
  Steering commands are normalized by the actuator limit, so command tracking
  was checked after converting back to radians. In r01 command-vs-designed
  steering RMSE was 0.00048 rad median across probes; feedback showed the
  expected finite actuator lag (step response median RMSE 0.099 rad, ramp
  0.055 rad), rather than a command-publisher failure. Speed-hold median error
  was about 0.031 m/s.
- The r02 driver exited successfully and the bag is closed/usable; the host
  wrapper then printed a shell syntax error after completion while the runner
  source was being edited. The runner now passes `bash -n` and `--help`; do not
  edit the runner while a capture is executing.
- A new 2-D speed-by-steering residual/support table is being added to the
  diagnostic output so every populated regime can be ranked directly; the
  existing 3-D mismatch/wheel-gap/throttle-gap audit is retained. The current
  GRU trajectory-teacher training continues in the background, pinned away
  from the batch simulator; no GRU result has yet been promoted.

### Live continuation after the expanded audit — 2026-10-07

- `openplane_yaw_error_command_train_r01_20261007` completed its 48 paired
  reset-isolated probes (144/144 phases), with no collision, invalid phase,
  or quality-gate failure. The cleaned training export has 9,045 samples in
  55 contiguous sequences. Packet join was 9,880/9,880; active odometry was
  39.76 Hz with 26.43 ms p95 receipt gap; command streams were 40 Hz. Bag:
  [`run_0.db3`](../../live_runs/openplane_yaw_error_command_train_r01_20261007/run/run_0.db3);
  export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_command_train_r01_dataset/manifest.json).
- `openplane_yaw_error_command_train_r02_20261007` completed the independent
  replication (144/144 phases; zero collision/quality failures; 9,467 samples
  in 57 sequences; packet join 100%; odometry 39.91 Hz, 26.14 ms p95 gap;
  command stream 40 Hz). Its training export is
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_command_train_r02_dataset/manifest.json).
- `openplane_yaw_error_command_validation_r03_20261007` is the independent
  whole-run validation capture, now complete and prepared as validation-only:
  144/144 phases, no collision/quality failure, 9,642 samples in 49 sequences,
  100% packet join, 39.95 Hz odometry with 26.01 ms p95 gap, and 40 Hz command
  streams. Its export is
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_command_validation_r03_dataset/manifest.json).
- The low-speed high-steer r01 training capture completed 192/192 schedule
  phases with zero quality failures. Its clean training export contains 9,393
  samples in 65 reset-delimited sequences; 64/64 scored phases were valid,
  collisions and bridge faults were zero, active streams were 39.89 Hz with
  26.13 ms odometry p95 receipt gap, and packet alignment was 99.990% (one
  unmatched startup packet; retained sequence packet IDs are contiguous).
  Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_lowspeed_steering_train_r01_dataset/manifest.json).
- The independent low-speed high-steer r02 training replication is now
  complete and admitted: 192/192 schedule phases, zero quality failures,
  9,270 samples in 69 reset-delimited sequences, 64/64 scored phases valid,
  zero collisions/faults, 100% packet alignment, 39.845 Hz odometry, and
  26.46 ms p95 receipt gap. Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_lowspeed_steering_train_r02_dataset/manifest.json).
- The independent low-speed high-steer r03 whole-run validation capture
  completed and passed: 192/192 schedule phases, no quality failures, 9,338
  samples in 66 sequences, 64/64 scored phases valid, zero collisions/faults,
  99.990% packet alignment (one unmatched startup packet), 39.923 Hz odometry,
  and 26.17 ms p95 receipt gap. Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_lowspeed_steering_validation_r03_dataset/manifest.json).
- The next scheduled family is the 4.5/5.5 m/s command/feedback-gap profile;
  it targets the low-speed portion of the persistent 4–6 m/s tail and will
  likewise use two train runs plus a separate validation run.
- Its first training capture, `openplane_yaw_error_command_lowspeed_train_r01_20261007`,
  passed and exported 5,274 samples in 33 sequences: 32/32 scored phases
  valid, zero collisions/faults, 99.982% packet alignment (one unmatched
  startup packet), 39.957 Hz odometry, 25.99 ms p95 gap, and 40 Hz command
  streams. Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_command_lowspeed_train_r01_dataset/manifest.json).
- The command-gap r02 training replication has since passed: 5,288 samples /
  32 sequences, 32/32 scored phases valid, zero collisions/faults, exact
  packet alignment, 39.946 Hz odometry (25.99 ms p95 gap), and 40 Hz command
  streams. Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_command_lowspeed_train_r02_dataset/manifest.json).
- Its r03 whole-run validation capture passed: 5,281 samples / 32 sequences,
  32/32 scored phases valid, zero collisions/faults, exact packet alignment,
  39.957 Hz odometry (26.00 ms p95 gap), and 40 Hz command streams. Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_command_lowspeed_validation_r03_dataset/manifest.json).
- Both independent command-gap training runs and the validation run are now
  complete and admitted. The next family, 4.5/6.5 m/s steering transitions,
  is underway.
- Its r01 training capture, `openplane_yaw_error_midspeed_steering_train_r01_20261007`,
  completed and passed: 11,484 samples / 65 sequences, 64/64 scored phases
  valid, zero collisions/faults, 99.992% packet alignment (one startup-only
  unmatched packet), 39.958 Hz odometry (25.98 ms p95 gap), and 40 Hz command
  streams. Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_midspeed_steering_train_r01_dataset/manifest.json).
- The mid-speed r02 repeat has completed and passed: 11,495 samples / 64
  sequences, 64/64 scored phases valid, zero collisions/faults, 99.992%
  packet alignment (one startup-only unmatched packet), 39.940 Hz odometry
  (26.01 ms p95 gap), and 40 Hz command streams. Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_midspeed_steering_train_r02_dataset/manifest.json).
- Its independent r03 whole-run validation capture passed: 11,504 samples /
  65 sequences, 64/64 scored phases valid, zero collisions/faults, exact
  packet alignment, 39.968 Hz odometry (26.03 ms p95 gap), and 40 Hz command
  streams. Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_midspeed_steering_validation_r03_dataset/manifest.json).
- The next scheduled family is high-speed steering onset/unwind at 10.5 and
  11.1 m/s, constrained to the measured steering frontier.
- Its r01 training capture completed and passed: 15,407 samples / 67
  sequences, 64/64 scored phases valid, zero collisions/faults, exact packet
  alignment, 39.955 Hz odometry (26.00 ms p95 gap), and 40 Hz command streams.
  Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_highspeed_steering_train_r01_dataset/manifest.json).
- The independent high-speed steering r02 training repeat passed: 15,439
  samples / 68 sequences, 64/64 scored phases valid, zero collisions/faults,
  99.994% packet alignment (one startup-only unmatched packet), 39.956 Hz
  odometry (26.00 ms p95 gap), and 40 Hz command streams. Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_highspeed_steering_train_r02_dataset/manifest.json).
- The independent high-speed r03 whole-run validation capture passed: 15,285
  samples / 66 sequences, 64/64 scored phases valid, zero collisions/faults,
  99.994% packet alignment (one startup-only unmatched packet), 39.938 Hz
  odometry (26.02 ms p95 gap), and 40 Hz command streams. Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_highspeed_steering_validation_r03_dataset/manifest.json).
- The last original seven-family profile is wheel-spin/throttle slew at
  8.5–11 m/s; its paired throttle increases/cuts will measure the actual
  encoder/body-speed divergence rather than assuming throttle-command change
  causes slip.
- Wheel-spin r01 training completed and passed: 11,847 samples / 48 sequences,
  48/48 scored phases valid, zero collisions/faults, 99.992% packet alignment
  (one unmatched startup packet), 39.946 Hz odometry (26.01 ms p95 gap), and
  40 Hz command streams. Export:
  [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_wheelspin_train_r01_dataset/manifest.json).
- Its independent r02 training replication is now running.
- The targeted schedule now has one shared builder for paired onset/unwind/
  reversal tests. All eight focused schedule checks pass for low-speed
  high-steer, mid-speed transitions, and high-speed measured-frontier
  transitions. The new profiles are registered in the runner; its shell
  syntax/help checks pass and it uses the established batch Explore procedure.
- First attempt to start the low-speed family exposed a missing Python CLI
  choice even though the schedule and shell profile existed. It stopped before
  actuator excitation; the 1.6 s startup-only bag was unusable and removed.
  CLI choices now derive from the complete yaw-profile tuple, and a focused
  test asserts every yaw schedule is CLI-exposed. The corrected r01 capture
  completed as recorded above; the r02 replication is complete and its r03
  validation run is in progress.
- The independent GRU trajectory teacher is still training (last saved
  validation evaluation: epoch 74, 32,499 pooled validation windows; best
  validation-selected checkpoint remains epoch 70). At the latest evaluation
  its pooled p95 errors were 0.099, 0.078, 0.079, 0.080, 0.082, and 0.088
  rad/s at 25/100/250/500/750/1000 ms, respectively. Its maximum errors still
  reached 1.06–1.85 rad/s. This is a promising comparator, not a
  solved all-sample model: training is incomplete, 25 ms p95 is only just
  below threshold, and no new targeted capture has been scored by it.
- The held-out final/test arrays remain sealed. No model from this full-band
  teacher study is integrated into MPC or odometry; production MPC retains
  the separately validated Y1 response surface.

No full-band yaw teacher is integrated into odometry or MPC. This capture suite
is to decide whether transition-data addition measurably improves whole-run
validation; it is not a claim that the current observer now meets 0.1 rad/s.

### Live progress — 2026-10-08

- The seventh/original-final capture, `openplane_yaw_error_wheelspin_validation_r03_20261007`, closed cleanly: 96/96 schedule phases, all 48 scored probes valid, zero collisions, zero timing faults, and command publication at 39.86 Hz. Its admitted export has 11,784 samples / 56 sequences, 100% packet alignment, 39.973 Hz odometry (26.12 ms p95), and no quality failures. The raw bag is [`run_0.db3`](../../live_runs/openplane_yaw_error_wheelspin_validation_r03_20261007/run/run_0.db3); the validation manifest is [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_wheelspin_validation_r03_dataset/manifest.json).
- Implemented `yaw_error_residual_steering_grid_gapfill` in the excitation schedule and runner. It covers ten omitted failing speed/steering cells with onset/reversal, both turn signs, step/ramp steering, and two event ages; there are 160 probes (480 phases) per capture. Its focused schedule test checks coverage, reset isolation, pairing, randomization, and the 3,600 s budget. The nine focused gapfill schedule tests pass, as does `bash -n` on the runner.
- Two startup attempts exposed runner wiring issues before any scored probe. First, the Python CLI still capped timeouts at 1,200 s; its 1.6 s bag had no phase or command data and was removed. Second, the new profile was accepted but omitted from the runner's reset-enabled profile list. The idle bag had 0/480 phases and only neutral commands; it was removed after confirming the recorded source streams were ~40 Hz. Both issues are fixed and recorded in the retained experiment logs.
- The actual first training replication, `openplane_yaw_error_residual_grid_train_r01_retry2_20261008`, completed and was admitted: 28,709 samples / 165 sequences; 160/160 scored probe phases valid; zero collisions/timing faults; exact packet alignment; odom at 39.9568 Hz (26.01 ms p95). Export: [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_residual_grid_train_r01_retry2_dataset/manifest.json).
- Independent training repetition r02, seed 202610082, also passed admission: 28,707 samples / 164 sequences, 160/160 probe phases valid, zero collisions/timing faults, 99.9966% packet alignment (one unmatched startup sample), and 39.958 Hz odometry (26.01 ms p95). Export: [`manifest.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_error_residual_grid_train_r02_dataset/manifest.json).
- The independent whole-run validation capture `openplane_yaw_error_residual_grid_validation_r03_20261008`, seed 202610083, is now running. It is the final planned capture; after admission, the next stage is whole-run model refit/scoring, not more data collection unless validation reveals a specific unsupported cell.
- The GRU trajectory teacher is still running in the background; its last persisted validation is epoch 74 / best epoch 70. No candidate is integrated into runtime odometry or MPC, and no production physics or reference trajectory was changed.

## Earlier at-a-glance state — historical, superseded by the update above

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
- **Done:** v11 exact-target-cell assessment, full-atlas nonlinear ExtraTrees
  comparison on whole-run validation, and one unseen off-grid Explore final
  capture scored against frozen v11/v1.
- **Finding:** ExtraTrees improves supported whole-run and new-capture one-step
  yaw scores; near-target windows are not uniformly better and unsupported
  cells still abstain.
- **Done:** four-fold grouped train-only comparison of the 14-feature tree
  against a causal 17-feature lagged-history version. Added history does not
  produce a reliable run-level improvement.
- **Done:** current IMU roll and roll rate failed as a global yaw-model
  extension; isolated cell wins were inconsistent across direction.
- **Done:** reviewed `Yaw_fitting.txt`, rejected every oracle-state feature,
  and fitted direct sensor-only yaw predictions at six command-conditioned
  horizons on 29 train / 23 whole-run validation captures.
- **Finding:** flattened 1.6 s history improves only the 25 ms horizon; the
  current-sample model is better at 100–750 ms. Neither meets the requested
  worst-error bound. Exact worst errors identify a 5 m/s steering-onset gap and
  a 3.5 m/s full-steer reversal gap in training.
- **Next:** gather only those two reset-isolated transition families in paired
  train/validation captures, refit, and re-score all horizons. Do not integrate
  a yaw model into MPC/odometry until the sign-reversal errors and whole-run
  transfer are addressed.

## Earlier machine/run state — historical, superseded by updates above

- The pinned Explore simulator was stopped after the final capture because no
  simulator or recorder work was active. The r01/r02 captures, combined
  dataset export, v11 fit, full-atlas tree sweep, history CV, and roll CV are
  complete. No analysis or recording process is active. The host has 16
  logical CPUs, about 7.8 GiB memory available, and 125 GiB disk free.
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
The candidate has a reproducible saved model/scorer and completed one new
off-grid Explore final holdout. The next key gate is a free recursive rollout
using predicted state only; the one-step score does not establish an offline
lap simulator. Simulator physics, runtime code, and the reference trajectory
remain unchanged.

The frozen v1 ExtraTrees checkpoint is
[`yaw_extratrees_fullband_v1.joblib`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/yaw_extratrees_fullband_v1.joblib)
with manifest
[`yaw_extratrees_fullband_v1_manifest.json`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/yaw_extratrees_fullband_v1_manifest.json), SHA-256
`8b5f8e7622968c797115879012c2274e6ed72104e80559171c9bc13ab7386c53`. Replay
from the saved artifact reproduced the 113,908-row comparison exactly; all
six final-test points have direct model support for both signs and both
turn-in/unwind phases.

The independent capture
`openplane_yaw_atlas_extratrees_final_20261007_r01` completed 72/72 phases /
24/24 maneuvers with zero harness quality failures, collisions, or timing
faults and 39.71 Hz command output. Closed-bag audit: 5,821/5,821 packet
joins, zero packet gaps, 39.949 Hz state/feedback streams, 39.998 Hz command
streams, state p95 receipt gaps 25.82–25.89 ms, and zero collision-count
changes. Whole-bag receipt rate is 34.76 Hz only because it includes reset
boundaries; active phase rates remain about 39.95 Hz. The API container used
the local image ID whose repo digest exactly matches the runner's pinned
image, avoiding a local Docker digest-inspect parser error without changing
image contents. The clean bag was assigned `final_test` and was not used for
fitting.

The scorer initially counted its prepended 500 ms phase-history context as
part of the test interval, while the dataset correctly exports only
non-negative phase-time samples. The observed in-window packet coverage is at
least 98.13% for every point. The scorer now uses the same labeled interval;
the existing 98% gate was not weakened. A focused test protects this rule.
The complete reset-separated archive is
[`yaw_extratrees_final_holdout_r01_dataset`](../../live_runs/racing_model_diagnostics_20261007/yaw_extratrees_final_holdout_r01_dataset/manifest.json);
the phase-window scoring view is
[`yaw_extratrees_final_holdout_r01_scoring_view_dataset`](../../live_runs/racing_model_diagnostics_20261007/yaw_extratrees_final_holdout_r01_scoring_view_dataset/manifest.json).

On the 2,136 supported final-test transitions, ExtraTrees v1 scores RMSE
0.03555 rad/s, MAE 0.01685, p95 0.06243, and 97.99% below 0.1 rad/s. Frozen
v11 direct-phase scores those exact same samples at RMSE 0.04889, MAE 0.01713,
p95 0.10040, and 94.99% below 0.1; persistence is 0.10189 rad/s on the same
support. Both direct atlases abstain on the same 410 other transitions. At the
full maneuver-window level, ExtraTrees has lower RMSE in all 24 conditions.
Within the narrower near-target windows it wins 20/24; the four losses are
the two repeats and both turn signs at 8.62 m/s / about ±0.058 rad (roughly
0.035 vs 0.009 rad/s RMSE). This is one independent capture, so no run-level
confidence interval is claimed. Frozen scores:
[`ExtraTrees r01`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/yaw_extratrees_final_holdout_r01_score.json)
and [`v11 r01`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/v11_final_holdout_r01_score.json).

### 2026-10-07 — grouped test of lagged yaw/command history

Ran four-fold `GroupKFold` by whole capture using the 32 clean training
captures only (215,304 rows). The existing 14-feature tree and 17-feature
variant adding the prior yaw increment, steering rate, and throttle rate were
evaluated on the same 181,465 out-of-fold transitions and 438 supported
speed/steering/phase keys. The fit gates remain 40 samples and two contributing
runs per cell. Validation was discovered for split auditing but not used in
fold fitting/scoring; test and final-test arrays were not read.

| Predictor | RMSE | MAE | p95 absolute | fraction <0.1 rad/s |
|---|---:|---:|---:|---:|
| 14-feature ExtraTrees v1 | 0.056484 | 0.013554 | 0.05842 | 97.177% |
| + lagged history | 0.056587 | 0.013464 | 0.05745 | 97.296% |
| Huber-ridge comparator | 0.078130 | 0.019093 | 0.08630 | 95.736% |

The history-v1 paired run-macro RMSE difference is +0.000063 rad/s (15 wins,
17 losses; bootstrap 95% CI [−0.000168, +0.000315]), so history is not a
repeatable improvement and is not promoted. The original tree beats the linear
comparator on 30/32 runs (mean difference −0.02071 rad/s; 95% CI
[−0.02504, −0.01634]). This is still one-step truth-conditioned prediction,
not free recursive rollout. Full speed-band metrics and per-run values are in
[`yaw_history_grouped_cv.json`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/yaw_history_grouped_cv.json).

The focused diagnostics use `imu_attitude_frames` already present in the clean
dynamics archives (roll and roll-rate are valid for 7,863/7,865 samples in the
recent combined capture). Four-fold grouped CV evaluated current-sample roll
and roll-rate against v1 on identical train-fold rows: 181,221 transitions,
437 supported cells. Baseline v1 RMSE is 0.057098 rad/s; roll-augmented RMSE is
0.057358, MAE 0.013848 versus 0.013586, and p95 0.05882 versus 0.05965. The
paired run-macro RMSE difference is +0.000230 (10 wins, 22 losses; bootstrap
95% CI [−0.0000003, +0.0004563]). Roll does not improve the full supported
domain. A few +turn-in cells improve, but nearby opposite-sign cells regress;
the effects are too sparse and asymmetric to support a regime-specific rule.
Reports:
[`roll grouped CV`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/yaw_imu_roll_grouped_cv.json)
and [`roll by exact cell`](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/yaw_imu_roll_grouped_cv_by_cell.json).

### Unseen-speed evidence and current gap

The only final-test speed-transfer run so far tested six unseen speed values
(4.62, 6.62, 7.62, 8.12, 8.62, 10.62 m/s), but at low steering only
(0.033–0.133 rad). The exact-cell tree improves the one-step score over v11 on
2,136 supported transitions (RMSE 0.03555 vs 0.04889 rad/s); one 7.62 m/s,
0.133-rad negative-turn sequence is materially worse than neighboring points
(0.0639 RMSE, 0.363 max error). The final run is one capture and cannot
establish broad speed transfer.

The atlas is not continuous across speed cells: each prediction selects one
exact 0.5 m/s × 0.025 rad × response-phase cell, with only within-cell speed
and steering offsets as features. A frozen-support audit found 12 high-angle
(|steering|≥0.20 rad), steady cells between 8.0 and 10.25 m/s, but no
supported high-angle turn-in/unwind cells above 7.75 m/s. A separate final
holdout schedule now probes four off-grid high-speed/high-angle steady points
(8.37/0.302, 8.62/0.427, 9.12/0.302, 9.37/0.427 m/s/rad), both directions and
two repeats. Schedule/shell checks passed (57 schedule tests); the capture
and scoring are now complete, as recorded below.

### Production input compatibility — next implementation gate

The tree v1 feature builder currently computes speed, yaw rate, lateral speed,
and speed rate from simulator rigid-state truth. This is acceptable for a
one-step plant-identification diagnostic but is not a deployable odom/MPC
feature contract. The dataset also records the contemporaneous odometry state
`[u,v,r]` and permitted actuator/encoder/IMU inputs. Next, fit and whole-run
cross-validate an observer-compatible variant using only those current-time
signals and GT only as the label. Then check an MPC-state/control-only model
separately; MPC rollout must not consume future recorded sensors. Only after
these input-compatible variants beat their production baselines on grouped
whole runs and a new sealed capture should they be wired into runtime code.
The current one-step truth-feature tree has **not** been integrated into either
odometry or MPC.

### 2026-10-07 — reviewer playbook: continuous local yaw schedule

Read and evaluated the external experiment playbook at
`/home/akselmo/Downloads/SDU_APEX_EXISTING_DATA_EXPERIMENT_PLAYBOOK_2026-10-07.md`.
The tested part was its mechanism/phase-specific model-bank recommendation:
use a continuous scheduling surface *inside measured joint support*, retain
separate response dynamics, and validate on whole held-out captures. This was
not an attempt to fit one universal model or to execute every unrelated
throttle, braking, odometry, and optimizer proposal in the playbook.

No simulator was started for this analysis. It used the already-recorded
high-speed/high-steer training capture
[`speed-surface training bag`](../../live_runs/openplane_yaw_highsteer_speed_surface_train_20261007_r01/run/run_0.db3)
and the already-recorded independent probe capture
[`high-steer holdout bag`](../../live_runs/openplane_yaw_atlas_extratrees_highsteer_final_20261007_r01/run/run_0.db3).
The held-out observer-state replay is
[`sensor-odometry replay`](../../live_runs/racing_model_diagnostics_20261007/yaw_sensor_state_candidate/highsteer_final_sensor_odom_replay/replayed/replayed_0.db3).
These are separate captures, but the training experiment is one capture with
reset-isolated repetitions; individual phases are not independent runs.

The training capture contains 48 reset-isolated conditions: three speed
anchors (measured approximately 8.170, 8.732, and 9.222 m/s), four steering
magnitudes (approximately 0.350, 0.425, 0.475, and 0.500 rad), both turn signs,
and two repeats. Sidecar-to-sensor-odometry joining covered 100% of its rows.
For steady yaw response, a continuous PCHIP schedule in measured speed and
absolute physical steering was fit, with turn sign applied by symmetry. The
measured absolute equilibrium yaw-rate knots (rad/s) are:

| Speed (m/s) | 0.350 rad | 0.425 rad | 0.475 rad | 0.500 rad |
|---:|---:|---:|---:|---:|
| 8.170 | 0.78310 | 0.83145 | 0.88848 | 0.92775 |
| 8.732 | 0.74603 | 0.79320 | 0.85045 | 0.88770 |
| 9.222 | 0.71705 | 0.76205 | 0.81808 | 0.85325 |

This surface is explicitly supported only for measured speed 8.170–9.222 m/s
and steering magnitude 0.350–0.500 rad. It is not a 0–12 m/s model. The held
out 8.62 m/s / 0.427 rad condition is inside the joint support; the 8.37/0.302,
9.12/0.302, and 9.37/0.427 conditions are outside at least one training
boundary and were not counted as validation. In particular, they do not
justify extrapolating the surface.

Interpolation checks on training-only capture data withheld an entire speed
anchor or steering knot at a time. Leaving out the middle speed anchor and
interpolating between the two outer anchors gave 0.002285 rad/s RMSE over 32
condition/phase estimates. Leaving out the 0.425-rad steering knot gave
0.00398 rad/s RMSE; leaving out 0.475 rad gave 0.00137 rad/s. These are useful
smoothness checks, but the repeated conditions share one capture and therefore
do not provide run-level uncertainty.

On the separate held-out capture (not used to fit this surface), the in-support
8.62 m/s / 0.427 rad
equilibrium response had 0.00230 rad/s RMSE and 0.00416 rad/s p95 absolute
error over eight reset/sign samples. The legacy equilibrium response on those
same samples was wrong by about 8.35 rad/s. Again, the eight samples are
nested within one capture, not eight independent runs. Important holdout
qualification: this capture had already been used in the earlier ExtraTrees
final-test work, so it is not an unopened model-family holdout. The current
surface was fit only from the separate speed-surface training bag, but these
scores are exploratory evidence, not a sealed final estimate.

For dynamics, a one-pole response was fit to training transitions using only
current sensor-odometry state and steering feedback as model inputs; simulator
truth supplied the next-yaw training label. The fitted time constant is
approximately 0.0120 s. On 206 supported transitions from four held-out probe
sequences, one-step RMSE was 0.00780 rad/s pooled (0.00772 macro-averaged by
sequence), versus 0.01100 for persistence and 6.832 for the production C yaw
model at these conditions.

The same schedule and one-pole state transition were then rolled recursively
through the four complete held-out probe sequences. The C vehicle-model
actuator/longitudinal stages were retained; future recorded sensor state was
not fed to the rollout. On scored in-support event-relative samples:

| Horizon | Existing production yaw RMSE | Scheduled local candidate yaw RMSE |
|---:|---:|---:|
| 100 ms | 3.022 rad/s | outside high-steer support; legacy fallback |
| 250 ms | 8.304 rad/s | 0.02125 rad/s |
| 500 ms | 8.287 rad/s | 0.00648 rad/s |
| 750 ms | 9.176 rad/s | 0.00299 rad/s |
| 1500 ms | 8.286 rad/s | 0.02286 rad/s |
| 2000 ms | 8.286 rad/s | 0.00598 rad/s |

At 25 ms both have zero error at the initial-state point; at 100 ms and during
unsteered intervals the candidate is outside support and falls back to the
production yaw model. The table therefore compares candidate performance only
where its measured joint support applies; it is not a claim that the fallback
is accurate. Four sequences come from one independent capture, so no
run-level confidence interval is claimed. The rollout supports a promising
short local response model for held/outward/reversal high-steer events, but
does not validate the low-angle turn-in from zero to 0.350 rad, arbitrary
steering transitions, practice-track transfer, or a full-lap offline plant.

**Decision:** the playbook's continuous, mechanism-specific scheduling is
viable for this measured high-speed/high-steer yaw mechanism and is materially
better than the current production yaw response in the supported hold region.
The evidence does not support replacing the general production model, extending
the surface to the full speed/steering range, or claiming a full-lap accuracy
gain. Production Y1, MPC defaults, odometry, raceline, and runtime physics were
left unchanged. The specialist has not yet been integrated into MPC/optimizer
parity or tested on a practice lap; it is therefore an offline candidate, not
a production improvement. The held-out score is exploratory because its
capture was previously used for another model family. The remaining
implementation gate is a separate,
opt-in specialist path that preserves Y1, followed by C/CasADi parity and a
collision-watched practice-transfer run. Before that, the model needs measured
low-angle entry coverage or a demonstrated safe/accurate transition from the
existing model into its 0.350-rad support.

The fitted values, support gates, source-bag SHA-256 hashes, and validation
metrics are preserved in the machine-readable, non-runtime artifact
[`yaw_highspeed_highsteer_scheduled_candidate.json`](../../config/racing/yaw_highspeed_highsteer_scheduled_candidate.json).
Focused source-stamp/label/support regression checks pass (`4 passed` via
`python3 -m pytest -q tools/racing/test_fit_race_domain_yaw_residual.py`). No
runtime build or simulator run was performed for this offline experiment.

Runtime integration was deliberately not approximated by swapping a CSV:
the competition MPC currently has one active empirical yaw surface, and its
configured surface is Y1. Replacing that slot would remove the existing
low-speed specialist. The runtime loader also expects a fixed ten-angle/q
table, whereas this candidate is a measured four-angle physical-steering
schedule, and the candidate's one-pole state transition applies through the
tested transient while the current optional surface time-constant hook is a
hold-phase-only blend. A faithful implementation therefore needs a second
surface/model slot with the same equations and derivatives in C and the
optimizer/CasADi path; it must retain Y1 and preserve an explicit support
boundary. The current local result is evidence to justify that implementation,
not a reason to silently replace the working production model.

### 2026-10-07 — locate the full-band ExtraTrees maximum error

Replayed the frozen full-band v1 checkpoint against its already-opened
whole-run validation partition, then compared the worst sample's feature
vector with the exact model-cell training rows. Test/final-test arrays were
not read. The replay reproduced the aggregate maximum absolute error exactly:
**1.220963 rad/s**.

| Quantity | Worst validation sample |
|---|---:|
| Capture | `openplane_dyn_coupled_validation_r02_20261002` |
| Speed | 6.9946 m/s |
| Physical steering | −0.1042 rad |
| Response phase / fitted cell | steady-or-low-rate; cell center 6.75 m/s, −0.100 rad |
| Current yaw rate | −1.3025 rad/s |
| Predicted next yaw rate | −1.3191 rad/s |
| Actual next yaw rate | −0.0981 rad/s |
| Absolute error | 1.2210 rad/s |
| Cell training support | 4 runs, 77 rows |

Thus the point was **not outside the fitted speed/steering/phase cell**. But
one input was outside the training range *within that cell*: steering-command
minus steering-feedback was +0.1184 rad, versus a training range of
−0.0434 to +0.0122 rad. Other model features were within their per-cell
training min/max. This is a fitted-cell prediction on an actuator-mismatch
combination the model had not seen—not evidence that every point in the cell
is covered. It suggests a useful next diagnostic around large command/feedback
disagreement and phase labeling; it does not by itself prove that actuator
lag caused the yaw collapse.

### Data-gap check for that actuator-mismatch condition

Searched the currently admitted clean train and validation archives for this
exact fitted cell (speed center 6.75 m/s, steering center −0.10 rad, phase
steady/low-rate). The current archive discovery has 80 training rows from five
runs in the cell; none has a positive command-minus-feedback gap above
0.05 rad. The frozen v1 checkpoint used 77 of those rows from four runs. The
validation partition has 67 rows from four runs, but only four samples exceed
0.08 rad of positive gap, across two captures; these validation examples are
diagnostic evidence, not adequate training coverage.

**Specific missing data:** repeated 25 ms transitions near 6.8–7.0 m/s and
physical steering near −0.10 rad (plus the mirrored +0.10-rad case), where
steering-command minus measured-steering-feedback spans near zero through
approximately +0.12 rad and the mirrored negative mismatch. Capture the
command and physical steering histories/rates, current and next GT yaw rate,
current yaw history, throttle command/feedback, rear-wheel speeds, and body
speed/lateral speed together. Include command reversals/unwinds that create the
mismatch naturally; label phase from both physical steering motion and
command-versus-feedback error rather than treating low measured steering rate
as sufficient evidence of steady hold.

The existing fit gate is at least 40 transitions in a cell from at least two
contributing runs; here the important unit is independent repeated mismatch
events across runs, not hundreds of adjacent frames from one event. Fit using
training captures only and retain a new whole-run holdout for this mismatch
condition. Score one-step and recursive yaw prediction and confirm the
near-zero-mismatch behavior does not regress. The four existing validation
examples can guide the target profile, but must not be relabeled as fresh
holdout data. No new simulation was started for this gap check.

### 2026-10-07 — review of `Yaw_fitting.txt` and causal multi-horizon yaw teacher

The external proposal was read end-to-end. Its core idea is accepted as a
research direction: model yaw-rate trajectories from causal sensor/actuator
history and the future *candidate commands* known to MPC, then distill only a
validated teacher to a differentiable controller model. It is not accepted as
permission to ensemble every model family or to integrate an unvalidated
teacher directly into production.

The error audit adds a concrete mechanism to test. Across the frozen ExtraTrees
validation transitions, grouped by steering command-minus-feedback gap:

| Absolute steering command-feedback gap | Samples | Yaw-rate RMSE | Fraction below 0.1 rad/s | Maximum error |
|---|---:|---:|---:|---:|
| <0.025 rad | 104,971 | 0.0225 rad/s | 98.98% | 0.847 rad/s |
| 0.025–0.05 rad | 5,672 | 0.0932 rad/s | 89.74% | 0.929 rad/s |
| 0.05–0.10 rad | 2,389 | 0.1305 rad/s | 87.15% | 1.062 rad/s |
| ≥0.10 rad | 876 | 0.2961 rad/s | 58.56% | 1.221 rad/s |

These are held-out validation data, not proof of causality. For the worst row,
the command changes from −0.104 to +0.014 rad while physical steering remains
−0.104 rad for one sample; at the next sample physical steering catches up and
yaw rate changes from −1.303 to −0.098 rad/s. Same-packet simulator heading
change and IMU yaw-rate labels change in the same interval, so this is a real
command/actuator-history transition in the capture, not evidence of a dropped
row. The current exact-cell model uses current mismatch but has no training
examples for this mismatch in that speed/steering cell. The decisive gap is
therefore actuator-transition history and support, not simply “another
speed/steering bin.”

The old ExtraTrees atlas remains a useful diagnostic comparator, but its input
features include same-time simulator-truth yaw rate, speed, and lateral
velocity; it is **not** an online sensor-only predictor. A first offline-only
teacher fit was stopped and invalidated when its feature audit found that it
consumed `frames[:,0:3]`. Those fields are simulator-oracle
`/autodrive/roboracer_1/odom`, not production sensor odometry. Its generated
models and report were removed; none of its metrics are valid or promoted. The
corrected fit is being prepared from current/past encoder, IMU,
actuator-feedback/command, roll and roll-rate channels, plus only the planned
steering/throttle commands through each prediction horizon. Future measured
feedback and future truth are never inputs. It compares current-sample versus
up-to-1.6 s causal history at 25, 100, 250, 500, 750, and 1000 ms, holding out
whole captures. Test/final-test arrays remain unopened. The corrected feature
set intentionally excludes body `u/v`: the body-velocity fields stored in
these archives come from the simulator oracle.

The audit also found that a clean combined archive containing five training and
five validation runs had been skipped by the older “one split per archive”
loader. Its five training captures were already represented by pure-split
archives; the five clean validation captures were not. The yaw teacher now
admits only mixed archives whose manifest contains train and validation **and
no test/final-test**, verifies each run's split and quality gate before reading
its numeric arrays, and deduplicates by run ID and fingerprint. This increases
whole-run validation from 18 to 23 captures. The combined archive's five
validation views have 2–5 fewer rows than their per-run manifest counts (under
0.1%); those source/archive count differences are recorded and archive sequence
bounds are used—no data is padded or interpolated.

#### Disposition of every major proposal in `Yaw_fitting.txt`

| Proposal | Decision | Evidence / boundary |
|---|---|---|
| Predict a full future yaw-rate trajectory conditioned on past history and candidate future controls | **Promote; test underway** | Directly matches the MPC question and exposes short- versus long-horizon error. Future commands are permitted model inputs; future sensor/truth samples are forbidden. |
| Retain all causal yaw-relevant signals, then ablate | **Promote** | Corrected teacher uses current/past steering/throttle feedback and commands, wheel speeds, IMU acceleration/yaw, roll/rate. Stored body `u/v` is oracle-derived and excluded. Per-signal and grouped ablations follow only if this baseline improves. |
| Define experts by mechanism, with continuous speed conditioning | **Promote as model organization, not as a fixed seven-expert commitment** | Y1 has strong narrow high-steer evidence; the atlas shows many supported cells but large transition errors. Do not create Y0–Y6 experts without a held-out gain for each. |
| Add a sticky HMM/HSMM or learned temporal gate | **Defer** | The diagnostic supports missing actuator-transition history; no evidence yet that a learned gate beats causal history plus a small explicit phase model. Gate flicker/latency would add MPC risk. |
| Benchmark TCN, GRU/LSTM, Transformer, ExtraTrees, GP and CNP | **Adapt** | First compare a causal tree baseline and a recursive SUBNET/RSSM comparator on identical run splits. Add temporal neural families only if history matters and tree error remains structured. Local GP/CNP are not justified as the first step by sparse independent runs. |
| Direct finite-horizon predictor in addition to recursive dynamics | **Promote** | The new teacher directly predicts six horizons; it does not recursively feed future recorded measurements back. Recursive free-run remains a separate required evaluation before calling it a plant. |
| Direct and dynamic heads sharing a history encoder | **Defer until direct-vs-recursive comparison** | This is useful only if both solve distinct measured failure modes; two heads are not automatically an improvement. |
| Keep ExtraTrees as a teacher/comparator | **Promote as a frozen comparator; reject it as the teacher target** | Its open-run one-step performance is useful for diagnosis, but current features include truth-derived state and its maximum held-out error is 1.221 rad/s. Train against simulator truth, not ExtraTrees predictions. |
| Trajectory, peak timing/magnitude, shape, and reversal losses | **Promote as evaluation criteria; defer custom loss fitting** | Report direct yaw-rate error at each horizon, bias, p95/p99/max, per-run and speed/steer strata first. Peak/zero-cross metrics are added when their labels and events are unambiguous. |
| Heavy-tailed probabilistic output | **Defer** | First establish point-prediction error and determine whether the 25 ms response is predictable from allowed history/commands. Uncertainty cannot hide a large point error; calibration requires separate held-out captures. |
| Exhaustive history study from 100 ms to 4 s | **Adapt/promote** | The first comparison uses current state and a sparse 0–1.6 s history. If history helps, extend to 2–4 s and fit regime-specific history length under grouped whole-run validation; do not brute-force every combination. |
| Exhaustive causal-signal ablation | **Promote after the history baseline** | Roll and wheel/sensor groups are included. Remove each group using identical run splits and compare trajectory metrics, not training loss. |
| Steady sweeps, chirp/multisine, turn-in, unwind, reversal, longitudinal-coupling captures | **Conditional; only fill demonstrated gaps** | Existing clean archives already include steering/throttle dynamic captures and whole-run validation. The exact 6.8–7.0 m/s, ±0.10 rad, ≥0.10 rad command-feedback transition is missing from training. Do not run all proposed experiments unless residual/gap analysis shows they are needed. |
| Ensemble-disagreement active learning | **Promote after at least two independently useful teachers exist** | Current atlas support and validation errors identify gaps, but there is not yet a calibrated ensemble disagreement score. New runs must target a decision-relevant gap, not a rectangular sweep. |
| Per-horizon, per-regime metrics with equal weight per run | **Promote** | Existing atlas reports run-level uncertainty; the new report includes six horizons, run-macro/worst-run metrics, speed/steering strata, p95/p99/max, and under-0.1/0.05 fractions. |
| Whole-capture generalization tests | **Promote** | Comparisons use the existing train/validation split at whole-run level. Final-test captures remain sealed and will not be used to choose features or checkpoints. |
| Assume a different best model for every regime and ensemble them | **Reject as an assumption; test only as a later hypothesis** | The external proposal names candidate experts without results on this simulator. Select a mixture only if run-held-out gains and support boundaries beat the strongest single teacher. |
| Distill the teacher to a compact differentiable MPC model | **Promote as the deployment path, but not yet** | A tree teacher is non-differentiable. Distillation is allowed only after causal inputs, multi-horizon accuracy, support/uncertainty, and recursive or command-conditioned free-run checks pass. Preserve production Y1 while adding any proven specialist. |
| Hierarchical desired-yaw-rate control | **Defer outside this identification decision** | It changes control architecture rather than identifying the observed yaw response. Revisit only if an accurate yaw teacher still cannot be used by the existing MPC. |

The source papers support trying history-based identification and local
model-error augmentation, not assuming success on this vehicle. SUBNET has
reported F1TENTH identification using measured data
([Szécsi et al., IFAC 2024](https://doi.org/10.1016/j.ifacol.2024.08.542));
the CNP yaw-rate paper studies high-fidelity simulation scenarios and different
vehicle conditions
([Ullrich et al., CDC 2023](https://arxiv.org/abs/2407.06605)); and the cited
2026 gated-LSTM mixture paper evaluates NARMA-10, not an RC vehicle
([Rajabi & Salehi, 2026](https://arxiv.org/abs/2608.08980)). The autonomous
racing error-dynamics paper likewise supports a nominal-model-plus-local-error
architecture, but does not validate this model
([Xue et al., ICRA 2024](https://publications.ri.cmu.edu/learning-model-predictive-control-with-error-dynamics-regression-for-autonomous-racing)).
These references justify experiments; only this repo's held-out data can
promote a model.

At that earlier checkpoint, no model had been integrated into odometry or
MPC, no production model or raceline had been changed, and the targeted
capture had not yet started. The subsequent 2026-10-08 audit and capture
status are recorded below; they supersede the then-future tense in this
checkpoint.

### 2026-10-08: full >0.1 rad/s error audit and targeted replication

The threshold in this section is **absolute yaw-rate prediction error in
rad/s**, not yaw-angle error in rad. The audit uses the corrected v3 direct
multi-horizon teacher and its 32 whole-run validation captures. At the 250 ms
horizon, 27 speed-by-absolute-steering cells with at least 100 held-out rows
have p95 error above 0.1 rad/s. The errors are not confined to a single speed
or to high steering: the model has a broad residual tail across 1.5–10 m/s and
0–0.525 rad.

| 250 ms region | Held-out p95 error | Train samples / independent runs | Interpretation |
|---|---:|---:|---|
| 1.5–2 m/s, 0.10–0.20 rad | 0.242 | 1,452 / 8 | Thin independent-run support; data may help. |
| 1.5–2 m/s, 0.20–0.35 rad | 0.305 | 3,088 / 6 | Few independent runs despite many adjacent samples; transitions/regime coverage are the gap. |
| 1.5–2 m/s, 0.35–0.525 rad | 0.512 | 1,831 / 5 | Clear support gap and large residual; directly targeted by the current capture. |
| 2–3 m/s, 0.10–0.20 rad | 0.391 | 1,460 / 12 | Errors persist with moderate run support; more steady rows alone are unlikely to solve it. |
| 2–3 m/s, 0.20–0.35 rad | 0.362 | 5,188 / 15 | Not a simple lack-of-data case. |
| 2–3 m/s, 0.35–0.525 rad | 0.625 | 4,219 / 13 | Highest tail; high-steer replication tests transfer/repeatability. |
| 3–4 m/s, 0.20–0.35 rad | 0.373 | 4,677 / 16 | Dense enough to reject raw sample count as the sole cause. |
| 4–6 m/s, 0.10–0.20 rad | 0.239 | 9,582 / 27 | Strong evidence of response/model mismatch, not a lack of ordinary samples. |
| 6–8 m/s, 0.10–0.20 rad | 0.216 | 13,649 / 26 | Same: many runs and samples, persistent tail. |
| 4–8 m/s, 0–0.10 rad | 0.175–0.185 | 45k–56k / 35–42 | Very dense cells still miss the 0.1-rad/s p95 target. |
| 8–10 m/s, 0–0.20 rad | 0.148–0.173 | 10k–48k / 16–30 | Mostly supported, residual remains structured. |

The exact cell audit (including all cells, sample counts, independent run and
sequence counts, and steering/throttle/wheel-speed-mismatch cross-tabs) is
[`expanded_error_support_audit_v3_20261008.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_multihorizon_teacher_v3_residual_grid/expanded_error_support_audit_v3_20261008.json).
The most defensible conclusion is mixed: thin independent-run coverage is a
real issue around 1.5–2 m/s, especially at moderate/high steering, but the
largest broad-band errors remain after thousands of rows and 10–40 runs.
That points to missing response/history structure and rare transitions, not
merely insufficient grid density.

The largest 250 ms error examples expose at least two different mechanisms:

* At 3.76 m/s and 0.20 rad, the prediction is 1.074 versus 2.694 rad/s
  (error −1.620 rad/s), with 8.72 m/s² lateral acceleration and about
  1.98 m/s rear-wheel/body-speed mismatch. Slip is plausible in this event.
* In a separate held-out high-steer reversal at 3.47 m/s and −0.42 rad,
  prediction is 0.186 versus −1.421 rad/s (error +1.607 rad/s), despite
  steering command/feedback gap ≈0.0001 rad, throttle gap ≈0.0004, and
  wheel/body mismatch ≈0.056 m/s. Slip or actuator tracking alone cannot
  explain that wrong-sign response.

This distinction informed a finite, reset-isolated open-plane test rather than
another broad sweep. Profile `yaw_error_lowspeed_highsteer_replication` has
144 scored probes per capture, crossing 1.5/2.5/3.5 m/s with 0.35/0.42/0.50
rad, onset and reversal, left/right turn, step and 0.30 s ramp, and two event
ages; phase order is randomized. It records sensor/actuator feedback, wheel
encoders, IMU, commands, packet timing, and simulator truth for offline labels.
The two training captures completed (r01: 21,869 samples/148 sequences; r02:
21,796/145), each with all 144 probes, zero collisions/timing faults/packet
sequence gaps, and approximately 39.96 Hz data with p95 intervals near 26 ms.
Independent validation r03 completed all 144 probes and passed the whole-bag
gate: 21,540 prepared samples, 154 reset/packet sequences, zero collisions or
timing/quality failures, and exact packet alignment. It was not used for
fitting. On this high-steer capture, frozen v3 current-only ExtraTrees at
250 ms scored RMSE 0.186, p95 0.503, and 71.5% within 0.1 rad/s. Its worst
sub-bins were 1.5–2 m/s at 0.20–0.35 rad (p95 0.482) and 0.35–0.525 rad
(0.563), plus 2–3 m/s at 0.20–0.35 rad (0.560). These overlap the thin
training-support cells above, so a train-only refit using r01/r02 is
warranted; this does not justify another broad steady-state sweep. The capture
used the established batch-mode Explore simulator; no production physics,
Odom, MPC, or raceline was changed.

### What the learned models have—and have not—shown

There is measurable progress from temporal models, but no full-spectrum,
deployment-ready predictor yet.

* On the 24 historical whole-run validation captures shared with the tree
  model, the best checkpoint of the causal direct-yaw GRU has run-macro RMSE
  0.0497, 0.0457, 0.0458, 0.0478, 0.0507, and 0.0567 rad/s at 25, 100, 250,
  500, 750, and 1,000 ms. The corresponding v3 ExtraTrees values are
  0.0536, 0.0657, 0.0685, 0.0679, 0.0670, and 0.0695. The GRU is better on
  most shared runs, particularly at 100–1,000 ms. However, its largest errors
  remain 1.1–1.9 rad/s, and that checkpoint has not yet been scored on the
  newer error-conditioned holdouts. Its current longer training job is still
  running; the values above are its best checkpoint so far, not a final
  promoted result.
* Retraining the ExtraTrees teacher after adding the two residual-grid train
  captures improved the new targeted validation r03 (250 ms RMSE 0.122→0.081,
  p95 0.270→0.161 rad/s) but worsened mean RMSE on most of the 24 shared old
  validation runs. It is therefore a local gain, not a global model
  improvement. A 1.6 s history variant is also worse overall, although
  localized high-steer results motivate a sensor-gated history comparison.
* The very large throttle-response dataset is usable and has already supported
  model experiments; it must not be dismissed. Its sequence archive has 1,508
  reset-isolated sequences and 725,834 samples, including 770 sequences from
  r04 and 738 from a separate r05 capture. The structured-GRU model improved
  held-out r05 yaw-rate RMSE over the earlier mixed GRU: 0.251 vs 0.319 rad/s
  at 250 ms, 0.417 vs 1.473 at 1 s, and 0.685 vs 1.810 over the common
  11.475 s trajectory. That is real progress, but still not near 0.1 rad/s;
  its whole-trajectory position error was also about 10.72 m. The test has only
  13 steering-angle bootstrap clusters and is not a full speed-by-steering
  generalization proof. The source-aligned GRU report and independent-capture
  comparison are in
  [`throttle_surface_sourcealigned_gru_11s_20260930`](../../live_runs/derived_dynamics_learning_20260928/throttle_surface_sourcealigned_gru_11s_20260930/training_report.json)
  and
  [`r05_external_model_comparison_20260930.json`](../../live_runs/derived_dynamics_learning_20260928/throttle_surface_40hz_sourcealigned_dataset_20260930/r05_external_model_comparison_20260930.json).
* Earlier direct-sequence Transformer/RSSM and hybrid plant teachers were
  evaluated on practice transfer and free-running segments. None has yet
  established small full-lap motion error; they remain offline comparators,
  not usable production plants.

The same audit also exposed a distinct crawl-speed support hole: for 0.5–1.0
m/s with 0.1–0.5 rad steering, several speed/steer cells have only 13–16
training samples across 3–4 runs (4–5 sequences), while held-out p95 reaches
0.27–1.10 rad/s. In the 1.0–1.5 m/s band, support rises to 23–40 samples
across 5–6 runs (8–16 sequences), but held-out p95 still reaches 0.26–1.38.
The active 1.5–3.5 m/s suite does not hold the car in that band; its approach
is straight. A second finite profile, `yaw_error_crawl_steering_gapfill`, is
now implemented for 0.60/1.00 m/s × 0.20/0.35/0.50 rad with the same randomized
onset/reversal, both-sign, step/ramp and two-delay design (96 probes/288
phases per capture). Its focused schedule test, runner syntax check, and help
registration pass. The first launch revealed its schedule minimum (1,699.4 s)
exceeded the generic 1,200 s cap; that rejection happened before any probe.
The cap is now profile-specific at 3,600 s. Crawl training capture r01 is
currently running with the established batch-mode Explore simulator, using
reset-isolated probes; if it passes admission, follow with an independent
training repeat and a held-out validation capture.

The updated support/error audit including the new high-steer r03 is
[`expanded_error_support_audit_v4_20261008.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_multihorizon_teacher_v3_residual_grid/expanded_error_support_audit_v4_20261008.json).
Across all admitted validation runs, current-only v3 at 250 ms scores RMSE
0.0929, p95 0.1902, and 86.8% within 0.1 rad/s; its 1.6 s history tree is
worse (0.0988 RMSE, 0.2129 p95, 84.3% within). Longer tree history alone does
not solve the broad residual.

Next: finish and quality-admit crawl r01, then capture r02 and independent
r03; fit a separate train-only ExtraTrees candidate using the targeted
training captures; compare v3 and the candidate on identical historical and
new whole-run validation, with per-regime errors and run-level deltas. Finish
the in-progress causal GRU training and score its frozen checkpoint on new
held-out captures it has not seen. Keep candidates offline-only unless they
improve the relevant held-out regimes without materially degrading broad
validation. No model has been integrated into Odom or MPC.

`score_yaw_gru_capture.py` provides the whole-run, validation-only scoring
entry point, with per-horizon speed-by-steering error bins; truth speed is used
only to label diagnostic bins. It rejects training and checkpoint-selection
runs and does not feed future sensor or truth values to the model. Its
synthetic regime-binning check passes. No model has been integrated into Odom
or MPC.

#### Leakage audit correction

The first teacher attempt was interrupted before any result could be used.
The source was checked against `signal_semantics.py` and the archive writer:
`frames[:,0:3]` are the simulator `/autodrive/.../odom` twist. That stream is
an oracle label/debug stream and is not an admissible controller or observer
input. The model files/report from that attempt were deleted; source bags and
datasets were not changed. In particular, a zero error from comparing
`frames[:,2]` with simulator rigid-state yaw rate is a self-comparison and is
excluded from all production-accuracy claims. The separate IMU-yaw versus
simulator-label characterization remains an offline sensor diagnostic only.

#### Corrected direct-teacher fit and gap diagnosis

The corrected fit completed using 29 eligible training captures (four short
captures were excluded by the 1.6 s history + 1 s target requirement) and all
23 eligible whole-run validation captures. No test/final-test arrays were
opened. Inputs were only steering/throttle feedback and commands, rear wheel
speeds, IMU acceleration/yaw, and IMU roll/roll-rate history; labels were future
simulator rigid-body yaw rate. `frames[:,0:3]` did not enter the features.

| Horizon | Current-only RMSE | 1.6 s history RMSE | IMU-persistence RMSE | Best under 0.1 rad/s | Worst error (current / history) |
|---:|---:|---:|---:|---:|---:|
| 25 ms | 0.0572 | **0.0515** | 0.0971 | 94.8% | 1.259 / 1.076 |
| 100 ms | **0.0702** | 0.0770 | 0.3035 | 89.6% | 1.964 / 1.673 |
| 250 ms | **0.0754** | 0.0805 | 0.5388 | 88.6% | 1.727 / 1.869 |
| 500 ms | **0.0769** | 0.0803 | 0.7350 | 88.6% | 1.826 / 1.980 |
| 750 ms | **0.0748** | 0.0813 | 0.8485 | 88.7% | 1.994 / 1.952 |
| 1000 ms | **0.0770** | 0.0792 | 0.9231 | 87.6% | 1.967 / 1.950 |

The 25 ms history model beat the current-only model on 21/23 validation runs;
the run-paired 95% bootstrap CI for mean RMSE change was [−0.00955,
−0.00483] rad/s. At 100–750 ms the current-only model had lower mean RMSE;
paired run CIs exclude zero at 100, 250, and 750 ms. The 500 ms and 1000 ms
differences are inconclusive. Both model variants beat current-IMU persistence on all 23 runs at
every horizon. This establishes useful command-conditioned forecasting over
persistence, but emphatically **does not** meet a maximum-error target of 0.1
rad/s: the worst errors are 1.1–2.0 rad/s and wrong-sign predictions remain.

Detailed errors, grouped by speed/steering/command-feedback mismatch and the
top held-out samples, are retained in
[`yaw_multihorizon_teacher_error_diagnosis.json`](../../live_runs/racing_model_diagnostics_20261007/yaw_multihorizon_teacher_v1/yaw_multihorizon_teacher_error_diagnosis.json).
The most important failures are:

1. At 25 ms, the worst sensor-only error is 1.259 rad/s in
   `openplane_dyn_coupled_validation_r01`, at 4.98 m/s with steering feedback
   0.0036 rad and command 0.2824 rad. The clean training set has **zero** rows
   in the matched 4.5–5.5 m/s, |feedback|<0.05 rad, 0.20–0.35 rad command
   condition. For |command−feedback|≥0.1 rad, current-only RMSE rises to
   0.252 rad/s (1,561 samples), versus 0.031 rad/s below 0.025 rad mismatch.
2. At 100–1000 ms, the worst repeated error occurs in
   `openplane_race_domain_validation_20261001_r04` around 3.47 m/s and +0.42
   rad feedback while the planned command reverses to −0.42 rad. Training has
   **zero** matching 3–4 m/s full-steer cross-sign command/feedback rows; only
   four high-mismatch rows exist across one unrelated train run. Both models
   predict the wrong yaw sign through this reversal.

This is enough evidence to justify a narrowly scoped new capture; it is not a
request for another broad sweep. The new schedule
`yaw_mismatch_transition_train` has 16 shuffled, reset-isolated probe
conditions per capture: 5.0 m/s neutral-to-±0.28 rad steering onset and 3.5
m/s ±0.42 rad full reversal, each as a one-tick step and a 0.30 s ramp, both
directions, two repeats. Each condition has a matched speed approach and
settle; the recorder retains actual steering/throttle feedback, wheel speeds,
IMU, and command streams. The plan is two independently seeded captures:
r01 assigned `train`, r02 assigned `validation` by explicit whole-run split
override. No test/final-test data will be reused. The profile's focused
schedule tests passed (2 tests). This describes the 2026-10-07 plan; the
2026-10-08 low-speed/high-steer suite was run instead and is detailed above.

Historical 2026-10-07 decision: **promote the causal direct multi-horizon yaw
forecaster as a useful offline MPC research oracle; reject the claim that the
then-current tree teacher is accurate/trustworthy over all regimes or ready
for odometry/MPC integration.** Longer history alone was not promoted for
horizons beyond 25 ms. The 2026-10-08 results and follow-up decisions are in
the dated section above.

### 2026-10-08: error-support audit, cadence correction, and finite gap tests

This is the latest status; older statements above that crawl r01 is still
running are superseded here.

#### What the held-out error map says about data scarcity

The audited threshold is **0.1 rad/s future yaw-rate error**, not 0.1 rad yaw
angle. On the current-only v3 model at 250 ms, 27 speed-by-|steering| cells
with at least 100 held-out rows have p95 above 0.1 rad/s. Only two of those
cells have fewer than 100 training rows:

| Speed and |steering| | Validation rows; p95 | Training rows / runs / sequences | Interpretation |
|---|---:|---:|---:|---|
| 0–0.5 m/s, 0.2–0.35 rad | 134; 0.141 rad/s | 88 / 3 / 5 | Sparse; not yet sampled by the crawl profile's measured speed (its 0.60 m/s target approaches at about 0.52 m/s). |
| 0–0.5 m/s, 0.35–0.525 rad | 133; 0.110 rad/s | 90 / 2 / 4 | Sparse and marginally over threshold; determine whether stable sub-0.5 m/s steering is a meaningful operating regime before expanding tests. |

The most severe cells are not explained by low raw sample count. At 250 ms,
1.5–2 m/s and 0.35–0.525 rad has p95 0.558 with 5,759 training rows / 7
runs / 96 sequences; 2–3 m/s at 0.35–0.525 has p95 0.444 with 8,343 / 15 /
154; 2–3 m/s at 0.2–0.35 has p95 0.392 with 7,949 / 17 / 359; and 3–4 m/s
at 0.2–0.35 has p95 0.344 with 7,009 / 18 / 196. The low-speed/high-steer
replication and crawl captures target missing transient coverage, but more
steady samples in these already populated cells are not justified.

Full 250 ms current-only v3 error map for all cells with at least 100 held-out
samples and p95 above 0.1 rad/s (v4 audit). “Over 0.1” is the fraction of
held-out rows whose absolute error exceeds that threshold.

| Speed (m/s) | |steering| (rad) | Validation rows / runs / sequences | p95 (rad/s) | Rows >0.1 | Training rows / runs / sequences |
|---|---:|---:|---:|---:|---:|
| 0–0.5 | 0.2–0.35 | 134 / 4 / 7 | 0.141 | 9.7% | 88 / 3 / 5 |
| 0–0.5 | 0.35–0.525 | 133 / 3 / 6 | 0.110 | 7.5% | 90 / 2 / 4 |
| 1.5–2 | 0–0.1 | 6,016 / 29 / 1,027 | 0.200 | 9.8% | 10,645 / 42 / 1,718 |
| 1.5–2 | 0.1–0.2 | 767 / 4 / 82 | 0.302 | 45.6% | 1,543 / 10 / 162 |
| 1.5–2 | 0.2–0.35 | 2,517 / 7 / 88 | 0.338 | 85.5% | 5,080 / 8 / 168 |
| 1.5–2 | 0.35–0.525 | 2,868 / 6 / 53 | 0.558 | 58.2% | 5,759 / 7 / 96 |
| 2–3 | 0–0.1 | 12,711 / 31 / 1,004 | 0.160 | 8.2% | 21,938 / 43 / 1,676 |
| 2–3 | 0.1–0.2 | 982 / 11 / 188 | 0.640 | 43.4% | 1,794 / 14 / 362 |
| 2–3 | 0.2–0.35 | 4,196 / 14 / 188 | 0.392 | 68.3% | 7,949 / 17 / 359 |
| 2–3 | 0.35–0.525 | 4,582 / 13 / 79 | 0.444 | 21.1% | 8,343 / 15 / 154 |
| 3–4 | 0.1–0.2 | 2,135 / 14 / 125 | 0.223 | 14.9% | 3,299 / 19 / 228 |
| 3–4 | 0.2–0.35 | 3,865 / 15 / 113 | 0.344 | 21.7% | 7,009 / 18 / 196 |
| 3–4 | 0.35–0.525 | 3,702 / 8 / 53 | 0.360 | 21.9% | 6,784 / 11 / 97 |
| 4–6 | 0–0.1 | 25,561 / 30 / 761 | 0.176 | 9.0% | 44,984 / 42 / 1,170 |
| 4–6 | 0.1–0.2 | 5,362 / 17 / 192 | 0.239 | 32.3% | 9,582 / 27 / 291 |
| 4–6 | 0.2–0.35 | 7,085 / 18 / 165 | 0.234 | 16.4% | 10,032 / 23 / 197 |
| 4–6 | 0.35–0.525 | 3,153 / 11 / 77 | 0.240 | 18.2% | 5,078 / 12 / 82 |
| 6–8 | 0–0.1 | 39,314 / 28 / 667 | 0.185 | 14.7% | 56,249 / 35 / 934 |
| 6–8 | 0.1–0.2 | 9,127 / 23 / 335 | 0.216 | 24.6% | 13,649 / 26 / 376 |
| 6–8 | 0.2–0.35 | 15,704 / 21 / 274 | 0.147 | 8.6% | 16,778 / 22 / 271 |
| 6–8 | 0.35–0.525 | 11,821 / 16 / 126 | 0.114 | 5.9% | 11,561 / 18 / 118 |
| 8–10 | 0–0.1 | 27,677 / 23 / 431 | 0.147 | 12.4% | 48,051 / 30 / 658 |
| 8–10 | 0.1–0.2 | 5,954 / 16 / 191 | 0.173 | 11.3% | 9,976 / 22 / 319 |
| 8–10 | 0.2–0.35 | 3,099 / 17 / 101 | 0.129 | 6.6% | 5,334 / 17 / 166 |
| 8–10 | 0.35–0.525 | 1,925 / 10 / 45 | 0.121 | 5.3% | 5,118 / 11 / 92 |
| 10–12 | 0–0.1 | 24,662 / 18 / 201 | 0.147 | 11.9% | 29,328 / 18 / 269 |
| 10–12 | 0.1–0.2 | 8,602 / 18 / 157 | 0.129 | 7.8% | 11,301 / 18 / 201 |

There are 32 speed/steering cells with at least 100 held-out rows; all 32
contain at least one >0.1 rad/s residual, but five have p95 at or below 0.1
and therefore only a rarer tail: 0–0.5 m/s near-straight (p95 .0012, max
.908), 0.5–1 m/s near-straight (p95 .049, max .530), 1–1.5 m/s
near-straight (p95 .094, max .670), 3–4 m/s near-straight (p95 .087, max
.750), and 10–12 m/s at 0.2–0.35 rad (p95 .0998, max .364). This separates
“some outlier exists” from a regime whose 95th-percentile error itself fails.

A more specific cross-cut does reveal a separate plausible data hole: in
3–4 m/s, 0.2–0.35 rad, and rear wheel/body speed mismatch above 1 m/s, the
training set has 78 rows / 13 runs / 26 sequences; held-out support is 80
rows / 13 runs and p95 is 0.695 rad/s. A second related slice, 2–3 m/s,
0.35–0.525 rad, and 0.25–0.5 m/s wheel/body mismatch, has 247 training rows /
6 runs / 45 sequences; held-out p95 is 0.659 rad/s over 147 rows. These are
conditional support gaps inside populated broad speed/steering cells. They
are not the same failure as wrong-sign high-steer reversals with little
wheel/body mismatch. Do not add a mid-speed wheelspin run until the train-only
refit and separate GRU evaluation establish that these conditional residuals
persist.

In contrast, broad near-straight errors at 4–8 m/s and the 4–6 m/s,
0.1–0.2-rad region have thousands to tens of thousands of training samples
over many independent runs. Those remain model/latent-transition problems,
not a lack of generic operating-point coverage. Steering command/feedback
mismatch worsens error but does not explain high-steer failures when the
feedback matches the command.

#### Finite capture progress

The first crawl capture, `openplane_yaw_error_crawl_steering_train_r01_retry2_20261008`,
completed 288/288 phases and all 96 probes, with no collision, timing-fault,
invalid phase, or quality failure. Its admitted training export contains
13,082 rows in 98 reset-delimited sequences with exact packet alignment
(13,662/13,662); manifest and phase gates pass. The actual approach speeds
were approximately 0.52 m/s for the 0.60 target and 0.91 m/s for the 1.00
target. Independent train r02 completed the same 288 phases / 96 probes with
zero collision or quality failure; its export has 13,079 rows / 99 sequences
and 100% packet alignment (13,658/13,658). Held-out validation r03 also
completed 288/288 phases and 96/96 probes with zero collision or quality
failure; its validation export has 13,201 rows / 98 sequences and 100% packet
alignment (13,683/13,683). Its manifest split is explicitly `validation`,
and it has not been used for fitting or checkpoint selection.

#### Corrected active cadence and a real timing observation

`tools/analyze_open_plane_dynamics.py` previously averaged the entire bag,
including the deliberate reset pauses between 288 phases, so it reported a
false low rate. It now calculates cadence inside phase intervals, excludes
between-phase/reset gaps, and retains the existing 38 Hz, 35 ms p95, and 60
ms max-gap gates. The focused synthetic check verifies both that reset gaps
do not lower a 40 Hz active rate and that an actual in-phase stall still fails
the max-gap test (2 checks pass).

On r01/r02/r03, active odometry rates were 39.956/39.960/39.949 Hz, p95
receipt intervals 26.11/26.10/26.05 ms, and maxima 128.41/122.00/103.89 ms.
Odometry had 9/9/7 intervals over 60 ms in the active phases; other streams
showed 7–10. Packet sequence IDs remain contiguous and request cadence is
about 25 ms, but individual responses have 62–126 ms source/arrival gaps and
92–167 ms request/response latency in r01. The r03 capture was collected with
the GRU training process temporarily suspended; its lower maximum/over-limit
count is consistent with reduced host contention, but three runs are not
enough to claim causality. Each run passes packet-continuity and collision
admission but **fails the stricter active max-receipt-gap cadence gate**. The
rare stalls cannot explain broad p95 model-error tails; they can contribute
some maximum outliers. Do not describe the source as perfectly regular 40 Hz.

#### Learned-model status and next action

An earlier causal GRU benchmark on 24 shared whole-run validation captures
did improve run-macro yaw-rate RMSE versus ExtraTrees at every measured
horizon: 25/100/250/500/750/1000 ms were approximately
0.0497/0.0457/0.0458/0.0478/0.0507/0.0567 rad/s versus
0.0536/0.0657/0.0685/0.0679/0.0670/0.0695. At 250 ms that is about a 33%
RMSE reduction. This is meaningful partial forecasting improvement, not a
maximum-error guarantee: historical outliers remained above 1 rad/s, and it
has not been tested on the new sealed crawl/high-steer captures or integrated
into Odom/MPC.

For the retraining's current best checkpoint (epoch 70), the exact 24-run
validation metrics were:

| Horizon | Run-macro RMSE (rad/s) | Pooled RMSE (rad/s) | Pooled p95 abs error (rad/s) | Within 0.1 | Max abs error (rad/s) |
|---:|---:|---:|---:|---:|---:|
| 25 ms | 0.0497 | 0.0558 | 0.0995 | 95.03% | 1.098 |
| 100 ms | 0.0457 | 0.0528 | 0.0789 | 96.61% | 1.737 |
| 250 ms | 0.0458 | 0.0546 | 0.0799 | 96.46% | 1.691 |
| 500 ms | 0.0478 | 0.0559 | 0.0806 | 96.36% | 1.861 |
| 750 ms | 0.0507 | 0.0603 | 0.0841 | 96.06% | 1.684 |
| 1000 ms | 0.0567 | 0.0643 | 0.0913 | 95.66% | 1.836 |

This resolves an apparent ambiguity: the GRU's **95th-percentile** error is
below 0.1 rad/s on those historical validation runs, but its worst errors are
still 1.1–1.9 rad/s. It is a useful broad predictor with a significant
failure tail, not an everywhere-accurate observer. The 24 runs were used for
checkpoint selection, so the crawl r03 and high-steer r03 scores are the
important untouched generalization checks.

The trajectory GRU retraining has a saved best checkpoint at epoch 70 and
validation history through epoch 76. Its report remains `running`; the process
was resumed after r03 closed, and no final-evaluation report exists yet. The
tree v4 train-only refit was then launched after all three crawl manifests
were admitted. Both remain in progress; do not call either result complete or
promote them. The v3 residual map above remains the test-selection baseline
until the new held-out scoring finishes.

Next sequence: complete and admit crawl r02 as training; capture/admit crawl
r03 as validation; train the fixed v4 candidate using only existing and new
training runs; compare it and the completed GRU on identical held-out runs,
including per-run and speed/steering/wheel-mismatch cuts. Only if the
3–4 m/s wheel-mismatch residual persists should one additional finite
step-versus-ramp throttle test be run there. Do not collect more broad
high-steer or high-speed steady data: their dominant failing cells already
have adequate raw support. Keep all models research-only until they pass
whole-run and recursive/trajectory evaluation.

The remaining test plan is deliberately conditional and finite, not an open
ended sweep:

| Trigger after v4/GRU held-out scoring | Additional capture, only if still failing | Replication/split |
|---|---|---|
| The two 0–0.5 m/s cells at 0.2–0.525 rad still have p95 >0.1 | One actual crawl target near 0.4–0.45 m/s × 0.20/0.35/0.50 rad; onset and reversal, both signs, one-tick step vs 0.30 s ramp, 0.25/0.75 s event ages. 48 probes / 144 phases per capture. Verify achieved speed before claiming the bin is covered. | Two independent train runs and one whole-run validation run. |
| The 2–3 m/s high-steer wheel-mismatch or 3–4 m/s mid-steer >1 m/s wheel/body mismatch slices remain high | Paired, reset-matched throttle increases at (2.5 m/s, 0.42/0.50 rad) and (3.5 m/s, 0.25/0.35 rad); both turn signs; +0.05/+0.10 command changes; step vs 0.30 s ramp to the same final command. Hold steering after a matched settle and record actual throttle feedback, both encoders, body twist, IMU/roll, and packet timing. 16 probes / 48 phases per capture. | Two independent train runs and one untouched whole-run validation run. Reject a probe if measured command/feedback or achieved operating point does not match the intended pair. |
| Errors persist in populated 1.5–12 m/s cells with low wheel/body mismatch and matched steering | No more operating-point data by default. Use the existing high-steer turn-in/reversal and race-domain captures to evaluate a causal temporal model or targeted feature ablation; collect new data only for a newly identified missing transition. | Preserve whole runs for validation; do not randomly split time samples. |

The selected yaw-teacher training support reaches 11.401 m/s simulator-truth
speed; the broader admitted source manifests include longitudinal feedback
up to about 11.4 m/s. There is no evidence above that. Thus the validated
empirical envelope is 0–about 11.4 m/s, not a proven 0–12 m/s model. Existing
full-throttle captures already reach the observed upper edge, so a new 12 m/s
test is not planned unless logs establish that the simulator can exceed
11.4 m/s without changing the allowed input/physics.

## 2026-10-08 continuation — complete support audit, candidate comparison, and fine-crawl capture

### Updated error and data-support diagnosis

- Completed `expanded_error_support_audit_v5_20261008.json` against the frozen
  v3 predictor and the newly admitted whole-run captures. It used 54 training
  runs / 492,013 samples and 34 validation runs / 255,396 samples for support
  and held-out analysis. Test/final-test arrays remain sealed.
- The new audit's overall v3 250 ms result is RMSE 0.1153 rad/s, p95 0.2364,
  and 85.08% within 0.1 rad/s. This is harder than the prior validation mix
  because it includes newly captured low-speed transition runs; it is not a
  regression on the same old runs. On the same 32 old validation runs, v3's
  run-macro RMSE was 0.0781 at 250 ms.
- Expanded whole-run validation by horizon (these are yaw-rate errors in
  rad/s, not yaw-angle errors in radians):

  | Horizon | RMSE | Absolute-error p95 | Within 0.1 | Maximum absolute error |
  |---:|---:|---:|---:|---:|
  | 25 ms | 0.0610 | 0.1292 | 93.07% | 1.9515 |
  | 100 ms | 0.1004 | 0.2181 | 87.42% | 1.6874 |
  | 250 ms | 0.1153 | 0.2364 | 85.08% | 1.6199 |
  | 500 ms | 0.1215 | 0.2491 | 83.82% | 1.6350 |
  | 1,000 ms | 0.1265 | 0.2602 | 83.02% | 1.7055 |

  This shows that large errors remain even one control step ahead, and the
  aggregate error grows as the forecast horizon lengthens.
- Added a v4 cell audit on the same 34 complete validation runs. At 250 ms,
  v4 changes overall RMSE/p95/within-0.1 from v3's 0.1153/0.2364/85.08% to
  0.0968/0.2029/85.49%. This improves aggregate RMSE but still leaves a large
  tail and does not meet a 0.1-rad/s maximum-error requirement. It is not
  evidence for production integration.
- New whole-run v3 results at 250 ms: residual-grid r03 RMSE 0.0811 / p95
  0.1612 / 90.8% within 0.1; low-speed-high-steer r03 0.1856 / 0.5034 / 71.5%;
  crawl r03 0.3511 / 0.7230 / 44.6%. At 1 s the crawl and high-steer run
  RMSEs rise to 0.4026 and 0.2215 respectively. This confirms the error is
  concentrated in low-speed dynamics rather than a universal sensor-rate or
  packet-alignment failure.
- Data scarcity is now quantified at finer low-speed bins. Below 0.5 m/s,
  |steer|=0.2–0.35 has only 92 training samples / 4 runs / 6 sequences and
  |steer|=0.35–0.525 has 91 / 3 / 5. The crawl captures added substantial
  0.5–2 m/s high-steer data, but their test phase reached almost no samples
  below 0.5 m/s while steering was applied. At 0.5–1.5 m/s the 0.1–0.2 rad
  band is also thin (for example, 1–1.5 m/s: 329 training rows across 10 runs;
  the held-out p95 is 0.796 rad/s). This is a specific, testable data gap.
- Conversely, many larger-error bands are not data-empty: 1.5–4 m/s high
  steering has thousands of training rows across 9–18 independent runs;
  4–8 m/s low/mid steering has tens of thousands across many runs; and 8–12
  m/s low steering is similarly populated. Their error is principally a
  model/transition-description problem, so they are excluded from another
  broad steady-state sweep. Rare >0.1 outliers remain even in some cells with
  p95 below threshold; per-bin count and tail fraction are retained in v5.
- The full 250 ms speed-by-steering classification is in the v5 audit JSON.
  Compact cell summary below gives `p95 | train samples / independent runs`;
  a cell is called sparse only where counts/runs are low, not merely because
  its validation p95 exceeds 0.1:

  | Speed (m/s) | 0–0.1 rad | 0.1–0.2 rad | 0.2–0.35 rad | 0.35–0.525 rad |
  |---|---:|---:|---:|---:|
  | 0–0.5 | p95 0.001 | 47,420/45; tails 1.3% | 0.235 | 56/5 — sparse | 0.182 | 92/4 — sparse | 0.110 | 91/3 — sparse |
  | 0.5–1 | 0.232 | 8,815/44 — dense | 0.793 | 108/6 — sparse | 0.814 | 1,575/5 — few runs | 0.765 | 716/5 — few runs |
  | 1–1.5 | 0.351 | 10,294/44 — dense | 0.796 | 329/10 — sparse | 0.765 | 4,733/10 — modest runs | 0.773 | 2,384/9 — modest runs |
  | 1.5–2 | 0.255 | 11,797/44 — dense | 0.490 | 1,696/12 | 0.531 | 7,220/10 | 0.563 | 6,677/9 |
  | 2–3 | 0.160 | 21,938/43 — dense | 0.640 | 1,794/14 | 0.392 | 7,950/18 — dense | 0.444 | 8,343/15 — dense |
  | 3–4 | p95 0.087 | 18,108/42; tails 4.1% | 0.223 | 3,299/19 | 0.344 | 7,009/18 — dense | 0.360 | 6,784/11 — dense |
  | 4–6 | 0.176 | 44,984/42 — dense | 0.239 | 9,582/27 — dense | 0.234 | 10,032/23 — dense | 0.240 | 5,078/12 |
  | 6–8 | 0.185 | 56,249/35 — dense | 0.216 | 13,649/26 — dense | 0.147 | 16,778/22 — dense | 0.114 | 11,561/18 — dense |
  | 8–10 | 0.147 | 48,051/30 — dense | 0.173 | 9,976/22 — dense | 0.129 | 5,334/17 — dense | 0.121 | 5,118/11 |
  | 10–12 | 0.147 | 29,328/18 — dense | 0.129 | 11,301/18 — dense | p95 0.100 | 1,762/7; 5.0% tails | no held-out support |

  The 10–12 m/s, 0.35–0.525 rad cell is unvalidated, not a measured error
  region. The existing measured 11.1 m/s steering frontier tops out at 0.18
  rad; do not extrapolate the high-speed envelope or request high-angle tests
  there without first establishing that those combinations are physically
  reachable and safe.

  This shows a real *base operating-point* gap concentrated below 1.5 m/s,
  especially below 0.5 m/s at nonzero steering and 0.5–1 m/s above 0.1 rad.
  It also shows large residual error in many populated cells; repeating those
  steady cells is not a justified test. The audit's feature intersections
  expose additional sparse *joint states*: e.g. 3–4 m/s, 0.2–0.35 rad,
  wheel/body mismatch above 1 m/s has only 78 training samples; 1–1.5 m/s,
  0.35–0.525 rad with mismatch above 1 m/s has 311; several 0.5–2 m/s
  steering-feedback-gap strata have only 100–400 rows. Those are candidates
  for targeted excitation, not evidence that every failure is wheel slip.
- The full joint 250 ms cross-cuts, using validation cells with at least 50
  samples and the sparse rule `training <500 rows OR <8 runs`, contain 101
  failing speed×steering×steering-feedback-gap cells (58 sparse, 28 dense),
  63 speed×steering×throttle-feedback-gap cells (22 sparse, 34 dense), and
  136 speed×steering×wheel/body-mismatch cells (63 sparse, 47 dense). The
  categories overlap; they do not imply 122 independent missing-data points.
  Most are transitions in combinations already represented in the broad
  matrix, so use paired excitation and run-level validation rather than a
  Cartesian repeat of every cross-cut.
- The high-speed throttle mismatch case already has dedicated, clean whole-run
  captures: two training plus one validation run at 8.5/9.5/11 m/s and
  0.06/0.10 rad, with both turn signs, +0.03 and -0.12 throttle changes, and
  step/ramp pairs. In the 10–12 m/s, <0.1 rad, throttle command-feedback gap
  0.25–1.0 slice, v3 has 186 validation rows across 6 runs and 377 training
  rows across 9 runs; p95 is 0.626 rad/s. V4's p95 is 0.665, so more
  same-condition data is not the next step. The current/history tree variants
  are similarly poor there; first score the held-out GRU on this exact slice.
  Do not repeat this high-speed test unless that evaluation shows a data-limited
  residual that new excitation can distinguish.
- Inspection of largest held-out errors identifies two distinct failure
  signatures. At 3.56–3.96 m/s and 0.20 rad, the model underpredicts yaw rate
  by about 1.45–1.62 rad/s while measured wheel/body speed mismatch is
  0.65–2.64 m/s, lateral acceleration is 7.5–9.5 m/s², and roll is
  0.048–0.062 rad. This is consistent with an under-supported slip-coupled
  response. Separately, around 3.47 m/s and ±0.42 rad, errors of about
  1.56–1.61 rad/s occur with near-zero steering/throttle command mismatch
  and only 0.056–0.081 m/s wheel/body mismatch; those are not explained by
  wheelspin alone and point to a history/transition-model failure. At
  0.9–1.3 m/s and about 0.4–0.5 rad, several >1.4 rad/s errors coincide
  with 3.0–3.3 m/s wheel/body mismatch. These associations are diagnostics,
  not causal proof.
- All auditable new captures use exact packet IDs and the 25 ms simulator
  timebase. Receipt cadence is about 40 Hz with rare >60 ms gaps; those rare
  stalls cannot explain the broad low-speed error fractions. No capture used
  future truth as model input.

### Targeted tree and neural-model evidence

- The v4 ExtraTrees refit adds the two crawl and two low-speed-high-steer
  training runs, keeps each r03 whole run in validation, and leaves final/test
  data sealed. On the same 32 old validation runs, its 250 ms run-macro RMSE
  is 0.08145 versus v3's 0.07814; a paired whole-run bootstrap 95% interval
  for the increase is [0.00169, 0.00488] rad/s (7/32 runs improved). This is a
  small but consistent tradeoff against the old domain.
- On the previously unseen targeted r03 runs, v4 does improve the trained
  low-speed families: crawl 250 ms RMSE/p95 changes 0.3511/0.7230 to
  0.1820/0.3858; low-speed-high-steer changes 0.1856/0.5034 to
  0.1555/0.3268. These are real transfer gains, but neither meets 0.1
  rad/s-tail accuracy. The residual-grid r03 remains a useful covered
  comparator. This supports adding data in the sparse low-speed cells, while
  arguing against a claim that more data alone solves the full-band problem.
- The same v4 held-out report still has severe, structurally different errors:
  about 1.77 rad/s at 3.47 m/s and 0.50 rad with only 0.055 m/s rear-wheel vs
  body-speed mismatch, and about 1.58 rad/s at 3.86 m/s and 0.20 rad with
  2.63 m/s mismatch. The first cannot be attributed to wheelspin and is
  consistent with a transition/history response failure; the second is
  compatible with a slip-coupled response. More samples of the broad
  speed/steering cells alone will not discriminate these mechanisms.
- GRU v1's epoch-70 checkpoint was good on its historical checkpoint-selected
  runs (run-macro RMSE 0.0503 rad/s), but fails the new unseen crawl/high-steer
  captures badly (250 ms run-macro RMSE 0.4109 and 0.3357). Its old training
  process had no new data and no validation gain through epoch 76; it is
  paused with the epoch-70 checkpoint/report preserved.
- GRU v2 uses 50 whole-run training captures, including the new
  targeted r01/r02 runs. Both crawl/high-steer r03 runs are explicitly
  excluded from checkpoint selection. It was paused at best epoch 56 with
  run-macro checkpoint score 0.07102 rad/s on its historical validation set;
  that score is not comparable to v3's expanded validation and no r03 scoring
  has been done. It remains research-only. This GRU and the newer v3/v4
  offline teachers are not integrated into runtime.
- Runtime distinction: `mpc_competition.yaml` already enables the Y1 empirical
  yaw-response surface through `vehicle_model.c`; that supported high-steer
  specialist remains active and must not be described as absent. The current
  new teachers are not in the MPC. Odometry still reports synchronized IMU yaw
  rate directly and does not use a learned yaw predictor; its learned yaw
  model integration remains future work pending causal, held-out validation.
  In `odometry_observer.cpp`, output yaw angle/rate pass through from IMU and
  pose uses IMU heading with midpoint body-velocity integration. A future yaw
  teacher must not replace those measurements with an unvalidated prediction;
  its plausible odometry role is a tested propagation/correction during a
  demonstrated sensor dropout or a separately validated body-velocity estimate.

### Finite suite for the confirmed remaining data gap

- Expanded `yaw_error_crawl_fine_gapfill` after the cell audit to a full
  24-point low-speed rectangle: speeds 0.35, 0.45, 0.65, 0.90, 1.20, and
  1.45 m/s crossed with steering magnitudes 0.10, 0.20, 0.35, and 0.50 rad.
  Each point is paired across turn signs, onset/reversal, step/0.30 s ramp,
  and 0.25/0.75 s event ages, with a reset-matched speed approach per probe.
  This is 384 probes / 1,152 phases and a conservative schedule estimate of
  6,778 s (113 min) per run. Timeout is 9,000 s; two independent training
  captures and one untouched whole-run validation capture are planned.
- Added a second, conditional profile `yaw_error_lowspeed_wheelspin_gapfill`
  for sparse high wheel/body mismatch combinations: twelve measured
  speed/steering points from 0.75 to 3.5 m/s, including the audited 2.5 m/s
  moderate-steer (0.25/0.35 rad) and high-steer (0.42/0.50 rad) slices, both
  signs, +0.05/+0.10/+0.20 throttle changes, and step/ramp pairs to the same
  endpoint (144 probes, 288 phases, approximately 42 min per run). Even the
  largest endpoint stays
  below the existing 0.50 command cap. It records throttle feedback,
  encoders, body motion, IMU/roll, and packet timing. Run this profile only if
  the refit still has high residuals in those sparse mismatch strata; do not
  use it as a broad substitute for modeling populated regimes. If triggered,
  use two independent training captures and one untouched whole-run validation
  capture; admit a probe only when actual throttle feedback and measured
  wheel/body mismatch confirm that it exercised the intended condition.
- The first, narrower crawl capture
  `openplane_yaw_error_crawl_fine_train_r01_20261008` stopped at 140/672
  phases after one active odometry gap reached 252.8 ms, exceeding the
  experiment's 250 ms stale-source abort threshold. It had 32.3–33.8 Hz active
  receipt cadence, 113.9 ms p95 gaps, and 253.2 ms maximum; it is not admitted
  as a training dataset. During it, both tree refit and GRU v2 training were
  CPU-active. A no-fit moving smoke run on the same pinned simulator/bridge
  completed all 20 phases with zero collisions/faults: odom and sensor cadence
  39.96 Hz, p95 gap 25.85 ms, max gap 46.27 ms, and request/response latency
  26.4 ms median / 27.2 ms p95. This strongly implicates host compute
  contention in the earlier data stall (the exact Unity/socket scheduling
  contribution is not yet isolated), rather than a persistent 20 Hz bridge
  ceiling. Both GRU fits remain paused during simulator work.
- Current full training capture
  `openplane_yaw_error_crawl_fine_train_r01_retry2_20261008` completed in the
  official pinned Explore image via the existing batch-mode/Xvfb procedure;
  no GUI and no `-no-graphics` mode were used. The runner reports
  `schedule complete`, 1,152/1,152 phases, 67,706 commands at 39.55 Hz, zero
  quality failures, and no abort. Its closed bag still must be converted to a
  split-labelled model archive and checked for *measured* probe-state support;
  requested conditions alone do not establish coverage.
- The throttle gapfill profile was expanded to include ±0.20 alongside
  ±0.05/±0.10, and the focused schedule checks passed. That profile remains
  conditional and has not been run.
- The prior live `/odom` rolling monitor included a 927 ms interval spanning
  a reset/idle period; it is not a valid active-probe cadence verdict. Closed
  bag phase-aware analysis is authoritative for the capture.
- Next: keep CPU-bound fitting paused while checking this closed bag's phase
  cadence, collision/fault gates, and measured support. If admitted, create
  independent training r02 and whole-run validation r03 only where this
  capture still leaves an identified model/data question. Refit on training
  runs only and score frozen candidates on r03. Do not collect more
  1.5–12 m/s steady-state points: most errors there persist with thousands of
  training rows, so broad repetition is not justified.

### 2026-10-08: exhaustive >0.1 rad/s classification and finite follow-up suite

- Clarification: the audited `0.1` limit is **absolute yaw-rate prediction
  error in rad/s**, not yaw-angle error in radians. The target is yaw rate
  250 ms ahead; horizons from 25 ms to 1 s are separately evaluated.
- At 250 ms, the v3 current-only predictor scores RMSE 0.1153 rad/s, p95
  0.2364, and 85.08% within 0.1 on 292,203 held-out examples. The v4 tree
  lowers pooled RMSE to 0.0968 but p95 remains 0.2029 and maximum is 1.770;
  on the 32 shared older validation runs, run-macro error is worse than v3
  with paired bootstrap CI for the increase [0.00169, 0.00488] rad/s. Neither
  fit establishes that the error tail has been solved.
- Across horizons 25/100/250/500/1000 ms, v3 has respectively 20/29/35/36/30
  failing speed×steering cells (at least 50 held-out samples and p95>0.1).
  Of these, 4/6/6/6/5 are sparse by the current rule (under 500 train rows or
  under 8 train runs); 13/21/27/28/23 have at least 1,000 rows and 10 runs.
  These classifications overlap with other feature cuts. They show that
  broad sample count is not the principal explanation for most failures.
- At 250 ms, 27 speed×steering cells have at least 100 validation rows and
  p95>0.1; only two have fewer than 100 training rows, both below 0.5 m/s at
  nonzero steering. Low speed is not uniformly solved by row count either:
  the 0.5–1 m/s, 0.2–0.35 rad cell has 1,575 training rows but only five
  independent runs, and p95 is 0.814. This justifies the already-running
  low-speed *transition* matrix, not more arbitrary steady-state repetitions.
- Joint feature cuts reveal missing combinations hidden by populated base
  cells. At 250 ms there are 101 failing speed×steering×steering-feedback-gap
  cells (58 sparse), 63 speed×steering×throttle-feedback-gap cells (22
  sparse), and 136 speed×steering×wheel/body-speed-mismatch cells (63
  sparse). The categories overlap and do not represent independent test
  points. Notable examples from the expanded v3/v4 error/support audits:
  - 3–4 m/s, 0.2–0.35 rad, rear-wheel/body mismatch 1–5 m/s: train 78 rows
    across 13 runs; validation 80 rows across 13 runs; p95 0.707 rad/s.
  - 2–3 m/s, 0.35–0.525 rad, mismatch 0.25–0.5 m/s: in v3, train 247
    rows across 6 runs; validation 147 rows; p95 0.659 rad/s. The v4 fit
    reduces this slice's p95, so it is not a clear persistent-data-gap trigger
    without checking the newer held-out score.
  - 2–3 m/s, 0.2–0.35 rad, throttle command/feedback gap 0.05–0.1: train
    873 rows / 7 runs; validation 405 rows; p95 0.666 rad/s.
  - 4–8 m/s, moderate/high steering and 0.025–0.05 throttle gap: several
    cells have only 33–53 training rows across 2–3 runs, with validation
    p95 about 0.27–0.57 rad/s. This is a possible *joint-transition*
    coverage gap, not a lack of ordinary steering or throttle samples.
  - 10–12 m/s, under 0.1 rad steering, throttle gap 0.25–1: train 377 rows /
    9 runs; validation 186 / 6; p95 0.665 rad/s. This exact event already
    has dedicated high-speed throttle captures. Score the causal GRU on the
    held-out run before authorizing a repeat or longer post-cut hold.
- The largest residuals confirm at least two distinct mechanisms rather than
  one universal data shortage:
  - At 3.76 m/s and 0.20 rad, v3 underpredicts yaw rate by 1.620 rad/s while
    rear wheel speed exceeds body speed by 1.98 m/s. At 0.895 m/s and 0.397
    rad, mismatch is 3.26 m/s and error is -1.47 rad/s. These support a
    wheel-slip/history interaction; the corresponding joint training cells
    are sparse and are covered by the staged throttle-pair captures.
  - At 3.47 m/s and -0.42 rad, v3 errs by +1.61 rad/s with only 0.056 m/s
    wheel/body mismatch and essentially zero command/feedback steering gap.
    The v4 maximum is 1.77 rad/s near 3.47 m/s and +0.50 rad with 0.055 m/s
    mismatch. These are wrong-direction/high-steer response errors in base
    cells with thousands of training rows; more generic throttle tests do not
    explain them. They require temporal-model/label diagnosis and the unseen
    GRU test, not a wheelspin label imposed after the fact.
  - At 11.24 m/s and near-zero steering, the high-speed validation capture
    contains -1.337 rad/s error during a throttle cut: command-feedback gap
    -0.488 and wheel/body mismatch +1.015 m/s. This is already in the
    dedicated high-speed throttle dataset, so a repeat is contingent on the
    frozen GRU failing this exact held-out event.
- Read-through of target construction found no obvious horizon off-by-one or
  future-measurement leakage: the direct teacher consumes past/current sensor
  observations and candidate commands k…k+h-1, then targets simulator yaw
  rate at k+h; all windows stop at reset/sequence boundaries. The GRU sees 65
  consecutive 25 ms sensor rows (1.6 s) and the future command sequence. Its
  inputs include rear wheel speeds, IMU longitudinal/lateral acceleration,
  yaw rate, roll/rate, and actuator feedback/commands, but no explicit packet
  age, measured body sideslip, or front-wheel speed. Therefore wheelspin and
  lateral state are only inferable indirectly from history and IMU response.
  This is a concrete candidate limitation for the dense wrong-direction
  high-steer errors, not yet proof that adding any one feature will improve
  held-out prediction. One selected packet-clock training/validation archive
  contains 99,906 rows in 321 sequences; its manifest states that `/odom`
  source stamps are joined to bridge packet timing, only consecutive packet
  IDs remain in a sequence, resets are hard boundaries, and model intervals
  are fixed 25 ms (receipt/request times are diagnostic only). Thus training
  windows do not bridge known packet gaps, but the model also does not observe
  packet age. Packet-age correlation and residual response timing remain to
  be tested on existing timing-labelled captures without changing the user's
  fixed-25-ms physical-time assumption.
- Many other failing cells, including 4–8 m/s near-straight and matched-
  steering high-angle reversals, have tens of thousands of base samples and
  many runs. Those are not justified by a new broad capture. They point to
  latent history/transition representation, target construction, or model
  generalization; a new test is warranted only if a held-out residual slice
  identifies an unexcited transition variable.
- The new mid-speed schedule is part of the CLI/runner profile set, uses a
  seeded randomized order, and passes the focused schedule test for signed
  throttle changes, paired endpoints, reset-isolated approaches, and command
  bounds. The support-gap schedule class now passes 14 tests total; shell
  syntax and `git diff --check` pass. This test has not been launched.
- If all three targeted families ultimately prove necessary (crawl,
  low/mid-speed wheelspin, and mid-speed throttle gaps), the planned total is
  about 9.15 simulator-hours for two train captures plus one whole-run
  validation capture per family. The latter two families are conditional on
  frozen-model scores, so this is an upper bound, not an instruction to run
  redundant data collection.
- Finite capture plan and gating:
  1. Finish `yaw_error_crawl_fine_gapfill` r01. The current matrix spans
     0.35–1.45 m/s, 0.10/0.20/0.35/0.50 rad, both signs, onset/reversal,
     step/ramp, and two event ages. If quality and actual-speed gates pass,
     add independent r02 training and r03 whole-run validation captures.
     Three captures are about 5.65 simulator-hours at the conservative
     estimate; a failed cadence/coverage capture is not admitted.
  2. Refit the fixed v4 teacher and causal GRU using r01/r02 only, then score
     the frozen models on r03 and the named joint-error slices. Keep all
     final-test data sealed. Do not train, select checkpoints, or run heavy
     CPU fitting while capture timing is being measured.
  3. Only if the 2–3.5 m/s wheel/body-mismatch or throttle-gap errors persist,
     run the already implemented `yaw_error_lowspeed_wheelspin_gapfill`:
     12 measured speed/steering points, both signs, +0.05/+0.10/+0.20
     throttle changes, step versus 0.30 s ramp to the same endpoint, with
     actual throttle feedback, wheel speeds, body motion, IMU/roll, and
     packet timing. It is 144 probes / 288 phases / about 42 minutes per
     capture; use two training runs and one untouched whole-run validation
     run (about 2.1 additional hours). Require the measured mismatch/gap to
     enter the intended bin; a commanded step alone is not proof of coverage.
  4. A second already-implemented conditional profile,
     `yaw_error_midspeed_throttle_gapfill`, targets the separate sparse
     4–8 m/s throttle-gap cells: (4.5 m/s, 0.275/0.42 rad),
     (6.5 m/s, 0.15/0.275/0.42 rad), and (8.0 m/s, 0.42 rad), both turn
     signs, ±0.04/±0.08 throttle changes, and step/ramp pairs. This is 96
     probes / 192 phases / about 27.9 minutes per capture; two train and one
     validation run add about 84 minutes. Run it only if the frozen GRU still
     fails these exact held-out
     cells after the current refit. Require actual throttle feedback to
     enter the audited 0.025–0.05 or 0.05–0.1 gap bands and check measured
     speed/steering before accepting samples.
  5. Do not repeat the dedicated 8.5–11 m/s low-angle throttle suite unless
     the held-out GRU and the post-crawl refit both fail its exact transient
     slice. Do not sample 10–12 m/s at 0.35–0.525 rad: there is no measured
     feasible support there, so that cell is unvalidated rather than a known
     error.
  6. Once targeted gaps are filled, stop collecting yaw data unless another
     whole-run validation identifies a new sparse joint condition. For dense
     low-mismatch cells, proceed with model/feature/temporal diagnosis instead
     of multiplying nearly identical bags.
- The final runner line is authoritative for phase counts. Earlier live
  progress notes that counted only `approach_*` log messages (for example
  362) were not total phase counts and are superseded by the completed 1,152
  phase result above.

### 2026-10-08: fine-crawl r01 capture closed and cadence-gated

- Bag:
  `live_runs/openplane_yaw_error_crawl_fine_train_r01_retry2_20261008/run/run_0.db3`
  (653,083 total messages; 54,579 samples on each of the six 40 Hz telemetry
  streams; 384 probe phases). The runner completed all 1,152 scheduled phases
  with 0 quality failures and 39.55 Hz command publication.
- Closed-bag analysis passed its quality gates. Receipt-time rates were
  39.951–39.952 Hz; stream gap p95 was 25.82–25.90 ms, maximum 55.58–56.68 ms,
  and there were zero active gaps above 60 ms. Cross-phase/reset gaps were
  excluded from active cadence. Collision count stayed zero, bridge timing
  faults were zero, and all 384 probe phases were valid (0 invalid).
- The first analyzer invocation falsely rejected the new `probe_yawerr_*`
  naming pattern because its phase classifier knew only older capture labels.
  The classifier now recognizes these probes, reports each named quality
  gate, and no longer overwrites the accepted probe list with the legacy
  phase-block list. The focused phase/rate tests pass (6 tests), and rerunning
  the closed-bag analysis produced `quality gates: PASS`. This was an analyzer
  classification defect, not bad capture data.
- The remaining admission step is model-dataset conversion and condition-level
  support audit: quantify actual speed, steering feedback, yaw response,
  wheel/body mismatch, command/feedback lag, packet continuity, and phase
  sample count for each signed onset/reversal × step/ramp × delay condition.
  Only then decide whether r02/r03 are useful replication or held-out data.
- At the 04:29 CEST process check, the two GRU trainers were stopped (`T`),
  but a separate Explore simulator process from
  `openplane_yaw_transport_smoke_r01_20261008` remained active at roughly one
  CPU core. Its experiment log shows only a CLI timeout-validation error and
  the recorder contains a tiny smoke bag; it is not part of the completed
  fine-crawl capture. It is being treated as a leftover simulator service,
  not as a successful test run.

### 2026-10-08: error-support audit and crawl speed-control pilot

The follow-up support audit is complete and is summarized in
[`YAW_ERROR_SUPPORT_CLASSIFICATION_20261008.md`](YAW_ERROR_SUPPORT_CLASSIFICATION_20261008.md).
The short version: at 250 ms, 34 speed×steering cells have p95 yaw-rate error
above 0.1 rad/s; 27 are dense by independent-run/sample support, 5 sparse, and
2 intermediate. Across 25–1000 ms, the number of failing cells stays between
20 and 34. Repeating generic speed/angle holds is therefore not justified for
most cells.

The largest specific 250 ms failure is a +0.50 to −0.50 rad steering
reversal at roughly 3.47 m/s. Ground-truth yaw changes from +1.406 to
−0.833 rad/s while the teacher predicts +0.937. Steering feedback follows
within about 50 ms, and current wheel/body mismatch is only 0.055 m/s. The
matching event-conditioned training support is just 43 samples across 9
sequences in 2 runs; on the independent validation capture, 22/31 samples
exceed 0.1 rad/s (RMSE 0.718, p95 1.510, max 1.770). This is a sparse
high-rate reversal-response problem hidden inside an otherwise dense bin.
A separate acceleration/wheelspin tail has 2.63 m/s rear-wheel/body mismatch
and needs targeted paired throttle tests only if the frozen-model refit still
fails its exact held-out slice. Extending the teacher's history window to
1.6 s did not improve its 250 ms error.

The bounded `openplane_yaw_error_crawl_speedhold_kp100_r01_20261008` run
completed 96 probes and passed stream/collision/fault gates: all six streams
were 39.952 Hz, gap p95 25.83–25.91 ms, gap max 53.18 ms, with no active gap
over 60 ms, collision, or bridge fault. `kp=1.0, ki=0.05` moved measured
median probe speeds much closer to their 0.60/1.00 m/s targets (0.673–0.706 /
1.067–1.098 m/s), compared with the old-default-gain medians near 1.09/1.50.
But only 43.7% of plateau samples were within ±0.15 m/s, and the original
phases had `validate_speed=false`; its `quality gates: PASS` must not be
interpreted as speed-target validation. A future schedule-harness change now
requires speed validation for crawl probes, with a focused test confirming
that setting. No full crawl retest has been launched. The next planned capture
is a reset-isolated 1–5% throttle calibration, then the crawl matrix only if
actual speed gates pass.

No production odom, MPC, vehicle-physics, or observer model was changed by
this audit. Test-harness speed validation is the only implementation change;
the current yaw teacher remains offline research-only.

### 2026-10-08: regime-by-regime response status and unseen GRU scores

Created the findings-only [`yaw response regime map`](YAW_RESPONSE_REGIME_MAP_20261008.md).
The answer to “has every region been identified like the 8.17–9.22 m/s,
0.35–0.50 rad fit?” is **no**. Two local time-constant laws are supported:
Y1 hold at 2.5–3.5 m/s / 0.30–0.42 rad (`tau=0.129314 s`) and the high-speed,
high-steer candidate at 8.170–9.222 m/s / 0.350–0.500 rad (`tau=0.0119551 s`).
Turn-in/unwind constants in Y1 are too sparse for promotion. The crawl law
`r=0.961264*u_odom*tan(delta)/0.324` is validated only for steady 0.237–1.25
m/s, 1–5% throttle and tested angles; it gives no higher-speed boundary.

The broad GT-conditioned exact-cell atlas is a different kind of result: 478
supported cell/phase predictors on a 0.5 m/s × 0.025 rad grid, trained from
628 occupied cells (578 have at least two training runs). On its matched
whole-run validation support it covers 92.17% of transitions at RMSE 0.04396,
p95 0.03789 rad/s, with 97.97% below 0.1; it still abstains on 7.83%, has a
1.221 rad/s maximum, and observed speed reaches only 11.37 m/s. This is a
useful broad one-step predictor, not a local time-constant model everywhere
and not full 0–12 m/s Cartesian coverage.

Scored the complete v2 GRU checkpoint on the two later validation captures
excluded from checkpoint selection. Crawl r03 gives 25 ms RMSE/p95
0.12591/0.26562 and 1 s 0.12955/0.27538 rad/s. Low-speed/high-steer r03 gives
0.09709/0.22185 at 25 ms and 0.19037/0.35482 at 1 s. Within crawl r03 at
25 ms, low steer (0.5–1.0 m/s, 0–0.1 rad) is 0.0961/0.1359 (n=85), while
high steer (0.5–1.0 m/s, 0.35–0.525 rad) is 0.1982/0.3433 (n=63). This is
evidence that the two steering regimes do not share the GRU's accuracy; it
does not locate a sharp physical boundary. The GRU is command-conditioned on
the recorded future command sequence, so these are offline forecast scores,
not a sensor-only odometry result.

The next analysis is to fit an event-conditioned next-yaw model from the
existing high-steer reversal and low-speed/high-steer training captures, then
verify exact joint-condition support in an independent existing validation
run before scoring. Gather a new capture only if this support audit finds the
required event/interaction strata missing. No simulator was launched and no
runtime odometry/MPC/physics model was changed for this analysis.

### 2026-10-08: crawl-law transfer boundary and targeted reversal validation

Corrected a speed-coordinate mismatch in the first broad boundary calculation:
the published crawl coefficient was identified using longitudinal odometry
speed, so its frozen transfer score must use that same input. Refit the compact
law on the clean crawl training capture, obtaining `k=0.961181` with odometry
speed (matching the saved `0.961264`) and `k=0.938317` with truth speed. Then
scored the published odometry-speed law against simulator-truth yaw on the
existing clean validation runs, selecting steady rows with
`|d(delta)/dt|<0.10 rad/s`, `|du_GT/dt|<0.25 m/s^2`, and
`|dr_GT/dt|<2 rad/s^2`. This is a transfer diagnostic, not a new model fit.

The most informative brackets are now: the fixed law remains close at 2.5–4.0
m/s through 0.20–0.25 rad (273 samples / 2 runs, RMSE 0.024 rad/s), but fails
by 0.30–0.35 rad (601 / 2, RMSE 1.908); at 4–6 m/s it remains close through
0.10–0.15 rad (833 / 5, RMSE 0.031) but fails by 0.15–0.20 rad (391 / 4,
RMSE 1.041); and at 6–8 m/s it is already degraded at 0.04–0.10 rad (2,341 /
5, RMSE 0.262, p95 0.589). The exact cell support is uneven, so these are
brackets, not sharp thresholds. Full counts and error summaries are in the
[`regime map`](YAW_RESPONSE_REGIME_MAP_20261008.md). Its held-out runs were
excluded from the crawl coefficient fit, but had been used in broader atlas
development; do not treat them as final sealed evaluation.

The prior support audit had only two high-steer-reversal training captures and
no independent validation capture with adequate samples in the same joint
event region. That gap has since been addressed by the completed
`openplane_yaw_error_highsteer_reversal_validation_r03_20261008` run, described
in the following 2026-10-08 update. The earlier “running” progress snapshot is
historical and superseded by the completed-run record.

Added [`score_frozen_yaw_atlas.py`](../../tools/racing/specialists/score_frozen_yaw_atlas.py)
to replay a saved atlas's coefficients against one explicitly admitted
whole-run validation capture, reporting direct-cell/direct-phase coverage and
a persistence baseline on identical samples. Before using it on r03, replayed
the frozen v11 report on its existing high-steer validation r03: all 4,643
direct-phase predictions and RMSE/p95/max values matched the original report
exactly (`0.04504985 / 0.01163422 / 0.65409992 rad/s`). This verifies the
coefficient replay path, not a new vehicle-model gain.

The completed-run quality gate, train-only v12/v13 refits, and r03 event scores
are recorded in the following 2026-10-08 update. The model uses current
truth-labelled motion plus current actuator/encoder values for offline
identification; it is not an odometry or MPC runtime input contract. No
production code has been changed.

### 2026-10-08 follow-up: actuator prediction takes priority

Following the new steering-reversal evidence, the immediate focus is the
command-to-steering-feedback model rather than another broad yaw-feature fit.
The latest 3–4 m/s / 0.30–0.525-rad actuator evaluation uses a one-step target
`steering_feedback[k+1]`, the 25-ms packet grid, 48,144 train rows from 25
clean train captures, and 13,735 r03 validation rows (10,072 within controlled
onset/unwind/reversal probes). It explicitly compares command alignments
k, k−1, and k−2, hold, the 3.2-rad/s limiter, and the release-or-limit hybrid.

The one-packet command history is best among the three discrete alignments.
The hybrid is the best simple model overall (RMSE 0.02738 rad), but has a
0.919-rad worst error. The regime split is real: onset favors a 3.2-rad/s
limit (0.00327-rad RMSE); unwind favors direct delayed-command following
(0.01950 vs 0.04354 rad for the limiter); reversal remains mixed (0.04476-rad
limiter RMSE, max 0.919 rad). In some held-out 25-ms full-angle reversals,
steering feedback changes almost the full 1-rad span in one sample, while
other reversal probes from similar settings are rate-limited.

An ExtraTrees correction to the hybrid reduced pooled RMSE (0.02738→0.02176
rad) and worst error (0.919→0.723 rad), but the paired condition analysis did
not support keeping it: it worsened 179/216 condition RMSEs, improved 37, and
mean condition-RMSE delta was +0.00174 rad (95% bootstrap CI
[-0.00100,+0.00448]). It also made some held-out reversals predict the wrong
steering sign. The candidate is rejected and no learned actuator artifact was
saved.

Two yaw-atlas changes—predicted next steering change (v14), then an explicit
release flag (v15)—barely changed event RMSE; tails stayed at 0.68–1.04 rad/s.
This confirms that adding continuous global features is not enough. The next
modeling step is to classify/represent actuator onset, release, reversal, and
command age as distinct states and fit the one-step feedback transition from
training captures; score the frozen candidate on r03. Do not integrate a
candidate until its reversal predictions no longer have near-full-range misses
and it does not harm onset/unwind. If those states remain ambiguous at 25 ms,
use the existing bag timing/command history to isolate the missing state before
requesting more simulation. Production odometry, MPC, and physics remain
unchanged.

### 2026-10-08 actuator timing correction and frozen r03 replay

This subsection supersedes the preliminary same-sign-release and timing
interpretation immediately above. The actuator rule was corrected to allow a
direct delayed-command target whenever steering magnitude decreases, including
through a sign change. On the held-out r03 capture, this simple hybrid scored
0.02136-rad one-step steering-feedback RMSE (p95 0.00020, max 0.99981) versus
0.03095 for a fixed 3.2-rad/s limiter and 0.03546 for direct command following.
Across 216 paired conditions, the hybrid reduced condition RMSE against the
fixed limiter by 0.01344 rad (95% bootstrap CI 0.00909–0.01817), with 82
conditions better and 22 worse. The gain is local to 2.5–4.5 m/s and
|steering| 0.30–0.525 rad; the nearly 1-rad maximum reversal error remains.

The learned ExtraTrees residual is rejected: it slightly lowers pooled RMSE
to 0.02018 but worsens 186/216 paired conditions (mean condition-RMSE delta
+0.00370 rad, 95% CI +0.00250–+0.00497). A hand-coded extra-queue-on-new-sign-
reversal rule is also rejected: it triggers for 26 r03 samples, but worsens
paired condition performance (mean +0.00436 rad, 95% CI +0.00113–+0.00794),
and reversal RMSE rises 0.03356→0.04753 rad. Neither candidate was integrated.

Added [`analyze_steering_actuator_timing.py`](../../tools/racing/specialists/analyze_steering_actuator_timing.py)
and retained machine-readable per-probe results under
`live_runs/racing_model_diagnostics_20261008/yaw_highsteer_reversal_v11_v16_magnitude_release/`.
Raw ROS receipt-time latency is recorded only as a transport diagnostic; it is
not used as physical `dt`. Using packet sequence and the required fixed 25-ms
interval, command-to-feedback onset counts were:

- Train r01: 149/216 at 2 packets, 34 at 3, 25 at 1, and 8 at 0 or 4–7.
- Train r02: 187/216 at 2, 23 at 3, and 6 at 1.
- Held-out r03: 190/216 at 2, 24 at 3, and 2 at 1.

All measured onset intervals were packet-sequence-contiguous. The current
one-previous-command model therefore represents the dominant two-packet
command-to-feedback onset; a minority takes one more packet, and r01 is much
more variable. A global extra delay does not transfer.

The corrected v16 yaw atlas was replayed on the same r03 holdout. On shared
direct-phase support, yaw RMSE changed only 0.06500→0.06489 rad/s; onset,
reversal, and unwind changes were similarly negligible. The maximum event
errors remain roughly 0.68, 0.96, and 1.04 rad/s. No yaw improvement is claimed.

The planned condition-only onset-mode check is complete; it did not predict
minority modes. The bridge request records then exposed command-update age as
a likely explanation for the extra packet, documented in the follow-up below.
Next score the separated command-to-request and request-to-feedback stages on
existing runs. No new simulation or production odometry/MPC/physics change was
made.

### 2026-10-08 onset-mode predictability check on existing captures

Completed a first classifier check on the three existing 216-condition timing
captures; no new run was started. A random forest trained on r01/r02 using the
condition descriptors (event, speed, steering magnitude/sign, transition
profile and duration) scored 87.04% accuracy / 0.330 balanced accuracy on r03.
The always-two-packet baseline scored 87.96% / 0.333 balanced accuracy, so
the fitted classifier did not predict the minority delay modes. ExtraTrees
scored 82.87% / 0.326. The r03 outcomes are 190 two-packet, 24 three-packet,
and 2 one-packet responses. Exact-condition onset-mode agreement was 62.0%
between r01/r02 and 76.9% between r02/r03. Thus condition parameters alone do
not determine the observed onset count across repetitions; this is compatible
with unobserved packet-phase/state variability but does not identify its
cause.

The confirmed actuator fit is therefore limited to a local, one-step
statistical gain over the fixed rate limiter; its near-1-rad worst cases and
unpredictable transition tail still rule out runtime promotion. A coarse
command-start packet-index phase check (modulo 2/3/4/5/8/10/20) did not predict
the minority onset modes. A deeper join to each bag's bridge packet diagnostics
then found that command-update age at request time does: on r03 the median was
10.22 ms for the 190 two-packet cases and 22.30 ms for the 24 three-packet
cases; r02 replicated the separation (10.27 vs 22.16 ms). Request/response
latency did not separate them on r03 (26.41 vs 26.27 ms median).

For r03, using a 0.01-rad command-change threshold, the first request carrying
the changed command was one packet after the observed ROS command change for
189/190 two-packet cases and two packets after it for 23/24 three-packet cases.
Feedback then began moving one packet after the changed command was first
included in a request in those cases. This strongly suggests the extra packet
is command-update timing against the bridge's 25-ms send boundary, followed by
an approximately one-packet actuator/feedback response—not a longer physical
servo time constant. It is not definitive proof of Unity's apply tick because
the bridge has no echoed applied-command ID or native frame counter; packet
sequence is assigned locally on receipt. The MPC receives a Float32 steering
feedback value and receive-age estimate, not the diagnostic packet ID or the
request's command snapshot. So bags can align this path more precisely than
the live permitted input stream currently can.

Source inspection confirms the current runtime model uses a fixed 25-ms
physical steering queue (`physical_steering_delay_s`) while
`command_actuation_delay_s` is 0.0 in `mpc_competition.yaml`. That fixed queue
models the approximately one-packet response after a request carries the new
command; it does not model the extra 0/25-ms command-update-to-request phase
seen in the bag join. Fresh measured steering feedback anchors the current
physical angle, but the future MPC rollout still uses its fixed command queue.
This identifies a plausible missing timing term, not yet a validated runtime
fix: the legal steering feedback has no header/command ID, so exact request
phase is not currently observable to MPC. No production code or parameters
were changed.

Next: score a split offline model—command update to bridge request, then
feedback response after the carrying request—on the existing r01/r02/r03
captures. Keep the bridge-side timing diagnostic out of competition MPC inputs
unless an allowed runtime signal can provide equivalent causally available
timing. No odom, MPC, or physics code was changed.

### 2026-10-08 command-age feature check and two-packet counterfactual

The bridge timing join already points to command-update phase relative to its
25-ms request boundary as the main source of the one-/three-packet onset
variation. I tested whether that phase could be approximated from only
competition-visible causal history by adding time-since-change features for
steering command/feedback and throttle command/feedback (25-ms fixed packet
grid, 0.005 change threshold, 400-ms cap). The model remains research-only;
simulator truth is only the next-step yaw target and scoring truth. No future
sensor or GT inputs were added.

The candidate was fitted on the same sealed 62 training runs and evaluated on
the same 36 whole-run validation captures (299,242 rows). The baseline was the
command-intent local atlas with 0/25/50/100-ms history. On the 266,562 common
local-support rows:

| Score | Existing v2 | Command-age v4 |
|---|---:|---:|
| Local one-step yaw RMSE | 0.028935 | 0.029139 rad/s |
| p95 absolute error | 0.020700 | 0.020567 rad/s |
| Maximum absolute error | 1.353620 | 1.375296 rad/s |
| Fraction below 0.1 rad/s | 98.8847% | 98.8911% |

Run-clustered v4-minus-v2 RMSE was +0.000330 rad/s, with a 95% bootstrap CI
of +0.000043 to +0.000636; v4 improved 12/36 validation runs and worsened
24/36. The tiny p95/fraction changes do not offset the significantly worse
run-level RMSE and maximum tail. Reject v4; keep it as a comparator and do not
integrate it into odometry or MPC. Its report and bundle are under
`live_runs/racing_model_diagnostics_20261008/sensor_only_yaw_regime_atlas_command_age_v4/`.

I also tested the “always two packets” counterfactual by joining all 216 r03
probe-condition yaw scores to their measured packet-grid onset counts. The
available controlled high-steer captures do not show 95% two-packet behavior:
r01 was 149/216 (69.0%), r02 187/216 (86.6%), r03 190/216 (88.0%), and the
combined 3-run total was 526/648 (81.2%). On r03, restricting evaluation to
the 190 two-packet conditions changed the existing v16 yaw atlas RMSE only
from 0.072695 to 0.072314 rad/s; the maximum stayed 1.037442 rad/s. That
maximum occurs on a two-packet 4.0-m/s, 0.42-rad unwind condition, so the
largest yaw miss is not explained by the exceptional packet count. The
two-packet-only rows have 97.80% below 0.1 rad/s, but their >1-rad/s tail
still violates the desired error bound. Filtering the other 12% of r03
conditions therefore does not solve yaw prediction and would hide behavior
the live system still encounters.

There is positive evidence for the source of the actuator timing modes. In
r03, command-update age at bridge request time had median 10.22 ms for
two-packet cases versus 22.30 ms for three-packet cases, while request/response
latency did not separate them. The changed command first appeared in a bridge
request one packet after the ROS command update in 189/190 two-packet cases,
and two packets after it in 23/24 three-packet cases; feedback then usually
moved one packet after that request. This is consistent with command update
landing on different sides of the bridge's periodic request boundary,
followed by about one packet of feedback response. It is not proof of the
Unity apply tick (no echoed command ID/native frame counter exists). The
bridge-side timing topic was used only offline and is not a legal competition
MPC input. Since sensor-only command-age features did not improve held-out yaw,
this timing diagnosis does not justify a runtime change.

The data-driven atlas is still not a full 0–12 m/s model: the 36 validation
runs reach 11.299 m/s and 0.5236 rad; the 11.5–12.0 m/s training bin is empty.
Among 640 validation speed/steering cells with at least 20 rows, 511 have a
local expert and 46 supported cells fail the p95/95%-within-0.1-rad/s gate.
Those cells do not imply every speed/steering combination is physically
feasible. No simulator was launched and no production Odom/MPC code, physics,
or tuning was changed. Next focus is the high-error unwind/reversal response
that remains even when command-to-feedback onset is exactly two packets;
continue using per-regime models, retain unsupported cells as unsupported,
and evaluate on whole unseen runs before any promotion.

### 2026-10-08 local unwind response under the fixed-two-packet hypothesis

The exact worst high-steer yaw row was traced through neighboring fixed-grid
samples. At 3.97 m/s and +0.42 rad, steering feedback stayed at 0.4199 rad
while the command had already moved to zero. One packet later the feedback
snapped to 0.0016 rad and GT yaw rate fell from 1.2816 to 0.2386 rad/s. A
second capture contains the same response: current yaw 1.2816 to 0.2847.
These are not one-/three-packet outliers; the r03 probe labels both as
two-packet onset.

I implemented
[`fit_yaw_two_packet_release_specialist.py`](../../tools/racing/specialists/fit_yaw_two_packet_release_specialist.py)
as a reproducible, narrow model fit. It uses the existing r01/r02 training
captures, GT only as the next-yaw target, and the bridge timing reports only
offline to isolate the hypothesized fixed-two-packet events. It fits a
sign-symmetric first-order release law at 4.0–4.5 m/s wheel-speed proxy,
±0.425-rad steering cells:

`r[k+1] = 0.23116 * r[k]` when a near-zero steering command changed one
25-ms packet ago, steering feedback is still stale, and yaw remains large.

Five training events from r01/r02 give decay 0.76884. The untouched r03
two-packet examples (three transitions, both steering signs) score RMSE
0.03528 rad/s, maximum 0.05766, and 100% below 0.1. This supports a real,
local nonlinear response description under the fixed-two-packet assumption.

The counterexample is equally important: r03 has one sensor-history state
that satisfies the same gate but takes three packets. Its yaw remains at
-1.2798 rad/s rather than decaying, so the two-packet law errs by 0.98389
rad/s. Bridge request timing identifies it as a three-packet response; the
currently visible command/feedback history does not. Applying the override
blindly over both signed cells worsens whole-cell validation RMSE
0.11835→0.13089 and maximum error 0.63743→0.98389, even though the fraction
below 0.1 rises 89.3%→92.2%. Therefore this candidate is saved as a conditional
offline comparator, not wired into MPC or odometry. The generated model and
full report are in
`live_runs/racing_model_diagnostics_20261008/yaw_release_decay_two_packet_v1/`.

This reconciles the “95%” premise with actual evidence. In these dedicated
captures two-packet onset occurred in 526/648 probes overall (81.2%) and
190/216 in r03 (88.0%), not 95%. The variable delay is strongly associated
with command-update age at the bridge's periodic request boundary: median
10.22 ms for two-packet versus 22.30 ms for three-packet cases, with no
separating request/response latency. The controller must not consume the
bridge diagnostic topic. No simulator or production runtime was changed.

Next: extend this explicit response-law procedure to the other speed × signed
steering × turn-in/unwind/reversal regions, first using existing captures.
Keep packet-mode filtering strictly as an offline diagnostic; any model
intended for the competition stack must also account for the observed
three-packet branch through legal causal inputs or a deterministic command
schedule. Do not claim full 0–12 m/s coverage: training has no 11.5–12.0 m/s
samples, and several held-out local cells remain unsupported or above the
0.1-rad/s criterion.

### 2026-10-08 audit of the largest held-out errors

I classified the atlas's 100 largest supported whole-run validation errors
instead of treating them as one generic fit problem: 69 are unwind, 22 turn-in,
6 hold, and 3 reversal. Nine of these 100 pair a stale/repeated archived IMU
yaw value with a simulator-truth yaw that has already changed by more than
0.1 rad/s. This is an input/label timing issue, not evidence that the raw bag
or the simulator truth is corrupt.

The worst example (dynamic-coupled validation r01, packet 6212) makes the
mechanism explicit. The exported sensor frame contains IMU yaw -1.1286 rad/s,
but same-packet simulator yaw is -0.1667 rad/s and next-packet truth is
+0.5554 rad/s. The raw IMU message has source stamp 1.83 ms before that packet
and reports -0.1667 rad/s; it was received about 14 microseconds after the
odometry callback. `load_capture` deliberately aligns IMU to odometry receipt
time using `_causal_scalar`, so it selected the previous IMU packet. The
fixed-25-ms exporter then pairs that receipt-causal input with same-packet
truth. This faithfully represents “latest IMU at the odometry callback,” but
mixes an input one packet old with a current-packet target. The correct model
must either carry causal IMU source-age/packet phase as a feature or explicitly
evaluate same-source-time alignment while accounting for what the runtime
controller can have received. I am not dropping these samples or silently
using future truth as an input.

Most of the large unwind errors are different: the current IMU and truth agree,
but yaw drops sharply when steering feedback reaches zero. In independent
swerve validation r01/r03 around 6.34 m/s, feedback is about ±0.0419 rad and
yaw about ±0.785 rad/s; after the command has been sent and the feedback snaps
to zero, next-packet truth is about ±0.017 rad/s. The v2 atlas predicts about
±0.35–0.54 rad/s. A neighboring sample with nearly the same yaw and steering
still persists for one packet before the release. The same release signature
appears in the 2026-10-08 mid-speed capture at constant throttle, so throttle
cut is not a sufficient explanation. A dynamic-coupled sample at 8.44 m/s
also changes from -1.232 to -0.103 rad/s as feedback releases from -0.0639 to
-0.0016 rad, while the atlas predicts -1.213. This is a speed-dependent
actuator/yaw release regime, not one isolated 6.3-m/s anomaly.

Command age is correlated with the response but is not a sufficient selector.
In low-steering unwind data, validation medians for next/current yaw retention
are about 0.01–0.04 at 25-ms command age, versus about 0.50–0.67 at age zero
in several 5.5–7.0-m/s wheel-speed cells. However, the response is bimodal
within those broad groups. I tested a training-only, run-balanced retention
law `r[k+1] = rho * r[k]`, stratified by 0.5-m/s wheel-speed, 0.05-rad
absolute-steering, and 0/25-ms command-age cells. On the targeted 1,123
held-out low-steer unwind rows it worsened RMSE 0.1489→0.1800 rad/s, p95
0.3598→0.4199, and fraction below 0.1 from 74.2% to 63.5%; only 3 of 24
validation runs improved. Simpler speed/age and steering/age versions were
worse still. This experiment is rejected; the group median is not a safe
replacement for a selector that resolves the two response modes.

A distinct high-error class is command reversal before measured steering
feedback reverses. In steering-validation r03 at 3.50 m/s, feedback is still
+0.3293 rad and yaw +0.877 rad/s while the command is -0.35 rad. The bridge
request first carries the negative command on that packet; next-packet truth
is -0.421 rad/s before feedback has caught up. The atlas predicts the old
positive-yaw direction. This is a command-to-plant phase/reversal regime and
must not be pooled with steady turn-in or release.

The high-steering fixed-two-packet release law remains a useful local
comparator, but the existing whole-cell counterexample still prevents blind
runtime promotion. The new all-outlier audit adds a second caution: the yaw
dataset itself needs explicit source-age accounting before its worst-tail
metrics can be interpreted as pure vehicle-model error. No simulator was
started and no odometry/MPC runtime, physics, or competition inputs changed.

Next work: audit the raw timestamp/source-age join across the high-error
validation examples, then fit separate release and pre-feedback reversal
specialists using training captures only. Compare on whole unseen runs and
report both the timing-stratified error and ordinary full-run error. Only a
candidate that improves its intended regime without worsening the remaining
run-level distribution is eligible for implementation.

### 2026-10-08: GT-target clarification and census of every large error

The fitting target is simulator ground-truth yaw rate at the next 25-ms tick,
not production `/odom` and not a GT-derived approximation to production
odometry. The frozen sensor-only atlas target is
`GT_yaw_rate[k+1] - IMU_yaw_rate[k]`; adding its prediction to the current
permitted gyro measurement yields the predicted next yaw rate. Its inputs and
expert selector use only current/past competition-observable sensors,
commands, measured wheel speed, measured steering, and a causal turn event.
GT speed and current GT yaw appear only in the offline audit's diagnostic
strata. A prior low-speed specialist draft briefly included GT-derived body
speed features; that leakage was removed before its reported refit/evaluation.

The frozen v2 atlas was rescored over 299,242 transitions from 36 whole-run
validation captures; test/final-test data remained sealed. One-step error is
0.03694 rad/s RMSE, 0.04012 rad/s p95, and 1.35362 rad/s maximum. There are
5,871 samples above 0.1 rad/s (1.962%). The run-macro RMSE is 0.03330
rad/s (median 0.03631; range 0.00167–0.05713), and the mean per-run fraction
within 0.1 rad/s is 98.09% (range 96.16–100%). These are correlated
25-ms samples and one-step rate scores, not independent trials, recursive
rollout, odometry position, or full-lap accuracy.

Error-event breakdown exposes a highly nonuniform failure distribution:

| Event | Validation rows | >0.1 rad/s | Error rate | Local-expert rows / errors | Fallback rows / errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hold | 232,199 | 596 | 0.26% | 229,163 / 551 | 3,036 / 45 |
| Reversal | 2,722 | 754 | 27.70% | 591 / 89 | 2,131 / 665 |
| Turn-in | 35,276 | 1,300 | 3.69% | 22,727 / 980 | 12,549 / 320 |
| Unwind | 29,045 | 3,221 | 11.09% | 14,081 / 1,353 | 14,964 / 1,868 |

Overall local-expert coverage is 89.08%. Large-error frequency is 2,973 / 266,562
(1.12%) on rows with local experts, versus 2,898 / 32,680 (8.87%) on global
event fallback rows. Thus unsupported regimes materially amplify the tail,
especially reversals, but lack of a local expert is not the whole problem:
unwind and turn-in also fail inside supported experts, and the supported
reversal error rate is still 15.06%. Many local-error experts have limited
support (780 of 2,973 outliers come from experts trained on fewer than 120
examples; 135 have only two independent training runs).

Across all 5,871 outliers, 3,221 are unwind, 1,300 turn-in, 754 reversal, and
596 hold. Observable-condition flags overlap and are diagnostic associations,
not causal proof: 3,022 (51.5%) have steering command/feedback gap >0.05 rad;
221 (3.8%) have current IMU-vs-current-GT yaw disagreement >0.1 rad/s;
485 (8.3%) have absolute rear-wheel/GT speed mismatch >1 m/s; and 775
(13.2%) have throttle command/feedback gap >0.05. In particular, current
IMU-vs-GT phase mismatch is not the dominant explanation for the full tail;
most failures occur while those two rates agree within 0.1 rad/s. Steering
transition state is much more common, especially reversal (592/754) and
unwind (1,791/3,221), but 1,983 errors have none of the five audited threshold
signatures. Those remain unresolved rather than being assigned a speculative
physics cause.

Speed-conditioned error rates are elevated at 4–6 m/s (1,283 / 37,048,
3.46%), versus 1.44% at 6–8 and 1.71% at 10–12 m/s; however, every evaluated
band contains large errors. This is not evidence for a speed cutoff or a
single missing-data interval. The complete row-level audit, with the selected
expert/fallback source, its training sample/run support, and GT values clearly
marked as diagnostics, is in
[`sensor_yaw_large_error_audit.json`](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/sensor_yaw_large_error_audit.json)
and [`selected_atlas_errors_over_0p1.csv`](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/selected_atlas_errors_over_0p1.csv).

The two-packet release fit remains GT-targeted but is not a production fix:
conditional exact-two-packet validation is promising, yet its observable
sensor history does not distinguish a matching three-packet case, and blind
application worsens whole-cell RMSE 0.11835→0.13089 with maximum error
0.63743→0.98389. Bridge packet timing remains offline-only and cannot be used
as a competition controller selector. Do not replace GT labels with production
odometry, and do not route the model using GT speed/yaw.

Next: use the all-row census to fit/test a training-only specialist or model
revision for unwind and command-ahead-of-feedback reversal, with particular
attention to fallback-supported cells and low-run-count experts. Score the
same whole-run validation captures and include event-specific and full-run
metrics. No new simulation is justified by this audit alone; the existing
captures contain the failure modes. No production Odom/MPC, physics, or
trajectory files were changed, and no simulator was launched.

One diagnostic response to the GT-target correction is a sensor-only
low-speed combined-slip specialist. On its targeted three whole-run
validation subset it changes yaw-rate RMSE 0.09081→0.08088 rad/s, p95
0.21049→0.15211, maximum 0.88564→0.75918, and errors above 0.1 from 40 to
30. It wins on two of three runs and loses on the crawl-steering run; the
three-run bootstrap interval for run-macro RMSE improvement crosses zero
([-0.08949, 0.00419]). Its unwind subset regresses (0.06852→0.11277 RMSE),
so it is a bounded research candidate, not a universal solution or runtime
promotion. A sensor-only 100-ms acceleration-corrected wheel-speed proxy
barely changes pooled metrics and does not remove the tail; it is not adopted.
The report and model are in
[`yaw_low_speed_combined_slip_v1`](../../live_runs/racing_model_diagnostics_20261008/yaw_low_speed_combined_slip_v1/).

I also compared the existing command-age v4 against v2 separately in every
event, using the frozen GT-targeted whole-run validation reports. It gives a
small reversal RMSE change (global 0.17279→0.16262, local 0.11955→0.11917
rad/s), but the local p95 is essentially flat (0.25140→0.25166) and its
maximum error worsens (1.35362→1.37530). It worsens unwind RMSE (global
0.08273→0.08323; local 0.09192→0.09298) and turn-in local RMSE
0.05246→0.05276. This does not solve the main tail and confirms that command
age alone is not a useful release/reversal selector. Keep v4 rejected; do not
promote it on the basis of its small event-average reversal change.

### 2026-10-08 follow-up: unwind/reversal neighborhood and timing join

A training-only adjacent-cell expert fit was evaluated across all 36
whole-run validation captures. It reduced one-step GT-yaw RMSE from 0.03694
to 0.03117 rad/s and the >0.1-rad/s count from 5,871 to 3,814. Unwind
outliers fell 3,221→1,208, but reversal only fell 754→710 and the global
maximum stayed 1.352 rad/s. The candidate's run-macro result beats the prior
fallback-gap model on 31/36 runs (95% paired-run bootstrap CI for RMSE delta
[-0.00388,-0.00220] rad/s). It is a substantive offline one-step improvement,
not recursive-rollout or production odometry evidence; its 646 tree experts
occupy about 176 MB and it is not integrated.

I joined candidate errors in the held-out 216-condition high-steer r03 run to
the measured 25-ms command-to-feedback onset. In matched probe conditions,
three-packet response phases had higher candidate window RMSE than two-packet
phases in 5/6 unwind pairs and 7/9 reversal pairs. Mean paired RMSE increases
were 0.0480 (95% interval [0.0038,0.1021]) and 0.0531 ([0.0265,0.0794])
rad/s. This timing effect is not a complete explanation: many two-packet
rows still exceed 0.1. In the narrow full-steer unwind gate, nearly identical
legal sensor/command states lead to next-GT yaw of about 0.25 versus 1.28
rad/s depending on the measured response phase. Existing bridge diagnostics
associate this branch with command update relative to the bridge request
boundary, not variable request-response latency. The debug timing topic stays
offline-only; current legal history does not yet identify the future branch.

The row/phase join is reproducible with
[`analyze_yaw_transition_timing_residuals.py`](../../tools/racing/specialists/analyze_yaw_transition_timing_residuals.py);
the results are
[`timing_residual_join.json`](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/unwind_reversal_neighborhood_v2_supported/timing_residual_join.json).

### 2026-10-08 policy correction and exact-two-packet error diagnosis

The packet-response assumption is now strict: **only an exactly two-packet
response phase is valid. Every other count, including three packets, is
invalid and excluded from yaw fitting, model selection, and validation.** The
earlier historical notes above that compare or model three-packet responses
are superseded; retain them only to explain how the old hypothesis arose, not
as evidence or a target. The focused r09 evaluator selects only
`reversal/2_packet` and `unwind/2_packet`, and states this policy in its output.
Its dedicated collector excludes all other response phases before model
selection/scoring and does not load the old two-versus-three-packet classifier.
The regime map has the same warning at its top.

The old candidate-selection reference for unwind was itself a
`unwind/3_packet` comparison, so the prior mirror-augmentation selection is
withdrawn. I recomputed plain-versus-mirrored model choice with leave-one
training-capture-out folds using only exact-two-packet rows: mirroring lowered
run-macro RMSE in all four folds for reversal (0.1676→0.1497 rad/s) and unwind
(0.0526→0.0466). The fresh r09 holdout was not used to choose the model.

I reran the candidate evaluation on whole-run r09 using only those two groups.
There are 90 reversal rows and 162 unwind rows. The mirrored sensor-history
ExtraTrees candidate improves over gyro persistence but still fails the
0.1 rad/s maximum-error requirement:

| Exact-two-packet group | Candidate RMSE / p95 / max (rad/s) | Rows over 0.1 | Persistence RMSE / max |
|---|---:|---:|---:|
| Reversal | 0.1373 / 0.3332 / 0.4478 | 27 / 90 | 0.5355 / 0.9810 |
| Unwind | 0.1094 / 0.1927 / 0.8235 | 19 / 162 | 0.3180 / 0.9956 |

The command-at-receipt versus bridge-request steering gap is not a complete
cause: reversal error varies non-monotonically across gap bins, and unwind has
12 errors above 0.1 among 119 rows whose gap is within 0.05 rad. The bridge
diagnostic was used only offline. A causal next-steering estimate improves
reversal RMSE 0.1393→0.1027 and lowers its >0.1 count 21→16, but max error is
still 0.4326. It does not improve unwind (RMSE 0.1083→0.1077; >0.1 count
15→20; max 0.7750→0.7881). Actuator prediction therefore explains part of
reversal, not both groups.

The high-error conditions are represented in training, but sparsely: the
3.5 m/s, 0.42 rad step-unwind condition has 4–6 valid phases (16–24 scored
rows) across 3–4 captures for the two turn signs. The worst reversal-step
conditions also have multiple training captures. These are not empty bins,
but the present model remains inaccurate there.

I decomposed the target into current IMU-to-GT yaw mismatch plus the next
25-ms GT yaw change. On all 90 reversal and 162 unwind rows, current yaw
mismatch is exactly zero in the aligned data; the error is in predicting the
physical next-step yaw change, not correcting current yaw. A model trained on
that GT yaw-change target produces numerically identical held-out errors to
the direct next-yaw residual target.

As an information-upper-bound test, I added current GT body velocity,
finite-difference body acceleration, sideslip angle, and rear-wheel/body speed
mismatch to offline models. This did not help reversal (RMSE 0.1425→0.1482,
errors >0.1: 32→33). For unwind, RMSE changed 0.1113→0.1048, but errors >0.1
increased 18→21 and max remained 0.7938. These GT values are diagnostic only;
they do not explain away the large error tail.

The earlier r07 GT-body-state score is superseded: its selector checked each
row's event but not the enclosing probe's event, mixing event-phase histories.
The corrected r09 analysis gates both phase and row to the same event and
exact-two-packet class.

Current conclusion: these are genuine next-step yaw-dynamics errors.
Command/feedback timing is a partial reversal factor; current GT u/v and their
simple derivatives do not resolve the tail. No model was promoted or
integrated into odometry/MPC, and no simulation was launched. Next, compare
the high-error states against training examples using only the valid
two-packet groups; do not fit, score, or draw conclusions about three-packet
responses.
Detailed interpretation and next step are in
[`YAW_RESPONSE_REGIME_MAP_20261008.md`](YAW_RESPONSE_REGIME_MAP_20261008.md).

### 2026-10-08 exact-two residual follow-up: age, history depth, and local experts

This update supersedes the preceding note that r10 was still in progress. The
finished 40-Hz r10 capture is
`openplane_yaw_error_highsteer_reversal_residual_train_r10_20261008`. It ran
480 scheduled phases, with 146 response phases passing the exact-two-packet
gate. Of the classified response phases, 12 three-packet and 2 one-packet
phases were excluded; no non-two-packet row was fitted or scored. The capture
measured 39.60 Hz command delivery and had no collision or bridge fault. Its
admitted dataset is
[`r10 manifest`](../../live_runs/openplane_yaw_error_highsteer_reversal_residual_train_r10_20261008_dataset/manifest.json).
The capture is training-only; r03 and r09 remain separate whole-run
evaluations, and no test/final-test data were opened.

All reported errors in this section are absolute one-step **yaw-rate** error
in rad/s at the next fixed 25-ms packet sample—not yaw-angle radians. The
exact-two collector is the only source for these tables. One-, three-, and
all other packet counts are invalid and excluded.

#### Where the held-out failures are

After admitting r10 to training, the training-only-selected mirrored yaw
candidate with the causal `hybrid_release_or_limit_k_minus_1` steering
feature gives:

| Whole-run holdout / event | Rows | Baseline mirrored >0.1 | Candidate >0.1 | Candidate RMSE / p95 / max (rad/s) |
|---|---:|---:|---:|---:|
| Ordinary 40-Hz r03, reversal | 156 | 13 | 7 | 0.1275 / 0.0861 / 0.8817 |
| Ordinary 40-Hz r03, unwind | 538 | 3 | 2 | 0.0211 / 0.0260 / 0.3813 |
| Packet-phase r09, reversal | 90 | 20 | 15 | 0.0902 / 0.2036 / 0.3690 |
| Packet-phase r09, unwind | 162 | 17 | 18 | 0.1092 / 0.1769 / 0.8755 |

The ordinary-rate reversal improvement is real but incomplete: 149/156 samples
are within 0.1, while one sample is still 0.882 rad/s wrong. Its seven misses
are at 3.0–4.0 m/s and 0.35–0.50 rad; six are 25–50 ms after the command
transition and one is 50–150 ms. The two remaining r03 unwind misses are both
from the 3.5 m/s, 0.42-rad, 100-ms ramp; they occur at about 99 and 125 ms.
The candidate is accurate outside these short transition windows.

The separate r09 packet-phase holdout is materially different. Reversal misses
are all early: 5/26 in the first 25 ms and 10/23 from 25–50 ms, with none after
50 ms. Unwind has 18/162 misses, including 13/60 from 50–150 ms and one
0.876-rad/s miss at 26 ms. r09 uses a different setpoint/packet-phase stimulus
and is not pooled with ordinary 40-Hz captures.

The actual-next-steering experiment is an offline forbidden-input diagnostic,
not a usable observer input. On ordinary r03, training with measured next
steering as an oracle feature reduces the extended-history reversal result to
0/156 above 0.1 (max 0.0939 rad/s). This indicates that next-steering
forecast error is an important part of the ordinary-rate residual. However,
the same oracle does not solve r09: it still has 14/90 reversal and 19/162
unwind errors above 0.1 without symmetry. Therefore actuator forecast error is
not the complete explanation; the packet-phase yaw response itself remains
under-modelled.

#### Falsification of two simple fixes

First, I changed the offline evaluator to compare current/past histories through
100 ms against the same histories through 500 ms. Left/right symmetry,
training-only whole-capture folds, and exact-two filtering were retained.
Longer history made no useful threshold improvement:

| Holdout / event | 100-ms history: >0.1, max | 500-ms history: >0.1, max |
|---|---:|---:|
| r03 reversal | 7/156, 0.8817 | 7/156, 0.8798 |
| r03 unwind | 2/538, 0.3813 | 3/538, 0.4418 |
| r09 reversal | 15/90, 0.3690 | 14/90, 0.3425 |
| r09 unwind | 18/162, 0.8755 | 21/162, 0.7757 |

The r09 reversal change is only one sample and the unwind tail worsens. A
longer raw history window alone is therefore not the missing state.

Second, I tested separate measured-state experts for speed-only (3 bins),
steering-only (3 bins), and joint speed×absolute-steering (9 bins), against one
global event-specific model. Bins use current measured rear-wheel mean speed
and current steering feedback; all models use the same causal steering
feature and the same exact-two rows. The architecture was selected using only
whole-capture training folds. The global candidate won for both events:

| Event | Global LOCO misses | Speed-only | Steering-only | Joint 9-cell |
|---|---:|---:|---:|---:|
| Reversal | 80 | 102 | 85 | 91 |
| Unwind | 56 | 77 | 71 | 94 |

No cell fell back for lack of the minimum support in the two holdouts, so this
is not a hidden coverage failure. The local bins sometimes lowered a single
run's RMSE, but worsened training-fold threshold counts and did not transfer to
r09. Current evidence rejects hard speed/steering bins as the fix for this
particular transient residual; it does not reject regime-specific models in
other operating regions.

#### Current diagnosis and next action

The high-steer ordinary-rate residual is a mixture: the yaw predictor can be
close when next steering is known, but its causal next-steering forecast has
rare large errors; a few residuals also remain even with the predicted
steering feature. In the separate packet-phase stimulus, even oracle next
steering leaves large yaw errors, so at least one additional dynamic state or
response mode is missing. The data do not support explaining this with speed
or steering alone, and simply extending sensor history from 100 to 500 ms did
not cure it.

The next model experiment should use a compact latent actuator/tire-response
state driven only by current and past steering feedback/commands, wheel
speeds, throttle, IMU, and timing. It must be selected by whole-capture
training folds, then checked separately on r03 and r09; r09 must remain a
different stimulus domain unless ordinary production setpoints match it. A
causal event-age or actuator-state feature is the targeted hypothesis, not
future measured steering or phase metadata. If that state cannot be inferred
from existing captures, the minimum new-data design is paired exact-two
reversal/unwind probes with identical current speed/steering/command state but
deliberately varied pre-transition steering and throttle/wheel-speed history.
Do not run another broad sweep. This update did not start a simulator, change
odometry/MPC, alter physics, or admit a non-two-packet sample.

Machine-readable results:

- [short-history r03](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/predicted_next_steering_yaw_audit_r03_loco.json)
- [short-history r09](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/predicted_next_steering_yaw_audit_r09_loco.json)
- [500-ms-history r03](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/predicted_next_steering_yaw_audit_r03_extended_loco.json)
- [500-ms-history r09](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/predicted_next_steering_yaw_audit_r09_extended_loco.json)
- [measured-state local experts](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/exact2_speed_steering_local_model_audit.json)
Existing data are enough to identify the phase mechanism; no additional
open-plane run was made. A later real-sim test is justified only to evaluate a
specific deterministic bridge/control-phase change or a causal phase estimate
from permitted timestamps. Runtime odometry/MPC, simulator physics, and topic
subscriptions remain unchanged.

### 2026-10-08 exact-two-packet residual localization and targeted capture

The acceptance rule for this analysis is strict: **only samples whose measured
response is exactly two simulator packets are included**. One-, three-, and
all other packet-count samples are invalid and excluded from fitting, model
selection, and scoring. The residual target reported here is next-tick
simulator-GT yaw-rate error in **rad/s** (not yaw-angle error in radians).

The remaining error is concentrated in fast transients, not steady turning.
On the ordinary 40-Hz-command r03 whole-run capture, the mirrored sensor-history
candidate's reversal group has 16/156 samples above 0.1 rad/s (RMSE 0.1520,
p95 0.4024, maximum 0.8656). Fourteen of its 16 threshold violations occur
within 50 ms of the command transition (14/40 samples in that window); after
50 ms only 2/116 samples exceed the threshold. The same candidate transfers
substantially better on ordinary-rate unwind: 3/538 above 0.1 (RMSE 0.0203,
maximum 0.3607), with all three outliers between 50 and 150 ms. On the
200-Hz-setpoint packet-phase captures, both reversal and unwind degrade; those
captures are kept as a separate stimulus context and are not pooled as if
they represented 40-Hz commands.

The error is in physical next-step yaw response: aligned current IMU yaw rate
matches current simulator truth exactly on the audited samples. The existing
legal sensor-history model and a causal predicted-next-steering feature do not
remove the tail. Future-measured steering and bridge debug timing remain
diagnostic only and are never predictor inputs. Therefore the current evidence
does not justify a runtime correction yet.

The 40-Hz open-plane run
`openplane_yaw_error_highsteer_reversal_residual_train_r10_20261008` is in
progress. It repeats only the poorly supported 3–4 m/s, 0.35–0.50 rad
high-steering reversal and unwind pockets, with signed turns, randomized order,
two transition ages, and reset isolation. This is a focused attempt to separate
the early command/steering-response transient from the established later
response, not another broad sweep. At the time of this update it is running
without a reported collision; no r10 samples have been admitted to fitting or
scoring yet. After completion, the bag must pass probe/capture checks, and only
exactly-two-packet rows will be exported. The first evaluation will score the
pre-existing frozen candidate on r10 before r10 is admitted to training. Any
revised model must then improve held-out whole captures and reduce the
phase-level maximum/tail; an average-RMSE gain alone is not sufficient for the
requested 100% within-0.1 threshold.

No odometry/MPC model, runtime topic, or simulator physics has been changed.

The rerun now records response-count histograms so the exclusion is auditable.
For example, it found 34 three-packet phases in r01, 23 in r02, 24 in r03,
16 in r04, and 10 in r06; **all were excluded**, as were every other
non-two-packet count. These are exclusion counts only, not evidence about a
three-packet behavior or a regime. The report lists `packet_response_steps_included`
as `[2]` for every capture:
[`targeted_group_candidate_comparison_r09.json`](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/targeted_group_candidate_comparison_r09.json).
