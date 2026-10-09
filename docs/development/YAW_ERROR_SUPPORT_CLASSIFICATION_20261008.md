# Yaw prediction error and data-support audit — 2026-10-08

## Scope and meaning of the threshold

The audited `0.1` threshold is absolute **future yaw-rate prediction error in
rad/s**, not yaw angle in radians. The principal table below is the 250 ms
prediction horizon. Errors are scored against simulator rigid-body truth. The
v3 and v4 audits use whole-run validation captures; test/final-test splits
remain sealed.

The full machine-readable audits are:

- `live_runs/racing_model_diagnostics_20261007/yaw_multihorizon_teacher_v3_residual_grid/expanded_error_support_audit_v5_20261008.json`
- `live_runs/racing_model_diagnostics_20261007/yaw_multihorizon_teacher_v4_targeted_gaps/expanded_error_support_audit_v1_20261008.json`
- `live_runs/racing_model_diagnostics_20261007/yaw_multihorizon_teacher_v5_gap_refit_20261008/expanded_error_support_audit_v1_20261008.json`

All cell error/support counts below describe the frozen v4 baseline, before
admitting the new high-steer r01 and low-throttle r02 training exports. They
are the diagnosis used to target data collection, not post-refit claims. The
cell and joint-support audit must be rerun on the retrained candidate before
deciding whether conditional sweeps are still needed.

## Follow-up capture status

The focused high-steering event-gap profile is implemented as
`yaw_error_highsteer_reversal_gapfill`. It randomizes 216 reset-isolated
probes over 3.0/3.5/4.0 m/s and measured steering points 0.35–0.50 rad,
both turn signs, onset/unwind/reversal events, 25/100/300 ms transitions,
and 0.25/0.75 s response ages. Each probe records the existing 40 Hz
odometry, wheel encoders, actuator command/feedback, IMU, and bridge timing;
the probe speed gate is enabled. The conservative schedule budget is
3,817.4 s per capture, including the five-second finish margin.

Training capture `openplane_yaw_error_highsteer_reversal_train_r01_20261008`
completed all 648 phases (216/216 valid probes, zero invalid, zero collisions,
zero bridge timing faults, not aborted). The 40 Hz source streams measured
39.76–39.77 Hz with p95 gaps 26.95–27.27 ms; exact packet IDs were available
for all joined samples. Receipt-gap maxima reached 205–211 ms in a small tail,
so the export uses the fixed 25 ms packet timebase and breaks sequences at
packet discontinuities rather than interpreting receipt jitter as physics dt.
The complete whole-run export contains 33,410 samples in 231 reset/gap-isolated
sequences (3.1 MB compressed); its persistent files are under
`live_runs/racing_model_diagnostics_20261007/yaw_error_highsteer_reversal_train_r01_dataset/`.

All 216 probe medians were within 0.15 m/s of their requested speed, across
2.973–2.989 m/s, 3.471–3.485 m/s, and 3.962–3.977 m/s groups. Steering
feedback was within 0.05 rad for at least 95% of target-plateau samples in
207/216 probes. The exceptions were almost entirely fast, full-angle
reversals: 8/24 25 ms reversal probes had less than 95% of plateau samples
within 0.05 rad, with command-to-feedback p90 gaps up to 0.236 rad. All
24/24 300 ms reversal ramps reached the requested target throughout the
measured plateau. This is direct evidence that transient steering slew and
response history matter; it is not a sparse steady speed/angle cell. The
whole-run bag audit reports 216/216 expected probes, 12 probe sequences split
at packet discontinuities, and no collision/fault/abort.

The independent high-steer training repeat r02 completed with the same
216/216-probe, reset-isolated schedule and passed its bag/timing/feedback audit.
Both r01/r02 remain training-only. A separate r03 whole-run final-test capture
is not justified unless a model selected on validation shows a repeatable
region-level gain. After each bag closes, export only a quality-passing run with
`tools/vehicle_dynamics_learning/prepare_dataset.py --continuous-whole-run
--fixed-packet-timebase`, assigning its explicit train/validation split. The
exporter preserves 25 ms packet time, breaks sequences at recorded resets,
and keeps simulator truth as labels. Train on r01/r02; leave r03 entirely out
of fitting/checkpoint selection and score it once as a whole-run holdout.

The failed crawl-speed hold has a separate calibration profile implemented,
`yaw_error_crawl_throttle_calibration`. Its 70 randomized conditions are
1/2/3/4/5% fixed throttle × 0/±0.20/±0.35/±0.50 rad steering × two repeats,
with reset to spawn for every condition and an eight-second recorded response.
It deliberately has no commanded-speed gate: measured speed, throttle
feedback, wheel/body mismatch, yaw, IMU, and timing determine which low-speed
cells it actually covers. The schedule check passes and its conservative
budget is under 17 minutes.

The first attempt, `openplane_yaw_error_crawl_throttle_calibration_train_r01_20261008`,
was interrupted at 24/210 phases and is excluded as a complete-run fit. Eight
probe conditions completed with valid phase markers and eight reset-recovery
events; the runner then started a ninth reset for the next condition and
deasserted it during shutdown. A post-stop bag audit confirms zero collisions,
zero bridge faults, full packet-ID matching, and 39.95 Hz within-phase sensor
streams (p95 gaps about 25.8 ms). The interruption followed an incorrect
initial reading of the profile dispatch, not a missing reset.
The partial per-condition audit is
`live_runs/openplane_yaw_error_crawl_throttle_calibration_train_r01_20261008/partial_calibration_audit.json`,
generated by `tools/racing/specialists/audit_yaw_crawl_throttle_calibration.py`.
The full randomized replacement,
`openplane_yaw_error_crawl_throttle_calibration_train_r02_20261008`, completed
all 210 phases and passed the stricter audit: 70/70 probes, 70/70 reset
recoveries, no missing conditions, no invalid probes, zero collisions,
zero bridge faults, 100% simulator-packet matching, and 39.95 Hz within-phase
streams (25.91 ms p95 odometry gap, 53.32 ms maximum). Its machine-readable
result is `live_runs/openplane_yaw_error_crawl_throttle_calibration_train_r02_20261008/calibration_audit.json`.
The persistent training export contains 25,228 samples in 70 reset-isolated
sequences under
`live_runs/racing_model_diagnostics_20261007/yaw_error_crawl_throttle_calibration_train_r02_dataset/`.
The aborted pilot is not admitted to model training.

Across all 70 conditions, final-two-second median speeds by throttle command
were 0.244, 0.489, 0.732, 0.974, and 1.213 m/s for 1–5%, respectively. By
condition, the measured range was 0.237–1.25 m/s: 26 conditions below
0.5 m/s, 28 in 0.5–1.0, and 16 in 1.0–1.5. Thus this test covers the actual
low-speed band that the earlier target-speed crawl schedule failed to attain.
Across the 35 throttle×steering combinations, both repetitions had identical
final-window speed and yaw medians at reported precision; the maximum paired
wheel/body-mismatch difference was 0.0026 m/s. Steering feedback p90 error
was at most 0.0002 rad, throttle feedback error was below 9e-10, and the
largest condition-level wheel/body-mismatch median was 0.0246 m/s (maximum
0.1131). For 0.20–0.50 rad, signed yaw was symmetric and measured 0.928–0.953
of the no-slip bicycle rate `u*tan(steering)/0.324`; absolute roll stayed
below 0.019 rad. This measured surface shows steady low-speed high-steer
response is orderly, so those points do not explain the large transient
errors; it is not itself a yaw-observer validation.

The analyzer also fit the local steady relation using runtime-like inputs
(odometry body speed and measured steering feedback) and simulator-truth yaw
only as the label. A single yaw-authority factor `k=0.961264` in
`r = k*u_odom*tan(delta_feedback)/0.324` reduces the 70-point absolute-error
p95 from 0.0608 to 0.00625 rad/s and the maximum from 0.0646 to 0.00728. When
both repetitions of each throttle×steering condition are left out together,
p95 is 0.00704 and max 0.00837 rad/s. Leaving each 1–5% throttle level out in
turn (unseen speed level) gives p95 0.00705 and max 0.00836. No held-out point
in this capture exceeds 0.1 rad/s. This is a strong regime-local candidate,
not independent-run validation: all conditions share one capture, the target
is steady yaw rather than a 25–1000 ms transient, and it has not been
integrated into odometry or MPC.

An independent-capture check was attempted on the existing crawl/low-speed
validation exports using only their causal measured streams: a 0.5 s window
had to remain within 0.10 m/s odometry-speed range, 0.01 rad steering range,
and 0.005 throttle-feedback range, at 0.2–1.5 m/s and 0.20–0.50 rad steering.
The crawl-steering validation capture has 5,294 samples in the speed/steering
domain but none satisfy that steady-window gate; the low-speed steering and
high-steer validation captures have only 8 and 7 domain samples respectively,
also with none satisfying the gate. These runs therefore cannot independently
validate the steady correction. If the refit still relies on this correction,
the minimum follow-up is one separately randomized 70-condition calibration
repeat, scored as an untouched whole-run validation—not another dense
speed-by-angle sweep.

## Model-level result

At 250 ms, v3 has RMSE 0.1153 rad/s, p95 0.2364, and 85.08% of validation
samples within 0.1. v4 reduces pooled RMSE to 0.0968 and p95 to 0.2029, with
85.49% within 0.1, but its worst error rises from 1.620 to 1.770 rad/s. On
the 32 common older validation runs, v4's run-macro RMSE is worse than v3;
the paired bootstrap interval for the increase is +0.00169 to +0.00488
rad/s. This is a modest pooled improvement, not a solved yaw model.

Across horizons 25/100/250/500/750/1000 ms, v4 has respectively 20/24/34/34/33/30
speed-by-steering cells with at least 100 validation samples and p95 error
above 0.1. At 250 ms, 5 of those 34 cells are sparse by the audit rule
(`<500` training rows or `<8` independent training runs), 27 are dense
(`>=1000` rows and `>=10` runs), and 2 are intermediate. More generic samples
cannot explain most of the remaining failures.

| Horizon | Failing cells | Sparse | Dense | Intermediate |
|---:|---:|---:|---:|---:|
| 25 ms | 20 | 4 | 13 | 3 |
| 100 ms | 24 | 5 | 17 | 2 |
| 250 ms | 34 | 5 | 27 | 2 |
| 500 ms | 34 | 4 | 28 | 2 |
| 750 ms | 33 | 4 | 27 | 2 |
| 1000 ms | 30 | 4 | 24 | 2 |

## Every speed × steering cell with systematic error at 250 ms

Rows are `validation samples / independent validation runs` and
`training samples / independent training runs`. “Systematic” here means the
cell p95 exceeds 0.1; observed isolated tail errors are listed separately.

| Speed m/s / abs steering rad | Validation n / runs | Training n / runs | v3 p95 | v4 p95 | v4 within 0.1 | v4 samples >0.1 | Support |
|---|---:|---:|---:|---:|---:|---:|---|
| 0–0.5 / 0.2–0.35 | 138 / 5 | 92 / 4 | 0.182 | 0.214 | 3.6% | 133 | sparse |
| 0–0.5 / 0.35–0.525 | 133 / 3 | 91 / 3 | 0.110 | 0.218 | 0.0% | 133 | sparse |
| 0.5–1 / 0–0.1 | 5,092 / 30 | 8,815 / 44 | 0.232 | 0.154 | 94.2% | 294 | dense |
| 0.5–1 / 0.2–0.35 | 766 / 6 | 1,575 / 5 | 0.814 | 0.380 | 5.6% | 723 | sparse |
| 0.5–1 / 0.35–0.525 | 340 / 5 | 716 / 5 | 0.765 | 0.475 | 19.4% | 274 | sparse |
| 1–1.5 / 0–0.1 | 5,926 / 30 | 10,294 / 44 | 0.351 | 0.235 | 90.9% | 542 | dense |
| 1–1.5 / 0.1–0.2 | 146 / 4 | 329 / 10 | 0.796 | 0.520 | 37.7% | 91 | sparse |
| 1–1.5 / 0.2–0.35 | 2,389 / 7 | 4,733 / 10 | 0.765 | 0.395 | 33.1% | 1,598 | dense |
| 1–1.5 / 0.35–0.525 | 1,186 / 6 | 2,384 / 9 | 0.773 | 0.530 | 28.6% | 847 | intermediate |
| 1.5–2 / 0–0.1 | 6,669 / 30 | 11,797 / 44 | 0.255 | 0.281 | 90.6% | 625 | dense |
| 1.5–2 / 0.1–0.2 | 854 / 5 | 1,696 / 12 | 0.490 | 0.417 | 78.3% | 185 | dense |
| 1.5–2 / 0.2–0.35 | 3,637 / 8 | 7,220 / 10 | 0.531 | 0.352 | 47.6% | 1,906 | dense |
| 1.5–2 / 0.35–0.525 | 3,348 / 7 | 6,677 / 9 | 0.563 | 0.509 | 30.7% | 2,321 | intermediate |
| 2–3 / 0–0.1 | 12,712 / 32 | 21,938 / 43 | 0.160 | 0.190 | 91.6% | 1,062 | dense |
| 2–3 / 0.1–0.2 | 982 / 11 | 1,794 / 14 | 0.640 | 0.709 | 53.4% | 458 | dense |
| 2–3 / 0.2–0.35 | 4,196 / 14 | 7,950 / 18 | 0.392 | 0.417 | 25.5% | 3,125 | dense |
| 2–3 / 0.35–0.525 | 4,582 / 13 | 8,343 / 15 | 0.444 | 0.444 | 69.1% | 1,416 | dense |
| 3–4 / 0.1–0.2 | 2,135 / 14 | 3,299 / 19 | 0.223 | 0.234 | 85.2% | 317 | dense |
| 3–4 / 0.2–0.35 | 3,865 / 15 | 7,009 / 18 | 0.344 | 0.368 | 78.5% | 832 | dense |
| 3–4 / 0.35–0.525 | 3,702 / 8 | 6,784 / 11 | 0.360 | 0.253 | 90.5% | 351 | dense |
| 4–6 / 0–0.1 | 25,561 / 30 | 44,984 / 42 | 0.176 | 0.185 | 90.8% | 2,356 | dense |
| 4–6 / 0.1–0.2 | 5,362 / 17 | 9,582 / 27 | 0.239 | 0.262 | 53.0% | 2,521 | dense |
| 4–6 / 0.2–0.35 | 7,085 / 18 | 10,032 / 23 | 0.234 | 0.242 | 83.6% | 1,159 | dense |
| 4–6 / 0.35–0.525 | 3,153 / 11 | 5,078 / 12 | 0.240 | 0.270 | 81.4% | 586 | dense |
| 6–8 / 0–0.1 | 39,314 / 28 | 56,249 / 35 | 0.185 | 0.190 | 85.3% | 5,781 | dense |
| 6–8 / 0.1–0.2 | 9,127 / 23 | 13,649 / 26 | 0.216 | 0.217 | 74.3% | 2,342 | dense |
| 6–8 / 0.2–0.35 | 15,704 / 21 | 16,778 / 22 | 0.147 | 0.154 | 91.1% | 1,392 | dense |
| 6–8 / 0.35–0.525 | 11,821 / 16 | 11,561 / 18 | 0.114 | 0.118 | 93.3% | 794 | dense |
| 8–10 / 0–0.1 | 27,677 / 23 | 48,051 / 30 | 0.147 | 0.147 | 88.2% | 3,271 | dense |
| 8–10 / 0.1–0.2 | 5,954 / 16 | 9,976 / 22 | 0.173 | 0.168 | 88.3% | 698 | dense |
| 8–10 / 0.2–0.35 | 3,099 / 17 | 5,334 / 17 | 0.129 | 0.147 | 93.1% | 215 | dense |
| 8–10 / 0.35–0.525 | 1,925 / 10 | 5,118 / 11 | 0.121 | 0.123 | 94.2% | 111 | dense |
| 10–12 / 0–0.1 | 24,662 / 18 | 29,328 / 18 | 0.147 | 0.146 | 91.2% | 2,182 | dense |
| 10–12 / 0.1–0.2 | 8,602 / 18 | 11,301 / 18 | 0.129 | 0.139 | 90.8% | 793 | dense |

Three additional cells have individual errors above 0.1 but p95 at or below
0.1 (rare-tail rather than broad failure):

| Speed / steering | Validation n | Errors >0.1 | p95 | Maximum | Training support |
|---|---:|---:|---:|---:|---:|
| 0–0.5 m/s / 0–0.1 rad | 27,897 | 353 | 0.0002 | 0.8873 | 47,420 rows / 45 runs |
| 3–4 m/s / 0–0.1 rad | 10,753 | 460 | 0.0892 | 0.7618 | 18,108 rows / 42 runs |
| 10–12 m/s / 0.2–0.35 rad | 1,614 | 78 | 0.0948 | 0.3693 | 1,762 rows / 7 runs |

The 0–0.1 rad near-straight cells have large sample counts, so their isolated
outliers are not a broad data-coverage problem. The 10–12 m/s, 0.2–0.35 rad
tail has only seven training runs and is a small support gap, but most samples
are already under threshold.

## What is data-limited versus model-limited

1. **Confirmed low-speed support gaps:** below 0.5 m/s with steering above
   0.2 rad, only 91–92 training rows from 3–4 runs exist. At 0.5–1 m/s and
   0.2–0.525 rad, there are 716–1,575 rows but only five training runs. Run
   diversity and actual state coverage are inadequate here.
2. **Dense cells with large residual:** from 1–4 m/s at moderate/high steering
   and across 4–12 m/s, most failing cells have thousands of training rows
   and 10–43 runs. The errors survive v4 and cannot be explained by a simple
   absence of speed/steering points. These need model/history/interaction
   diagnosis, not a repeated generic sweep.
3. **Joint transition gaps hidden by the 2-D table:** at 250 ms, the expanded
   audit finds 103 failing speed×steering×steering-command/feedback-gap cells
   (58 sparse), 63 failing speed×steering×throttle-gap cells (22 sparse), and
   137 failing speed×steering×wheel/body-speed-mismatch cells (64 sparse).
   These sets overlap; they are not 300 independent test points. The notable
   3–4 m/s, 0.2–0.35 rad, wheel/body mismatch >1 m/s slice has only 78 train
   rows / 13 runs and p95 0.707 rad/s. A separate 2–3 m/s, 0.2–0.35 rad,
   throttle-gap 0.05–0.1 slice has 873 rows / 7 runs and p95 0.666.
4. **Dense high-steer wrong-direction samples:** the largest residuals around
   3.47 m/s and ±0.42–0.50 rad occur with only about 0.055 m/s wheel/body
   mismatch and negligible command/feedback steering gap. The base bins have
   thousands of rows. This points to response history/representation rather
   than more wheelspin tests. Do not attribute it to slip without evidence.
5. **Already-covered high-speed throttle cut:** the 11.24 m/s, near-zero-steer
   error occurs during a throttle cut and is represented in the existing
   high-speed throttle captures. Do not repeat it unless the frozen model
   continues to fail the exact held-out slice.

## Time-history diagnosis: a sparse reversal event hidden by dense bins

The 250 ms worst-held-out sample is not a steady corner. It is from
`openplane_yaw_error_lowspeed_highsteer_validation_r03_20261008`: at about
3.47 m/s, measured steering is +0.50 rad and current yaw rate is +1.406 rad/s.
The steering command reverses from +0.50 to -0.50 rad inside the 250 ms
forecast window. In the same capture, physical steering follows the reversal
within about 50 ms, lateral acceleration changes from about +4.9 to -4.2
m/s², and ground-truth yaw reaches -0.833 rad/s. The teacher predicts
+0.937 rad/s, retaining the old turn direction: error 1.770 rad/s. At the
sample start, command/feedback steering gap is zero and rear-wheel/body-speed
mismatch is only 0.055 m/s. This case is therefore not explained by missing
steady-state data, a current actuator tracking error, or wheelspin; it is a
fast reversal response that the direct ExtraTrees teacher does not represent
well.

An event-conditioned recount finds only 43 matching 3–4 m/s, |steering|≥0.45
rad, command-reversal/yaw-reversal training samples across 9 reset-isolated
sequences in 2 training runs. The independent validation run has 31 such
samples across 4 sequences; 22/31 exceed 0.1 rad/s, with RMSE 0.718, p95
1.510, and maximum 1.770 rad/s. Thus the ordinary speed×steering cell is
dense, but the sequence-conditioned reversal is genuinely thin and has too
few independent training runs. These 25 ms rows overlap within each event:
the effective independent count is at most 9 training reversal episodes and
4 validation episodes, not 43 and 31 independent samples. The data gap is in
event history, not in the static speed/angle rectangle.

A distinct held-out failure is acceleration plus rear-wheel spin. At about
3.86 m/s and 0.20 rad steering, measured yaw is 2.654 rad/s while the teacher
predicts 1.071; rear-wheel mean exceeds body speed by 2.63 m/s, with
longitudinal acceleration about 3.28 m/s² and lateral acceleration about
8.72 m/s². Steering command tracks feedback and throttle command/feedback
gap is small at that instant. This is consistent with an under-supported
wheel/body-slip interaction, but is an association rather than proof of tire
force causality. The exact 3–4 m/s, 0.2–0.35 rad, >1 m/s mismatch cell has
only 78 training / 80 validation samples and p95 error 0.707 rad/s.

Simply adding more past history is not a demonstrated fix: at 250 ms the
1.6 s-history ExtraTrees variant has RMSE 0.1046, p95 0.2293, and 83.20%
within 0.1, versus the current-history variant's 0.0968, 0.2029, and 85.49%.
That comparison is exploratory and uses fewer eligible validation samples,
but it rejects the claim that a longer history window alone has already
solved the event response.

## v5 refit after adding the new clean captures

The completed v5 refit uses 58 clean training runs (638,702 samples) and 34
whole-run validation captures (255,396 samples). The source audit confirms no
test or final-test arrays were read. New data include the two high-steer
reversal repeats and crawl/low-throttle captures. The same v4 estimator and
feature recipe were refit, so this isolates the effect of additional examples
more than an architecture change.

The additions did not generalize as a broad accuracy improvement. On the 34
paired validation runs, the bootstrap interval for the change in run-macro
RMSE (v5 minus v4) is above zero at all current-only horizons and at 100–1000
ms for the 1.6 s-history variant. The 25 ms history result is indistinguishable
from zero. Values are rad/s:

| Horizon | Current-only run-macro ΔRMSE (95% paired bootstrap CI) | 1.6 s history ΔRMSE (95% paired bootstrap CI) |
|---:|---:|---:|
| 25 ms | +0.00088 [+0.00041, +0.00134] | +0.00025 [−0.00021, +0.00069] |
| 100 ms | +0.00244 [+0.00154, +0.00338] | +0.00238 [+0.00175, +0.00299] |
| 250 ms | +0.00150 [+0.00040, +0.00251] | +0.00135 [+0.00028, +0.00240] |
| 500 ms | +0.00412 [+0.00258, +0.00571] | +0.00213 [+0.00094, +0.00359] |
| 750 ms | +0.00551 [+0.00348, +0.00798] | +0.00306 [+0.00141, +0.00509] |
| 1000 ms | +0.00226 [+0.00023, +0.00477] | +0.00427 [+0.00219, +0.00689] |

At 250 ms, current-only pooled p95 changes only from 0.20290 to 0.20312
rad/s; the 0.35–0.525 rad steering-band p95 stays 0.29198 to 0.29214. The
history model's pooled p95 moves 0.22927 to 0.22504, but its run-macro RMSE
still worsens and high-steer p95 moves 0.27743 to 0.27945. No v5 variant is
promoted. This is direct evidence that adding broad clean data alone did not
close the dense error regions; the remaining collection must target verified
joint-support gaps, while dense failures require a model/feature explanation.

The absolute v5 validation error remains horizon-dependent. “Over 0.1” below
is the fraction of samples with absolute yaw-rate error >0.1 rad/s; it is not
an angle-error fraction. Both models are evaluated on whole unseen runs:

| Horizon | Current RMSE | Current p95 | Current >0.1 | 1.6 s-history RMSE | History p95 | History >0.1 |
|---:|---:|---:|---:|---:|---:|---:|
| 25 ms | 0.0602 | 0.1242 | 6.7% | 0.0568 | 0.1176 | 6.7% |
| 100 ms | 0.0874 | 0.1932 | 12.0% | 0.0958 | 0.2183 | 15.2% |
| 250 ms | 0.0979 | 0.2031 | 14.6% | 0.1053 | 0.2250 | 16.9% |
| 500 ms | 0.1074 | 0.2316 | 16.2% | 0.1089 | 0.2246 | 17.5% |
| 750 ms | 0.1119 | 0.2456 | 17.3% | 0.1109 | 0.2368 | 19.3% |
| 1000 ms | 0.1124 | 0.2462 | 17.3% | 0.1074 | 0.2326 | 18.8% |

RMSE and p95 are rad/s. The increasing >0.1 fraction with forecast horizon,
together with worse paired run-macro scores after the broad-data refit, is
consistent with missing response-state/history representation, not simply too
few rows in the full speed/steering grid.

The largest run-level 250 ms errors are concentrated, not uniformly bad:
`lowspeed_steering_validation_r03` has p95 0.393 rad/s and only 70.6% within
0.1; `crawl_steering_validation_r03` has p95 0.377 and 58.3% within; and
`lowspeed_highsteer_validation_r03` has p95 0.335 and 76.7% within. The next
two dynamic-coupled runs have p95 0.249/0.244. This confirms two overlapping
needs: genuine low-speed sample/run diversity, and a model that handles rapid
steering/slip history rather than treating each sample as an isolated
speed-angle point.

The expanded v5 audit still finds systematic speed×steering cells at all
forecast horizons. The support split (validation cell n≥100 and cell p95
>0.1 rad/s; sparse means <500 training rows or <8 independent training runs)
is:

Across all six horizons, 37 distinct speed×steering cells fail at one or more
horizons; 18 fail at every horizon from 25 ms through 1,000 ms. Sixteen of
those 18 are dense by the support rule and two are the known 0.5–1 m/s
moderate/high-steering gaps. Thus the repeatable broad residual is a model
limitation; a finite low-speed gap-fill addresses only the two support-limited
cells, not the other 16.

| Horizon | Failing cells | Sparse | Dense | Intermediate |
|---:|---:|---:|---:|---:|
| 25 ms | 20 | 4 | 16 | 0 |
| 100 ms | 25 | 4 | 21 | 0 |
| 250 ms | 34 | 4 | 30 | 0 |
| 500 ms | 34 | 3 | 31 | 0 |
| 750 ms | 32 | 2 | 30 | 0 |
| 1000 ms | 30 | 2 | 27 | 1 |

At 250 ms, the four remaining sparse speed×steering failures are all in the
low-speed edge. Above 1.5 m/s, the failing speed×steering cells are already
dense by this criterion; the prominent 1–4 m/s moderate/high-steering errors
and 4–12 m/s residuals therefore need model/feature work, not another broad
speed/angle sweep. Joint interaction bins remain more informative: among
validation strata with at least 100 samples and p95 >0.1 rad/s, there are 77
steering-command/feedback-gap strata (27 sparse), 57 throttle-gap strata (12
sparse), and 119 wheel/body-mismatch strata (39 sparse). These sets overlap.

The table below is the complete post-v5 250 ms speed×steering classification.
It contains every cell with at least 100 held-out samples and p95 absolute
error above 0.1 rad/s. Support labels use the audit rule: sparse is fewer than
500 training rows **or** fewer than 8 independent training runs; dense is at
least 1,000 rows **and** at least 10 runs. Dense is only a marginal
speed/steering support statement; it does not mean the causal transition or
slip history is covered.

| Speed m/s | abs steer rad | Validation n / runs | Training n / runs | p95 rad/s | Errors >0.1 | Support |
|---|---:|---:|---:|---:|---:|---|
| 0–0.5 | 0.20–0.35 | 138 / 5 | 5,993 / 6 | 0.175 | 94 | sparse |
| 0–0.5 | 0.35–0.525 | 133 / 3 | 2,954 / 5 | 0.206 | 133 | sparse |
| 0.5–1 | 0–0.1 | 5,092 / 30 | 15,930 / 48 | 0.125 | 286 | dense |
| 0.5–1 | 0.20–0.35 | 766 / 6 | 9,846 / 7 | 0.333 | 576 | sparse |
| 0.5–1 | 0.35–0.525 | 340 / 5 | 4,749 / 7 | 0.442 | 229 | sparse |
| 1–1.5 | 0–0.1 | 5,926 / 30 | 18,009 / 48 | 0.215 | 551 | dense |
| 1–1.5 | 0.1–0.2 | 146 / 4 | 3,013 / 11 | 0.462 | 103 | dense |
| 1–1.5 | 0.20–0.35 | 2,389 / 7 | 12,072 / 12 | 0.374 | 1,562 | dense |
| 1–1.5 | 0.35–0.525 | 1,186 / 6 | 5,818 / 11 | 0.536 | 757 | dense |
| 1.5–2 | 0–0.1 | 6,669 / 30 | 16,909 / 47 | 0.303 | 613 | dense |
| 1.5–2 | 0.1–0.2 | 854 / 5 | 3,814 / 13 | 0.461 | 175 | dense |
| 1.5–2 | 0.20–0.35 | 3,637 / 8 | 11,243 / 11 | 0.359 | 1,469 | dense |
| 1.5–2 | 0.35–0.525 | 3,348 / 7 | 8,949 / 10 | 0.530 | 2,477 | dense |
| 2–3 | 0–0.1 | 12,712 / 32 | 30,392 / 46 | 0.207 | 1,090 | dense |
| 2–3 | 0.1–0.2 | 982 / 11 | 2,434 / 17 | 0.668 | 409 | dense |
| 2–3 | 0.20–0.35 | 4,196 / 14 | 9,072 / 21 | 0.436 | 2,169 | dense |
| 2–3 | 0.35–0.525 | 4,582 / 13 | 16,186 / 18 | 0.405 | 1,330 | dense |
| 3–4 | 0.1–0.2 | 2,135 / 14 | 3,875 / 21 | 0.242 | 339 | dense |
| 3–4 | 0.20–0.35 | 3,865 / 15 | 11,721 / 20 | 0.360 | 854 | dense |
| 3–4 | 0.35–0.525 | 3,702 / 8 | 19,038 / 13 | 0.251 | 347 | dense |
| 4–6 | 0–0.1 | 25,561 / 30 | 45,652 / 44 | 0.195 | 2,571 | dense |
| 4–6 | 0.1–0.2 | 5,362 / 17 | 9,664 / 29 | 0.264 | 2,335 | dense |
| 4–6 | 0.20–0.35 | 7,085 / 18 | 10,081 / 25 | 0.261 | 1,272 | dense |
| 4–6 | 0.35–0.525 | 3,153 / 11 | 5,139 / 14 | 0.274 | 591 | dense |
| 6–8 | 0–0.1 | 39,314 / 28 | 56,249 / 35 | 0.194 | 6,090 | dense |
| 6–8 | 0.1–0.2 | 9,127 / 23 | 13,649 / 26 | 0.226 | 2,483 | dense |
| 6–8 | 0.20–0.35 | 15,704 / 21 | 16,778 / 22 | 0.164 | 1,504 | dense |
| 6–8 | 0.35–0.525 | 11,821 / 16 | 11,561 / 18 | 0.124 | 798 | dense |
| 8–10 | 0–0.1 | 27,677 / 23 | 48,051 / 30 | 0.153 | 3,830 | dense |
| 8–10 | 0.1–0.2 | 5,954 / 16 | 9,976 / 22 | 0.171 | 751 | dense |
| 8–10 | 0.20–0.35 | 3,099 / 17 | 5,334 / 17 | 0.141 | 227 | dense |
| 8–10 | 0.35–0.525 | 1,925 / 10 | 5,118 / 11 | 0.124 | 105 | dense |
| 10–12 | 0–0.1 | 24,662 / 18 | 29,328 / 18 | 0.147 | 2,750 | dense |
| 10–12 | 0.1–0.2 | 8,602 / 18 | 11,301 / 18 | 0.131 | 720 | dense |

Three additional cells have individual >0.1 rad/s errors but p95 at or below
0.1; these are rare tails rather than broad cell failures: 0–0.5 m/s / 0–0.1
rad (332/27,897 errors; p95 0.0012; 69,414 training rows / 49 runs),
3–4 m/s / 0–0.1 rad (505/10,753; p95 0.0946; 29,737 rows / 44 runs), and
10–12 m/s / 0.2–0.35 rad (80/1,614; p95 0.0993; 1,762 rows / 7 runs).
Only the last has a meaningful support-diversity gap; re-evaluate it after
the targeted refit rather than adding data now.

The exact post-v5 crawl cells justify one targeted transient test. Below
0.5 m/s at 0.20–0.35 rad, validation has 138 samples (p95 0.175 rad/s; 94
over 0.1) and training support 5,993 rows across 6 runs. At 0.35–0.525 rad,
there are 133 validation samples (p95 0.206; all 133 over 0.1) but only 2,954
training rows across 5 runs. The measured low-throttle calibration increased
row counts but mostly added steady-state samples in one run; it did not add
enough independent transient captures. The new suite below fills just these
two bands at calibrated 0.24/0.49 m/s and ±0.25/0.40/0.50 rad.

The multi-horizon target/index audit found no evident off-by-one or future
measurement leakage. Direct predictions use past/current sensor features and
candidate commands through the target interval; the GRU uses a causal history
and future command sequence, not future truth. Its current feature contract
still lacks explicit packet age, sideslip, and front-wheel speed; these remain
model hypotheses to test on existing data, not automatic additions.

## New fine-crawl capture: clean bag, wrong speed conditions

`openplane_yaw_error_crawl_fine_train_r01_retry2_20261008` completed all
1,152/1,152 phases. Its closed bag passed quality gates: 39.951–39.952 Hz
streams, gap p95 25.82–25.90 ms, maximum 55.58–56.68 ms, no active gap above
60 ms, zero collision changes, zero bridge timing faults, and 384/384 valid
probe phases. The fixed-25-ms whole-run training export contains 53,252
samples in 384 reset-isolated sequences; its per-probe audit is
`live_runs/racing_model_diagnostics_20261007/yaw_error_crawl_fine_train_r01_dataset/probe_support_audit.json`.

But its requested speeds were not achieved during the probes. Median measured
speeds were approximately:

| Requested speed | Measured probe speed, median across requested steering/event conditions |
|---:|---:|
| 0.35 m/s | 0.83 m/s |
| 0.45 m/s | 0.94 m/s |
| 0.65 m/s | 1.13 m/s |
| 0.90 m/s | 1.41 m/s |
| 1.20 m/s | 1.68 m/s |
| 1.45 m/s | 1.89 m/s |

Only 5.6% of plateau samples were within 0.15 m/s of the requested target.
Steering feedback did reach the requested angle: 99.3% of plateau samples
were within 0.05 rad. Odom speed closely matched simulator truth (median
absolute difference 0.011 m/s, maximum 0.055 m/s), so the discrepancy is not
an odometry-speed error. The probe throttle command/feedback median was about
0.081/0.080; rear-wheel/body-speed mismatch median was 0.523 m/s. Only 29
samples from 20 **unscored settle phases** had both speed below 0.5 m/s and
steering feedback at least 0.2 rad. This run is valid training transition
data at its measured states, but it does not fill the sub-0.5 m/s high-steer
gap and must not be described as doing so.

The same test-controller issue affected the earlier three `yaw_error_crawl`
captures: in training r01, the 0.60 m/s request produced about 1.09 m/s
median probe speed and the 1.00 request about 1.51 m/s. The implementation
explains the bias: `THROTTLE_FEEDFORWARD` has its first point at 2.5 m/s and
uses 0.10 throttle for every lower target; speed-hold gains were only
`kp=0.04`, `ki=0`, and the overspeed guard permits target +0.50 m/s. The
1,508-sequence throttle-surface dataset has 725,834 samples but only 21
throttle levels: 0%, 5%, …, 100%; it contains no 1–4% data needed to calibrate
this crawl controller. Thus the test-speed mismatch is an experimentally
confirmed control-design gap, not a simulator physics change or an odom
ground-truth mismatch.

The bounded real-sim gain check
`openplane_yaw_error_crawl_speedhold_kp100_r01_20261008` completed 96 probes
with `kp=1.0`, `ki=0.05`. It was cadence-clean: all six telemetry streams
were 39.952 Hz, gap p95 was 25.83–25.91 ms, maximum gap 53.18 ms, zero active
gaps exceeded 60 ms, zero collisions, zero bridge faults, and all 96 probes
reached the commanded steering plateau. Its median measured speeds improved
substantially over the old default-gain crawl runs, but were still above
target:

| Requested speed | Median measured speed across steering angles |
|---:|---:|
| 0.60 m/s | 0.673–0.706 m/s |
| 1.00 m/s | 1.067–1.098 m/s |

Across the 96 probes, only 43.7% of target-plateau samples were within
±0.15 m/s of the requested speed; the per-probe fraction ranged from 26.9%
to 100%, and 80/96 probes were below 50%. The run's `quality gates: PASS`
does **not** certify target-speed accuracy: the generated probe phases had
`validate_speed=false`, so the p50/p95 speed gates were not applied. The
same measure was only 4.7%, 5.8%, and 5.2% on the three old default-gain
crawl runs, so the gain change materially improved speed targeting without
making it reliable yet. The
unscored settle intervals did contain 185 samples below 0.5 m/s with
|steering|≥0.2 rad, but those are too brief to count as a proper transition
matrix. This confirms a test-controller limitation as well as the yaw-model
gap. Future crawl schedules now set `validate_speed=true`; the speed-hold
pilot itself remains data, not a validated low-speed capture.

## Finite follow-up test suite

1. **Low-throttle crawl calibration — completed and admitted as training.**
   The randomized 1/2/3/4/5% × 0/±0.20/±0.35/±0.50 rad, two-repeat test
   completed 70/70 reset-isolated conditions. Its final-two-second measured
   speed mapping and steady yaw response are documented above and archived in
   the clean fixed-timebase dataset. Do not repeat it.
2. **Sub-crawl steering-transition gap fill — training r04 completed and admitted.**
   The post-v5 audit reduced the true speed×steering support gap to below
   0.5 m/s at 0.20–0.525 rad; higher-speed cells are dense in that marginal
   projection. The targeted `yaw_error_subcrawl_steering_gapfill` schedule
   tests measured 0.24/0.49 m/s at 0.25/0.40/0.50 rad, both signs, onset,
   unwind and reversal, step versus 300 ms ramp, and 250/750 ms response ages.
   It has 144 reset-isolated probes / 432 phases and a conservative duration
   of 2,541.6 s. Training r01 was only a rejected preflight attempt: the
   runner started recording but rejected the 3,600 s timeout before issuing
   any excitation. Attempts r02 and r03 also issued no excitation: the
   launcher had omitted this new profile from its reset-enabled list, leaving
   zero subscribers on the required reset command. The r02 bag recorder was
   then terminated while an open SQLite bag was being inspected; active bags
   must not be read before rosbag closes. Both attempts are invalid and
   excluded. The launcher now enables the reset subscriber for this profile,
   and the harness reports each readiness gate. Training r04 completed 432/432
   phases (144/144 valid probes, zero invalid phases, zero collisions, zero
   bridge timing faults, not aborted). The closed-bag audit found all 144
   expected probes, target steering within 0.05 rad on every scored plateau,
   26–37 plateau samples per probe, and no collision/fault/abort. The fixed
   25 ms whole-run export passed packet, stream, and collision gates and
   contains 19,438 samples in 144 reset-isolated sequences. Requested 0.24 and
   0.49 m/s targets were actually measured at median 0.237–0.248 and
   0.492–0.497 m/s respectively across steering/event groups. The audit and
   export are `yaw_error_subcrawl_steering_train_r04_audit.json` and
   `yaw_error_subcrawl_steering_train_r04_dataset/` under
   `live_runs/racing_model_diagnostics_20261007/`. Its one detected packet-gap
   split is not crossed by any exported sequence. No simulator physics or
   runtime odometry/MPC behavior changed. Do not run the old 5.65-hour crawl
   rectangle; first evaluate the v6 refit with the admitted wheel-spin r03 and
   sub-crawl r04 captures. Only add a new capture if the held-out error and
   support audit still identify the same sparse regime.
3. **Missing 3–4 m/s high-steer reversal dynamics:** the implemented
   `yaw_error_highsteer_reversal_gapfill` matrix uses (3.0 m/s, 0.42/0.50 rad),
   (3.5 m/s, 0.42/0.50 rad), and (4.0 m/s, 0.35/0.42 rad), both signs,
   onset/unwind/reversal, 25/100/300 ms steering transitions, and 0.25/0.75 s
   delay. This is 216 probes per capture; two training and one sealed whole-run
   final-test capture are about 3.2 simulator-hours at the conservative reset
   budget. Training r01 completed with all 216 probes passing the requested
   speed gate, zero collisions/faults, and 207/216 probes meeting the
   ≥95%-within-0.05-rad steering-feedback criterion. Randomized training
   repeat r02 completed and passed: 216/216 probes, 648/648 phases, 216 reset
   epochs, zero collisions/faults, every probe within the speed gate, and all
   plateau steering samples within 0.05 rad. Its exported archive has 34,799
   samples in 216 reset-isolated sequences, 39.949–39.950 Hz sensor streams
   (p95 gaps 25.84–25.92 ms), zero packet-sequence gaps, and 99.997% packet
   matching. The v5 fit actually used 54 full training runs (four short
   captures were excluded from horizon fitting) and 34 independent validation
   runs; test/final-test arrays were not read. Paired run-level
   RMSE is worse for the current-only model at all six horizons; the history
   model is statistically unchanged only at 25 ms and worse at 100–1000 ms.
   Therefore do not collect high-steer r03 merely to add samples. The post-fit
   joint audit and low-wheel-slip capture decide whether any regime-specific
   follow-up is warranted.
   Score yaw, lateral acceleration, roll/rate, actuator response delay, and
   wheel/body mismatch separately. Keep 10–12 m/s steering within the existing
   empirical frontier; do not make the speed-angle grid rectangular by
   extrapolation.
4. **Targeted wheel-slip suite — matched-start controller diagnosis:** the
   baseline has sparse, severe 2–3.5 m/s wheel/body-mismatch and throttle-gap
   slices that the high-steer reversal and crawl-calibration runs do not
   excite. The existing `yaw_error_lowspeed_wheelspin_gapfill` schedule covers
   12 measured speed/steering points (0.75–3.5 m/s, 0.20–0.50 rad), both
   signs, +0.05/+0.10/+0.20 throttle steps versus ramps, reset-isolated
   approaches; its 288-phase budget is 3,081.6 s.

   Two starts exposed a test-controller calibration failure; neither bag is
   admitted to training. r01 aborted at 7/288 phases on the first 0.75 m/s
   probe: the generic 2.5 m/s feedforward floor supplied about 6.5% throttle,
   and the weak 0.04 speed gain could not bring the car back from 1.62 m/s to
   the matched start. r02 used Kp=1.0, Ki=0.05 and aborted on its first 2.5 m/s
   probe: the controller oscillated between 1.48 and 2.96 m/s, commanded up to
   50% throttle, and rear-wheel surface speed reached 10.72 m/s. Lateral/yaw
   motion and steering were zero; the failed criterion was speed stability.
   This is not evidence against the yaw model—it is a measured test-controller
   failure, and r02 demonstrates that an overly aggressive gain creates the
   wheelspin the experiment is meant to observe.

   The experiment controller now uses the empirical crawl relation from the
   completed 1–5% calibration (truth-speed medians 0.244/0.489/0.732/0.974/
   1.213 m/s) joined to the existing 2.5 m/s, 10% nominal anchor, only for this
   low-speed wheel-slip profile. This changes no simulator physics or runtime
   odometry/MPC behavior. Corrected training r03 completed 288/288 phases,
   144/144 paired probes, and passed the stream, reset, collision, and
   actuator-feedback audit. It was exported as one clean whole-run training
   sequence set (23,540 samples / 147 sequences; fixed 25 ms packet timebase;
   100% packet match). Start-speed pairs were closely matched (median
   difference 0, p95 0.0009 m/s, max 0.0041 m/s; steering start error max
   zero). The intended throttle changes produced actual post-stimulus
   wheel/body mismatch; speed rose substantially after some steps, so those
   observations must be modeled against measured state and not treated as
   steady target-speed points. This archive postdates v5 and was not included
   in v6; v7 must include it before deciding whether this interaction needs
   another capture.
5. **Conditional 2.5–8 m/s throttle-gap suite:** if the frozen GRU/refit still
   fails the sparse joint cells after admitting wheel-spin r03 and the
   sub-crawl capture, use the existing `yaw_error_midspeed_throttle_gapfill`
   schedule at 2.5/4.5/6.5/8.0 m/s and measured steering 0.15–0.42 rad,
   paired ±0.04/±0.08 step/ramp treatments. Its added 2.5 m/s, 0.15 rad point
   targets the sparse 2–3 m/s, 0.1–0.2 rad, 0.05–0.1 throttle-gap stratum
   (p95 0.513 rad/s; 984 training rows across 7 runs). Two train plus one
   validation capture are a bounded ~1.6-hour suite. Require actual feedback
   to enter the target throttle-gap bands; stop if the specific held-out
   joint strata resolve.
6. **No new generic matrix for dense cells.** Refit/compare causal models and
   diagnose the exact residual history/state for those cells. Do not retest
   the already-covered high-speed near-zero-steer throttle cut or extrapolate
   into unmeasured 10–12 m/s high steering.
7. **Rare 10–12 m/s, 0.2–0.35 rad tail:** this cell has 78/1,614 validation
   samples above 0.1 rad/s, but p95 is 0.0948 and training support is 1,762
   rows across seven runs. Treat it as a small run-diversity gap, not a broad
   operating-region failure. Re-score after the targeted refit first; only if
   the same tail repeats, add two independent, reset-isolated captures at
   measured 10–11 m/s and 0.20–0.30 rad, both signs, with steady and
   onset/unwind segments. Stay inside the empirically attained envelope and
   use a separate run for validation.

The conditional suites together are a finite upper bound, not a request to
run every possible combination. Re-evaluate support after each admitted
whole-run result and stop collecting for any regime whose held-out error is
resolved or whose failures remain dense and model-limited.

## Dynamic-state stratification of the v5 residual (2026-10-08)

The additional whole-run validation audit is
`live_runs/racing_model_diagnostics_20261007/yaw_multihorizon_teacher_v5_gap_refit_20261008/expanded_error_support_audit_v3_dynamic_state_20261008.json`.
It adds diagnostic-only bins for current absolute IMU lateral acceleration,
absolute roll, roll-rate, and simulator-truth body sideslip. Truth sideslip is
used only to explain residuals; it is not an allowed prediction input. No
test/final-test split was read.

At 250 ms, the pooled fraction above 0.1 rad/s and pooled RMSE increase with
lateral acceleration: `|ay|<1` gives 6.6% / 0.066 rad/s; `2–4` gives 37.9% /
0.178; `4–6` gives 18.4% / 0.119; and `8–12 m/s²` gives 29.7% / 0.107. The
non-monotonic bins show acceleration magnitude alone is not a sufficient
model. The association with sideslip is stronger: `|beta|<0.01` gives 7.3%
above threshold / 0.065 RMSE; `0.05–0.1` gives 29.8% / 0.163; `0.1–0.2` gives
55.1% / 0.219; and `0.2–0.5 rad` gives 65.0% / 0.228. These are pooled
associations, not causal effects; samples overlap within runs and adjacent
time windows.

The present teacher already receives causal IMU lateral acceleration, roll,
and roll-rate (plus rear wheel speeds, actuator feedback, and commands).
Therefore the audit does not justify merely adding those same channels. Roll
and acceleration are correlated with difficult slip states, but are not by
themselves an explanation that solves them. Sideslip is absent from the
teacher inputs; it is available here only as a truth label for diagnosis. A
future plant teacher could predict a latent/lateral-velocity state recursively,
but it must not read future truth or measured future sideslip during rollout.

For dynamic bins with at least 100 held-out rows and p95 above 0.1, the
speed×steering×lateral-acceleration audit has 92 failing bins (19 sparse by
`<500` training rows or `<8` runs; 59 dense); the equivalent roll-rate audit
has 107 (13 sparse, 73 dense), and sideslip has 63 (9 sparse, 38 dense).
Thus state-conditioned sparsity exists, but most failures remain despite
dense marginal support. In the high-sideslip bins the explicit sparse
failures are confined to sub-crawl (<0.5 m/s) conditions: three held-out bins
have 115–319 samples, p95 0.117–0.217 rad/s, and only 84–544 training rows
across 3–7 runs. This reinforces a narrow sub-crawl/high-slip data gap while
rejecting a broad additional speed×steering sweep.

The worst current-only 250 ms sample remains a fast 3.47 m/s high-steer yaw
reversal: error 1.648 rad/s despite near-zero steering/throttle feedback
gaps, only 0.056 m/s rear-wheel/body mismatch, `ay=-4.72 m/s²`, roll
`-0.0457 rad`, and roll-rate `0.039 rad/s`. Other high-error samples include
large wheel/body mismatch and lateral acceleration. The different contexts
mean no single measured roll or acceleration threshold explains the tail.

## Current targeted captures and next gating

The bounded paired throttle-gap capture has two clean training runs and one
independent validation run, all complete. For r01, all 112/112 intended
throttle changes followed feedback with the correct sign and approximately
1.0 gain; 56 step/ramp pairs began at matched speed (median difference
0.00006 m/s; maximum 0.00303 m/s). Its export has 22,168 samples in 120
reset/gap-isolated sequences. For r02, all 112/112 changes also followed with
the correct sign (feedback gain p10/p50/p90 1.0000/1.0000/1.0000; maximum
delta residual 2.15e-8). Its 56 step/ramp pairs began within 0.0027 m/s at
p95 and 0.0081 m/s maximum. The clean export has 21,855 samples in 129
sequences, exact 25 ms timebase, 99.996% packet match, and no quality
failures. Both runs completed all 224 phases at 39.83–39.84 Hz with no
collisions, bridge faults, or aborts. The proper response metric is the
measured trajectory after each command, since speed is expected to change
during the stimulus. Independent validation r03 also completed 224/224
phases and 112/112 probes, at 39.83 Hz, with zero collision/fault/abort; its
probe audit confirms all throttle changes followed feedback with correct sign
and unity gain. After adjusting each probe by its own initial mismatch, the
positive 0.08 step-minus-ramp mismatch-change median is +0.069 m/s in r03,
positive in 13/14 pairs; body speed after the stimulus is 0.093 m/s lower
with the step. For +0.04, the mismatch-change median is only +0.009 m/s and
step is larger in 8/14 pairs. This independently reproduces the nonlinear
throttle-slew effect in the direction expected from wheelspin. r03 is
exported separately as validation (22,083 samples / 126 sequences, 25 ms
timebase, 99.9956% packet match) and has not entered fitting.

Paired-response analysis of the two training audits finds an effect from
positive throttle slew, but also a matching limitation that must be retained
in the interpretation. Starts were exceptionally close in body speed
(r01/r02 p95 pair difference 0.0014/0.0026 m/s), steering, and throttle;
initial wheel/body mismatch was less consistently matched (median pair
difference 0.021/0.034 m/s; p95 2.075/0.870 m/s, with outliers). Therefore
I compare the change in plateau median mismatch from each probe's own
pre-stimulus median, not raw
post-stimulus mismatch alone. Across 14 speed/steering/signed-turn pairs for
a positive 0.08 normalized-throttle change (8 percentage points), the
step-minus-ramp difference in that within-probe mismatch change has median
+0.095 m/s in r01 (14/14 positive) and +0.191 m/s in r02 (12/14 positive).
The paired body-speed median after the stimulus is lower with a step by
0.070/0.081 m/s. For +0.04, mismatch-change differences are smaller (+0.015 /
+0.039 m/s median); throttle cuts have near-zero median differences and
mixed signs. Thus the strongest supported result is narrower than the raw
p90 contrast: a rapid +0.08 step usually adds more wheel/body mismatch than
a ramp, and leaves lower body speed, but the effect is state- and
turn-dependent. These plateau-window comparisons are not a complete
transient response curve; r03 confirms the positive-step effect direction,
but this evidence alone does not justify a runtime change.

Two prior low-speed wheelspin pilots do not constitute full coverage: r01
completed 7/288 phases with 3 valid probes, r02 completed 1/288 with none;
both aborted on reset-state recovery, with zero collisions or timing faults.
The subsequent r03 completed all 288 phases and 144 probes. Preserve the
partial bags but exclude them as complete-run training captures; do not rerun
the whole wheelspin matrix unless the post-v6 exact-cell audit still shows a
gap. The completed r03 wheelspin archive and r04 sub-crawl archive were added
to v6. The strict held-out audit below shows no global one-step or multistep
improvement; the wheelspin validation cell regressed slightly. These captures
remain valid evidence, but do not justify claiming a teacher improvement.

The wheelspin r03 is not merely a generic low-speed repeat: its audit has
nine probes near 3.55–3.94 m/s at 0.20 rad steering, with rear-wheel/body
speed mismatch p90 above 0.5 m/s in four probes and above 1 m/s in three
(maximum 3.95 m/s). That directly exercises the previously thin 3–4 m/s,
moderate-steering, high-slip/high-lateral-acceleration interaction. This
should increase relevant training support in v6; the measured held-out
one-step and multistep results below show no global improvement.

## 40 Hz odometry relevance and latest held-out audit (2026-10-08)

For odometry, the relevant teacher is the 25 ms / one-step model. The v5
`current` variant uses only the current encoder, steering/throttle feedback,
IMU acceleration/yaw-rate/attitude, and command already issued; the separate
`history_1p6s` variant adds past observations. Simulator truth yaw-rate is
only the target. No future sensor values, simulator pose, or truth-derived
speed/sideslip are model inputs. The longer-horizon direct
forecasts are different tasks: their training examples include the recorded
command sequence throughout the forecast window. Their errors should not be
used as the per-tick odometry score.

The strict v5/v6 audits froze training sources to each fit report, scored all
34 fit-validation runs plus the new independent paired-throttle validation
run r03, and did not read test/final-test arrays. At 25 ms, v5 scores
323,635 held-out samples: RMSE 0.05992 rad/s, p95 0.12258 rad/s, and 93.40%
within 0.1 rad/s. Carrying the current gyro rate forward gives RMSE 0.09630,
p95 0.20350, and 90.19% within 0.1. V5 beats that baseline on all 35 held-out
runs; the paired run-macro RMSE improvement is 0.0321 rad/s (95% bootstrap
interval 0.0267–0.0375). V5 is a real improvement in one-step rate
prediction, but not a near-zero-error model. V6, after adding the
completed wheelspin and sub-crawl training captures, is worse: RMSE 0.06296,
p95 0.12684, and 92.76% within 0.1. Its 250 ms RMSE is 0.1049 rad/s versus
v5's 0.0973, and 17.29% of v6 samples exceed 0.1 rad/s at that horizon. More
rows alone did not buy accuracy.

Using truth speed/absolute steering only as offline diagnostic labels, and
requiring at least 100 validation samples and p95 above 0.1 rad/s, the
one-step speed×steering grid has 22 failing cells for v5 (16 dense by
≥1,000 training rows and ≥10 runs; six lack that support) and 24 for v6
(18 dense, five intermediate, one sparse). The 25 ms failures are not all
data gaps. Joint conditioning on steering feedback gap, throttle gap,
wheel/body-speed mismatch, lateral acceleration, roll rate, and sideslip
reveals more failures, but those tables overlap and are diagnostic bins, not
independent model scores. The worst measured marginal failures include
3–4 m/s at 0.1–0.2 rad (v5 p95 0.234 rad/s, 21 training runs) and
4–6 m/s at 0.1–0.2 rad (p95 0.223, 29 training runs). A few sub-crawl/high-
steering cells remain run-diversity limited; the rest should first be tested
with one-step regime specialists, not another generic sweep.

An additional causal one-step replay numerically integrated predicted yaw
rate over contiguous held-out sequences without IMU-orientation correction.
Across 1,373 reset/gap-isolated blocks, the model's p95 absolute terminal
heading drift was 3.96° (maximum 20.75°); carrying the measured gyro rate
forward was 2.00° p95 (maximum 4.83°). On two continuous 260 s validation
captures, model peak drift was 22.10° and 49.06°, compared with 2.97° and
4.14° for gyro persistence. The model improves instantaneous RMSE but has
temporally correlated residuals and is not safe to substitute as a standalone
gyro integrator.

That replay is not the production odometry path. The current node integrates
measured IMU gyro rate and, when the IMU quaternion step is within its 0.30 rad
gate, applies a 1.0-gain quaternion correction at each packet; large quaternion
jumps take the gyro-only fallback. Therefore these yaw-teacher metrics do not
prove an odometry pose improvement. Do not replace the production gyro or
integrate this teacher yet. The useful next experiment is a one-step,
sensor-only regime-specialist comparison on the frozen whole-run train/validation
split, followed by replay through the actual quaternion-correction policy and
run-level heading/position error. Route experts only with runtime-available
signals; truth speed/steering bins may diagnose and score offline but must not
be used as an online gate. Collect new data only if that comparison identifies
a specific repeatedly failing regime with insufficient independent-run
support. No simulator was launched for this audit, and no runtime odom, MPC,
or physics code was changed.

### Complete threshold census of the frozen v2 atlas (2026-10-08)

The row-level held-out census is saved at
[`yaw_large_error_audit_v1/sensor_yaw_large_error_audit.json`](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/sensor_yaw_large_error_audit.json)
and its full 5,871-row `|error| > 0.1 rad/s` list is
[`selected_atlas_errors_over_0p1.csv`](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/selected_atlas_errors_over_0p1.csv).
It covers 299,242 one-step transitions from 36 whole-run validation captures;
test/final-test data were not read for fitting or scoring. The model target is
next-tick simulator-GT yaw rate; no production odometry target or GT-derived
feature/selector was used. GT speed, current GT yaw, and wheel-minus-GT speed
are post-fit diagnostics only.

The census has 5,871/299,242 errors above 0.1 rad/s (1.962%), RMSE 0.03694,
p95 0.04012, and maximum 1.35362 rad/s. The event breakdown is hold
596/232,199 (0.26%); reversal 754/2,722 (27.70%); turn-in 1,300/35,276
(3.69%); unwind 3,221/29,045 (11.09%). Unwind contributes the largest number
of outliers, but reversal has the highest per-sample failure rate.

Local experts cover 89.08% of rows. Of large errors, 2,973 use a local expert
(1.12% of locally covered rows) and 2,898 use the global event fallback
(8.87% of fallback rows). Fallback error rate is 31.21% for reversal and
12.48% for unwind; local-expert error rate is 15.06% for reversal and 9.61%
for unwind. Unsupported regimes therefore amplify the tail, especially
reversal, but are not the only issue: unwind/turn-in errors also happen inside
trained local experts, and supported reversal remains poor. Of the local
expert outliers, 780 have fewer than 120 training rows and 135 come from
experts with only two independent training runs.

Among outliers, 51.5% have a steering command/feedback gap above 0.05 rad;
this occurs in 78.5% of reversal and 55.6% of unwind outliers. Only 3.8% have
current IMU yaw differing from same-tick GT by >0.1 rad/s, 8.3% have
wheel-vs-GT speed mismatch >1 m/s, and 13.2% have throttle
command/feedback mismatch >0.05. These signatures overlap and do not prove
causality. There are 1,983 outliers that cross none of the five tested
thresholds; stale IMU, wheelspin, or actuator mismatch alone cannot explain
the full tail. Errors occur in every measured speed band; 4–6 m/s has the
largest band rate in this split at 3.46%, not an exclusive failing region.

The largest coherent groups remain steering-release yaw collapse and
command reversal ahead of measured feedback. The smaller packet-phase class
is real but not the dominant explanation across all validation rows. This is
one-step yaw-rate error, not recursive heading drift, full odometry, or
full-lap MPC accuracy. Do not feed GT or `/bridge_packet_timing` to a runtime
selector. The two-packet law remains a local comparator: its exact-mode gain
does not transfer to the indistinguishable three-packet branch, and blind
whole-cell use regressed.

Follow-up analysis separated support-gap repair from remaining transient
error. A train-only neighborhood candidate reduced the whole-run held-out
count above 0.1 from 5,871 to 3,814, mainly by reducing unwind errors
(3,221 to 1,208); reversal changed only 754 to 710. The r03 timing join found
higher unwind and reversal errors in 3-packet than matched 2-packet
transitions, but many 2-packet errors remain. The available legal sensor
history does not yet distinguish the timing branch before steering feedback
moves, and the large candidate artifact is not production-ready. Full metrics,
matched conditions, and the next deterministic-phase investigation are in
[`YAW_RESPONSE_REGIME_MAP_20261008.md`](YAW_RESPONSE_REGIME_MAP_20261008.md).
