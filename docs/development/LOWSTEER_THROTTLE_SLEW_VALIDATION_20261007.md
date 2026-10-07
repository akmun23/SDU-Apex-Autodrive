# Low-steering throttle-slew validation — 2026-10-07

## Why this test exists

The existing randomized throttle-slew study covered 4.5/6.5/7.5 m/s at
0.12/0.20 rad and higher-steering frontier cells. Practice telemetry showed
that the fast 9g line spends much of its 4.5–7.5 m/s range below 0.12 rad.
This validation fills that specific gap at 0.08/0.10 rad; it is not another
broad throttle grid.

Each capture has 24 paired conditions: 3 speeds × 2 steering magnitudes × 2
turn directions × throttle increase/decrease. Ramp-versus-step order and
condition order are seeded and randomized. Each individual probe begins after
a simulator reset to spawn and a fresh speed approach, preventing lateral or
yaw motion from one swerve from contaminating the other member of its pair.
The capture records the existing sensor/command set and retains the 40 Hz
packet-sequence timebase. Simulator state is used offline for scoring only.

## Captures and quality

Three independent captures passed every quality gate:

| Run | Complete pairs | Resets | `/odom` receive rate | Collision delta | Timing faults |
| --- | ---: | ---: | ---: | ---: | ---: |
| r02 | 24/24 | 48/48 | 39.961 Hz | 0 | 0 |
| r03 | 24/24 | 48/48 | 39.961 Hz | 0 | 0 |
| r04 | 24/24 | 48/48 | 39.963 Hz | 0 | 0 |

All runs completed 96/96 phases, had no invalid phases, and passed the
analyzer's matched-start, command/feedback, packet-continuity, reset-position,
and 40 Hz gates.

The first pilot, r01, is retained but excluded from validation: it completed
five valid pairs, then correctly aborted when a 6.5 m/s, 0.10-rad swerve left
large lateral velocity/yaw that did not recover within the four-second
matched-state window. During the first half-second of that wait, median
absolute lateral body speed was about 5.0 m/s and median absolute yaw rate
about 1.39 rad/s. The schedule was then improved to reset and re-approach
before each probe. The pilot's five pairs and abort diagnosis are in
`live_runs/racing_model_diagnostics_20261007/lowsteer_validation_r01_partial.json`;
the incomplete capture is not included in the replicated estimates.

## Measured result

The frozen primary outcome is the paired step-minus-ramp change in mean
absolute rear-wheel/body-speed mismatch over 0.10–1.00 s. Positive means the
rapid step had more mismatch than the gradual ramp. Across 72 valid paired
conditions, all 72 were positive. The mean was **+0.3336 m/s**. Equal-weight
run effects were r02 **+0.3340**, r03 **+0.3344**, and r04 **+0.3324 m/s**;
the three-run cluster-bootstrap interval was **[+0.3324, +0.3344] m/s**.
With only three independent runs, treat this as strong repeatability in these
captured conditions, not a universal-population guarantee.

Effects remained positive in every tested speed, steering, turn, and
throttle-direction group:

| Condition | Mean step-minus-ramp mismatch change |
| --- | ---: |
| 4.5 m/s | +0.2047 m/s |
| 6.5 m/s | +0.4129 m/s |
| 7.5 m/s | +0.3831 m/s |
| 0.08 rad | +0.2893 m/s |
| 0.10 rad | +0.3779 m/s |
| Throttle increase | +0.5669 m/s |
| Throttle reduction | +0.1003 m/s |
| Left / right turn | +0.3367 / +0.3305 m/s |

The wheel-speed longitudinal-slip proxy also favored the ramp in all 72 pairs
(mean step-minus-ramp change **+0.0547**). This is an encoder/kinematic proxy,
not a direct tire-force or contact-patch slip measurement.

There is a real acceleration tradeoff: rapid steps produced **+0.671 m/s²**
more longitudinal acceleration change on average than ramps in this short
window (**+1.139 m/s²** for throttle increases; **+0.203 m/s²** for reductions).
The acceleration difference grew with speed. A throttle-rise limiter can
reduce the measured wheel/body mismatch, but it is not a free lap-time gain;
practice transfer must decide whether that trade is worthwhile.

The combined machine-readable analysis is
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/lowsteer_validation_r02_r04_pair_analysis.json`.
Individual quality reports and bags remain beside their run IDs under
`live_runs/`.

## Existing high-steering paired evidence and remaining gap

The earlier eight-run validation already covers the high-steering paired
step/ramp question; this grid must not be repeated. Its 256 valid pairs include
all turn directions and throttle increase/decrease at 4.5, 6.5, and 7.5 m/s
with steering magnitudes 0.30 and 0.42 rad. Each high-steering speed/angle cell
has 32 pairs total across eight independent runs; the positive-throttle
increase subset has 16 pairs per cell, with one outcome in each turn direction
per run.

For positive throttle increases, the step-minus-ramp mean absolute rear-wheel/
body-speed residual over 0.10–1.00 s is **+0.2363 m/s** overall (run-cluster
bootstrap 95% interval **[+0.2207, +0.2637]**); all eight run-level effects are
positive. The corresponding encoder-derived slip-ratio proxy difference is
**+0.0378** (95% interval **[+0.0360, +0.0409]**), also positive in all eight
runs. These are kinematic proxies, not direct tire-force measurements.

The same paired high-steering cells show an average step-minus-ramp body
longitudinal acceleration difference of **−0.0434 m/s²** (95% interval
**[−0.0530, −0.0344]**). Thus, in these tested transients, the gradual ramp
reduced wheel/body mismatch without sacrificing average body acceleration;
this is consistent with less torque going into wheelspin, but does not by
itself establish a faster lap or causal tire-force mechanism. The effect is
not uniform enough to extrapolate beyond the tested cells.

| Speed | Steering | Step-minus-ramp wheel/body residual | Run-bootstrap 95% interval |
| ---: | ---: | ---: | ---: |
| 4.5 m/s | 0.30 rad | +0.2501 m/s | [+0.2433, +0.2579] |
| 4.5 m/s | 0.42 rad | +0.2559 m/s | [+0.2514, +0.2616] |
| 6.5 m/s | 0.30 rad | +0.2130 m/s | [+0.2035, +0.2249] |
| 6.5 m/s | 0.42 rad | +0.2209 m/s | [+0.2181, +0.2239] |
| 7.5 m/s | 0.30 rad | +0.2697 m/s | [+0.1846, +0.4275] |
| 7.5 m/s | 0.42 rad | +0.2083 m/s | [+0.1996, +0.2183] |

The adjacent high-speed tests do not fill the paired positive-throttle rate
comparison above 7.5 m/s: their 9.5–11.1 m/s steering magnitudes are only
0.06–0.20 rad, and their 9.5 m/s paired throttle probes are reductions. That
gap has now been tested directly at 8 m/s and high steering as described
below.

## High-steer throttle-rate sweep at 8 m/s

Three independent validation runs compared a +0.08 normalized-throttle change
from the measured 8 m/s feedforward baseline (0.325 to 0.405) using an
instantaneous step and 0.15/0.30/0.60 s ramps. Each ramp duration had a
separately reset-matched step in both turn directions at 0.30 and 0.42 rad.
All 36 pairs passed matching; actual stimulus speeds were 7.903–7.944 m/s,
with measured steering feedback at 0.30 or 0.4199 rad. All three runs passed
40 Hz, encoder matching, reset, and capture-quality gates: 12/12 pairs and
24/24 reset recoveries per run, zero collision/timing faults, and command
streams at 39.84 Hz.

The table reports step-minus-ramp changes over 0.10–1.00 s, averaged first
within run and then over the three independent runs. Positive wheel-residual
and slip-proxy changes mean the ramp reduced the measured transient mismatch;
negative acceleration change means the ramp yielded greater body acceleration.

| Ramp duration | Rise rate | Wheel/body residual change (m/s) | Slip-ratio proxy change | Body acceleration change (m/s²) |
| ---: | ---: | ---: | ---: | ---: |
| 0.15 s | 0.533/s | +0.1181 [0.1127, 0.1266] | +0.0146 [0.0140, 0.0155] | +0.0390 [−0.0161, +0.0690] |
| 0.30 s | 0.267/s | +0.1995 [0.1951, 0.2069] | +0.0242 [0.0238, 0.0250] | −0.0106 [−0.0172, −0.0058] |
| 0.60 s | 0.133/s | +0.2485 [0.2349, 0.2695] | +0.0292 [0.0276, 0.0319] | −0.0661 [−0.0834, −0.0380] |

Brackets are 95% run-cluster bootstrap intervals (3 independent runs). Across
the measured cells, slower ramps monotonically reduced the wheel/body mismatch
and slip proxy. The body-acceleration result is steering-dependent, however:

| Ramp duration | Steering | Step-minus-ramp body acceleration (m/s²) |
| ---: | ---: | ---: |
| 0.15 s | 0.30 rad | +0.0287 [−0.0806, +0.1075] |
| 0.15 s | 0.42 rad | +0.0492 [+0.0203, +0.0788] |
| 0.30 s | 0.30 rad | +0.0294 [+0.0043, +0.0524] |
| 0.30 s | 0.42 rad | −0.0506 [−0.0868, −0.0220] |
| 0.60 s | 0.30 rad | −0.0969 [−0.1480, −0.0427] |
| 0.60 s | 0.42 rad | −0.0352 [−0.0536, −0.0188] |

Positive means the step accelerated the body more; negative means the ramp did.
At the same 0.30 s ramp, the acceleration effect reverses between 0.30 and
0.42 rad even though the wheel/body mismatch is reduced by the ramp in both
cells. Therefore a single global throttle slew rule would hide a measured
steering interaction. These are short-horizon open-plane comparisons, not a
lap-time improvement or direct tire-force measurements; they provide
angle-conditioned targets for a gated plant/controller candidate.

The closed-bag analysis is
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/highsteer_rate_sweep/all3_paired_analysis.json`;
the three raw bags and per-run analyses remain in `live_runs/`.

## Throttle increment and rise-rate follow-up

Three new independent 8 m/s runs repeated the full steering/rate matrix with a
larger +0.12 throttle change (0.325 to 0.445). All 36 matched pairs passed,
all 72 reset recoveries passed, every run passed the capture and 40 Hz gates,
and no speed-governor intervention occurred; the highest observed speed in
the first run was 10.71 m/s. The table is step-minus-ramp, averaged across
0.30/0.42 rad and both turn directions; intervals resample whole runs, not
40 Hz samples.

| Ramp duration | Actual rise rate | Wheel/body residual change (m/s) | Slip-ratio proxy change | Body acceleration change (m/s²) |
| ---: | ---: | ---: | ---: | ---: |
| 0.15 s | 0.800/s | +0.3260 [0.3208, 0.3297] | +0.0403 [0.0396, 0.0408] | +0.0085 [−0.0105, +0.0402] |
| 0.30 s | 0.400/s | +0.6169 [0.6101, 0.6226] | +0.0746 [0.0738, 0.0752] | −0.0644 [−0.0796, −0.0565] |
| 0.60 s | 0.200/s | +0.8392 [0.8308, 0.8471] | +0.0989 [0.0978, 0.0999] | −0.0342 [−0.0461, −0.0279] |

The wheel residual and slip-proxy benefits of a ramp replicate at both steering
magnitudes; for +0.12 their effect sizes are similar at 0.30 and 0.42 rad.
At +0.08, the acceleration effect at a 0.30 s ramp reversed sign between
0.30 and 0.42 rad. Thus steering remains necessary for explaining the body
response even where the wheel-speed proxy moves consistently.

The +0.16 increment (0.325 to 0.485) exposed a test-design boundary. Two
12-pair exploratory runs each yielded only 4 fully matched pairs: the 11.2 m/s
safety governor changed throttle during many ramps, especially at 0.30 rad and
the 0.60 s ramp. Those pairs fail the command/feedback profile gate and are
excluded from effect estimates; the new analyzer now records governor ticks,
peak phase speed, invalid condition IDs, and explicit capture-quality reasons.
No collisions or bridge timing faults occurred. A targeted third run restricted
to 0.42 rad and 0.15/0.30 s ramps passed all 4/4 pairs, 8/8 resets, and 40 Hz
checks. These are the governor-clean +0.16 cells; do not infer the rejected
cells from their raw bag signals.

There was a second design issue: comparing +0.08 and +0.12 at the same ramp
durations changes both total throttle increment and rise rate. That shows the
combined actuator effect but does not identify increment-size nonlinearity by
itself. A randomized factorial profile crossed +0.08/+0.12 increments with
matched 0.133/0.267/0.533 normalized-throttle/s rise rates at 8 m/s, both
0.30/0.42 rad steering magnitudes, and both turn directions. Each ramp has a
reset-matched step to its own final command. The three captures completed
without collisions, timing faults, speed-governor interventions, or reset
failures; each received the measured sensor streams at about 39.96 Hz.

The audit exposed one important missing match variable. The old pair gate
matched body speed, lateral speed, yaw rate, steering, and throttle, but did
not match the initial rear-wheel/body-speed residual. In factorial r02, the
0.42-rad left-turn +0.08 pair started with a 0.926 m/s wheel/body residual in
the step phase and near zero in the ramp phase. Its other start-state fields
matched, but it was not a valid throttle comparison. The analyzer now also
compares each rear wheel's pre-stimulus residual, reconstructed from the
common and left/right-asymmetry terms, with a 0.20 m/s per-wheel tolerance.
This tolerance is above the 95th-percentile natural pair difference
(0.079 m/s) in these captures and rejected only that badly unmatched pair;
the other two factorial exclusions are a steering-waveform mismatch in r01
and this residual mismatch in r02. Result: **70/72 usable factorial pairs**
(r01 23/24, r02 23/24, r03 24/24). Invalid pairs are excluded from every
effect estimate and are retained with explicit reasons in the JSON report.

For the crossed factorial subset, the 0.10–1.00 s step-minus-ramp means,
averaged over steering and turn, show a clear throttle-increment × rate
interaction. Positive residual/slip values mean the gradual ramp reduced the
encoder-derived wheel/body mismatch; acceleration is step minus ramp.

| Increment | Rise rate (/s) | Residual change (m/s), 95% run-bootstrap CI | Slip-ratio proxy change, 95% CI | Body acceleration change (m/s²), 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 0.08 | 0.133 | +0.239 [+0.230, +0.248] | +0.0281 [+0.0270, +0.0293] | −0.034 [−0.096, +0.018] |
| 0.08 | 0.267 | +0.196 [+0.193, +0.199] | +0.0236 [+0.0231, +0.0239] | −0.064 [−0.081, −0.036] |
| 0.08 | 0.533 | +0.122 [+0.108, +0.130] | +0.0151 [+0.0134, +0.0162] | +0.024 [−0.037, +0.058] |
| 0.12 | 0.133 | +0.959 [+0.950, +0.967] | +0.1107 [+0.1097, +0.1116] | +0.276 [+0.267, +0.285] |
| 0.12 | 0.267 | +0.769 [+0.761, +0.774] | +0.0917 [+0.0906, +0.0923] | −0.076 [−0.082, −0.067] |
| 0.12 | 0.533 | +0.489 [+0.483, +0.499] | +0.0599 [+0.0591, +0.0612] | −0.016 [−0.055, +0.004] |

These rows are the 8 m/s factorial profile only, with three run clusters per
cell (the two invalid pairs are excluded). The larger +0.12 increment causes
substantially more measured wheel/body mismatch at every shared rate. The
acceleration effect changes sign across rates and increments, so a single
global rate cap is not justified. This is a local, empirically measured
response surface—not yet a general tire-force model.

The full crossed matrix now covers four matched initial speeds, with three
independent randomized captures at each speed. Each run has 24 ramp/step
pairs, both 0.30/0.42 rad steering magnitudes and turn directions, +0.08/+0.12
throttle increments, and 0.133/0.267/0.533 normalized-throttle/s ramp rates.
Across the 12 runs, all 576 reset recoveries passed, all sensor streams passed
the 40 Hz receive gate, and there were no collisions, bridge timing faults,
or speed-governor interventions. The 4.5, 6.5, and 7.5 m/s strata each have
72/72 valid pairs. The 8.0 m/s stratum has 70/72: one initial steering-waveform
mismatch and one initial wheel/body residual mismatch are excluded. Total:
**286/288 usable pairs**.

The speed labels refer to the matched speed at the throttle stimulus, not a
speed held constant through the response. The observed stimulus and transient
peak ranges were:

| Initial speed | Matched stimulus speed (m/s) | Peak transient speed (m/s) | Valid pairs |
| ---: | ---: | ---: | ---: |
| 4.5 | 4.406–4.431 | 6.344–7.337 | 72/72 |
| 6.5 | 6.589–6.626 | 8.478–9.447 | 72/72 |
| 7.5 | 7.545–7.586 | 9.412–10.367 | 72/72 |
| 8.0 | 7.903–7.944 | 9.760–10.708 | 70/72 |

The table is the paired step-minus-ramp change in mean absolute rear-wheel /
body-speed residual over 0.10–1.00 s. Positive means the ramp reduced the
measured mismatch. Each interval bootstraps the three run-level effects, not
the 40 Hz samples.

| Increment | Rise rate (/s) | 4.5 m/s | 6.5 m/s | 7.5 m/s | 8.0 m/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.08 | 0.133 | +0.304 [+0.302, +0.306] | +0.271 [+0.264, +0.276] | +0.248 [+0.244, +0.251] | +0.239 [+0.230, +0.248] |
| 0.08 | 0.267 | +0.250 [+0.248, +0.251] | +0.215 [+0.211, +0.220] | +0.202 [+0.198, +0.209] | +0.196 [+0.193, +0.199] |
| 0.08 | 0.533 | +0.161 [+0.158, +0.163] | +0.134 [+0.132, +0.135] | +0.118 [+0.114, +0.124] | +0.122 [+0.108, +0.130] |
| 0.12 | 0.133 | +0.897 [+0.893, +0.900] | +0.941 [+0.935, +0.949] | +0.962 [+0.957, +0.968] | +0.959 [+0.950, +0.967] |
| 0.12 | 0.267 | +0.775 [+0.770, +0.782] | +0.781 [+0.778, +0.787] | +0.768 [+0.763, +0.774] | +0.770 [+0.761, +0.774] |
| 0.12 | 0.533 | +0.439 [+0.434, +0.445] | +0.489 [+0.485, +0.496] | +0.500 [+0.498, +0.503] | +0.489 [+0.483, +0.499] |

This is already a useful regime-conditioned calibration table, not a single
global coefficient. Across all four speeds, the larger +0.12 increment
produced about 3–4 times the step/ramp wheel-mismatch contrast of +0.08 at
the same rate. Faster ramps consistently reduced the ramp's mismatch
advantage. The speed dependence is not uniform: for +0.08 the contrast
decreases with speed, while for +0.12 it rises from 4.5 to 7.5 m/s and then
levels off. That is a real speed × increment interaction in the measured
regime.

Acceleration makes the choice non-monotonic. At +0.12 and 0.133/s, the
rapid step produced 0.19–0.30 m/s² more body acceleration than the ramp. At
0.267/s the sign reversed at every speed: the ramp produced about
0.064–0.082 m/s² more body acceleration than the step while still reducing
the wheel/body mismatch by about 0.77–0.78 m/s. Thus 0.267/s is the strongest
candidate rate to carry into a high-steering practice transfer; it is not
yet a validated lap-time improvement or a runtime change. The 0.133/s ramp
still minimizes mismatch more, at a measurable acceleration cost.

### Does measured roll explain the paired mismatch?

As an offline diagnostic, a ridge model was fit to the step-minus-ramp wheel
residual contrast using the test condition (initial speed, throttle increment
and rise rate, steering magnitude, turn direction, and their predeclared
interactions: squared increment/rate/steering, increment×rate, increment×steer,
rate×steer, speed×increment/rate/steer, and turn×increment/steer). The ridge
penalty was fixed at 1.0. A second model added the paired post-transient
change in absolute IMU roll and absolute roll rate. Standardization was fit
on the training captures only; evaluation held out one entire capture at a time.
This used 285 of 286 valid factorial pairs with both roll features (one
alignment-missing sample is described below). Standardization and ridge
coefficients were fit inside each fold; one entire capture was held out in
each of 12 folds. Adding measured roll **worsened** equal-capture mean RMSE
for wheel residual from 0.02114 to 0.02196 m/s (+3.9%), and worsened the
slip-ratio proxy from 0.00444 to 0.00449 (+1.2%). Individual folds were
mixed, including all three 4.5 m/s residual folds getting worse. After
removing condition-cell means, the residual contrast correlations were only
0.18 for roll change and −0.11 for roll-rate change. The expanded, held-out
evidence therefore does not support roll as a useful predictor here. The
features are measured after the throttle transient, so they cannot be used
as future inputs in a recursive plant rollout. Do not add roll to the plant,
observer, odometry, or MPC based on this result.

The only missing roll measurement was one IMU attitude sample in the 6.5 m/s
r02 step phase: the latest causal IMU packet was 45.8 ms old, beyond the
30 ms alignment limit. That pair remains valid for the wheel/actuator
analysis; only its roll-feature comparison is omitted.

The analyzer emits a 96-row CSV grouped by initial speed, increment, actual
rise rate, steering magnitude, and turn, with per-run means and run-to-run
spread. It contains the step-minus-ramp contrasts and each profile's separate
pre-relative wheel-residual, slip-proxy, body-acceleration, and roll response.
This is a direct lookup/calibration input for offline controller and
optimizer work; the 40 Hz bags remain source data, not the deliverable by
themselves.
Practice transfer remains the gate before runtime integration: the existing
0.267/s low-steering overlay reduced the high-mismatch tail slightly but did
not improve lap time beyond baseline variability, so this new high-steering
surface has not been inserted into MPC or odometry.

The combined, re-gated analysis is
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/highsteer_rate_sweep/all_slew_increment_rate_surface_20261007.json`;
its compact surface is
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/highsteer_rate_sweep/throttle_slew_empirical_surface_20261007.csv`.
The factorial-only report is
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/highsteer_rate_sweep/factorial_all3_paired_analysis.json`.
The four-speed factorial report and its 96-row empirical surface are
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/highsteer_rate_sweep/factorial_all4_speeds_paired_analysis.json`
and
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/highsteer_rate_sweep/factorial_all4_speeds_empirical_surface_20261007.csv`.
The matching 4.5 and 7.5 m/s three-capture reports and speed-specific surfaces
are stored beside those outputs. The four-speed report keeps all strata and
run clusters separate; no cross-speed pooling is used for the confidence
intervals.
The 6.5 m/s factorial report and its 24-row detailed surface are
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/highsteer_rate_sweep/factorial_6p5_all3_paired_analysis.json`
and
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/highsteer_rate_sweep/factorial_6p5_empirical_surface_20261007.csv`.
The combined rate-sweep summary keeps speed and capture profile separate;
there is no cross-speed or cross-profile pooling in its rate-factorial table.
Per-run reports and raw bags remain beside their run IDs. The two incomplete
broad +0.16 captures remain stored but are quality-failed and contribute only
their valid, reset-matched cells.

## Candidate and practice transfer

The existing high-steering evidence and this low-steering replication support
testing a rise-only cap of **0.267 normalized throttle/s** (the measured
0.08 throttle change over a 0.30 s ramp) at 4.5–7.5 m/s and steering magnitude
at least 0.08 rad. The isolated development overlay is
`config/racing/throttle_slew_lowsteer_candidate.yaml`; the baseline actuator
configuration, MPC, odometry, physics, map, and reference trajectory are not
changed by this overlay. Its focused production-actuator replay test passes.

Three collision-guarded practice transfers used the same rebuilt
MPC/integration, practice map, and frozen 9g reference trajectory. Each reached
12 total laps (warmup + 10 scored + extra), had zero collisions, and logged no
MPC fallback cycles. The trajectory SHA256 remained
`051043e5979c18de1a3fe3b3c5a40967d37d3ffbd0a220e763c31b8c6df75dd6`; the map
SHA256 remained
`fcc4416fbab3c84aed6394e23d792f5533c914a777b743556b5f2b568807985d`.

| Run | Overlay | Mean / best lap (s) | Collision delta | MPC optimal / rejected | |CTE| p95 (m) |
| --- | --- | ---: | ---: | ---: | ---: |
| frozen baseline | none | 5.7633 / 5.7018 | 0 | 2880 / 1 | 0.1028 |
| low-steer r01 | 0.267/s, 4.5–7.5 m/s, ≥0.08 rad | 5.7736 / 5.7148 | 0 | 2863 / 6 | 0.1012 |
| low-steer r02 | same | 5.7780 / 5.7328 | 0 | 2865 / 6 | 0.0962 |
| high-speed r01 | 0.267/s, 6.0–7.5 m/s, ≥0.08 rad | 5.7678 / 5.7348 | 0 | 2862 / 6 | 0.1009 |

These screens do **not** establish faster lap time. The two broad-gate means
were 5.7736 and 5.7780 s; the high-speed-only screen was 5.7678 s, all within
the baseline run's 0.031 s lap-to-lap standard deviation. Best laps also did
not beat the baseline. MPC still converged almost every cycle, but the
candidate runs had six rejected cycles each versus one in baseline.

The offline practice-bag traction analysis does show a specific candidate
effect in the targeted low-steering 6–7.5 m/s cell (positive-throttle scored
samples, 0.08–0.12 rad command):

| Run | Samples | Median absolute mismatch (m/s) | p95 absolute mismatch (m/s) | Samples > +0.25 m/s |
| --- | ---: | ---: | ---: | ---: |
| frozen baseline | 70 | 0.0779 | 0.1779 | 2/70 |
| low-steer r01 | 82 | 0.0792 | 0.1447 | 0/82 |
| low-steer r02 | 72 | 0.0678 | 0.1552 | 0/72 |
| high-speed-only r01 | 79 | 0.0763 | 0.1519 | 0/79 |

So far the repeatable signal is a smaller high-mismatch tail, not a better
typical mismatch or faster lap. These are samples nested within only one
baseline run and two broad-gate runs (plus one narrower-gate run); do not treat
the sample counts as independent-run statistical confidence. Whole-run speed
tracking p95 was 0.249 m/s in baseline, 0.281/0.254 m/s in broad-gate runs,
and 0.250 m/s in the narrower-gate run. One-step MPC speed-error p95 improved
in broad-gate runs (0.215/0.205 versus 0.260 m/s baseline), but this did not
transfer into lower full-run speed error or lap time. The narrow-gate candidate
is therefore a research overlay only; no production/default configuration
was changed.

The three practice bags, reports, and traction feature tables are retained under
their `live_runs/practice_throttle_slew_*_candidate_*` directories and
`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/`.
The audit of existing coverage remains the basis for test selection: the prior
eight-run validation already covers bidirectional high-steering step/ramp
transitions from 4.5 through 7.5 m/s, so those grids were not repeated. Current
work is limited to the missing 8 m/s rate-versus-increment interaction. The
next decision depends on whether the crossed-rate captures validate and whether
their whole-run effects replicate; no wider grid is scheduled absent a specific
new gap in that result.
