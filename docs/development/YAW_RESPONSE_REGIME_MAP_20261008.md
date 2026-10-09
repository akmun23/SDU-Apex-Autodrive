# Yaw response findings by measured regime — 2026-10-08

> **Packet-response policy correction (2026-10-08):** For the targeted yaw
> transition analyses, only samples belonging to an exactly two-packet response
> phase are valid. Any other packet count, including three, is invalid and must
> be excluded from fitting, selection, validation, and conclusions. Historical
> sections below that compare two- and three-packet responses are retained for
> traceability only; their three-packet evidence and any conclusions depending
> on it are superseded and must not guide modeling.

This is a findings-only map: a model is listed only with its measured joint
support. “Unresolved” means no validated response law is established there; it
does not mean the car cannot enter that state. Speed/steering limits below are
measured support boundaries, not physical limits.

For offline fitting, simulator-truth yaw rate and speed are labels/current-state
coordinates. A forecast may not use future simulator truth or future measured
sensors as inputs. This work concerns the next 25 ms yaw-rate transition; it is
not a 30-step MPC rollout or an odometry integration claim.

## Response laws actually identified

| Region | Identified response | Evidence and boundary |
|---|---|---|
| Crawl steady response: 0.237–1.25 m/s, 1–5% throttle, physical steering 0 and ±0.20/0.35/0.50 rad | `r ≈ 0.961264 · u_odom · tan(delta_feedback) / 0.324` describes the steady yaw-rate surface. | 70 reset-isolated conditions in one capture. Leave-both-repeats-out p95 error 0.00704 rad/s, max 0.00837; leaving one throttle level out gives p95 0.00705, max 0.00836. This is strong within-capture steady-state evidence, not independent-run or transient validation. It does not establish how far the law extends above 1.25 m/s. Source and caveats: [error/support audit](YAW_ERROR_SUPPORT_CLASSIFICATION_20261008.md). |
| Low-speed, high-steer hold: 2.5–3.5 m/s, |steering| 0.30–0.42 rad, low steering rate | Empirical signed equilibrium surface at speed knots 2.5/3.0/3.5 and steering knots 0.30/0.35/0.42, with hold relaxation `r_next = r_eq + (r-r_eq) exp(-dt/0.129314 s)`. | Two whole-run validation captures. Hold-only RMSE was 0.01054 and 0.03714 rad/s. Across all supported transitions, including sparse turn-in samples outside the fitted hold behavior, RMSE was 0.211 and 0.173. Turn-in tau 0.071 s (52 training samples) and unwind tau 0.401 s (27 samples) are too sparse and are not promoted. This identifies a hold model, not a full transient law. Source: [Y1 fit report](../../live_runs/racing_model_diagnostics_20261007/y1_authority_loss_fit/report.json). |
| High-speed, high-steer hold/short transient: 8.170–9.222 m/s, |steering| 0.350–0.500 rad | Shape-preserving equilibrium surface over 3 speed × 4 steering knots; first-order yaw response with `tau = 0.0119551 s`. | Leave-middle-speed-out equilibrium RMSE 0.002285 rad/s; leave-steering-knot-out 0.00398/0.00137; one held-out equilibrium RMSE 0.00230; one-step RMSE 0.00780. Recursive errors: 0.02125 at 250 ms, 0.00648 at 500 ms, 0.00299 at 750 ms, 0.02286 at 1.5 s, and 0.00598 at 2 s. Its current yaw state is sensor-odometry yaw, with simulator truth as the target—not a GT-initialized response fit. The holdout is four sequences from one capture previously used for another model-family final test, so it is exploratory, not independent-run confirmation. No low-angle entry, practice transfer, or full-lap support. Candidate remains offline-only: [model artifact](../../config/racing/yaw_highspeed_highsteer_scheduled_candidate.json). |
The two supported time-constant regions are distinct: 0.129 s at low-speed/high-steer hold versus 0.01196 s at high-speed/high-steer hold. That difference is measured evidence against extending one fitted tau across all regimes. It does not establish a smooth boundary between them.

## Broad one-step local predictor (not a bank of validated time constants)

The broadest ground-truth-conditioned model is the exact-cell, phase-specific
local yaw-transition atlas. Its axes are 0.5 m/s speed cells × 0.025 rad signed
physical-steering cells × turn-in / near-steady / unwind. Each cell predicts
the next 25 ms yaw-rate increment from current truth yaw/speed, recent yaw
increment, current actuator/encoder/IMU values, steering/throttle rates and
command-feedback gaps, wheel/body-speed mismatch, lateral velocity, and body
speed rate. Future truth is not an input. This is an empirical one-step
transition model, not a time-constant description.

- The requested grid has 24 × 43 = 1,032 speed/steering cells. Training data
  occupy 628; 578 have data from at least two training runs. The fitted atlas
  contains 478 supported cell/phase models (320 unphased cell models are also
  reported). It does not fill unsupported cells.
- Observed data reach 11.37 m/s, not 12 m/s. The 11.5–12.0 m/s row is empty;
  the full Cartesian speed × steering product is not measured and includes
  combinations that may not be reachable.
- On identical direct-phase support, 113,908 / 123,585 validation transitions
  (92.17%) are covered. The exact-cell ExtraTrees model scored RMSE 0.04396,
  p95 absolute error 0.03789 rad/s, 97.97% below 0.1 rad/s, and worst error
  1.221 rad/s. This is useful broad one-step prediction, but it is not “near
  zero everywhere”: 7.83% of transitions have no direct-phase model and a
  small tail remains large. Validation was used during model-family
  development; it is not a blind final estimate.
- The separate off-grid final capture tested six speeds from 4.62 to 10.62 m/s
  only at low steering (0.033–0.133 rad). It supports some low-steering speed
  interpolation, not high-steer transfer.

Artifacts: [atlas report](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/fullband_yaw_regime_atlas_report.json), [ExtraTrees comparison](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/extratrees_full_atlas_comparison.json), [held-out score](../../live_runs/racing_model_diagnostics_20261007/fullband_yaw_regime_atlas_v11_lowangle_unwind/yaw_extratrees_final_holdout_r01_score.json).

## Where the crawl relation stops matching

The crawl coefficient was re-fit on its training capture using simulator-truth
yaw/speed and, separately, the same odometry-speed input as the published law.
For `u_GT`, fitted `k=0.938317`; for `u_odom`, `k=0.961181`, reproducing the
previous `0.961264` coefficient. The fixed published law was then scored on
the clean whole-run validation captures used by the v11 atlas, without refitting
the coefficient. “Steady” samples were selected with
`|d(delta_feedback)/dt| < 0.10 rad/s`, `|d(speed_GT)/dt| < 0.25 m/s^2`,
`|d(yaw_GT)/dt| < 2 rad/s^2`, speed at least 0.1 m/s, and steering at least
0.04 rad. The table reports the original odometry-speed law's yaw-rate error;
counts are samples / distinct validation runs.

| GT speed | physical `|delta|` | support | fixed crawl law RMSE / p95 abs error (rad/s) | interpretation |
|---|---:|---:|---:|---|
| 1.5–2.5 m/s | 0.20–0.25 rad | 496 / 2 | 0.008 / 0.008 | still close |
| 1.5–2.5 m/s | 0.40–0.45 rad | 575 / 3 | 1.261 / 1.611 | clear high-steer departure |
| 2.5–4.0 m/s | 0.10–0.15 rad | 624 / 2 | 0.006 / 0.006 | still close |
| 2.5–4.0 m/s | 0.20–0.25 rad | 273 / 2 | 0.024 / 0.024 | still close |
| 2.5–4.0 m/s | 0.30–0.35 rad | 601 / 2 | 1.908 / 1.914 | large departure; boundary is between sampled bands |
| 4.0–6.0 m/s | 0.10–0.15 rad | 833 / 5 | 0.027 / 0.031 | still close |
| 4.0–6.0 m/s | 0.15–0.20 rad | 391 / 4 | 1.041 / 1.836 | large departure |
| 6.0–8.0 m/s | 0.04–0.10 rad | 2,341 / 5 | 0.262 / 0.589 | degradation begins even at low steering |
| 8.0–10.0 m/s | 0.04–0.10 rad | 2,233 / 6 | 1.061 / 1.313 | crawl law no longer useful |
| 10.0–11.5 m/s | 0.04–0.10 rad | 2,161 / 6 | 1.490 / 1.896 | large speed-dependent departure |

This rejects one speed-independent bicycle factor over the race domain. The
measured transition is regime-dependent: the low-steer relation extends to
about 4–6 m/s at small angles, while its high-steer failure moves to lower
angles as speed rises; by 6–8 m/s even the 0.04–0.10 rad band has degraded.
The data do **not** locate a sharp threshold between the sampled cells (for
example, 0.25–0.30 rad at 2.5–4 m/s). Nor do these steady-state scores explain
turn-in, unwind, reversal, or throttle/wheelspin history. Validation runs here
were held out from the crawl coefficient fit but were used in broader atlas
development, so this is a targeted transfer check, not a sealed final test.

The crawl relation itself is only directly calibrated through 1.25 m/s and
1–5% throttle. These broader validation rows show where its frozen prediction
continues or fails; they do not authorize deployment outside the stated
calibration region. The Y1 hold specialist begins at 2.5 m/s and 0.30 rad, so
the 1.25–2.5 m/s transition and exact steering boundary remain unresolved.

On two later whole-run captures explicitly excluded from GRU checkpoint
selection, the v2 command-conditioned GRU scored:

| Held-out regime | 25 ms yaw-rate RMSE / p95 | 1,000 ms RMSE / p95 |
|---|---:|---:|
| Crawl steering validation r03 | 0.12591 / 0.26562 rad/s | 0.12955 / 0.27538 rad/s |
| Low-speed/high-steer validation r03 | 0.09709 / 0.22185 rad/s | 0.19037 / 0.35482 rad/s |

The previously quoted GRU table (25 ms run-macro RMSE 0.0497, improving to
0.0457 at 100 ms) came from the checkpoint-selection validation, not these
later unseen captures. It remains a useful comparator, but it is not evidence
of generalization to the full speed/steering range.

Within the crawl capture at 25 ms, 0.5–1.0 m/s and 0–0.1 rad had RMSE/p95
0.0961/0.1359 (85 samples), while 0.5–1.0 m/s and 0.35–0.525 rad had
0.1982/0.3433 (63 samples). At 1.0–1.5 m/s, the corresponding low-steer
values were 0.0987/0.1648 (475 samples), versus 0.1814/0.3573 (382 samples)
at high steering. This supports treating low/high steering as different
response regimes; it does not locate a sharp boundary. The GRU receives the
recorded future command sequence for its full forecast, so its 25 ms score is
an offline command-conditioned forecast, not a sensor-only odometry estimate.
Score files: [crawl r03](../../live_runs/racing_model_diagnostics_20261007/yaw_gru_trajectory_teacher_v2_targeted_gaps/unseen_crawl_validation_r03_score.json), [low-speed/high-steer r03](../../live_runs/racing_model_diagnostics_20261007/yaw_gru_trajectory_teacher_v2_targeted_gaps/unseen_lowspeed_highsteer_validation_r03_score.json).

## New held-out high-steering transition region: 3–4 m/s, 0.35–0.50 rad

The independent reset-isolated validation capture
`openplane_yaw_error_highsteer_reversal_validation_r03_20261008` is complete.
It contains 216 randomized physical conditions across 3.0/3.5/4.0 m/s,
0.35/0.42/0.50 rad, both turn directions, and onset/unwind/reversal command
transitions. All 648 approach/settle/probe phases passed; there were zero
collisions, zero quality failures, and no bridge timing faults. The prepared
fixed-25-ms validation archive has 34,807 samples / 217 sequences. The recorded
stream audit measured 39.969 Hz, with no gaps above 60 ms. The entire r03 run
was held out of fitting; r01/r02 supplied the new training captures.

This is now the best measured transition-specific test, but the yaw model is
not yet good enough to call this regime solved. On controlled event windows,
v12's direct-phase atlas covers 94.6% of the 16,969 transitions. Its RMSE / p95
are 0.0676 / 0.0355 rad/s for onset (5,309 rows), 0.0889 / 0.0673 for reversal
(5,267), and 0.0673 / 0.0052 for unwind (5,482). Tails matter: maximum errors
are 0.680, 0.959, and 1.042 rad/s respectively. The p95 can look small because
the rare failure is below the worst 5%; the maximum and fraction below 0.1
must be read with it.

The v11 atlas covered 49.5% of these controlled event transitions. Adding
r01/r02 to the training data raised whole-run r03 phase coverage to 97.1%, but paired accuracy on the
22,592 samples supported by both atlases changed only from 0.06883 to 0.06866
rad/s RMSE. Thus the data materially filled support, while v12 did not materially
improve prediction where v11 already had a model. Adding previous command slew
and command/feedback gaps (v13) produced only small same-support changes:
event RMSE became 0.0675 onset, 0.0851 reversal, and 0.0648 unwind. The largest
unwind miss remained 1.038 rad/s. This is not a promoted yaw law.

The held-out samples identify a specific mechanism worth modeling: the
steering actuator does not follow one response rule for all command changes.
Predicting next steering feedback from the one-packet-delayed command using a
3.2 rad/s rate limit works well on onset (r03 RMSE 0.00464 rad), while directly
following the delayed command is better on unwind (0.01117 versus 0.02403
rad). Reversals remain difficult (0.04298 rate-limited versus 0.07455 direct;
rare errors near 0.92 rad remain). A hybrid “direct on same-sign release,
rate-limit otherwise” reduced the reversal p95 to about 0.00025 rad but did
not remove the 0.919-rad worst case. Those are actuator-next-state scores, not
yaw scores, and apply only to this measured speed/steering/command domain.

One concrete yaw failure occurred during a release at 3.97 m/s: current GT yaw
rate was 1.2816 rad/s, feedback steering was 0.420 rad, and delayed steering
command was 0. The next packet reported feedback steering 0.002 rad and GT yaw
rate 0.2386 rad/s, a 1.043-rad/s change in 25 ms. The broad v12 model predicted
1.2810 rad/s. This points to a fast actuator release/yaw transient rather than
an absent steady-state cell. A single exponential through this pair would imply
about 15 ms time constant if the target were zero, but two points do not
identify a response law; that number is explicitly not accepted as a fitted
tau.

The existing v2 GRU was also scored on all 12,171 valid one-step windows in
r03: 25-ms RMSE 0.08375, p95 0.14871, maximum 0.93194 rad/s, with 92.72% below
0.1. Gyro persistence on the same rows was 0.14094 / 0.30975 / 1.43760, with
91.73% below 0.1. So the GRU improves aggregate one-step prediction over
persistence, but does not meet the 0.1-rad/s maximum-error goal. In the
3–4 m/s, 0.4–0.55-rad subset (4,510 samples), GRU RMSE was 0.06778 versus
0.07916 for persistence; their p95 values were 0.05185 and 0.03030 and their
maximum errors 0.842 and 0.963. This is a modest aggregate gain with a
substantial rare-error tail, not “very high accuracy” for this regime.

The v14/v15 yaw fits tested explicit predicted steering change and a separate
same-sign-release indicator. Those used an overly restrictive same-sign rule.
The corrected v16 fit allows magnitude decrease across a sign change. On r03,
v16 versus v13 changed direct-phase RMSE only 0.06500→0.06489 rad/s on the
33,164 transitions supported by both, and direct-cell RMSE 0.07769→0.07683 on
33,940 shared transitions. Event RMSE changes were likewise tiny: onset
0.067524→0.067510, reversal 0.085052→0.084803, unwind
0.064797→0.064472 rad/s. The rare maxima remain 0.680/0.959/1.037 rad/s.
This does not establish a useful yaw-model gain; v16 remains offline.

### Actuator-next-packet fit on the same high-steer validation data

Added [`evaluate_steering_actuator_one_step.py`](../../tools/racing/specialists/evaluate_steering_actuator_one_step.py)
to predict measured steering feedback at k+1 from actuator/command history
through k. It trained an ExtraTrees residual correction on 48,144 domain rows
from 25 clean training runs and scored 13,735 r03 rows in 2.5–4.5 m/s,
|steering| 0.30–0.525 rad. The controlled probe slice contains 10,072
transitions across 216 randomized conditions. No future truth or sensor value
is an input; simulator truth is used only to select the offline speed domain.

The one-sample-delayed command is the best simple alignment among k/k−1/k−2.
For k−1, the corrected rule—follow the delayed command directly when its
magnitude decreases, including across a sign change; otherwise rate-limit at
3.2 rad/s—gives r03 RMSE 0.02136 rad, p95 0.00020, max 0.99981. It improves on
the fixed limiter (0.03095 RMSE) and direct following (0.03546). Across 216
paired probe conditions it beats the fixed limiter by a mean 0.01344 rad in
condition RMSE (95% bootstrap CI 0.00909–0.01817); 82 conditions improve and
22 worsen. This is a measurable local actuator-model gain, not a yaw or
full-pipeline gain. Event scores are onset 0.00354 RMSE / 0.04975 max,
reversal 0.03356 / 0.99981, and unwind 0.01950 / 0.41890 rad. The rare tail
remains unacceptable for a general model.

An ExtraTrees residual lowers pooled RMSE to 0.02018 but worsens 186/216
paired condition RMSEs; mean condition-RMSE change is +0.00370 rad (95% CI
+0.00250–+0.00497). It is rejected. A separate causal rule that uses command
`k−2` for one transition when a new command sign reversal is pending fires on
only 26 r03 samples and also worsens condition-level performance: mean change
+0.00436 rad (95% CI +0.00113 to +0.00794), with 62 conditions worse and 15
better. Reversal RMSE rises from 0.03356 to 0.04753 rad. It is rejected as a
global fix, despite helping a few individual step profiles.

Added [`analyze_steering_actuator_timing.py`](../../tools/racing/specialists/analyze_steering_actuator_timing.py)
to compare raw receipt-time onset with the required 25-ms simulator packet
grid. Receipt-time estimates center around 60–65 ms but are not treated as
physical time because message receipt jitter is not simulator `dt`. Packet
sequence onset counts are more useful: r01 has 149/216 transitions at 2 steps,
34 at 3, 25 at 1, and 8 at 0 or 4–7; r02 has 187 at 2, 23 at 3, and 6 at 1;
held-out r03 has 190 at 2, 24 at 3, and 2 at 1. All measured onset intervals
were packet-sequence-contiguous. Thus the existing one-previous-command MPC
queue corresponds to the dominant two-packet command-to-feedback onset, while
a minority has another packet of delay and r01 is substantially more
variable. This is real run-to-run/mode uncertainty, not a basis for globally
adding one queue step. A representative r03 full-angle reversal holds old
feedback through packet `p+2` and begins slewing by `p+3`; that is one packet
later than the dominant response.

No actuator/yaw candidate is integrated into odometry or MPC. The yaw atlas
shows no material r03 gain from the corrected feature; the sign-reversal
queue heuristic worsens paired conditions; and the simple actuator hybrid is
supported only in this 2.5–4.5 m/s, 0.30–0.525-rad region. The next step is to
use training-only r01/r02 histories to find a causal state/timing signal that
predicts the 1/2/3-packet mode, then evaluate that frozen rule on r03 by
condition and event. The first condition-only and coarse packet-index-phase
checks failed, but bridge request records then exposed command-update age as
a useful timing signal; that decomposition is reported below. Do not infer
physical time from ROS receipt jitter or request another run until the
existing-data analysis identifies a missing observable.

### Existing-data onset-mode predictability check

The first timing-mode check is complete using the three already-collected
captures; no simulator was started. A mode classifier using only randomized
probe descriptors (event, speed, steering magnitude/sign, transition type and
duration), trained on r01/r02 and evaluated on r03, does not predict the
minority response modes. The r03 majority baseline (always predict 2 packets)
is 190/216 correct (87.96% accuracy, 0.333 balanced accuracy). A random forest
gets 87.04% / 0.330 balanced accuracy and an ExtraTrees classifier gets
82.87% / 0.326. Both largely predict the 2-packet class; neither identifies
the 24 three-packet or 2 one-packet cases usefully. A further check binned
command-start packet sequence modulo 2, 3, 4, 5, 8, 10, and 20, learning the
most common training mode per bin. None improved the held-out result: each
still predicts 2 packets for all r03 probes (87.96% accuracy, 0.333 balanced
accuracy). This coarse packet-index phase is not a useful mode predictor.

The same exact probe condition does not have a fixed onset mode across runs:
agreement is 62.0% for train r01 vs r02 and 76.9% for train r02 vs held-out
r03. This supports run-/packet-phase variability or another unobserved state,
but does not establish its cause. It means a deterministic lookup by speed,
steering and transition profile is not justified. However, the bridge packet
diagnostics expose a better explanatory signal: age of the latest steering
command when each 40-Hz request is emitted, plus the actual steering value
included in that request. The follow-up below checks this command-side timing
against the feedback onset.

### Separating command-delivery phase from actuator response

The three existing bags contain `/bridge_packet_timing` records. For each
request these report the bridge request sequence, the command snapshot sent,
command-update age at request time, and request-to-response transport latency.
These were joined to the same packet-sequence-aligned steering-command and
feedback records used above. On held-out r03, request-to-response latency was
stable (p50 26.44 ms, p95 27.23 ms, p99 27.59 ms, maximum 30.80 ms). It did not
increase in the three-packet-onset group: at command onset, median response
latency was 26.41 ms for two-packet cases and 26.27 ms for three-packet cases.

Command-update age did separate those groups: its median was 10.22 ms for the
190 two-packet cases, 22.30 ms for the 24 three-packet cases, and 5.35 ms for
the two one-packet cases. This pattern replicated in train r02 (two-packet
median 10.27 ms; three-packet median 22.16 ms). It indicates that the extra
packet is associated with the command update landing close to/after the
bridge's 25-ms request boundary, rather than with slower feedback delivery.

The request-by-request command snapshot supports that interpretation. Using a
0.01-rad command-change threshold, on r03 the first bridge request containing
the changed command occurred one packet after the observed ROS command change
in 189/190 two-packet cases, and two packets after it in 23/24 three-packet
cases. In those same cases, measured steering feedback began moving one packet
after the changed command was first included in a bridge request. Four cases
deviated from this common decomposition. Thus the apparent two/three-packet
command-to-feedback “actuator delay” is mostly a command-update/request-phase
delay followed by an approximately one-packet feedback response. This is a
strong empirical explanation for the extra packet, not proof of the Unity
physics apply tick: the bridge has no echoed applied-command ID or native
simulator frame counter, and its packet sequence is assigned locally on
receipt.

The runtime observability difference matters. The bridge stamps sensor headers
with local receive time; the MPC stores the latest steering feedback and its
steady-clock receive age (freshness limit 75 ms), but does not consume the
bridge packet ID or request command snapshot. Bag analysis can retrospectively
join packet channels and use the fixed 25-ms sequence, while the live MPC
cannot identify which request carried the command from the permitted steering
feedback value alone. The current runtime model has a fixed 25-ms physical
steering queue (`physical_steering_delay_s`) and zero extra command-application
delay (`command_actuation_delay_s: 0.0`). Thus it represents the measured
request-to-response interval, but not the observed variable command-update-to-
request phase. The current evidence does not justify collapsing that extra
phase into a longer physical servo constant. It is not impossible to model the
*effective* path offline; runtime prediction of its phase needs a causal signal
or a justified statistical phase model, neither of which is established yet.

Next: score the separated offline stages on the existing r01/r02/r03 captures,
then test whether command-update age can be causally recovered from permitted
runtime input timestamps. If it cannot, evaluate a training-only distribution
for the request phase rather than pretending its exact value is known. Do not
make the competition MPC subscribe to diagnostic packet topics. No new
data-derived phase model has been integrated.

## Regimes still without an accepted local response law

| Regime / mechanism | Current evidence | Status |
|---|---|---|
| 0–2.5 m/s transient yaw, especially |steering| ≥0.20 rad | Crawl steady points plus targeted crawl/high-steer train and validation runs. The current-variant v5 25 ms teacher improves RMSE over gyro persistence on these runs but has p95 0.176/0.161 rad/s in crawl/high-steer, respectively. Its report includes command conditioning, so do not call it a sensor-only observer. | No validated time-constant or low-speed transient law; low-steer/high-steer split is established qualitatively, exact boundary unresolved. |
| 3–4 m/s, |steering| 0.35–0.50 rad: turn-in, unwind, reversal | Two 216-probe training captures and one independent 216-condition validation capture. A causal one-step actuator hybrid beats the fixed rate limiter in paired condition RMSE, but retains a 0.9998-rad max tail. Packet-grid command-to-feedback onset is 2 steps in 149/216, 187/216, and 190/216 conditions across r01/r02/r03, with a 1–3+ step tail. Bridge request diagnostics show that the extra packet tracks command-update age (about 22 ms vs 10 ms) rather than request-response latency; most transitions have one packet from command inclusion to feedback movement. | No accepted yaw response law or runtime actuator model. v16 has negligible yaw gain; global-delay, learned-residual, condition-only, and coarse packet-phase candidates fail. Next model command-update/request phase separately from feedback response; preserve the diagnostic topic outside competition MPC inputs. The GRU improves aggregate RMSE over gyro persistence but misses the ≤0.1 maximum-error goal. |
| 4–8 m/s, broad steering | 85 equilibrium points exist at 4.5/6.5/7.5 m/s; a first-/second-order local fit was attempted. Many fitted retentions hit the 0.995 stability cap, indicating the data did not identify those taus. The phase-conditioned one-step atlas predicts supported cells. | Equilibrium surface and one-step predictors exist; no trustworthy local response-time constants. |
| 8–11.37 m/s outside 8.17–9.22 × 0.35–0.50 rad | Full-band one-step atlas has supported cells; low-angle off-grid speed test exists. | No validated extension of the high-speed/high-steer tau beyond its stated rectangle; some high-speed/steering cells are sparse. |
| 11.37–12 m/s | No samples in the current full-band training grid. | Unsupported; obtain data only for physically achieved states, and do not invent a speed/steering Cartesian envelope. |
| Command onset/reversal × throttle slew × wheel/body mismatch interactions | Current one-step atlas includes steering/throttle rates, actuator command-feedback mismatch, mean rear-wheel/body-speed mismatch and lateral velocity. Paired throttle and high-steer reversal datasets exist. Rear-wheel split and IMU roll did not improve full-domain grouped validation. | Factors are available to the predictor, but their separate local response laws and interaction boundaries are not yet identified. |

## 2026-10-08 follow-up: command-age candidate and fixed-two-packet check

A sensor-only local yaw atlas candidate added causal time-since-change for
steering/throttle commands and feedback (25-ms fixed grid, capped at 400 ms),
in addition to the prior 0/25/50/100-ms sensor history. It used the same
62-train/36-validation split and the same 299,242 held-out one-step targets as
the command-intent v2 atlas. On the 266,562 rows where both local atlases had
support, RMSE changed from 0.028935 to 0.029139 rad/s. Run-clustered
v4-minus-v2 RMSE was +0.000330 rad/s (95% bootstrap CI +0.000043 to
+0.000636; 12/36 runs better). The feature revision is rejected; it does not
improve the local predictor. The full comparison is preserved beside the
candidate report in
`live_runs/racing_model_diagnostics_20261008/sensor_only_yaw_regime_atlas_command_age_v4/`.

The fixed-two-packet counterfactual was checked on the matched r03 high-steer
probe conditions. In that capture, exactly 190/216 conditions (88.0%) had a
two-packet command-to-feedback onset, not 95%. Across all 216 conditions,
the v16 one-step yaw atlas RMSE was 0.072695 rad/s; on only the 190 two-packet
conditions it was 0.072314 rad/s. The worst error remained 1.0374 rad/s and
occurred during a two-packet unwind at 4.0 m/s and 0.42 rad steering. Thus
excluding one-/three-packet cases changes aggregate RMSE by only 0.000381
rad/s (about 0.5%) and does not remove the yaw-response failure. This is a
validation filter, not evidence that a model trained only on that subset will
transfer to the discarded timing modes.

A narrow first-order unwind law was then fitted from the five matched
two-packet training events at wheel-speed bin 4.0–4.5 m/s and signed steering
centers ±0.425 rad. For a near-zero steering command changed one 25-ms packet
ago while measured steering is still stale and yaw magnitude is high, the
training-only fit is
`r[k+1] = 0.23116 * r[k]` (decay fraction 0.76884). On the three independent
r03 two-packet events, absolute errors are 0.0120, 0.0163, and 0.0577 rad/s.
The same sensor/command gate also fires on the r03 three-packet event; there
the yaw has not yet decayed, and this law errs by 0.9839 rad/s. Over the 103
held-out rows in both signed cells, applying the fixed-two-packet override
raises RMSE from 0.11835 to 0.13089 rad/s and the maximum from 0.63743 to
0.98389, despite improving the under-0.1 fraction from 89.3% to 92.2%. The
candidate and full score are preserved at
`live_runs/racing_model_diagnostics_20261008/yaw_release_decay_two_packet_v1/`;
the reusable fitter is
[`fit_yaw_two_packet_release_specialist.py`](../../tools/racing/specialists/fit_yaw_two_packet_release_specialist.py).

This is a useful local response law *conditional on exactly two packets*, but
not a safe all-mode rule. On r03 the negative-steering two- and three-packet
samples have almost the same observed command age (25 ms), stale feedback age
(400 ms capped), steering, and yaw. Yet the two-packet sample moves from
-1.2815 to -0.2842 rad/s while the three-packet sample remains at -1.2798
rad/s. The bridge request diagnostic identifies the timing mode; the
competition-visible history does not currently identify it reliably. Retain
the coefficient as a research comparator, not a runtime branch, until the
request phase is made deterministic or a permitted causal predictor of it is
validated.

The timing variation itself has a concrete likely source in the bridge
diagnostics: command-update age at request time had median 10.22 ms for the
r03 two-packet cases and 22.30 ms for its three-packet cases; request/response
latency did not separate them. In 189/190 two-packet cases the changed command
first appeared in a bridge request one packet after the ROS command update;
in 23/24 three-packet cases it appeared two packets later. Feedback then
usually moved one packet after the request carrying the new command. This
supports a command-update phase relative to the 25-ms bridge request boundary,
followed by the feedback response; it is not evidence for a randomly varying
physical servo constant. The bridge timing topic remains offline diagnostic
only and must not be consumed by the competition MPC.

The sensor-only v2 atlas still does not cover the entire requested domain:
validation reaches 11.299 m/s and 0.5236 rad, but training has no samples in
the 11.5–12.0 m/s bin. Of 640 validation speed/steering cells with at least
20 rows, 511 have local-expert support and 46 supported cells fail the current
p95/within-0.1-rad/s criterion. These are measured cells, not a full Cartesian
or physically feasible envelope. No new run, MPC/odom integration, or runtime
parameter change was made. The next fit should concentrate on yaw-event
failures that persist inside the two-packet subset (especially unwind and
reversal), rather than spending more effort on command-age features that did
not transfer. Keep unsupported cells explicit; do not claim full-band
accuracy or promote a candidate from these one-step scores alone.

Frozen coefficient replay remains available in
[`score_frozen_yaw_atlas.py`](../../tools/racing/specialists/score_frozen_yaw_atlas.py)
and reproduces the earlier v11 score exactly on its original validation run.

### 2026-10-08: unwind/reversal mismatch cause and held-out neighborhood fit

The unwind/reversal neighborhood candidate has now been scored on all 36
whole-run validation captures, with simulator-GT next yaw rate as the target
and no test/final-test data read. Relative to the frozen v2 atlas it reduces
one-step RMSE from 0.03694 to 0.03117 rad/s, p95 from 0.04012 to 0.03095,
and samples above 0.1 from 5,871 to 3,814 (98.04% to 98.73% within 0.1).
The maximum remains about 1.352 rad/s: the severe tail is not solved.

The improvement is strongly event-specific. Unwind errors above 0.1 fall
from 3,221/29,045 to 1,208/29,045; unwind RMSE/p95 fall from
0.08484/0.17774 to 0.05715/0.08619 rad/s. Reversal errors fall only from
754/2,722 to 710/2,722, and RMSE/p95 only from 0.17436/0.40714 to
0.17082/0.40189. The candidate beats its earlier fallback-gap-fill version
on 31/36 validation runs (paired run-macro RMSE delta -0.00305 rad/s,
95% bootstrap interval [-0.00388, -0.00220]). This is a real held-out
one-step gain for unwind, not evidence that reversal is fixed.

The controlled high-steer r03 bag independently tests whether command-to-
feedback packet phase contributes to those two events. Across command-onset
windows, candidate error rates were:

| Probe event | 2-packet response | 3-packet response | Matched conditions |
|---|---:|---:|---:|
| Unwind | 46/1,488 (3.1%) >0.1; RMSE 0.0475 | 14/191 (7.3%); RMSE 0.1241 | 5/6 pairs worse for 3 packets; mean paired RMSE increase 0.0480 rad/s, bootstrap 95% CI [0.0038, 0.1021] |
| Reversal | 103/1,507 (6.8%) >0.1; RMSE 0.0705 | 21/214 (9.8%); RMSE 0.1193 | 7/9 pairs worse for 3 packets; mean paired RMSE increase 0.0531 rad/s, bootstrap 95% CI [0.0265, 0.0794] |

These matched comparisons support a packet-phase effect, but also show it is
not the whole explanation: many two-packet rows remain above threshold, and
some three-packet phases are predicted well. In the narrow 4.0 m/s,
0.42-rad unwind gate, two and three packet cases have nearly identical
competition-visible values at prediction time: command age 25 ms, feedback
age 400 ms, steering about 0.42 rad, command about zero, and yaw about
1.28 rad/s. Yet one next-tick GT yaw is about 0.24–0.28 rad/s and the
three-packet case remains about 1.28 rad/s. A fixed two-packet decay law gets
the three two-packet validation examples within 0.058 rad/s but misses the
three-packet example by 0.984 rad/s; applying it to the full 103-row cell
raises RMSE 0.11835 to 0.13089. Do not use that law as an unconditional
runtime selector.

The likely cause is command/request phase, not a different steady-state yaw
coefficient: existing bridge diagnostics show update age at the 25-ms request
boundary separates most two- and three-packet transitions, while request-to-
response latency does not. Most transitions then begin feedback movement one
packet after the changed command is included in a request. The competition
MPC may not consume `/bridge_packet_timing`; this remains offline diagnostic
data. The available sensor/command history does not yet reliably identify
which timing branch will occur before feedback moves. Thus the model can
identify an ambiguous/high-risk state, but cannot guarantee its exact next
GT yaw from current legal inputs while the bridge phase varies.

I also tested a compliant timestamp proxy on the three captures: bag receipt
time of the steering-command change minus the latest permitted odometry source
stamp. A single threshold fitted to r01/r02 reaches only 66.8% accuracy on
r03, versus 88.8% for always predicting the dominant two-packet class; it
recovers 8/24 three-packet cases (balanced accuracy 0.522 versus 0.500 for the
majority baseline). This proxy is not a usable phase selector. Bag receipt
time is less precise than the MPC's own command-publication timestamp, so this
does not rule out that more precise internal timing; it must be checked against
the actual MPC callback/command path before relying on it.

The candidate is preserved at
[`unwind_reversal_neighborhood_v2_supported`](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/unwind_reversal_neighborhood_v2_supported/)
and the timing join at
[`timing_residual_join.json`](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/unwind_reversal_neighborhood_v2_supported/timing_residual_join.json).
The candidate bundle is about 176 MB (646 local tree experts); it is not a
runtime-ready artifact and has not been replayed through production odometry
or MPC. Keep it as an offline comparator, not as a production promotion.

No additional open-plane capture is needed to establish this diagnosis: two
training captures plus the independent 216-condition r03 run already contain
the phase variation and matched cases. The next useful action is a narrowly
scoped deterministic command/request-phase change or a legal, causal estimate
from command and sensor timestamps, followed by a real simulation comparison.
Do not collect another generic yaw sweep. No simulator was launched for this
analysis; physics, reference raceline, odometry, MPC, and runtime topic
subscriptions were unchanged.

### New high-error classes found on 2026-10-08

The 100 largest supported v2 held-out errors are 69 unwind, 22 turn-in, 6
hold, and 3 reversal samples. They separate into at least three mechanisms:

1. Receipt-causal IMU/GT packet mismatch: 9/100 have current archived IMU yaw
   differing from same-packet truth by >0.1 rad/s, with the IMU often repeated
   while truth changes. In the largest case, packet 6212's archive input is
   -1.1286 rad/s, while its raw same-source-time IMU and GT are -0.1667. The
   IMU was received ~14 µs after odometry, so the extractor's intentional
   receipt-causal join picked the prior packet. This is a measurement-age
   feature/alignment issue, not a vehicle-dynamics conclusion.
2. Steering release near zero: at 5.5–8.5 m/s, yaw can collapse when measured
   steering snaps to zero; v2 often predicts yaw persistence. Similar-looking
   states can persist one packet before release, so command age alone is not a
   reliable mode selector. A simple fitted retention law worsened the
   targeted validation distribution and is rejected.
3. Command reversal ahead of feedback: at 3.50 m/s, steering validation r03
   has +0.329-rad feedback and +0.877-rad/s yaw while the command has reversed
   to -0.35 rad. The bridge request carries the negative command before the
   measured feedback turns; next truth is -0.421 rad/s. This requires a
   separate causal reversal response, not a steady-state yaw fit.

Detailed counts, packet traces, the rejected targeted-law metrics, and the
next data-alignment steps are recorded in
[`YAW_REGIME_MODELING_PROGRESS_20261007.md`](YAW_REGIME_MODELING_PROGRESS_20261007.md).
No runtime yaw model was promoted from these findings.

## Latest exact-two-packet high-steer residual status — 2026-10-08

This section supersedes the earlier “r10 in progress” status above. Only
exactly two-packet samples were used; one-, three-, and every other packet
count were excluded. The target is next-25-ms yaw-rate error in rad/s.

The r10 training capture is complete and admitted: 146 scored two-packet
response phases from 480 scheduled phases, 39.60-Hz command delivery, no
collision or bridge fault. Adding it to the training set and selecting by
whole-capture folds produces a causal mirrored candidate that lowers held-out
reversal errors from 13/156 to 7/156 on ordinary 40-Hz r03, and 20/90 to
15/90 on separate packet-phase r09. It does not make all samples fall under
0.1 rad/s; worst held-out errors remain 0.882 and 0.369 rad/s, respectively.
Unwind is 2/538 over threshold in r03 but 18/162 in r09, where the candidate
slightly worsens the count versus the mirrored baseline (17/162).

Ordinary r03 reversal misses are concentrated 25–50 ms after command
transition (6/20 in that age bin); r09 reversal misses are 5/26 in the first
25 ms and 10/23 from 25–50 ms, with none after 50 ms. r03 unwind's two misses
share one 3.5-m/s, 0.42-rad ramp condition at 99 and 125 ms. r09 unwind's
largest miss is 0.876 rad/s at 26 ms, and 13/60 errors in the 50–150-ms bin
exceed 0.1. These are transition-response failures, not broad steady-state
speed/steering errors.

The actual-next-steering oracle nearly clears ordinary r03, but still leaves
14/90 r09 reversal and 19/162 r09 unwind errors above 0.1 rad/s. Thus the
ordinary-rate group is partly limited by actuator prediction; the distinct
packet-phase group also has yaw-dynamics error that steering knowledge alone
cannot explain. Expanding input history from 100 to 500 ms did not improve the
threshold count. Separate 3-speed, 3-steering, and 9-cell speed×steering
experts all lost to the global event-specific model in exact-two training
folds (80 reversal misses for global versus 85–102 local; 56 unwind versus
71–94 local). Do not promote the local bins or the longer-history model.

Current status: useful but incomplete reversal gain; no model reaches 100%
within 0.1 rad/s, and nothing from this experiment is integrated into odometry
or MPC. Next hypothesis is a causal latent actuator/tire-response state, not
another speed/steering-only partition. If existing data cannot identify that
state, gather only paired, reset-isolated exact-two reversal/unwind probes
with matched current state and deliberately varied pre-transition steering
and throttle/wheel-speed history. See the full [progress and evidence](YAW_REGIME_MODELING_PROGRESS_20261007.md#2026-10-08-exact-two-residual-follow-up-age-history-depth-and-local-experts)
and machine-readable reports linked there.

The subsequent complete v2 validation census confirms this is not only a
top-100 phenomenon: among 299,242 one-step validation transitions, 5,871
(1.962%) exceed 0.1 rad/s. Error rates by event are 0.26% hold, 3.69% turn-in,
11.09% unwind, and 27.70% reversal. The current local atlas still uses
simulator truth only for the next-yaw target and offline scores; online-style
features/selection remain sensor/command-derived. Reversal is both sparse
(only 591 local-expert rows from 2,722 validation rows) and inaccurate even
where locally supported (89/591 >0.1). Unwind is inaccurate in both local
experts (1,353 errors) and global fallback (1,868 errors). Detailed metrics
and every >0.1 sample, including expert/fallback provenance and training
support, are in
[`yaw_large_error_audit_v1`](../../live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/).
The next model work should therefore treat release and pre-feedback reversal
as separate transition-response regimes, retain GT as the supervised target,
and never use GT speed/yaw or bridge packet timing as an online selector.
