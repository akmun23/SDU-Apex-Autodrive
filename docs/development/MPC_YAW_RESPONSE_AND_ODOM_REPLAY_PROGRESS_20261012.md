# Yaw-response and odometry replay progress — 2026-10-12

## Data-access integrity correction — 2026-10-12

**The new yaw-model results in this document are exploratory, not clean
sealed-set validation.** The source archive
`practice_transfer_validation_20261001_r03/openplane_dynamics.npz` contains
12 train, 5 validation, 2 unseen-practice, 1 test, and 1 final-test run. The
new jobs selected/fitted models using only the intended train/development
rows, and their practice scoring used only the two named practice runs, but
the shared loader decompressed the archive's numeric arrays before filtering
by run ID. Therefore the test and final-test arrays were read into process
memory. We cannot claim those partitions were never accessed, and none of the
recent scores is admissible as untouched sealed-set evidence. No evidence
shows that test/final-test rows entered fitting, feature selection, or
checkpoint selection; the breach is archive-level data access, not known
label-driven tuning.

The results and artifacts are retained as exploratory diagnostics, explicitly
not promoted as held-out validation. Both the yaw row collector and shared
`train_nssm._load_dataset()` now check only `run_ids`/`run_splits` first and
refuse to decompress any archive containing `test` or `final_test`. A targeted
check monkeypatched the numeric loader to fail if called and confirmed refusal
occurs before that loader. The complete audit and affected output families
are listed in
[`SEALED_ARCHIVE_ACCESS_AUDIT_20261012.md`](SEALED_ARCHIVE_ACCESS_AUDIT_20261012.md).
Future work uses the separately admitted, manifest-gated full-spectrum
corpus, whose run-level splits are established before numeric arrays are
opened.

## Scope and safeguards

This work used existing captures only. No simulator was launched, and no
production physics/controller/raceline was changed. The recent small-model
results below are exploratory under the access correction above. The yaw
experiment is an isolated response diagnostic: future body `u/v` and
steering-command history are supplied from truth at each 25 ms step, so it is
not a full plant rollout and is not suitable to integrate into MPC or
odometry.

## What the existing evidence says

The MPC-available HGB history model uses body `u/v/yaw-rate` and steering
command/history, not wheel speed, measured actuator feedback, or IMU
acceleration. On the two frozen practice-transfer captures its teacher-forced
one-step yaw-rate RMSE is 0.0274 rad/s; 98.6% of samples are within 0.1 rad/s.
That score does not survive recursive use: with oracle future `u/v` and
steering, its yaw-rate error passes 0.1 rad/s by about 0.275–0.30 s and
continues to grow. A seven-value residual-gain sweep selected full gain on
development validation, but the corresponding practice recursive errors
remain poor. This candidate is therefore not promoted.

## Stable-feedback candidate result

A new training-only model study constrains feedback explicitly as
`r_next = rho * r_current + f(u, v, steering-command history)`. The learned
function excludes yaw-rate history; `rho` is selected only on development
whole-run validation. Candidate values are 0, 0.5, 0.8, and 0.95. The selected
value is `rho = 0.8`. On the two frozen practice transfers, recursive yaw-rate
RMSE averaged over the six reset-delimited sequences per capture is 0.158 and
0.159 rad/s. Mean sequence RMSE at 0.1, 0.25, 0.5, and 1.0 s is respectively
0.054, 0.053, 0.057, and 0.081 rad/s. A teacher-forced one-step check of this
model is 0.0477 rad/s RMSE, p95 0.0907, with 96.6% of samples within 0.1 rad/s.

This is a meaningful reduction against the previous MPC-input HGB history
model's recursive practice RMSE of about 0.33 rad/s, but it is not a solved
yaw predictor: full-sequence RMSE remains above 0.1, the candidate sacrifices
one-step accuracy versus the prior HGB history model's 0.0274 rad/s, and its
development performance differs sharply by source (0.190 rad/s race-domain
composite versus 0.0072 rad/s paired-throttle composite). This mismatch is
being traced by regime before any promotion. Even a good result remains a
teacher-forced yaw-only diagnostic until it is tested using estimated (not
truth) body state and converted to a runtime-compatible model.

Detailed report and selected model are
`live_runs/derived_dynamics_learning_20260928/odom_wheel_signal_correction_20261011/mpc_prediction_analysis_20261012/stable_ar_yaw_response_20261012.json` and its sibling `stable_ar_yaw_response_20261012_models/yaw_ar_rho_0.8.joblib`.

The recursive residual diagnosis localizes the practice-transfer failures:

- At 2–4 m/s, recursive yaw-rate RMSE is 0.169 rad/s (46.7% within 0.1).
- At 4–6 m/s, it is 0.162 rad/s (50.4% within 0.1).
- At 6–8 m/s, it falls to 0.131 rad/s (56.8% within 0.1).
- At 0.2–0.3 rad steering command, it is 0.192 rad/s (31.0% within 0.1).
- At absolute throttle-command slew 1–2 norm/s, it is 0.249 rad/s (29.7% within 0.1; 182 samples). Above 2 norm/s it is 0.174 rad/s (29.1%; 110 samples).

The longest consecutive failure stretches last about 0.8–0.9 s around 3.7–4.1 m/s, 0.17–0.20 rad steering, and nonzero throttle slew. This is consistent with missing coupled turn-in/throttle-response dynamics, not merely high-speed saturation. The residual report is `stable_ar_yaw_residual_signal_diagnosis_20261012.json` beside the model report.

The throttle-command-history ablation is complete and **does not improve transfer**. At fixed `rho = 0.8`, one-step practice RMSE worsens from 0.0477 to 0.0513 rad/s; full recursive practice RMSE worsens from 0.158/0.159 to 0.185/0.180 rad/s. It is rejected. So the throttle-slew correlation is not resolved by simply feeding the requested throttle history to this response map.

Further residual stratification finds higher recursive errors with large measured lateral acceleration (≥8 m/s²: 0.278 rad/s RMSE, 227 samples), large throttle-feedback/command mismatch (≥0.1: 0.230, 179 samples), and encoder-derived wheel/body speed mismatch of 0.5–1 m/s (0.237, 94 samples). Steering-feedback/command mismatch is usually small and does not separate the error as strongly. These are associations, not causal proof, and the bins contain serially correlated samples. Existing data now supports a focused check of whether measured lateral acceleration explains the error; if it does, it must be predicted internally for recursive rollout rather than consumed as a future sensor.

The IMU lateral-acceleration ablation improves teacher-forced one-step practice yaw error only slightly (0.0477 → 0.0449 rad/s RMSE; 96.6% → 97.3% within 0.1). Supplying future measured `a_y` as an oracle gives recursive yaw-rate RMSE 0.144/0.143 rad/s, only about a 10% improvement over the no-`a_y` candidate, while trajectory position error is worse (1.83/1.66 m versus 0.79/0.69 m averaged over reset-delimited sequences). This does not establish `a_y` as the missing causal term and is not a useful deployable rollout. The report is `stable_ar_yaw_imu_ay_ablation_20261012.json` beside the other model results.

Since local feature additions did not fix recursive drift, the next implementation uses a smooth differentiable response function trained directly on 1 s recursive yaw windows, with the stable recurrence retained. This specifically tests whether the remaining error comes from HGB's piecewise one-step fit and teacher-forced objective rather than simply missing one measured input. Training remains limited to train runs; development runs select checkpoint, practice stays held out.

The 1 s recursive MLP run completed on the GTX 1080 using 529,064 training samples across 2,264 accepted sequences (six sequences were excluded by completeness checks); development selected epoch 75. On the two frozen practice transfers, recursive yaw-rate RMSE is 0.023/0.024/0.033/0.035 rad/s at 0.1/0.25/0.5/1 s, much better than the stable HGB candidate through 1 s. But one-step RMSE is 0.0491 rad/s (94.2% within 0.1, slightly worse than the stable HGB's 0.0477 and 96.6%), and full-sequence RMSE is 0.180 rad/s, worse than the HGB candidate's 0.158/0.159. Thus this is a useful short-horizon result, not a successful full-motion model. The report/model are `recursive_yaw_mlp_20261012.json` and `recursive_yaw_mlp_20261012.pt` in the same analysis directory.

A matched-compute 4 s recursive-window MLP run also completed, with the same features and training split and a 160-step objective. Development selected epoch 75 (composite 0.0987 rad/s), but transfer failed: practice one-step RMSE rose to 0.0813 rad/s with only 78.6% within 0.1, recursive full-sequence RMSE rose to 0.367 rad/s, and 1 s RMSE was 0.0534. This is worse than both the 1 s MLP and stable HGB candidates, so the long-window MLP is rejected. Its output is `recursive_yaw_mlp_4s_20261012.json` and `.pt`.

The contrast—good 1 s MLP rollout, poor 4 s transfer despite a good development score—shows a substantial development-to-practice domain mismatch, not simply a lack of longer training windows. A joint-coverage audit now shows that the dominant 3–4 m/s, 0.2–0.3 rad practice region has abundant ordinary samples in race-domain training (about 22,900 rows across 11 captures), but the high-throttle-slew subset is only 84–88 adjacent samples across 39 episodes and four captures; one training capture supplies 72 of those rows/35 episodes. The paired throttle-slew training set has no samples in that joint regime (its maximum recorded throttle-command slew there is about 0.27 command/s). The two held-out practice captures contain 78 such rows in this 3–4 m/s band. Thus the unresolved response is not an overall lack of speed/steering coverage; it is sparse, highly concentrated coverage of the coupled transition that most resembles the failing practice condition.

This evidence motivated one controlled architecture test: keep the same features, 1 s recursive objective, training rows, and development-only checkpoint selection, but replace fixed `rho=0.8` with a smooth state-conditioned pole bounded to `[0.2, 0.98]`. That run completed and failed recursive transfer as described below. It remains a yaw-only oracle-input experiment (future body `u/v` and steering command are supplied), not a complete vehicle model; no candidate is integrated into odometry or MPC.

The selected adaptive-pole checkpoint fails the frozen practice transfer: teacher-forced one-step RMSE is 0.0356 rad/s (98.5% within 0.1), but recursive RMSE is 0.111 at 0.1 s, 0.199 at 0.25 s, 0.256 at 0.5 s, and 0.231 over full sequences. Its learned pole is concentrated near 0.94–0.96 (95th percentile near the imposed 0.98 ceiling), so small transition biases persist instead of being corrected; this is a failed rollout candidate, not evidence for that physical time constant. In the exact 2–4 m/s, 0.2–0.3 rad practice subset, adaptive-pole MLP one-step error is 0.0245 RMSE but recursive error is 0.305; the fixed-pole MLP is 0.0621 one-step and 0.261 recursive. For throttle slew ≥1 command/s in that same subset, their recursive errors are 0.321 and 0.296 respectively. The earlier stable-AR HGB remains better overall in this region (about 0.169 recursive RMSE at 2–4 m/s), but it also exceeds the 0.1 target. These paired metrics confirm that the main failure is recursive transition response, not just one-step fitting.

The adaptive candidate and full per-capture results are in `recursive_yaw_mlp_adaptive_pole_20261012.json` and its `.pt` artifact. It is rejected for runtime use. Its fitting/checkpoint selection did not use test/final-test rows, but those arrays were decompressed by the shared loader; treat this result as exploratory, not as sealed-set validation.

## Production odometry replay correction

The exact production sensor-odometry node was replayed from left/right encoder
and IMU topics in an existing paired-throttle capture. Simulator truth was
used only as an offline label. There are 6,047 valid-phase samples. Correct
reset-local scoring aligns each of 64 valid phase sequences independently;
the earlier single alignment across the full reset-separated capture is
invalid and must not be cited as odometry drift.

| Quantity | Result on this capture |
| --- | ---: |
| Forward-speed RMSE | 0.174 m/s |
| Lateral-speed RMSE | 0.0151 m/s |
| Yaw-rate RMSE | 0 rad/s (current odom passes gyro yaw rate through) |
| Mean phase-level relative-position RMSE | 0.102 m |
| 95th percentile phase-level position RMSE | 0.175 m |
| Worst phase-level position RMSE | 0.578 m |
| Forward-speed RMSE above 8 m/s | 0.233 m/s (1,824 samples) |

Reading the v7 odometry diagnostics on the same 6,047 source-aligned rows gives
the following signal comparison against simulator body-forward speed. These
are not independent wheel-speed-truth errors:

| Signal | Bias (m/s) | RMSE (m/s) | p95 absolute error (m/s) |
| --- | ---: | ---: | ---: |
| Raw encoder-derived rear-pair mean | +0.272 | 0.380 | 0.680 |
| Production rolling raw wheel speed | +0.267 | 0.379 | 0.684 |
| Mapped wheel speed | −0.055 | 0.225 | 0.470 |
| Pre-wheel-update speed prediction | −0.018 | 0.165 | 0.307 |
| Final odometry speed | −0.022 | 0.174 | 0.344 |

At 8 m/s and above, the encoder-pair mean is +0.453 m/s biased with 0.508 m/s
RMSE; at absolute steering ≥0.4 rad it is +0.187 m/s biased with 0.215 m/s
RMSE. The observer is in turn mode on 74.7% of samples, uses wheel updates on
90.0%, and rejects burst samples on 7.8%. The production mapped input and
pre-wheel prediction are already much closer to body speed than raw encoder
rate. This indicates the existing observer is suppressing a substantial
encoder/body mismatch. It does not establish whether the raw discrepancy is
physical tire slip, encoder timing/conversion, or both, and a single training
capture is not enough to retune the observer.

The regime breakdown is in
`live_runs/derived_dynamics_learning_20260928/odom_wheel_signal_correction_20261011/production_replay_training_20261012/train_r01/production_odom_score_trace_by_regime_20261012.json`.

The highest-speed error is a concrete odometry diagnosis from one existing
capture, not evidence of a general fix. A single-run result is insufficient to
change production odometry. The corrected report is
`live_runs/derived_dynamics_learning_20260928/odom_wheel_signal_correction_20261011/production_replay_training_20261012/train_r01/production_odom_score_by_phase_20261012.json`.
The source bag SHA256 is
`e105007781331f6283f710284356baf071eee0516556c3841f9029f753e12bc5`.

## Code and artifacts created during this work

- `tools/racing/offline/controller/evaluate_mpc_deployable_yaw_features.py`
- `tools/racing/offline/controller/evaluate_mpc_deployable_yaw_recursive.py`
- `tools/racing/offline/controller/evaluate_mpc_deployable_yaw_damping.py`
- `tools/racing/offline/controller/fit_stable_ar_yaw_response.py`
- `tools/racing/offline/controller/train_recursive_yaw_mlp.py` (fixed-pole and bounded adaptive-pole modes)
- `tools/vehicle_dynamics_learning/score_replayed_production_odom.py` now
  reports pose error per valid phase rather than incorrectly aligning across
  reset-separated phases.

No candidate has been integrated into the runtime stack. No new simulation
data has been collected.

## Matched practice A/B: coherent-packet pose blend — 2026-10-12

The first live `0.5` and `1.0` overlay attempts both collided. They were
initially treated as candidate evidence, but their manifests show an important
build mismatch: those runs used `SDU_APEX_BUILD_MPC=0`, whereas the known-good
R02 reference had `build_mpc=1`. Those collision bags remain preserved as
diagnostics and are **not** evidence against either blend value. In both,
packet intervals stayed near 25 ms and no source-order reversal occurred
before the car began leaving the line; the broad failure-atlas label
`ODOM_WHEEL_BURST` was not the initiating event.

I reran the comparison using the exact practice map and unchanged reference
trajectory, pinned practice simulator, current MPC and localization builds,
and only changed `coherent_packet_pose_blend`. Both new runs pre-armed the
collision watcher before the simulator connected and confirmed collision
count zero before motion. The `0.5` candidate ran 10 scored laps with zero
collisions (the collision-only watcher allowed several extra laps after the
standard protocol); the matched `1.0` control ran exactly one warmup, ten
scored laps, and one extra lap with zero collisions. Both manifests record
`build_mpc=1`, `build_localization=1`, identical map, trajectory, simulator,
controller image, MPC config, and path-tracking config.

| Metric | `0.5` candidate | `1.0` control |
| --- | ---: | ---: |
| Scored laps / collisions | 10 / 0 | 10 / 0 |
| Mean / best lap | 6.0937 / 6.0257 s | 6.1172 / 6.0617 s |
| Absolute CTE p95 | 0.0921 m | 0.0895 m |
| Map-normal error p95 | 0.0440 m | 0.0434 m |
| Map-tangential error p95 | 0.1645 m | 0.1642 m |
| MPC accepted / rejected | 4097 / 6 | 3035 / 6 |
| One-step speed-error p95 | 0.1777 m/s | 0.1810 m/s |
| One-step yaw-rate-error p95 | 0.0956 rad/s | 0.0908 rad/s |

The lap-time difference is small and is only one independent run per setting;
it is encouraging, not a claim of a statistically established pace gain. AMCL,
CTE, and MPC errors are effectively unchanged at this resolution. The
important targeted result is the same-bag offline A/B: I replayed the exact
same encoder/IMU capture twice, changing only the pose blend, then compared
each reconstructed Odom trajectory with simulator truth after one rigid pose
alignment at the start of each scored lap. Mean over the ten scored laps:

| Relative `/odom` position metric | Blend `1.0` | Blend `0.5` | Change |
| --- | ---: | ---: | ---: |
| Per-lap position RMSE | 0.1210 m | 0.0991 m | −18.1% |
| Mean per-lap p95 position error | 0.1800 m | 0.1565 m | −13.1% |
| Per-lap endpoint error | 0.1541 m | 0.1346 m | −12.6% |

The paired replay holds the observed driving and sensor sequence fixed, so
this isolates the pose-integration parameter from closed-loop run variation.
The observer implementation confirms that the blend changes only the
longitudinal velocity used for pose integration when the current encoder
packet passes coherence gates; reported body `u/v/yaw-rate` is unchanged by
the blend. The one-step body-speed regression seen between the two independent
live runs is therefore a trajectory/run-condition difference, not a
parameter-induced estimator regression.

This initially supported promoting the bounded `0.5` pose blend. A later
offline comparison found that using the observer body-speed state alone was
better on the additional capture; that result supersedes the `0.5` default.
The current config disables coherent-packet pose fusion, keeps packet
coherence checks in the observer, and does not change AMCL/MPC weights,
vehicle physics, reference trajectory, or competition topic access. The run bags and
all replay/report outputs are retained under:

- `live_runs/odom_pose_blend_050_current_mpc_20261012_r01/`
- `live_runs/odom_pose_blend_100_current_mpc_20261012_r01/`
- matched MPC/localization build provenance in each `run_manifest.json`

The first, MPC-build-mismatched attempts remain under
`odom_pose_blend_050_closedloop_20261012_r01/` and
`odom_pose_blend_100_current_control_20261012_r01/`; they are not used for
the promotion result. This correction improves relative Odom pose integration
on the tested practice path. It does not solve the remaining body-speed model
error or independently prove high-steer transfer; those are the next Odom
questions. No MPC model or raceline candidate has been promoted by this A/B.

## Odom-only follow-up: fresh capture and timing-grid hypothesis

The deployed `use_coherent_packet_velocity_for_pose: false` setting was
rechecked against another complete practice capture using an offline replay of
the production sensor-odometry node. The replay used encoder and IMU inputs;
simulator truth was used only for scoring. On 3,101 matched samples across 13
reset-separated sequences, reported twist was identical for the two pose
settings (`u` RMSE 0.09994 m/s, `v` RMSE 0.00934 m/s, gyro yaw-rate RMSE 0).
The per-sequence macro relative-position RMSE was 0.084745 m with packet-speed
pose fusion disabled, versus 0.094633 m with a 0.5 blend (10.4% lower). The
sample-weighted position RMSE was 0.12883 m versus 0.13973 m (7.8% lower).
Thus the setting changes pose integration, not the reported velocity
estimator. One sequence still reaches 0.5199 m p95 position error, so this is
a bounded improvement—not a high-accuracy or drift-free Odom result.

The fixed-25-ms hypothesis was also tested offline because the sensor capture
has receipt-time jitter. It was not promoted. Quantizing all observer timing
changed burst/recovery decisions and caused a substantial held-out braking
sequence regression; narrowing the fixed interval to pose/physics integration
only still worsened the 16-lap practice capture's macro pose RMSE from 0.08548
to 0.08671 m. The source, YAML, and replay-tool hooks for these temporary
fixed-grid variants have been removed. Production continues to use measured
source intervals.

Current Odom status: the pose-only correction has replicated on independent
captures without changing twist, but forward-speed RMSE remains about 0.10
m/s on this capture, and a braking/high-yaw sequence has approximately 0.52 m
p95 relative-position error. Gyro yaw rate is passed through rather than
predicted. The next Odom work should isolate the speed residual in that
sequence (including encoder burst/recovery and braking transitions) and test
one causal estimator change on held-out captures before promotion. No MPC,
controller, raceline, or simulator change is part of this follow-up.

## Strict future-command plant forecast — 2026-10-12

This directly tested whether an offline plant can forecast 750 ms from a known
initial physical state and then only the recorded steering/throttle command
stream. No simulator was launched and no future sensor, odometry, or truth
state was supplied to the predictor. Future ground truth was used only after
the rollout to score it.

Two six-lap practice captures (`practice_model_validation_r02` and `_r03`)
were exported from their active intervals only. Each active interval has lap
counter transitions 0–6, zero collisions, zero timing faults, and passing
40 Hz cadence checks. The full-bag manifest still records `experiment_aborted`;
the reason for that overall status was not established in this analysis, so it
was not hidden or relabeled. The targeted active-interval gate is what admitted
these six completed laps. The persistent
dataset contains 3,031 samples and 12 reset/lap sequences. It was produced
with `--practice-active-interval --practice-expected-laps 6`; this explicit
lap-count option fixes the exporter’s former hard-coded 12-lap assumption
without weakening the interval’s collision, timing, or cadence checks.

The tested model was the existing four-wheel grey-box checkpoint
`best_four_wheel_greybox.pt`, not production Odom or MPC. At forecast start it
receives simulator rigid-body state and pose, current steering and throttle
feedback, current IMU acceleration as its load-transfer initial condition,
and an inferred wheel/slip state (rear wheel rates from current encoder
channels; front wheel rates initialized kinematically). The forecast starts
at the end of a 1.975 s sequence prefix; that prefix is not fed as a time
history to the model. After initialization, only the recorded
steering/throttle commands are inputs. Thus this is a favorable
ground-truth-state initialization, with a causal instantaneous IMU value, and
no future sensor/truth leakage. It is not a deployable Odom estimate of the
initial state.

The command-only rollout fails badly by the requested horizon. Values below
are the macro mean of the six scored sequences in each independent capture;
position, heading, and speed are absolute errors at the stated endpoint, not
whole-window RMSE:

| Forecast horizon | R02 position | R02 heading | R02 speed | R03 position | R03 heading | R03 speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 ms | 0.0026 m | 0.0065 rad | 0.0092 m/s | 0.0024 m | 0.0066 rad | 0.0164 m/s |
| 100 ms | 0.0117 m | 0.0343 rad | 0.148 m/s | 0.0133 m | 0.0314 rad | 0.180 m/s |
| 250 ms | 0.0854 m | 0.0730 rad | 0.498 m/s | 0.0930 m | 0.0823 rad | 0.558 m/s |
| 500 ms | 0.384 m | 0.343 rad | 1.316 m/s | 0.368 m | 0.181 rad | 1.225 m/s |
| 750 ms | **0.888 m** | **0.436 rad** | **2.384 m/s** | **0.833 m** | **0.314 rad** | **2.005 m/s** |

The model’s error begins in yaw response during turns, then body-speed error
grows quickly: across the individual sequences the first 25 ms yaw-rate
absolute error is roughly 0.44–0.66 rad/s, and by 50 ms many sequences have
already accumulated about 0.15–0.16 m/s speed error. This is not a small
integration-only drift. The complete free runs diverge by roughly 9.5–10.2 m
at their endpoints, so this candidate is rejected for open-loop plant/MPC use.

For context, an earlier EDSSM command-only replay on the same two practice
captures did substantially better over its first 750 ms: per-run position
radial RMSE was 0.0174/0.0169 m, 750 ms endpoint position error
0.0351/0.0297 m, forward-speed RMSE 0.0373/0.0349 m/s, and yaw-rate RMSE
0.0615/0.0604 rad/s. That model starts from an observed 80-frame history
(1.975 s) plus state and pose, then consumes commands only. It is therefore
not the same single-initial-state test above. Its full-capture rollout still
drifted substantially: radial position RMSE 1.17/2.19 m and endpoint error
1.79/3.11 m. In addition, the old dataset loader decompressed sealed test and
final-test arrays before filtering by run ID; that archive-level access breach
makes the old report exploratory, not clean sealed-set evidence. It is a
promising short-horizon comparator, not a perfect or promoted plant.

This test does **not** show that every offline model fails, and it does not
invalidate the separate production-Odom replay improvement documented above.
That improvement changes pose integration only; the observer still receives
new sensors every 25 ms and has not been shown to predict 750 ms open-loop.
No model or parameter from this failed rollout was integrated into Odom or
MPC, and no runtime physics, MPC weights, or raceline was changed.

Artifacts:

- Export manifest and compact dataset:
  [`practice_two_run_dataset`](../../live_runs/derived_dynamics_learning_20260928/command_only_750ms_rollout_20261012/practice_two_run_dataset/manifest.json)
- 25 ms–5 s free-run score:
  [`four_wheel_greybox_practice_unseen_first_steps.json`](../../live_runs/derived_dynamics_learning_20260928/command_only_750ms_rollout_20261012/four_wheel_greybox_practice_unseen_first_steps.json)
- Checkpoint:
  `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/greybox_highspeed_steering_h0p5_20261002/best_four_wheel_greybox.pt`

Verification: focused dataset-preparation unit tests pass (9/9), both edited
Python modules parse, and `git diff --check` passes. No compile, full build,
simulator, or MPC test was run.

## Odom-only follow-up: bounded lateral IMU fusion screen — 2026-10-12

I screened whether a complementary lateral-velocity update can improve the
current sensor-only kinematic estimate without allowing unbounded IMU
integration. At each sample the candidate used only current/past sensor
signals:

```text
v_pred = v_previous + dt * (ay_rear - yaw_rate * u_wheel)
v_hat  = (1 - alpha) * v_kinematic + alpha * v_pred
```

`ay_rear` removes the IMU-to-rear-axle angular-acceleration term using the
causal gyro difference. `u_wheel` was the average measured rear-wheel surface
speed with the production speed-scale curve; no simulator truth entered the
estimate. Truth was used only to score. The screen approximates the production
turn-mode gate and is therefore a candidate screen, not an exact C++ node
replay.

The blend was selected only on two clean high-steering training captures.
Their baseline rear-lateral-velocity RMSE was 0.00524 and 0.00531 m/s. The
best tested blend was only marginally better (`alpha=0.2` on one and `0.1` on
the other). On the two independent six-lap practice validation captures,
`alpha=0` (the existing kinematic estimate) was best on both:

| Practice capture | Existing estimate | `alpha=0.1` | `alpha=0.2` | `alpha=1.0` |
| --- | ---: | ---: | ---: | ---: |
| R02 `v` RMSE | 0.01657 m/s | 0.02060 m/s | 0.02610 m/s | 2.741 m/s |
| R03 `v` RMSE | 0.01599 m/s | 0.01980 m/s | 0.02500 m/s | 2.611 m/s |

This rejects lateral-IMU blending as a production improvement on the current
evidence. Full integration has severe accumulated drift; even the bounded
blend transfers in the wrong direction on both practice captures. The C++
observer, YAML, and runtime behavior were not changed by this screen.

An independent timing check on those practice captures also found that using
the current IMU acceleration sample gives better interval-acceleration
agreement than averaging current and previous samples. For R02/R03, current
sample versus trapezoid-average RMSE was 0.337/0.338 versus 0.506/0.502 m/s²
longitudinally, and 0.234/0.242 versus 0.353/0.356 m/s² laterally. This does
not establish perfect IMU timing; it rejects the proposed extra smoothing for
these captures.

**Updated Odom status:** the last saved production-node practice replay gives approximately
0.080–0.086 m/s forward-speed RMSE, 0.0098–0.0101 m/s rear-lateral-velocity
RMSE, and near-zero yaw-rate residual because Odom publishes the gyro yaw rate
directly. These saved results are not a fresh replay of every uncommitted
worktree edit, and are not proof of accuracy in every speed/steering regime.
The previously documented pose-integration gain
is limited to pose; the lateral fusion experiment produced no accepted
correction. The best next Odom target remains the causal longitudinal-speed
residual during wheel/vehicle mismatch and braking. The strict 750 ms
command-only plant forecast remains a separate failed task; the production
Odom estimator is updated from sensors every sample and is not itself a
750 ms open-loop predictor.

No simulator, MPC, compile, or test/final-test run was started for this
screen. The combined archive containing sealed test/final-test runs was not
opened; only the clean two-run training dataset and clean two-run practice
validation dataset were read. A PAYDAY2 process was observed using high CPU at
the time of the machine check; no simulator, ROS bag replay, or MPC test
process was active.

## Odom continuation: coherent-burst recovery and positive wheel innovation — 2026-10-12

The earlier speed-residual replay isolated a state-selection bug in both
straight and turn recovery paths: `coherent_burst_recovery` required the
rolling encoder-window estimate to be coherent, but the accepted update still
selected the single-packet derivative. The observer now uses the rolling
estimate for both `window_wheel_recovery` and `coherent_burst_recovery`.
This is a source-path correction, not a wheel-radius or vehicle-physics
change.

A second, narrowly gated update was added for a positive wheel/body innovation
during an established turn: only when pre-update Odom speed is at least
7 m/s, current measured longitudinal IMU acceleration is positive, and the
wheel update exceeds the causal rigid-body IMU prediction. The accepted
estimate is then 87.5% IMU prediction and 12.5% wheel update. It uses only the
existing encoder and IMU inputs; simulator truth is an offline score label.
The positive-acceleration gate is important: the first version without it
reduced practice speed error but worsened reset-local position on the 9.5 m/s
throttle-reduction phases. That broad version was not retained.

The gated candidate was selected on the existing 64 valid randomized
swerve/throttle phases, then recursively replayed against both independent
six-lap practice captures. The comparison below is to the same current-worktree
observer with the coherent-recovery fix but without the new wheel/IMU blend:

| Capture | Forward-speed RMSE, recovery-only → gated | Reset-local position RMSE, recovery-only → gated | High wheel-mismatch/high-throttle `u` RMSE |
| --- | ---: | ---: | ---: |
| Training excitation, 6,047 samples / 64 phases | 0.14343 → 0.12792 m/s | 0.08099 → 0.08034 m phase-macro | — |
| Practice R02, 1,344 samples | 0.07248 → 0.06684 m/s | 0.02220 → 0.02071 m per ~0.775 s window | 0.19372 → 0.03204 m/s (18 samples) |
| Practice R03, 1,312 samples | 0.07382 → 0.07083 m/s | 0.02325 → 0.02294 m per ~0.775 s window | 0.12187 → 0.06030 m/s (15 samples) |

On the training excitation, speed RMSE also falls from 0.13970 to 0.12622
m/s at 6–8 m/s and from 0.16641 to 0.13534 m/s above 8 m/s. Phase-level
analysis localizes the strongest speed gains to throttle-increase turn phases
around 6.5–7.5 m/s. The no-acceleration-gate variant regressed position in
most 9.5 m/s throttle-reduction phases, which motivated the gate. The current
gate removes most of that path regression; residual phase errors are still
largest around 9.5 m/s throttle reduction and 4.5 m/s / 0.42 rad turn
transients.

These are real improvements to the sensor-only Odom replay, not proof of a
perfect observer. The practice position numbers are reset-local short-window
scores, not full six-lap absolute Odom drift. `v` remains about 0.0098–0.0101
m/s RMSE on practice; yaw-rate is copied from gyro, so its near-zero residual
does not independently validate a yaw predictor. The full command-only
750 ms plant remains inaccurate and has not been repaired by these Odom
changes.

The prior 58-run sensor-only linear residual model was also checked against
this changed observer. It is version-stale: applying it as an offline
one-step correction raises R02/R03 speed RMSE to 1.538/1.503 m/s. It is not
integrated. A new run-level fit is in progress from the clean
`attitude_dataset_20260929_clean` manifest: exactly 58 existing training bags
are selected before any bag opens, and only left/right encoders plus IMU are
replayed into the updated observer. The manifest's validation, test, and
final-test run bags are not part of this fit. The purpose is to determine
whether refitting a sensor-only correction on the changed observer source
helps; it will be scored on the same practice captures before any C++
integration is considered.

Current artifacts:

- Training-only excitation baseline: [`recovery-fix score`](../../live_runs/derived_dynamics_learning_20260928/current_worktree_odom_replay_20261012/coherent_recovery_fix_train_r01/score.json)
- Positive-acceleration-gated training replay: [`training score`](../../live_runs/derived_dynamics_learning_20260928/current_worktree_odom_replay_20261012/turn_imu_wheel_fusion_7ms_positive_ax_train_r01/score.json)
- Practice R02/R03: [`R02 score`](../../live_runs/derived_dynamics_learning_20260928/current_worktree_odom_replay_20261012/turn_imu_wheel_fusion_7ms_positive_ax_r02/score.json), [`R03 score`](../../live_runs/derived_dynamics_learning_20260928/current_worktree_odom_replay_20261012/turn_imu_wheel_fusion_7ms_positive_ax_r03/score.json)
- New source-matched residual retrain artifacts are being written under `live_runs/derived_dynamics_learning_20260928/odom_u_residual_retrain_current_observer_20261012/fit/`.

No simulator or MPC was run. The production physics, MPC, AMCL, raceline, and
topic subscriptions were not changed. The observer source now contains the
recovery correction and gated update; it has been rebuilt by the real
recorded-bag replay procedure. The competition container itself has not yet
been rebuilt or validated in a closed-loop run.

## Continuation: request-time replay and AMCL startup correction — 2026-10-12

The previously identified request-time Odom gains were retained as the
baseline: on independent R02/R03 practice captures, forward-speed RMSE fell
from `0.10169/0.10124` to `0.08545/0.07887 m/s`, and reset-local 0.775 s
position RMSE fell from `0.03425/0.03473` to `0.02331/0.02606 m`. These are
short-window offline replays, not a full-lap predictor claim.

A guarded live bridge check was attempted with the established pinned
practice simulator procedure. It produced 807 sensor packets in about 20 s
(approximately 40 Hz), all with zero throttle and steering, and no collision.
It is **not** a driving/Odom validation: AMCL published no map pose, so MPC
kept neutral output and Odom remained at its reset origin. The run was
stopped by the no-progress watcher, and every container was shut down.

The AMCL rejection was traced to inconsistent startup gating. The saved
start anchor is `(0.8006, 3.1583, -1.5708)`; its nearest trajectory point is
0.184 m away. The confident, stable scan candidate was `(0.253, 3.201,
-1.455)`: 0.55 m from the start anchor and within its configured 1.20 rad
heading tolerance, but 0.690 m from the raceline, just beyond the 0.65 m
gate. In anchor-position startup mode, AMCL publishes the saved start pose
propagated by causal Odom rather than that raw scan-cluster translation. The
old gate nevertheless checked the raw cluster against the raceline, preventing
the designed anchor correction from publishing and leaving the controller
waiting indefinitely.

[`amcl_node.cpp`](../../f1tenth_localization/gpu_amcl_cpp/src/core/amcl_node.cpp)
now evaluates the raceline distance and heading gate against the
anchor-propagated pose when that startup mode is active. The scan still must
pass covariance, cluster-weight, start-anchor, heading, and five consecutive
stable-scan checks; outside that mode the original raw-pose raceline gate is
unchanged. This source correction has not yet been rebuilt or validated in a
live run.

The source-matched longitudinal residual fit completed on exactly the 58
clean training bags admitted by
`attitude_dataset_20260929_clean/manifest.json`; validation/test/final-test
bags were not opened. It sends only left/right encoders and IMU through the
current production Odom node, with simulator truth used only as an offline
label. The resulting one-step residual table contains 311,873 rows. A global
linear correction failed run-disjoint CV: median run RMSE increased from
0.0875 to 0.1911 m/s, with only 3/58 runs improved. The trainer uses NumPy
weighted Ridge and run-group folds; a parity check matched scikit-learn Ridge
to `1.1e-16` maximum prediction difference and confirmed run-disjoint folds.

The whole-run command-only predictor remains unsolved. Neither this residual
fit nor the startup-gate fix establishes full-state 750 ms/lap prediction or
perfect pose/speed. No simulator physics, MPC weights, raceline, or
competition topic subscriptions were changed.

## Continuation: current-source Odom fit and full-speed data audit — 2026-10-12

The active first-pass retraining job uses the 58 clean train captures in
`attitude_dataset_20260929_clean/manifest.json` (1.53 GiB of source bags). Its
metadata reaches 8.41 m/s longitudinal `u`; it does **not** independently
support a 0–12 m/s claim. This corrects an earlier note that mistakenly
attributed a 20.8 m/s capture from another manifest to this 58-run split.
At the latest check, 48/58 captures had replayed through the current rebuilt
Odom node; the job is still running, with training artifacts kept in the
workspace rather than temporary storage.

Before opening more bags, the race-domain manifest was audited using only its
run metadata. It admits four additional clean `train` rows not in the 58-run
set: two full-steering captures reaching 9.59 and 11.01 m/s, one high-throttle
capture reaching 20.80 m/s, and one practice capture. The same manifest has
separate clean 11.1 m/s validation captures; those are reserved for evaluation
and are not admitted to fitting. Its test/final-test bags are not opened.
Duplicate run IDs across the two training manifests have matching source path
and size; their derived-feature fingerprints differ, so identity is checked
against the source bag, not that view-specific fingerprint.

The trainer now supports merging clean train-only rows across manifests with
identity-conflict checks and an optional planar-speed ceiling. The expanded
fit will reuse hash-verified Odom replay caches, fit only samples at or below
12 m/s planar speed, and include the four disjoint clean train captures. This
captures the relevant 0–12 m/s portion of the long throttle run without
letting its >12 m/s samples steer the fit. A separate fit tool will compare
three input profiles with run-disjoint folds: the current seven Odom/sensor
features; those plus the permitted steering/throttle feedback sensors; and a
command-output feature reference that is explicitly non-deployable.

The competition technical guide lists steering and throttle feedback as
sensor inputs and steering/throttle command topics as actuator outputs
([official data-stream table](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#23-data-streams)).
Accordingly, a command-conditioned fit will not be integrated into Odom. A
feedback-conditioned fit may be considered only after held-out transfer and
the required compliant runtime wiring are verified.

The Unity-source instrumentation build was also attempted in batch mode. The
checkout confirms `Fixed Timestep: 0.001` s and the four-wheel collider setup
documented in `UNITY_WHEELCOLLIDER_PHYSX_SOURCE_AUDIT_20261012.md`, but the
build stops at C# compilation because this public source checkout contains
neither the `RGLUnityPlugin` nor `SocketIO` dependencies required by its LiDAR
and bridge scripts. No instrumented player was produced or run, and no sensor
or physics replacement was substituted. Resolving this dependency gap is
separate from the existing-bag Odom fit and is not being represented as a
validated simulator.

The residual under test is only a one-step correction to forward speed. It
does not estimate lateral speed, yaw, wheel spin, pose drift, or a free-running
future trajectory. A good score here would be a useful Odom component, not a
complete vehicle model. The current candidate must still beat the frozen
practice baselines on run-level `u` and reset-local integrated position while
not materially regressing `v` or yaw before any production integration.

Next evaluation order: complete the expanded 62-run, <=12 m/s train-only
replay/fit now in progress; evaluate the existing disabled high-yaw IMU
latent-velocity observer against the two extreme-turn training captures and
unchanged practice R02/R03; then consider the clean race-domain validation
R04/R05 only for a candidate selected without them. Report `u`, `v`, yaw rate,
and reset-local integrated position separately. No learned residual will be
wired into production without run-level transfer gains and no material
cross-state regression. A separate source-build telemetry effort is still
required for any command-only full-state simulator: ordinary ROS bags contain
no independent per-wheel RPM/contact-slip labels, and the public Unity build
must first pass same-command motion parity against the pinned simulator before
internal WheelCollider telemetry is trusted.

## Follow-up: residual gate failure and high-yaw physics observer A/B — 2026-10-12

The 58-run CV outlier reduction did not transfer. A train-only linear expert
was gated to samples where encoder-mean speed and Odom `u` differ by at least
5 m/s. It reduced run-macro CV RMSE from `0.701` to `0.197 m/s`, but left
median run RMSE unchanged and helped only 3/58 runs. On held-out practice R02
and R03 it activates on 96/1,344 and 88/1,312 scored samples, respectively,
and raises forward-speed RMSE from `0.0789` to `0.4014 m/s` and from `0.0708`
to `0.4066 m/s`. It is rejected and is not integrated.

Root cause of this false activation: during those practice samples, Odom is
already near truth at `5–7 m/s`, while the 100 ms encoder estimate is near
zero and IMU `a_x` is about `-7.1 m/s²`. The same instantaneous features appear
in extreme-turn training data with both near-zero and very large negative
truth-minus-Odom residuals. Thus the instantaneous mismatch is not a reliable
error indicator; it conflates encoder dropout with body-frame reverse/sliding
motion. Increasing linear-model complexity on these same instantaneous inputs
would not resolve that ambiguity.

The code already contains a disabled physics-based high-yaw observer: it
integrates a private signed body-velocity state using rigid-body equations and
causal IMU, then publishes its `u` only when wheel speed exceeds the integrated
state by a configured slip margin. An offline-only replay switch was added to
`tools/replay_sensor_odometry_bag.py`; no production setting has changed.

On extreme-turn training capture `train1`, its ≤12 m/s rows show `u` RMSE
falling from `15.84` to `15.20 m/s` overall and from `16.65` to `11.65 m/s`
for `|yaw rate| ≥ 2.5 rad/s`. Among 981 high-yaw rows where simulator-truth
body `u` is negative, sign accuracy rises from `0%` to `79.9%` and `u` RMSE
falls from `20.05` to `9.65 m/s`. This is a substantial high-yaw speed/sign
gain, but not a sufficiently accurate estimate.

The pose result is mixed and blocks promotion: across 504 valid phases,
phase-macro position RMSE changes `10.16 → 9.96 m`, but phase p95 changes
`38.82 → 39.69 m`, mean endpoint error slightly worsens `15.05 → 15.20 m`,
and position RMS restricted to high-yaw samples worsens `10.72 → 13.04 m`.
The candidate does not publish its latent lateral velocity, and per-phase
analysis shows sizeable pose regressions even where `u` RMSE improves. This
points to unresolved lateral motion / trajectory integration, not a reason to
ship the high-yaw `u` override alone. On practice R02/R03 the option changed
no state samples, so it neither helps nor regresses ordinary laps.

The paired replay on independent extreme-turn training capture `train2` is
complete. In the `|yaw rate| >= 2.5 rad/s` subset (`N=5,158`), `u` RMSE
improves `19.39 → 14.13 m/s`; for negative truth `u` at high yaw (`N=1,048`),
it improves `21.17 → 10.82 m/s`. Lateral-speed RMSE remains unchanged at
`2.40 m/s`. Across its 552 reset phases, phase-macro position RMSE changes
`12.45 → 12.19 m`; within high-yaw samples, sample-weighted position RMSE
worsens `13.91 → 14.93 m` while macro-per-phase RMSE improves `7.04 → 6.75 m`.
For high-yaw negative-`u` samples, position RMSE worsens `8.72 → 12.50 m`.
Thus `u` sign/magnitude gains replicate on the second capture, but do not
transfer to consistently better pose. This paired result rejects promotion
of the `u`-only variant and motivates one narrow candidate: publish the
already-integrated latent `v` only when the same high-yaw slip gate replaces
`u`, then compare state and reset-local position metrics on both captures.
The scorer now reports `0–12 m/s`, high-yaw, negative-`u`, and matched
relative-position subsets explicitly. Practice replay remains unchanged;
no simulator is running and production configuration is untouched.

## Physics-first state decomposition — 2026-10-12

The source audit confirms that motion need not be learned as one opaque
input/output map. The Unity vehicle already computes a causal state chain:
actual actuator feedback and rate-limited wheel steering; per-wheel motor and
brake torque; wheel angular speed; tire contact slip and normal load; tire
forces; rigid-body acceleration; then pose integration. The development-only
`OpenPlaneWheelDynamicsCapture` hook records these at the 1 ms physics step,
including four WheelCollider RPMs, actual wheel angles/torques, `WheelHit`
forward/sideways slip and contact force, body velocity/angular velocity/pose,
and contact-frame geometry. It is read-only and is invoked only in
development builds when `SDU_APEX_DYNAMICS_DIR` is set. The ring buffer holds
300 seconds at 1 kHz and reports overwritten samples. No such capture CSV is
present in `live_runs`; this source capability is not yet a data result.
The hook has now been extended to include each collider's configured wheel
mass alongside radius, damping, and suspension metadata; Unity exposes this
property in its [2022.3 WheelCollider reference](https://docs.unity3d.com/2022.3/Documentation/Manual/class-WheelCollider.html).
This improves the eventual spin-response identification inputs, but does not
expose the native solver's exact rotational inertia.

For planar body state `(u,v,r)` and wheel position `(x_i,y_i)`, the first
kinematic split is wheel-contact velocity
`(u-r*y_i, v+r*x_i)`, rotated by the actual wheel steer angle. Comparing its
longitudinal component to `R_i*omega_i` gives longitudinal slip; wheel-frame
lateral velocity gives slip angle. These are per-wheel quantities, not one
global speed/steering correction. The tire model then maps `(slip, normal
load)` to `(Fx_i,Fy_i)`. Summing forces and moments gives
`u_dot=sum(Fx)/m+r*v`, `v_dot=sum(Fy)/m-r*u`, and
`r_dot=sum(x_i*Fy_i-y_i*Fx_i)/I_z` in the planar approximation. Pose follows
`X_dot=u*cos(psi)-v*sin(psi)`, `Y_dot=u*sin(psi)+v*cos(psi)`,
`psi_dot=r`. Wheel spin follows motor torque minus brake torque, tire
reaction torque `R_i*Fx_i`, and wheel damping. The Unity public API exposes
the slip and contact load but not the native WheelCollider force solver or
all hidden wheel/suspension state; therefore this is a structured model to
identify, not an exact source-code reimplementation of PhysX.

The direct-versus-hidden state split is:

| Quantity | Development truth/capture | Compliant ROS runtime |
| --- | --- | --- |
| World pose and body `(u,v,r)` | Rigidbody position, rotation, velocity and angular velocity | No truth pose/velocity; IMU plus integrated odometry, with AMCL/LiDAR correcting global pose |
| Wheel spin `omega_i` | RPM for all four WheelColliders at 1 kHz | Cumulative left/right rear encoder angles at packet rate; front spin is hidden |
| Actual steering and motor/brake torque | Per-wheel steer angle and torque at 1 kHz | Steering/throttle feedback; torque reconstructed from the controller's piecewise map |
| Tire slip and normal load | Per-wheel `WheelHit` forward/side slip and normal contact force | Not reported; rear longitudinal slip can only be estimated from encoder speed minus estimated contact speed |
| Tire forces | Applied internally by WheelCollider, not returned as separate tangential-force components | Inferred from acceleration, wheel spin and a fitted force model |

This is why direct equations can anchor the model but cannot make the runtime
observer omniscient. Ground truth and 1 kHz capture are teacher signals only;
hidden runtime states must be estimated from permitted measurements and
carried causally.

The compliant runtime sensor set is poorer than that development teacher:
rear encoder angle changes, IMU acceleration/angular rate/attitude, and
actuator feedback. It does not expose four wheel RPMs or per-wheel contact
slip/load. Thus rear wheel circumferential speed is directly measurable on
valid encoder intervals, while actual tire slip is an estimate obtained by
subtracting predicted rear contact speed (including yaw and track width).
Front-wheel spin and individual tire forces remain latent. GT and WheelHit
telemetry may label and validate the offline plant, but must not become
runtime inputs to compliant Odom or MPC.

Actuator mapping is not an arbitrary fit: the audited F1TENTH prefab uses
Ackermann front steering with a steering-rate limit, common AWD motor torque,
and common four-wheel braking. In this controller, zero throttle sets motor
torque to zero and applies the configured brake torque to every wheel; it is
not free coasting. With the prefab's CAWB brake type, negative throttle goes
through the negative motor-torque path rather than either independent-brake
branch. Model identification must therefore use actual feedback and this
piecewise command-to-wheel-torque map. Exact numeric prefab values and the
public-Explore versus pinned-competition build caveat are recorded in
`UNITY_WHEELCOLLIDER_PHYSX_SOURCE_AUDIT_20261012.md`.

The Unity IMU source computes world-velocity finite difference, subtracts
`Physics.gravity` only when `GravityCompensation` is false, then rotates and
remaps into body axes. The public F1TENTH prefab sets the flag true, so its
published linear acceleration includes gravity projected into the body frame
when rolled/pitched. This is not automatically an odometry bug: raw inertial
acceleration is appropriate for integrating velocity when paired with the
body-frame rotation terms; tire-force inference instead needs gravity removed
in the current attitude. Treating those two uses as the same signal would
create a systematic error. A source-to-bag check on the existing raw-wheel
dataset's train and validation partitions used only fixed 25 ms rows
(639,105 with valid IMU attitude/sensor data; test/final-test were excluded).
ROS IMU horizontal acceleration versus the packet's copied Unity acceleration
differs by RMSE `[0.111, 0.043] m/s²` on train and `[0.040, 0.110] m/s²` on
validation, with near-zero bias. This confirms the bag channels preserve the
source acceleration convention to within packet alignment/sampling noise.
Under the standard body roll/pitch convention, the implied horizontal gravity
projection reaches p99 magnitudes of about `0.52 m/s²` forward and
`1.07 m/s²` lateral. Those terms matter for force inference, but are not an
excuse to subtract gravity blindly from the velocity-integrator input.

For a separate consistency check, finite-differencing packet-rate body
velocity and adding the body-frame transport term `omega × v` is noticeably
noisier than the Unity acceleration channel (fixed-grid RMSE on train is
`[0.49, 0.36, 0.20] m/s²`; validation `[0.94, 0.32, 0.13]`). This is why
25 ms GT velocity differences should not be treated as clean instantaneous
tire-force labels; a 1 kHz development capture is materially useful here.

The 62-capture train-only forward-speed residual is rejected: selected
regularization raises median-run RMSE `0.0927 → 0.6422 m/s`, with only 3/62
runs improving. The latent-lateral publication candidate is also not
promotable based on `train1`: high-yaw `u` RMSE improves `17.14 → 12.06 m/s`,
but high-yaw `v` RMSE worsens `2.56 → 6.58 m/s` (p95 absolute `4.88 → 16.85
m/s`); 0.775 s relative-position RMSE changes only `7.038 → 6.884 m`.
Independent `train2` confirms rejection: on `|r|≥2.5 rad/s`, `u` RMSE
improves `19.90 → 14.64 m/s`, while `v` RMSE worsens `2.40 → 7.32 m/s`
(p95 absolute `4.75 → 19.93 m/s`). Across all scored samples, `v` RMSE
worsens `0.92 → 2.86 m/s`; reset-local position RMSE changes only
`8.72 → 8.54 m` over median 0.775 s windows. The experiment-only option to
publish this latent `v` has been removed; the pre-existing `u` observer
remains disabled by default. Neither candidate changes production behavior.

The existing raw-wheel dataset also permits a direct rear longitudinal-slip
proxy on valid 25 ms rows: `R*omega_left/right - (u ∓ r*0.118 m)`, with
simulator-truth `(u,r)` used only as offline labels. Test/final-test partitions
were excluded. The existing raw-encoder-valid mask requires adjacent packet
sequence and a plausible source interval, then converts the angle difference
on the fixed 25 ms grid. The slip grouping conditions on measured throttle
feedback at the sample, not on command-to-feedback packet delay, so it does
not assume a 1/2/3-packet actuator response. To avoid letting the very long
throttle sweep dominate, the summary is run-macro over runs with at least 100
valid rows. In 0–4 m/s with
actual throttle feedback above 0.5, the median of per-run median absolute
slip proxy is `10.52 m/s` across six train runs (range `10.41–17.87`) and
`10.49 m/s` across four validation runs (range `10.37–11.00`), consistent
with repeatable wheel spin in those captures. At 4–8 m/s and actual zero
throttle, the corresponding run-macro values are `5.47 m/s` across seven
train runs (range `5.22–7.53`) and `5.68 m/s` across four validation runs
(range `5.40–7.53`); the signed pooled proxy is about `-6.8 m/s` in both
splits, consistent with the source's zero-throttle brake branch. These are
large, physically interpretable wheel/body mismatches, not direct `WheelHit`
slip ratios; they are matched-regime proxies, not a universal slip law. They
show that existing data already identifies why throttle and braking cannot
be modeled as a smooth scalar gain. No new simulation was needed.

Completed next checks: the bag IMU matches the Unity packet acceleration
within alignment noise; attitude-derived gravity projection is material to
force estimation; and the existing train/validation encoder data yield a
large throttle/braking-dependent rear slip proxy. Next, fit and validate the
separate actuator, wheel-spin, tire-force, and rigid-body blocks. A
command-only rollout must recursively carry those states; future measured
sensors and truth remain labels only. Per-wheel force-response identification
still needs the 1 kHz `WheelHit` capture, which does not yet exist in the data
folder. Do not infer full accuracy from one-step scores or from a 1 kHz
integrator alone.

After removal of the failed latent-`v` publication switch, the
`f1tenth_localization` package was rebuilt in the development image with
build/install/log outputs under container `/tmp`; the targeted build passed.
A first attempt failed before compilation because colcon tried to create its
log under the read-only mounted workspace; the retry's explicit `/tmp` log
base resolved that without writing build artifacts into the repository. The
Unity C# project remains unbuilt because its external bridge/LiDAR dependencies
are absent in this checkout.

## Exact pinned-player decode and causal model work — 2026-10-12

Static vehicle configuration has now been decoded from the exact competition
player image, including the active serialized `VehicleController` rather than
inferred from the public Explore prefab. The repeatable extraction is
`live_runs/unity_exact_binary_audit_20261012/tools/extract_exact_vehicle_config.py`;
its retained result and hashes are
`live_runs/unity_exact_binary_audit_20261012/exact_vehicle_controller_config.json`.
The pinned controller selects autonomous front Ackermann steering, CAWD, and
CAWB. It has throttle limit 1, motor-torque field 428 N·m, 30° steering limit,
and 183.345993°/s steering-rate limit. For nonzero throttle, CAWD distributes
the total controller torque equally to four wheels (107 N·m/wheel at full
command); at exactly zero it applies 428 N·m brake torque to each wheel.
Negative throttle is negative motor torque in the active branch, not a
separate brake input. The 32.5 mm controller radius is unused by CAWD; the
59 mm collider radius is separate and is the geometry used by encoder code.
Exact Unity-local COM and wheel positions are in
`UNITY_SIMULATOR_ASSET_AUDIT.md`.

The public prefab defaults are 85.6 N·m/manual, but the active public Explore
scene overrides `MotorTorque=428` and `DrivingMode=1`, exactly matching the
pinned controller. Thus the prefab defaults alone do not invalidate
source-built Explore runs; each capture still needs its scene/player
provenance confirmed. A persistent development-player project now exists at
`live_runs/unity_exact_binary_audit_20261012/source_project_pinned_config/`.
The copied prefab and scene are byte-identical to the public source, with no
vehicle physics or serialized configuration changed. The scene overrides are
recorded in `pinned_config_overlay_manifest.json`. The read-only wheel
recorder is development-only and environment-gated. The project has not yet
been built or run, so no improvement in model accuracy is claimed.

The pinned build's extracted settings also show a 1 ms physics step, separate
from 25 ms sensor/bridge samples. This sets the appropriate substep scale for
an offline teacher; smaller integration steps alone do not recover hidden
wheel spin, tire slip, load transfer, or native tire forces.

The causal state sequence to implement and validate is:

1. **Actuation:** use the exact rate-limited steering/Ackermann map and the
   piecewise throttle-to-wheel torque/brake map. In diagnosis, observed
   actuator feedback can condition the next state; in recursive prediction,
   actuator state must be generated from commands and its own prediction.
2. **Wheel-frame kinematics:** in the guide's body axes (forward `x`, left
   `y`, up `z`), wheel-point velocity is
   `vx_body_i=u-r*y_i`, `vy_body_i=v+r*x_i`. Rotate by each actual steer angle
   to obtain the local longitudinal/lateral velocity, then compare
   longitudinal surface speed `R*omega_i` and lateral slip. Sign is not yet
   reconciled: Unity documents `WheelHit.forwardSlip` as negative in
   acceleration, while the guide's displayed formula can be positive when
   `R*omega_i > vx`; wheel-RPM orientation and API conventions must be checked
   with a straight-drive/brake capture before using signed slip. The guide's
   normalized slip equations are also singular near zero longitudinal speed;
   that region needs a tested representation, not blind division.
3. **Wheel/suspension state:** evolve wheel spin from motor torque, brake
   torque, tire reaction torque and damping. Estimate dynamic normal loads
   from sprung/unsprung suspension motion and body attitude/acceleration.
   Wheel mass/radius and spring/damper settings are known, while effective
   rotational inertia and per-wheel runtime load remain unobserved.
4. **Tires and rigid body:** apply separate longitudinal/lateral spline
   responses, transform/sum wheel forces and moments about the COM, and
   integrate Newton-Euler body motion plus world pose. Fit unknown force/load
   scaling and coupled effects; do not assume curve ordinates are a constant
   friction coefficient.
5. **Sensor-only deployment:** infer hidden wheel/slip/load states causally
   from permitted rear encoders, IMU, and actuator feedback. Ground truth and
   development-only WheelHit values are training labels, never runtime input.

The Unity guide supplies the structure and slip definitions, while the exact
WheelCollider solver remains native. Next: compile this one development
build; run a matched command trace against the pinned player; compare public
pose, velocity and 40 Hz sensor outputs against repeated-run variation. Admit
hidden wheel/slip/load telemetry only if that A/B passes. Then fit causal
blocks on training data and score complete free rollouts on held-out
validation/practice runs, reporting forward/lateral velocity, yaw, wheel
spin, position and orientation separately. No source build, causal-model
candidate, or accuracy gain is yet validated.
