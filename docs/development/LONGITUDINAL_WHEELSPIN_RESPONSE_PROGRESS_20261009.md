# Longitudinal and rear-wheel response — 2026-10-09

## Scope

This phase starts with forward body motion (`u` / effective longitudinal
acceleration `a_x`) and rear-wheel spin response. It uses existing whole-run
captures and already frozen training/validation reports. The later `v` / `a_y`
and coupled-transition phase is intentionally not started here. Simulator truth
is an offline label only; any future rollout must use its own predicted state,
past sensor/actuator history, and commanded input. No future truth or feedback
may enter a prediction.

No simulator was launched, no bag or checkpoint was changed, and no model was
integrated into odometry or MPC. Test/final-test data were not used for fitting
or candidate scoring. One accidental aggregate support query did read summary
statistics spanning those partitions; that access is recorded below, so those
partitions must not be described as untouched blind evidence.

## Existing whole-run evidence

### One-step transition baseline

The frozen parent analysis
[`first_transition_attribution_wp14_parent_v1.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/encoder_raw_state_teacher_v1/first_transition_attribution_wp14_parent_v1.json)
scores the next 25 ms transition after a prior 2 s context. It uses seven
independent whole-run OpenPlane validation captures (71,564 transitions),
with two separate practice captures for transfer diagnostics. No test or
final-test run was used.

| One-step target | Seven OpenPlane validation runs | Two practice diagnostics |
|---|---:|---:|
| Forward `u` | 0.0348 m/s RMSE; −0.0083 m/s bias | 0.0460 m/s RMSE; −0.0137 m/s bias |
| Rear-wheel pair, filtered 100 ms target | 0.3179 m/s RMSE; +0.2100 m/s bias | 0.4045 m/s RMSE; +0.2565 m/s bias |

This is a teacher-forced local-transition score, not a free-running simulator.
It shows the body-speed step is predicted much better than the wheel state;
the wheel pair is also biased high. The validation sample is broad, but its
wheel output uses the stored smoothed wheel-rate target, not the instantaneous
fixed-25 ms encoder rate.

### The wheel target changes the apparent error substantially

The frozen same-transition comparison
[`wp16_frozen_parent_one_step_v3.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp16_frozen_parent_one_step_v3.json)
compares the two rear-wheel targets on identical transition masks. It scores
seven independent OpenPlane validation runs and two practice captures; the
checkpoint is unchanged and no fitting occurred in this evaluation.

| Wheel-rate target | OpenPlane validation macro-run RMSE | Practice macro-run RMSE |
|---|---:|---:|
| Fixed-25 ms encoder increment | 0.7572 m/s | 1.0012 m/s |
| Stored trailing ~100 ms rate | 0.3106 m/s | 0.4046 m/s |

These values do not establish that the encoder is wrong or that the smoother
is truth. The raw target exposes much larger packet-to-packet variation while
the trailing target suppresses it and adds delay. They must remain separate
measurement targets; neither should be renamed “tire slip.” The raw-rate
comparison has a wide run-cluster interval on OpenPlane (0.449–1.284 m/s),
and just two independent practice runs, so confidence about its error scale is
limited.

### Where the frozen parent first-step errors concentrate

The seven-run stratification in the same WP14 report is descriptive, with
fixed reporting bins rather than inferred physical regimes:

| Validation condition | Starts | Forward-speed RMSE | Rear-wheel-pair RMSE |
|---|---:|---:|---:|
| `|wheel/body mismatch|` 0–0.10 m/s | 9,763 | 0.0430 m/s | 0.220 m/s |
| 0.50–1.00 m/s | 13,437 | 0.0376 m/s | 0.315 m/s |
| 1–2 m/s | 951 | 0.0674 m/s | 0.754 m/s |
| 2–4 m/s | 1,714 | 0.0438 m/s | 0.573 m/s |
| 4–20 m/s | 3,053 | 0.0350 m/s | 0.615 m/s |
| Absolute throttle slew 0–0.25/s | 70,682 | 0.0341 m/s | 0.308 m/s |
| Absolute throttle slew 8–22/s | 202 | 0.1053 m/s | 1.074 m/s |

Large existing wheel/body mismatch and rapid throttle changes coincide with
much larger wheel-pair prediction errors. The fastest-slew sample is sparse
relative to the full validation set, and these are associations in scored
transitions—not causal proof that throttle slew alone creates wheelspin. `u`
does not show the same monotonic error growth with mismatch. This reinforces
separating body acceleration from wheel rotation instead of treating rear
wheel speed as a proxy for `u`.

### What existing longitudinal revisions did

There is some positive, but conflicting, held-out evidence:

- The frozen rigid-acceleration one-step candidate reduced `a_x` RMSE during
  the pre-existing throttle-pickup region by 0.344 m/s² versus its matched
  control (six validation captures; run-cluster 95% interval for candidate
  minus control: −0.642 to −0.050 m/s²). Across all validation starts, its
  `a_x` difference was effectively zero (+0.0009 m/s² RMSE, interval −0.037
  to +0.049). On large wheel/body-mismatch starts it was numerically worse
  (+0.104 m/s², interval −0.013 to +0.244). It has no wheel-spin output, so
  this is a localized body-acceleration result, not a joint response model.
  Source:
  [`rigid_acceleration_one_step_regions_20261004.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/rigid_acceleration_one_step_regions_20261004.json).
- The existing `axwheel_best` recursive candidate reduced full-capture `u`
  RMSE by 8–9% and rear-wheel-pair RMSE by 18–19% on each of two unseen
  practice captures. Position RMSE improved on one capture (2.006→1.866 m)
  and worsened on the other (2.821→2.951 m). This is a promising comparator
  for the new phase, but two runs do not establish general transfer or an
  accurate offline simulator. No checkpoint-selection use was recorded.
  Source:
  [`full_practice_replay_v11_axwheel_best_vs_last/full_replay.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/encoder_raw_state_teacher_v1/full_practice_replay_v11_axwheel_best_vs_last/full_replay.json).
- A separate six-run open-plane recursive comparison improved its `u` score
  but worsened wheel-state and position scores; it is not a globally better
  plant. Source:
  [`longitudinal_revision_validation_v1`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/encoder_raw_state_teacher_v1/longitudinal_revision_validation_v1/).
- The paired throttle-response lookup predicts a localized early
  step-versus-ramp wheel/body mismatch and slip-proxy effect, but its
  integrated longitudinal-acceleration result did not demonstrate a gain.
  It predicts an intervention difference, not absolute `a_x` or wheel speed.
  Source:
  [`THROTTLE_RESPONSE_DATA_REUSE_AND_RACE_DOMAIN_TEST_20261007.md`](THROTTLE_RESPONSE_DATA_REUSE_AND_RACE_DOMAIN_TEST_20261007.md).

## Current conclusion

Existing data already show that (1) one-step `u` is relatively predictable,
(2) one-step wheel rotation is substantially harder and depends strongly on
whether the target is a noisy 25 ms rate or a smoothed 100 ms rate, and (3) a
localized throttle-pickup acceleration gain exists in one candidate. They do
not yet give a single response that is both accurate for `u`/`a_x` and rear
wheel spin across held-out whole runs, nor do they justify production
integration. The `axwheel_best` checkpoint is retained as a candidate
comparator, not promoted.

## Next work in this phase

1. Reconcile acceleration and wheel target semantics on the same contiguous
   25 ms transitions: packet-aligned simulator body `a_x`, `u`-difference
   effective acceleration, fixed-25 ms encoder rate, and the trailing 100 ms
   rate. Keep the two wheel-rate targets distinct and quantify their relation
   to the kinematic rear contact-speed residual (a proxy, not tire force).
2. On training captures only, test whether prior wheel/body mismatch,
   throttle command/feedback change, and wheel-rate history explain held-out
   `a_x` and common/differential rear-wheel acceleration beyond current
   state/commands. Score every output per independent validation capture,
   with speed, steering, throttle, and mismatch breakdowns; do not choose
   arbitrary bins as physical regimes.
3. Compare only the targeted transition candidate, persistence/kinematic
   baselines, and the retained `axwheel_best` comparator. If the `u`/wheel
   gains do not transfer across whole runs and practice, stop that candidate
   and report the specific unresolved response. Do not integrate it into
   odometry/MPC yet.
4. After this phase has a defensible result, begin `v`/`a_y`, then evaluate
   the coupled longitudinal/lateral transition. No simulator was launched in
   the first pass. The practice-transfer audit below shows that the two
   available independent practice captures do not cover the aggressive speed /
   steering envelope; a targeted new capture will be needed before claiming
   transfer over that envelope.

## Continuation — matched model and one-step audits (2026-10-09)

No source bags, datasets, or checkpoints were changed. All comparisons below
use frozen training outputs; practice captures are scored only, and test/final
splits remain sealed. No simulator was launched and no production odometry/MPC
model was modified.

### Same-start recursive transfer comparison

The existing 10-second raw-encoder EDSSM plant, its 5-second parent, both
wheel-head fine-tunes, two longitudinal-head fine-tunes, and the 10-second
pose-weighted variant were run on the same four frozen 5-second windows from
the two unseen practice captures. These are 2 independent runs, not 4
independent samples. The comparison includes constant-initial-wheel-speed
persistence as a wheel baseline.

| Frozen model | 5 s position RMSE (m) | forward-speed RMSE (m/s) | yaw-rate RMSE (rad/s) | raw rear-wheel pair RMSE (m/s) |
|---|---:|---:|---:|---:|
| 5 s parent | 0.232 | 0.412 | 0.182 | 0.863 |
| 10 s plant | 0.515 | 0.282 | 0.133 | 0.723 |
| Wheel-head only | 0.355 | 0.360 | 0.220 | 1.214 |
| Wheel-head + body distillation | 0.263 | 0.352 | 0.254 | **0.606** |
| 10 s pose-weighted | 0.402 | 0.333 | 0.202 | 1.083 |
| Constant initial wheel-speed persistence | — | — | — | 1.841 |

The wheel-head/body-distilled candidate cuts the raw encoder pair error by
about 30% relative to the 5-second parent and by about 67% relative to
persistence on these windows. It is not a complete model improvement: its
5-second yaw-rate error is about 40% worse than the parent, and its position
error is about 13% worse. The non-distilled wheel-head candidate is worse than
the parent for the wheel pair. No candidate dominates all state outputs, so
none is promoted.

The two `longitudinal_distilled_*` checkpoints have exactly identical model
weights to the 5-second parent, despite 1,200 reported training steps. Their
validation selection retained the warm start. Their practice scores are
therefore not new model results and they are excluded from the comparison
table above.

### Matched one-step (u/a_x) and wheel response

To distinguish local transition quality from recursive drift, a new audit
scores every raw-encoder-valid 25 ms transition with the preceding 80-frame
history. It evaluates 2,058 transitions from the same 2 practice runs. The
speed distribution is 3.06 / 3.93 / 6.35 m/s at the 5th / 50th / 95th
percentiles; absolute physical steering is 0.015 / 0.149 / 0.270 rad. There
are no samples above 0.30 rad, above 6.35 m/s at the 95th percentile, or in
the 8–12 m/s range. This is not validation of the full requested domain.

| One-step predictor | (u) RMSE (m/s) | (v) RMSE (m/s) | yaw-rate RMSE (rad/s) | effective (a_x) RMSE (m/s²) | raw-wheel pair RMSE (m/s) |
|---|---:|---:|---:|---:|---:|
| Fixed-40-Hz HGB candidate | **0.0113** | **0.0034** | **0.0260** | **0.497** (midpoint-label head; 0.453 inferred from its (u)) | **0.396** |
| EDSSM 5 s parent | 0.0539 | 0.0595 | 0.0517 | 2.156 | 0.854 |
| EDSSM wheel-head/body-distilled | 0.0428 | 0.0250 | 0.0594 | 1.712 | 0.799 |
| EDSSM 10 s plant | 0.0483 | 0.0367 | 0.0845 | 1.934 | 0.791 |

For reference, one-step persistence errors on these same rows are 0.0725 m/s
for `u`, 0.0106 m/s for `v`, and 0.0782 rad/s for yaw rate. The zero-
acceleration (a_x) baseline is 2.92 m/s²; raw wheel-speed persistence is
0.867 m/s. The HGB transition candidate therefore has a strong local
one-step result in this low-to-moderate practice domain. The table's HGB
direct (a_x) head uses the midpoint transport convention used in its
training labels; the inferred-from-(u) value uses the same current-state
transport convention as the EDSSM comparison. They should not be conflated.

The recurrent models' one-step wheel-acceleration targets remain poor
(roughly 32–34 m/s² RMSE against the fixed-25-ms finite difference), only
slightly better than zero acceleration. This derivative is especially noisy
because the raw encoder signal is itself a rate inferred from an angle
difference; it is not direct tire angular velocity or torque. State-speed
prediction is the more meaningful wheel target here.

### What the two time scales establish—and do not

The HGB candidate is a much better one-step transition on these two practice
runs, but the exact-start recursive comparison already shows it is not a
usable long-horizon plant: at 5 s its position component RMSE is about 9.86 m
and its forward-speed RMSE about 0.97 m/s, versus 0.515 m and 0.282 m/s for
the 10-second EDSSM. The EDSSM is much more stable over this short low-steer
practice sample, but it still has substantial accumulated pose error and no
validated high-speed/high-steering transfer. The one-step HGB should remain a
local-transition comparator, not be naively fed its own predictions for a
lap-length rollout. Since multiple frozen candidates have now been inspected
on these two practice runs, they are no longer an untouched final test set;
use them only as transfer diagnostics and reserve any new candidate's final
generalization claim for a new independent capture.

An initial audit report was discarded after a batching bug was found: it used
batch positions rather than frame indices when constructing the EDSSM history.
The corrected evaluator now uses each actual global frame index and checks the
model's explicit (u/v/r) integration equations at every batch. Only the
corrected report is retained. Intermediate/incomplete reports from this
comparison were removed; the source data and candidate checkpoints were not.

### Current assessment and next action

There is a real, quantified one-step improvement for longitudinal/body and
raw-wheel prediction, but no accurate offline simulator yet. The one-step
specialist and recursive teacher currently have complementary failure modes:
the specialist drifts when recursively fed its own state, while the recursive
teacher does not match the specialist's local (u/a_x) and raw-wheel
accuracy. The targeted wheel fine-tune also trades yaw accuracy for better
wheel estimates. These are implementation/architecture issues visible in
existing evaluations, not a reason to gather another broad throttle sweep.

The next modeling pass should preserve the HGB local-response result while
making the recursive plant robust to its own predicted-state distribution,
and should treat wheel-speed prediction as a separate measured state without
equating encoder-rate derivatives to tire force. Use training captures for
fitting, existing validation captures for candidate selection, and keep the
two practice runs diagnostic-only. Because those practice runs contain no
high-speed/high-steering samples, a single compact, targeted open-plane
transfer capture is justified only after a candidate has changed; it should
cover the missing domain and be held out from fitting/selection. No runtime
integration or (v/a_y) phase should begin until the longitudinal/wheel
tradeoff has a defensible recursive result.

### Reproducible artifacts

- Same-start checkpoint and persistence comparison:
  [`wheel_head_practice_transfer_same_starts_v4_20261009.json`](../../live_runs/derived_dynamics_learning_20260928/wheel_head_practice_transfer_same_starts_v4_20261009.json)
- Corrected matched one-step (u/a_x)/wheel audit:
  [`edssm_hgb_one_step_practice_dynamics_v4_20261009.json`](../../live_runs/derived_dynamics_learning_20260928/edssm_hgb_one_step_practice_dynamics_v4_20261009.json)
- Reproducible evaluators:
  [`compare_fixed40_wheel_heads_practice_transfer.py`](../../tools/vehicle_dynamics_learning/compare_fixed40_wheel_heads_practice_transfer.py),
  [`evaluate_edssm_one_step_practice_dynamics.py`](../../tools/vehicle_dynamics_learning/evaluate_edssm_one_step_practice_dynamics.py)

An initial inline prototype was stopped before scoring because repeated
row-by-row access was decompressing NPZ arrays in the loop. It produced no
model, output artifact, or data change. The next analysis should use sequence
vectorization or the existing loader before any further fitting.

### Continuation — recursive wheel response and omitted-state audit (2026-10-09)

This pass used the fixed-40-Hz whole-run dataset and the already captured
practice runs. There was no simulator launch, no source-bag change, no use of
test/final-test rows, and no change to runtime physics, odometry, or MPC.

#### Recursive wheel-spin candidate: reject

The teacher-forced contact-relative wheel model has a one-step macro pair RMSE
of **0.249 m/s** on seven OpenPlane validation runs and **0.394 m/s** on two
practice runs. The new model trained on a 50/50 mixture of teacher-forced and
free-running training rows scores **0.326 m/s** and **0.399 m/s**, respectively.
Thus its intended rollout-aware training did not improve the local wheel
transition on either held-out group.

The matched 5-second full-vehicle comparison is also mixed in the wrong way
for promotion. On six OpenPlane validation runs, the candidate-minus-parent
changes are: position **+0.870 m** (95% run-bootstrap interval −0.211 to
+2.034), `u` **+0.146 m/s** (+0.024 to +0.298), filtered wheel pair
**+0.072 m/s** (+0.008 to +0.127), and raw encoder pair **+0.458 m/s**
(−0.033 to +1.240). On the two practice runs, position improves by 2.159 m
and raw encoder error by 0.487 m/s, but filtered wheel error worsens by
0.143 m/s and `u` by 0.075 m/s. This is not a globally useful plant
improvement; the checkpoint remains rejected and is retained only as a
comparator.

This does not mean the candidate is “bad because it crashes”: it is an offline
prediction comparison, and there was no driving run. It means the same
recursively predicted wheel state does not transfer consistently into the
body model across held-out runs.

#### The miss is not explained by a simple lack of scalar coverage

The causal one-step audit covered 393,637 training transitions and 75,094
validation transitions. Training spans the full steering range represented
in validation (through 0.524 rad); the validation values lie inside the
training 1st–99th percentile for steering and the contact-relative encoder
residual proxy. Only 2.9% of validation speed samples and 2.7% of throttle
slew samples fall outside their respective training 1st–99th percentile
ranges. The difficult `openplane_full_input_validation_20260928` capture has
median absolute steering 0.30 rad, 95th-percentile steering 0.524 rad, and
95th-percentile absolute wheel/body residual 10.78 m/s; those levels are
represented in training.

Across fixed descriptive partitions (not asserted physical regimes), the
rollout-trained wheel model has higher teacher-forced RMSE than its parent in
every populated speed band from 0–12 m/s and every populated steering band,
including 0.4–0.5 rad. Its one-step error on the difficult full-input run is
0.832 m/s versus 0.486 m/s for the parent. This is evidence against “we just
need more examples at the same speed/steering values.” It does not prove
complete joint/history coverage; the more likely issue is that the learned
correction is not the right transition law for the state it is given.

The full-input run is still diagnostically important: it is the largest
one-step miss, not an excluded out-of-domain sample. The model has not yet
explained its response there.

#### Roll/pitch ablation: no transfer gain

Because the wheel model omitted roll and pitch, a separate causal ablation
was trained on whole-run training captures with current/past IMU roll, pitch,
and their changes over 25–400 ms. It never reads future attitude. On the same
attitude-valid rows, the one-step wheel-pair RMSE changes from 0.2491 to
0.2483 m/s on OpenPlane validation (paired seven-run interval −0.0051 to
+0.0028 m/s) and from 0.3942 to 0.4010 m/s on practice (both practice runs
worse). The full-input validation run improves by 0.0123 m/s, but the aggregate
validation result is near zero and practice degrades. Roll/pitch is therefore
not promoted as a universal wheel-spin correction. A model that uses roll in
free rollout would also have to predict roll internally; that step was not
attempted because the one-step transfer result did not justify it.

#### Longitudinal and acceleration status

The fixed-40-Hz one-step transition candidate remains a genuine local gain:
on seven validation runs, `u` RMSE is 0.00588 m/s versus 0.00656 for its
reference (paired interval −0.00079 to −0.00056 m/s); effective `a_x` RMSE is
0.2215 versus 0.2590 m/s² (−0.0453 to −0.0298); filtered rear-wheel pair
RMSE is 0.0910 versus 0.0987 m/s. On the two practice runs the corresponding
values improve from 0.01364 to 0.01237 m/s, 0.567 to 0.536 m/s², and 0.1618
to 0.1569 m/s. These are **one-step, truth-context** scores. They do not make
the model a recursive plant: on the matched 5-second practice replay the
one-step specialist diverges substantially more than the recursive teacher.

The simulator `a_x/a_y` labels at `t+1` are better aligned with kinematic
acceleration derived from the next rigid-body state than labels at `t`, and
the exact midpoint rigid-body equations reconstruct the simulator-truth
`u/v` transitions to numerical precision. However, the trained direct-`a_x`
head did not remain accurate when recursively integrated; the direct-`a_y`
and kinematic-derived-`a_y` candidates also failed recursive evaluation.
These tests isolate the present bottleneck to predicting the next force /
acceleration response from the available state and controls—not to an error in
the algebraic body-frame transport equations. Neither acceleration head is
promoted.

#### What the evidence says about missing physics

The repo’s [simulator asset audit](UNITY_SIMULATOR_ASSET_AUDIT.md) records the
known scene-level mass, wheel radius, suspension spring/damper, geometry,
steering, and friction-curve landmarks. Existing ROS/bag-derived datasets do
not contain each wheel’s true angular speed, local longitudinal/lateral slip,
normal load, suspension state, or motor/brake torque. The rear encoder minus
kinematic contact speed used here is only an **encoder-derived proxy**, not
ground-truth tire slip. The current `a_x/a_y` truth labels constrain aggregate
body force, but cannot uniquely identify four individual tire forces or their
load-dependent response.

The repository already has a read-only instrumentation candidate at
[`OpenPlaneWheelDynamicsCapture.cs`](../../tools/unity/OpenPlaneWheelDynamicsCapture.cs).
However, the asset audit notes that its public source scene is not proven
equivalent to the pinned player. It cannot be treated as new truth until an
identical-command motion A/B establishes equivalence. No new simulator or
instrumentation run was started in this pass.

#### Decision and next concrete work

1. Keep the fixed-40-Hz one-step `u/a_x` model, teacher-forced contact-relative
   wheel model, recursive teacher, and isolated `u` + contact-wheel hybrid as
   separate research comparators. Do not integrate them into production
   odometry/MPC or call any of them a complete offline simulator.
2. Do not repeat broad throttle/steering sweeps: scalar support is already
   broad, and the current regression is not localized to an unrepresented
   scalar range. Continue using whole-run validation to isolate how recursive
   wheel-state errors feed back into `u` and `a_x`.
3. For a physically interpretable wheel-spin model, the decisive missing
   labels are per-wheel RPM, local forward/sideways slip, contact/load and
   drive/brake torque, synchronized at the physics tick. First verify whether
   the existing instrumentation candidate can run against a behavior-
   equivalent simulator; if not, keep the wheel model empirical and explicitly
   proxy-limited rather than fitting invented force parameters.
4. Do not advance to `v/a_y` as a modeling phase until the longitudinal/wheel
   recursive gap is resolved or the missing measurement requirement is
   explicitly accepted. The exploratory `a_y` attempts in this pass are
   rejected candidates, not a completed lateral model.

#### New reproducible artifacts

- Recursive contact-slip fit and evaluation:
  [`fit_recursive_contact_slip_candidate.py`](../../tools/vehicle_dynamics_learning/fit_recursive_contact_slip_candidate.py),
  [`evaluation.json`](../../live_runs/derived_dynamics_learning_20260928/recursive_contact_slip_candidate_20261009/evaluation.json)
- Causal attitude ablation:
  [`fit_contact_slip_attitude_ablation.py`](../../tools/vehicle_dynamics_learning/fit_contact_slip_attitude_ablation.py),
  [`evaluation.json`](../../live_runs/derived_dynamics_learning_20260928/contact_slip_attitude_ablation_20261009/evaluation.json)
- Best partial hybrid research comparator:
  [`isolated_dagger_u_contact_slip_candidate_20261009.joblib`](../../live_runs/derived_dynamics_learning_20260928/isolated_dagger_u_contact_slip_candidate_20261009.joblib)
- Direct-acceleration and lateral candidates are retained only with their
  explicit failed recursive evaluations under
  [`derived_dynamics_learning_20260928`](../../live_runs/derived_dynamics_learning_20260928/).

### Continuation — compose the yaw atlas into the offline plant (2026-10-09)

The requested yaw source is the existing sensor-only, command-intent atlas
(`sensor_only_yaw_regime_atlas_command_age_v4`), not the generic EDSSM yaw
head. It predicts the next yaw rate from current/past sensor history and uses
supported local speed/steering/event experts, with its trained pooled event
expert where a local cell is absent. The atlas is not refit here.

Before recursive evaluation, the new evaluator rebuilt the atlas inputs on
all 17,167 valid transitions in the three held-out high-steer captures. Its
predictions match the original atlas feature/routing implementation to a
maximum absolute difference of `5.96e-8 rad/s`. On those rows the atlas scores
0.0205 rad/s RMSE, 0.00134 rad/s p95 absolute error, and 99.93% below 0.1
rad/s; the maximum is still 0.952 rad/s, so the small p95 does not erase its
rare-error tail. On the 39 specifically matched recursive start windows
(13 per capture), the measured-input one-step RMSE is 0.00290 rad/s and the
maximum is 0.0113 rad/s. The two available practice captures contribute only
one eligible 5-second start each, so their transfer scores are diagnostic,
not a strong independent estimate.

The yaw atlas was then made the yaw-rate head in a research-only hybrid replay:
HGB predicts forward speed, actuator feedback and rear wheels; EDSSM predicts
lateral velocity and roll; the atlas predicts yaw. The initial history is
measured, but every future atlas feature is generated causally from predicted
states and recorded commands. No future sensor/truth is fed into the plant.
This is not an odometry/MPC integration and no simulator was launched.

On the three high-steer runs, the hybrid is initially promising for position
(0.75-second macro position RMSE 0.054 m versus 0.181 m for EDSSM), but its
yaw-rate error is already worse (0.166 versus 0.061 rad/s). By 5 seconds it
fails badly: hybrid position RMSE is 8.81 m, `v` RMSE 15.40 m/s, yaw-rate RMSE
1.57 rad/s, versus EDSSM's 2.02 m, 0.053 m/s and 0.300 rad/s. Practice has
only two windows; the hybrid also loses to EDSSM there at 5 seconds (3.25 m
versus 0.624 m position RMSE). A second causal feature construction used HGB's
explicit effective-`a_x` prediction and EDSSM's predicted `a_y`, rather than
derivatives of mixed model states. It changed the high-steer 5-second result
only slightly (8.81 m position, 1.57 rad/s yaw error), rejecting acceleration
feature construction as the main explanation. Neither hybrid is promoted.

#### What causes the recursive yaw failure

A counterfactual input-sensitivity replay substituted one *recorded future
sensor group at a time* into the atlas, solely to score the next yaw estimate;
these oracle values never update the recursive plant. At 5 seconds, across the
three high-steer runs, feeding the recursively predicted sensor history gives
per-step next-yaw RMSEs of 2.147, 0.774 and 1.786 rad/s. Replacing only the
IMU yaw-rate history with its recorded values reduces those errors to 0.013,
0.047 and 0.023 rad/s. Replacing only acceleration, wheels, roll, or actuator
feedback does not produce a comparable reduction. Supplying all recorded
sensor inputs also returns the atlas to near-zero one-step error. At 2 seconds
one run remains an exception: yaw history alone does not repair it, while all
sensor groups together do, indicating at least one interaction remains there.

This pins down the current limitation: the atlas is a high-accuracy
sensor-conditioned **one-step observer transition**, but feeding its own
predicted yaw back as a substitute for the next IMU sample is not stable over
a long free rollout. The dominant recurrence error is in the predicted yaw
history, not the HGB longitudinal acceleration feature. This does not negate
the atlas's measured-input accuracy; it means the atlas alone is not yet a
closed-loop offline yaw plant. Do not claim lap-level simulation accuracy or
integrate this hybrid. The next targeted modeling step is to make yaw
recursion stable under its own predicted history while retaining the atlas as
the one-step teacher, then evaluate that closure on whole runs not used to fit
it. The exceptional 2-second run must remain in that check.

#### Reproducible artifacts

- Research evaluator composing HGB, EDSSM lateral/roll, and the frozen yaw
  atlas: [`evaluate_hybrid_longitudinal_edssm_candidate.py`](../../tools/vehicle_dynamics_learning/evaluate_hybrid_longitudinal_edssm_candidate.py)
- Full recursive and sensor-substitution report:
  [`highsteer_practice_input_sensitivity_evaluation.json`](../../live_runs/derived_dynamics_learning_20260928/hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/highsteer_practice_input_sensitivity_evaluation.json)
- Earlier acceleration-input ablation reports in the same folder preserve the
  state-derivative and model-head comparisons; these are offline artifacts,
  not successful runtime changes.

### Full-regime recursive yaw-residual follow-up (2026-10-11)

The preceding counterfactual established that the frozen yaw atlas is highly
accurate when given measured history, but its own predicted yaw history causes
recursive drift. A training-only, run-balanced DAgger residual was therefore
fit around the **frozen** atlas. The atlas, HGB longitudinal heads, and EDSSM
were not refit. The residual has no hand-selected speed/steering gate; it adds
to the atlas's causal predicted yaw-rate history. Training used 173 five-second
windows from 12 training runs (34,600 state transitions before DAgger
regenerations). The training windows span approximately 0–11.97 m/s and
0–0.524 rad in their marginal support, but joint support is sparse: only 42 of
132 coarse 1 m/s by 0.05 rad cells contain at least 200 transitions. This is
not evidence of dense coverage over the full speed/steering rectangle.

On paired whole-run validation, the residual substantially reduces recursive
yaw drift and path error at 5 seconds:

| Held-out cohort | 5 s position RMSE, baseline → residual | 5 s yaw-rate RMSE, baseline → residual | Other important result |
|---|---:|---:|---|
| Broad OpenPlane (6 runs) | 9.05 → 4.76 m | 0.607 → 0.175 rad/s | `v`: 6.67 → 1.15 m/s |
| High-steering (3 runs) | 6.85 → 3.42 m | 1.357 → 0.163 rad/s | `v`: 11.69 → 2.26 m/s |
| Practice diagnostic (2 runs) | 7.62 → 3.18 m | 0.546 → 0.312 rad/s | `u`: 0.171 → 1.70 m/s and filtered wheel-pair error: 0.970 → 1.818 m/s |

For broad and high-steering validation, paired run-bootstrap intervals for
the 5-second position and yaw-rate changes exclude zero in the improving
direction. The very small practice cohort is only a transfer diagnostic. Its
large position/yaw improvement is accompanied by unacceptable longitudinal
and wheel-state regression, so the candidate does **not** qualify as an
accurate plant. At 12 seconds it still has large errors: broad position and
`v` RMSE are 15.29 m and 4.78 m/s; high-steering values are 10.69 m and
4.97 m/s. The short-horizon comparison is also not uniformly better (for
example, high-steering position is worse at 0.75 and 2 seconds). This is a
real but incomplete result: recursive yaw closure improved, while coupling to
forward speed, wheels, and initial state remains unresolved.

The 2026-10-09 matched DAgger `v`-head and `u+v`-head ablations remain
research-only. They reduce their targeted velocity errors, but either worsen
position on practice or fail to preserve the high-steering path gain. A
parent-HGB fallback gate preserves practice by not activating there, but
worsens high-steering position (5-second RMSE 5.11 → 5.43 m); it is rejected.
These results support keeping the yaw residual as a diagnostic comparator,
not integrating the DAgger velocity heads or the fallback gate.

#### Evaluation partition incident

During read-only exploratory analysis, an aggregate speed/steering support
query inadvertently included rows summarized across the `test` and
`final_test` partitions. Those aggregates were not used to fit any model,
select the yaw residual, tune it, or report candidate accuracy. They are not
reported here and must not be used as blind-test evidence. The evaluation in
this section explicitly selected training rows for fitting and validation /
practice rows for scoring; it did not load test/final-test scores. No data
were changed. Future work must continue to exclude test/final-test partitions
and treat them as no longer pristine with respect to aggregate support access.

#### Current model status and next work

- **Yaw atlas:** retain as a high-accuracy measured-history one-step observer
  candidate. It has not been shown to be a stable free-running yaw plant by
  itself.
- **Full-regime DAgger yaw residual:** retain as a research-only comparator.
  It improves 5-second recursive yaw/path results on broad and high-steering
  validation and gives a practice position/yaw gain, but does not preserve
  practice longitudinal and wheel accuracy and remains inaccurate at 12 s.
- **Forward speed / wheel-spin model:** no complete recursive improvement is
  established. The fixed-40-Hz teacher-forced `u/a_x` gain and wheel models
  remain separate experiments; the DAgger `u/v` candidates have not resolved
  their coupled recursive failure.
- **Runtime:** no change to odometry, MPC, AMCL, the competition container,
  simulator physics, or the reference trajectory. No simulation was run.

Next, diagnose practice `u` and wheel divergence under the improved yaw
residual using only existing training/validation/practice records, and check
whether the yaw-residual benefit survives a corrected, causally consistent
coupled longitudinal transition. Keep candidate model selection on the
existing validation/practice cohorts; do not use test/final-test data.

#### Reproducible artifacts

- Full-regime yaw residual fit/evaluator:
  [`fit_full_regime_yaw_atlas_dagger_corrector.py`](../../tools/vehicle_dynamics_learning/fit_full_regime_yaw_atlas_dagger_corrector.py)
- Paired held-out results and research-only model:
  [`evaluation.json`](../../live_runs/derived_dynamics_learning_20260928/full_regime_yaw_atlas_dagger_20261011/evaluation.json),
  [`full_regime_yaw_residual.joblib`](../../live_runs/derived_dynamics_learning_20260928/full_regime_yaw_atlas_dagger_20261011/full_regime_yaw_residual.joblib)
- Matched DAgger velocity-head and fallback evaluations:
  [`matched_evaluation.json`](../../live_runs/derived_dynamics_learning_20260928/dagger_v_yaw_atlas_composition_20261009/matched_evaluation.json),
  [`matched_uv_evaluation.json`](../../live_runs/derived_dynamics_learning_20260928/dagger_v_yaw_atlas_composition_20261009/matched_uv_evaluation.json),
  [`matched_parent_fallback_gate.json`](../../live_runs/derived_dynamics_learning_20260928/dagger_v_yaw_atlas_composition_20261009/matched_parent_fallback_gate.json)

### Paired first-divergence and longitudinal-target comparison (2026-10-11)

This follow-up follows the prescribed order: identify when each predicted
state departs from simulator truth; alter only training-fitted longitudinal
and rear-wheel heads with the yaw atlas and yaw residual frozen; then replay
the exact same held-out/practice windows. No simulator was launched and no
test/final-test partition was selected or scored.

#### Where rollout error starts and grows

The trace is sampled at 40 Hz (25 ms). All state channels have a finite
one-step residual at the first predicted sample; “first divergence” therefore
does not mean a delayed fault onset. The useful distinction is which errors
grow first and become large. The yaw-residual rollout changes its yaw
prediction at 50 ms, `u`/`v` at 75 ms, and practice rear-wheel predictions at
325 ms; practice steering and throttle feedback do not change until 450 ms and
1.25 s respectively. In other words, the large practice wheel and yaw errors
are not preceded by a large actuator-feedback prediction error.

Practice (two held-out diagnostic runs) error RMSE illustrates the sequence:

| Time | Yaw rate | Rear-wheel pair | Forward `u` | Lateral `v` | Steering feedback | Throttle feedback | Position |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 ms | 0.0127 rad/s | 0.0429 m/s | 0.0007 m/s | 0.0047 m/s | 0.0003 rad | ~0 | 0.0022 m |
| 250 ms | 0.2027 | 0.1527 | 0.0042 | 0.0109 | 0.0009 | 0.0004 | 0.0029 |
| 500 ms | 0.3775 | 0.4009 | 0.0124 | 0.0094 | 0.0003 | 0.0004 | 0.0326 |
| 750 ms | 0.3739 | 0.5684 | 0.0109 | 0.0088 | 0.0003 | 0.0006 | 0.1336 |
| 2 s | 0.7319 | 1.2345 | 0.5078 | 0.0998 | 0.0004 | 0.0012 | 1.267 |
| 5 s | 0.5464 | 0.9696 | 0.1707 | 4.9868 | 0.0017 | 0.0031 | 7.624 |

These are per-state macro-run RMSEs against truth, not errors between the two
candidate predictions. Practice first develops a sizeable yaw-rate and wheel
speed discrepancy (about 250–500 ms); `u` remains close until later, and
actuator-feedback errors stay small. Position error follows integration and
then grows. At 5 s, the large lateral-velocity error also contributes. This
does not prove wheel error causes yaw error; the channels are coupled in the
rollout. On high-steering validation, the distinguishing late failure is
instead a simultaneous yaw/`v` blow-up (baseline yaw-rate `1.357 rad/s`, `v`
`11.69 m/s` at 5 s), while speed and wheel errors are still much smaller.
Thus one longitudinal correction cannot explain all cohorts.

The frozen yaw residual itself helps the 5 s path and yaw comparison, but it
trades errors between states on practice: it reduces yaw-rate RMSE from 0.546
to 0.312 rad/s and position from 7.62 to 3.18 m, while raising `u` error from
0.171 to 1.70 m/s and rear-wheel error from 0.970 to 1.82 m/s. This justifies
keeping the yaw model unchanged during this pass and diagnosing the coupled
state rollout rather than broadening/refitting the yaw atlas.

Full 25 ms curves, per-run final errors, and same-window baseline/candidate
traces are in
[`first-divergence evaluation`](../../live_runs/derived_dynamics_learning_20260928/full_regime_yaw_first_divergence_20261011/evaluation.json),
reproduced by
[`diagnose_yaw_residual_first_divergence.py`](../../tools/vehicle_dynamics_learning/diagnose_yaw_residual_first_divergence.py).

#### Targeted training-only correction and transfer check

Two three-generation, run-balanced DAgger models were trained from the same
12 training captures (173 initial windows, 98,233 accumulated transition rows
after DAgger regeneration). They correct only HGB's forward-speed and two
rear-wheel increments. The yaw atlas, yaw residual, lateral head, and actuator
heads remain frozen. The candidates differ only in their training labels:

- **State correction:** target includes accumulated rollout-state error,
  `truth_next - predicted_current - base_increment`.
- **Transition increment:** target is the one-step ground-truth transition
  residual, `truth_next - truth_current - base_increment`.

At 5 s, paired macro-run RMSE change versus the frozen yaw-residual plant
(negative is better):

| Cohort | Label | `u` | Rear wheels | `v` | Yaw rate | Position | Pose yaw |
|---|---|---:|---:|---:|---:|---:|---:|
| Broad validation | state correction | −0.740 m/s | −0.156 m/s | −0.234 m/s | −0.041 rad/s | −0.579 m | +0.056 rad |
| Broad validation | transition increment | −0.718 | −0.155 | +0.310 | −0.002 | −0.580 | +0.004 |
| High-steering validation | state correction | −0.013 | +0.001 | +0.128 | +0.004 | +0.272 | +0.008 |
| High-steering validation | transition increment | −0.057 | −0.003 | +0.131 | +0.009 | +0.066 | −0.011 |
| Practice (2 runs) | state correction | −1.622 | −1.728 | −0.078 | −0.242 | −2.464 | −0.046 |
| Practice (2 runs) | transition increment | −1.453 | −1.119 | −0.080 | −0.047 | −1.620 | +0.086 |

The apparent practice gain is not a transferable plant improvement: both
variants worsen high-steering 5 s position in all 3/3 runs and worsen `v` and
yaw-rate in all 3/3 runs. The transition-increment version improves practice
`u`, wheel, `v`, yaw-rate, and position in both practice runs at 5 s, but
worsens pose-yaw in both; at 2 s it worsens yaw-rate by `+0.266 rad/s` while
improving `u` and wheels. Broad validation is mixed across runs and variables.
Neither candidate meets the cross-motion-variable transfer requirement.

A four-way practice-only head ablation further showed strong non-additive
coupling. With the state-correction labels, wheel heads alone reduce practice
0.75 s yaw-rate RMSE `0.263 → 0.049 rad/s` and position `0.123 → 0.046 m`,
while the `u` head alone worsens them (`0.397 rad/s`, `0.221 m`). With the
transition-increment labels, wheel-only does not give that early gain
(`0.284 rad/s`, `0.125 m`); combining heads gives `0.292 rad/s`, `0.134 m`.
At 5 s the transition-increment combined candidate improves position to
`1.56 m` from `3.18 m`, but pose-yaw regresses from `0.105` to `0.191 rad`.
This is evidence that the state-update/label semantics materially change
coupled rollout behaviour—not evidence for promoting either set of heads.

The exact candidate artifacts and reports are:

- [`state-correction candidate`](../../live_runs/derived_dynamics_learning_20260928/yaw_conditioned_u_wheel_dagger_20261011/evaluation.json)
- [`transition-increment candidate`](../../live_runs/derived_dynamics_learning_20260928/yaw_conditioned_u_wheel_increment_dagger_20261011/evaluation.json)
- [`state-correction head ablation`](../../live_runs/derived_dynamics_learning_20260928/yaw_conditioned_u_wheel_head_ablation_20261011/evaluation.json)
- [`transition-increment head ablation`](../../live_runs/derived_dynamics_learning_20260928/yaw_conditioned_u_wheel_increment_head_ablation_20261011/evaluation.json)
- [`fit_yaw_conditioned_u_wheel_dagger.py`](../../tools/vehicle_dynamics_learning/fit_yaw_conditioned_u_wheel_dagger.py) and [`evaluate_yaw_conditioned_u_wheel_head_ablation.py`](../../tools/vehicle_dynamics_learning/evaluate_yaw_conditioned_u_wheel_head_ablation.py)

#### Decision and next step

Keep both candidates research-only; do not integrate either into odometry,
MPC, or the runtime plant. At the time of that comparison, marginal support
alone did not justify more simulation data. The subsequent regime analysis
below identifies one specific sparse, high-error joint cell. No simulation,
runtime physics, reference trajectory, or competition configuration was
changed.

#### Raw-wheel regime support audit (2026-10-11)

The raw-encoder transition residual was followed by a held-out regime audit.
It used only training, validation, and the two designated unseen-practice
captures; test/final-test partitions were not opened. The simulator was not
launched. The report separates speed, absolute steering, their joint bins,
throttle command/slew, and a wheel/body-speed-mismatch proxy:

- [`regime support report`](../../live_runs/derived_dynamics_learning_20260928/raw_encoder_transition_residual_20261011/regime_support_analysis.json)
- [`regime audit script`](../../tools/vehicle_dynamics_learning/analyze_raw_wheel_transition_regimes.py)

No validation/practice joint bin was completely absent from training, but
mere nonzero support is misleading. A specific weakly supported, high-error
cell is 8.0–9.0 m/s body speed and 0.50–0.55 rad absolute steering. The broad
training set has only 41 one-step rows in this cell across two captures, and
the separate high-steering training set has none; the broad held-out run has
31 rows. In that held-out cell, raw-encoder-pair one-step RMSE is 2.462 m/s
for the baseline and 2.446 m/s for the fitted residual. The improvement is
negligible and the absolute error remains large. The 41 training rows are
time-correlated samples, not 41 independent experiments.

Within this cell, the validation error is concentrated in steady-throttle
rows with large encoder/contact-speed mismatch: zero throttle slew has 22
rows at 2.911 m/s baseline RMSE, versus 9 throttle-decreasing rows at
0.414 m/s; mismatch at least 4 m/s has 10 rows at 3.318 m/s, versus 13 rows
below 2 m/s at 2.301 m/s. These are overlapping descriptive groups, not a
causal separation. The cell includes both steering signs. Representative
source samples show encoder-derived surface rate falling from about
10.9–12.7 m/s to 0.47 m/s in one 25 ms transition while simulator body
forward speed remains about 8 m/s. The paired encoder source intervals are
about 24.2–25.6 ms, their source samples are within about 1.1 ms of the state
sample, and packet sequence increments by one. Thus these examples are not
explained by a missing bridge packet or stale packet alignment. They indicate
an abrupt wheel/body-speed mismatch response in the recorded signal, but do
not by themselves distinguish actual simulated wheel spin-down from an
encoder-derived measurement artifact. The rigid-state truth contains chassis
motion, not an independent per-wheel angular-speed truth channel.

Across all rows, the residual changes one-step raw encoder RMSE only slightly:
0.253→0.248 m/s on broad validation, 0.160→0.157 m/s on high-steering
validation, and 0.399→0.397 m/s on practice (two runs). This does not repair
the specific high-error cell, nor does it validate a physical wheel-dynamics
teacher. The target is encoder-angle-derived surface rate; it must not be
described as simulator wheel-speed ground truth. The previous recursive
checks also showed no cross-cohort gain, so this residual remains
research-only and is not integrated into odometry or the offline plant.

**Data decision:** a narrowly targeted open-plane capture is justified now;
another broad sweep is not. The unsupported region is the joint condition
around 8.0–8.3 m/s, ±0.50–0.524 rad, especially high encoder/contact mismatch
with throttle held at zero. Collect independent repetitions in both turn
directions, including matched steady-zero-throttle and throttle-cut entries,
and retain the raw cumulative encoder angles, their source stamps, actuator
commands/feedback, packet sequence, and simulator chassis truth. That test
should determine whether the rapid wheel-rate changes repeat as a genuine
simulated response and provide enough independent training episodes to test
the local predictor. Do not fit or promote a new model from the present
41-row cell alone. The older datasets do not establish this specific
high-speed/high-steering response with adequate independent-run coverage.

No model/runtime/physics/trajectory changes were made. The fitted raw-wheel
residual and all DAgger candidates remain research artifacts only.

#### First-divergence replay and focused feedback-gap follow-up (2026-10-11)

The next diagnostic replay used the same initial states, command sequences,
validation runs, and two unseen practice runs for the frozen baseline and
yaw-corrected rollout. It compares when the yaw correction first changes each
predicted state, then scores each state against simulator truth; these are
different questions. Full curves and per-run results are in
[`first-divergence evaluation`](../../live_runs/derived_dynamics_learning_20260928/full_regime_yaw_first_divergence_20261011/evaluation.json).

On practice, the yaw correction first changes yaw rate at 50 ms, body `u/v` at
75 ms, rear-wheel state at 325 ms, steering feedback at 450 ms, and throttle
feedback at 1.25 s. At 25 ms, both rollouts still have the same errors:
`u` 0.00070 m/s, `v` 0.00474 m/s, yaw rate 0.01274 rad/s, rear-wheel pair
0.04290 m/s, steering feedback 0.00025 rad, and throttle feedback 0.000041.
The yaw correction helps yaw-rate error at 50 ms (0.02654→0.01151 rad/s), but
then worsens it at 75 ms (0.02389→0.02794) and 100 ms
(0.00589→0.04937). By 5 s, yaw-rate RMSE improves 0.546→0.312 rad/s and
position RMSE 7.624→3.181 m, while forward-speed RMSE regresses
0.171→1.701 m/s and rear-wheel-pair RMSE 0.970→1.818 m/s. Integrated pose
yaw improves 1.307→0.105 rad. This establishes that a better-looking path/yaw
does not establish a better plant, and does not justify changing or widening
the yaw atlas.

The training-support audit identified the narrower wheel state that warranted
one focused follow-up: physical steering near ±0.50 rad, zero throttle command,
throttle feedback still near 0.5, and raw encoder speed above body speed at
8–9.22 m/s. A first matched step/ramp capture reached only 8.26–8.55 m/s in
that state. A second capture was therefore limited to the remaining upper
band; it produced 17 valid transitions at 8.78–9.04 m/s, with 3.10–4.27 m/s
wheel/body mismatch. Both captures are preserved with bags, derived datasets,
and paired analyses:

- [`lower feedback-gap capture`](../../live_runs/openplane_yaw_highsteer_wheel_feedbackgap_train_20261011_r01/)
- [`upper feedback-gap capture`](../../live_runs/openplane_yaw_highsteer_wheel_feedback_upper_train_20261011_r01/)

Both runs completed all 32 phases with zero collisions, zero quality failures,
zero timing faults, all expected reset recoveries, and accepted stream rates
near 40 Hz. Their command rates were 39.85 and 39.86 Hz. Each capture is one
run, so the paired-response bootstrap has only one independent run and its
degenerate interval is not population-level uncertainty. The step-minus-ramp
wheel/body residual effect over 0.10–1.00 s is +0.596 m/s in the lower capture
and +0.769 m/s in the upper capture: in these runs, the rapid throttle cut
produces the larger residual proxy. This is a replicated directional
observation across two captures, not a statistically independent multi-run
estimate or tire-force measurement.

The two captures contributed 660 and 742 valid training transitions to one
combined raw-encoder residual fit (368,499 transitions, 18 training runs).
The yaw atlas, yaw residual, filtered-wheel head, physics, and runtime remained
frozen; validation and practice windows were replayed unchanged, and
test/final-test data were not used. The candidate report is
[`combined feedback-gap candidate evaluation`](../../live_runs/derived_dynamics_learning_20260928/raw_encoder_transition_residual_with_lower_upper_feedbackgap_train_20261011/evaluation.json).

At one step, only the raw-encoder prediction changes: RMSE improves by
0.00461 m/s on broad validation and 0.00401 m/s on high-steering validation,
but worsens by 0.00091 m/s on practice. Recursive 5 s candidate-minus-baseline
RMSE deltas (negative is better) are:

| Cohort | Position (m) | `u` (m/s) | `v` (m/s) | Yaw rate (rad/s) | Filtered rear wheels (m/s) | Raw encoder wheels (m/s) | Integrated pose yaw (rad) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Broad validation | −0.441 | −0.030 | −0.720 | −0.018 | +0.006 | −0.008 | −0.016 |
| High-steering validation | +0.031 | −0.003 | +0.022 | −0.008 | −0.006 | +0.021 | −0.002 |
| Unseen practice | −0.067 | −0.712 | −0.060 | −0.220 | −0.546 | +0.382 | +0.191 |

The practice position average hides run disagreement: position error worsens
by 0.191 m on practice run r02 and improves by 0.324 m on r03. In contrast,
raw-encoder error worsens on both practice runs (+0.467 and +0.297 m/s), and
integrated pose-yaw error also worsens on both (+0.206 and +0.175 rad). Thus
the combined candidate does improve body forward/lateral speed, yaw rate, and
filtered-wheel rollout on practice, but it fails the required all-state
transfer gate because measured-wheel and integrated-pose errors regress; it
also slightly regresses position on high-steering validation. **Do not promote
or integrate it.** It remains a research artifact.

This identifies the next modeling issue more narrowly than “collect more
data”: the learned raw-encoder transition can improve several recursive body
states while making the raw-wheel observation and integrated pose-yaw
prediction less accurate. The available signals do not provide independent
per-wheel simulator ground truth, so those quantities must not be conflated.
The upper capture reached 9.04 m/s, leaving only the 9.04–9.22 m/s tail of the
held-out condition without a matching new transition. However, the combined
candidate still worsens raw-wheel and integrated pose-yaw error on both
practice runs, despite adding coverage in both the lower and upper bands.
That cross-run failure is not evidence that another nearby speed sweep is the
next fix; diagnose the raw-measurement versus physical-wheel latent coupling
offline first, and request more data only if that diagnosis needs the remaining
9.04–9.22 m/s tail. The yaw atlas remains frozen. Both simulators and
experiment containers are stopped; no odometry, MPC, physics, or production
model was changed.
