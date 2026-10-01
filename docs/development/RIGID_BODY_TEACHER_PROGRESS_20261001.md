# 3D rigid-body plant teacher — progress, 2026-10-01

## Bottom line

The first command-only 3D teacher is now implemented and trained. Adding the
packet acceleration as an offline label materially improved it over the prior
3D candidate, but it still does **not** beat the older 2D GRU on complete-lap
position prediction and is not ready for offline-simulator, MPC, or odometry
use. No production controller, localization, simulator physics, or runtime
topic policy was changed.

The latest model is
`live_runs/derived_dynamics_learning_20260928/plant_teacher_rigidbody_gru_acceleration_20261001/`.
Its full training and scoring record is
[`training_report.json`](../../live_runs/derived_dynamics_learning_20260928/plant_teacher_rigidbody_gru_acceleration_20261001/training_report.json).

## Data and splits

The schema-6 dataset is
`live_runs/derived_dynamics_learning_20260928/plant_teacher_mixed_dataset_full3d_fixed25_20261001/`:
850,576 aligned rows, 7,797 reset-/gap-separated sequences, 116 run records,
and exact 25 ms simulator time steps. All rows have finite rigid-state and
linear-acceleration labels. Whole-run splits are 79 train, 3 validation, 30
test, and 2 final-test; two additional records are excluded.

At an 11-second rollout horizon, only six training run IDs have sequences
long enough: five practice captures and the throttle-surface capture. The
throttle-surface bag contributes 735 of 740 eligible sequences, so those
sequences are numerous but are not 735 independent run-level replicates.
Validation has two complete 12-lap practice runs. The final-test practice bag
has been inspected in earlier 2D-model work, so it is a diagnostic holdout,
not a fresh blind confirmation.

## Verified frame contract

The packet fields are not all expressed at the same rigid-body point/frame.
On four practice captures:

- Packet linear velocity is body-frame COM velocity. `u` matches the exported
  longitudinal state; `v_rear = v_com_y - L*r` matches lateral state exactly.
- Packet angular-velocity `z` matches exported yaw rate exactly.
- Packet quaternion is the world attitude. Packet position `x/y` exactly
  matches the exported rear-axle pose in the inspected captures.
- With `L = 0.15532 m`, rear-axle velocity for pose integration is
  `v_rear_3d = v_com + omega x [-L, 0, 0]`.

Using the COM velocity directly to integrate rear-axle position was a real
reference-point error. Applying the offset reduced per-step XY integration
residuals, for example on the NTU 12-lap capture, from about
`[0.0048, 0.0064] m` to `[0.0023, 0.0049] m` RMSE. Correcting this changed one
full-run teacher score only slightly, however, so it was not the dominant
multi-metre drift source.

Packet linear acceleration is body-frame too. Against finite-difference
COM-velocity acceleration with the rigid transport term included, direct
acceleration labels have component RMSE around `0.36–0.64 m/s^2` on the
practice captures and `0.16 m/s^2` on the throttle-surface capture. Rotating
those labels as though they were world-frame increases the aggregate error to
over `5 m/s^2`. The acceleration labels are therefore useful offline teacher
targets, not rollout inputs.

## Model design and empirical results

[`train_rigid_body_teacher.py`](../../tools/vehicle_dynamics_learning/train_rigid_body_teacher.py)
predicts effective body-frame translational acceleration, body angular
acceleration, actuator rates, and rear-wheel-speed rates with a causal GRU.
Rigid-body body-frame transport, quaternion attitude integration, and
rear-axle world-position integration remain explicit. After initialization,
free rollouts use only the predicted state, predicted attitude, and logged
steering/throttle commands. Future simulator state, acceleration, encoder,
or actuator measurements are losses/score labels only.

| Candidate | Validation selection | Final-test 12-lap practice result | Decision |
| --- | --- | --- | --- |
| Initial 3D, 2 s rollout | Short-sequence-heavy validation; later found imbalanced | 13.11 m XY RMSE, 16.04 m endpoint, 1.64 rad attitude RMSE | Reject |
| 3D with derivative labels, 2 s rollout | Run-balanced short windows plus full-run selection; best step 1200 | 13.24 m RMSE, 10.91 m endpoint, 1.60 rad | Reject |
| 3D derivative model, 11 s rollout | Best step 600; 5.66 m mean validation XY RMSE, 1.80 rad | 6.00 m RMSE after rear-point correction, 1.01 m endpoint, 1.82 rad | Reject |
| Warm-started 11 s model, corrected rear point | Best step 800; 6.77 m validation RMSE, 1.18 rad | 11.77 m RMSE, 11.51 m endpoint, 1.42 rad | Reject; validation-to-test transfer failed |
| Acceleration-supervised 11 s model | Best step 800; 6.75 m validation RMSE, 0.70 rad | **5.91 m RMSE, 4.83 m endpoint, 1.06 rad** | Best 3D candidate so far, still reject for use |
| Existing 2D mixed GRU (same final-test bag) | Earlier held-out report; quality flag noted there | 1.52 m RMSE, 2.54 m endpoint, 0.16 rad | Keep as comparator; still not negligible lap drift |

The acceleration-supervised candidate’s local final-test errors are much
smaller than its full-run error: at 1 s it has `0.28 m` XY error and about
`0.09 m/s` body-velocity error; by 5 s position error is `1.30 m`, by 10 s
`2.87 m`, and full-run XY RMSE reaches `5.91 m` with a `1.06 rad` attitude
RMSE. This is a real improvement over the immediately preceding 3D model, but
not an improvement over the present 2D comparator.

The first 3D run also showed that selecting from short windows was inadequate:
the original validation pool contained 549 short OpenPlane sequences but only
two complete practice trajectories. The trainer now balances validation
windows by run and includes complete practice-run rollouts in checkpoint
selection. Some checkpoints had better short-window loss while full-run drift
became much worse; those checkpoints were rejected.

## Hybrid comparison on complete held-out practice runs (2026-10-01)

The proposed hybrid was evaluated over six entire 12-lap captures (about
73–76 s each), using 16 truth samples only to initialize both models and then
commands plus each model's own predictions. No future truth or sensor values
were used. The evaluator and full per-run metrics are
[`evaluate_hybrid_rigid_body_teacher.py`](../../tools/vehicle_dynamics_learning/evaluate_hybrid_rigid_body_teacher.py)
and
[`hybrid_2d3d_comparison_20261001.json`](../../live_runs/derived_dynamics_learning_20260928/hybrid_2d3d_comparison_20261001.json).

| Run-level group | Runs | Planar mean XY RMSE | Hybrid mean XY RMSE | Planar mean endpoint | Hybrid mean endpoint | Hybrid wins |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Held-out test | 3 | 2.96 m | 3.32 m (+12.1%) | 6.19 m | 7.11 m (+14.9%) | 0/3 RMSE, 0/3 endpoint |
| Validation | 2 | 8.87 m | 8.86 m (−0.1%) | 9.28 m | 9.33 m (+0.6%) | 1/2 RMSE, 1/2 endpoint |
| Previously inspected final-test capture | 1 | 1.53 m | 1.49 m (−2.8%) | 2.14 m | 2.89 m (+35.2%) | RMSE only |

On the three held-out test runs, the hybrid is worse on both position RMSE and
endpoint drift for every run. Its full-attitude error is also worse than the
planar model's yaw error on all six runs (not a perfectly identical metric,
since the hybrid reports quaternion geodesic attitude error). Across all six
runs the hybrid's mean RMSE is 3.6% worse and mean endpoint error is 9.2%
worse. The small RMSE win on the previously inspected final-test capture does
not transfer and accompanies a larger endpoint and attitude error. **Reject
the hybrid; retain the planar GRU as the comparator and do not integrate the
3D teacher.**

The comparison script initially could not run because PyTorch was absent from
the host and both existing development images. It was then run successfully
with CPU PyTorch 2.14.1 in a persistent user-level analysis environment. This
does not change the simulator container or production dependencies.

## Root-cause follow-up: local accuracy versus recursive drift

The planar 5-second GRU was then scored both in free rollout and one-step
teacher-forced mode on the same six captures. For one-step scoring, each
prediction uses the true current state/history, then is compared to the next
25 ms truth sample; this is a transition diagnostic, not a valid simulator
rollout. Its yaw-rate RMSE is `0.015–0.024 rad/s` in the highest observed
steering band (`|steering feedback| = 0.24–0.32 rad`) and `0.027–0.037 rad/s`
in the `0.16–0.24 rad` band, with near-zero bias in both validation runs.

In recursive rollout, the two validation runs instead show yaw-rate bias of
`−0.050` and `−0.053 rad/s` overall, and `−0.130/−0.142 rad/s` in the
`0.16–0.24 rad` steering band. Their heading error grows to about 3 rad by
the final lap boundary and position endpoints are about 9 m. This contrast
points to accumulated state/hidden-state feedback error, not an inability to
fit a local high-steering transition. Wheel-surface-speed errors also grow
with rollout; their relationship to the yaw drift remains a hypothesis, not
yet a demonstrated cause.

The reusable free-running evaluator now reports per-lap state errors and
yaw-rate errors grouped by truth speed/steering regime. Its report is
[`planar_full_diagnostics_20261001.json`](../../live_runs/derived_dynamics_learning_20260928/planar_full_diagnostics_20261001.json).
Its implementation also fixes an existing report-writing `NameError` by
reading the bag path from the manifest.

## Long-horizon loss experiments (2026-10-01)

The 60-second heading-only candidate
(`plant_teacher_planar_gru_60s_heading_20261001`) was selected at step 1800,
but failed complete-run scoring. Across the three named held-out test runs,
mean XY RMSE increased from 2.96 m to 18.97 m and endpoint error from 6.19 m
to 34.54 m; it lost all three runs. Across all six evaluated captures, mean
XY RMSE increased from 4.69 m to 16.07 m. It is rejected.

The next candidate added differentiable rear-axle XY trajectory supervision
to the recursive state loss, using the exact 25 ms timebase and simulator
pose only as an offline label. Its implementation is in
[`train_nssm.py`](../../tools/vehicle_dynamics_learning/train_nssm.py), and
its reproducible report and complete-run diagnostics are in
[`training_report.json`](../../live_runs/derived_dynamics_learning_20260928/plant_teacher_planar_gru_60s_xy_20261001/training_report.json)
and
[`full_run_diagnostics_20261001.json`](../../live_runs/derived_dynamics_learning_20260928/plant_teacher_planar_gru_60s_xy_20261001/full_run_diagnostics_20261001.json).
The run used 2,400 recursive steps (60 s), 96 hidden units, batch size 2,
and position-loss weight 0.05 with 0.5 m robust-loss scale. Best validation
checkpoint: step 2200; normalized state RMSE 0.164, position trajectory
radial RMSE 5.96 m, endpoint RMSE 6.97 m, and integrated heading error RMSE
8.36 rad on the eight validation windows.

This position-only candidate improved the two validation captures, where the
existing comparator had large drift, but generalized poorly. Across all six
complete 12-lap captures mean XY RMSE rose from 4.69 m to 6.24 m, endpoint
error from 6.54 m to 8.57 m, and heading RMSE from 0.90 rad to 1.83 rad; it
won on only two of six runs (both validation runs). All three named test runs
and the final-test run were worse. These captures include previously examined
data, so they are diagnostic holdouts, not a new blind confirmation.

The specific failure is a recursive yaw-rate bias. In free rollout the
candidate has roughly `-0.23 rad/s` yaw-rate bias on every capture, while its
one-step truth-fed yaw error remains much smaller (about `-0.026 rad/s`
overall). The one-step high-steering bins are not the worst: absolute bias is
near zero at `|steering| = 0.16–0.32 rad`, while the `0–0.08 rad` bin has
`-0.046 rad/s` bias. Thus position loss alone does not prevent exposure-driven
feedback drift, and it appears to trade away a small but persistent yaw-rate
error to fit XY on the validation trajectories.

The balanced XY+heading candidate is
`plant_teacher_planar_gru_60s_xy_heading_20261001`; its report and per-run
scores are
[`training_report.json`](../../live_runs/derived_dynamics_learning_20260928/plant_teacher_planar_gru_60s_xy_heading_20261001/training_report.json)
and
[`full_run_diagnostics_20261001.json`](../../live_runs/derived_dynamics_learning_20260928/plant_teacher_planar_gru_60s_xy_heading_20261001/full_run_diagnostics_20261001.json).
It starts from random weights and uses the same five long practice training
runs, but adds heading weight `0.005` to the position weight `0.05`. Best
checkpoint was step 2400. Validation-window position RMSE is 1.40 m and
integrated-heading RMSE 0.188 rad; named-test windows are 1.29 m and the
previously inspected final-test windows 1.53 m.

On six complete 12-lap practice captures, the balanced candidate lowers mean
position RMSE from 4.69 m to 2.18 m, endpoint error from 6.54 m to 3.57 m,
and heading RMSE from 0.90 rad to 0.40 rad. It wins 5/6 runs, including all
three named test captures; the previously inspected final-test capture is the
one regression (1.53 m to 2.36 m position RMSE). Paired run-bootstrap
intervals remain broad and include no improvement (RMSE ratio change roughly
−60% to +6% at 95%). With six correlated historical captures this is
promising evidence, not a statistically conclusive result or blind
confirmation.

## Remaining domain failure and next training design

The long-horizon candidates only have enough pose-labeled sequences for
60-second training on five practice bags. The practice trajectories contain
essentially no simultaneous `speed >= 5 m/s` and `|steering| >= 0.16 rad`
samples. Shorter pose-labeled windows cover 15 independent training bags and
include open-plane full-input excitation and the throttle surface sweep, with
thousands of high-speed/high-steering samples. The current 60-second model
does not use these windows.

On four open-plane source captures excluded from the 60-second candidate's
training, one-step yaw-rate RMSE in the `speed >= 5 m/s`, `|steering| >= 0.16`
region was about `0.27–0.29 rad/s` on the two full-input excitation captures,
but free-running RMSE grew to `2.20–2.44 rad/s`. On the throttle-sweep
capture, local error is lower (`0.14 rad/s`) but free-running error rises to
`7.47 rad/s`; its weighted XY RMSE across reset-separated phases is 26.96 m.
The older broad-data 5-second comparator is also poor there (18.19 m XY RMSE,
1.50 rad/s high-regime yaw RMSE). These are single-capture/phase diagnostics,
not hundreds of independent replicates.

This isolates the current limit: long practice rollouts learn lap-scale
behavior but fail to retain the broad input envelope; local fit alone does
not prevent recursive high-speed/high-steering divergence.

## Multi-scale warm-start result

The next candidate adds a 2-second, high-steering-sampled auxiliary rollout
to the 60-second pose/heading objective and initializes from the existing
broad-data 5-second GRU. It reuses that checkpoint's normalization and records
all inherited training-run IDs so those captures cannot be misreported as
held out. The training extension is in
[`train_nssm.py`](../../tools/vehicle_dynamics_learning/train_nssm.py); the
one-step mixed-horizon CLI/gradient smoke test passed. The complete record is
[`training_report.json`](../../live_runs/derived_dynamics_learning_20260928/plant_teacher_multiscale_warmstart_20261001/training_report.json),
with full-run results in
[`full_run_and_extreme_diagnostics_20261001.json`](../../live_runs/derived_dynamics_learning_20260928/plant_teacher_multiscale_warmstart_20261001/full_run_and_extreme_diagnostics_20261001.json)
and the broad 5-second comparator's matching extreme-capture scores in
[`baseline_extreme_holdout_diagnostics_20261001.json`](../../live_runs/derived_dynamics_learning_20260928/plant_teacher_multiscale_warmstart_20261001/baseline_extreme_holdout_diagnostics_20261001.json).

Training used 2,400-step/60-second long rollouts plus 80-step/2-second
auxiliary rollouts, with 30% of the auxiliary batch sampled from windows
whose mean absolute steering is at least 0.16 rad. It was initialized from
the broad 5-second GRU and fine-tuned at learning rate `3e-4`. Early stopping
selected step 600 (the run stopped at step 2200 after eight non-improving
evaluations). On validation, long-horizon XY RMSE is 1.89 m and heading RMSE
0.324 rad; auxiliary XY RMSE is 2.02 m and heading RMSE 0.911 rad.

On the same six complete practice captures as the comparator, mean XY RMSE
improves 4.69 m → 2.66 m (−43%), endpoint error 6.54 m → 5.05 m (−23%), and
heading RMSE 0.90 rad → 0.53 rad (−41%). It wins 4/6 runs. The paired,
run-bootstrap 95% intervals still cross no change (relative XY-RMSE change
about −53% to +20%; endpoint −49% to +68%), and the candidate loses on the
srate38 run and the previously inspected final-test speed-headroom run. This
is a useful transfer improvement, not statistically proven superiority.

Crucially, the separate `openplane_full_input_excitation_20260927_holdout_cache`
was not used by the pretrained model or this fine-tune. Across its 549
reset-/gap-separated sequences (804 s aggregate), the multi-scale model has
1.43 m weighted XY RMSE versus 1.26 m for the broad 5-second comparator. In
the `speed >= 5 m/s`, `|steering| >= 0.16 rad` region (7,559 scored samples),
yaw-rate RMSE is 0.561 versus 0.490 rad/s; at `speed >= 7 m/s` it is 0.463
versus 0.346 rad/s. The candidate remains bounded and close to the comparator,
unlike the scratch long-only model, but has a positive yaw bias (+0.19 to
+0.23 rad/s) there. This is one whole capture, not thousands of independent
replicates, and it is now a consumed diagnostic holdout rather than a blind
future test.

The result is the clearest improvement so far: full-lap drift is substantially
lower than the broad comparator on most practice captures, while high-speed /
high-steering transfer stays near that comparator rather than exploding. The
offline model is still not accurate enough to replace production odometry or
to claim the full 0–10+ m/s envelope solved.

## Joint high-speed/high-steering correction

On the 11 throttle-slew captures, the preceding multi-scale model improved
overall yaw-rate RMSE on all runs but worsened the joint high-speed/high-steer
bin on all 11: mean `0.095 → 0.157 rad/s` for `speed >= 5 m/s` and
`|steering| >= 0.16 rad`. This exposed a specific omission in the earlier
sampler: high steering windows were often low speed. The helper now supports
requiring at least half of a sampled auxiliary rollout to be simultaneously
above the speed and steering thresholds.

The focused candidate is
`plant_teacher_multiscale_highspeed_focus_20261001`, initialized from the
preceding multi-scale model. It uses 2,294 eligible joint-regime windows
across three training captures; half of the auxiliary batches are sampled
from those windows. Its best checkpoint is step 1800. Artifacts are
[`training_report.json`](../../live_runs/derived_dynamics_learning_20260928/plant_teacher_multiscale_highspeed_focus_20261001/training_report.json)
and
[`targeted_full_and_extreme_diagnostics_20261001.json`](../../live_runs/derived_dynamics_learning_20260928/plant_teacher_multiscale_highspeed_focus_20261001/targeted_full_and_extreme_diagnostics_20261001.json).

On the repeated 45% and 65% throttle-slew captures, target-model aggregate XY
RMSE is 0.189 m versus 0.238 m for the broad 5-second model and 0.230 m for
the preceding multi-scale model; endpoints are 0.367 m versus 0.471 m and
0.470 m. Overall yaw RMSE is also lower (0.241 versus 0.267 and 0.223
rad/s). Crucially, joint-regime yaw RMSE returns to 0.097 rad/s, essentially
the broad comparator's 0.095 and well below the preceding candidate's 0.157.
It wins the high-regime yaw comparison against the preceding candidate on all
11 captures.

The result is not uniform across throttle profile: at 45%, XY RMSE is worse
than both comparators (0.166 m versus 0.138 m broad and 0.092 m multi-scale);
at 65%, it is substantially better (0.145 m versus 0.235 m and 0.222 m).
On the six practice captures it averages 2.78 m position RMSE versus 4.69 m
for the broad comparator, but it loses to the preceding multi-scale model
(2.78 m versus 2.66 m) and improves only 3/6 runs against the broad model.
Thus the new candidate recovers the high-throttle/high-speed regime without
dominating the racing and lower-throttle model.

On the separate full-input-excitation holdout, it improves joint-regime yaw
RMSE `0.561 → 0.480 rad/s` and XY RMSE `1.43 → 1.16 m` versus the preceding
multi-scale model; high-regime yaw bias is approximately zero. It also edges
the broad comparator on this single capture (`0.480` versus `0.490 rad/s`).
These results support a regime-conditioned/mixture teacher as the next
modeling direction, but the 11 throttle captures are repeats of two test
profiles, not 11 independent throttle conditions. Run-level means are
descriptive; they do not establish statistical significance across the full
operating domain.

## Causal predicted-state switch diagnostic (2026-10-01)

The two GRUs were next evaluated behind a causal hard gate: use the focused
high-speed/high-steering specialist when the *shared fused predicted state*
has speed at least 7 m/s and absolute steering feedback at least 0.16 rad;
otherwise use the multi-scale generalist. Each model receives that same fused
predicted state and the logged command. Both hidden states are advanced every
step. Simulator truth is used only for initialization history and scoring,
never to select the model during rollout. The evaluator is
[`evaluate_regime_conditioned_teacher.py`](../../tools/vehicle_dynamics_learning/evaluate_regime_conditioned_teacher.py),
and per-sequence scores and gate counts are in
[`regime_conditioned_gate_speed7_steer016_20261001.json`](../../live_runs/derived_dynamics_learning_20260928/plant_teacher_multiscale_highspeed_focus_20261001/regime_conditioned_gate_speed7_steer016_20261001.json).

The gate preserves practice behavior: it selected the specialist on zero
steps of all six practice captures, so mean run-level position RMSE is
effectively unchanged at 2.658 m (generalist 2.658 m; specialist 2.775 m).
It also did not activate on any of the five 45% throttle-slew captures; their
mean per-capture position RMSE stays 0.167 m, while always using the
specialist gives 0.180 m.

On the six repeated 65% throttle captures, the gate selected the specialist
for 35.6% of steps and lowered mean per-capture position RMSE to 0.156 m,
versus 0.196 m for the always-specialist model and 0.282 m for the
generalist. It wins all six captures against both source models. In the
truth-binned high-speed/high-steering region, yaw-rate RMSE is 0.089 rad/s;
the specialist alone is better at 0.078, while the generalist is 0.116.
Thus the position gain comes with a small high-regime yaw penalty relative to
always using the specialist.

On the separate full-input excitation capture, the gate selected the
specialist on 10.3% of steps (131 transitions) and scored 1.370 m position
RMSE, between generalist 1.430 m and specialist 1.156 m. High-regime yaw RMSE
was likewise intermediate: 0.530 versus 0.561 and 0.480 rad/s. The gate is
not a universal winner, and these already-inspected captures are diagnostic,
not blind confirmation. The throttle evidence still consists of only two
command profiles; repeated captures do not establish transfer to the full
speed/steering/throttle envelope. The hard threshold is preliminary and has
no hysteresis; retain both source models and do not integrate this experiment
into MPC or odometry.

## Next work

1. Freeze these candidate weights and thresholds. Do not tune the gate further
   on the consumed captures; that would turn the diagnostic holdouts into
   training/selection data.
2. Blindly confirm on at least one new full-lap practice run and independent
   open-plane runs spanning low/high throttle and the joint high-speed/high-
   steering regime. Preserve exact 25 ms simulator cadence; receipt-time
   jitter is not physical `dt`.
3. Keep all experimental teachers out of MPC and odometry until the blind
   runs show repeatable full-run improvement without unacceptable local
   high-regime losses.

## Warm-started causal mixture teacher (in progress, 2026-10-01)

Following the gate diagnostic, a four-expert causal mixture is being
fine-tuned from the multi-scale generalist. Each expert starts as an exact
copy of the existing transition head and shares its GRU state; the learned
router begins untrained. Before any optimization, a numerical comparison on
32 real dataset rows confirmed the lifted mixture's one-step transition
matches the generalist to `2.4e-7` normalized-state maximum error. A two-step
backpropagation smoke test produced finite loss/gradients, all experts updated,
and router gradients became nonzero after the experts separated.

The training uses the same whole-run split, 2,400-step/60-second pose+heading
objective, and 80-step auxiliary objective with half its batches drawn from
the joint `speed >= 5 m/s`, `|steering| >= 0.16 rad` region. It requests three
members under an eight-hour total budget, up to 20,000 optimizer steps/member.
The output/checkpoints are persisted under
`plant_teacher_multiscale_moe4_causal_warmstart_20261001/`. At a status check
58 minutes after launch, the process was still active at about 426% CPU, but
the only saved checkpoint remained the step-1 initialization and there was no
new validation report. Do not interpret that checkpoint as a trained
improvement. It is pinned to CPUs 8–11 and nice level 10 so the already-running
Explore simulator is not crowded. The analysis environment has CPU-only
PyTorch despite a local GTX 1080.

A representative CPU benchmark using this exact four-expert, hidden-size-96
architecture and the 2,400+80-step batch-2 backward pass took 1.46 s with one
PyTorch intra-op thread versus 2.13 s with ten threads on the same four-core
allocation. This is a synthetic-input timing comparison, not a model-accuracy
result. `train_nssm.py` now exposes `--cpu-threads` (default 1) and applies it
on CPU; this change does not alter the already-running process. The live run
has not been restarted, to preserve any in-memory optimization progress until
its validation state is understood.

After training finishes, score the ensemble on complete practice rollouts,
the consumed full-input capture, and the earlier partial throttle-surface
capture (which was excluded from this candidate's training archive). The
throttle surface was already examined in earlier model work, so it is a
diagnostic transfer check, not blind confirmation. Preserve the current
generalist and focused GRU as comparators. No runtime model has changed.

The model remains offline-only. The current evidence does not justify using it
for racing or localization.
