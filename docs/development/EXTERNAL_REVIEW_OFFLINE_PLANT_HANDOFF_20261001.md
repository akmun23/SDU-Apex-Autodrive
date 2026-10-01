# External review handoff: offline vehicle plant model

**Snapshot:** 2026-10-01. The training and simulator jobs from this work session
have been stopped. This is an evidence summary for an independent reviewer;
it is not a claim that a usable plant model has been achieved.

## 1. The question to review

Can the existing simulator data support a causal, recursively stable vehicle
plant model that is accurate enough to compare racelines and tune MPC weights?
The model should predict motion from its current state/history and future
steering/throttle commands. It may use simulator truth as offline labels, but
must not receive future truth or future measured sensor feedback during a
rollout. Training/selection/test separation must be by complete runs, not
randomly divided frames.

The intended use is offline experimentation, not immediate production
integration. No learned plant or observer has been promoted into MPC, odometry,
localization, or the simulator's physics. Runtime competition topic policy is
unchanged; simulator truth and additional channels in the data are development
labels only.

The required horizon has been revised: **do not require a 12-lap replay**.
Twelve-lap reference captures are roughly 73–76 seconds, or about 6.1 seconds
per lap. Six laps (about 36–38 seconds) is the hard upper bound for useful
offline evaluation; one-, two-, and three-lap accuracy should determine whether
the model is already useful for weight/raceline experiments. Existing longer
bags can be cropped for this evaluation; another 12-lap run is not required.

## 2. Current conclusion

There is a great deal of recorded data, but not an equally large number of
independent long trajectories. The strongest historical planar candidate is
promising on some practice replays, but run-to-run drift remains material and
uncertain. New, clean high-speed straight-line holdouts show severe recursive
failure in both available generalist and high-speed-specialist teachers. No
current checkpoint is fit to serve as a trusted offline simulator.

The actionable failure is not “too few frames.” At 40 Hz, adjacent samples are
highly correlated. Long-horizon supervision currently comes from only a handful
of independent runs, and the learned state transition does not remain correct
when recursively driven by its own predictions. In a fresh throttle hold, one
teacher predicts that speed will fall while full throttle remains applied; the
specialist tracks speed better in one run but still develops enormous pose
drift. More epochs on the same weak objective are not evidence of progress.

## 3. Work stopped and artifacts

The active four-expert mixture run was stopped after about 87 minutes of CPU
use. Its output directory is
`live_runs/derived_dynamics_learning_20260928/plant_teacher_multiscale_moe4_causal_warmstart_20261001/`.
It contains `member_00.pt` with metadata `step: 1`, but no `training_report.json`
and no validated post-training checkpoint. Treat it as an interrupted
initialization, **not a trained candidate**. The configured primary rollout
was 2,400 samples = 60 seconds (about ten practice laps), longer than the new
six-lap evaluation ceiling. Do not resume it as-is.

The batch-mode simulator container `sdu_apex_sim_explore` was stopped too. At
handoff time, no project training process, simulator, or open-plane experiment
was left running. No data was deleted. `live_runs/` is approximately 15 GB and
contains 247 bag database files; the catalog below, not the raw file count,
should guide inclusion.

Three new reset-isolated real simulator captures and their derived final-test
dataset are retained:

- Bags: `live_runs/openplane_highspeed_straight_holdout_20261001_r01_t060/`,
  `..._r02_t080/`, and `..._r03_t100/` (10 seconds each at fixed throttle
  commands 0.60, 0.80, and 1.00; zero steering).
- Derived set:
  `live_runs/derived_dynamics_learning_20260928/highspeed_straight_holdout_dataset_20261001/`
  (1,202 samples, 3 sequences, all explicitly assigned `final_test`).
- Scores:
  [`highspeed_straight_holdout_generalist_warmstart_scores_20261001.json`](../../live_runs/derived_dynamics_learning_20260928/highspeed_straight_holdout_generalist_warmstart_scores_20261001.json)
  and
  [`highspeed_straight_holdout_highspeed_focus_scores_20261001.json`](../../live_runs/derived_dynamics_learning_20260928/highspeed_straight_holdout_highspeed_focus_scores_20261001.json).

All three passed reset recovery, collision/fault checks, bag closure, and active
odometry rate checks: approximately 39.96–39.97 Hz, with p95 sample intervals
about 25.8–25.9 ms. Throttle feedback followed each command after the initial
reset/command transition. These captures were made after the compared
checkpoints and have not been used for training or checkpoint selection.

### What the new holdout shows

| Throttle | Peak body speed | Generalist: 9.6 s position RMSE / endpoint error | High-speed specialist: position RMSE / endpoint error |
| ---: | ---: | ---: | ---: |
| 0.60 | 14.24 m/s | 50.10 m / 95.32 m | 67.18 m / 144.80 m |
| 0.80 | 18.66 m/s | 50.07 m / 96.68 m | 33.18 m / 77.37 m |
| 1.00 | 18.42 m/s | 53.03 m / 119.93 m | 60.13 m / 115.86 m |

These are plainly unusable pose errors, even though the test is only 10 seconds
and straight. On the 0.60 run, truth/predicted forward speed was 8.14/5.83 m/s
at 2 seconds, 14.24/5.56 m/s at 5 seconds, and 14.24/4.95 m/s at 9.5 seconds
for the generalist. The speed specialist predicted 9.05, 13.18, and 14.04 m/s
at those points. Its better speed trace on this run did not prevent severe
heading/position drift. Across all three runs neither model generalizes
reliably.

The simulator data also show wheel-surface speed separating from body speed:
at 9 seconds on the 1.00 throttle run, rear wheel surface speed is about
25 m/s while body speed is about 18 m/s. At 0.60 throttle these are about
15 m/s and 14 m/s. This is consistent with substantially increased wheelspin
or traction saturation at full throttle, but these signals alone do not prove a
tire-force mechanism. The three runs are a useful diagnostic, not a complete
throttle-response law; they are straight-line only and were run in ascending
throttle order.

## 4. Data inventory and what it can establish

### Canonical schema-6 mixed dataset

`live_runs/derived_dynamics_learning_20260928/plant_teacher_mixed_dataset_full3d_fixed25_20261001/`
is the current broad plant dataset. Its manifest records 850,576 rows,
7,797 contiguous sequences and 116 run records: 79 train, 3 validation,
30 test, 2 final-test, 1 replay-excluded, and 1 source-player-mismatch-excluded.
Split counts are by whole run. Inputs include body-frame motion, steering and
throttle feedback, rear wheel surface speeds, and command channels. Same-packet
simulator pose, rigid state, and linear acceleration are retained as offline
labels/diagnostics. Every model time step is fixed at the simulator's 25 ms;
receipt-time jitter is not treated as vehicle physics time. Missing packet
sequence numbers break a trajectory.

Coverage is broad but uneven: maximum observed speed is 20.80 m/s and the
99th percentile is 15.36 m/s. At least 5 m/s and absolute steering at least
0.16 rad occur in 54,262 training rows, 9,985 validation rows, 29,049 test
rows, and 9,747 final-test rows. These are correlated samples, not independent
experiments, and not every speed/steering pairing is physically feasible.
At the prior 60-second horizon, only five train, two validation, three test,
and one final-test run have complete pose-labelled sequences. At roughly
36 seconds, those same five long training runs and small held-out groups remain
the limiting run-level evidence. Shorter windows draw from more runs, but do
not substitute for long free-running validation.

The manifest currently records six import errors. Two bags genuinely lack
command topics needed for this causal plant target. Four other import failures
were `UnboundLocalError` exceptions in the exporter (three rootless captures
and the closed throttle-surface `r03` capture). The exporter bug has since been
patched and regression-checked against an empty/failed bag and a good bag, but
the canonical dataset has **not** been rebuilt from those four bags. Recheck
those specific files with the fixed exporter before deciding they are unusable;
do not rebuild or retrain blindly.

### Additional curated datasets

The entry point is
[`VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md`](VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md).
It separates:

- Phase-salvage verified archive: 1,804 sequences (908 train, 896 held-out).
  Keep the speed-target-mismatch (45 phases) and unverified (18 phases)
  archives separate; they are not baseline evaluation data.
- Fine throttle/steering transition surface from `r04` and `r05_resume`:
  1,508 usable replicated condition keys, representing 754 conditions with two
  valid repeats. It spans 5% throttle transition points and 13 steering
  settings from -30 to +30 degrees. This is valuable local transition data,
  but it is fragmented/reset-isolated and is not a continuous-lap plant by
  itself. `r03` remains unreviewed and may recover through the exporter fix.
- Paired throttle ramp/step study: ten clean runs and 240 matched ramp/step
  pairs (4.5 and 6.5 m/s, both turn directions, two high-steering magnitudes).
  It found a short-lived encoder/kinematic-speed residual response; this is a
  proxy, not direct wheel slip or tire force, and the pooled later response was
  inconclusive.
- Practice-transfer captures: useful complete-run tests, but only a few
  independent trajectories. Multiple 12-lap bags exist; they can be cropped
  to six laps or less for the requested practical evaluation.
- A closed throttle-transition sweep stopped at 237/5,050 pairs may contain
  unique straight-line conditions; compare condition keys before inclusion.

The older schema-3 dataset, continuous-sequence view, and one-run transfer view
are documented in the catalog. Use the schema-6 manifest for current teacher
training unless a specific comparison requires an older view. Do not pool
random overlapping frames across splits or merge unverified salvage into the
final test set.

### Important observability limit

Earlier provenance checks found recorded IMU orientation and angular rates to
be exact duplicates of the odometry orientation and angular rates. They are
not an independent roll sensor in these captures. There are no individual
wheel vertical loads, contact forces, or tire-force labels in the identified
data. A learned model can estimate effective motion response; it cannot claim
to have recovered per-wheel tire forces or load transfer from absent labels.

## 5. What has been tried and decisions supported by evidence

- **Planar GRU / broad-data teacher:** keep as the baseline comparator, not as
  a trusted simulator. Earlier 12-lap evaluation averaged 2.658 m XY RMSE over
  six captures; the high-steering specialist averaged 2.775 m. Both are too
  inaccurate for uncritical raceline/weight search.
- **Rigid-body/3D recurrent teacher with attitude and acceleration targets:**
  rejected. It was markedly worse than the planar GRU on held-out practice
  rollouts. Adding more physical outputs did not fix recursive state drift.
- **Long-horizon objective variants:** heading-only diverged badly. Position-
  only improved the two difficult validation captures but regressed on held-out
  runs. A balanced XY-plus-heading model is the most promising historical
  result: mean XY RMSE 2.18 m versus 4.69 m for its comparator over six
  previously evaluated captures, winning 5/6. However, the paired run-bootstrap
  interval included no improvement, and the final-test capture regressed from
  1.53 m to 2.36 m. Treat this as a candidate for bounded re-evaluation, not a
  confirmed gain. Full details are in
  [`RIGID_BODY_TEACHER_PROGRESS_20261001.md`](RIGID_BODY_TEACHER_PROGRESS_20261001.md).
- **Future measured actuator/encoder oracle ablation:** replacing predicted
  actuator and wheel channels with their future measured values barely changed
  body-state error on a hard open-plane run. This argues against actuator or
  encoder prediction being the sole source; recursive body-state dynamics,
  latent/history representation, or target alignment remain more likely.
- **High-steering sampling / throttle-rate / mixture models:** results have
  been mixed or worse on whole-run and fresh holdout data. Keep the plain GRU
  and high-steering candidate as offline comparators only; no runtime
  integration is supported.
- **Throttle ramp/step effect:** there is evidence of an early response in a
  wheel-speed residual proxy, but no demonstrated universal or lasting effect
  that justifies adding a throttle-slew correction to production dynamics.
- **Fresh 0.60/0.80/1.00 straight holds:** the latest holdouts establish a
  concrete high-speed recursive generalization failure and possible
  throttle-dependent wheelspin/saturation. They do not identify the nonlinear
  mechanism or establish curved high-speed behavior.

## 6. Code and workspace state

Research changes include schema-6 data preparation with exact packet-sequence
25 ms timing, a causal command-only plant trainer, free-running and per-run
evaluators, 3D/structured teacher experiments, throttle-surface analysis, and
an open-plane reset smoke runner. The smoke runner now performs an initial
simulator reset before declaring the spawn and writes standard phase markers.
The exporter now handles a capture with no extractable phases without raising
an unbound-local exception and correctly aggregates simulator-label validity
across sequences. No change to simulator physics or production MPC/odometry
behavior was made as part of this modeling work.

The working tree contains substantial uncommitted research changes and new
reports/scripts. Preserve them for review; do not reset or clean the tree as
part of this handoff. The following reports are useful entry points:

- [`RIGID_BODY_TEACHER_PROGRESS_20261001.md`](RIGID_BODY_TEACHER_PROGRESS_20261001.md)
- [`VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md`](VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md)
- [`OFFLINE_PLANT_NEXT_STEPS_20260930.md`](OFFLINE_PLANT_NEXT_STEPS_20260930.md)
- [`OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md`](OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md)
- [`tools/vehicle_dynamics_learning/README.md`](../../tools/vehicle_dynamics_learning/README.md)

The local persistent analysis environment is
`/home/akselmo/.local/share/sdu-apex-modeling-venv/` (CPU-only PyTorch
2.14.1+cpu). The host has a GTX 1080, but the stopped training job ran on CPU;
`nvidia-smi` showed no model-training process using the GPU. GPU acceleration
should be benchmarked rather than assumed beneficial for this small,
step-recursive GRU implementation.

## 7. Suggested independent review questions / minimum next work

1. Is the 25 ms packet-index timebase and target/frame alignment correct for
   the new holds? Confirm using packet sequence, simulator pose, wheel-speed,
   and command traces before fitting.
2. Why does the generalist's forward-speed prediction decay under a sustained
   throttle command, and why does the specialist's improved speed prediction
   still create large pose error? Inspect model state evolution and training
   support; do not infer the answer from pooled one-step RMSE.
3. Re-export only the four bags with exporter exceptions. Decide whether the
   extra `r03` throttle data changes independent high-speed support. Keep
   missing-command bags out of this command-driven plant target.
4. Re-score the strongest balanced candidate on complete held-out runs, but
   only through at most six laps. Report 1/2/3/6-lap trajectory RMSE, endpoint
   and heading error, per-run results, and run-cluster uncertainty. Use the
   fresh three-hold run as a diagnostic final test, not a tuning set.
5. If training again, compare a small number of causal architectures/objectives
   on fixed whole-run splits and select on recursive trajectory error. The
   six-lap ceiling is an evaluation bound, not a requirement to backpropagate
   through 36 seconds on every optimizer update. Do not spend compute on
   thousands of hyperparameter combinations until an offline simulator beats
   the current comparator on independent runs.
6. Decide whether the recorded evidence can identify only an effective
   command-to-motion model, or whether new safe, physically feasible curved
   high-speed trajectories are necessary. Do not repeat a broad grid or request
   unavailable tire-force labels.

Until those checks show a material, repeatable improvement, retain all learned
models as research comparators and do not use them to tune production MPC
weights or claim improved racelines.
