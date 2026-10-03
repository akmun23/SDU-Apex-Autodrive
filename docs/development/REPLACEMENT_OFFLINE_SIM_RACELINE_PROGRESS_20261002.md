# Replacement offline simulator and raceline optimizer progress

**Status:** WP0–WP14 are complete; WP15 onward is blocked because the frozen
candidate was not accepted at WP14. No offline-simulator or raceline-optimizer
acceptance gate has passed. This log follows the 2026-10-02 handoff in WP
order. A completed work package means its stated artifact/evidence exists; it
does not imply overall success.

## Objective and rules

Build a command-driven offline simulator whose recursive prediction of body,
rear wheels, actuators, and pose is accurate enough to rank aggressive raceline
and speed-profile candidates over the empirically feasible 0–12 m/s domain.
The current conservative trajectory is a baseline, not a target. A merely
completed lap is not evidence that the plant model is accurate. AutoDRIVE
remains the final confirmation source. Do not modify simulator physics or
promote truth-dependent models to runtime.

Preserve whole-run splits, validation/test separation, fixed 25 ms timing,
reset isolation, and the handoff's dynamic-coupled coverage gates. Do not
continue the rejected four-wheel-force nominal, broad RSSM sweeps, or raceline
optimization before simulator-ranking gates pass.

## Ordered work-package status

| WP | Work | Status |
|---:|---|---|
| 0 | Freeze HEAD, data, captures, models, controller, map/trajectory, TUM/Exact-MinTime artifacts | Done; 130 files hashed and re-verified |
| 1 | Score production MPC model MPC0 on frozen practice starts | Done; 135 starts, 2 independent runs, no rollout failures |
| 2 | Canonical plant-regression command | Done for the frozen RSSM baseline; all five cohorts scored |
| 3 | Reset-isolated dynamic-coupled train/validation/final plans and tests | Done; detailed handoff coverage passes |
| 4 | At least 3 fresh train and 2 independent validation whole runs | Done; all 5 fresh-start runs passed whole-run quality gates |
| 5 | Immutable schema-9 teacher dataset with provenance | Done; append-only and bitwise prefix-verified |
| 6 | Dynamic coverage and sufficiency report | Done; no planned-region capture gap found |
| 7 | Deterministic effective-dynamics state-space teacher | Done |
| 8 | Verify rigid-body acceleration transform from simulator truth | Done |
| 9 | E0 GRU-encoder EDSSM, three seeds | Done |
| 10 | E1 TCN-encoder EDSSM, three seeds | Done |
| 11 | Paired E0/E1/RSSM evaluation on identical starts | Done; candidates fail transfer gate |
| 12 | Latent-size comparison on winning encoder only | Done; z=32 remains best |
| 13 | Conditional soft mixture-of-experts experiment | Done; improves dynamic pose rollout but retains speed/wheel and practice-transfer deficits |
| 14 | One final blind dynamic capture after architecture freeze | Done; capture quality passed, frozen candidate not accepted |
| 15 | Five-member final plant ensemble and calibrated support | Blocked: WP14 did not accept the architecture |
| 16 | Integrate final plant in offline plant API | Blocked on WP15 |
| 17 | Track geometry, footprint clearance and collision validation | Blocked on WP16 |
| 18 | Full-stack known-lap replay against AutoDRIVE | Blocked on WP17 |
| 19 | Historical configuration ranking | Blocked on WP18 |
| 20 | Attribute/calibrate simulator mismatch | Blocked on WP19 |
| 21 | S1 simulator acceptance | Blocked; no validated plant to rank laps |
| 22 | Raceline periodic geometry parameterization | Blocked on S1 |
| 23 | Independent periodic speed-profile parameterization | Blocked on S1 |
| 24 | Cached full-stack candidate evaluator | Blocked on S1 |
| 25 | Speed-only optimization, fixed geometry | Blocked on S1 |
| 26 | AutoDRIVE baseline/mid/best speed-candidate confirmation | Blocked on WP25 |
| 27 | Trust-region geometry optimization | Blocked on R1 transfer |
| 28 | AutoDRIVE geometry-candidate confirmation | Blocked on WP27 |
| 29 | Joint path/speed optimization | Blocked on R1/R2 transfer |
| 30 | Supported planner-envelope export | Blocked on WP29 |
| 31 | MPC-weight optimization on validated evaluator | Blocked on S1 |
| 32 | Compact MPC student model | Blocked on S1 |
| 33 | Legal-sensor odometry residual/gate | Blocked on S1 |

## Evidence already present before this handoff

- Frozen practice-transfer benchmark: 135 common starts over two independent
  six-lap validation captures; both source runs pass the established quality
  gates. These runs remain validation-only.
- Existing MPC0 report: `production_mpc_practice_model_v1.json`, 135 windows,
  two runs, horizons through 2 s. It reports 0.75 s RMSE of 0.106 m/s in
  forward speed, 0.018 m/s lateral speed, 0.089 rad/s yaw rate, 0.026 m lateral
  path error, and 0.022 rad heading error. This is a model-only replay with
  truth-initialized state and recorded commands, not closed-loop counterfactual
  validation. Re-score/freeze it under WP1.
- High-steering validation covers ±0.5236 rad around 7.5 m/s. A focused RSSM
  improves local high-steer prediction but does not generalize sufficiently
  across practice and remains meters off in longer command-only rollouts.
- The frontier validation captures non-monotonic steering response near
  10.5 m/s. Existing lead RSSM and four-wheel grey-box are not suitable
  recursive plant simulators; do not reuse the grey-box as EDSSM nominal.
- Existing dynamic steering data has independent training runs `r01/r02` and
  one validation run, but it does not yet establish the handoff's three-band
  dynamic coupled coverage or its train/validation count.
- No dynamic-coupled run has yet been admitted to the schema-9 dataset. EDSSM,
  plant ensemble, and replacement raceline optimizer remain unimplemented.

### WP0 — replacement-simulator baseline

Created [`replacement_sim_baseline_registry_20261002.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_sim_baseline_registry_20261002.json)
at the reviewed HEAD. It fingerprints 130 files, including immutable practice,
frontier, high-steer, braking and existing dynamic-steering evidence; prior
registry/splits and model checkpoints; production MPC source/config/library;
the current practice map and trajectory; and TUM/Exact-MinTime source trees.
Every recorded file hash was recomputed successfully. The source checkout was
clean before adding this handoff's progress and registry helper; the registry
records its actual worktree state at freeze time. Test/final-test remain
unscored for this handoff.

### WP1 — MPC0 production model benchmark

Re-scored the unchanged compiled production model from the pinned API
environment against the frozen 135 common practice starts. The API image lacked
`ackermann_msgs`; installed the missing ROS message package only inside the
disposable scoring container, leaving the pinned image and repository
environment unchanged. Output:
[`mpc0_practice_transfer_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/mpc0_practice_transfer_report.json).

There were zero nonlinear rollout failures. At 0.75 s, macro state RMSE was
`u=0.106 m/s`, `v=0.018 m/s`, `yaw rate=0.089 rad/s`, cross-track `0.026 m`,
and heading `0.0218 rad`. At 2 s these were `0.112 m/s`, `0.024 m/s`,
`0.103 rad/s`, `0.145 m`, and `0.0569 rad`. These are truth-initialized,
recorded-command prediction errors on two runs; they do not validate a
counterfactual closed-loop plant or the missing high-speed/high-steering domain.

### WP2 — canonical recursive-plant regression

Added [`run_plant_regression_suite.py`](../../tools/vehicle_dynamics_learning/run_plant_regression_suite.py)
as the single command for all five handoff cohorts. It checks frozen input
hashes, validation splits, exact 25 ms timing, checkpoint/data feature layout,
and training/validation run disjointness. Outputs are immutable and stored in
[`plant_regression_rssm_h256z32_20261002_complete1`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/plant_regression_rssm_h256z32_20261002_complete1).
Future truth and sensors are never fed back during rollout. The current adapter
is the frozen lead RSSM; EDSSM must use this same suite before it can be treated
as the replacement.

The suite completed on 2 practice runs/135 common starts, 7 paired brake
conditions on 1 independent validation run (braking and release, 14 two-second
replays), 2 high-steer runs, frontier r03, and ordinary race-domain validation
r04/r05. No test/final-test captures were used. All 130 WP0 source hashes
remain intact.

The lead RSSM is not a satisfactory recursive offline simulator:

- Fresh practice starts: at 0.75 s, macro state RMSE is `u=0.200 m/s`,
  `v=0.028 m/s`, `r=0.188 rad/s`, position x/y `0.062/0.065 m`, heading
  `0.059 rad`; at 2 s, x/y are `0.328/0.161 m`, heading `0.120 rad`.
- Across the high-steer phases, per-phase position RMSE median is about
  9.3–10.0 m and p95 about 11.7 m. Phases are not independent replicates; the
  size of the mismatch still rules this out as a useful high-steer plant.
- Frontier r03 has 61 scored segments; median/p95 segment position RMSE is
  6.20/11.70 m. This is one independent run.
- Full-capture ordinary race-domain position RMSE is 179 m and 233 m on r04/r05,
  with endpoint errors 365 m/545 m. These long-run values are not lap errors;
  they demonstrate recursive drift, while two-second errors remain separately
  available in the raw report.
- Validation braking/release two-second position RMSE ranges 0.11–1.46 m and
  heading RMSE 0.035–0.709 rad across paired event labels from one run.

Short-horizon practice numbers alone overstate readiness. The held-out
high-steer and whole-run evidence confirms that this RSSM remains a comparator,
not the requested full-envelope simulator.

## Work log

### 2026-10-02 — handoff read and baseline audit

- Read all sections of `SDU_APEX_REPLACEMENT_OFFLINE_SIM_AND_RACELINE_OPTIMIZER_HANDOFF_2026-10-02.md`.
- Checkout is at the handoff's reviewed commit `71909824cef2ae18c61762b8d942c36fc0b8d15b`; initial worktree was clean. No `AGENTS.md` was found above the repository root.
- Existing baseline registry, frozen practice benchmark, braking fixtures,
  practice captures, high-steering/frontier captures, candidate checkpoints,
  MPC0 report, production source/config, map, and trajectories were located.
- No Docker containers were running at audit; pinned simulator and API images
  are present locally. Workspace filesystem had 181 GB free at audit.
- WP0 complete: the replacement registry contains 130 hashed files; all
  hashes re-verified without mismatch.
- WP1 complete: MPC0 was re-scored on 135 common starts across two runs, with
  zero model rollout failures. The report is tied to the new registry.
- WP2 complete: the canonical five-cohort command scored the frozen RSSM baseline.

### 2026-10-02 — WP3 dynamic-coupled capture plan

- An initial seven-point, step-only draft was rejected during a section-by-
  section check against the handoff before any simulator was started; it missed
  the prescribed 7.0/8.0 m/s points, band-A excitation frequencies, and band-C
  frontier transitions. The implemented schedule below replaces that draft.
- Implemented `race_domain_dynamic_coupled_train`, `_validation`, and `_final`
  in `tools/open_plane_excitation.py`. Every profile uses the same deterministic
  plan generator, with randomized condition order, turn direction, multisine
  phases, triangle frequency, and a stratified piecewise steering sequence.
- Nine reset-isolated conditions cover 5.0/6.5, 7.0/7.5/8.0/8.5, and
  9.5/10.5/11.1 m/s. Steering limits use existing measured support: the held-out
  full-lock capture at 7.5 m/s; the frontier through 0.20 rad at 9.5/10.5 m/s;
  and through 0.18 rad at 11.1 m/s. The 7.0/8.5 m/s limits remain at the nearest
  demonstrated 0.20/0.10 rad rather than assuming an unmeasured boundary. The
  final profile is defined but remains unused until WP14.
- Band A uses the handoff's 0.3/0.6/1.0/1.5 Hz multisine, a seeded triangle,
  and a 12-level Latin-hypercube piecewise sequence with 0.5 s minimum dwell,
  plus turn-in/unwind/reversal, throttle pickup/reduction, active braking, and
  release. Band B includes all four prescribed speeds and coupled maneuvers
  within measured steering limits. Band C traverses 0.04/0.08/0.12/0.16/0.20
  rad in both directions at 9.5/10.5 m/s, uses the measured 0.18 rad maximum
  at 11.1 m/s, and includes mixed-order and +/-0.08 reversal sequences. There
  is no white-noise steering or static Cartesian grid.
- The complete seed, conditions, waveform parameters, per-phase command plan,
  and actual 40 Hz command/feedback topics are retained in the bag. Each
  condition starts from the built-in reset and re-approaches speed. Existing
  collision, tilt, speed, and reset-recovery cutoffs remain active; simulator
  physics and runtime controller/localization code are unchanged.
- Tests passed before simulator use: 5 pure plan tests and 12 ROS-backed
  schedule tests, including handoff frequency/speed/frontier coverage,
  40 Hz waveform bounds, throttle ramp endpoints, reset grouping, braking,
  and seeded reproducibility. Python/shell syntax, profile CLI, and pinned-image
  ROS/rosbag preflight passed. The schedule is 84 phases and 401.7 s nominal;
  the reset-inclusive minimum timeout is 450.8 s, so the wrapper default is
  600 s.
- An initial ROS test invocation accidentally replaced the sourced ROS
  `PYTHONPATH`; rerunning with the ROS environment preserved passed all tests.
  No runtime package installation or repository dependency change was needed.
- The pre-existing `recursing_pasteur` controller container was left untouched.
  Read-only Docker network/port inspection showed it is isolated from the
  host-networked Explore simulator and ROS domain used by this experiment.
  Explore was started and stopped separately for the first capture.

Planned WP4 captures (each gets nine built-in spawn resets; no final/test
capture is included):

| Split | Run id | Seed | Status |
|---|---|---:|---|
| train | `openplane_dyn_coupled_train_r01_20261002` | 20261002 | captured; passed |
| train | `openplane_dyn_coupled_train_r02_20261002` | 20261003 | captured; passed |
| train | `openplane_dyn_coupled_train_r03_20261002` | 20261004 | captured; passed |
| validation | `openplane_dyn_coupled_validation_r01_20261002` | 20261005 | captured; passed |
| validation | `openplane_dyn_coupled_validation_r02_20261002` | 20261006 | captured; passed |

The per-run quality gate requires the excitation process to complete its
exact schedule with no phase-quality failures, plus the existing closed-bag
collision, bridge timing, 40 Hz stream, packet-alignment and feature-sequence
checks. Validation runs will be assigned by explicit whole-run split override;
no individual phase/sample from a validation capture will be moved to training.

### WP4 — capture 1 quality-gated whole run

Captured `openplane_dyn_coupled_train_r01_20261002` with seed `20261002` using a
fresh Explore simulator start in canonical `batchmode` with Xvfb (graphics
enabled; no GUI and no `-no-graphics`). The capture completed all 84/84 planned
phases, with 66 scored dynamic maneuvers, 18 expected unscored approach/settle
phases, zero invalid phases, zero phase-quality failures, and a clean driver
exit. The Explore instance was then stopped; no other container was modified.

Closed-bag quality results:

- Zero collisions at both capture endpoints; zero bridge timing faults.
- Packet alignment: 12,803/12,803 (100%). All nine required command/sensor
  streams passed: measured rates 39.966–40.000 Hz; worst p95 gap 26.96 ms;
  worst maximum gap 47.36 ms (sensor allowance 60 ms; command allowance 120
  ms).
- Nine rising reset events were recorded, matching the nine plan conditions.
- Simulator-truth speed peak was 11.367 m/s (<12 m/s); maximum configured
  tilt metric was 6.61° (<8°). World position remained on the plane throughout
  (x −0.16..240.86 m, y −159.07..186.11 m, z 0.054..0.064 m; no terminal fall).
- Actuator streams were present on 99.984% of aligned samples. Command and
  feedback ranges both reached steering ±0.5236 rad and throttle 0..0.50.
  Descriptive absolute command-feedback error was steering MAE 0.0179 rad / p95
  0.0787 rad and throttle MAE 0.0026 / p95 0.0038; this records actuator lag,
  not an invented rejection threshold.
- The 12,689-sample, 9-sequence preliminary quality export is at
  [`capture_qc_r01`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/capture_qc_r01/manifest.json).
  This uses the existing schema-7 preparer only as a per-run QC/export artifact;
  it is not the WP5 schema-9 replacement teacher dataset.

The second fresh-start training run, `openplane_dyn_coupled_train_r02_20261002`
(seed `20261003`), also completed all 84 phases with 66 scored maneuvers, 18
expected approach/settle phases, zero invalid phases, and zero phase-quality
failures. Its closed-bag gates pass: zero collisions/timing faults, 12,797/12,797
packet matches, nine reset epochs, and all required streams near 40 Hz (worst
p95 gap 27.16 ms; worst max gap 48.61 ms). It exported 12,681 samples in nine
sequences with 100% sensor-feature validity. Truth speed peaked at 11.276 m/s
and configured tilt at 6.43°; position remained on the plane (x −0.16..280.90 m,
y −69.59..144.40 m, z 0.054..0.064 m), with no terminal fall. Steering feedback
reached ±0.5236 rad and throttle feedback 0..0.50. The schema-7 QC/export is at
[`capture_qc_r02`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/capture_qc_r02/manifest.json),
not yet part of the required schema-9 dataset.

The third fresh-start training run, `openplane_dyn_coupled_train_r03_20261002`
(seed `20261004`), completed the same 84-phase schedule with 66 scored
maneuvers, 18 expected approach/settle phases, zero invalid phases, and zero
phase-quality failures. Its gates pass: zero collisions/timing faults,
12,804/12,804 packet matches, nine reset epochs, and 39.962–40.000 Hz required
streams (worst p95 gap 27.13 ms; worst max gap 48.53 ms). It exported 12,689
samples in nine sequences; sensor-feature validity was 99.976%. Truth speed
peaked at 11.283 m/s and configured tilt at 6.61°. Position remained on the
plane with stable height (x −0.16..250.27 m, y −149.88..116.61 m,
z 0.054..0.064 m), with no terminal fall. Feedback again reached steering
±0.5236 rad and throttle 0..0.50. Its schema-7 per-run QC/export is
[`capture_qc_r03`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/capture_qc_r03/manifest.json).

The first whole-run validation capture, `openplane_dyn_coupled_validation_r01_20261002`
(seed `20261005`), used a fresh simulator start and was explicitly assigned to
the validation split. It completed 84/84 phases (66 scored, 18 approach/settle),
with zero invalid phases or phase-quality failures. Closed-bag checks pass:
zero collisions/timing faults, 12,801/12,801 packet matches, nine reset epochs,
and 39.963–40.000 Hz required streams (worst p95 gap 27.11 ms; worst max gap
48.59 ms). It exported 12,680 samples in nine sequences with 99.961% sensor
validity. Truth speed peaked at 11.287 m/s, tilt at 6.43°; position stayed on
the plane with stable z (x −0.16..269.76 m, y −169.40..148.87 m,
z 0.054..0.064 m), with no fall. Steering feedback reached ±0.5236 rad and
throttle 0..0.50. The whole-run QC/export is at
[`capture_qc_validation_r01`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/capture_qc_validation_r01/manifest.json).

The second whole-run validation capture, `openplane_dyn_coupled_validation_r02_20261002`
(seed `20261006`), likewise used a fresh Explore start and explicit validation
split. It completed all 84 phases (66 scored, 18 approach/settle), with zero
invalid phases or phase-quality failures. Closed-bag gates pass: zero
collisions/timing faults, 12,800/12,800 packet matches, nine resets, and
39.965–40.000 Hz required streams (worst p95 gap 26.97 ms; worst max gap
50.29 ms). There were 12,681 exported samples in nine sequences and 99.968%
sensor-feature validity. Truth speed peaked at 11.281 m/s, configured tilt at
6.50°, and stable height remained 0.054..0.064 m (x −0.16..277.04 m,
y −92.58..161.05 m), with no fall. Feedback reached steering ±0.5236 rad and
throttle 0..0.50. Its QC/export is at
[`capture_qc_validation_r02`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/capture_qc_validation_r02/manifest.json).

### WP5 — immutable schema-9 replacement teacher dataset

Built [`replacement_teacher_dataset_v1`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/replacement_teacher_dataset_v1/manifest.json)
from the frozen schema-8 race-domain parent and exactly the five WP4 captures.
No legacy bags were re-extracted, no sample-level split was used, and the final
dynamic profile remains blind. The combined NPZ contains 721,041 rows, 3,806
sequences and 24 whole runs: training 528,509 rows / 2,137 sequences / 15 runs;
validation 110,611 / 571 / 7; test 40,958 / 549 / 1; final-test 40,963 / 549 / 1.
The five appended captures contribute 63,420 rows and 45 reset-isolated
sequences. All 34 pre-existing arrays were independently verified bitwise
identical over their original schema-8 prefix. The manifest records frozen
parent and registry hashes, capture/bag/QC hashes, run IDs, profiles, seeds,
condition plans and the blind-evaluation policy.

Builder and focused validation test:
[`build_replacement_teacher_dataset.py`](../../tools/vehicle_dynamics_learning/build_replacement_teacher_dataset.py)
and [`test_build_replacement_teacher_dataset.py`](../../tools/vehicle_dynamics_learning/test_build_replacement_teacher_dataset.py).
The three dataset-integrity tests passed; the complete build and independent
prefix-byte comparison passed. No production or simulator physics changed.

### WP6 — dynamic coverage and sufficiency

Built [`dynamic_coverage_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/dynamic_coverage_20261002/dynamic_coverage_report.json)
and its per-condition CSV with
[`build_dynamic_coverage_report.py`](../../tools/vehicle_dynamics_learning/build_dynamic_coverage_report.py).
The report counts independent whole runs and reset-isolated sequences—not just
sample rows—across body speed, measured steering and steering rate, throttle
command/feedback and their rates, yaw, valid IMU lateral acceleration and rear
wheel/body speed mismatch. It also records all phase events and speed/absolute-
steering support cells. It deliberately does not demand a rectangular grid of
physically unsupported speed/steering combinations.

Coverage finding: every one of the nine planned speed/steering-cap conditions
has three independent training runs and two independent validation runs. Both
steering signs were reached in each validation run; every validation condition
has a valid active-brake phase. All 2-D speed/absolute-steering cells observed
in the new captures occur in both splits; there are no train-only or
validation-only cells. Speed spans 0–11.367 m/s in training and 0–11.287 m/s in
validation. Measured steering reaches ±0.5236 rad at the demonstrated 7.5 m/s
condition; at the higher-speed frontier, the report preserves the narrower
measured caps instead of implying untested full lock. Throttle command and
feedback cover 0–0.50 in both splits. Every dynamic maneuver category—multisine,
triangle, piecewise steering, turn-in, reversal, throttle pickup/reduction,
braking/release, and frontier sweeps/mixed order—has independent train and
validation runs.

The held-out tails are retained and called out, not hidden: four valid IMU
lateral-acceleration samples in validation reach −13.12 m/s² versus −11.27 to
+11.39 m/s² in training; they occur at 7.89–8.44 m/s with small measured
steering and high yaw, while each training run already contains many samples in
that same speed/steering/high-yaw regime. One validation steering-rate sample
and one throttle-command-rate sample exceed the corresponding signed training
extrema; their magnitudes are represented by opposite-sign training transitions
(steering rate −35.04 rad/s; throttle rate −18.48 norm/s). These are isolated
tail differences, not an absent planner-relevant region, so WP6 does not justify
another capture. This conclusion is limited to the preregistered measured
envelope; steering beyond its high-speed caps and speeds above 12 m/s remain
unsupported.

The event topic leaves about 256–263 samples/run outside phase-event intervals
(capture lead-in, inter-event publication gaps and tail). They remain in the
quality-admitted dataset; the coverage report counts them as data but reports
the unassigned-event count explicitly. This is event-label coverage, not missing
sensor/packet data.

Follow-up against the two actual production-practice bags resolves the
apparent full-throttle support gap: `/autodrive/.../throttle_command` peaks at
0.33661/0.33647; feedback peaks at 0.337/0.336; target speed peaks at
8.13/8.12 m/s. The five fresh dynamic captures independently cover throttle
command and feedback through 0.50 while reaching 11.3 m/s. The single legacy
100% throttle sweep's extreme wheel/body mismatch is not used to define the
raceline domain. WP8 derives its high-mismatch regime threshold only from
training rows at body speed <=12 m/s and throttle command <=0.50; this leaves
four independent validation runs in that hard regime.

### WP7 — deterministic EDSSM dynamics module (complete)

Added [`effective_race_teacher.py`](../../tools/vehicle_dynamics_learning/effective_race_teacher.py)
with the prescribed seven-state vector (body COM `u/v/r`, steering/throttle
feedback, rear-left/rear-right wheel surface speeds), 80-step causal history,
32-dimensional deterministic latent state, generalized effective-acceleration
heads, explicit body-frame rigid coupling, actuator update, wheel integration,
and COM-to-rear-axle pose integration. Only commands are supplied after the
initial history. No four-wheel-force nominal or inferred front-wheel states
were added. The initial actuator fit was row-weighted; review showed the
343k-row legacy throttle sweep could dominate it. It is now train-only and
hierarchical by family/run/condition/sequence. It selects a one-sample (25 ms)
delay, alpha 0.402 for steering and 0.902 for throttle. A synthetic test with
short and long sequences confirms the long sequence does not dominate this fit.
The shared training/scoring entry points and recursive evaluations are complete.
This implementation remains offline-only; no claim of plant acceptance is made.

### WP8 — body-frame acceleration label validation (complete)

Added [`validate_effective_acceleration_labels.py`](../../tools/vehicle_dynamics_learning/validate_effective_acceleration_labels.py)
and focused quaternion/frame tests. Persistent results are in
[`acceleration_label_validation_20261002/report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/acceleration_label_validation_20261002/report.json).
The report scores only train/validation whole runs (15/7); test and final-test
were not read for model selection. It compares unrotated and quaternion-inverse
simulator-acceleration axes, and start/end/interval-mean time alignment, against
finite-difference body-COM acceleration. Normal, high-steer, zero-throttle
deceleration and in-envelope wheel/body-mismatch-proxy strata have independent
run metrics.

The same-packet state convention is verified against rear-axle odometry:
`u_rear = u_com` with zero measured error; `v_rear = v_com - 0.15532*r` with
validation RMSE `2.3e-8 m/s`; yaw rate is identical with zero measured error.
The unrotated interval-mean acceleration is best in both splits. Validation
`ax/ay` correlations are `0.951/0.999`; two-axis RMSE is `0.594 m/s²` pooled
and `0.412 m/s²` macro-run (`0.392 m/s²` training macro-run). Quaternion-inverse
rotation instead has `0.068/0.156` correlations and `6.09 m/s²` pooled RMSE.
Therefore the acceleration channels must not be quaternion-rotated. Training
targets will be finite differences of body-COM truth; simulator acceleration
is a frame/alignment diagnostic, not the direct target.

Held-out regime macro-run combined RMSE for the selected interval-mean
alignment is `0.247 m/s²` in high steering (15,487 transitions, five runs),
`0.773 m/s²` in zero-throttle deceleration (3,721 transitions, all seven
runs), and `0.244 m/s²` in the high mismatch proxy tail (1,418 transitions,
four runs). The mismatch proxy is not asserted to be ground-truth tire slip.
This validates target consistency; it does not establish recursive vehicle
prediction accuracy.

The schema-9 split remains frozen; test and final-test remain unused.

### WP9/WP10 — E0/E1 training (complete)

Trained the prescribed deterministic effective state-space model with a 2 s
history, z=32, family/run/condition-balanced sampling, and 0.75/2/5 s rollout
curriculum. E0 uses a GRU history encoder; E1 changes only that encoder to a
dilated causal TCN. Three optimizer seeds were run for each encoder, all with
1,200 updates and the same sampling seed. All checkpoint selection used only
the seven registered open-plane validation runs; neither test nor final-test
was used. The six per-seed directories, training logs, summaries and selected
checkpoints are under
[`edssm_training_20261002`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/edssm_training_20261002/).

Lowest validation composite by encoder was E0 seed 101: `1.3438` (step 1200),
and E1 seed 101: `1.3595` (step 1200). Other seed minima were E0 `1.4634` and
`1.4718`, E1 `1.4293` and `1.5653`. The seed spread is material, but the best
E0/E1 models are close on the registered composite; neither score alone
establishes transfer.

### WP11 — paired recursive evaluation (complete; no architecture accepted)

Added
[`compare_effective_race_teacher.py`](../../tools/vehicle_dynamics_learning/compare_effective_race_teacher.py)
to replay the best registered E0/E1 checkpoints and frozen lead RSSM from the
same state history, pose and future steering/throttle commands. Body-state
errors are compared in the same COM reference; RSSM lateral velocity is
transformed from its rear-axle API convention before scoring. Pose remains
rear-axle referenced. Uncertainty is bootstrapped by whole run, not by treating
overlapping windows as independent. The fixed validation starts used during
checkpoint selection remain unchanged; a frontier label is added only after
start selection. Reports:

- [`paired_comparison_frontier_v1`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/edssm_training_20261002/paired_comparison_frontier_v1/paired_comparison.json): E0/E1, dynamic validation and production-practice transfer.
- [`paired_comparison`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/edssm_training_20261002/paired_comparison/paired_comparison.json): first paired run before adding explicit frontier labels; same primary starts/metrics.

Key radial position RMSEs (candidate / lead RSSM, meters):

| Cohort | Horizon | E0 GRU | E1 TCN |
|---|---:|---:|---:|
| Dynamic validation, 6 independent whole runs | 2 s | 1.694 / 1.709 | 1.960 / 1.709 |
| Dynamic validation, 6 independent whole runs | 5 s | 6.760 / 8.248 | 8.164 / 8.248 |
| Steering frontier, 5 independent whole runs | 5 s | 6.578 / 5.400 | 7.753 / 5.400 |
| Frozen practice benchmark, 2 runs / 135 starts | 2 s | 0.800 / 0.388 | 0.790 / 0.388 |
| Frozen practice benchmark, 2 runs / 135 starts | 5 s | 2.454 / 1.702 | 2.530 / 1.702 |

The E0 GRU improves the broad dynamic-validation 5 s mean over the RSSM, but
that benefit does not hold on the high-speed steering-frontier subset and is
accompanied by a large practice regression. At 2 s in practice, E0's paired
radial-position error difference is `+0.412 m` (95% whole-run paired bootstrap
CI `+0.402..+0.422 m`); at 5 s it is `+0.751 m` (`+0.727..+0.776 m`). There
are only two independent practice runs, so their run-level interval is
descriptive and weakly powered. E1 has the same direction of practice
regression. These results fail the handoff's no-practice-regression criterion;
neither EDSSM is suitable for integration or raceline optimization.

The main observed failure is recursive heading/yaw drift, not only one-step
speed error. In the dynamic subset, E0's 5 s heading RMSE is `0.805 rad`; the
frontier-specific 5 s radial error is worse than RSSM. Do not read the broad
5 s mean as a usable improvement.

### WP12 — GRU latent-size study (complete; z=32 retained)

Extended the trainer with a latent-size parameter while preserving the z=32
checkpoint layout. Trained z=16 and z=64 GRU controls with the same seed,
sampler, curriculum and 1,200-update budget as z=32 seed 101. Their validation
composites were z=16 `1.4603`, z=32 `1.3438`, and z=64 `1.5379`, so the handoff
stop rule retains z=32. The paired rollouts are in
[`latent_size_comparison_v1`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/edssm_training_20261002/latent_size_comparison_v1/paired_comparison.json).
At 5 s dynamic validation radial position error was z=16 `8.129 m`, z=32
`6.760 m`, and z=64 `8.126 m`; practice errors were `2.040`, `2.454`, and
`2.229 m`, respectively. Both alternatives remain well behind RSSM on
practice; z=16's modest practice advantage over z=32 does not offset its worse
dynamic score or the baseline gap.

WP13 was run because WP11 exposed the handoff's stated regime tradeoff:
E0 is better than RSSM over broad dynamic validation at 5 s and in some
high-steering captures, but worse on the explicit high-speed frontier and
practice transfer. Added a two-expert soft transition mixture: both experts
predict bounded effective accelerations; a softmax router sees only the
current predicted physical state, current command and causal latent state. The
same explicit rigid-body integration and actuator model are shared. No
speed/steering threshold chooses an expert. The single preregistered run used
the same data, sampler, split, seed and curriculum; router probabilities are
not collapsed (validation macro-run mean `0.615/0.385`). Implementation and
focused router/rollout tests are in
[`effective_race_teacher.py`](../../tools/vehicle_dynamics_learning/effective_race_teacher.py)
and [`test_effective_race_teacher.py`](../../tools/vehicle_dynamics_learning/test_effective_race_teacher.py).

The model selected at step 1000 has validation composite `0.9438`; the final
step-1200 composite rose to `1.1143`, so the earlier checkpoint was retained.
Matched paired results are in
[`moe_comparison_v1`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/edssm_training_20261002/moe_comparison_v1/paired_comparison.json).
At 5 s, radial position RMSE is `5.143 m` overall dynamic validation and
`4.224 m` on the steering frontier, compared with RSSM `8.248 m` and `5.400 m`.
High-steering 5 s is `4.026 m` vs `9.512 m`; ordinary is `4.493 m` vs
`7.579 m`; zero-throttle is `3.039 m` vs `4.341 m`. The wheel-mismatch proxy
is `6.066 m` vs `10.466 m`, but that hard regime exists in only one independent
validation run. These are substantial open-plane gains, not yet a useful lap
simulator.

The no-practice-regression gate still fails: against RSSM, MoE practice radial
position RMSE is `0.742 m` vs `0.388 m` at 2 s and `3.688 m` vs `1.702 m` at
5 s over the frozen 135 starts and only two independent runs. Accordingly it
is not integrated and no raceline optimization is allowed. The z=32 two-expert
checkpoint was frozen solely for WP14's single blind final dynamic capture in
[`architecture_freeze_wp14_20261002.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/architecture_freeze_wp14_20261002.json).
WP13 implementation checks passed: three focused unit tests plus checkpoint
compatibility and syntax checks. No simulator physics, production MPC,
odometry, or competition-container code was changed.

### WP14 — final blind dynamic evaluation (complete; architecture not accepted)

Ran only the frozen `race_domain_dynamic_coupled_final` profile, run
`openplane_dyn_coupled_final_r01_20261002`, seed `20261007`, in the pinned
Explore simulator using the established batchmode/Xvfb launch. The final
capture was never added to the training dataset or used for checkpoint
selection. The simulator was stopped after the bag closed; the unrelated
`recursing_pasteur` container was left running.

The schedule completed 84/84 phases (66 scored and 18 approach/settle), with
zero phase-quality failures. Closed-bag checks pass: zero collisions and
bridge timing faults, 12,799/12,799 packet matches, nine reset epochs, 12,695
exported samples, and 100% validity for sensor, pose, rigid-state,
acceleration, and attitude labels. All required streams were approximately
40 Hz (minimum 39.979 Hz; p95 gaps approximately 25.7 ms). Speed reached
11.281 m/s, steering feedback reached ±0.5236 rad, and throttle feedback
reached 0.50. The bag is
[`run_0.db3`](../../live_runs/openplane_dyn_coupled_final_r01_20261002/run/run_0.db3);
its schema-7 continuous-whole-run final-test QC export is
[`capture_qc_final_r01`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/capture_qc_final_r01/manifest.json).

Scored the frozen two-expert GRU once against the registered lead RSSM using
56 non-overlapping 5 s command-only windows within the nine reset epochs.
These windows are nested within one capture: the statistical unit is one whole
run, so run-level uncertainty is not estimable and no CI is reported. Both
models start each window from the same 80-frame observed history and pose; only
future steering/throttle commands are supplied. Full report and exact hashes
are in
[`blind_final_comparison.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/edssm_training_20261002/blind_final_comparison_wp14_v1/blind_final_comparison.json).

Final-run mean window RMSE (candidate / lead RSSM):

| Horizon | Position radial | Heading | Forward speed | Lateral speed | Yaw rate | Rear-wheel speed |
|---:|---:|---:|---:|---:|---:|---:|
| 0.75 s | 0.266 / 0.258 m | 0.118 / 0.134 rad | 0.197 / 0.106 m/s | 0.084 / 0.061 m/s | 0.510 / 0.534 rad/s | 0.612 / 0.222 m/s |
| 2 s | 1.089 / 1.754 m | 0.192 / 0.326 rad | 0.272 / 0.148 m/s | 0.080 / 0.068 m/s | 0.559 / 0.627 rad/s | 1.014 / 0.226 m/s |
| 5 s | 3.468 / 8.107 m | 0.272 / 0.750 rad | 0.403 / 0.228 m/s | 0.076 / 0.107 m/s | 0.580 / 0.705 rad/s | 1.593 / 0.243 m/s |

The candidate substantially reduces 2–5 s pose/heading error on this one
dynamic run, but is worse in forward-speed prediction at every horizon and
rear-wheel prediction by 176% at 0.75 s, 348% at 2 s, and 555% at 5 s. At 5 s,
candidate position radial RMSE remains 3.47 m (p95 across windows 8.42 m),
not negligible for reliable lap simulation. The high-steering label appears
in only one non-overlapping 5 s window; that slice is descriptive, not an
independent confirmation. This run therefore does not establish an accurate
full-envelope plant, and cannot erase WP11's frozen practice-transfer
regression (MoE 5 s position RMSE 3.688 m vs RSSM 1.702 m).

Per the handoff, the architecture is **not accepted**. Do not tune this
checkpoint against the final capture. WP15 and all dependent offline
simulator/track/raceline/weight-optimization packages remain blocked; no
raceline was generated or promoted using this plant. The run is useful held-out
evidence of a specific tradeoff—better long-horizon pose integration,
substantially worse longitudinal and wheel-state prediction—not a completed
offline simulator. WP14 is complete as an evaluation step, not as project
acceptance. No subagent review was run because the handoff's required work is
not complete and its acceptance gate failed.

### WP14 follow-up — first-divergence check and heading-constrained fit (2026-10-03; rejected)

Revisited the registered practice replay and residual attribution before
changing the teacher. The first `>1 m/s` rear-wheel-pair error occurs at
1.275 s in both practice runs and in the parent and targeted variants. The
forward-speed error does not cross 0.5 m/s until 6.875–6.900 s. On the parent,
position error crosses 0.5 m at 4.225–4.600 s, before forward speed does. This
puts early divergence in the wheel channels, followed by accumulated path
error; a low-throttle `ax` correction alone cannot explain the path drift.
At 25 ms the practice effective-`ax` residual is biased −1.07 m/s² (RMSE
2.25; 135 transitions across two runs); by 250 ms the pooled bias is near
zero, but RMSE remains 1.10 m/s². Thus the short transient is poorly modeled
even though the large forward-speed threshold is late. Measured roll/rate
posthoc covariates add only 0.6% `ax`, 1.7% `ay`, and 0.55% yaw-acceleration
explanatory accuracy on the practice runs. This is weak evidence for roll as
the practice root cause, despite modest dynamic-run value; no roll change is
justified from it.

Tried one targeted revision: fine-tune only longitudinal and rear-wheel
acceleration heads, add a final integrated-heading penalty (weight 5), retain
the low-throttle forward-speed emphasis (weight 4), and distill lateral speed
and yaw rate from the frozen parent (weight 10). Training used only the
registered train split; whole-run dynamic validation selected the warm-start
parent at score `1.0984`, while every trained checkpoint was worse (final
`1.1538`). The diagnostic final checkpoint also failed practice transfer.
Across isolated laps, mean position RMSE rose from `0.530` to `0.598 m` on r02
and `0.630` to `0.657 m` on r03; mean heading RMSE rose from `0.047` to
`0.071 rad` and `0.027` to `0.060 rad`; rear-wheel pair RMSE rose from
`0.514` to `0.809 m/s` and `0.541` to `0.841 m/s`. Full-capture metrics also
regressed. The extra heading term did not solve the recursive heading issue;
the trained `last.pt` is diagnostic only. No checkpoint was promoted or
integrated.

Training-loss wiring and its focused mathematical test are in
[`train_effective_race_teacher.py`](../../tools/vehicle_dynamics_learning/train_effective_race_teacher.py)
and [`test_effective_race_teacher.py`](../../tools/vehicle_dynamics_learning/test_effective_race_teacher.py).
All 19 focused EDSSM tests and Python compilation passed. Full-capture plus
isolated-lap replay is
[`full_practice_replay_v14_heading5_diagnostic`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/encoder_raw_state_teacher_v1/full_practice_replay_v14_heading5_diagnostic);
the reproducible run is under
[`edssm_gru_z32_e2_rollresidual_10s_axwheel_vyaw_distill10_lowthrottle4_heading5_lr3e5_seed101`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/encoder_raw_state_teacher_v1/edssm_gru_z32_e2_rollresidual_10s_axwheel_vyaw_distill10_lowthrottle4_heading5_lr3e5_seed101).
The two practice runs have already informed model experiments, so they are
development validation, not an untouched final test. No simulator was launched
and no simulator/MPC/odometry physics or runtime code was changed.

WP15 remains blocked. The evidence points next to a separate measurement/plant
question before another broad fit: determine whether the early rear-wheel
divergence is in the raw encoder-derived surface rates, the filtered odometry
wheel state, or the simulator's body-vs-wheel dynamics. Any next teacher change
must then be evaluated on untouched whole runs; these two repeatedly used
practice captures cannot provide a final unbiased promotion gate.

### WP14 follow-up — first-error attribution, roll check, and causal high-steer projection (2026-10-03)

This follow-up implemented the current request's ordered checks against the
existing frozen data. It remains diagnostic work: the offline plant is still
not accurate enough for a lap replay, and WP15 onward remains blocked.

#### Where error begins

The existing one-transition evaluator was extended to report steering and
throttle feedback as well as body, wheel, yaw, and pose errors, with signed
steering/yaw and operating-condition bins. It evaluates 25 ms transitions
initialized with measured state and the preceding causal 80-frame history;
only the next command is applied. Across seven independent dynamic-validation
runs, forward-speed RMSE is `0.043 m/s`, rear-wheel-pair RMSE `0.381 m/s`, and
yaw-rate RMSE `0.152 rad/s`. Across the two practice-validation runs these are
`0.056 m/s`, `0.375 m/s`, and `0.108 rad/s` respectively. This local accuracy
does not survive recursion.

The strongest conditional clues are:

- Dynamic validation at `|steering| >= 0.40 rad`: one-step forward-speed,
  wheel-pair, and yaw-rate RMSE are `0.065 m/s`, `0.681 m/s`, and `0.201
  rad/s`; below `0.10 rad` they are `0.040 m/s`, `0.355 m/s`, and `0.154
  rad/s`.
- The dynamic signed-steering split is markedly asymmetric in the parent:
  negative `−0.524..−0.30 rad` has one-step speed/wheel/yaw RMSE
  `0.083 m/s / 0.869 m/s / 0.236 rad/s`; positive `+0.30..+0.524 rad` has
  `0.025 m/s / 0.245 m/s / 0.125 rad/s`.
- Steering-rate bins show increasing local error: at `0.5..2 rad/s`, speed,
  wheel, and yaw RMSE are `0.058 m/s / 0.504 m/s / 0.193 rad/s`. Only 47
  validation transitions reach `5..10 rad/s`, so that extreme-rate estimate
  is too sparse for a confident conclusion.
- Throttle-slew error is concentrated in a sparse tail. At `8..22 command/s`,
  speed and wheel RMSE reach `0.110 m/s` and `1.108 m/s` over 202 dynamic
  transitions. Most data (`70,682 / 71,564` transitions) is below `0.25/s`.
- Wheel/body mismatch is informative but not a monotonic single explanation:
  dynamic `1..2 m/s` mismatch has `0.076 m/s` speed and `0.781 m/s` wheel
  RMSE; `2..4 m/s` has `0.051 m/s` and `0.590 m/s`. Practice's `0..0.10 m/s`
  mismatch slice has `0.332 m/s` wheel RMSE, while the `0.25..0.50 m/s` slice
  has `0.174 m/s`. These are conditioned associations, not causal effects.
- In practice, the `0..0.05` throttle-command slice has `0.915 m/s` wheel RMSE
  over 172 transitions; the `0.20..0.30 rad` steering slice has `0.144 m/s`
  wheel RMSE. Its modest speed range and two-run count constrain inference.

The previous pointwise recursive report found rear-wheel-pair error crosses
`1 m/s` at about `1.275 s`, while forward-speed error crosses `0.5 m/s` only
around `6.9 s`; position error crosses `0.5 m` earlier, around `4.2..4.6 s`.
The onset is therefore wheel/body/heading state consistency and accumulated
integration error, not simply an inaccurate instantaneous forward-speed
prediction. The new paired recursive report uses 135 dynamic starts across six
runs (10 s) and 31 practice starts across two runs (5 s). Parent cumulative
time-horizon RMSE is:

| Domain and horizon | Position radial | Heading | Forward speed | Yaw rate | Rear-wheel pair |
|---|---:|---:|---:|---:|---:|
| Dynamic, 10 s | 4.804 m | 0.343 rad | 0.234 m/s | 0.232 rad/s | 0.927 m/s |
| Practice, 5 s | 2.301 m | 0.303 rad | 0.185 m/s | 0.181 rad/s | 0.661 m/s |

These are RMSE over all samples through each horizon, averaged equally by
source run (not endpoint-only errors). They are nowhere near negligible lap
error. Roll is also predicted internally by the parent; recursive roll-angle
RMSE grows to `0.352 rad` by 10 s on dynamic data and is `0.060 rad` at 5 s on
practice data.

#### Does measured roll explain the residuals?

The existing leave-one-whole-run-out post-hoc model was checked on six dynamic
validation captures and two practice captures. Adding absolute measured roll
and roll rate as *diagnostic-only covariates* changes log-absolute-residual
RMSE by `−2.1%` for effective longitudinal acceleration, `−17.8%` lateral
acceleration, and `−8.2%` yaw acceleration on dynamic runs; wheel-acceleration
residual changes only `−0.4%`. Lateral/yaw improvement has the same sign on all
six dynamic runs. Practice gains are much smaller (`−0.24%`, `−1.56%`,
`−0.45%`) and wheel residual worsens `+0.81%` across the two runs.

Thus roll carries held-out dynamic information, especially for lateral/yaw
residuals, but this is not evidence that *future measured* roll belongs in a
plant rollout. The parent already initializes roll/rate from observed history
and predicts both internally; subsequent transitions consume only predicted
roll/rate and predicted lateral acceleration. No future IMU sample was exposed
to the rollout. Previous oscillator/neural roll comparisons are already
archived; the oscillator's better roll fit came with severe practice pose,
heading, speed, and wheel regressions, so it was not repeated or promoted.

#### Targeted symmetry revision and decision

Held-out mirrored-transition analysis found close left/right physical symmetry
in the dynamic high-steering captures, while the parent had the large signed
steering residual imbalance above. A 50% broad training-reflection augmentation
was tried first and rejected: the warm-start parent remained the best
validation checkpoint, and its last checkpoint worsened practice wheel/yaw
errors. A global trajectory-average reflection experiment reduced dynamic
10 s position RMSE `4.804 -> 3.499 m`, but catastrophically worsened practice
5 s position `2.301 -> 5.795 m`, yaw-rate `0.181 -> 0.422 rad/s`, and heading
`0.303 -> 0.845 rad`; it is rejected.

The narrower revision is `HighSteerStepwiseReflectionModel`: at each causal
transition, it projects the parent and reflected transition onto left/right
symmetry only when current steering feedback or the current steering command
reaches the pre-existing `0.30 rad` boundary. It feeds the projected physical
state back into the next transition and advances each branch's latent state
from projected acceleration. Below the boundary the parent transition is
unchanged. The threshold is not fit to practice data. Mathematical tests verify
high-steering reflection equivariance and exact parent identity below the
threshold.

On all 135 dynamic windows, 10 s position changes only `4.804 -> 4.688 m`
(paired run delta `−0.116 m`, 95% paired-t interval `[-0.442, +0.210]`); wheel
error changes `0.927 -> 0.918 m/s`. The 18 frozen windows whose command reaches
`0.30 rad` cover four runs. At 5 s in this high-steering subset, position
changes `1.830 -> 1.611 m`, heading `0.386 -> 0.304 rad`, and wheel error
`0.911 -> 0.813 m/s`, but run-paired intervals include no change because the
independent-run count is four. The practice captures peak at `0.276 rad`, so
the candidate is exactly identical to the parent there; practice 5 s position
error remains `2.301 m`. This is a useful, physically grounded comparator, not
a demonstrated full-domain improvement or accepted plant.

Artifacts and reproducibility:

- [`first_transition_attribution_wp14_parent_signed_actuator_v2.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/encoder_raw_state_teacher_v1/first_transition_attribution_wp14_parent_signed_actuator_v2.json)
- [`turn_reflection_symmetry_validation_v1.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/encoder_raw_state_teacher_v1/turn_reflection_symmetry_validation_v1.json)
- [`turn_reflection_projection_parent_v1.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/encoder_raw_state_teacher_v1/turn_reflection_projection_parent_v1.json) (rejected global-average experiment)
- [`turn_reflection_stepwise_projection_parent_v1.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/encoder_raw_state_teacher_v1/turn_reflection_stepwise_projection_parent_v1.json) (parent, causal targeted revision, paired run deltas, roll errors, and high-steering subset)
- [`turn_reflection_projection.py`](../../tools/vehicle_dynamics_learning/turn_reflection_projection.py), [`score_turn_reflection_projection.py`](../../tools/vehicle_dynamics_learning/score_turn_reflection_projection.py), and the focused tests under `tools/vehicle_dynamics_learning/test_turn_reflection_*`.

All prediction branches use command inputs only; measured future truth is used
only for scoring. The practice start indices were reused from the frozen
comparison, and the raw-wheel practice derivative was accepted only after
checking its recorded source-dataset hash against the frozen comparison hash.
These validation captures have already informed earlier model selection and
the experiments in this follow-up; they are not blind final-test evidence. A
future promotion still needs untouched whole-run data after the model choice
is frozen.
The focused mathematical suite passes (18 tests); Python compilation and
`git diff --check` pass. No simulator, production MPC, odometry, or physics
configuration changed. No recorded practice-lap replay was run: neither this
revision nor the parent meets the preceding motion-accuracy gate. Offline
simulator acceptance, full-lap replay, and raceline optimization remain
blocked; the next revision must reduce wheel/yaw recursive drift without
trading away practice transfer, then be checked on new whole-run captures.
