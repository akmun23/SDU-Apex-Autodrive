# Offline vehicle-dynamics learning

This package turns the existing wall-free Explore ROS bags into a reproducible
dataset and trains a recurrent nonlinear state-transition surrogate. It is
offline development tooling only: it does not launch the simulator, modify
physics, change the competition controller, or add a runtime topic subscriber.

For the current source-of-truth dataset, run inventory, cleanup decisions, and
the next evidence-based capture plan, start at
[`VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md`](../../docs/development/VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md).

## Checkpoint: 2026-09-28

Audit warning: the old schema-v2 oracle and sensor datasets both contain
captures marked `aborted=true` that were nonetheless labeled clean. All
numeric observer/plant results in this historical checkpoint are exploratory
until rebuilt and re-evaluated from the corrected capture-completion gate.

Two distinct models have now been evaluated on the exported open-plane data:

- A **sensor-only state observer** estimates present rear-axle `u` and `v`
  from causal encoder, IMU, actuator-feedback, command and sample-time inputs.
  On the one full-input validation run its RMSE is `0.109 m/s` in `u` and
  `0.071 m/s` in `v`, versus `4.017` and `0.983 m/s` for the simple
  encoder/zero-lateral-speed baseline. On seven named whole-run holdouts, all
  runs beat that baseline; the worst `u/v` RMSE is `0.166/0.231 m/s`. IMU
  yaw-rate is passed through directly because it already matches the offline
  body-state label closely. A single final full-input run was also evaluated
  once after fitting; it is no longer an untouched final test.
- An **oracle-current-state plant model** receives true current body-state
  history and logged future commands, then recursively predicts future body
  motion. Its 750 ms error beat a persistence/trend baseline on the named
  open-plane holdouts, but the first run produced only one of five requested
  ensemble members. It therefore has no usable ensemble uncertainty estimate
  and is only a model-family benchmark, not a validated free-running plant.

High-speed/high-steering observer strata also improved substantially over the
encoder-only baseline on held-out open-plane samples (for example, above
`8 m/s`, `|steer| >= 0.42 rad`, throttle `>= 0.3`: `u/v` RMSE
`0.099/0.005` versus `3.650/0.052 m/s`). Those samples come from only three
bags and overlapping windows are not independent trials. Neither result has
passed practice-track transfer. Three candidate practice bags have timing
fault flags and do not pass the current strict quality gate; the gate was not
relaxed and no practice promotion score is claimed. See the engineering
checkpoint for artifact locations, caveats, and the next specific steps.

No model has been connected to MPC, odometry, AMCL, or EKF. Runtime topic
policy and simulator physics were not changed.

## 2026-09-29 structured-body model evaluation

The handoff's physics-structured family has been implemented and evaluated:
predict aggregate effective accelerations
`a_x_eff`, `a_y_eff`, and `alpha_z_eff` at the center of mass (COM), then
integrate rear-axle state with the measured longitudinal COM offset `L`:
`u_rear_dot=a_x_eff+r*(v_rear+L*r)`,
`v_rear_dot=a_y_eff-r*u_rear-L*alpha_z_eff`,
`r_dot=alpha_z_eff`. Here `L=0.15532 m`; pose integration remains rear-axle
referenced, as confirmed by open-plane pose/twist consistency.
Compared models were a no-latent acceleration MLP, latent dimensions 2/4/8,
and a linear prior plus learned residual, against a fold-local plain-GRU
control. Five grouped whole-run folds trained only on original training runs;
the paired score uses independent runs, not overlapping windows.

The first tournament appeared to leave the GRU as the best general teacher,
but a subsequent audit found that the source dataset had admitted aborted
captures into training. That tournament and its numeric comparisons are
therefore **provisional and superseded**; do not use them for model selection.
The old NPZ/report are preserved with `AUDIT.md` markers. A corrected
capture-completion gate now rejects aborted runs, and both the no-attitude
and attitude-conditioned grouped CVs are being rerun before drawing a model
conclusion. None of these research models is connected to MPC or odometry.
The final corrected results and limitations will be recorded in
[`STRUCTURED_BODY_STATE_MODEL_EVALUATION_20260929.md`](../../docs/development/STRUCTURED_BODY_STATE_MODEL_EVALUATION_20260929.md)
and the JSON under `live_runs/derived_dynamics_learning_20260928/`.

A targeted Explore experiment reached measured body speed about `8.2 m/s`.
The 8.0/8.3 m/s surfaces completed; high-steering 8.3 m/s probes failed the
predeclared matched-speed gate and are excluded from model fitting. A
separate 8.6 m/s probe sequence later produced a delayed rollover after
steering returned to zero (tilt exceeded 120 degrees; collision count stayed
zero). The 8-degree tilt and 9 m/s speed stops remain unchanged; this is a
physical-support limit, not a reason to weaken the experiment abort.

Because body prediction frames previously omitted measured roll/pitch, the
schema-v3 exporter now carries a separate four-channel IMU attitude view:
roll, pitch, and their body gyro rates. A follow-up grouped-CV ablation
conditions models on the observed attitude history, holds attitude at its
last observed value during rollout, and never supplies future IMU samples.
This is a predictive identification test—not an attitude-state simulator or
a runtime sensor dependency. An independent whole-run holdout is preserved
outside training. Results are not considered complete until both the
no-attitude and attitude-conditioned reports have been compared on identical
run folds and the selected model has been scored once on that holdout.

## 2026-09-30 full-trajectory plant checkpoint

The mixed-domain dataset combines validated Explore runs with active intervals
from complete practice-track captures. The canonical inventory and model
comparison are recorded in
[`OFFLINE_PLANT_TRAINING_UPDATE_20260930.md`](../../docs/development/OFFLINE_PLANT_TRAINING_UPDATE_20260930.md).
Use its manifest as the run catalog; records failing the full-stream,
collision, or sequence gates are not included in the derived training/evaluation
arrays. A subsequent interval-level salvage pass left the 93-run canonical
archive unchanged and wrote three separately gated datasets from the 24
whole-capture rejects; details and exact splits are in the
[`vehicle-dynamics data catalog`](../../docs/development/VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md)
and the
[`offline plant next-step plan`](../../docs/development/OFFLINE_PLANT_NEXT_STEPS_20260930.md).

The salvage tool is `salvage_rejected_phases.py`. It only reads the closed bags
named by the canonical manifest and writes separate `verified`,
`speed_target_mismatch`, and `unverified` NPZ archives plus a per-phase JSON
audit. It does not rewrite source bags or merge auxiliary tiers into the
baseline. Run it only in an environment with the repository's ROS 2 message
packages available:

```bash
python3 tools/vehicle_dynamics_learning/salvage_rejected_phases.py
```

Free-running evaluation uses the full captured command trace and a single
rear-axle pose anchor. After an initial true-state history, the model receives
only its own predicted body/actuator/wheel state and future steering/throttle
commands. The plain mixed-domain GRU is currently stronger than the
physics-structured Euler model and the first integrated-heading-loss
candidate for whole-run position prediction. Even the plain model accumulates
meter-scale unanchored position error, so no candidate is yet suitable as a
trusted offline lap simulator or runtime model.

The development-only open-plane excitation runner accepts
`SDU_APEX_EXPERIMENT_PROBE_DWELL_S` to lengthen only validated probe phases;
the default `0` preserves existing profile durations. Keep the existing
speed/tilt/collision/timing gates. Longer dwell is for matched, feasible probes,
not for expanding high-speed/full-lock combinations. For example, a
single-speed transition surface may use a 4-second dwell and longer timeout:

```bash
SDU_APEX_EXPERIMENT_PROFILE=isolated_transition_full_surface \
SDU_APEX_EXPERIMENT_SPEED_MPS=4.5 \
SDU_APEX_EXPERIMENT_PROBE_DWELL_S=4 \
SDU_APEX_EXPERIMENT_TIMEOUT_S=900 \
./tools/run_open_plane_experiment.sh
```

For throttle/steering interaction, use the development-only
`throttle_transition_surface` profile. The refined design samples throttle
targets every 5% from a 0% baseline, then probes 5%, 10%, 20%, 40%, and full-
throttle increases from each 10% baseline. It repeats each transition twice
with a seeded randomized order. At 13 steering settings from -30° to +30° in
5° increments, this gives 1,508 reset-isolated conditions rather than 65,650
conditions in an exhaustive 1% throttle matrix at the same 13 steering values.
This is a broad screening
surface, not a claim that unsampled throttle histories are identical; the
repeated response curves reveal where follow-up refinement is warranted.

Each condition begins after the simulator's built-in reset has returned the
car to its spawn. It records a 4-second baseline at the starting throttle and
fixed steering, then an 8-second step response at the target throttle, and
resets again. It deliberately does not wait for the body motion to settle:
high-steering telemetry has shown strongly oscillatory speed and yaw, so an
equilibrium gate can prevent those useful transients from ever completing.
The response bag records commands, actual throttle/steering feedback,
odometry, encoders, IMU, collision count and bridge timing at the existing
40 Hz rate. Actual actuator feedback mismatch is reported separately from
completion so deadband/quantization remains visible. Response fitting requires
the actual throttle and steering to track within tolerance and the stream to
pass its cadence gate.

There is no test speed cap, distance cap, or per-transition timeout. Maximum
speed and displacement are observations, not stop conditions. A rollover
invalidates that condition and requests the normal reset; any collision or
bridge timing fault aborts the entire capture immediately. Reset recovery is
verified at the spawn before another condition begins. The reset input is
development-only and explicitly disabled in the competition launch.

Steering is fixed during each throttle transition. Set
`SDU_APEX_EXPERIMENT_STEERING_ANGLES_RAD` to comma-separated signed radians;
the refined capture uses 13 values spaced by 5° across the full `+/-0.5236`
rad command range. Keep distinct captures under distinct run IDs and inspect
the closed bags before using them for training. Do not combine a reset gap
with active 40 Hz cadence statistics.

Start the simulator in one terminal, then launch the capture in another.

```bash
SDU_APEX_SIM_TRACK=explore SDU_APEX_SIM_MODE=batchmode ./tools/start_simulator.sh
```

```bash
SDU_APEX_EXPERIMENT_PROFILE=throttle_transition_surface \
SDU_APEX_EXPERIMENT_STEERING_ANGLES_RAD='-0.5236,-0.4363333,-0.3490667,-0.2618,-0.1745333,-0.0872667,0,0.0872667,0.1745333,0.2618,0.3490667,0.4363333,0.5236' \
SDU_APEX_EXPERIMENT_THROTTLE_TARGET_STEP_PERCENT=5 \
SDU_APEX_EXPERIMENT_THROTTLE_BASELINE_STEP_PERCENT=10 \
SDU_APEX_EXPERIMENT_THROTTLE_REPEAT_COUNT=2 \
SDU_APEX_EXPERIMENT_RUN_ID=openplane_throttle_5pct_5deg_20260930_r04 \
SDU_APEX_EXPERIMENT_SEED=20260930 \
./tools/run_open_plane_experiment.sh
```

After the recorder closes, the runner audits the transition bag and writes
`throttle_transition_analysis.json` plus `analysis.log` beside it. To repeat
the audit manually:

```bash
python3 tools/analyze_open_plane_throttle_transitions.py \
  live_runs/openplane_throttle_5pct_5deg_20260930_r04/run/run_0.db3
```

The audit verifies requested condition and replicate coverage, reset recovery
and spawn consistency, zero collisions/timing faults, actual throttle tracking,
and active-phase stream cadence (reset gaps are excluded). It reports
within-condition repeat variability alongside speed, displacement and actuator
response; it does not impose a speed or distance stop.

The reset-isolated throttle matrix intentionally probes throttle transitions
at the selected fixed steering angles without a speed governor; its observed
rollover conditions are recorded separately from collision-aborted runs. This
does not imply every speed/steering combination is feasible or that a learned
model should extrapolate across unobserved states. For track data, use complete
runs of feasible racing trajectories and reserve new raceline/speed
combinations for unseen evaluation.

## Continuous race-domain capture (0–12 m/s)

Use the development-only `race_domain_continuous` profile to capture longer
closed-loop histories without reset or omitted phase intervals. Its 65 seeded
four-second blocks cover target speeds from 2 to 11.1 m/s, paired turn directions,
steering magnitudes scaled down with speed, a corner-exit/unwind sequence, and
descending speed targets that exercise throttle cuts and the simulator's active
brake. Actual steering is further limited by measured speed; a 11.2 m/s command
governor and 11.9 m/s hard cutoff keep the experiment inside the race domain.
Existing collision and 8-degree tilt stops remain active. This changes only the
development command schedule, not simulator physics.

The first completed run exposed a capture-design bug: the old speed
feedforward table ended at 8.5 m/s, so 9.5/10.5/11.0 m/s targets produced only
0.367/0.388/0.397 throttle and the car topped out at 9.595 m/s. A stable,
near-straight throttle-sweep analysis found preliminary plateaus near
8.49/9.66/10.82 m/s at 0.35/0.40/0.45 command, from one source run. The
continuous profile now uses the linear inverse of those observations only
above 8.49 m/s to improve excitation; it remains capped at 0.50. This is a
capture-only calibration to be verified on new bags, not a vehicle model or
runtime control change. Schedule version 2 moves its upper target from 11.0 to
11.1 m/s, remaining below the unchanged 11.2 m/s governor. The completed
version-2 validation run reached 11.109 m/s with zero >12 m/s samples and
passed the whole-run gates; this validates data coverage, not model accuracy.

Start the same pinned Explore simulator in batchmode. Redirect Unity logs to
`/dev/null` to avoid the simulator's repeated external WebSocket diagnostics
flooding the terminal; the bridge, experiment, recorder and quality logs remain
in the workspace:

```bash
SDU_APEX_SIM_TRACK=explore \
SDU_APEX_SIM_MODE=batchmode \
SDU_APEX_SIM_LOG_FILE=/dev/null \
./tools/start_simulator.sh
```

In another terminal, run one independent capture at a time with a unique ID and
seed. The run writes only the existing minimal dynamics topics at the bridge's
40 Hz rate:

```bash
SDU_APEX_EXPERIMENT_PROFILE=race_domain_continuous \
SDU_APEX_EXPERIMENT_RUN_ID='openplane_race_domain_train_YYYYMMDD_rNN' \
SDU_APEX_EXPERIMENT_SEED=17010 \
SDU_APEX_EXPERIMENT_TIMEOUT_S=300 \
./tools/run_open_plane_experiment.sh
```

For the targeted brake/turn-in follow-up, keep the same simulator launcher and
run one seeded capture at a time. Change the run ID and seed for independent
train/validation repetitions; the profile preserves the matching plan in its
bag events and the dataset builder admits only an exact completed plan:

```bash
SDU_APEX_EXPERIMENT_PROFILE=race_domain_brake_boundary \
SDU_APEX_EXPERIMENT_RUN_ID='openplane_race_domain_brake_boundary_train_r01' \
SDU_APEX_EXPERIMENT_SEED=17020 \
SDU_APEX_EXPERIMENT_TIMEOUT_S=270 \
./tools/run_open_plane_experiment.sh
```

Restart the simulator through the same batchmode command between independent
captures so each starts from the same simulator spawn. Assign new complete runs
to validation/test/final-test before rebuilding datasets; never split a run's
windows across roles. A stopped or aborted pilot is diagnostic only, not a
training or holdout run. Zero throttle means active braking in this vehicle
interface; there is no separate coast input to claim or fit.

The focused `race_domain_brake_boundary` follow-up keeps the same 11.2 m/s
governor, 11.9 m/s hard cutoff, collision abort, and 8-degree tilt stop. It
adds seeded left/right steering sweeps at 9.5, 10.5, and 11.1 m/s, increasing
the largest tested angles by only 0.01 rad from the prior clean profile (0.09
rad at 9.5/10.5; 0.05 rad at 11.1). Eight matched target-speed reductions
preserve steering through braking, followed by corner-exit/unwind sequences.
The plan returns to 3 m/s between the high-speed brake trials and the exit
sequence. Its 51 four-second blocks take 204 seconds. This is a targeted development
experiment, not a feasibility claim: measured steering/throttle feedback,
speed, tilt, and the existing abort gates determine usable samples. Rebuild
the immutable race-domain view only after the bag passes the same stream,
timing, packet, and collision quality gates.

The coverage audit showed that this legacy profile still left moderate-steering
9–12 m/s braking unsupported. The follow-up `race_domain_moderate_braking`
profile therefore uses 64 six-second blocks (384 seconds): a straight approach,
signed steering steps through 0.10–0.14 rad at 9.5/10.5 m/s and 0.10–0.12 rad
at 11.1 m/s, then randomized paired turn-in, positive-throttle reduction,
zero-throttle active braking, and steering-release sequences in both
directions. It includes corner exits/unwind and returns to 3 m/s before each
high-speed transition. Its profile-specific steering ceiling applies only to
this development test; the production vehicle and simulator are unchanged.
The 8-degree tilt abort, collision abort, 11.2 m/s governor, and 11.9 m/s hard
cutoff remain active. Zero throttle is active braking; this simulator interface
has no distinct coast or negative-throttle command. Block start events capture
measured speed and actuator feedback, so pairs that fail to reach their
requested approach speed must be marked unmatched rather than counted as
matched braking trials.

Run one capture per fresh Explore batchmode start, then assign the whole run to
its split before rebuilding the dataset:

```bash
SDU_APEX_EXPERIMENT_PROFILE=race_domain_moderate_braking \
SDU_APEX_EXPERIMENT_RUN_ID='openplane_race_domain_moderate_braking_train_r01' \
SDU_APEX_EXPERIMENT_SEED=17022 \
SDU_APEX_EXPERIMENT_TIMEOUT_S=430 \
./tools/run_open_plane_experiment.sh
```

## Why supervised dynamics identification, not RL first

These bags are rich in commanded inputs and state labels, but they do not carry
a reward for the lap-time/collision tradeoff and do not expose reliable
per-wheel force vectors. Direct/offline RL would therefore optimize through
an uncertain value model or infer a reward that is not present in the data.
First fit and validate the transition law. Once a free-running surrogate has
acceptable held-out error and calibrated uncertainty, it can support model-
based MPC/policy search; any candidate policy still needs a real-simulator
check. A learned body-transition model is not claimed to recover individual
tire forces.

## Dataset semantics

`prepare_dataset.py` scans `live_runs/openplane*/run/run_0.db3` read-only and
exports a compact NPZ plus a JSON manifest. It rejects any capture marked
aborted, in addition to collision/timing failures. It keeps validated phase samples,
uses causal encoder-derived rear wheel surface speeds, excludes pre-phase
context overlap, and breaks sequences at packet gaps rather than interpolating
across them. It reports every discovered bag, including rejected or
source-player runs, and preserves the original bags unchanged.

The archive contains separate aligned feature views (schema versions 2–3):

- `frames`: nine **oracle-plant** features: truth body `u, v, r`, steering and
  throttle feedback, rear-left/right encoder-derived surface speed, then
  steering and throttle commands.
- `sensor_frames`: ten **causal sensor-observer** features: steering and
  throttle feedback, rear-left/right encoder-derived surface speed, IMU
  `ax`, `ay`, IMU yaw rate, steering and throttle commands, and sample `dt`.
  `sensor_valid` marks rows with all required sensors present; missing values
  are not synthesized. Observer targets are offline bridge-state `u, v`.
- Schema 3 adds `imu_attitude_frames` and `imu_attitude_valid`, four causally
  aligned roll/pitch and roll/pitch-rate measurements. It does not change the
  nine oracle-plant or ten sensor-observer feature layouts.

The plant experiment uses bridge state as history and target. During rollout it
receives the logged future command trace and feeds back only its own predicted
body/actuator/wheel state; no future ground-truth state or sensor feedback is
injected. It estimates the plant prediction floor when current body state is
known accurately. It is **not** a sensor-only estimator. Conversely, the
observer receives only causal sensor/command history at each update and
estimates contemporaneous state; it is **not** an open-loop plant forecast.

Production MPC/odometry must never subscribe to a restricted truth topic. This
offline dataset/tooling does not authorize a runtime ground-truth input.

Splits are by complete run: named holdouts are `test`,
`openplane_full_input_validation_20260928` is validation, and
`openplane_full_input_validation_20260929` is `final_test`. Exact sequence
fingerprints prevent identical runs from crossing split boundaries. The
20260929 run has already been inspected once for this checkpoint, so it is no
longer an untouched final test; do not use it for future model selection.
Reserve a genuinely new full run as the next final holdout.

## Prepare once, then move the bundle to the compute PC

Run from the repository root, with the same ROS Python environment used by
the existing analysis scripts:

```bash
python3 tools/vehicle_dynamics_learning/prepare_dataset.py \
  --root live_runs \
  --output-dir /path/to/new/apex_vehicle_dynamics_dataset
```

The command fails rather than overwriting a non-empty output directory. Copy
`openplane_dynamics.npz` and `manifest.json` to the compute machine. The NPZ
contains only derived 40 Hz state/action sequences, not duplicate raw bags.
Inspect the manifest first: it includes quality failures, domain ranges, split
assignments, duplicate fingerprints, run counts and the extraction errors.

## Train on an available GPU

Install the stable PyTorch build selected for that machine's OS, Python and
CUDA/ROCm/CPU platform using the official selector:
<https://pytorch.org/get-started/locally/>. Install NumPy in the same
environment. Training does not require ROS or the original 2.74 GiB of bags.

```bash
python3 tools/vehicle_dynamics_learning/train_nssm.py \
  /path/to/apex_vehicle_dynamics_dataset/openplane_dynamics.npz \
  --output-dir /path/to/new/apex_nssm_run \
  --device auto \
  --members 5 \
  --time-budget-hours 12
```

The budget is shared across ensemble members. Training samples whole temporal
windows with runs balanced, uses 16 observed steps to initialize a GRU state,
and optimizes recursive 32-step rollouts (about 0.8 s at 40 Hz). It validates
at 1, 4, 10, 20 and 30 steps, keeps the best whole-run validation checkpoint,
and early-stops stale members. Use `--score-final-test` once, after model and
hyperparameter selection, to report the untouched final holdout.

Outputs are one checkpoint per member and `training_report.json`. The report
records device, seeds, exact training run IDs, checkpoints, validation
trajectories and—if requested—the final-test metrics. It scores each member
and the ensemble mean; member spread is compared with held-out error as a
diagnostic, not assumed to be calibrated uncertainty. No model is integrated
into MPC automatically.

To train the sensor-only observer instead, use the same schema-v2 or schema-v3 dataset:

```bash
python3 tools/vehicle_dynamics_learning/train_sensor_observer.py \
  /path/to/apex_vehicle_dynamics_dataset/openplane_dynamics.npz \
  --output-dir /path/to/new/apex_sensor_observer_run \
  --device auto \
  --time-budget-hours 2
```

The observer consumes a causal sensor sample stream with a 16-step burn-in
and scores 32 updates (about `0.8 s` at 40 Hz); at no point is body truth an
input. `--score-test` and `--score-final-test` are explicit evaluation options.
The current final run has been consumed, so a future untouched evaluation must
use a new bag/run assignment rather than repeatedly scoring 20260929.

## Promotion gates

Do not promote because one-step RMSE is small. A candidate must:

- Beat persistence and the existing empirical body-model baseline on whole
  held-out runs at all reported horizons, including 0.75 s.
- Improve or preserve error in the high-steering/high-speed bins, not only the
  dominant low-steering samples.
- Show useful held-out uncertainty ranking: larger ensemble disagreement
  should coincide with larger errors. The current report's spread/error
  correlation and empirical coverage are diagnostics, not proof of calibration;
  unsupported regions still need an explicit out-of-domain policy.
- For the plant surrogate, remain stable in recursive rollout without
  injecting future ground-truth state, IMU, encoder, or actuator feedback.
- For the observer, retain causal sensor-only inputs and establish state
  accuracy on separate practice runs.
- Transfer to separate practice runs before either model is considered for
  MPC or odometry use.

This is an identified *aggregate* plant surrogate. Per-wheel tire-force
recovery still requires synchronized wheel-local slip/load/force labels from a
player proven equivalent to the pinned Explore build.

## 2026-10-01 3D rigid-body teacher

The current schema-6 data and model checkpoint are documented in
[`RIGID_BODY_TEACHER_PROGRESS_20261001.md`](../../docs/development/RIGID_BODY_TEACHER_PROGRESS_20261001.md).
The dedicated trainer is `train_rigid_body_teacher.py`. It uses the simulator's
body-frame COM velocity/angular-rate labels, packet acceleration as an
offline-only teacher target, and explicit rigid-body/quaternion/rear-axle
integration. Future rollout inputs remain predicted state plus commands only.

The acceleration-supervised model improves on the preceding 3D candidate, but
still loses to the current 2D GRU on the same complete practice capture. Keep
both as offline comparators; neither change is connected to MPC or odometry.
