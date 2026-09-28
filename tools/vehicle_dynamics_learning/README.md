# Offline vehicle-dynamics learning

This package turns the existing wall-free Explore ROS bags into a reproducible
dataset and trains a recurrent nonlinear state-transition surrogate. It is
offline development tooling only: it does not launch the simulator, modify
physics, change the competition controller, or add a runtime topic subscriber.

## Checkpoint: 2026-09-28

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
exports a compact NPZ plus a JSON manifest. It keeps validated phase samples,
uses causal encoder-derived rear wheel surface speeds, excludes pre-phase
context overlap, and breaks sequences at packet gaps rather than interpolating
across them. It reports every discovered bag, including rejected or
source-player runs, and preserves the original bags unchanged.

The archive contains two aligned feature views (schema version 2):

- `frames`: nine **oracle-plant** features: truth body `u, v, r`, steering and
  throttle feedback, rear-left/right encoder-derived surface speed, then
  steering and throttle commands.
- `sensor_frames`: ten **causal sensor-observer** features: steering and
  throttle feedback, rear-left/right encoder-derived surface speed, IMU
  `ax`, `ay`, IMU yaw rate, steering and throttle commands, and sample `dt`.
  `sensor_valid` marks rows with all required sensors present; missing values
  are not synthesized. Observer targets are offline bridge-state `u, v`.

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

To train the sensor-only observer instead, use the same schema-v2 dataset:

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
