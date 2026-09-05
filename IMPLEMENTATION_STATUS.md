# SDU Apex AutoDRIVE implementation status

Status date: 2026-09-05  
Repository: `/home/akselmo/Documents/GitHub/SDU-Apex-Autodrive`  
Scene used for calibration: official open ground, no track

## Current result

The fixed train/validation odometry fit is strong in accelerating, steady,
and most frozen-wheel regimes, but it is not yet accepted as a universal
under-5% estimator. The remaining failure is concentrated in low-speed
deceleration:

| regime, 1--3 m/s | relative median | relative p95 |
|---|---:|---:|
| accelerating | 0.365% | 3.869% |
| steady | 0.000% | 0.016% |
| decelerating | 1.663% | 11.655% |
| frozen | 0.460% | 2.352% |
| frozen while accelerating | 0.419% | 4.430% |
| all regimes | 0.024% | 2.669% |

The production sensor-odometry header has therefore not been replaced by the
large generated candidate. Runtime remains sensor-only: encoders and IMU go
into odometry; simulator ground truth and AMCL are not inputs to odometry.

## The only CSVs to analyze for the odometry model

Use exactly these two files:

1. Training/fitting data:  
   `sdu_apex_autodrive/artifacts/calibration/raw/identification_grid_full_20260904/identification_grid_20260904_113534.csv`
2. Independent validation data:  
   `sdu_apex_autodrive/artifacts/calibration/raw/identification_grid_fusion_candidate_20260904/identification_grid_20260904_123232.csv`

Fit on the first file only, then score on the second file only. Do not merge
them, fit on the validation file, or use any other CSV as a substitute. The
fitter command and retained artifact paths are recorded in
`sdu_apex_autodrive/artifacts/calibration/MANIFEST.yaml`.

The current candidate report is:

`sdu_apex_autodrive/artifacts/calibration/derived/odom_fusion_train_full_holdout_observer_median_20260904/metrics.csv`

The generated candidate header is deliberately not retained: it was an
uninstalled 110 MB artifact. It can be regenerated from the two canonical
CSVs using the command in the manifest, and the production header remains
unchanged.

## What has been implemented

- Physical wheel radius remains 0.0590 m; slip is represented by a wheel-speed
  map and confidence gates, not by changing geometry.
- Encoder increments are timestamped, paired, and guarded against resets,
  rollovers, impossible source-time steps, and burst/zero quantisation.
- A causal two-state longitudinal observer estimates body speed and IMU bias.
- Acceleration, steady-state, deceleration, frozen-wheel braking, and
  frozen-wheel launch are treated as separate regimes.
- Braking rejects the encoder as soon as the filtered IMU identifies active
  deceleration; frozen encoders continue through IMU propagation and a bounded
  slip prior instead of forcing speed to zero.
- Stop confirmation requires quiet IMU, encoder, and odometry evidence.
- The bridge command clock is paced independently at 40 Hz, while incoming
  telemetry is not fabricated or interpolated at runtime.
- AMCL remains an independent global-position correction path and is not fed
  directly into the local odometry observer.

## Why the remaining tail is difficult

During the failing braking samples the encoder is zero, stale, or already at
the next wheel speed, while the IMU supplies acceleration but no absolute
speed. The two permitted runtime sensors can consequently have nearly
identical causal features for different ground-truth speeds.

The validation recording also contains source-time bursts. In the same brake
episode, adjacent ground-truth odometry messages can change speed by several
tenths of a metre per second over 1--3 ms while the IMU acceleration implies a
much smaller physical change. Position and twist in those bursts are not
always mutually consistent. Filtering those labels for reporting would hide
the problem, so the canonical score retains the raw timestamp-aligned ground
truth target.

This is why more forest capacity, median tree aggregation, temporal filters,
state-history features, braking priors, slip blends, and alternative causal
branches were tested but not promoted: none reduced the isolated low-speed
deceleration p95 below 5% on the untouched validation file.

## Verification already completed

- Python calibration/controller tests: 50 passed.
- Humble Docker build: `f1tenth_localization` and `sdu_apex_autodrive` built
  successfully with the active source.
- Candidate C++ header syntax check: passed with C++17 warnings-as-errors.
- `git diff --check`: passed before cleanup.
- Native IMU, encoder, odometry, diagnostics, and simulator odometry source
  events in both canonical recordings are approximately 40 Hz; recorder row
  rate is higher and must not be mistaken for sensor cadence.
- No new simulator run was started for this cleanup.

## Cleanup policy

Calibration clutter was removed rather than archived. The workspace retains
only the two canonical odometry CSVs, the current candidate metrics, and the
small wheel-map provenance CSV. Superseded controller experiments,
old validation runs, derived reports, duplicate fitters, rejected observer fit
outputs, and the Docker-owned rejected-controller archive were removed. Only
the default calibration profile and the reusable identification-grid harness
remain. Active runtime source, launch files, tests, map, raceline, and the
40-Hz bridge were preserved.

## Still open

1. Find a causal braking estimator that passes the isolated low-speed p95
   requirement without using ground truth or another live run.
2. Only after that candidate passes the fixed validation, promote it to the
   production header and rerun the Humble build/tests.
3. Then validate the full track map/raceline, AMCL, EKF, Pure Pursuit, and
   controller stack separately.
