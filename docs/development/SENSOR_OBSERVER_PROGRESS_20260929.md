# Sensor-only movement observer — 2026-09-29 checkpoint

## Objective

The primary objective is a causal, sensor-only estimate of the car's actual
rear-axle motion across the operating envelope, especially wheelspin,
high-speed/high-curvature, and aggressive-steering regimes. Lap time is a
downstream outcome, not the observer's acceptance metric. Simulator bridge
odometry is permitted only as an offline training/scoring label; the learned
observer receives steering/throttle feedback and commands, both encoders, IMU
acceleration/yaw-rate, and sample interval. The current GRU estimates body
`u/v`; yaw rate is passed through from the gyro. It does not identify
per-wheel tire forces or provide an absolute map pose.

No simulator physics, production odometry, MPC, localization, or competition
runtime code/configuration was changed. The learned model and the
`wheel_burst_catchup_accel_mps2=0` replay remain offline comparators.

## Evidence added on 2026-09-29

The clean schema-v3 sensor GRU was trained on whole-run training data and
checkpoint-selected using the validation split. The comparisons below use
separate whole-run captures, source-stamp matched to exact sensor-only
replays of the production odometry node. The bridge's simulator odometry was
read offline for labels only. Clean captures were recorded near 40 Hz.

### Six throttle-slew high-demand runs

Across six randomized, clean 6.5 m/s-start throttle-slew holdouts, maximum
measured speed was 8.54 m/s and measured steering reached about `±0.42 rad`.
Run-level bootstrap means (runs, not 40 Hz samples, are the independent
units):

| Measure | Sensor GRU | Production odometry | Offline no-catch-up replay |
| --- | ---: | ---: | ---: |
| `u` RMSE, all scored samples | 0.094 m/s | 4.600 m/s | 0.326 m/s |
| `v` RMSE, all scored samples | 0.0075 m/s | 0.0255 m/s | 0.0101 m/s |
| `u` RMSE, speed ≥6 m/s and `|steer|` ≥0.40 rad | 0.078 m/s | 4.542 m/s | 0.343 m/s |
| `v` RMSE, same high-demand subset | 0.0094 m/s | 0.0237 m/s | 0.0133 m/s |

All six runs improved in each listed comparison. Paired whole-run bootstrap
95% CIs for the GRU-minus-no-catch-up `u` RMSE difference were `[-0.277,
-0.188] m/s` overall and `[-0.315, -0.215] m/s` in the high-demand subset.
At speed ≥8 m/s, the GRU's lateral estimate is *not* uniformly better than
the no-catch-up replay (0/6 run wins); that remains a specific weak region.

### Independent mixed full-input holdout

The clean `openplane_full_input_excitation_20260927_holdout_cache` capture is
a different, untouched test run. It completed all 552 phases with no collision,
abort, or timing fault; sensor streams averaged 39.98 Hz. It reached 8.36 m/s
and the steering limit `±0.524 rad`. Exact replays matched all 23,232 scored
source timestamps. On this run:

| Measure | Sensor GRU | Production odometry | Offline no-catch-up replay |
| --- | ---: | ---: | ---: |
| `u` RMSE, all scored samples | 0.308 m/s | 13.096 m/s | 12.025 m/s |
| `v` RMSE, all scored samples | 0.202 m/s | 0.790 m/s | 0.791 m/s |
| `u/v` RMSE, speed ≥6 m/s and `|steer|` ≥0.42 rad | 0.237 / 0.0108 m/s | 15.209 / 0.0883 m/s | 13.747 / 0.0796 m/s |
| `u/v` RMSE, speed ≥8 m/s | 0.305 / 0.0765 m/s | 13.453 / 0.173 m/s | 12.082 / 0.173 m/s |

The full-input run's overall and hardest-region errors support the claim that
the GRU learned useful nonlinear sensor-to-body-motion corrections beyond
these six throttle-slew profiles. It is still one independent mixed-profile
run, not broad population coverage. Only 23,232 of 40,363 clean samples are
scored by the fixed burn-in/rollout window protocol; unscored samples are not
silently counted as model successes.

### Short-horizon motion, not just velocity

The paired scorer now also integrates each 32-step estimated rear-axle twist
segment (median duration 0.776 s) in the world frame. Each segment is
initialized with the same simulator pose and heading, then compared with the
simulator path. Across the six throttle-slew runs, macro-run endpoint-error
RMSE was 0.050 m for the GRU, 0.168 m for no-catch-up replay, and 3.486 m for
production odometry. Across those six runs plus the mixed full-input run, it
was 0.075 m, 1.383 m, and 4.339 m respectively; the run-bootstrap 95% CI for
the GRU-minus-no-catch-up difference was `[-3.693, -0.099] m`.

This is explicitly a short-horizon relative-motion score with a truth pose
reset at every window. It does **not** establish long-run unanchored odometry
or localization accuracy. It isolates whether the estimated velocities
reconstruct local movement.

### Important ordinary-driving counterexample

On `practice_speed_headroom_12lap_20260927_01`, a clean 12-lap, 74.1-second
practice run with zero collisions and approximately 39.97 Hz sensor streams,
the frozen GRU is worse overall than production odometry:

| `u/v` RMSE | Sensor GRU | Production odometry | No-catch-up replay |
| --- | ---: | ---: | ---: |
| Practice run | 0.271 / 0.0186 m/s | 0.154 / 0.0104 m/s | 0.158 / 0.0104 m/s |

Therefore the GRU is a promising high-slip/high-demand observer candidate,
not a universal replacement. Any future estimator should preserve normal
regime accuracy while correcting the bad wheelspin/high-curvature regimes;
that needs a principled, calibrated regime-conditioned fusion model and fresh
whole-run confirmation. No candidate has been integrated into production.

## What the evidence says about the nonlinear behavior

The strongest identified odometry failure is not a fixed wheel-radius error.
During throttle wheelspin, rear encoder surface speed can become a poor proxy
for body speed. The production observer's aggressive burst catch-up can
follow that false wheel-speed burst and latch `u` high; once high, its wheel
recovery gate can reject otherwise useful lower wheel measurements. Setting
the catch-up acceleration to zero in offline replay sharply reduces error on
the six designed throttle-slew runs, but barely helps the different
full-input run. Thus this mechanism is real and repeatable, but not the whole
problem.

The learned model's useful signal is the **history-conditioned joint relation**
between encoders, IMU acceleration/yaw rate, steering/throttle feedback, and
commands. It estimates aggregate body movement; it does not reveal individual
contact-patch forces, friction curves, wheel loads, or a unique analytic
nonlinearity. IMU roll in these simulator bags is not an independent load
transfer measurement, so it cannot yet identify per-wheel load/slip physics.

## Coverage limits and next work

- The clean training and scored test captures top out at 8.54 m/s. There is no
  demonstrated accuracy at 10 m/s or above; high-speed lateral comparison is
  still mixed. Do not extrapolate the current model to that range.
- Keep the plain GRU and high-steering candidates as offline comparators. Do
  not put them into odometry/MPC yet.
- Next compare an observer correction/fusion model against production on
  ordinary practice and independent high-slip/high-curvature runs. Learn the
  regime weighting on training runs only, use run-grouped validation, and
  report uncertainty/OOD behavior rather than choosing a hand-set fallback.
- Extend motion evaluation from truth-reset 0.776 s segments to uninterrupted
  unanchored pose drift, with any sensor dropouts and initial-state assumptions
  explicitly accounted for.
- Collect new whole-run data only for uncovered `speed × steering × throttle`
  regions, especially the currently unsupported >8.5–10 m/s range if the
  simulator can reach it safely. Preserve collision and 40 Hz stream gates.
- After model selection is frozen, evaluate once on untouched full runs and
  then do separate real closed-loop practice transfer before considering any
  runtime change.

## Durable artifacts

Reports and replay outputs are retained under
`live_runs/derived_dynamics_learning_20260928/`:

- `sensor_observer_clean_20260929_uniform/observer_report.json` and
  `sensor_observer.pt` — training/checkpoint evidence.
- `sensor_observer_clean_20260929_uniform/paired_openplane/seven_run_observer_summary.json`
  — paired whole-run velocity and displacement bootstrap.
- `sensor_observer_clean_20260929_uniform/paired_openplane/full_input_holdout_cache_trajectory.json`
  — high-angle mixed-profile detail.
- `sensor_observer_clean_20260929_uniform/practice_speed_headroom_12lap_20260927_01.json`
  — separate practice transfer counterexample.
- `production_odom_replay_clean_20260929/` — source/config-verified offline
  sensor-only production-node replays, including the practice replay.

The scoring/replay tools are `score_openplane_observer_vs_odom.py`,
`score_practice_odom_replay.py`, `summarize_paired_observer_scores.py`, and
`replay_sensor_odometry_bag.py` under `tools/vehicle_dynamics_learning/` and
`tools/`.
