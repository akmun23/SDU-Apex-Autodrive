# Unseen practice-run model accuracy

**Result:** the larger frozen RSSM candidate predicts movement better than the
previous smaller RSSM on two new practice-simulator runs, especially in
position and heading after two seconds. The errors are still too large to call
this a trusted multi-lap offline simulator, and neither model is integrated
into production.

## What the numbers mean

The models were recursively rolled forward using the observed starting context
and future steering/throttle commands. Future ground truth and future sensor
measurements were not supplied to the rollout. Each model was scored on the
same 64 windows from each of two complete, previously unseen practice drives.
The table pools those windows with equal weight per run. The 64 windows are
correlated; the independent evidence is **two drives**, not 128 experiments.

The previous model is `c2_h128_z16_seed17020`; the current candidate is
`c2_h256_z32_seed17021`. Lower is better.

| Prediction horizon | 2D position RMSE | Forward speed `u` RMSE | Sideways speed `v` RMSE | Yaw-rate RMSE | Heading error RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.25 s, previous → candidate | 3.3 → 2.3 cm | 0.316 → 0.272 m/s | 0.067 → 0.026 m/s | 0.235 → 0.218 rad/s | 1.19° → 1.03° |
| 0.75 s, previous → candidate | 17.1 → 12.0 cm | 0.246 → 0.226 m/s | 0.067 → 0.037 m/s | 0.446 → 0.289 rad/s | 5.51° → 4.42° |
| 2.0 s, previous → candidate | 90.1 → 48.4 cm | 0.282 → 0.247 m/s | 0.043 → 0.042 m/s | 0.362 → 0.331 rad/s | 14.5° → 9.95° |

Position is the recursively predicted car location compared with simulator
truth at the horizon. `u` and `v` are body-frame forward and sideways speed;
they are errors in velocity components, not position. Yaw rate is how quickly
the car rotates; heading is its orientation error. For example, after two
seconds the candidate has about **48 cm 2D position RMSE** and about **10°
heading RMSE** on these windows. That is substantially better
than the previous model's 90 cm and 14.5°, but not negligible.

The 2-second position result improved on each held-out run individually:
0.68 → 0.42 m on run 1, and 1.08 → 0.54 m on run 2. The combined normalized
body-state score improved about 15% at 0.25 s, 29% at 0.75 s, and 8% at 2 s;
the 2-second combined score was slightly worse on run 1 and clearly better on
run 2. Thus the position/heading gain is encouraging, but broader run-level
confirmation is still needed.

## What this does and does not establish

- These are real, unseen simulator trajectories, not a replay of training data.
- The captures reached about 7.8 m/s. They do not validate the 9–12 m/s edge of
  the intended race domain.
- The longest scored rollout was two seconds. No full-lap free-running error
  has been established for this candidate; a 48 cm two-second error is not
  evidence of negligible lap-scale drift.
- The production odometry/MPC and localization were not changed. The candidate
  has not been deployed, so live-car accuracy is unchanged by this experiment.
- In the second live production-stack capture, AMCL-vs-truth position error was
  5.0 cm median / 12.4 cm p95; integrated odometry drift was 16.6 cm median /
  27.1 cm p95. Those are separate runtime baseline measurements, not the
  learned plant's prediction errors.
- The two production-stack practice captures completed three laps without a
  collision. The second run's racing laps averaged 6.07 s. Those results do not
  demonstrate that the learned model caused faster driving.

## Reproducible score artifacts

- Run 1, previous: [`rssm_c2_h128_score.json`](../../live_runs/derived_dynamics_learning_20260928/practice_unseen_validation_20261001_r01/rssm_c2_h128_score.json)
- Run 1, candidate: [`rssm_c2_h256_score.json`](../../live_runs/derived_dynamics_learning_20260928/practice_unseen_validation_20261001_r01/rssm_c2_h256_score.json)
- Run 2, previous: [`rssm_c2_h128_score.json`](../../live_runs/derived_dynamics_learning_20260928/practice_unseen_validation_20261001_r02/rssm_c2_h128_score.json)
- Run 2, candidate: [`rssm_c2_h256_score.json`](../../live_runs/derived_dynamics_learning_20260928/practice_unseen_validation_20261001_r02/rssm_c2_h256_score.json)

The corresponding manifests document admitted continuous intervals, timing,
sensor/label validity, and the exclusions from each partial capture. Test runs
remain in the `unseen_practice` evaluation split and were not used for fitting
or checkpoint selection.
